from ast import parse
import os
import torch.nn as nn


from tqdm import tqdm
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from torch.utils.data import DataLoader
from zmq import device
from Vqgan import VQGAN_model
from utils import Custom_dataset, weights_init
from modules.discriminator import Discriminator
from lpips import LPIPS
from torchvision import transforms
import logging
import datetime


import wandb
wandb.login()
time = datetime.datetime.now().strftime("%d-%m-%H-%M")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"




class TrainVQ:
    def __init__(self,args):
        self.vqgan = VQGAN_model(args)
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.conf_optimizers(args)

        
        self.prepare_training()

        self.train(args)
    

    def conf_optimizers(self,args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))


        return opt_vq, opt_disc

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def train(self,args):
        transform = transforms.Compose([transforms.Resize(size=512),transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
        train_data = Custom_dataset(img_dir=args.dataset_path,size = 0.25,transform=transform)
        train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=False)
        steps_per_epoch = len(train_loader)
        model = self.vqgan
        if args.pretrained == True:
            print('loading checkpoints............')
            #import pdb;pdb.set_trace()
            checkpoint = torch.load(args.checkpoints)
            pretrained_dict = {key.replace("module.", ""): value for key, value in checkpoint['model_state_dict'].items()}
            #checkpoint = pretrained_dict['codebook.embedding.weight']
            #print('pretrain',pretrained_dict['codebook.embedding.weight'].shape)
            model.load_state_dict(pretrained_dict)

            #self.opt_vq.load_state_dict(checkpoint['opt_vq'])
            #self.opt_disc.load_state_dict(checkpoint['opt_disc'])
            pre_epoch = checkpoint['epoch']
            print(f'starting from epoch_{pre_epoch}')
        else:
            pre_epoch = 0
        #model = nn.DataParallel(model,device_ids=[0,1,2,3]).cuda()
        model = model.cuda()
        for epoch in range(pre_epoch,pre_epoch+args.epochs):
            with tqdm(range(len(train_loader))) as pbar:
                for i, img in zip(pbar,train_loader):
                    img = img.float()
                    img = img.to(device=args.device)
                    #print('gpus', torch.cuda.device_count())
                    #wandb.watch(model, log="all", log_freq=10)              
                    decoder_img,codebook_indices,q_loss = model(img)
                   

                    disc_real = self.discriminator(img)
                    disc_fake = self.discriminator(decoder_img)
                    disc_factor = self.vqgan.adopt_weight(args.disc_factor, epoch*steps_per_epoch+i, threshold=args.disc_start)

                    perceptual_loss = self.perceptual_loss(img, decoder_img)

                    rec_loss = torch.abs(img-decoder_img)
                    perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                    perceptual_rec_loss = perceptual_rec_loss.mean()
                    g_loss = -torch.mean(disc_fake)

                    λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                    vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)
                    
                    self.opt_vq.zero_grad()
                    vq_loss.sum().backward(retain_graph=True)

                    self.opt_disc.zero_grad()
                    gan_loss.backward()

                    self.opt_vq.step()
                    self.opt_disc.step()

                    checkpoints = {
                        "epoch":epoch,
                        "model_state_dict":model.state_dict(),
                        "opt_vq": self.opt_vq.state_dict(),
                        "opt_disc":self.opt_disc.state_dict(),
                        #"conv1": model.qunat_conv.get_weights(),

                    }
                    """if epoch % 10==0 and i % 500 == 0:
                        with torch.no_grad():
                            real_fake_images = torch.cat((img[:4], decoder_img[:4])) #add(1).mul(0.5)
                            vutils.save_image(real_fake_images, os.path.join("/home/moravapa/results/resume/resume2", f"{epoch}_{i}.jpg"), nrow=4)
"""
                    
                    wandb.log({'vqloss':vq_loss.mean(),'rec_loss':rec_loss.mean(),'per_loss':perceptual_loss.mean(),'gan_loss':gan_loss})

                    pbar.set_postfix(
                        Epoch_no = epoch,
                        VQ_Loss=np.round(vq_loss.mean().cpu().detach().numpy().item(), 5),
                        AN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3),
                        R_Loss = np.round(rec_loss.mean().cpu().detach().numpy().item(),3)
                    )
                
                    pbar.update(0)

                    if epoch % 2 == 0 and i % 500 == 0:
                        with torch.no_grad():
                            real_fake_images = torch.cat((img[:4], decoder_img[:4]))  # add(1).mul(0.5)
                            vutils.save_image(real_fake_images,os.path.join("/home/moravapa/Documents/Thesis/VQ_main/results", f"{epoch}_{i}.jpg"),
                                              nrow=4)

                            torch.save(checkpoints, os.path.join("/home/moravapa/Documents/Thesis/VQ_main/checkpoints", f"main_vq.pt"))





if __name__=='__main__':
    parser = argparse.ArgumentParser(description="VQ-main")
    parser.add_argument('--latent-dim', type=int, default=512, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=512, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=1, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-06, help='Learning rate (default: 0.0002)') #0.0000
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=5000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--pretrained',type=bool,default=True,help='resume training from previous checkpoints')
    parser.add_argument('--mode', type=bool, default=False, help='mode of the model')
    parser.add_argument('--checkpoints',type=str,default='/checkpoints',help='load checkpoints for resume the training or inference')
    parser.add_argument('--model', type=str, default='VQ_main', help='For saving the log for each model')
    args = parser.parse_args()


    
    args.dataset_path = r'/home/moravapa/Documents/Thesis/VQ_GAN/Data/sample'
    args.checkpoints = r'/home/moravapa/Documents/Thesis/synthetic/checkpoints/last.ckpt'

    base_dir = r'/home/moravapa/Documents/Thesis/VQ_GAN/outputs'
    dir = args.model + "_" + str(args.num_codebook_vectors) + "_" + str(args.latent_dim) + "_" + time
    # path = os.path.join(base_dir,dir)
    path = os.path.join(base_dir, dir)
    # import pdb;pdb.set_trace()
    if not os.path.isdir(path):
        os.mkdir(path)

    #wandb.init(project="main", name="project", dir=path)
    #wandb.config.update(args)
    #config = wandb.config
    #wandb.run.name = dir
    train_vggan = TrainVQ(args) 
