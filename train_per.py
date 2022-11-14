# Image generation using perceptual loss (Instance Normalization)

import argparse

import numpy as np
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import utils as vutils

from utils.utils_sin import Custom_dataset, weights_init
from syn_module.lpips import LPIPS
from syn_module.model import SYN_model
from syn_module.recon_freeze import Recon
from modules.model import Encoder, Decoder
import datetime
import matplotlib.pyplot as plt

import wandb
wandb.login(key='6b037e771b020c9d419f6468780b6a0640a9e9eb') #key='6b037e771b020c9d419f6468780b6a0640a9e9eb'
time = datetime.datetime.now().strftime("%d-%m-%H-%M")

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


class TrainVQ:
    def __init__(self, args):
        self.syn_encoder = SYN_model(args)
        self.real_encoder = SYN_model(args)
        self.syn_decoder = Decoder(args)
        self.real_decoder = Decoder(args)
        self.device = args.device
        self.perceptual_loss = LPIPS().eval().to(device=self.device)

        self.optimizer_encoder, self.optimizer_decoder = self.conf_optimizer(args) #, self.optimizer_decoder
        self.train_model(args=args)

    def conf_optimizer(self, args):
        lr = args.learning_rate
        enc = torch.optim.Adam(self.encoder.parameters(), lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))
        dec = torch.optim.Adam(self.decoder.parameters(), lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))
        #enc = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))
        return enc, dec



    def train_model(self, args):
        transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
                                        ])
        train_data = Custom_dataset(args.unreal_data, transform=transform)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
        #steps_per_epoch = len(train_loader)
        total_samples = len(train_loader)*args.batch_size

        encoder = self.encoder
        decoder = self.decoder
        lst = ['decoder']
        #check_path ="/home/moravapa/Documents/Thesis/Taming/ckpt/last.ckpt"#logs/2022-10-13T15-01-05_/checkpoints/last.ckpt"
        checkpoint = torch.load(args.checkpoints,map_location='cuda')
        #import pdb;pdb.set_trace()
        pretrained = {x.replace('decoder.',''):y for x,y in checkpoint['state_dict'].items() if x.split('.',1)[0] in lst }
        decoder.load_state_dict(pretrained)
        decoder.eval()
        decoder.requires_grad_(False)
        encoder = encoder.to(device=self.device)
        decoder = decoder.to(device=self.device)
        #model = self.model.to(device=self.device)
        #import pdb;pdb.set_trace()
        #model = nn.DataParallel(model, device_ids=[0,1]).to(device=self.device)
        for epoch in range(0, args.epochs):

            with tqdm(range(len(train_loader))) as pbar:
                for i, img in zip(pbar, train_loader):

                    #real_1 = img['A'].float().to(device=self.device)
                    #import pdb;pdb.set_trace()
                    real_2 = img.float().to(device=self.device)
                    #n
                    
                    feature = encoder(real_2)
                    recon_img = decoder(feature)
                    #recon_img = model(real_2)

                    perceptual_loss = self.perceptual_loss(real_2,recon_img)
                    rec_loss = torch.abs(real_2 - recon_img)
                    perceptual_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                    perceptual_rec_loss = perceptual_loss.mean()

                    self.optimizer_encoder.zero_grad()
                    self.optimizer_decoder.zero_grad()


                    perceptual_rec_loss.backward()
                    self.optimizer_encoder.step()
                    self.optimizer_decoder.step()
                    checkpoints = {
                        "epoch": epoch, "model_state_dict": encoder.state_dict(),
                    }
                    wandb.log({'perceptual_loss': perceptual_loss.mean().cpu().detach().numpy().item()})

                    pbar.set_postfix(
                        Epoch_no=epoch,
                        perceptual_loss=np.round(perceptual_rec_loss.cpu().detach().numpy().item(), 5),
                    )
                    pbar.update(0)
                    if epoch % 50 == 0 and i == total_samples//args.batch_size-25:
                        with torch.no_grad():
                            real_fake_images = torch.cat((real_2[:4], recon_img[:4]))#.add(1).mul(0.5)[:4]))
                            wandb.log({"Generated Images":wandb.Image(real_fake_images,caption='Image')})
                            vutils.save_image(real_fake_images, os.path.join("/scratch/moravapa/images/1", f"image_freeze{epoch}.jpg"), nrow=4)
                            torch.save(checkpoints,os.path.join("/scratch/moravapa/chkpt", f"perc_{epoch}.ckpt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQ-GAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--resolution', type=int, default=256, help='Number of channels of images (default: 3)')
    parser.add_argument('--ch_mult', type=list, default=[1,1,2,2,4], help='Number of channels of images (default: 3)')
    parser.add_argument('--out-channels', type=int, default=3, help='Number of out channels (default: 3)')
    parser.add_argument('--in_channels', type=int, default=128, help='Number of output channels in first layer (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=8, help='Input batch size for training (default: 6)')
    parser.add_argument('--dropout', type=int, default=0.0, help='droupout')
    parser.add_argument('--num_workers', type=int, default=8, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=401, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=0.001,help='Learning rate (default: 0.0002)')  # 2.25e-06
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=5000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1.,
                        help='Weighting factor for perceptual loss.')
    parser.add_argument('--pretrained', type=bool, default=False, help='resume training from previous checkpoints')
    parser.add_argument('--mode', type=bool, default=False, help='mode of the model')
    parser.add_argument('--checkpoints', type=str, default='/checkpoints',
                        help='load checkpoints for resume the training or inference')
    parser.add_argument('--model', type=str, default='Perceptualcode_with-IN', help='For saving the log for each model')
    args = parser.parse_args()
    #args.real_data = r'/home/moravapa/Documents/Thesis/synthetic/data/train1.txt'
    args.unreal_data = r'/home/moravapa/Documents/VQ_Main/data/train2.txt'
    args.checkpoints = r'/home/moravapa/Documents/ckpt/last.ckpt'
    #args.dataset_path = r'/home/moravapa/Documents/Thesis/data/'


    base_dir = r'/home/moravapa/Documents/VQ_Main/results'

    dir = args.model + "_" + time
    # path = os.path.join(base_dir,dir)
    path = os.path.join(base_dir, dir)
    # import pdb;pdb.set_trace()
    if not os.path.isdir(path):
        os.mkdir(path)

    wandb.init(project="Syn_train", name="project", dir=path)
    wandb.config.update(args)
    config = wandb.config
    wandb.run.name = dir
    train_vggan = TrainVQ(args)

