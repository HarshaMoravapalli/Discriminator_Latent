import argparse

import numpy as np
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from utils.utils_1 import Custom_dataset, weights_init
from syn_module.main_vq import VQGAN_model
from syn_module.syn_model import SYN_model
from syn_module.Discriminator import Discriminator
from syn_module.recon_freeze import Recon
import datetime
import matplotlib.pyplot as plt

import wandb
wandb.login(key='6b037e771b020c9d419f6468780b6a0640a9e9eb')

time = datetime.datetime.now().strftime("%d-%m-%H-%M")



class TrainVQ:
    def __init__(self, args):
        self.syn_model = SYN_model(args)
        self.mainvq = VQGAN_model(args)
        self.recon = Recon(args)


        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        self.opt_syn, self.opt_disc = self.conf_optimizer(args)
        self.train_model(args=args)

    def conf_optimizer(self, args):
        lr = args.learning_rate
        opt_syn = torch.optim.Adam(
            list(self.syn_model.encoder.parameters()) +
            list(self.syn_model.quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))

        return opt_syn, opt_disc

    def train_model(self, args):
        transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((256, 256)), transforms.ToTensor(),
                                        transforms.Normalize([0], [1])])
        train_data = Custom_dataset(args.real_data, args.unreal_data, transform=transform)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
        total_samples = len(train_loader)*args.batch_size
        # Encoder
        generator = self.syn_model
        checkpoint = torch.load(args.checkpoints)
        # main
        main_vq = self.mainvq
        #model_dict = main_vq.state_dict()
        lst = ['encoder', 'quant_conv']
        pretrained_dict = {x: y for x, y in checkpoint['state_dict'].items() if x.split('.', 1)[0] in lst}
        main_vq.load_state_dict(pretrained_dict)
        main_vq.eval()
        main_vq.requires_grad_(False)
        print('main:', main_vq.training)

        recon = self.recon
        lst_recon = ['decoder', 'post_quant_conv'] #,'quantize'
        pretrained_dict2 = {x: y for x, y in checkpoint['state_dict'].items() if x.split('.', 1)[0] in lst_recon}
        recon.load_state_dict(pretrained_dict2)
        recon.eval()
        recon.requires_grad_(False)
        print('recon:', recon.training)
        generator = generator.to(torch.device("cuda"))
        criterion = nn.BCELoss()
        criterion_GAN = nn.MSELoss()
        criterion_pixelwise = nn.L1Loss()

        real = 1  # np.random.uniform(0.7,1.2)
        fake = 0  # np.random.uniform(0.0,0.3)
        main_vq = main_vq.to(torch.device("cuda"))
        recon = recon.to(torch.device("cuda"))
        for epoch in range(0, args.epochs):
            with tqdm(range(len(train_loader))) as pbar:
                for i, img in zip(pbar, train_loader):

                    real_1 = img['A'].float().to(torch.device("cuda"))
                    real_2 = img['B'].float().to(torch.device("cuda"))

                    fake_lat = generator(real_2)
                    real_lat = main_vq(real_1)

                    # train discriminator
                    self.opt_disc.zero_grad()
                    d_real = self.discriminator(real_lat)
                    label = torch.tensor(real).expand_as(d_real).float().to(torch.device('cuda'))
                    d_real_loss = criterion_GAN(d_real, label)
                    d_real_loss.backward()

                    label.fill_(fake)
                    d_fake = self.discriminator(fake_lat.detach())
                    d_fake_loss = criterion_GAN(d_fake, label)
                    d_fake_loss.backward(retain_graph=True)
                    gan_loss = 0.5 * (d_real_loss + d_fake_loss)
                    self.opt_disc.step()

                    # train generator
                    self.opt_syn.zero_grad()
                    label.fill_(real)
                    d_fake = self.discriminator(fake_lat)
                    g_loss = criterion_GAN(d_fake, label)
                    g_loss.backward(retain_graph=True)
                    self.opt_syn.step()

                    checkpoints = {
                        "epoch": epoch, "state_dict": generator.state_dict(),
                    }
                    wandb.log({'D_loss': gan_loss.mean().cpu().detach().numpy().item(),
                               'g_loss': g_loss.cpu().detach().abs().numpy().item(), 'Epoch': epoch})

                    pbar.set_postfix(
                        Epoch_no=epoch,
                        D_loss=np.round(gan_loss.mean().cpu().detach().numpy().item(), 5),
                        G_loss=np.round(g_loss.cpu().detach().abs().numpy().item(), 5)
                    )
                    pbar.update(0)
                    if epoch % 30 == 0 and i == total_samples//args.batch_size-50:
                        decoder_image = recon(fake_lat)
                        real_fake_images = torch.cat((real_2[:4], decoder_image[:4]))  # add(1).mul(0.5)
                        save_image(real_fake_images,os.path.join("/home/moravapa/Documents/images/task1",f"task1_{epoch}.jpg"), nrow=4)


                    if epoch % 50 == 0 and i == total_samples//args.batch_size-50:
                        with torch.no_grad():
                            torch.save(checkpoints,os.path.join("/scratch/moravapa/chkpt/task1", f"task1_{epoch}.pt"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GAN")
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
    parser.add_argument('--batch-size', type=int, default=6, help='Input batch size for training (default: 6)')
    parser.add_argument('--dropout', type=int, default=0.0, help='droupout')
    parser.add_argument('--num_workers', type=int, default=8, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=301, help='Number of epochs to train (default: 50)')
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
    parser.add_argument('--model', type=str, default='task_1', help='For saving the log for each model')
    args = parser.parse_args()
    args.real_data = r'/home/moravapa/Documents/Thesis/data/train1.txt'
    args.unreal_data = r'/home/moravapa/Documents/Thesis/data/train2.txt'
    args.checkpoints = r'/home/moravapa/Documents/ckpt/last.ckpt'
    #args.enc_checkpoints = r'/home/moravapa/Documents/Thesis/synthetic/checkpoints/enc_synn_100.pt'

    base_dir = r'/scratch/moravapa/outputs'

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
