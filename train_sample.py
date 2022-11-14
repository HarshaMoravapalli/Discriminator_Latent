from ast import parse
from distutils.command.check import check
from multiprocessing.sharedctypes import Value
import os



from unicodedata import decimal
from cv2 import waitKey
from matplotlib import transforms
from tqdm import tqdm
import numpy as np
import argparse
import torch
import torch.nn as nn

import torch.nn.functional as F
from torchvision import utils as vutils
from torch.utils.data import DataLoader
from zmq import device
from Vqgan_sample import VQGAN_model
from model import Encoder,Decoder
from utils import Custom_dataset, weights_init
from modules.discriminator import Discriminator
from lpips import LPIPS
import wandb
wandb.login()
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms



import gc
gc.collect()
torch.cuda.empty_cache()

os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3"
#export CUDA_VISIBLE_DEVICES = 1,2


#print("torch1:",torch.cuda.device_count())

#device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class synthVQ:
    def __init__(self,args):
        self.encoder = Encoder(args=args)
        self.decoder = Decoder(args=args)
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)

        self.main(args)

    def main(self,args):
        transform = transforms.Compose([transforms.Resize(size=512),transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
        train_data = Custom_dataset(img_dir=args.dataset_path,size = 0.25,transform=transform)
        train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=False)
        steps_per_epoch = len(train_loader)
        #encoder_dict = self.encoder.state_dict()
        checkpoint = torch.load(args.checkpoints)
        pretrained_dict = {key.replace("module.", ""): value for key, value in checkpoint['model_state_dict'].items()}

        #for key in encoder_dict:
            #print(key)
           
        encoder = {key[8:]:value for key, value in pretrained_dict.items() if 'encoder' in key}
        decoder_dict = {key[8:]:value for key, value in pretrained_dict.items() if 'decoder' in key}
        codebook_dict = {key[9:]:value for key, value in pretrained_dict.items() if 'codebook' in key}
        
        self.encoder.load_state_dict(encoder)
        print(codebook_dict['embedding_weight'])
            
                

        #encoder = {key:value for key, value in pretrained_dict.items() if key in encoder_dict}
        #print(encoder)
        #print(pretrained_dict)
#

        



        

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="VQ-GAN")
    parser.add_argument('--latent-dim', type=int, default=512, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=512, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('-g', '--gpus', default=1, type=int,help='number of gpus per node')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=4, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=2500, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--checkpoints', type=str, default='/checkpoints', help='load save checkpoints')
    parser.add_argument('--pretrain', type=bool, default=True, help='load save checkpoints')
    parser.add_argument('--model', type=str, default='vqgan', help='model name')
    
    args = parser.parse_args()
    args.dataset_path = r'/home/moravapa/Documents/Thesis/VQ_GAN/Data/train'
    args.checkpoints = r'/home/moravapa/checkpoints/resume/vqgan_227.pt'

    
    '''wandb.init(project="run_1")
    wandb.config.update(args)
    config = wandb.config'''

    train_vggan = synthVQ(args)