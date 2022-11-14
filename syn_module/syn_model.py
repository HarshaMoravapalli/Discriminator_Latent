import torch
import torch.nn as nn

# from modules.vqgan_architecture import Encoder, Decoder
from syn_module.model import Encoder, Decoder
from modules.codebook import Codebook
from modules.discriminator import Discriminator
from utils.utils import weights_init


class SYN_model(nn.Module):
    def __init__(self, args):
        super(SYN_model, self).__init__()
        self.encoder = Encoder(args)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1)

    def forward(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_syn = self.quant_conv(encoded_images)

        return quant_syn

