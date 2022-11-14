import torch
import torch.nn as nn

# from modules.vqgan_architecture import Encoder, Decoder
from modules.model import Encoder, Decoder
from modules.codebook import Codebook


class VQGAN_model(nn.Module):
    def __init__(self, args):
        super(VQGAN_model, self).__init__()
        self.encoder = Encoder(args)
        #self.codebook = Codebook(args)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1)


    def forward(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        #codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        return quant_conv_encoded_images#, codebook_mapping,codebook_indices, q_loss

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor






