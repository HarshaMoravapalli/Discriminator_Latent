import torch
import torch.nn as nn

from modules.model import Encoder,Decoder
from modules.vec_quan import VectorQuantizer2 as VectorQuantizer

class VQModel(nn.Module):
    def __init__(self, args):
        super(VQModel, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.quantize = VectorQuantizer(args)
        self.quant_conv = nn.Conv2d(args.latent_dim,args.latent_dim,1)
        self.post_quant_conv = nn.Conv2d(args.latent_dim,args.latent_dim,1)

    def forward(self,image):
        i = self.encoder(image)
        i = self.quant_conv(i)
        quant,code_loss, info = self.quantize(i)
        quant = self.post_quant_conv(quant)
        dec_img = self.decoder(quant)
        return dec_img, code_loss, info

    
    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor
    
    def calculate_lambda(self, perceptual_loss, gan_loss):
        #import pdb;pdb.set_trace()
        last_layer = self.decoder.conv_out
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ


