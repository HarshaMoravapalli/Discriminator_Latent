import torch
import torch.nn as nn

from modules.model import Encoder,Decoder
from modules.vec_quan import VectorQuantizer2 as VectorQuantizer

class Recon(nn.Module):
    def __init__(self, args):
        super(Recon, self).__init__()
        self.decoder = Decoder(args)
        #self.quantize = VectorQuantizer(args)
        self.post_quant_conv = nn.Conv2d(args.latent_dim,args.latent_dim,1)

    def forward(self,lat):
        #quant,code_loss, info = self.quantize(lat)
        quant = self.post_quant_conv(lat)
        dec_img = self.decoder(quant)
        return dec_img

