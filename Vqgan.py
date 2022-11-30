import torch
import torch.nn as nn
 
#from modules.vqgan_architecture import Encoder, Decoder 
from model import Encoder, Decoder
from modules.codebook import Codebook

class VQGAN_model(nn.Module):
    def __init__(self,args):
        super(VQGAN_model,self).__init__()
        self.encoder = Encoder(args).apply(self.weights_enc)
        #self.decoder = Decoder(args)
        #self.codebook =  Codebook(args)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1)
        #self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1)


    def forward(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        #codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        #post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        #decoded_images = self.decoder(post_quant_conv_mapping)
        return quant_conv_encoded_images

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def weights_enc(self,m):
        print('first',m)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        pretrained_dict = {key.replace("module.", ""): value for key, value in checkpoint['model_state_dict'].items()}
        self.load_state_dict(pretrained_dict)

        #self.load_state_dict(torch.load(path))

    def calculate_lambda(self, perceptual_loss, gan_loss):
        #import pdb;pdb.set_trace()
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ


    def encode(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z):
        post_quant_conv_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(post_quant_conv_mapping)
        return decoded_images
