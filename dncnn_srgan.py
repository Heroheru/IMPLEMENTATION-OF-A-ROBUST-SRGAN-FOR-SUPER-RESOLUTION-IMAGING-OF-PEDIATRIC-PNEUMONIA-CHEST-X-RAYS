
from DNCNN import DnCNN
from srgan_model import Generator, Discriminator
import torch
import torch.nn as nn

class DNCNN_SRGAN(nn.Module):
    def __init__(self, channels = 1, 
                 num_of_layers=17, 
                 img_feat=1,
                 n_feats=64,
                 num_block=16,
                 scale=4):
        
        super(DNCNN_SRGAN, self).__init__()
        self.denoising_module = DnCNN(channels,num_of_layers)
        self.generator = Generator(
            img_feat=img_feat,
            n_feats=n_feats,
            num_block=num_block,
            scale=scale
        )

    def forward(self, x):

        # SRGAN -> DnCNN
        x_dncnn = (x + 1.0) / 2.0

        noise = self.denoising_module(x_dncnn)
        denoised = x_dncnn - noise

        denoised = torch.clamp(denoised, 0.0, 1.0)

        # DnCNN -> SRGAN
        denoised = denoised * 2.0 - 1.0

        sr, feat = self.generator(denoised)

        return sr, feat
    # def forward(self, x):

    #     noise = self.denoising_module(x)
    #     denoising_image = x - noise

    #     # PERBAIKAN WAJIB: Kunci di rentang [0.0, 1.0]
    #     denoising_image = torch.clamp(denoising_image, 0.0, 1.0)
        
    #     sr_image, feat = self.generator(denoising_image)
    #     return sr_image, feat
    
