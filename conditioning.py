import torch
from torch import Tensor
import torch.nn as nn

# Sinusoidal Time Embedding Module

class TimeEmbedding(nn.Module):
    '''
    via https://huggingface.co/blog/annotated-diffusion 
    '''
    def __init__(self, dim: int, i: int = 10000):   
        super().__init__()
        self.dim = dim
        self.i = i

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(self.i)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:  # zero pad if dim is odd
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb
        

# Image Conditioning Module
# input ViT features and output conditioning features for the unet model
class ImageConditioning(nn.Module):

    def __init__(self, in_channels: int, cond_channels: int):
        super().__init__()
        

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:

    