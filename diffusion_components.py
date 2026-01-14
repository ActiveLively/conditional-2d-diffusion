import torch
from torch import Tensor
import torch.nn as nn   


def add_singletons_like(v: Tensor, x_shape: tuple) -> Tensor:
    ## get batchsize (b) and data dimention (d ...)
    b, *d = x_shape
    ## add singletons for each dim in data dim
    return v.reshape(b, *[1] * len(d))

def extract(v: Tensor, t: Tensor, x_shape: tuple) -> Tensor:
    return add_singletons_like(v.gather(0, t), x_shape)

def linear_beta_schedule(timesteps: int) -> Tensor:
    return torch.linspace(1e-4, 0.02, timesteps)

def scaled_linear_beta_schedule(timesteps: int) -> Tensor:
    '''
    linear schedule, proposed in original ddpm paper
    '''
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s = 0.008):
    '''
    https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/classifier_free_guidance.py
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    '''
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)