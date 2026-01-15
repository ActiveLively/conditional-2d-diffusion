import torch
from torch import Tensor
import torch.nn as nn   
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from UNet import UNET


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

class DiffusionSampler(nn.Module):
    def __init__(self, timesteps: int, beta_schedule: str = 'scaled_linear'):
        super().__init__()

        ## computing scheduling parameters
        if beta_schedule == 'linear':
            beta = linear_beta_schedule(timesteps)
        elif beta_schedule == 'scaled_linear':
            beta = scaled_linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            beta = cosine_beta_schedule(timesteps)
        else:
            raise ValueError('Unrecognized beta_schedule.')
        alpha = 1.0 - beta
        alpha_bar = alpha.cumprod(dim = 0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value = 1.)

        ## adding as non trainable parameters
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('alpha_bar_prev', alpha_bar_prev)

    @property
    def device(self):
        return self.beta.device
    
    @property
    def timesteps(self):
        return len(self.beta)
    
    @torch.no_grad()
    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor = None):
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar = extract(self.alpha_bar, t, x0.shape) # alpha_bar is % of original signal preserved

        ## compute the forward sample q(x_t | x_0)
        ## draw a sample x_t ~ q(x_t | x_0)
        '''
        To-Do:
        See Equation (4) in: https://arxiv.org/pdf/2006.11239

        Also see Algoithm 1 (Training)
        '''
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0-alpha_bar)*noise # the sqrt of variance (alpha_bar) =  amplitude component that works w/ the pixel values
        
        return xt, noise
    
    @torch.no_grad()
    def p_sample(self, model: nn.Module, xt: Tensor, t: Tensor) -> Tensor:
        beta               = extract(self.beta, t, xt.shape)
        alpha              = extract(self.alpha, t, xt.shape)
        alpha_bar          = extract(self.alpha_bar, t, xt.shape)
        alpha_bar_prev     = extract(self.alpha_bar_prev, t, xt.shape)
        not_first_timestep = add_singletons_like(t > 0, xt.shape)
        ## compute mu and variance of the reverse sampling distribution p(x_{t-1} | x_t)
        ## draw a sample x_{t-1} ~ p(x_{t-1} | x_t)
        '''
        To-Do:
        See Equation (7) in: https://arxiv.org/pdf/2006.11239

        Also see Algorithm (2) Sampling
        '''
        mu = torch.divide(1.0, torch.sqrt(alpha)) * (xt - torch.divide(beta, torch.sqrt(1.0-alpha_bar)) *  model(xt, t))
        beta_tilda = torch.divide( (1.0 - alpha_bar_prev), (1.0 - alpha_bar)) * beta
        noise = torch.randn_like(xt)
        sample = mu + torch.sqrt(beta_tilda) * not_first_timestep * noise

        return sample
    
    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, xT: Tensor, num_images: int = 1) -> Tensor:

        xt = xT.clone()

        if num_images > 1:
            dt = int(self.timesteps / num_images)
            sequence = []
            times = []

        for i in reversed(range(self.timesteps)):
            t = torch.ones(xT.shape[0], device = xT.device).long() * i
            xt = self.p_sample(model, xt, t)

            if num_images > 1 and (i % dt == 0 or i == self.timesteps - 1):
                sequence.append(xt)
                times.append(i)

        if num_images > 1:
            return sequence, times
        else:
            return xt
        

