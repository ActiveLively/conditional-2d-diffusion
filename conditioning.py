from tkinter import Image
import torch
from torch import Tensor
import torch.nn as nn
from torchvision import datasets, transforms

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
class LeNet5(nn.Module):
    def __init__(self, in_channels: int, cond_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(128 * 8 * 8, 256)
        self.linear2 = nn.Linear(256, cond_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
def get_image_condition(file_path: str, condition_channels: int, device = 'cpu'):
    transform = transforms.Compose([
        transforms.Resize((32,32)), 
        transforms.ToTensor()
    ])
    img = Image.open(file_path).convert('L')
    img_resized = transform(img).unsqueeze(0).to(device)
    lenet5 = LeNet5(in_channels = 1, cond_channels = condition_channels).to(device)
    lenet5.to(device)
    lenet5.eval()

    with torch.no_grad():
        context = lenet5(img_resized)

    return context, context.shape 