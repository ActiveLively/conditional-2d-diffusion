import torch
from torch import Tensor, einsum
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from conditioning import TimeEmbedding

    
class DownBlockLossless(nn.Module):
    '''
    Downsamples the sequence length by a factor of 2. 
    If the input sequence is (batch_size x channels x length) 
    the output sequence will be (batch_size x channels x length // 2)

    from Tristan Shah
    '''
    def __init__(self, in_channels: int, out_channels):
        super().__init__()

        self.rearrange = Rearrange('b c (l p1) (w p2) -> b (c p1 p2) l w', p1 = 2, p2 = 2)
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size = 3, padding = 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.rearrange(x))

class DownBlockStride(nn.Module):
    '''
    Strided Convolution 
    '''
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)
    
class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.upsample(x))

class RMSNorm(nn.Module):
    def __init__(self, in_channels: int, eps: float = 1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.eps = eps

    def forward(self, x: Tensor, time_emb= None) -> Tensor:
        mean = torch.mean(x**2, dim=1, keepdim=True)
        inv_rms = torch.rsqrt(mean + self.eps)
        return x * inv_rms * self.scale
    
class AdaRMSNorm(nn.Module):
    def __init__(self, in_channels: int, time_emb_dim: int, eps: float = 1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.time_emb_dim = time_emb_dim
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, in_channels * 2)
        )

    def forward(self, x: Tensor, time_emb: Tensor) -> Tensor:
        scale_shift = self.mlp(time_emb)
        scale_shift = rearrange(scale_shift, 'b c -> b c 1 1')

        scale, shift = scale_shift.chunk(2, dim=1)
        mean = torch.mean(x**2, dim=1, keepdim=True)
        inv_rms = torch.rsqrt(mean + self.eps)
        return x * inv_rms * (1 + scale) + shift
    
class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.act = nn.SiLU()
        self.norm = nn.GroupNorm(8, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: Tensor, scale: Tensor = None, shift: Tensor = None) -> Tensor:
        # apply norm, act, then conv (Pre-norm)
        x =self.norm(x)
        if scale is not None and shift is not None:
            x = x * (1 + scale) + shift
        return self.conv(self.act(x))
    
class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp_in = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, in_channels * 2)
        )
        self.time_mlp_out = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)
        )
        self.block1 = Block(in_channels, out_channels)
        self.block2 = Block(out_channels, out_channels)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor, time_emb: Tensor) -> Tensor:
        time_emb_in = self.time_mlp_in(time_emb)
        time_emb_in = rearrange(time_emb_in, 'b c -> b c 1 1')
        scale1, shift1 = time_emb_in.chunk(2, dim=1)

        time_emb_out = self.time_mlp_out(time_emb)
        time_emb_out = rearrange(time_emb_out, 'b c -> b c 1 1')
        scale2, shift2 = time_emb_out.chunk(2, dim=1)

        h = self.block1(x, scale1, shift1)
        h = self.block2(h, scale2, shift2)
        return h + self.skip(x)

        
class CrossAttention(nn.Module):
    '''
    Docstring for CrossAttention
    https://arxiv.org/pdf/2106.05786 
    '''
    def __init__(self, in_channels: int, context_dim: int, heads: int, head_dim: int, time_emb_dim: int = None):
        super().__init__()
        self.heads = heads
        self.scale = head_dim ** -0.5
        hidden_dim = heads * head_dim

        # if time_emb_dim is not None:
        #     self.norm = AdaRMSNorm(in_channels, time_emb_dim)
        # else:
        #     self.norm = RMSNorm(in_channels, scale=1.0)


        self.q = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.to_kv = nn.Linear(context_dim, 2 * hidden_dim, bias=False)

        self.output = nn.Sequential(
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1),
            RMSNorm(in_channels)
        )
    
    def forward(self, x: Tensor, context: Tensor, time_emb: Tensor = None) -> Tensor:
        residual = x
        b, c, l, w = x.shape
        q = self.q(x)
        kv = self.to_kv(context)
        k, v = kv.chunk(2, dim=-1)

        q = rearrange(q, 'b (h c) l w -> b h c (l w)', h=self.heads)

        k = rearrange(k, 'b s (h c) -> b h c s', h=self.heads)
        v = rearrange(v, 'b s (h c) -> b h c s', h=self.heads)

        q = q * self.scale

        attn = torch.einsum('b h c i, b h c j -> b h i j', q, k)
        attn = torch.softmax(attn, dim=-1)

        sim = torch.einsum('b h c i, b h c j -> b h i j', q, k)
        sim = sim - torch.amax(sim, dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # (batch, heads, channels, length + width)
        attn_output = torch.einsum('b h i j, b h c j -> b h i c', attn, v)
        attn_output = rearrange(attn_output, 'b h (l w) c -> b (h c) l w', l=l, w=w)
        return self.output(attn_output) + residual


class LinearAttention(nn.Module):
    def __init__(self, in_channels: int, heads: int, head_dim: int, time_emb_dim: int = None):
        super().__init__()
        self.heads = heads
        self.scale = head_dim ** -0.5
        hidden_dim = heads * head_dim

        # use AdaRMSNorm if time_emb_dim is provided, otherwise use RMSNorm
        if time_emb_dim is not None:
            self.norm = AdaRMSNorm(in_channels, in_channels, time_emb_dim)
        else:
            self.norm = RMSNorm(in_channels)
        
        self.qkv = nn.Conv2d(in_channels, hidden_dim * 3, kernel_size=1, bias=False)
        self.output = nn.Sequential(
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1),
            RMSNorm(in_channels, scale=1.0)
        )

    def forward(self, x: Tensor, time_emb: Tensor = None) -> Tensor:
        residual = x
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x, time_emb))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (h c) l w -> b h c (l w)', h=self.heads)
        k = rearrange(k, 'b (h c) l w -> b h c (l w)', h=self.heads)
        v = rearrange(v, 'b (h c) l w -> b h c (l w)', h=self.heads)

        q = torch.softmax(q, dim=2) * self.scale
        k = torch.softmax(k, dim=3)

        context = torch.einsum('b h c i, b h c j -> b h i j', k, v) * self.scale
        out = torch.einsum('b h i j, b h c i -> b h j c', context, q)

        attn_output = rearrange(out, 'b h c (l w) -> b (h c) l w', l=h, w=w)
        return self.output(attn_output) + residual
        


class SelfAttention(nn.Module):
    def __init__(self, in_channels: int, heads: int, head_dim: int, time_emb_dim: int = None):
        super().__init__()
        self.heads = heads
        self.scale = head_dim ** -0.5
        hidden_dim = heads * head_dim

        # use AdaRMSNorm if time_emb_dim is provided, otherwise use RMSNorm
        # if time_emb_dim is not None:
        #     self.norm = AdaRMSNorm(in_channels, time_emb_dim)
        # else:
        #     self.norm = RMSNorm(in_channels, self.scale)

        self.qkv = nn.Conv2d(in_channels, hidden_dim * 3, kernel_size=1, bias=False )
        self.output = nn.Sequential(
            ## MODIFY ##
            nn.Conv2d(hidden_dim, in_channels, kernel_size = 1),
            RMSNorm(in_channels, self.scale)
        )
    
    def forward(self, x: Tensor, time_emb: Tensor = None) -> Tensor:
        residual = x
        b, c, h, w = x.shape
        # if time_emb is not None:
        #     x_norm = self.norm(x, time_emb)
        # else:
        #     x_norm = self.norm(x)

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (h c) l w -> b h c (l w)', h=self.heads)
        # q = rearrange(q, 'b (h c) l w -> b h (l w) c', h=self.heads)
        k = rearrange(k, 'b (h c) l w -> b h c (l w)', h=self.heads)
        v = rearrange(v, 'b (h c) l w -> b h c (l w)', h=self.heads)

        # attention formula: softmax(QK^T / sqrt(d_k))V
        q = q * self.scale
        sim = torch.einsum('b h c i, b h c j -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # attn_output = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        # (batch, heads, channels, length + width)
        attn_output = torch.einsum('b h i j, b h c j -> b h i c', attn, v)

        attn_output = rearrange(attn_output, 'b h (l w) c -> b (h c) l w', l=h, w=w)
        return self.output(attn_output) + residual
    
class GEGLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self._to_x_gate= nn.Conv2d(in_channels, out_channels * 2, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x_gate = self._to_x_gate(x)
        x, gate = x_gate.chunk(2, dim=1)
        return x * F.gelu(gate)
    
class FeedForward(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        inner_dim = out_channels * 4
        self.ff = nn.Sequential(
            #nn.Linear(in_channels, out_channels),
            GEGLU(in_channels, inner_dim),
            nn.Conv2d(inner_dim, in_channels, kernel_size=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ff(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, in_channels: int, heads: int, head_dim: int, context_dim: int, time_emb_dim: int = None):
        super().__init__()

        # Attention modules
        self.self_attn = SelfAttention(in_channels, heads, head_dim, time_emb_dim)
        self.cross_attn = CrossAttention(in_channels, context_dim, heads, head_dim, time_emb_dim)

        # Norm depending on time_emb
        if time_emb_dim is not None:
            self.norm1 = AdaRMSNorm(in_channels, time_emb_dim)
            self.norm2 = AdaRMSNorm(in_channels, time_emb_dim)
            self.norm3 = AdaRMSNorm(in_channels, time_emb_dim)
        else:
            self.norm1 = RMSNorm(in_channels)
            self.norm2 = RMSNorm(in_channels)
            self.norm3 = RMSNorm(in_channels)

        self.context_norm = nn.LayerNorm(context_dim)
        #FF block
        self.ff = FeedForward(in_channels, in_channels)

    def forward(self, x: Tensor, context: Tensor, time_emb: Tensor = None) -> Tensor:
        #residual = x
        x = self.self_attn(self.norm1(x, time_emb), time_emb) + x
        x = self.cross_attn(self.norm2(x, time_emb), self.context_norm(context), time_emb) + x
        return self.ff(self.norm3(x, time_emb)) + x

class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, heads: int = 4, dim_head: int = 32, time_emb_dim: int = 128, out_channels: int = 3):
        super().__init__()
        
        self.tim_mlp = nn.Sequential(
            TimeEmbedding(time_emb_dim), 
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.input_layer = nn.Conv2d(in_channels, 32, kernel_size=1)

        self.downs = nn.ModuleList([
            nn.ModuleList([
                ResBlock(32, 32, time_emb_dim),
                ResBlock(32, 32, time_emb_dim),
                TransformerBlock(32, heads, dim_head, 128, time_emb_dim),
                DownBlockStride(32, 64)
            ]),

            nn.ModuleList([
                ResBlock(64, 64, time_emb_dim),
                ResBlock(64, 64, time_emb_dim),
                TransformerBlock(64, heads, dim_head, 128, time_emb_dim),
                DownBlockStride(64, 128)
            ]),

            nn.ModuleList([
                ResBlock(128, 128, time_emb_dim),
                ResBlock(128, 128, time_emb_dim),
                TransformerBlock(128, heads, dim_head, 128, time_emb_dim),
                nn.Conv2d(128, 256, kernel_size=3, padding=1)
            ])
        ])

        self.mid_block1 = ResBlock(256, 256, time_emb_dim)
        self.mid_transform = TransformerBlock(256, heads, dim_head, 128, time_emb_dim)
        #self.mid_attention = SelfAttention(128, heads, dim_head, time_emb_dim)
        self.mid_block2 = ResBlock(256, 256, time_emb_dim)

        self.ups = nn.ModuleList([
            nn.ModuleList([
                ResBlock(256 + 128, 256, time_emb_dim),
                ResBlock(256 + 128, 256, time_emb_dim),
                TransformerBlock(256, heads, dim_head, 128, time_emb_dim),
                UpBlock(256, 128)
            ]),

            nn.ModuleList([
                ResBlock(128 + 64, 128, time_emb_dim),
                ResBlock(128 + 64, 128, time_emb_dim),
                TransformerBlock(128, heads, dim_head, 128, time_emb_dim),
                UpBlock(128, 64)
            ]),

            nn.ModuleList([
                ResBlock(64 + 32, 64, time_emb_dim),
                ResBlock(64 + 32, 64, time_emb_dim),
                TransformerBlock(64, heads, dim_head, 128, time_emb_dim),
                nn.Conv2d(64, 32, kernel_size=3, padding=1)
            ])
        ])

        self.output_res = ResBlock(32 + 32, 32, time_emb_dim)
        self.output_layer = nn.Conv2d(32, out_channels, kernel_size=1)  

    def forward(self, x: Tensor,  time_steps: Tensor, context: Tensor,) -> Tensor:
        b, c, l, w = x.shape
        time_emb = self.tim_mlp(time_steps)

        y = self.input_layer(x)
        r = y.clone()

        residuals = []

        for res1, res2, transform, down in self.downs:
            y = res1(y, time_emb)
            # print("after res1: ", y.shape)
            residuals.append(y)
            # y = res2(y, time_emb)
            # print("after res2: ", y.shape)
            y = transform(y, context, time_emb)
            # print("after transform: ", y.shape)
            residuals.append(y)
            y = down(y)
            # print("after down: ", y.shape)

        y = self.mid_block1(y, time_emb)
        # print("after mid block 1: ", y.shape)
        y = self.mid_transform(y, context, time_emb)
        # print("after mid transform: ", y.shape)
        y = self.mid_block2(y, time_emb)    
        # print("after mid block 2: ", y.shape)

        for res1, res2, transform, up in self.ups:
            y = res1(torch.cat((y, residuals.pop()), dim=1), time_emb)
            # print("after res1 up: ", y.shape)
            y = res2(torch.cat((y, residuals.pop()), dim=1), time_emb)
            # print("after res2 up: ", y.shape)
            y = transform(y, context, time_emb)
            # print("after transform up: ", y.shape)
            y = up(y)
            # print("after up: ", y.shape)
            

        y = self.output_res(torch.cat((y, r), dim=1), time_emb)
        y = self.output_layer(y)
        return y
