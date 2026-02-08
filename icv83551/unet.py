import copy
import torch
from torch import nn
import torch.nn.functional as F
import math


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def Upsample(dim, dim_out=None):
    """Upsample the image feature resolution a factor of 2."""
    # CRITICAL: Must use 'nearest' mode, not 'bilinear'
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    """Downsample the image feature resolution a factor of 2."""
    return nn.Conv2d(dim, default(dim_out, dim), kernel_size=2, stride=2)


class RMSNorm(nn.Module):
    """RMSNorm layer which is compute-efficient simplified variant of LayerNorm."""

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-8
        rms = torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + eps)
        return (x / rms) * self.g
        

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embedding for time steps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        # CRITICAL: Must use (half_dim - 1), not half_dim
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(nn.Module):
    """A conv block with feature modulation."""

    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.GELU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """A ResNet-like block with context dependent feature modulation."""

    def __init__(self, dim, dim_out, context_dim):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.context_dim = context_dim

        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(context_dim, dim_out * 2))
            if exists(context_dim)
            else None
        )

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        # CRITICAL: Must have dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, context=None):
        scale_shift = None
        if exists(self.mlp) and exists(context):
            context = self.mlp(context)
            # rearrange(context, "b c -> b c 1 1") equivalent
            context = context.reshape(context.shape[0], context.shape[1], 1, 1)
            scale_shift = context.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        # CRITICAL: Must apply dropout
        h = self.dropout(h)
        h = self.block2(h)
        return h + self.res_conv(x)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        condition_dim,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        uncond_prob=0.2,
    ):
        super().__init__()

        self.init_conv = nn.Conv2d(channels, dim, 3, padding=1)
        self.channels = channels

        dims = [dim] + [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        in_out_ups = [(b, a) for a, b in reversed(in_out)]

        context_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

        self.condition_dim = condition_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

        self.uncond_prob = uncond_prob

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        # Downsampling blocks
        for ind, (dim_in, dim_out) in enumerate(in_out):
            down_block = nn.ModuleList([
                ResnetBlock(dim_in, dim_in, context_dim=context_dim),
                ResnetBlock(dim_in, dim_in, context_dim=context_dim),
                Downsample(dim_in, dim_out),
            ])
            self.downs.append(down_block)

        # Middle blocks
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, context_dim=context_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, context_dim=context_dim)

        # Upsampling blocks
        for ind, (dim_in, dim_out) in enumerate(in_out_ups):
            up_block = nn.ModuleList([
                Upsample(dim_in, dim_out),
                ResnetBlock(dim_out * 2, dim_out, context_dim=context_dim),
                ResnetBlock(dim_out * 2, dim_out, context_dim=context_dim),
            ])
            self.ups.append(up_block)

        self.final_conv = nn.Conv2d(dim, channels, 1)

    def cfg_forward(self, x, time, model_kwargs={}):
        """Classifier-free guidance forward pass."""
        cfg_scale = model_kwargs.pop("cfg_scale")
        print("Classifier-free guidance scale:", cfg_scale)
        model_kwargs = copy.deepcopy(model_kwargs)

        eps_cond = self.forward(x, time, model_kwargs)
        eps_uncond = self.forward(x, time, {**model_kwargs, "text_emb": None})
        x = (cfg_scale + 1) * eps_cond - cfg_scale * eps_uncond

        return x

    def forward(self, x, time, model_kwargs={}):
        if "cfg_scale" in model_kwargs:
            return self.cfg_forward(x, time, model_kwargs)
    
        # Context from time
        context = self.time_mlp(time.float())
    
        # Get condition embedding
        cond_emb = model_kwargs["text_emb"]
        if cond_emb is None:
            cond_emb = torch.zeros(x.shape[0], self.condition_dim, device=x.device)
    
        # CRITICAL: Apply masking during training
        if self.training:
            mask = (torch.rand(cond_emb.shape[0], device=x.device) > self.uncond_prob).float()
            cond_emb = cond_emb * mask[:, None]
    
        # Add condition to context
        context = context + self.condition_mlp(cond_emb)
    
        # Initial convolution
        x = self.init_conv(x)
    
        # Downsampling path - append AFTER each ResNet
        skips = []
        for resnet1, resnet2, downsample in self.downs:
            x = resnet1(x, context)
            skips.append(x)
            
            x = resnet2(x, context)
            skips.append(x)
            
            x = downsample(x)
    
        # Middle blocks
        x = self.mid_block1(x, context)
        x = self.mid_block2(x, context)
    
        # Upsampling path - pop BEFORE each ResNet
        for upsample, resnet1, resnet2 in self.ups:
            x = upsample(x)
            
            x = torch.cat([x, skips.pop()], dim=1)
            x = resnet1(x, context)
            
            x = torch.cat([x, skips.pop()], dim=1)
            x = resnet2(x, context)
    
        return self.final_conv(x)

