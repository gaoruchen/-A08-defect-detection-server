import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange


from .ACT import *

# 默认参数一般是1024 8
__all__ = [          
           'AIFI_Flash',
           ]

# 基础模块
class RepBN(nn.Module):
    def __init__(self, channels):
        super(RepBN, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.bn(x) + self.alpha * x
        x = x.transpose(1, 2)
        return x


class LinearNorm(nn.Module):
    def __init__(self, dim, norm1, norm2, warm=0, step=300000, r0=1.0):
        super(LinearNorm, self).__init__()
        self.register_buffer('warm', torch.tensor(warm))
        self.register_buffer('iter', torch.tensor(step))
        self.register_buffer('total_step', torch.tensor(step))
        self.r0 = r0
        self.norm1 = norm1(dim)
        self.norm2 = norm2(dim)

    def forward(self, x):
        if self.training:
            if self.warm > 0:
                self.warm.copy_(self.warm - 1)
                x = self.norm1(x)
            else:
                lamda = self.r0 * self.iter / self.total_step
                if self.iter > 0:
                    self.iter.copy_(self.iter - 1)
                x1 = self.norm1(x)
                x2 = self.norm2(x)
                x = lamda * x1 + (1 - lamda) * x2
        else:
            x = self.norm2(x)
        return x

ln = nn.LayerNorm
linearnorm = partial(LinearNorm, norm1=ln, norm2=RepBN, step=60000)

# 基础Transformer Layer
class BaseTransformerLayer(nn.Module):
    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm1 = linearnorm(c1)
        self.norm2 = linearnorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature ** omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]


# Flash Attention实现
class FlashAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class AIFI_Flash(BaseTransformerLayer):
    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        self.flash_attn = FlashAttention(c1, num_heads, dropout)

    def forward(self, x):
        c, h, w = x.shape[1:]
        x = x.flatten(2).permute(0, 2, 1)
        x2 = self.flash_attn(x)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.fc2(self.dropout(self.act(self.fc1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()




def test_flash():
    print("\nTesting Flash Attention...")
    torch.manual_seed(42)
    batch_size, channels, height, width = 2, 256, 32, 32
    x = torch.randn(batch_size, channels, height, width)
    model = AIFI_Flash(channels)
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sample:\n{output[0, 0, :3, :3]}")





# 修改main函数
if __name__ == "__main__":
    test_flash()


