import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange

# 默认参数一般是1024 8
__all__ = [
           'AIFI_SLA',
           'AIFI_Flash',
           'AIFI_Dynamic',
           'AIFI_SLA_Flash',
           'AIFI_SLA_Dynamic',
           'AIFI_Flash_Dynamic',
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


class SmoothGELU(nn.Module):
    def __init__(self, alpha=1.0):
        super(SmoothGELU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return x * torch.sigmoid(self.alpha * x)

class LearnableGELU(nn.Module):
    def __init__(self):
        super(LearnableGELU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))  # 可学习参数

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + self.alpha * torch.pow(x, 3))))


class GELUWithDropout(nn.Module):
    def __init__(self, p=0.1):
        super(GELUWithDropout, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        gelu = 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
        return self.dropout(gelu)
    
class GELUResidual(nn.Module):
    def __init__(self):
        super(GELUResidual, self).__init__()

    def forward(self, x):
        gelu = 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
        return x + gelu  # 残差连接

class GELUWithLayerNorm(nn.Module):
    def __init__(self, hidden_size):
        super(GELUWithLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.layer_norm(x)
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))



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


class Swish(nn.Module):
    """
    Swish activation function: x * sigmoid(beta * x)
    beta is a learnable parameter initialized to 1.0
    """
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(1))  # 初始化beta为1.0
        
    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class HardSiLU(nn.Module):
    """
    Hard-SiLU activation function: x * min(max(0, x + 3)/6, 1)
    A computationally efficient approximation of SiLU
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.clamp(torch.clamp(x + 3, min=0) / 6, max=1)
    
class ModifiedSILU(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        return x * torch.sigmoid(self.alpha * x + self.beta)
    
class SpatialSILU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 生成channel-wise和spatial-wise的注意力权重
        self.spatial_alpha = nn.Conv2d(channels, channels, 1, groups=channels)
        
    def forward(self, x):
        x = x.unsqueeze(0)
        # 每个位置都有独立的activation斜率
        weights = self.spatial_alpha(x)
        return x * torch.sigmoid(weights * x)
    

class AttentionSILU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//4, 1),
            nn.ReLU(),
            nn.Conv2d(channels//4, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.unsqueeze(0)    
        att = self.channel_attention(x)
        return x * torch.sigmoid(att * x)
class EdgeAwareSILU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.edge_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.alpha = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        x = x.unsqueeze(0)
        # 提取局部结构信息来调制激活函数
        edge_info = self.edge_conv(x) - x
        return x * torch.sigmoid(x + self.alpha * edge_info)
class ContrastAwareSILU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.contrast_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, groups=channels)
        )
        
    def forward(self, x):
        x = x.unsqueeze(0)
        # 利用局部对比度信息来调整激活强度
        mean = self.contrast_pool(x)
        contrast = torch.abs(x - mean)
        return x * torch.sigmoid(x * (1 + contrast))

class AIFI_Flash(BaseTransformerLayer):
    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, 
                 act=SpatialSILU(channels = 256), normalize_before=False):
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

# SLA实现
class SimplifiedLinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.relu(q) * self.scale
        k = F.relu(k)
        context = k.transpose(-2, -1) @ v
        out = (q @ context)
        out = out.transpose(1, 2).reshape(B, N, C)
        _, n, c = out.shape
        h = w = int(n ** 0.5)
        out = out.transpose(1, 2).reshape(B, c, h, w)
        out = self.dwconv(out)
        out = out.flatten(2).transpose(1, 2)
        out = self.proj(out)
        return out
    
class AIFI_SLA(BaseTransformerLayer):
    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        self.sla = SimplifiedLinearAttention(c1, num_heads)

    def forward(self, x):
        c, h, w = x.shape[1:]
        x = x.flatten(2).permute(0, 2, 1)
        x2 = self.sla(x)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.fc2(self.dropout(self.act(self.fc1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()



# Dynamic Token Selection实现
class DynamicTokenSelector(nn.Module):
    def __init__(self, dim, mlp_ratio=0.25):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.token_score = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        nn.init.constant_(self.token_score[-1].bias, 1.0)
        nn.init.uniform_(self.token_score[-1].weight, -0.001, 0.001)

    def forward(self, x, keep_ratio=0.7):
        scores = self.token_score(x).squeeze(-1)
        scores = torch.sigmoid(scores)
        num_keep = max(1, int(x.shape[1] * keep_ratio))
        _, keep_indices = torch.topk(scores, num_keep, dim=1)
        batch_indices = torch.arange(x.size(0), device=x.device)[:, None].expand(-1, num_keep)
        return x[batch_indices, keep_indices], keep_indices


class AIFI_Dynamic(BaseTransformerLayer):
    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        self.token_selector = DynamicTokenSelector(c1)
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        c, h, w = x.shape[1:]
        x = x.flatten(2).permute(0, 2, 1)
        selected_tokens, keep_indices = self.token_selector(x)
        attn_out = self.ma(selected_tokens, selected_tokens, selected_tokens)[0]
        output = torch.zeros_like(x)
        batch_indices = torch.arange(x.size(0), device=x.device)[:, None].expand(-1, keep_indices.size(1))
        output[batch_indices, keep_indices] = attn_out
        x = x + self.dropout1(output)
        x = self.norm1(x)
        x2 = self.fc2(self.dropout(self.act(self.fc1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()


# 新增组合实现
class AIFI_SLA_Flash(BaseTransformerLayer):
    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        self.sla = SimplifiedLinearAttention(c1, num_heads)
        self.flash_attn = FlashAttention(c1, num_heads, dropout)

    def forward(self, x):
        c, h, w = x.shape[1:]
        x = x.flatten(2).permute(0, 2, 1)

        # 结合SLA和Flash Attention的输出
        x_sla = self.sla(x)
        x_flash = self.flash_attn(x)
        x2 = x_sla + x_flash  # 简单加权组合

        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.fc2(self.dropout(self.act(self.fc1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()


class AIFI_SLA_Dynamic(BaseTransformerLayer):
    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        self.sla = SimplifiedLinearAttention(c1, num_heads)
        self.token_selector = DynamicTokenSelector(c1)
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        c, h, w = x.shape[1:]
        x = x.flatten(2).permute(0, 2, 1)

        # SLA输出
        x_sla = self.sla(x)

        # Dynamic Token Selection输出
        selected_tokens, keep_indices = self.token_selector(x)
        attn_out = self.ma(selected_tokens, selected_tokens, selected_tokens)[0]
        output = torch.zeros_like(x)
        batch_indices = torch.arange(x.size(0), device=x.device)[:, None].expand(-1, keep_indices.size(1))
        output[batch_indices, keep_indices] = attn_out

        # 组合两种输出
        x2 = x_sla + output

        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.fc2(self.dropout(self.act(self.fc1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()


class AIFI_Flash_Dynamic(BaseTransformerLayer):
    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        self.flash_attn = FlashAttention(c1, num_heads, dropout)
        self.token_selector = DynamicTokenSelector(c1)
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        c, h, w = x.shape[1:]
        x = x.flatten(2).permute(0, 2, 1)

        # Flash Attention输出
        x_flash = self.flash_attn(x)

        # Dynamic Token Selection输出
        selected_tokens, keep_indices = self.token_selector(x)
        attn_out = self.ma(selected_tokens, selected_tokens, selected_tokens)[0]
        output = torch.zeros_like(x)
        batch_indices = torch.arange(x.size(0), device=x.device)[:, None].expand(-1, keep_indices.size(1))
        output[batch_indices, keep_indices] = attn_out

        # 组合两种输出
        x2 = x_flash + output

        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.fc2(self.dropout(self.act(self.fc1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()
    


def test_sla():
    print("\nTesting SLA...")
    torch.manual_seed(42)
    batch_size, channels, height, width = 2, 256, 32, 32
    x = torch.randn(batch_size, channels, height, width)
    model = AIFI_SLA(channels)
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sample:\n{output[0, 0, :3, :3]}")


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


def test_dynamic():
    print("\nTesting Dynamic Token Selection...")
    torch.manual_seed(42)
    batch_size, channels, height, width = 2, 256, 32, 32
    x = torch.randn(batch_size, channels, height, width)
    model = AIFI_Dynamic(channels)
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sample:\n{output[0, 0, :3, :3]}")

# 新增测试函数
def test_sla_flash():
    print("\nTesting SLA + Flash Attention...")
    torch.manual_seed(42)
    batch_size, channels, height, width = 2, 256, 32, 32
    x = torch.randn(batch_size, channels, height, width)
    model = AIFI_SLA_Flash(channels)
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sample:\n{output[0, 0, :3, :3]}")


def test_sla_dynamic():
    print("\nTesting SLA + Dynamic Token Selection...")
    torch.manual_seed(42)
    batch_size, channels, height, width = 2, 256, 32, 32
    x = torch.randn(batch_size, channels, height, width)
    model = AIFI_SLA_Dynamic(channels)
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sample:\n{output[0, 0, :3, :3]}")


def test_flash_dynamic():
    print("\nTesting Flash Attention + Dynamic Token Selection...")
    torch.manual_seed(42)
    batch_size, channels, height, width = 2, 256, 32, 32
    x = torch.randn(batch_size, channels, height, width)
    model = AIFI_Flash_Dynamic(channels)
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sample:\n{output[0, 0, :3, :3]}")


# 修改main函数
if __name__ == "__main__":
    # 测试单个模块
    test_sla()
    test_flash()
    test_dynamic()

    # 测试组合模块
    test_sla_flash()
    test_sla_dynamic()
    test_flash_dynamic()


