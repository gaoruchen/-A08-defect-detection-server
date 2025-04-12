import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange


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

class ExpandedGatingActivation(nn.Module):
    def __init__(self, alpha_init=1.0, beta_init=0.0):
        super(ExpandedGatingActivation, self).__init__()
        # 学习参数 alpha 和 beta，用于扩展门控范围
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, x):
        # 使用 arctan 作为激活函数，并引入 alpha 和 beta 参数来扩展门控范围
        return torch.atan(self.alpha * x + self.beta)
    
class LinearOscillationActivation(nn.Module):
    def __init__(self, alpha_init=1.0, beta_init=0.0):
        super(LinearOscillationActivation, self).__init__()
        # 学习参数 alpha 和 beta，控制正弦函数的行为
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, x):
        # 通过线性和振荡结合的形式定义激活函数
        return x * torch.sin(self.alpha * x + self.beta)