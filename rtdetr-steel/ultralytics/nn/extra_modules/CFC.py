import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
__all__ = ['CFC_FFT',
           'CFC_Performer',
           'CFC_Reformer',
           'CFC_Linformer',
           'SFC_AKConv'
           ]


class LocalAttenModule(nn.Module):
    def __init__(self, in_channels=256, inter_channels=32):
        super(LocalAttenModule, self).__init__()

        self.conv = nn.Sequential(
            Conv(in_channels, inter_channels, 1),
            nn.Conv2d(inter_channels, in_channels, kernel_size=3, padding=1, bias=False))

        self.tanh_spatial = nn.Tanh()
        self.conv[1].weight.data.zero_()
        self.keras_init_weight()

    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                # nn.init.xavier_normal_(ly.weight,gain=nn.init.calculate_gain('relu'))
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        res1 = x
        res2 = x

        x = self.conv(x)
        x_mask = self.tanh_spatial(x)

        res1 = res1 * x_mask

        return res1 + res2


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    # (1, 3, 6, 8)
    # (1, 4, 8,12)
    def __init__(self, grids=(1, 2, 3, 6), channels=256):
        super(PSPModule, self).__init__()

        self.grids = grids
        self.channels = channels

    def forward(self, feats):
        b, c, h, w = feats.size()
        ar = w / h

        return torch.cat([
            F.adaptive_avg_pool2d(feats, (self.grids[0], max(1, round(ar * self.grids[0])))).view(b, self.channels, -1),
            F.adaptive_avg_pool2d(feats, (self.grids[1], max(1, round(ar * self.grids[1])))).view(b, self.channels, -1),
            F.adaptive_avg_pool2d(feats, (self.grids[2], max(1, round(ar * self.grids[2])))).view(b, self.channels, -1),
            F.adaptive_avg_pool2d(feats, (self.grids[3], max(1, round(ar * self.grids[3])))).view(b, self.channels, -1)
        ], dim=2)



class CFC_CRB(nn.Module):
    def __init__(self, in_channels=256, grids=(6, 3, 2, 1)):  # 先ce后ffm
        super(CFC_CRB, self).__init__()
        self.grids = grids
        inter_channels = in_channels // 2
        self.inter_channels = inter_channels

        self.reduce_channel = Conv(in_channels, inter_channels, 3)
        self.query_conv = nn.Conv2d(in_channels=inter_channels, out_channels=32, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=inter_channels, out_channels=32, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=inter_channels, out_channels=self.inter_channels, kernel_size=1)
        self.key_channels = 32

        self.value_psp = PSPModule(grids, inter_channels)
        self.key_psp = PSPModule(grids, inter_channels)

        self.softmax = nn.Softmax(dim=-1)

        self.local_attention = LocalAttenModule(inter_channels, inter_channels // 8)
        self.keras_init_weight()

    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                # nn.init.xavier_normal_(ly.weight,gain=nn.init.calculate_gain('relu'))
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        # Input shape: torch.Size([2, 256, 20, 20])
        x = self.reduce_channel(x)  # 降维到 inter_channels
        # After reduce_channel shape: torch.Size([2, 128, 20, 20])
        m_batchsize, _, h, w = x.size()
        query = self.query_conv(x).view(m_batchsize, 32, -1).permute(0, 2, 1)  # b c n ->  b n c
        # Query shape after query_conv and reshape: torch.Size([2, 400, 32])
        key = self.key_conv(self.key_psp(x))  # b c s
        # Key shape after key_conv and PSPModule: torch.Size([2, 32, 50])
        sim_map = torch.matmul(query, key)
        # Similarity map shape after matmul: torch.Size([2, 400, 50])
        sim_map = self.softmax(sim_map)
        # Similarity map shape after softmax: torch.Size([2, 400, 50])
        value = self.value_conv(self.value_psp(x))  # b c s
        # Value shape after value_conv and PSPModule: torch.Size([2, 128, 50])
        context = torch.bmm(value, sim_map.permute(0, 2, 1))  # B C S * B S N ->  B C N
        # Context shape after bmm: torch.Size([2, 128, 400])
        context = context.view(m_batchsize, self.inter_channels, h, w)
        # Context shape after view: torch.Size([2, 128, 20, 20])
        context = self.local_attention(context)
        # Context shape after local_attention: torch.Size([2, 128, 20, 20])
        out = x + context
        # Output shape: torch.Size([2, 128, 20, 20])
        return out


# FFT 版本
class CFC_FFTv1(nn.Module):
    def __init__(self, in_channels=256, grids=(6, 3, 2, 1)):
        super(CFC_FFTv1, self).__init__()
        self.grids = grids
        inter_channels = in_channels // 2
        self.inter_channels = inter_channels

        self.reduce_channel = Conv(in_channels, inter_channels, 3)

        # 保留FFT相关操作
        self.fft_linear = nn.Linear(inter_channels, inter_channels)
        self.conv1d = nn.Conv1d(in_channels=50, out_channels=400, kernel_size=1)
        # 添加与CFC_CRB一致的PSP模块
        self.value_psp = PSPModule(grids, inter_channels)

        # 保留局部注意力模块
        self.local_attention = LocalAttenModule(inter_channels, inter_channels // 8)

        # 添加初始化方法
        self.keras_init_weight()

    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(ly.weight)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        # Input shape: torch.Size([2, 256, 20, 20])
        x = self.reduce_channel(x)  # 降维到 inter_channels
        # After reduce_channel shape: torch.Size([2, 128, 20, 20])
        m_batchsize, _, h, w = x.size()
        # 应用FFT
        x_fft = torch.fft.fft2(x)
        # After FFT shape: torch.Size([2, 128, 20, 20])
        # 取实部
        x_fft_real = x_fft.real
        # 应用线性变换
        x_fft_transformed = self.fft_linear(x_fft_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # After Linear transformation shape: torch.Size([2, 128, 20, 20])
        # 应用PSP模块
        value = self.value_psp(x_fft_transformed)
        # After value_psp shape: torch.Size([2, 128, 50])
        value = value.permute(0, 2, 1)  # 交换维度为 [2, 50, 128]
        value = self.conv1d(value)  # 经过1D卷积后, 形状为 [2, 400, 128]
        value = value.permute(0, 2, 1)  # 交换维度回到 [2, 128, 400]
        # 重塑为原始形状
        context = value.view(m_batchsize, self.inter_channels, h, w)
        # print("Context shape after view:", context.shape)  # 打印重塑后的张量形状
        # 应用局部注意力
        context = self.local_attention(context)
        # print("Context shape after local_attention:", context.shape)  # 打印局部注意力后的张量形状
        # 残差连接
        out = x + context
        # print("Output shape:", out.shape)  # 打印最终输出的张量形状
        return out



class CFC_FFT(nn.Module):
    def __init__(self, in_channels=256, grids=(6, 3, 2, 1)):
        super(CFC_FFT, self).__init__()
        self.grids = grids
        self.inter_channels = in_channels // 2  # 128

        self.reduce_channel = Conv(in_channels, self.inter_channels, 3)

        self.query_conv = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=self.inter_channels, out_channels=self.inter_channels, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=self.inter_channels, out_channels=self.inter_channels, kernel_size=1)

        self.fft_linear = nn.Linear(self.inter_channels, self.inter_channels)

        total_grids = sum(grid * grid for grid in grids)
        # 修改conv1d的输入通道数
        # self.conv1d = nn.Conv1d(in_channels=self.inter_channels, out_channels=self.inter_channels, kernel_size=1,stride=9)
        self.conv1d = nn.Conv1d(in_channels=450, out_channels=50, kernel_size=1)

        self.value_psp = PSPModule(grids, self.inter_channels)
        self.key_psp = PSPModule(grids, self.inter_channels)

        self.local_attention = LocalAttenModule(self.inter_channels, self.inter_channels // 8)

        # self.keras_init_weight()

    def forward(self, x):
        x = self.reduce_channel(x)  # (4, 128, 20, 20)
        m_batchsize, _, h, w = x.size()

        query = self.query_conv(x).view(m_batchsize, self.inter_channels, -1).permute(0, 2, 1)
        # (4, 400, 128)
        key = self.key_psp(x)
        # (4, 128, 50)
        value = self.value_psp(x)
        # (4, 128, 50)

        # 应用FFT
        x_fft = torch.fft.fft2(x)
        x_fft_real = x_fft.real
        x_fft_transformed = self.fft_linear(x_fft_real.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # [2, 128,20,20]

        # 将FFT结果与value结合
        value_combined = torch.cat([value, x_fft_transformed.view(m_batchsize, self.inter_channels, -1)], dim=2)
        # value_combined torch.Size([4, 128, 450])


        # 应用1D卷积
        value_combined = value_combined.permute(0, 2, 1)
        value_combined = self.conv1d(value_combined)
        value_combined = value_combined.permute(0, 2, 1)

        # print("value_combined", value_combined.shape)

        # 计算注意力
        attn = torch.bmm(query, key)  # (4, 400, sum(grids))
        attn = F.softmax(attn, dim=-1)

        # 应用注意力
        context = torch.bmm(attn, value_combined.permute(0, 2, 1))  # (4, 400, 128)
        context = context.permute(0, 2, 1).view(m_batchsize, self.inter_channels, h, w)

        # 应用局部注意力
        context = self.local_attention(context)

        # 残差连接
        out = x + context

        return out


# 假设 Conv, PSPModule, 和 LocalAttenModule 的定义保持不变


# 假设 Conv, PSPModule, 和 LocalAttenModule 的定义保持不变
# Performers 版本
class CFC_Performer(nn.Module):
    def __init__(self, in_channels=256, grids=(6, 3, 2, 1), num_features=256):
        super(CFC_Performer, self).__init__()
        self.grids = grids
        inter_channels = in_channels // 2
        self.inter_channels = inter_channels
        self.num_features = num_features

        self.reduce_channel = Conv(in_channels, inter_channels, 3)

        # 替换原有的query_conv, key_conv, value_conv
        self.q_proj = nn.Linear(inter_channels, inter_channels)
        self.k_proj = nn.Linear(inter_channels, inter_channels)
        self.v_proj = nn.Linear(inter_channels, inter_channels)

        # 随机投影矩阵
        self.random_projections = nn.Parameter(torch.randn(inter_channels, num_features))

        self.value_psp = PSPModule(grids, inter_channels)
        self.key_psp = PSPModule(grids, inter_channels)

        self.local_attention = LocalAttenModule(inter_channels, inter_channels // 8)
        self.keras_init_weight()

    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(ly.weight)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        # 输入 x: (4, 256, 20, 20)
        x = self.reduce_channel(x)  # 降维到 inter_channels
        # x: (4, 128, 20, 20)
        m_batchsize, C, h, w = x.size()
        # 重塑x以适应线性投影
        x_flat = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        # x_flat: (4, 400, 128)
        # 应用线性投影
        q = self.q_proj(x_flat)
        # q: (4, 400, 128)
        k = self.k_proj(self.key_psp(x).view(m_batchsize, C, -1).permute(0, 2, 1))
        # k: (4, S, 128), 其中S是PSP输出的总网格数
        v = self.v_proj(self.value_psp(x).view(m_batchsize, C, -1).permute(0, 2, 1))
        # v: (4, S, 128)
        # 应用随机特征近似
        q_feature = torch.exp(torch.einsum('bnc,cd->bnd', q, self.random_projections) / (C ** 0.25))
        # q_feature: (4, 400, 256)
        k_feature = torch.exp(torch.einsum('bsc,cd->bsd', k, self.random_projections) / (C ** 0.25))
        # k_feature: (4, S, 256)
        # 计算注意力
        kv = torch.einsum('bsd,bsc->bdc', k_feature, v)
        # kv: (4, 128, 256)
        qkv = torch.einsum('bnd,bdc->bnc', q_feature, kv)
        # qkv: (4, 400, 128)
        # 归一化
        normalizer = torch.einsum('bnd,bsd->bns', q_feature, k_feature).sum(dim=-1, keepdim=True)
        # normalizer: (4, 400, 1)
        output = qkv / normalizer
        # output: (4, 400, 128)
        # 重塑回原始形状
        output = output.permute(0, 2, 1).view(m_batchsize, C, h, w)
        # output: (4, 128, 20, 20)
        # 应用局部注意力
        output = self.local_attention(output)
        # output: (4, 128, 20, 20)
        # 残差连接
        out = x + output
        # out: (4, 128, 20, 20)
        return out



# Reformer 版本
class ReformerAttention(nn.Module):
    def __init__(self, dim, num_buckets=32, num_hashes=8):
        super().__init__()
        # dim=32
        self.dim = dim
        # num_buckets=32
        self.num_buckets = num_buckets
        # num_hashes=8
        self.num_hashes = num_hashes
        # (8,32)
        self.lsh_projection = nn.Parameter(torch.randn(num_hashes, dim))

    def hash_vectors(self, vectors):
        # k shape: (batch_size, seq_len_k, dim)
        # Concrete example: (4, 12, 32)
        projections = torch.einsum('hd,bnd->bhn', self.lsh_projection, vectors)
        buckets = torch.argmax(projections, dim=-1)
        return buckets

    def forward(self, q, k, v):
        # q shape: (batch_size, seq_len_q, dim)
        # Concrete example: (4, 400, 32)
        # k shape: (batch_size, seq_len_k, dim)
        # Concrete example: (4, 12, 32)
        # v shape: (batch_size, seq_len_v, dim)
        # Concrete example: (4, 12, 128)
        batch_size, seq_len, _ = q.shape

        # LSH Bucketing
        # q其实没用
        q_buckets = self.hash_vectors(q)
        # q_buckets shape: (batch_size, num_hashes)
        # Concrete example: (4, 8)

        k_buckets = self.hash_vectors(k)
        # k_buckets shape: (batch_size, num_hashes)
        # Concrete example: (4, 8)
        # Sort keys and values based on buckets
        _, k_indices = torch.sort(k_buckets, dim=-1)
        # k_indices shape: (batch_size, num_hashes)
        # Concrete example: (4, 8)
        k_sorted = torch.gather(k, 1, k_indices.unsqueeze(-1).expand(-1, -1, self.dim))
        # k_sorted shape: (batch_size, num_hashes, dim)
        # Concrete example: (4, 8, 32)

        v_sorted = torch.gather(v, 1, k_indices.unsqueeze(-1).expand(-1, -1, v.size(-1)))
        # v_sorted shape: (batch_size, num_hashes, v.size(-1))
        # Concrete example: (4, 8, 128)

        # Compute attention within buckets
        similarities = torch.bmm(q, k_sorted.transpose(1, 2))
        # similarities shape: (batch_size, seq_len_q, num_hashes)
        # Concrete example: (4, 400, 8)

        attention = F.softmax(similarities, dim=-1)
        # attention shape: (batch_size, seq_len_q, num_hashes)
        # Concrete example: (4, 400, 8)

        # Apply attention to values
        output = torch.bmm(attention, v_sorted)
        # output shape: (batch_size, seq_len_q, v.size(-1))
        # Concrete example: (4, 400, 128)
        return output


class CFC_Reformer(nn.Module):
    def __init__(self, in_channels=256, out_channels=128, grids=(6, 3, 2, 1)):
        super(CFC_Reformer, self).__init__()
        self.grids = grids
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.reduce_channel = Conv(in_channels, out_channels, 3)
        self.query_conv = nn.Conv2d(in_channels=out_channels, out_channels=32, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=out_channels, out_channels=32, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)

        self.value_psp = PSPModule(grids, out_channels)
        self.key_psp = PSPModule(grids, out_channels)

        self.local_attention = LocalAttenModule(out_channels, out_channels // 8)
        self.reformer_attention = ReformerAttention(dim=32)

        self.keras_init_weight()

    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        # x shape: (4, 256, 20, 20)
        x = self.reduce_channel(x)
        # x shape now: (4, 128, 20, 20)
        b, c, h, w = x.size()

        query = self.query_conv(x).view(b, 32, -1).permute(0, 2, 1)  
        # query shape: (4, 400, 32)  # 400 = 20 * 20
        # (b, h*w, 32)
        key = self.key_conv(self.key_psp(x))  
        # key shape: (4, 32, 12)  # 12 = sum(grids) = 6 + 3 + 2 + 1
        # (b, 32, sum(grids))

        value = self.value_conv(self.value_psp(x))  
        # value shape: (4, 128, 12)  # 128 = out_channels, 12 = sum(grids)
        # (b, out_channels, sum(grids))

        # Reformer-like attention
        # q shape: (batch_size, seq_len_q, dim)
        # Concrete example: (4, 400, 32)
        # k shape: (batch_size, seq_len_k, dim)
        # Concrete example: (4, 12, 32)
        # v shape: (batch_size, seq_len_v, dim)
        # Concrete example: (4, 12, 128)
        context = self.reformer_attention(query, key.permute(0, 2, 1), value.permute(0, 2, 1))
        # context shape: (4, 400, 128)
        # context shape: (b, h*w, out_channels)

        context = context.permute(0, 2, 1).view(b, self.out_channels, h, w)
        #  (4, 128, 20, 20)
        context = self.local_attention(context)
        #  (4, 128, 20, 20)
        out = x + context  # Residual connection
        # Shape: (4, 128, 20, 20)
        return out 


# Linformer 版本
class LinformerAttention(nn.Module):
    def __init__(self, input_size, channels, dim_k):
        super().__init__()
        self.E = nn.Linear(input_size, dim_k, bias=False)
        self.F = nn.Linear(input_size, dim_k, bias=False)
        self.dim_k = dim_k

    def forward(self, q, k, v):
        # q, k, v shape: (batch_size, seq_len, channels)
        batch_size, seq_len, _ = q.shape

        # Project k and v
        k_projected = self.E(k.transpose(1, 2)).transpose(1, 2)  # (batch_size, dim_k, channels)
        v_projected = self.F(v.transpose(1, 2)).transpose(1, 2)  # (batch_size, dim_k, channels)

        # Compute attention
        attn = torch.bmm(q, k_projected.transpose(1, 2))  # (batch_size, seq_len, dim_k)
        attn = F.softmax(attn / (self.dim_k ** 0.5), dim=-1)

        # Apply attention to v
        out = torch.bmm(attn, v_projected)  # (batch_size, seq_len, channels)

        return out

class CFC_Linformer(nn.Module):
    def __init__(self, in_channels=256, out_channels=128, grids=(6, 3, 2, 1), dim_k=64):
        super(CFC_Linformer, self).__init__()
        self.grids = grids
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim_k = dim_k

        self.reduce_channel = Conv(in_channels, out_channels, 3)
        self.query_conv = nn.Conv2d(in_channels=out_channels, out_channels=32, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=out_channels, out_channels=32, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)

        self.value_psp = PSPModule(grids, out_channels)
        self.key_psp = PSPModule(grids, out_channels)

        self.local_attention = LocalAttenModule(out_channels, out_channels // 8)

        # Calculate the total number of elements after PSP
        self.psp_size = sum(grid * grid for grid in grids)
        self.linformer_attention = LinformerAttention(self.psp_size, 32, dim_k)

        self.keras_init_weight()

    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(ly.weight)
                if hasattr(ly, 'bias') and ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        # x shape: (4, 256, 20, 20)
        x = self.reduce_channel(x)
        # x shape now: (4, 128, 20, 20)
        b, c, h, w = x.size()

        query = self.query_conv(x).view(b, 32, -1).permute(0, 2, 1)  # (b, h*w, 32)
        key = self.key_conv(self.key_psp(x))  # (b, 32, sum(grids))
        value = self.value_conv(self.value_psp(x))  # (b, out_channels, sum(grids))

        # Linformer attention
        context = self.linformer_attention(query, key.permute(0, 2, 1), value.permute(0, 2, 1))
        # context shape: (b, h*w, out_channels)

        context = context.permute(0, 2, 1).view(b, self.out_channels, h, w)
        context = self.local_attention(context)
        out = x + context  # Residual connection

        return out  # Shape: (4, 128, 20, 20)


class AKConv(nn.Module):
    def __init__(self, inc, outc, num_param=5, stride=1, bias=None):
        super(AKConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
                                  nn.BatchNorm2d(outc),
                                  nn.SiLU())  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the AKConv with different sizes.
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int))
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),
                torch.arange(0, mod_number))

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        # using Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        # using 1 × 1 Conv
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)

        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset

class SFC_AKConv(nn.Module):
    def __init__(self,inc):
        super(SFC_AKConv, self).__init__()

        hidc = inc[0]
        self.groups = 2

        # 使用 AKConv 替代原来的 Conv
        self.conv_8 = nn.Sequential(
            AKConv(256, hidc, num_param=5, stride=1),
            nn.Conv2d(hidc, hidc, kernel_size=1)  # 1x1 卷积来调整通道数
        )
        # 直接将 sp 转换为所需的 shape
        self.conv_32 = nn.Sequential(
            AKConv(128, hidc, num_param=5, stride=1),
            nn.Conv2d(hidc, hidc, kernel_size=1), # 1x1 卷积来调整通道数
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),  # 上采样到正确的空间尺寸
        )

        self.conv_offset = nn.Sequential(
            AKConv(hidc * 2, 64, num_param=5, stride=1),
            nn.Conv2d(64, self.groups * 4 + 2, kernel_size=3, padding=1, bias=False)
        )

        self.keras_init_weight()
        self.conv_offset[-1].weight.data.zero_()

    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        cp, sp = x
        # cp shape: torch.Size([2, 256, 80, 80])

        # sp需要上采样 来自上一模块
        # sp shape: torch.Size([2, 128, 20, 20])
        n, _, out_h, out_w = cp.size()
        # x_32
        sp = self.conv_32(sp)  # 转换为目标 shape
        # sp shape: torch.Size([2, 256, 80, 80])
        # 通过 AKConv, 1x1 Conv, 和上采样，
        # sp 的 shape 从 [2, 128, 20, 20] 变为 [2, 256, 80, 80]

        # x_8
        cp = self.conv_8(cp)
        # cp shape: torch.Size([2, 256, 80, 80])

        # 确保 cp 和 sp 的形状完全相同
        assert cp.shape == sp.shape, f"Shape mismatch: cp {cp.shape}, sp {sp.shape}"

        conv_results = self.conv_offset(torch.cat([cp, sp], 1))
        # conv_results shape: torch.Size([2, 10, 80, 80])

        sp = sp.reshape(n * self.groups, -1, out_h, out_w)
        cp = cp.reshape(n * self.groups, -1, out_h, out_w)
        # sp 和 cp shape: torch.Size([4, 128, 80, 80])
        # 2 * 2 = 4, 256 / 2 = 128

        offset_l = conv_results[:, 0:self.groups * 2, :, :].reshape(n * self.groups, -1, out_h, out_w)
        offset_h = conv_results[:, self.groups * 2:self.groups * 4, :, :].reshape(n * self.groups, -1, out_h, out_w)
        # offset_l 和 offset_h shape: torch.Size([4, 2, 80, 80])

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(sp).to(sp.device)

        # 空间变换网络（Spatial Transformation Network, STN）
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        # 将网格重复以匹配批量大小和分组数。
        grid = grid.repeat(n * self.groups, 1, 1, 1).type_as(sp).to(sp.device)

        # 应用偏移
        grid_l = grid + offset_l.permute(0, 2, 3, 1) / norm
        grid_h = grid + offset_h.permute(0, 2, 3, 1) / norm
        # grid_l 和 grid_h shape: torch.Size([4, 80, 80, 2])

        # 采样
        cp = F.grid_sample(cp, grid_l, align_corners=True)
        sp = F.grid_sample(sp, grid_h, align_corners=True)
        # cp 和 sp shape 保持不变: torch.Size([4, 128, 80, 80])

        cp = cp.reshape(n, -1, out_h, out_w)
        sp = sp.reshape(n, -1, out_h, out_w)
        # cp 和 sp shape 变回: torch.Size([2, 256, 80, 80])

        att = 1 + torch.tanh(conv_results[:, self.groups * 4:, :, :])
        # att shape: torch.Size([2, 2, 80, 80])

        sp = sp * att[:, 0:1, :, :] + cp * att[:, 1:2, :, :]
        # sp shape: torch.Size([2, 256, 80, 80])

        return sp
        # Final output shape: torch.Size([2, 256, 80, 80])

if __name__ == '__main__':
    x = torch.randn(4, 256, 20, 20)
    # mod = CFC_FFT()
    # mod = CFC_Performer()
    # mod = CFC_Reformer()
    mod = CFC_Linformer()
    out = mod(x)
    print(out.shape)


    cp = torch.randn(2, 256, 80, 80)
    sp = torch.randn(2, 128, 20, 20)

    model = SFC_AKConv(inc=[256, 128])

    out = model([cp,sp])
    print(out.shape)
