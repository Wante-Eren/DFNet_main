import torch

from torch import nn
from torch.nn import functional as F


class IN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Multiin(nn.Module):  # stereo attention block
    def __init__(self, out=1):
        super().__init__()
        self.out = out

    def forward(self, x):
        x1, x2 = x[:, :3, :, :], x[:, 3:, :, :]
        if self.out == 1:
            x = x1
        else:
            x = x2
        return x


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim  # 拼接的维度（默认为通道维度）

    def forward(self, x):
        # 如果输入是列表，则在指定维度拼接
        if isinstance(x, list):
            return torch.cat(x, dim=self.dim)
        # 如果输入是单个张量，则直接返回
        return x

class EFBlock(nn.Module):
    def __init__(self, c1, c2, reduction=16):
        super(EFBlock, self).__init__()
        self.mask_map_r = nn.Conv2d(c1 // 2, 1, 1, 1, 0, bias=True)
        self.mask_map_i = nn.Conv2d(c1 // 2, 1, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        self.bottleneck1 = nn.Conv2d(c1 // 2, c2 // 2, 3, 1, 1, bias=False)
        self.bottleneck2 = nn.Conv2d(c1 // 2, c2 // 2, 3, 1, 1, bias=False)
        self.se = SE_Block(c2, reduction)

    def forward(self, x):
        # print("EFBlock input:", x.shape)
        x_left_ori, x_right_ori = x[:, :3, :, :], x[:, 3:, :, :]
        x_left = x_left_ori * 0.5
        x_right = x_right_ori * 0.5

        x_mask_left = torch.mul(self.mask_map_r(x_left), x_left)
        x_mask_right = torch.mul(self.mask_map_i(x_right), x_right)

        out_IR = self.bottleneck1(x_mask_right + x_right_ori)
        out_RGB = self.bottleneck2(x_mask_left + x_left_ori)  # RGB
        out = self.se(torch.cat([out_RGB, out_IR], 1))

        return out


##################################CPCA Attention#########################################

class FeatureAdd(nn.Module):
    #  x + CPCA
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.add(x[0], x[1])


class CPCA_ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(CPCA_ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class CPCA(nn.Module):
    def __init__(self, channels, out_channels, channelAttention_reduce=4):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(1, 1), padding=0)
        self.ca = CPCA_ChannelAttention(input_channels=channels, internal_neurons=channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.dconv1_7 = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.dconv7_1 = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.dconv1_11 = nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)
        self.dconv11_1 = nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels)
        self.dconv1_21 = nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels)
        self.dconv21_1 = nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels)
        self.conv2 = nn.Conv2d(channels, out_channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, x):
        inputs = torch.cat((x[0], x[1]), dim=1)
        #   Global Perceptron
        inputs = self.conv1(inputs)
        inputs = self.act(inputs)

        inputs = self.ca(inputs)

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv1(x)
        out = spatial_att * inputs
        out = self.conv2(out)
        return out


##################################Transformer############################################
# 多头交叉注意力机制
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        assert (self.head_dim * num_heads == model_dim), "model_dim must be divisible by num_heads"

        self.query_vis = nn.Linear(model_dim, model_dim)
        self.key_vis = nn.Linear(model_dim, model_dim)
        self.value_vis = nn.Linear(model_dim, model_dim)

        self.query_inf = nn.Linear(model_dim, model_dim)
        self.key_inf = nn.Linear(model_dim, model_dim)
        self.value_inf = nn.Linear(model_dim, model_dim)

        self.fc_out_vis = nn.Linear(model_dim, model_dim)
        self.fc_out_inf = nn.Linear(model_dim, model_dim)

    def forward(self, vis, inf):
        batch_size, seq_length, model_dim = vis.shape

        # vis -> Q, K, V
        Q_vis = self.query_vis(vis)
        K_vis = self.key_vis(vis)
        V_vis = self.value_vis(vis)

        # inf -> Q, K, V
        Q_inf = self.query_inf(inf)
        K_inf = self.key_inf(inf)
        V_inf = self.value_inf(inf)

        # Reshape for multi-head attention
        Q_vis = Q_vis.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,
                                                                                            2)  # B, N, C --> B, n_h, N, d_h
        K_vis = K_vis.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V_vis = V_vis.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        Q_inf = Q_inf.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K_inf = K_inf.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V_inf = V_inf.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Cross attention: vis Q with inf K and inf Q with vis K
        # Q_vis 的形状为 (batch_size, num_heads, seq_length, head_dim)
        # K_inf 的形状为 (batch_size, num_heads, head_dim, seq_length)
        # 矩阵乘法后，scores_vis_inf 的形状为 (batch_size, num_heads, seq_length, seq_length)
        scores_vis_inf = torch.matmul(Q_vis, K_inf.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))
        scores_inf_vis = torch.matmul(Q_inf, K_vis.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))

        attention_inf = torch.softmax(scores_vis_inf, dim=-1)
        attention_vis = torch.softmax(scores_inf_vis, dim=-1)

        # attention_vis_inf 的形状为 (batch_size, num_heads, seq_length, seq_length)
        # V_inf 的形状为 (batch_size, num_heads, seq_length, head_dim)
        # out_vis_inf 的形状为 (batch_size, num_heads, seq_length, head_dim)
        out_inf = torch.matmul(attention_inf, V_inf)
        out_vis = torch.matmul(attention_vis, V_vis)

        # Concatenate and project back to the original dimension
        out_vis = out_vis.transpose(1, 2).contiguous().view(batch_size, seq_length, model_dim)
        out_inf = out_inf.transpose(1, 2).contiguous().view(batch_size, seq_length, model_dim)

        # out 的形状为 (batch_size, seq_length, model_dim)
        out_vis = self.fc_out_vis(out_vis)
        out_inf = self.fc_out_inf(out_inf)

        return out_vis, out_inf


# 前向全连接网络
class FeedForward(nn.Module):
    def __init__(self, model_dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(model_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout, max_len=6400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个从0到max_len-1的列向量，形状为 (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1)
        # 计算用于位置编码的除数项
        div_term = torch.exp(torch.arange(0, model_dim, 2) * -(torch.log(torch.tensor(10000.0)) / model_dim))

        pe = torch.zeros(max_len, model_dim)  # 初始化一个位置编码矩阵，形状为 (max_len, model_dim)，所有元素初始化为0
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列使用sin函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列使用cos函数

        pe = pe.unsqueeze(0)  # 在位置编码矩阵的第一个维度前添加一个新的维度，变为 (1, max_len, model_dim)
        self.register_buffer('pe', pe)  # 将位置编码矩阵 pe 注册为模型的一个缓冲区。缓冲区类似于模型参数，但在训练过程中不会更新

    def forward(self, x):
        # x 的形状为 (batch_size, seq_length, model_dim)
        # 从位置编码矩阵中选择前 seq_len 个位置的编码，形状为 (1, seq_len, model_dim)
        # 并将其与输入张量 x 相加
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# 编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.cross_attention = MultiHeadCrossAttention(model_dim, num_heads)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ff = FeedForward(model_dim, hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, vis, inf):
        attn_out_vis, attn_out_inf = self.cross_attention(vis, inf)
        vis = self.norm1(vis + attn_out_vis)
        inf = self.norm1(inf + attn_out_inf)

        ff_out_vis = self.ff(vis)
        ff_out_inf = self.ff(inf)

        vis = self.norm2(vis + ff_out_vis)
        inf = self.norm2(inf + ff_out_inf)

        return vis, inf


# Transformer编码器
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, hidden_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(model_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, vis, inf):
        vis = self.embedding(vis) * torch.sqrt(torch.tensor(self.embedding.out_features, dtype=torch.float32))
        inf = self.embedding(inf) * torch.sqrt(torch.tensor(self.embedding.out_features, dtype=torch.float32))

        vis = self.positional_encoding(vis)
        inf = self.positional_encoding(inf)

        for layer in self.layers:
            vis, inf = layer(vis, inf)

        return vis, inf


# 定义用于交叉注意力的网络
class CrossTransformerFusion(nn.Module):
    def __init__(self, input_dim, num_heads=2, num_layers=1, dropout=0.1):
        super(CrossTransformerFusion, self).__init__()
        self.hidden_dim = input_dim * 2
        self.model_dim = input_dim
        self.encoder = TransformerEncoder(input_dim, self.model_dim, num_heads, num_layers, self.hidden_dim, dropout)

    def forward(self, x):
        vis, inf = x[0], x[1]
        # 输入形状为 B, C, H, W
        B, C, H, W = vis.shape

        # 将输入变形为 B, H*W, C
        vis = vis.permute(0, 2, 3, 1).reshape(B, -1, C)
        inf = inf.permute(0, 2, 3, 1).reshape(B, -1, C)

        # 输入Transformer编码器
        vis_out, inf_out = self.encoder(vis, inf)

        # 将输出变形为 B, C, H, W
        vis_out = vis_out.view(B, H, W, -1).permute(0, 3, 1, 2)
        inf_out = inf_out.view(B, H, W, -1).permute(0, 3, 1, 2)

        # 在通道维度上进行级联
        out = torch.cat((vis_out, inf_out), dim=1)

        return out



######################################################
#try 1.0
######################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class EnhancedCrossFusion(nn.Module):
    def __init__(self, dim, num_heads=8, expansion=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # 维度压缩
        self.ch_reduce = nn.Sequential(
            nn.Conv2d(dim, dim//expansion, 1),
            nn.BatchNorm2d(dim//expansion),
            nn.SiLU()
        )

        # 跨模态QKV生成
        self.q_vis = nn.Conv2d(dim//expansion, dim, 1)
        self.kv_inf = nn.Conv2d(dim//expansion, dim*2, 1)
        
        # 空间增强（修正关键错误点）
        self.dw_conv = nn.Conv2d(
            in_channels=dim,          # 输入通道数必须等于输出通道数
            out_channels=dim, 
            kernel_size=3,
            padding=1,
            groups=dim                # 深度可分离卷积分组数=通道数
        )

        # 动态门控（通道对齐）
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim*2, dim//4, 1),  # 输入通道数=融合特征通道数x2
            nn.ReLU(),
            nn.Conv2d(dim//4, 2, 1),
            nn.Softmax(dim=1)
        )

        # 本地特征保留
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )

        # 维度恢复
        self.ch_expand = nn.Conv2d(dim//expansion, dim, 1)

    def forward(self, x):
        vis, inf = x
        B, C, H, W = vis.shape

        # 本地特征保留
        local_feat = self.local_conv(vis)

        # 维度压缩
        vis_reduced = self.ch_reduce(vis)
        inf_reduced = self.ch_reduce(inf)

        # 生成QKV
        q_vis = rearrange(self.q_vis(vis_reduced), 'b (h d) x y -> b h (x y) d', h=self.num_heads)
        k_inf, v_inf = self.kv_inf(inf_reduced).chunk(2, dim=1)
        k_inf = rearrange(k_inf, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
        v_inf = rearrange(v_inf, 'b (h d) x y -> b h (x y) d', h=self.num_heads)

        # 注意力计算
        attn_a=np.dot(q_vis,k_inf.transpose(-2, -1))
        attn = attn_a * self.scale
        #attn = (q_vis @ k_inf.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn_b = np.dot(attn,v_inf)
        attn_out = attn_b.transpose(1, 2)
        #attn_out = (attn @ v_inf).transpose(1, 2)
        attn_out = rearrange(attn_out, 'b (x y) d h -> b (h d) x y', x=H, y=W)

        # 空间增强（输入输出通道数必须相同）
        attn_out = self.dw_conv(attn_out)  # [B, C, H, W]

        # 动态门控融合
        fused = torch.cat([attn_out, local_feat], dim=1)
        gate_weights = self.gate(fused)
        output = gate_weights[:,0:1] * attn_out + gate_weights[:,1:2] * local_feat

        return output + vis  # 残差连接

##############################version 2##########################
class GatedCrossCAFM(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.in_channels = in_channels

        # 跨模态QKV生成
        self.self_q = nn.Conv2d(in_channels, in_channels, 1)
        self.other_kv = nn.Conv2d(in_channels, in_channels * 2, 1)

        # 门控机制
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, 1, 1),
            nn.Sigmoid()
        )

        # 本地特征保留
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

        self.out_conv = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        # 分支1：本地特征
        x_self = x[0]
        x_other = x[1]
        
        
        local_feat = self.local_conv(x_self)

        # 分支2：跨模态注意力
        B, C, H, W = x_self.shape

        # 生成Q/K/V
        q = self.self_q(x_self)  # 自身模态生成Q [B,C,H,W]
        kv = self.other_kv(x_other)  # 其他模态生成K/V [B,2C,H,W]
        k, v = torch.split(kv, [C, C], dim=1)

        # 注意力计算
        q = q.view(B, C, -1).permute(0, 2, 1)  # [B,HW,C]
        k = k.view(B, C, -1)  # [B,C,HW]
        v = v.view(B, C, -1).permute(0, 2, 1)  # [B,HW,C]

        attn = torch.bmm(q, k) / (C ** 0.5)  # [B,HW,HW]
        attn = F.softmax(attn, dim=-1)
        attn_out = torch.bmm(attn, v)  # [B,HW,C]
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)

        # 门控融合
        gate_weight = self.gate(x_self)  # [B,1,1,1]
        
        fused = local_feat + gate_weight * attn_out

        return self.out_conv(fused)  # 输出预处理后的特征


###############################################################################################


##################################################################################################
class CrossModalRefine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.vis_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        self.ir_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        self.fusion = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x):
        vis_feat = x[0]
        ir_feat = x[1]
        vis_out = self.vis_conv(vis_feat + ir_feat)
        ir_out = self.ir_conv(ir_feat + vis_feat)
        return self.fusion(torch.cat([vis_out, ir_out], dim=1))

#################################################################################
# ====================== 新增门控组件 ======================
class GatedMultiHeadCrossAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        # 复用原有的多头注意力机制
        self.cross_attn = MultiHeadCrossAttention(model_dim, num_heads)
        
        # 新增门控参数
        self.gate_vis = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.gate_inf = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.normal_(self.gate_vis, mean=0, std=0.02)
        nn.init.normal_(self.gate_inf, mean=0, std=0.02)

    def forward(self, vis, inf):
        # 复用原有注意力计算
        attn_vis, attn_inf = self.cross_attn(vis, inf)
        
        # 新增门控融合
        gate_vis = torch.sigmoid(self.gate_vis)
        gate_inf = torch.sigmoid(self.gate_inf)
        
        vis_out = gate_vis * attn_vis + (1 - gate_vis) * vis
        inf_out = gate_inf * attn_inf + (1 - gate_inf) * inf
        
        return vis_out, inf_out

class ResidualGatedFFN(nn.Module):
    def __init__(self, model_dim, hidden_dim, dropout=0.1):
        super().__init__()
        # 复用原有前馈网络
        self.ffn = FeedForward(model_dim, hidden_dim, dropout)
        
        # 新增门控参数
        self.gate = nn.Linear(model_dim, 1)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        
        # 门控系数计算
        gate = torch.sigmoid(self.gate(x))
        return residual + gate * self.ffn(x)

# ====================== 新增融合模块 ======================
class CrossModalGateFusion(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        # 新增门控卷积
        self.gate_conv = nn.Sequential(
            nn.Conv2d(2*model_dim, model_dim//2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(model_dim//2, 2, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, vis, inf):
        concat = torch.cat([vis, inf], dim=1)
        gates = self.gate_conv(concat)
        vis_gate, inf_gate = gates.chunk(2, dim=1)
        return vis_gate * vis + inf_gate * inf

# ====================== 增强版融合模块 ======================
class EnhancedCrossTransformerFusionV2(nn.Module):
    def __init__(self, input_dim, num_heads=2, num_layers=1, dropout=0.1):
        super().__init__()
        # 复用原有编码器结构
        self.encoder = TransformerEncoder(
            input_dim, 
            input_dim,  # 修改model_dim与input_dim一致
            num_heads,
            num_layers,
            input_dim*2,  # hidden_dim
            dropout
        )
        
        # 新增门控组件
        self.cross_gate = CrossModalGateFusion(input_dim)
        
        # 新增残差卷积
        self.res_conv_vis = nn.Conv2d(input_dim, input_dim, 3, padding=1)
        self.res_conv_inf = nn.Conv2d(input_dim, input_dim, 3, padding=1)

    def forward(self, x):
        vis, inf = x[0], x[1]
        B, C, H, W = vis.shape
        
        # 复用原有编码流程
        vis_seq = vis.permute(0,2,3,1).reshape(B, -1, C)
        inf_seq = inf.permute(0,2,3,1).reshape(B, -1, C)
        vis_enc, inf_enc = self.encoder(vis_seq, inf_seq)
        
        # 特征重建
        vis_out = vis_enc.view(B, H, W, C).permute(0,3,1,2)
        inf_out = inf_enc.view(B, H, W, C).permute(0,3,1,2)
        
        # 新增门控融合
        fused = self.cross_gate(vis_out, inf_out)
        
        # 残差连接后拼接
        return torch.cat([
            self.res_conv_vis(vis) + fused, 
            self.res_conv_inf(inf) + fused
        ], dim=1)




#############################################################################################################################################
# ===== 新增模块（名称全部包含Gated/V2标识）=====
class GatedMultiHeadCrossAtt(nn.Module):  # 增加"Gated"和缩短名称
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttention(model_dim, num_heads)
        self.gate_vis = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.gate_inf = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.normal_(self.gate_vis, std=0.02)
        nn.init.normal_(self.gate_inf, std=0.02)
    
    def forward(self, vis, inf):
        attn_vis, attn_inf = self.cross_attn(vis, inf)
        gate_vis = torch.sigmoid(self.gate_vis)
        gate_inf = torch.sigmoid(self.gate_inf)
        return (gate_vis*attn_vis + (1-gate_vis)*vis, 
                gate_inf*attn_inf + (1-gate_inf)*inf)

class GatedResFFN(nn.Module):  # 名称新增Gated
    def __init__(self, model_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.ffn = FeedForward(model_dim, hidden_dim, dropout)
        self.gate = nn.Linear(model_dim, 1)
        self.norm = nn.LayerNorm(model_dim)
    
    def forward(self, x):
        residual = x
        x = self.norm(x)
        return residual + torch.sigmoid(self.gate(x)) * self.ffn(x)

class GateFusionV2(nn.Module):  # 新增V2标识
    def __init__(self, channels):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Conv2d(2*channels, channels//2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels//2, 2, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, vis, inf):
        gates = self.gate_net(torch.cat([vis, inf], dim=1))
        return gates[:,0:1]*vis + gates[:,1:2]*inf

# ===== 最终融合模块 =====
class CrossTransformerGatedV2(nn.Module):  # 全新命名
    def __init__(self, in_channels, num_heads=2, num_layers=2):
        super().__init__()
        self.encoder = nn.ModuleList([
            nn.ModuleDict({
                'attn': GatedMultiHeadCrossAtt(in_channels, num_heads),
                'ffn': GatedResFFN(in_channels, in_channels*2)
            }) for _ in range(num_layers)
        ])
        self.fusion_gate = GateFusionV2(in_channels)
        self.res_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        
    def forward(self, x):
        vis, inf = x
        B, C, H, W = vis.shape
        
        # 编码处理
        vis_seq = vis.permute(0,2,3,1).reshape(B, -1, C)
        inf_seq = inf.permute(0,2,3,1).reshape(B, -1, C)
        
        for layer in self.encoder:
            vis_seq, inf_seq = layer['attn'](vis_seq, inf_seq)
            vis_seq = layer['ffn'](vis_seq)
            inf_seq = layer['ffn'](inf_seq)
        
        # 重建特征
        vis_feat = vis_seq.view(B, H, W, C).permute(0,3,1,2)
        inf_feat = inf_seq.view(B, H, W, C).permute(0,3,1,2)
        
        # 融合输出
        return self.res_conv(vis) + self.fusion_gate(vis_feat, inf_feat)



#######################################################################################

