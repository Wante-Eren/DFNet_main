# @Time    : 05/12/2022 10:57
# @Author  : BubblyYi
# @FileName: cxm_fusion.py
# @Software: PyCharm
import math

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class ChannelWeight(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeight, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim * 2 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 2 // reduction, self.dim),
            nn.Sigmoid())

    def forward(self, x):
        B, _, H, W = x.shape
        avg = self.avg_pool(x).view(B, self.dim)
        max = self.max_pool(x).view(B, self.dim)
        y = torch.cat((avg, max), dim=1)  # B 4C
        y = self.mlp(y).view(B, self.dim, 1)
        channel_weight = y.reshape(B, self.dim, 1, 1)
        return channel_weight


class SpatialWeight(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeight, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 1, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        B, _, H, W = x.shape
        spatial_weight = self.mlp(x).reshape(B, 1, H, W)
        return spatial_weight


class FeatureNormalModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FeatureNormalModule, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weight = ChannelWeight(dim=dim, reduction=reduction)
        self.spatial_weight = SpatialWeight(dim=dim, reduction=reduction)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        channel_weights = self.channel_weight(x)
        spatial_weights = self.spatial_weight(x)
        x = x + self.lambda_c * channel_weights * x + self.lambda_s * spatial_weights * x
        return x


# Feature Rectify Module
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 4 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        channel_weights = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(channel_weights).view(B, self.dim * 2)
        max = self.max_pool(channel_weights).view(B, self.dim * 2)
        channel_weights = torch.cat((avg, max), dim=1)  # B 4C
        channel_weights = self.mlp(channel_weights).view(B, self.dim * 2, 1)
        channel_weights = channel_weights.reshape(
            B, 2, self.dim, 1, 1).permute(
            1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        spatial_weights = torch.cat((x1, x2), dim=1)  # B 2C H W
        spatial_weights = self.mlp(spatial_weights).reshape(
            B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2 B 1 H W
        return spatial_weights


class FeatureRectifyModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FeatureRectifyModule, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)

        out_x1 = x1 + self.lambda_c * \
            channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * \
            channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        return out_x1, out_x2


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C //
                        self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C //
                        self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(B, -
                                      1, 2, self.num_heads, C //
                                      self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -
                                      1, 2, self.num_heads, C //
                                      self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        k1 = (k1.transpose(-2, -1) @ v1) * self.scale
        k1 = k1.softmax(dim=-2)
        k2 = (k2.transpose(-2, -1) @ v2) * self.scale
        k2 = k2.softmax(dim=-2)

        x1 = (q1 @ k2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x2 = (q2 @ k1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()

        return x1, x2


class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        u1, u2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, u1), dim=-1)
        y2 = torch.cat((y2, u2), dim=-1)
        x1 = self.norm1(x1 + self.end_proj1(y1))
        x2 = self.norm2(x2 + self.end_proj2(y2))
        return x1, x2

# Stage 2


class ChannelEmbed(nn.Module):
    def __init__(
            self, in_channels, out_channels, reduction=1,
            norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // reduction,
                kernel_size=1,
                bias=True),
            nn.Conv2d(
                out_channels // reduction,
                out_channels // reduction,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=out_channels // reduction),
            nn.ReLU(
                inplace=True),
            nn.Conv2d(
                out_channels // reduction,
                out_channels,
                kernel_size=1,
                bias=True),
            norm_layer(out_channels))
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        x = self.norm(residual + x)
        return x


class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(
            dim=dim,
            reduction=reduction,
            num_heads=num_heads)
        self.channel_emb = ChannelEmbed(
            in_channels=dim * 2,
            out_channels=dim,
            reduction=reduction,
            norm_layer=norm_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
    # 检查输入是否包含两个张量，否则抛出异常
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError(f"FeatureFusionModule expects a tuple or list of two tensors, but got {type(x)} with length {len(x) if isinstance(x, (tuple, list)) else 'N/A'}.")

    # 正确解包两个输入张量
        x1, x2 = x

        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2)
        merge = torch.cat((x1, x2), dim=-1)
        merge = self.channel_emb(merge, H, W)

        return merge