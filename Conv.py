######################################## Strip R-CNN   by AI Little monster  start ########################################
 
 
###   my blog :https://blog.csdn.net/m0_63774211?type=lately
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv, autopad,DWConv
from ultralytics.nn.modules.block import Bottleneck, C2f, C3k,C3k2
 
class StripMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
 
class Strip_Block(nn.Module):
    def __init__(self, dim, k1, k2):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial1 = nn.Conv2d(dim,dim,kernel_size=(k1, k2), stride=1, padding=(k1//2, k2//2), groups=dim)
        self.conv_spatial2 = nn.Conv2d(dim,dim,kernel_size=(k2, k1), stride=1, padding=(k2//2, k1//2), groups=dim)
 
        self.conv1 = nn.Conv2d(dim, dim, 1)
 
    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial1(attn)
        attn = self.conv_spatial2(attn)
        attn = self.conv1(attn)
 
        return x * attn
 
class Strip_Attention(nn.Module):
    def __init__(self, d_model,k1,k2):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = Strip_Block(d_model,k1,k2)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)
 
    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        # x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
 
class StripBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., k1=1, k2=19, drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Strip_Attention(dim, k1, k2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = StripMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
 
    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x
 
class C3k_Strip(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(StripBlock(c_) for _ in range(n)))
 
class C3k2_Strip(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_Strip(self.c, self.c, 2, shortcut, g) if c3k else StripBlock(self.c) for _ in range(n))
 
######################################## Strip R-CNN    by AI Little monster  end ########################################
######################################## APConv AAAI2025   by AI Little monster start  ########################################
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from ultralytics.nn.modules.conv import Conv,autopad
from ultralytics.nn.modules.block import Bottleneck, C2f,C3k,C3k2
 
from mmcv.cnn import ConvModule
from mmengine.model import caffe2_xavier_init, constant_init
 
from ultralytics.nn.modules.conv import Conv
 
 
class ContextAggregation(nn.Module):
    """
    Context Aggregation Block.
    Args:
        in_channels (int): Number of input channels.
        reduction (int, optional): Channel reduction ratio. Default: 1.
        conv_cfg (dict or None, optional): Config dict for the convolution
            layer. Default: None.
    """
 
    def __init__(self, in_channels, reduction=1):
        super(ContextAggregation, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = max(in_channels // reduction, 1)
 
        conv_params = dict(kernel_size=1, act_cfg=None)
 
        self.a = ConvModule(in_channels, 1, **conv_params)
        self.k = ConvModule(in_channels, 1, **conv_params)
        self.v = ConvModule(in_channels, self.inter_channels, **conv_params)
        self.m = ConvModule(self.inter_channels, in_channels, **conv_params)
 
        self.init_weights()
 
    def init_weights(self):
        for m in (self.a, self.k, self.v):
            caffe2_xavier_init(m.conv)
        constant_init(self.m.conv, 0)
 
    def forward(self, x):
        # n, c = x.size(0)
        n = x.size(0)
        c = self.inter_channels
        # n, nH, nW, c = x.shape
 
        # a: [N, 1, H, W]
        a = self.a(x).sigmoid()
 
        # k: [N, 1, HW, 1]
        k = self.k(x).view(n, 1, -1, 1).softmax(2)
 
        # v: [N, 1, C, HW]
        v = self.v(x).view(n, 1, c, -1)
 
        # y: [N, C, 1, 1]
        y = torch.matmul(v, k).view(n, c, 1, 1)
        y = self.m(y) * a
 
        return x + y
 
 
 
class PConv_att(nn.Module):  
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''
    
    def __init__(self, c1, c2, k, s):
        super().__init__()
 
        # self.k = k
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)
        self.att = ContextAggregation(c1)
 
    def forward(self, x):
        x =   self.att(x)
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))
 
######################################## APConv AAAI2025   by AI Little monster END  ########################################