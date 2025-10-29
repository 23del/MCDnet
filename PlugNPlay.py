import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

#   Efficient up-convolution block (EUCB)
class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x
# if __name__ == '__main__':
#     input = torch.randn(1, 32, 64, 64)  #B C H W
#
#     block = EUCB(in_channels=32, out_channels=32)
#
#     print(input.size())
#
#     output = block(input)
#     print(output.size())

# 论文题目：PlainUSR: Chasing Faster ConvNet for Efficient Super-Resolution
# 论文地址：https://openaccess.thecvf.com/content/ACCV2024/papers/Wang_PlainUSR_Chasing_Faster_ConvNet_for_Efficient_Super-Resolution_ACCV_2024_paper.pdf
class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool

class LocalAttention(nn.Module):
    ''' attention based on local importance'''

    def __init__(self, channels, f=16):
        super().__init__()
        self.body = nn.Sequential(
            # sample importance
            nn.Conv2d(channels, f, 1),
            SoftPooling2D(7, stride=3),
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(f, channels, 3, padding=1),
            # to heatmap
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Sigmoid(),
        )

    def forward(self, x):
        ''' forward '''
        # interpolate the heat map
        g = self.gate(x[:, :1].clone())
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        return x * w * g  # (w + g) #self.gate(x, w)

# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     block = LocalAttention(channels=32).to(device)
#     input = torch.rand(1, 32, 256, 256).to(device)
#
#     output = block(input)
#     print(input.shape)
#     print(output.shape)


# 论文地址：https://arxiv.org/pdf/2108.01072
# 论文：S2-MLPv2: Improved Spatial-Shift MLP Architecture for Vision
def spatial_shift1(x):
    b, w, h, c = x.size()
    x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
    x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
    x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
    x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
    return x

def spatial_shift2(x):
    b, w, h, c = x.size()
    x[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
    x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
    x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
    x[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
    return x

class SplitAttention(nn.Module):
    def __init__(self, channel=3, k=3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)  # bs,k,n,c
        a = torch.sum(torch.sum(x_all, 1), 1)  # bs,c
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))  # bs,kc
        hat_a = hat_a.reshape(b, self.k, c)  # bs,k,c
        bar_a = self.softmax(hat_a)  # bs,k,c
        attention = bar_a.unsqueeze(-2)  # #bs,k,1,c
        out = attention * x_all  # #bs,k,n,c
        out = torch.sum(out, 1).reshape(b, h, w, c)
        return out

class S2Attention(nn.Module):

    def __init__(self, channels=512):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attention = SplitAttention()

    def forward(self, x):
        b, c, w, h = x.size()
        x = x.permute(0, 2, 3, 1)
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:, :, :, :c])
        x2 = spatial_shift2(x[:, :, :, c:c * 2])
        x3 = x[:, :, :, c * 2:]
        x_all = torch.stack([x1, x2, x3], 1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        x = x.permute(0, 3, 1, 2)
        return x

# if __name__ == '__main__':
#     input = torch.randn(50, 512, 7, 7)
#     block = S2Attention(channels=512)
#     output = block(input)
#     print(output.shape)


# 论文；MCANet: A joint semantic segmentation framework of optical and SAR images for land use classification
# Github地址：https://github.com/Ray010221/MCANet
# 全网最全100➕即插即用模块GitHub地址：https://github.com/ai-dawang/PlugNPlay-Modules
#图像特征融合mcam  需要空间太大
class MCAM(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 dimension=2,
                 sub_sample=True,    # 子采样
                 bn_layer=True):     # 批量归一化
        super(MCAM, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g_sar = conv_nd(in_channels=self.in_channels,out_channels=self.inter_channels,kernel_size=1,stride=1,padding=0)

        self.g_opt = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta_sar = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.theta_opt = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi_sar = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.phi_opt = conv_nd(in_channels=self.in_channels,
                               out_channels=self.inter_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)

        if sub_sample:
            self.g_sar = nn.Sequential(self.g_sar, max_pool_layer)
            self.g_opt = nn.Sequential(self.g_opt, max_pool_layer)
            self.phi_sar = nn.Sequential(self.phi_sar, max_pool_layer)
            self.phi_opt = nn.Sequential(self.phi_opt, max_pool_layer)

    def forward(self, sar, opt):

        batch_size = sar.size(0)

        g_x = self.g_sar(sar).view(batch_size, self.inter_channels, -1)  # [bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta_sar(sar).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi_sar(sar).view(batch_size, self.inter_channels, -1)

        f_x = torch.matmul(theta_x, phi_x)
        f_div_C_x = F.softmax(f_x, dim=-1)

        g_y = self.g_opt(opt).view(batch_size, self.inter_channels, -1)  # [bs, c, w*h]
        g_y = g_y.permute(0, 2, 1)

        theta_y = self.theta_opt(opt).view(batch_size, self.inter_channels, -1)
        theta_y = theta_y.permute(0, 2, 1)

        phi_y = self.phi_opt(opt).view(batch_size, self.inter_channels, -1)

        f_y = torch.matmul(theta_y, phi_y)
        f_div_C_y = F.softmax(f_y, dim=-1)
        y = torch.einsum('ijk,ijk->ijk', [f_div_C_x, f_div_C_y])
        y_x = torch.matmul(y, g_x)
        y_y = torch.matmul(y, g_y)
        y = y_x * y_y
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *sar.size()[2:])
        y = self.W(y)
        return y

# if __name__ == '__main__':
#     block = MCAM(in_channels=96)
#     sar = torch.randn(2, 96, 32, 32)
#     opt = torch.randn(2, 96, 64, 64)
#     print("input:", sar.shape, opt.shape)
#     print("output:", block(sar, opt).shape)


# --------------------------------------------------------
# 论文：DEA-Net: Single image dehazing based on detail enhanced convolution and content-guided attention
# GitHub地址：https://github.com/cecret3350/DEA-Net/tree/main
# --------------------------------------------------------


from einops.layers.torch import Rearrange

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn
class ECA(nn.Module):
    """ECA模块：高效通道注意力"""
    def __init__(self, channels, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # (B, 1, C)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y.expand_as(x)
class ChannelAttention(nn.Module): #原版
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn
# class ChannelAttention(nn.Module):# 改进版
#     def __init__(self, reduction=8):
#         super(ChannelAttention, self).__init__()
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.reduction = reduction
#         self.ca = None  # 延后初始化
#
#     def forward(self, x):
#         if self.ca is None:
#             dim = x.shape[1]
#             self.ca = nn.Sequential(
#                 nn.Conv2d(dim, dim // self.reduction, 1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(dim // self.reduction, dim, 1),
#             ).to(x.device)
#         x_gap = self.gap(x)
#         return self.ca(x_gap)
class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2

class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        # self.ca = ECA(dim, k_size=3)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        out = result + initial
        return out

# 特征融合
# if __name__ == '__main__':
#     block = CGAFusion(32)
#     input1 = torch.rand(3, 32, 64, 64) # 输入 N C H W
#     input2 = torch.rand(3, 32, 64, 64)
#     output = block(input1, input2)
#     print(output.size())


# 论文：Reciprocal Attention Mixing Transformer for Lightweight Image Restoration(CVPR 2024 Workshop)
# 论文地址：https://arxiv.org/abs/2305.11474
# 全网最全100➕即插即用模块GitHub地址：https://github.com/ai-dawang/PlugNPlay-Modules
# H-RAMi(Hierarchical Reciprocal Attention Mixer)
class MobiVari1(nn.Module):  # MobileNet v1 Variants
    def __init__(self, dim, kernel_size, stride, act=nn.LeakyReLU, out_dim=None):
        super(MobiVari1, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.out_dim = out_dim or dim

        self.dw_conv = nn.Conv2d(dim, dim, kernel_size, stride, kernel_size // 2, groups=dim)
        self.pw_conv = nn.Conv2d(dim, self.out_dim, 1, 1, 0)
        self.act = act()

    def forward(self, x):
        out = self.act(self.pw_conv(self.act(self.dw_conv(x)) + x))
        return out + x if self.dim == self.out_dim else out

    def flops(self, resolutions):
        H, W = resolutions
        flops = H * W * self.kernel_size * self.kernel_size * self.dim + H * W * 1 * 1 * self.dim * self.out_dim  # self.dw_conv + self.pw_conv
        return flops

class MobiVari2(MobiVari1):  # MobileNet v2 Variants
    def __init__(self, dim, kernel_size, stride, act=nn.LeakyReLU, out_dim=None, exp_factor=1.2, expand_groups=4):
        super(MobiVari2, self).__init__(dim, kernel_size, stride, act, out_dim)
        self.expand_groups = expand_groups
        expand_dim = int(dim * exp_factor)
        expand_dim = expand_dim + (expand_groups - expand_dim % expand_groups)
        self.expand_dim = expand_dim

        self.exp_conv = nn.Conv2d(dim, self.expand_dim, 1, 1, 0, groups=expand_groups)
        self.dw_conv = nn.Conv2d(expand_dim, expand_dim, kernel_size, stride, kernel_size // 2, groups=expand_dim)
        self.pw_conv = nn.Conv2d(expand_dim, self.out_dim, 1, 1, 0)

    def forward(self, x):
        x1 = self.act(self.exp_conv(x))
        out = self.pw_conv(self.act(self.dw_conv(x1) + x1))
        return out + x if self.dim == self.out_dim else out

    def flops(self, resolutions):
        H, W = resolutions
        flops = H * W * 1 * 1 * (self.dim // self.expand_groups) * self.expand_dim  # self.exp_conv
        flops += H * W * self.kernel_size * self.kernel_size * self.expand_dim  # self.dw_conv
        flops += H * W * 1 * 1 * self.expand_dim * self.out_dim  # self.pw_conv
        return flops

class HRAMi(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, mv_ver=1, mv_act=nn.LeakyReLU, exp_factor=1.2, expand_groups=4):
        super(HRAMi, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        if mv_ver == 1:
            self.mobivari = MobiVari1(dim + dim // 4 + dim // 16 + dim, kernel_size, stride, act=mv_act, out_dim=dim)
        elif mv_ver == 2:
            self.mobivari = MobiVari2(dim + dim // 4 + dim // 16 + dim, kernel_size, stride, act=mv_act, out_dim=dim,
                                      exp_factor=2., expand_groups=1)

    def forward(self, attn_list):
        for i, attn in enumerate(attn_list[:-1]):
            attn = F.pixel_shuffle(attn, 2 ** i)
            x = attn if i == 0 else torch.cat([x, attn], dim=1)
        x = torch.cat([x, attn_list[-1]], dim=1)
        x = self.mobivari(x)
        return x

    def flops(self, resolutions):
        return self.mobivari.flops(resolutions)

# if __name__ == '__main__':
#     hrami = HRAMi(dim=64)
#
#     # Create sample input tensors
#     # Assume the input tensors have spatial dimensions of 32x32, 16x16, 8x8, etc.
#     input = [
#         torch.randn(1, 64, 32, 32),  # Level 0
#         torch.randn(1, 64, 16, 16),  # Level 1
#         torch.randn(1, 64, 8, 8),  # Level 2
#         torch.randn(1, 64, 32, 32)  # Level 3 (final level)
#     ]
#
#     # Pass the input through HRAMi
#     output = hrami(input)
#
#     # Print the shapes of input and output
#     print(f"Input shapes: {[attn.shape for attn in input]}")
#     print(output.size())

class CMA_Block(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, downsample=True):
        super(CMA_Block, self).__init__()

        self.downsample = downsample
        self.pool = nn.AvgPool2d(2, 2) if downsample else nn.Identity()

        self.conv1 = nn.Conv2d(in_channel, hidden_channel, 1)
        self.conv2 = nn.Conv2d(in_channel, hidden_channel, 1)
        self.conv3 = nn.Conv2d(in_channel, hidden_channel, 1)

        self.scale = hidden_channel ** -0.5

        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, rgb, freq):
        B, C, H, W = rgb.shape

        # 1. 降采样后处理
        rgb_ds = self.pool(rgb)
        freq_ds = self.pool(freq)

        h_ds, w_ds = rgb_ds.shape[2], rgb_ds.shape[3]

        # 2. 投影变换
        q = self.conv1(rgb_ds)  # B, C, h, w
        k = self.conv2(freq_ds)
        v = self.conv3(freq_ds)

        # 3. flatten 后注意力计算
        q = q.flatten(2).transpose(1, 2)  # B, N, C
        k = k.flatten(2)                  # B, C, N
        attn = torch.bmm(q, k) * self.scale
        attn = attn.softmax(dim=-1)

        # 4. 注意力权重应用到 V
        v = v.flatten(2).transpose(1, 2)  # B, N, C
        z = torch.bmm(attn, v).transpose(1, 2).view(B, -1, h_ds, w_ds)

        # 5. 上采样恢复尺寸（如果下采样过）
        if self.downsample:
            z = F.interpolate(z, size=(H, W), mode='bilinear', align_corners=False)

        out = rgb + self.conv4(z)
        return out
