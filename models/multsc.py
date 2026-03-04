import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch.nn import functional as F
import torchvision
from .unets_parts import *


class DeformConv(nn.Module):

    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()

        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3
        )
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.dwconv(self.fc1(x), H, W))
        out = self.fc2(ax)
        return out


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm2d, self).__init__()  # 调用父类构造函数
        assert c_num >= group_num  # 断言 c_num 大于等于 group_num
        self.group_num = group_num  # 设置分组数量
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))  # 创建可训练参数 gamma
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))  # 创建可训练参数 beta
        self.eps = eps  # 设置小的常数 eps 用于稳定计算

    def forward(self, x):
        N, C, H, W = x.size()  # 获取输入张量的尺寸
        x = x.view(N, self.group_num, -1)  # 将输入张量重新排列为指定的形状
        mean = x.mean(dim=2, keepdim=True)  # 计算每个组的均值
        std = x.std(dim=2, keepdim=True)  # 计算每个组的标准差
        x = (x - mean) / (std + self.eps)  # 应用批量归一化
        x = x.view(N, C, H, W)  # 恢复原始形状
        return x * self.gamma + self.beta  # 返回归一化后的张量


class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
        self.DDW = DeformConv(in_ch, kernel_size=(3, 3), padding=1,
                              groups=in_ch)

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.DDW(out)
        out = self.pointwise_conv(out)
        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,  # 输出通道数
                 group_num: int = 16,  # 分组数，默认为16
                 gate_treshold: float = 0.5,  # 门控阈值，默认为0.5
                 torch_gn: bool = False  # 是否使用PyTorch内置的GroupNorm，默认为False
                 ):
        super().__init__()  # 调用父类构造函数
        # 初始化 GroupNorm 层或自定义 GroupBatchnorm2d 层
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold  # 设置门控阈值
        self.sigomid = nn.Sigmoid()  # 创建 sigmoid 激活函数
        self.threshold_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 得到 [B, C, 1, 1]
            nn.Flatten(),  # 得到 [B, C]
            nn.Linear(oup_channels, oup_channels // 4),
            nn.ReLU(),
            nn.Linear(oup_channels // 4, 1),
            nn.Sigmoid()  # 输出0~1之间的阈值
        )

    def forward(self, x):
        gn_x = self.gn(x)  # 应用分组批量归一化
        w_gamma = self.gn.gamma / sum(self.gn.gamma)  # 计算 gamma 权重
        reweights = self.sigomid(gn_x * w_gamma)  # 计算重要性权重
        adaptive_threshold = self.threshold_generator(x).unsqueeze(-1).unsqueeze(-1)
        info_mask = (reweights >= adaptive_threshold).float()
        noninfo_mask = (reweights < adaptive_threshold).float()
        x_1 = info_mask * x  # 使用信息门控掩码
        x_2 = noninfo_mask * x  # 使用非信息门控掩码
        x = self.reconstruct(x_1, x_2)  # 重构特征
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)  # 拆分特征为两部分
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)  # 拆分特征为两部分
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)  # 重构特征并连接


class CRU(nn.Module):
    def __init__(self, op_channel: int, alpha: float = 1 / 2, squeeze_radio: int = 2, group_size: int = 2,
                 group_kernel_size: int = 3):
        super().__init__()  # 调用父类构造函数
        self.up_channel = up_channel = int(alpha * op_channel)  # 计算上层通道数
        self.low_channel = low_channel = op_channel - up_channel  # 计算下层通道数
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)  # 创建卷积层
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)  # 创建卷积层
        # 上层特征转换
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)  # 创建卷积层
        self.DSC1 = DepthwiseSeparableConvolution(up_channel // squeeze_radio, op_channel)
        # 下层特征转换
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)  # 创建卷积层
        self.DSC2 = DepthwiseSeparableConvolution(low_channel // squeeze_radio, low_channel // squeeze_radio)
        self.advavg = nn.AdaptiveAvgPool2d(1)  # 创建自适应平均池化层

    def forward(self, x):
        # 分割输入特征
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # 上层特征转换
        Y1 = self.GWC(up) + self.DSC1(up)
        # 下层特征转换
        Y2 = torch.cat([self.PWC2(low), self.DSC2(up)], dim=1)
        # 特征融合
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(self, op_channel: int, group_num: int = 16, gate_treshold: float = 0.5, alpha: float = 1 / 2,
                 squeeze_radio: int = 2, group_size: int = 2, group_kernel_size: int = 3):
        super().__init__()  # 调用父类构造函数
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.SRU = SRU(op_channel, group_num=group_num, gate_treshold=gate_treshold)  # 创建 SRU 层
        self.CRU = CRU(op_channel, alpha=alpha, squeeze_radio=squeeze_radio, group_size=group_size,
                       group_kernel_size=group_kernel_size)  # 创建 CRU 层
        mip = max(8, op_channel // 32)
        self.conv1 = nn.Conv2d(op_channel, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_sigmoid()
        self.conv_h = nn.Conv2d(mip, op_channel, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, op_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.SRU(self.pool_h(x))
        x_w = self.pool_w(x)
        x_w = self.CRU(x_w.permute(0, 1, 3, 2))

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


# Optimized MBConv
class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_rate=4, kernel_size=3, stride=1):
        super(MBConv, self).__init__()
        mid_ch = in_ch * expansion_rate
        self.use_residual = (in_ch == out_ch and stride == 1)

        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(8, mid_ch)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                               groups=mid_ch, bias=False)
        self.norm2 = nn.GroupNorm(8, mid_ch)
        self.act2 = nn.GELU()
        self.ScConv = ScConv(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False)
        self.norm3 = nn.GroupNorm(8, mid_ch)  # LayerNorm

    def forward(self, x):
        residual = x
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.conv2(x)
        x = self.act2(self.norm2(x))
        x = self.ScConv(x)  # Apply spatial convolution or similar attention mechanism
        x = self.norm3(self.conv3(x))
        if self.use_residual:
            x += residual

        return x


# Optimized Refine Cross Attention
class Multiscale_spatial_channel_calibration(nn.Module):
    def __init__(self, attention_position, in_dim, dim, mbconv_expansion_rate=4):
        super(Multiscale_spatial_channel_calibration, self).__init__()
        self.attention_position = attention_position
        self.conv1 = ScConv(dim)
        self.conv2 = ScConv(dim)
        self.conv3 = ScConv(dim)

        target_idx = attention_position
        ch_down_in = in_dim[target_idx - 1]
        ch_mid = in_dim[target_idx]
        ch_up_in = in_dim[target_idx + 1]

        self.down = Down(ch_down_in, ch_mid)
        self.DDW = DepthwiseSeparableConvolution(ch_mid, ch_mid)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch_up_in, ch_mid, kernel_size=2, stride=2),  # 2x deconv
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_mid, ch_mid, kernel_size=3, padding=1),  # 3x3 conv to fix boundaries
            nn.ReLU(inplace=True)
        )
        self.conv1_1 = nn.Conv2d(3 * dim, dim, kernel_size=1, bias=False)
        self.linear_downsample = ScConv(dim)

    def forward(self, x):
        target_idx = self.attention_position

        embed_down = self.down(x[target_idx - 1])
        embed_mid = self.DDW(x[target_idx])
        embed_up = self.up(x[target_idx + 1])

        embed_down = self.conv1(embed_down)
        embed_mid = self.conv2(embed_mid)
        embed_up = self.conv3(embed_up)

        embed = torch.cat([embed_down, embed_mid, embed_up], dim=1)
        embed = self.conv1_1(embed)
        embed = self.linear_downsample(embed)
        return embed






