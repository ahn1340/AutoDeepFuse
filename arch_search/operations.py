"""
So far use only operations for testing.
"""

import torch
import torch.nn as nn

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else
    FactorizedReduce(C, C, affine=affine),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool3d(3, stride=stride,
                                                           padding=1),
    #'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool3d(3, stride=stride,
    #                                                       padding=1),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1,
                                                      affine=affine),
    '2_1d_conv_3x3': lambda C, stride, affine: Conv2_1D(C, C, 3, stride, 1,
                                                        affine=affine),
    '3d_conv_1x1': lambda C, stride, affine: Conv3D(C, C, 1, stride, 0,
                                                    affine=affine),
    '3d_conv_3x3': lambda C, stride, affine: Conv3D(C, C, 3, stride, 1,
                                                    affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: Conv3D(C, C, 3, stride, padding=2,
                                                     dilation=2, affine=affine),
}


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        # Added dilation here. Might cause error in the future.
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv3d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=False),
            nn.BatchNorm3d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class Conv3D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=1,
                 affine=True):
        super(Conv3D, self).__init__()
        self.op = nn.Sequential(
            ReLUConvBN(C_in,
                       C_out,
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=padding,
                       dilation=dilation,
                       )
        )

    def forward(self, x):
        return self.op(x)


# Depthwise separable convolution + 1D convolution (channelwise). Repeated twice as in the original paper.
class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv3d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=C_in, bias=False),
            nn.Conv3d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(C_in, affine=affine),

            nn.ReLU(inplace=False),
            nn.Conv3d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=C_in, bias=False),
            nn.Conv3d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

# 2D spatial + 1D temporal convolution. hidden channel size computed to approximate the total
# number of parameters to equal that of 3D convolution of same size.
class Conv2_1D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(Conv2_1D, self).__init__()
        if type(kernel_size) == int:
            t = d = kernel_size
        else:
            raise ValueError("Kernel size should be an int. Tuple not supported")

        # hidden size estimation to get a number of parameter similar to the 3d case
        self.hidden_size = int((t * d ** 2 * C_in * C_out) / (d ** 2 * C_in + t * C_out))

        self.conv2d = nn.Conv2d(C_in, self.hidden_size, t, stride, padding, bias=False)
        self.conv1d = nn.Conv1d(self.hidden_size, C_out, d, stride, padding, bias=False)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.bn1 = nn.BatchNorm2d(self.hidden_size)
        self.bn2 = nn.BatchNorm1d(C_out)

    def forward(self, x):
        # 2D convolution
        batch, channels, num_segments, height, width = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch * num_segments, channels, height, width)

        x =self.relu1(x)
        x = self.conv2d(x)
        x = self.bn1(x)

        # 1D convolution
        c, h, w = x.size(1), x.size(2), x.size(3)
        x = x.view(batch, num_segments, c, h, w)
        x = x.permute(0, 3, 4, 2, 1).contiguous()
        x = x.view(batch * h * w, c, num_segments)

        x = self.relu2(x)
        x = self.conv1d(x)
        x = self.bn2(x)

        # Final Output
        out_c, out_t = x.size(1), x.size(2)
        x = x.view(batch, h, w, out_c, out_t)
        x = x.permute(0, 3, 4, 1, 2)

        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:,:,::self.stride,::self.stride].mul(0.)


