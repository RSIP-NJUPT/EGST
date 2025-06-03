import math
import torch
from torch import nn
from einops.layers.torch import Rearrange
import os
import matplotlib.pyplot as plt
import numpy as np



class Conv2d_ad(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
        theta=1.0,
    ):
        super(Conv2d_ad, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_ad = (
            conv_weight - self.theta * conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        )
        conv_weight_ad = Rearrange(
            'c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3]
        )(conv_weight_ad)
        return conv_weight_ad, self.conv.bias


class Conv2d_cd(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
        theta=1.0,
    ):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_cd = torch.cuda.FloatTensor(
            conv_shape[0], conv_shape[1], 3 * 3
        ).fill_(0)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
        conv_weight_cd = Rearrange(
            'c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3]
        )(conv_weight_cd)
        return conv_weight_cd, self.conv.bias

class Conv2d_prewittx(nn.Module):
    def __init__(
        self,
        out_channels,
    ):
        super().__init__()
        self.out_channels = out_channels
        # init scale & bias
        scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 1e-1
        self.scale = nn.Parameter(torch.FloatTensor(scale))

        bias = torch.randn(self.out_channels) * 1e-1
        bias = torch.reshape(bias, (self.out_channels,))
        self.bias = nn.Parameter(torch.FloatTensor(bias))
        # init mask 左-右
        self.mask = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)
        for i in range(self.out_channels):
            self.mask[i, 0, 0, 0] = 1.0
            self.mask[i, 0, 1, 0] = 1.0
            self.mask[i, 0, 2, 0] = 1.0
            self.mask[i, 0, 0, 2] = -1.0
            self.mask[i, 0, 1, 2] = -1.0
            self.mask[i, 0, 2, 2] = -1.0
        self.mask = nn.Parameter(data=self.mask, requires_grad=False)

    def get_weight(self):
        return self.scale * self.mask, self.bias


class Conv2d_prewitty(nn.Module):
    def __init__(
        self,
        out_channels,
    ):
        super().__init__()
        self.out_channels = out_channels
        # init scale & bias
        scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 1e-1
        self.scale = nn.Parameter(torch.FloatTensor(scale))

        bias = torch.randn(self.out_channels) * 1e-1
        bias = torch.reshape(bias, (self.out_channels,))
        self.bias = nn.Parameter(torch.FloatTensor(bias))
        # init mask 上-下
        self.mask = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)
        for i in range(self.out_channels):
            self.mask[i, 0, 0, 0] = 1.0
            self.mask[i, 0, 0, 1] = 1.0
            self.mask[i, 0, 0, 2] = 1.0
            self.mask[i, 0, 2, 0] = -1.0
            self.mask[i, 0, 2, 1] = -1.0
            self.mask[i, 0, 2, 2] = -1.0
        self.mask = nn.Parameter(data=self.mask, requires_grad=False)

    def get_weight(self):
        return self.scale * self.mask, self.bias


class Conv2d_prewitt45(nn.Module):
    def __init__(
        self,
        out_channels,
    ):
        super().__init__()
        self.out_channels = out_channels
        # init scale & bias
        scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 1e-1
        self.scale = nn.Parameter(torch.FloatTensor(scale))

        bias = torch.randn(self.out_channels) * 1e-1
        bias = torch.reshape(bias, (self.out_channels,))
        self.bias = nn.Parameter(torch.FloatTensor(bias))
        # init mask 右上-左下
        self.mask = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)
        for i in range(self.out_channels):
            self.mask[i, 0, 0, 1] = 1.0
            self.mask[i, 0, 0, 2] = 1.0
            self.mask[i, 0, 1, 0] = -1.0
            self.mask[i, 0, 1, 2] = 1.0
            self.mask[i, 0, 2, 0] = -1.0
            self.mask[i, 0, 2, 1] = -1.0
        self.mask = nn.Parameter(data=self.mask, requires_grad=False)

    def get_weight(self):
        return self.scale * self.mask, self.bias


class Conv2d_prewitt135(nn.Module):
    def __init__(
        self,
        out_channels,
    ):
        super().__init__()
        self.out_channels = out_channels
        # init scale & bias
        scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 1e-1
        self.scale = nn.Parameter(torch.FloatTensor(scale))

        bias = torch.randn(self.out_channels) * 1e-1
        bias = torch.reshape(bias, (self.out_channels,))
        self.bias = nn.Parameter(torch.FloatTensor(bias))
        # init mask 右下-左上
        self.mask = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)
        for i in range(self.out_channels):
            self.mask[i, 0, 0, 0] = -1.0
            self.mask[i, 0, 0, 1] = -1.0
            self.mask[i, 0, 1, 0] = -1.0
            self.mask[i, 0, 1, 2] = 1.0
            self.mask[i, 0, 2, 1] = 1.0
            self.mask[i, 0, 2, 2] = 1.0
        self.mask = nn.Parameter(data=self.mask, requires_grad=False)

    def get_weight(self):
        return self.scale * self.mask, self.bias

class Conv2d_sobelx(nn.Module):
    def __init__(
        self,
        out_channels,
    ):
        super().__init__()
        self.out_channels = out_channels
        # init scale & bias
        scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 1e-1
        self.scale = nn.Parameter(torch.FloatTensor(scale))

        bias = torch.randn(self.out_channels) * 1e-1
        bias = torch.reshape(bias, (self.out_channels,))
        self.bias = nn.Parameter(torch.FloatTensor(bias))
        # init mask 左-右
        self.mask = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)
        for i in range(self.out_channels):
            self.mask[i, 0, 0, 0] = 1.0
            self.mask[i, 0, 1, 0] = 2.0
            self.mask[i, 0, 2, 0] = 1.0
            self.mask[i, 0, 0, 2] = -1.0
            self.mask[i, 0, 1, 2] = -2.0
            self.mask[i, 0, 2, 2] = -1.0
        self.mask = nn.Parameter(data=self.mask, requires_grad=False)

    def get_weight(self):
        return self.scale * self.mask, self.bias


class Conv2d_sobely(nn.Module):
    def __init__(
        self,
        out_channels,
    ):
        super().__init__()
        self.out_channels = out_channels
        # init scale & bias
        scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 1e-1
        self.scale = nn.Parameter(torch.FloatTensor(scale))

        bias = torch.randn(self.out_channels) * 1e-1
        bias = torch.reshape(bias, (self.out_channels,))
        self.bias = nn.Parameter(torch.FloatTensor(bias))
        # init mask 上-下
        self.mask = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)
        for i in range(self.out_channels):
            self.mask[i, 0, 0, 0] = 1.0
            self.mask[i, 0, 0, 1] = 2.0
            self.mask[i, 0, 0, 2] = 1.0
            self.mask[i, 0, 2, 0] = -1.0
            self.mask[i, 0, 2, 1] = -2.0
            self.mask[i, 0, 2, 2] = -1.0
        self.mask = nn.Parameter(data=self.mask, requires_grad=False)

    def get_weight(self):
        return self.scale * self.mask, self.bias


class Conv2d_sobel45(nn.Module):
    def __init__(
        self,
        out_channels,
    ):
        super().__init__()
        self.out_channels = out_channels
        # init scale & bias
        scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 1e-1
        self.scale = nn.Parameter(torch.FloatTensor(scale))

        bias = torch.randn(self.out_channels) * 1e-1
        bias = torch.reshape(bias, (self.out_channels,))
        self.bias = nn.Parameter(torch.FloatTensor(bias))
        # init mask 右上-左下
        self.mask = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)
        for i in range(self.out_channels):
            self.mask[i, 0, 0, 1] = 1.0
            self.mask[i, 0, 0, 2] = 2.0
            self.mask[i, 0, 1, 0] = -1.0
            self.mask[i, 0, 1, 2] = 1.0
            self.mask[i, 0, 2, 0] = -2.0
            self.mask[i, 0, 2, 1] = -1.0
        self.mask = nn.Parameter(data=self.mask, requires_grad=False)

    def get_weight(self):
        return self.scale * self.mask, self.bias


class Conv2d_sobel135(nn.Module):
    def __init__(
        self,
        out_channels,
    ):
        super().__init__()
        self.out_channels = out_channels
        # init scale & bias
        scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 1e-1
        self.scale = nn.Parameter(torch.FloatTensor(scale))

        bias = torch.randn(self.out_channels) * 1e-1
        bias = torch.reshape(bias, (self.out_channels,))
        self.bias = nn.Parameter(torch.FloatTensor(bias))
        # init mask 右下-左上
        self.mask = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)
        for i in range(self.out_channels):
            self.mask[i, 0, 0, 0] = -2.0
            self.mask[i, 0, 0, 1] = -1.0
            self.mask[i, 0, 1, 0] = -1.0
            self.mask[i, 0, 1, 2] = 1.0
            self.mask[i, 0, 2, 1] = 1.0
            self.mask[i, 0, 2, 2] = 2.0
        self.mask = nn.Parameter(data=self.mask, requires_grad=False)

    def get_weight(self):
        return self.scale * self.mask, self.bias


class Conv2d_scharr45(nn.Module):
    def __init__(
        self,
        out_channels,
    ):
        super().__init__()
        self.out_channels = out_channels
        # init scale & bias
        scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 1e-1
        self.scale = nn.Parameter(torch.FloatTensor(scale))

        bias = torch.randn(self.out_channels) * 1e-1
        bias = torch.reshape(bias, (self.out_channels,))
        self.bias = nn.Parameter(torch.FloatTensor(bias))
        # init mask 右上-左下
        self.mask = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)
        for i in range(self.out_channels):
            self.mask[i, 0, 0, 1] = 3.0
            self.mask[i, 0, 0, 2] = 10.0
            self.mask[i, 0, 1, 0] = -3.0
            self.mask[i, 0, 1, 2] = 3.0
            self.mask[i, 0, 2, 0] = -10.0
            self.mask[i, 0, 2, 1] = -3.0
        self.mask = nn.Parameter(data=self.mask, requires_grad=False)

    def get_weight(self):
        return self.scale * self.mask, self.bias


class Conv2d_scharr135(nn.Module):
    def __init__(
        self,
        out_channels,
    ):
        super().__init__()
        self.out_channels = out_channels
        # init scale & bias
        scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 1e-1
        self.scale = nn.Parameter(torch.FloatTensor(scale))

        bias = torch.randn(self.out_channels) * 1e-1
        bias = torch.reshape(bias, (self.out_channels,))
        self.bias = nn.Parameter(torch.FloatTensor(bias))
        # init mask 右下-左上
        self.mask = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)
        for i in range(self.out_channels):
            self.mask[i, 0, 0, 0] = -10.0
            self.mask[i, 0, 0, 1] = -3.0
            self.mask[i, 0, 1, 0] = -3.0
            self.mask[i, 0, 1, 2] = 3.0
            self.mask[i, 0, 2, 1] = 3.0
            self.mask[i, 0, 2, 2] = 10.0
        self.mask = nn.Parameter(data=self.mask, requires_grad=False)

    def get_weight(self):
        return self.scale * self.mask, self.bias


class Conv2d_scharrx(nn.Module):
    def __init__(
        self,
        out_channels,
    ):
        super().__init__()
        self.out_channels = out_channels
        # init scale & bias
        scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 1e-1
        self.scale = nn.Parameter(torch.FloatTensor(scale))

        bias = torch.randn(self.out_channels) * 1e-1
        bias = torch.reshape(bias, (self.out_channels,))
        self.bias = nn.Parameter(torch.FloatTensor(bias))
        # init mask
        self.mask = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)
        for i in range(self.out_channels):
            self.mask[i, 0, 0, 0] = 3.0
            self.mask[i, 0, 1, 0] = 10.0
            self.mask[i, 0, 2, 0] = 3.0
            self.mask[i, 0, 0, 2] = -3.0
            self.mask[i, 0, 1, 2] = -10.0
            self.mask[i, 0, 2, 2] = -3.0
        self.mask = nn.Parameter(data=self.mask, requires_grad=False)

    def get_weight(self):
        return self.scale * self.mask, self.bias


class Conv2d_scharry(nn.Module):
    def __init__(
        self,
        out_channels,
    ):
        super().__init__()
        self.out_channels = out_channels
        # init scale & bias
        scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 1e-1
        self.scale = nn.Parameter(torch.FloatTensor(scale))

        bias = torch.randn(self.out_channels) * 1e-1
        bias = torch.reshape(bias, (self.out_channels,))
        self.bias = nn.Parameter(torch.FloatTensor(bias))
        # init mask
        self.mask = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)
        for i in range(self.out_channels):
            self.mask[i, 0, 0, 0] = 3.0
            self.mask[i, 0, 0, 1] = 10.0
            self.mask[i, 0, 0, 2] = 3.0
            self.mask[i, 0, 2, 0] = -3.0
            self.mask[i, 0, 2, 1] = -10.0
            self.mask[i, 0, 2, 2] = -3.0
        self.mask = nn.Parameter(data=self.mask, requires_grad=False)

    def get_weight(self):
        return self.scale * self.mask, self.bias


class Conv2d_laplacian(nn.Module):
    def __init__(
        self,
        out_channels,
    ):
        super().__init__()
        self.out_channels = out_channels
        # init scale & bias
        scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 1e-1
        self.scale = nn.Parameter(torch.FloatTensor(scale))

        bias = torch.randn(self.out_channels) * 1e-1
        bias = torch.reshape(bias, (self.out_channels,))
        self.bias = nn.Parameter(torch.FloatTensor(bias))
        # init mask
        self.mask = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)
        for i in range(self.out_channels):
            self.mask[i, 0, 0, 1] = -1.0
            self.mask[i, 0, 1, 0] = -1.0
            self.mask[i, 0, 1, 2] = -1.0
            self.mask[i, 0, 2, 1] = -1.0
            self.mask[i, 0, 1, 1] = 4.0
        self.mask = nn.Parameter(data=self.mask, requires_grad=False)

    def get_weight(self):
        return self.scale * self.mask, self.bias



class EGConv(nn.Module):
    def __init__(self, dim):
        super(EGConv, self).__init__()
        self.dim = dim
        self.conv_shx = Conv2d_scharrx(dim)
        self.conv_shy = Conv2d_scharry(dim)
        self.conv_sh45 = Conv2d_scharr45(dim)
        self.conv_sh135 = Conv2d_scharr135(dim)
        self.conv_lpl = Conv2d_laplacian(dim)
        self.conv3x3 = nn.Conv2d(dim, dim, 3, padding=1, bias=True, groups=dim)

    def forward(self, x):
        w1, b1 = self.conv3x3.weight, self.conv3x3.bias
        w2, b2 = self.conv_shx.get_weight()
        w3, b3 = self.conv_shy.get_weight()
        w4, b4 = self.conv_sh45.get_weight()
        w5, b5 = self.conv_sh135.get_weight()
        w6, b6 = self.conv_lpl.get_weight()
        w = w1 + w2 + w3 + w4 + w5 + w6
        b = b1 + b2 + b3 + b4 + b5 + b6
        out = nn.functional.conv2d(
            input=x, weight=w, bias=b, stride=1, padding=1, groups=self.dim
        )

        # self.save_feature_map(x, out)
        return out
    
    def save_feature_map(self, x, out, save_dir=r'work_dirs/gt_224x224_40k_brandenburg_s1/vis'):
        x = x.detach().cpu().numpy()
        out = out.detach().cpu().numpy()

        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "x.npy"), x)
        np.save(os.path.join(save_dir, "out.npy"), out)
        pass





if __name__ == '__main__':
    x = torch.randn(1, 30, 32, 32).cuda()
    net = EGConv(30).cuda()
    out = net(x)
    print(out.shape)
    pass
