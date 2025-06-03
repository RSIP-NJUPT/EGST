import torch
import torch.nn as nn


class ASLKA(nn.Module):
    """RF=7

    Args:
        nn (_type_): _description_
    """

    def __init__(self, in_chans, out_chans=None, RF=23, temporal_dim=8):
        super().__init__()
        if RF == 7:
            k1, k2, p2, d2 = 3, 3, 2, 2
        elif RF == 11:
            k1, k2, p2, d2 = 3, 5, 4, 2
        elif RF == 23:
            k1, k2, p2, d2 = 5, 7, 9, 3
        elif RF == 35:
            k1, k2, p2, d2 = 5, 11, 15, 3
        elif RF == 41:
            k1, k2, p2, d2 = 5, 13, 18, 3
        elif RF == 53:
            k1, k2, p2, d2 = 5, 17, 24, 3
        else:
            raise NotImplementedError

        # horizontal large kernel
        self.hlk_conv1 = nn.Conv3d(
            in_chans,
            in_chans,
            kernel_size=(1, 1, k1),  # t h w
            stride=1,
            padding=(0, 0, k1//2),
            groups=in_chans,
        )
        self.hlk_conv2 = nn.Conv3d(
            in_chans,
            in_chans,
            kernel_size=(1, 1, k2),
            stride=1,
            padding=(0, 0, p2),
            groups=in_chans,
            dilation=(1, 1, d2),
        )
        # vertical large kernel
        self.vlk_conv1 = nn.Conv3d(
            in_chans,
            in_chans,
            kernel_size=(1, k1, 1),
            stride=1,
            padding=(0, k1//2, 0),
            groups=in_chans,
        )
        self.vlk_conv2 = nn.Conv3d(
            in_chans,
            in_chans,
            kernel_size=(1, k2, 1),
            stride=1,
            padding=(0, p2, 0),
            groups=in_chans,
            dilation=(1, d2, 1),
        )
        # temporal large kernel
        # RF_list = [7, 11, 23, 35, 41, 53]
        # if temporal_dim >= RF:
        #     pass
        # else:
        #     true_RF = min(RF_list)
        #     for i, rf in enumerate(RF_list):
        #         if rf > temporal_dim:
        #             true_RF = RF_list[i-1]
        self.tlk_conv1 = nn.Conv3d(
            in_chans,
            in_chans,
            kernel_size=(3, 1, 1),
            stride=1,
            padding=(1, 0, 0),
            groups=in_chans,
        )
        self.tlk_conv2 = nn.Conv3d(
            in_chans,
            in_chans,
            kernel_size=(3, 1, 1),
            stride=1,
            padding=(2, 0, 0),
            groups=in_chans,
            dilation=(2, 1, 1),
        )

        self.pw_conv = nn.Conv3d(in_chans, out_chans if out_chans is not None else in_chans, 1)

    def forward(self, x):
        B, C, T, H, W = x.shape

        attn = self.hlk_conv1(x)
        attn = self.hlk_conv2(attn)
        attn = self.vlk_conv1(attn)
        # conv with dilation>1
        attn = self.vlk_conv2(attn)
        attn = self.tlk_conv1(attn)
        attn = self.tlk_conv2(attn)
        attn = self.pw_conv(attn)

        return x * attn  # B, C, T, H, W


if __name__ == "__main__":
    # 3D ==================================
    x = torch.randn((1, 2, 12, 224, 224)).cuda()
    net = ASLKA(2, RF=53).cuda()
    out = net(x)
    print(out.shape)
    pass