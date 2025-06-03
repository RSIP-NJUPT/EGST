import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F

from .adconv import ADConv
from .aslka import ASLKA
from ...builder import BACKBONES

def cosine_similarity(x):
    B, N1, N2, C = x.shape
    x_norm = x / x.norm(dim=-1, keepdim=True)  # 对每个节点的特征向量进行归一化
    similarity_matrix = x_norm @ x_norm.transpose(-1, -2)

    return similarity_matrix  # B, N1, N2, N2


class Conv3d_BN_ACT(nn.Module):
    def __init__(
        self,
        in_chans,
        out_chans,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        dilation=1,
        act_layer=nn.GELU,
    ):
        super(Conv3d_BN_ACT, self).__init__()
        self.conv = nn.Conv3d(
            in_chans,
            out_chans,
            kernel_size,
            stride,
            padding,
            bias=False,
            dilation=dilation,
            groups=groups,
        )
        self.bn = nn.BatchNorm3d(out_chans)
        self.act = act_layer()

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x  # B C T H W


class ConditionalPositionEncoding(nn.Module):
    """
    Conditional Position Encoding from https://arxiv.org/abs/2102.10882

    Args:
        nn (_type_): _description_
    """

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim),
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.proj(x) + x

        return x  # B, C, T, H, W


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of output channels. Default: 60.
        temporal_dim (int): Number of input image channels. Default: 16.
    """

    def __init__(
        self,
        in_chans=3,
        embed_dim=60,
        temporal_dim=16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.temporal_dim = temporal_dim
        # 2↓ for H W
        self.conv_down = Conv3d_BN_ACT(
            in_chans, embed_dim // 2, kernel_size=3, stride=(1, 2, 2), padding=1
        )
        # 2↓ for T H W
        self.conv_down2 = Conv3d_BN_ACT(embed_dim // 2, embed_dim, 3, 2, 1)
        self.ad_conv1 = ADConv(
            embed_dim // 2,
        )
        self.ad_conv2 = ADConv(
            embed_dim,
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        assert self.temporal_dim == T
        x = self.conv_down(x)
        x = self.ad_conv1(x.transpose(1,2).reshape(B*T, self.embed_dim // 2, H // 2, W // 2))  # B*T, C, H, W
        x = x.reshape(B, T, self.embed_dim // 2, H // 2, W // 2).transpose(1, 2) # B, C, T, H, W

        x = self.conv_down2(x)
        x = self.ad_conv2(x.transpose(1,2).reshape(B*T//2, self.embed_dim, H // 4, W // 4))  # B, C*T, H, W
        x = x.reshape(B, T // 2, self.embed_dim, H // 4, W // 4).transpose(1, 2) # B, C, T, H, W

        return x  # B, C, T, H, W


class PatchMerging(PatchEmbed):
    def __init__(
        self,
        in_chans=3,
        embed_dim=60,
        temporal_dim=16,
    ):
        super().__init__(in_chans, embed_dim, temporal_dim)
        self.embed_dim = embed_dim
        # 2↓ for T H W
        self.conv_down = Conv3d_BN_ACT(
            in_chans, embed_dim, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        assert self.temporal_dim == T
        x = self.conv_down(x)

        return x  # B, C, T, H, W


class GFFN(nn.Module):
    """Grouped FFN

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        in_chans,
        out_chans=None,
        num_groups=5,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        self.num_groups = num_groups
        _in_chans = in_chans // num_groups
        _out_chans = out_chans // num_groups
        self.fc1 = nn.Linear(_in_chans, 4 * _in_chans)
        self.act = act_layer()
        self.fc2 = nn.Linear(4 * _in_chans, _out_chans)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if x.ndim == 4:
            B, N1, N2, C = x.shape
            x = x.reshape(B, N1, N2, self.num_groups, -1)
            x = self.drop(self.act(self.fc1(x)))
            x = self.drop(self.fc2(x))
            x = x.view(B, N1, N2, C)  # B, N1, N2, C
        elif x.ndim == 5:
            B, T, H, W, C = x.shape
            x = x.reshape(B, T, H, W, self.num_groups, -1)
            x = self.drop(self.act(self.fc1(x)))
            x = self.drop(self.fc2(x))
            x = x.view(B, T, H, W, -1)  # B, T, H, W, C

        return x  # B, T, H, W, C / B, N1, N2, C


class SelfGraphAttention(nn.Module):

    def __init__(self, 
        dim,
        attn_type="local",
        window_size=[7, 7, 7],
        num_groups=5,
        keep_ratio=0.5):
        super().__init__()
        # graph
        self.attn_type = attn_type
        self.keep_ratio = keep_ratio
        self.ws = window_size
        
        self.linear = GFFN(
            in_chans=dim,
            out_chans=dim,
            num_groups=num_groups,
        )
        # sa
        self.qkv = GFFN(
            in_chans=dim,
            out_chans=dim * 3,
            num_groups=num_groups,
        )
        self.proj = GFFN(
            in_chans=dim,
            out_chans=dim,
            num_groups=num_groups,
        )

    def forward(self, x):
        B, _T, H, W, C = x.shape
        qkv = self.qkv(x)  # B, T, H, W, 3C
        qkv = qkv.view(B, _T, H, W, 3, C).permute(4, 0, 1, 2, 3, 5)  # 3 B T H W C

        # ======================== start of window partition =======================
        ws_t, ws_h, ws_w = self.ws
        pad_flag = False
        if _T % ws_t != 0:
            pad_flag = True
            padding = (
                0,
                0,
                0,
                ws_t - _T % ws_t,
            )
            qkv = F.pad(qkv.reshape(3, B, _T, -1), padding, "replicate")
            x = F.pad(x.reshape(B, _T, -1), padding, "replicate")
        T = qkv.shape[2]

        # ws_t = min(ws_t, _T)
        nw_t, nw_h, nw_w = T // ws_t, H // ws_h, W // ws_w
        nw = nw_t * nw_h * nw_w
        # local is actually global!
        if nw == 1:
            self.attn_type = "local"
        N = ws_t * ws_h * ws_w
        if self.attn_type == "local":
            q, k, v = (
                qkv.reshape(3, B, nw_t, ws_t, nw_h, ws_h, nw_w, ws_w, C)
                .permute(0, 1, 2, 4, 6, 3, 5, 7, 8)
                .reshape(3, B, nw, N, C)  # 3, B, nw, N, C
                .contiguous()
            )
            x = (
                x.reshape(B, nw_t, ws_t, nw_h, ws_h, nw_w, ws_w, C)
                .permute(0, 1, 3, 5, 2, 4, 6, 7)
                .reshape(B, nw, N, C)  # B, nw, N, C
                .contiguous()
            )
            nk = max(N - int(N*self.keep_ratio) + 1, 1)
        elif self.attn_type == "global":
            q, k, v = (
                qkv.reshape(3, B, nw_t, ws_t, nw_h, ws_h, nw_w, ws_w, C)
                .permute(0, 1, 3, 5, 7, 2, 4, 6, 8)
                .reshape(3, B, N, nw, C)  # 3, B, N, nw, C
                .contiguous()
            )
            x = (
                x.reshape(B, nw_t, ws_t, nw_h, ws_h, nw_w, ws_w, C)
                .permute(0, 2, 4, 6, 1, 3, 5, 7)
                .reshape(B, N, nw, C)  # B, N, nw, C
                .contiguous()
            )
            nk = 1
        # ======================== end of window partition =======================
        # self attention
        attn = q @ k.transpose(-1, -2) / (C**-0.5)
        # B, N, nw, nw for "local" | B, nw, N, N for "global"
        attn = attn.softmax(dim=-1)

        # graph attention
        cossim_score = cosine_similarity(x)
        cossim_score = torch.clamp(cossim_score, min=0)
        
        score = cossim_score * attn
        score = score / (score.sum(dim=-1, keepdim=True) + 1e-12)

        self.threshold, _index = torch.kthvalue(
            score, nk, dim=-1, keepdim=True
        )
        normed_adj_matrix = torch.where(
            score < self.threshold, 0.0, score
        )

        x = (attn + normed_adj_matrix) @ v
        # ======================== start of window merging =======================
        if self.attn_type == "local":
            x = (
                x.reshape(B, nw_t, nw_h, nw_w, ws_t, ws_h, ws_w, C)
                .permute(0, 1, 4, 2, 5, 3, 6, 7)
                .reshape(B, T, H, W, C)
                .contiguous()
            )
        elif self.attn_type == "global":
            x = (
                x.reshape(B, ws_t, ws_h, ws_w, nw_t, nw_h, nw_w, C)
                .permute(0, 4, 1, 5, 2, 6, 3, 7)
                .reshape(B, T, H, W, C)
                .contiguous()
            )
        # ======================== end of window merging =======================
        if pad_flag:
            x = x[:,:-(ws_t - _T % ws_t),...]
        x = self.proj(x)  # B, T, H, W, C

        return x


class BasicLayer(torch.nn.Module):
    def __init__(
        self,
        in_chans,
        attn_type="local",
        window_size=[7, 7, 7],
        num_groups=5,
        keep_ratio=0.3
    ):
        super().__init__()
        self.attn_type = attn_type
        branch_chans = in_chans // 2
        self.sa_graph_mixer = SelfGraphAttention(
            branch_chans,
            attn_type,
            window_size,
            num_groups=num_groups,
            keep_ratio=keep_ratio)

    def forward(self, x):
        
        B, T, H, W, C = x.shape

        x1, x2 = torch.chunk(x, 2, dim=-1)  # B, T, H, W, C
        y1 = self.sa_graph_mixer(x1) + x1  # B, T, H, W, C

        y = torch.cat((y1, x2), dim=-1)  # B, T, H, W, C

        return y  # B, T, H, W, C


class BasicBlock(torch.nn.Module):
    def __init__(
        self,
        in_chans,
        block_type="early",
        num_groups=5,
        window_size=[7, 7, 7],
        keep_ratio=0.5,
        RF=23,
        temporal_dim=8,
    ):
        super().__init__()

        self.block_type = block_type
        self.ffn = GFFN(
            in_chans,
            in_chans,
            num_groups,
        )
        if block_type == "early":  # for early stages, i.e., stage1 and stage2 
            self.ln1 = nn.LayerNorm(in_chans)
            self.ln2 = nn.LayerNorm(in_chans)
            self.mixer = ASLKA(in_chans, RF=RF, temporal_dim=temporal_dim)
        elif block_type == "later":  # for later stages, i.e., stage3, 4.
            self.ln1 = nn.LayerNorm(in_chans)
            self.ln2 = nn.LayerNorm(in_chans)
            self.ln3 = nn.LayerNorm(in_chans)
            self.local_mixer = BasicLayer(
                in_chans,
                "local",
                window_size=window_size,
                num_groups=num_groups,
                keep_ratio=keep_ratio
            )
            self.global_mixer = BasicLayer(
                in_chans,
                "global",
                window_size=window_size,
                num_groups=num_groups,
                keep_ratio=keep_ratio
            )

    def forward(self, x):
        B, T, H, W, C = x.shape
        if self.block_type == "early":
            x = x + self.mixer(self.ln1(x).permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)  # B, C, T, H, W
            x = x + self.ffn(self.ln2(x))  # B, T, H, W, C
        elif self.block_type == "later":
            x = x + self.local_mixer(self.ln1(x))  # B, T, H, W, C
            x = x + self.global_mixer(self.ln2(x))  # B, T, H, W, C
            x = x + self.ffn(self.ln3(x))  # B, T, H, W, C

        return x  #  B, T, H, W, C



class GraphTransformerStage(torch.nn.Module):

    def __init__(
        self,
        in_chans,
        window_size,
        block_type="early",
        depth=1,
        num_groups=5,
        keep_ratio=0.5,
        RF=23,
        temporal_dim=8,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                BasicBlock(
                    in_chans,
                    block_type,
                    num_groups,
                    window_size,
                    keep_ratio=keep_ratio,
                    RF=RF,
                    temporal_dim=temporal_dim,
                )
                for _ in range(depth)
            ]
        )
        # conditional position embedding
        self.cpes = nn.ModuleList(
            [ConditionalPositionEncoding(in_chans) for _ in range(depth)]
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        for cpe, blk in zip(self.cpes, self.blocks):
            x = cpe(x)  # PEG here
            x = x.permute(0, 2, 3, 4, 1)  # B, T, H, W, C
            x = blk(x)  # B, T, H, W, C
            x = x.permute(0, 4, 1, 2, 3)  # B, C, T, H, W
        return x  # B, C, T, H, W


@BACKBONES.register_module()
class GraphTransformer(nn.Module):
    """ GraphTransformer 是最初用来训练的原始的网络，
    和 EGSTNet/mmseg/models/backbones/egst/egstnet.py 中的 EGSTNet 是完全一样的，仅有部分模块名称改变。
    可以直接学习 EGSTNet/mmseg/models/backbones/egst/egstnet.py 中的 EGSTNet，
    但预训练的权重仍是加载 GraphTransformer。
    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        in_chans=2,
        num_classes=8,
        embed_dims=[120, 240, 360, 480],
        window_size=[7, 7, 7],
        depths=[1, 1, 2, 1],
        dataset_flag="slovenia",
        nodes_keep_ratio=0.3,
        RF=23,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.dataset_flag = dataset_flag
        if self.dataset_flag == "slovenia":
            temporal_dim_list = [16, 8, 4, 2, 1]  # slovenia
        elif self.dataset_flag == "brandenburg":  # for brandenburg
            temporal_dim_list = [48, 24, 12, 6, 3]
        elif self.dataset_flag == "pastisr":  # for pastisr
            temporal_dim_list = [80, 40, 20, 10, 5]
        # encoder
        self.stages = nn.ModuleList()
        self.patch_embeds = nn.ModuleList()
        for i in range(len(depths)):  # 0 1 2 3
            _stage = GraphTransformerStage(
                embed_dims[i],
                window_size=window_size,
                block_type="early" if i < 2 else "later",
                depth=depths[i],
                num_groups=5,
                keep_ratio=nodes_keep_ratio,
                RF=RF,
                temporal_dim=temporal_dim_list[i+1] # for ASLKA
            )
            self.stages.append(_stage)

            # patch embed
            if i == 0:
                _patch_embed = PatchEmbed(
                    in_chans=in_chans,
                    embed_dim=embed_dims[i],
                    temporal_dim=temporal_dim_list[i],
                )
            else:  # downsampling 2x
                _patch_embed = PatchMerging(
                    in_chans=embed_dims[i - 1],
                    embed_dim=embed_dims[i],
                    temporal_dim=temporal_dim_list[i],
                )

            self.patch_embeds.append(_patch_embed)

        decoder_dim = 512
        self.lateral_layers = nn.ModuleList(
            [
                nn.Conv3d(in_c, decoder_dim, kernel_size=1)
                for i, in_c in enumerate(embed_dims)
            ]
        )

        # final upsample
        self.up_x4 = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_dim, decoder_dim, 2, 2, 0
            ),
            nn.GELU(),
            nn.Conv2d(decoder_dim, decoder_dim, 3, 1, 1, groups=decoder_dim),
            nn.GELU(),
            nn.ConvTranspose2d(
                decoder_dim, decoder_dim, 2, 2, 0
            ),
            nn.GELU(),
            nn.Conv2d(decoder_dim, decoder_dim, 3, 1, 1, groups=decoder_dim),
            nn.GELU(),
        )
        # segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_dim, num_classes, 1),
        )

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            import math

            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv3d):
                fan_out = (
                    m.kernel_size[0]
                    * m.kernel_size[1]
                    * m.kernel_size[2]
                    * m.out_channels
                )
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            # logger = get_root_logger()
            # logger.info(f"load model from: {pretrained}")

            weights = None
            self.load_from(weights)

        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward_features(self, x):
        B, C, T, H, W = x.shape
        encoder_feats = []  # all B C T H W
        for i, stage in enumerate(self.stages):
            # patch embed before stage 1, after that patch embed is actually patch merging
            x = self.patch_embeds[i](x)  # B C T H W
            x = stage(x)  # B C T H W
            encoder_feats.append(x)

        return encoder_feats

    def forward_features_up(self, encoder_feats):
        laterals = [
            torch.mean(lateral(fea), dim=2, keepdim=False)
            for lateral, fea in zip(self.lateral_layers, encoder_feats)
        ]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='bicubic'
            )
        return laterals[0]  # B C T H W

    def forward(self, x):
        B, C, T, H, W = x.shape
        # padding for x
        if self.dataset_flag == "slovenia":  # no padding for slovenia
            desired_T = 16
        elif self.dataset_flag == "brandenburg":  # brandenburg 41->48 for dim T
            desired_T = 48
        elif self.dataset_flag == "pastisr":  # pastisr 65/70/71->80 for dim T
            desired_T = 80

        padding = (
            0,
            0,
            0,
            0,
            (desired_T - T) // 2,
            desired_T - T - (desired_T - T) // 2,
        )
        x = F.pad(x, padding, "replicate")

        encoder_feats = self.forward_features(x)
        pfn_fea = self.forward_features_up(encoder_feats)  # B C H W
        logits = self.seg_head(self.up_x4(pfn_fea))  # B C H W

        return logits



if __name__ == "__main__":
    import torch
    import time
    # Prepare input tensor and network
    x = torch.randn((1, 2, 12, 224, 224)).cuda()
    net = GraphTransformer(
        dataset_flag="slovenia",
        embed_dims=[120, 240, 360, 480],
    ).cuda()
    out = net(x)
    print(out.shape)