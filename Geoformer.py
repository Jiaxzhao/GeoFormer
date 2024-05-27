import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.ChangeFormerBaseNetworks import *
from models.CCA import CC_module
# Feature differencing and pass it through Transformer decoder

class Geoformer(nn.Module):

    def __init__(self,input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=96):
        super(Geoformer, self).__init__()


        self.embed_dims = [96, 192, 384, 768]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.
        self.attn_drop = 0.
        self.drop_path_rate = 0.3


        # Transformer Encoder
        self.Tenc0 = SwinTransformer0(pretrain_img_size=256,patch_size=4,in_chans=input_nc,window_size=7)


        self.TDec_x2 = DecoderTransformer_v3(input_transform='multiple_select', in_index=[0, 1, 2, 3],
                                             align_corners=False,
                                             in_channels=self.embed_dims, embedding_dim=self.embedding_dim,
                                             output_nc=output_nc,
                                             decoder_softmax=decoder_softmax, feature_strides=[2, 4, 8, 16])

    def forward(self, x1, x2):
        [fx1, fx2] = [self.Tenc0(x1), self.Tenc0(x2)]
        cp = self.TDec_x2(fx1, fx2)

        return cp


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    # 将layer分成若干个windows，然后在每个windows内attention计算

    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

#将若干个windows合并为一个layer。

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.输入维度的数量
        num_heads (int): Number of attention heads.应用注意力头的数量
        window_size (int): Window size.窗口的尺寸
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels输入通道的数量.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # 通过一个线性层将4C降为2C
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # 将xreshape为维度：(B, H, W, C)
        x = x.view(B, H, W, C)

        # padding判断输入图像的长宽是否可以被二整除，因为我们是通过2倍来进行下采样的
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        # 切片操作，通过切片操作将将相邻的2*2的patch进行拼接
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        # 将合并好的patch通过c维进行拼接
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        # 将x的维度重置为B H/2*W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        # 通过一个线性层进通道降维
        x = self.reduction(x)

        return x
class Conv2drd(nn.Module):
    def __init__(self, pdc, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1,
                 bias=False):
        super(Conv2drd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.pdc = pdc

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return self.pdc(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def RPDC(x, weights, bias=None, stride=1, padding=1, dilation=1, groups=1):
    assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
    assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
    assert padding == dilation, 'padding for ad_conv set wrong'

    shape = weights.shape
    weights = weights.view(shape[0], shape[1], -1)
    weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)  # clock-wise
    y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    return y

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
        self.Conv2drd11=Conv2drd(RPDC,in_channels=192,out_channels=192,kernel_size=3)
        self.Conv2drd22=Conv2drd(RPDC,in_channels=384,out_channels=384,kernel_size=3)
        self.Conv2drd33=Conv2drd(RPDC,in_channels=768,out_channels=768,kernel_size=3)
        # self.SE1=SE(inp=192)
        # self.SE2=SE(inp=384)
        # self.SE3=SE(inp=768)
        self.conv_down1=conv_down(in_channels=192*2,out_channels=192)
        self.conv_down2=conv_down(in_channels=384*2,out_channels=384)
        self.conv_down3=conv_down(in_channels=768*2,out_channels=768)

    def forward(self, x, H, W, x_cov):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))


        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            if x_cov.shape[0]==1:
                x_down = self.downsample(x, H, W)
                Wh, Ww = (H + 1) // 2, (W + 1) // 2
                return x, H, W, x_down, Wh, Ww
            else:
                x1 = x.permute(0, 2, 1)
                x1= x1.reshape(x1.shape[0],x1.shape[1],int((x1.shape[2])** 0.5),int((x1.shape[2])** 0.5))
                x_cov=x1*x_cov
                if x_cov.shape[1]==192:
                    x1=self.conv_down1(torch.cat((x1,x_cov),dim=1))
                    x1=self.Conv2drd11(x1)+x1
                if x_cov.shape[1]==384:
                    x1=self.conv_down2(torch.cat((x1,x_cov),dim=1))
                    x1=self.Conv2drd22(x1)+x1
                if x_cov.shape[1]==768:
                    x1=self.conv_down3(torch.cat((x1,x_cov),dim=1))
                    x1=self.Conv2drd33(x1)+x1
                x1=x1.flatten(2).transpose(1, 2)
                x_down = self.downsample(x1, H, W)
                Wh, Ww = (H + 1) // 2, (W + 1) // 2
                return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W

def conv_down(in_channels,out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                  padding=0, bias=False))

def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )



class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

        self.Conv2drd00= Conv2drd(RPDC, in_channels=inp, out_channels=inp,
                                   kernel_size=3)

    def forward(self, x):
        if self.downsample:
            x1 = self.Conv2drd00(x)
            x=x+x1
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            x1 = self.Conv2drd00(x)
            x = x + x1
            return x + self.conv(x)
def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )
    layers = nn.ModuleList([])
    for i in range(depth):
        if i == 0:
            layers.append(block(inp, oup, image_size, downsample=True))
        else:
            layers.append(block(oup, oup, image_size))
    return nn.Sequential(*layers)
class SwinTransformer0(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=256,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 num_blocks=[2,2,2,2,2],
                 channels_head=[64, 96],
                 block_types=['C']
                 ):
        super().__init__()


        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.in_channels = [96, 192, 384, 768]


        block = {'C': MBConv}
        self.s0 = self._make_layer(
            conv_3x3_bn, in_chans, channels_head[0], num_blocks[0], (256 // 2, 256 // 2))
        self.s1 = self._make_layer(
            block[block_types[0]], channels_head[0], channels_head[1], num_blocks[1], (256 // 4, 256 // 4))



        self.MBConv12 = self._make_layer(block[block_types[0]],  self.in_channels[0], self.in_channels[1], num_blocks[2], (256 // 8, 256 // 8))
        self.MBConv23 = self._make_layer(block[block_types[0]], self.in_channels[1], self.in_channels[2], num_blocks[3],
                                         (256 // 16, 256 // 16))
        self.MBConv34 = self._make_layer(block[block_types[0]], self.in_channels[2], self.in_channels[3], num_blocks[4],
                                         (256 // 32, 256 // 32))




        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()
        #self.init_weights()

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        # 模板
        ckpt = torch.load('/home/ipiu301/Downloads/upernet_swin_tiny_patch4_window7_512x512.pth')['state_dict'] # 加载预训练权重
        model_dict = self.state_dict()  # 得到我们模型的参数

        # 判断预训练模型中网络的模块是否修改后的网络中也存在，并且shape相同，如果相同则取出
        pretrained_dict = {k[9:]: v for k, v in ckpt.items() if k[9:] in model_dict}

        assert len(pretrained_dict) !=0

        model_dict.update(pretrained_dict)  # 利用预训练模型的参数，更新模型
        self.load_state_dict(model_dict)

    def forward(self, x):
        """Forward function."""
        # MB
        x = self.s0(x)
        x = self.s1(x)
        # x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)

        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        x22 = []
        x33 = []
        x44 = []
        ones= np.ones(1)
        ones= torch.from_numpy(ones).cuda()
        for i in range(self.num_layers):
            layer = self.layers[i]
            n, hw, d = x.shape
            x1 = x.permute(0, 2, 1).reshape(n, -1, Wh, Ww)
            if x1.shape[1] ==96:
                x_1_2 = self.MBConv12(x1)
                x22.append(x_1_2)
            if x1.shape[1] ==192:
                x_2_3 = self.MBConv23(x1)
                x33.append(x_2_3)
            if x1.shape[1] ==384:
                x_3_4 = self.MBConv34(x1)
                x44.append(x_3_4)

            if x1.shape[1] == 96:
                x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww,ones)
            if x1.shape[1] == 192:
                x22 = x22[0]
                # x = x.flatten(2).transpose(1, 2).permute(0,1,2)
                x_out, H, W, x, Wh, Ww = layer(x,Wh,Ww,x22)
            if x1.shape[1] == 384:
                x33 = x33[0]
                # x = x.flatten(2).transpose(1, 2).permute(0,1,2)
                x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww,x33)
            if x1.shape[1] == 768:
                x44 = x44[0]
                # x = x.flatten(2).transpose(1, 2).permute(0,1,2)
                x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww,x44)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
            outs.append(out)
        return tuple(outs)



def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)
#Difference module
def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )

#Intermediate prediction module
def make_prediction(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    )
class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x)
        return x
class Laplac_CNN(nn.Module):
    def __init__(self):
        super(Laplac_CNN, self).__init__()
        Laplac = np.array([[[0.25, 0, 0.25],
                            [0, -1, 0],
                            [0.25, 0, 0.25]]])

        Laplac=torch.tensor(Laplac.repeat(256,0),dtype=torch.float32).unsqueeze(dim=1)
        self.weight = nn.Parameter(Laplac)  # 自定义的权值

    def forward(self, x):
        out = F.conv2d(x,self.weight,stride=1, padding=1, groups=256)
        return out


class DecoderTransformer_v3(nn.Module):
    """
    Transformer Decoder
    """

    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True,
                 in_channels=[96, 192, 384, 768], embedding_dim=64, output_nc=2,
                 decoder_softmax=False, feature_strides=[2, 4, 8, 16]):
        super(DecoderTransformer_v3, self).__init__()
        # assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]

        # settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # MLP decoder heads
        # self.head_pool=nn.AdaptiveAvgPool2d((128, 128))
        self.Lap = Laplac_CNN()

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)
        # Mluti CCA

        self.CCA = CC_module(self.embedding_dim)

        # convolutional Difference Modules
        self.diff_c4 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)

        # taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        # Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=self.embedding_dim * len(in_channels), out_channels=self.embedding_dim,
                      kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        self.conv_down4=conv_down(in_channels=int(self.embedding_dim*2),out_channels=self.embedding_dim)

        # Final predction head
        self.convd2x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.convd1x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)

        # Final activation
        self.output_softmax = decoder_softmax
        self.active = nn.Sigmoid()

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':

            inputs = [inputs[i] for i in self.in_index]
            # inputs(list4):0:(1,64,64,64),1:(1,128,32,32),2:(1,320,16,16),3:(1,512,8,8)
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2):
        # Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        # img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []

        # Stage 4: x1/32 scale

        # 8，64，64
        # c4_1：8，512，8，8
        _c4_1 = self.linear_c4(c4_1).permute(0, 2, 1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0, 2, 1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4_2, _c4_1 = self.CCA(_c4_2, _c4_1)
        head42 = self.Lap(_c4_2) +_c4_2
        # head41 = self.Lap(_c4_1) +_c4_1
        c44 = self.conv_down4(torch.cat((_c4_2, _c4_1), dim=1))
        _c4 = self.diff_c4(torch.cat((c44,head42), dim=1))
        p_c4 = self.make_pred_c4(_c4)
        # 1,2,8,8
        outputs.append(p_c4)
        # _c4_up = resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)
        # 1,256,64,64

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0, 2, 1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        # 8，64，16，16
        _c3_2 = self.linear_c3(c3_2).permute(0, 2, 1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3_2, _c3_1 = self.CCA(_c3_2, _c3_1)
        c33 = self.conv_down4(torch.cat((_c3_2, _c3_1), dim=1))
        head32 = self.Lap(_c3_2) + _c3_2
        # head31 = self.Lap(_c3_1) + _c3_1
        # head3 = conv_down(torch.cat((head31, head32), dim=1), dim=256)
        _c3 = self.diff_c3(torch.cat((c33,head32), dim=1)) + F.interpolate(_c4, scale_factor=2, mode="bilinear")
        _c43=_c3+ F.interpolate(_c4, scale_factor=2, mode="bilinear")
        p_c3 = self.make_pred_c3(_c3)
        # 1,2,16,16
        outputs.append(p_c3)
        # _c3_up = resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)
        # 1,256,64,64

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0, 2, 1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        # 1,256,32,32
        _c2_2 = self.linear_c2(c2_2).permute(0, 2, 1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2_2, _c2_1 = self.CCA(_c2_2, _c2_1)
        head22 = self.Lap(_c2_2) + _c2_2
        # head21 = self.Lap(_c2_1) + _c2_1
        c22 = self.conv_down4(torch.cat((_c2_2, _c2_1), dim=1))
        # c22 = conv_down(torch.cat((_c2_2, _c2_1), dim=1), dim=256)
        _c2 = self.diff_c2(torch.cat((c22,head22), dim=1))+F.interpolate(_c3, scale_factor=2, mode="bilinear")
        _c32 = _c2 + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        _c42 = _c32 + F.interpolate(_c3, scale_factor=2, mode="bilinear") + F.interpolate(_c43, scale_factor=2,mode="bilinear")
        p_c2 = self.make_pred_c2(_c2)
        # 1,2,32,32
        outputs.append(p_c2)
        # 1,256,64,64

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0, 2, 1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        # 1,256,64,64
        _c1_2 = self.linear_c1(c1_2).permute(0, 2, 1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1_2, _c1_1 = self.CCA(_c1_2, _c1_1)
        # c11 = conv_down(torch.cat((_c1_2, _c1_1), dim=1), dim=256)
        c11 = self.conv_down4(torch.cat((_c1_2, _c1_1), dim=1))

        head12 = self.Lap(_c1_2) + _c1_2
        _c1 = self.diff_c1(torch.cat((c11,head12), dim=1)) + F.interpolate(_c2, scale_factor=2, mode="bilinear")
        # _c21
        _c21 = _c1 + F.interpolate(_c2, scale_factor=2, mode="bilinear")
        # _c31
        _c31 = _c1 + _c21 + F.interpolate(_c32, scale_factor=2, mode="bilinear")
        # _c41
        _c41 = _c1 + _c31 + _c21 + F.interpolate(_c42, scale_factor=2, mode="bilinear")
        # 1,256,64,64
        p_c1 = self.make_pred_c1(_c1)
        # 1,2,64,64
        outputs.append(p_c1)

        # Linear Fusion of difference image from all scales
        _c = self.linear_fuse(torch.cat((_c41, _c31, _c21, _c1), dim=1))
        # 1,256,64,64



        # Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)
        #1,256,128,128
        x = self.dense_2x(x)
        # 1,256,128,128
        # Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        # 1,256,256,256
        # Residual block
        x = self.dense_1x(x)
        # 1,256,256,256

        # Final prediction
        cp = self.change_probability(x)

        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs
