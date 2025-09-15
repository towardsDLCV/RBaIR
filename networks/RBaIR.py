import numbers
from einops import rearrange
import einops

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.MoCE import *
from networks.arch_utils import *


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

##########################################################################
## Cross Attention
class Channel_Cross_Attention(nn.Module):
    def __init__(self, dim, num_head, bias):
        super(Channel_Cross_Attention, self).__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1), requires_grad=True)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        assert x.shape == y.shape, 'The shape of feature maps from image and features are not equal!'

        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_head)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)
        out = self.project_out(out)
        return out


class DeformAttn(nn.Module):

    def __init__(self, dim, n_heads=1, n_groups=1, stride=1, ksize=3, offset_range_factor=4):
        super(DeformAttn, self).__init__()
        self.dim = dim
        self.offset_range_factor = offset_range_factor
        self.n_head_channels = dim // n_heads
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.n_groups = n_groups

        self.n_group_channels = self.dim // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.ksize = ksize
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0
        self.conv_offset = nn.Sequential(nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
                                         LayerNorm(self.n_group_channels, 'WithBias'),
                                         nn.GELU(),
                                         nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False))
        self.proj_q = nn.Conv2d(dim, self.dim,
                                kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(dim, self.dim,
                                kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(dim, self.dim,
                                kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(self.dim, dim,
                                  kernel_size=1, stride=1, padding=0)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2
        # ref = ref[None, ...].expand(B, -1, -1, -1) # B * g H W 2
        return ref

    def forward(self, prompt, kv):
        B, C, Hp, Wp = kv.size()
        B, C, H, W = prompt.size()
        dtype, device = kv.dtype, kv.device
        q = self.proj_q(prompt)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg

        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
        offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        pos = offset + reference
        x_sampled = F.grid_sample(input=kv.reshape(B * self.n_groups, self.n_group_channels, Hp, Wp),
                                  grid=pos[..., (1, 0)],  # y, x -> x, y
                                  mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)
        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, H, W)
        return self.proj_out(out)


class DeformAttnBlock(nn.Module):

    def __init__(self, dim, n_heads=1,
                 n_groups=1,
                 stride=1, ksize=3, offset_range_factor=4):
        super(DeformAttnBlock, self).__init__()
        self.norm11 = LayerNorm(dim, 'WithBias')
        self.norm12 = LayerNorm(dim, 'WithBias')
        self.attn = DeformAttn(dim=dim,
                               n_heads=n_heads, n_groups=n_groups, stride=stride,
                               ksize=ksize, offset_range_factor=offset_range_factor)

    def forward(self, x, kv, plot=False):
        x = x + self.attn(self.norm11(x), self.norm12(kv), plot=plot)
        return x


class CrossAttnBlock(nn.Module):

    def __init__(self, dim, num_head):
        super(CrossAttnBlock, self).__init__()
        self.norm11 = LayerNorm(dim, 'WithBias')
        self.norm12 = LayerNorm(dim, 'WithBias')
        self.attn = Channel_Cross_Attention(dim=dim, num_head=num_head, bias=False)

    def forward(self, x, kv):
        x = x + self.attn(self.norm11(x), self.norm12(kv))
        return x


class rbf(nn.Module):

    def __init__(self,
                 min_v,
                 max_v,
                 num_k):
        super(rbf, self).__init__()
        self.min_v = min_v
        self.max_v = max_v
        self.num_k = num_k
        center = torch.linspace(min_v, max_v, num_k)
        self.center = nn.Parameter(center, requires_grad=False)
        self.denominator = (max_v - min_v) / (num_k - 1)

    def gaussian_basis_func(self, x):
        return torch.exp(-((x[..., None] - self.center) / self.denominator) ** 2)

    def forward(self, x):
        return self.gaussian_basis_func(x)


class DynamicRBFA(nn.Module):

    def __init__(self, dim=48*4, num_head=4, hidden_dim=128, max_value=2., min_value=-2., num_k=8,
                 n_groups=2, offset_range_factor=1, ksize=3, stride=1):
        super(DynamicRBFA, self).__init__()
        self.max_value = max_value
        self.min_value = min_value
        self.num_k = num_k
        self.hidden_dim = hidden_dim

        self.conv_x = nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.conv_y = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, bias=False)

        self.DaCA = DeformAttnBlock(dim=hidden_dim, n_heads=num_head,
                                    n_groups=n_groups, offset_range_factor=offset_range_factor, ksize=ksize,
                                    stride=stride)

        rbf_dim = hidden_dim // 2
        self.rbf_dim = rbf_dim
        self.transformation = nn.Linear(hidden_dim, rbf_dim)
        self.norm_rbf = nn.LayerNorm(rbf_dim)
        self.rbf = rbf(min_v=min_value, max_v=max_value, num_k=num_k)
        self.dynamic_W1 = nn.Linear(hidden_dim, num_k * rbf_dim)
        self.dynamic_W2 = nn.Linear(rbf_dim, hidden_dim)

        self.compression = nn.Linear(hidden_dim, hidden_dim // 2)
        self.act = nn.GELU()
        self.rbf_linear, self.emb_linear = nn.Linear(hidden_dim // 2, hidden_dim), nn.Linear(hidden_dim // 2, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

        self.conv_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.cross_attn = CrossAttnBlock(dim=dim, num_head=num_head)

    def plot_curve(
            self,
            w,
            input_index: int,
            output_index: int,
            num_pts: int = 1000,
            num_extrapolate_bins: int = 2):
        ng = self.rbf.num_k
        h = self.rbf.denominator
        assert input_index < self.hidden_dim
        assert output_index < self.hidden_dim
        w = w[0, output_index, input_index * ng: (input_index + 1) * ng]  # num_grids,

        x = torch.linspace(
            self.rbf.min_v - num_extrapolate_bins * h,
            self.rbf.max_v + num_extrapolate_bins * h,
            num_pts, device=w.device)  # num_pts, num_grids
        with torch.no_grad():
            y = (w * self.rbf(x.to(w.dtype))).sum(-1)
        return x, y

    def no_weight_plot_curve(
            self,
            w,
            input_index: int,
            output_index: int,
            num_pts: int = 1000,
            num_extrapolate_bins: int = 2):
        ng = self.rbf.num_k
        h = self.rbf.denominator
        assert input_index < self.hidden_dim
        assert output_index < self.hidden_dim
        w = w[0, output_index, input_index * ng: (input_index + 1) * ng]  # num_grids,

        x = torch.linspace(
            self.rbf.min_v - num_extrapolate_bins * h,
            self.rbf.max_v + num_extrapolate_bins * h,
            num_pts, device=w.device)  # num_pts, num_grids
        with torch.no_grad():
            y = (self.rbf(x.to(w.dtype))).sum(-1)
        return x, y

    def forward(self, x, y):
        B, C, Hp, Wp = y.shape
        u = y.clone()
        x = self.conv_x(x)
        y = self.conv_y(y)

        deg_feat = self.DCCA(y, x)

        emb = y.mean(dim=(-2, -1))
        deg_emb = deg_feat.mean(dim=(-2, -1))

        trans_emb = self.transformation(emb)
        basis = self.rbf(self.norm_rbf(trans_emb)).reshape(B, -1)

        W1 = self.act(self.dynamic_W1(deg_emb))
        W2 = self.act(self.dynamic_W2(trans_emb))
        dynamic_W = W2.unsqueeze(-1) @ W1.unsqueeze(1)

        rbf_out = (basis.unsqueeze(1) @ dynamic_W.transpose(-2, -1)).squeeze(1)
        compression = self.act(self.compression(rbf_out + emb))
        sel_w = F.softmax(torch.stack([self.emb_linear(compression), self.rbf_linear(compression)], dim=1), dim=1)

        out = emb * sel_w[:, 0] + rbf_out * sel_w[:, 1]
        out = self.act(self.out_linear(out)).unsqueeze(-1).unsqueeze(-1) * deg_feat

        out = F.interpolate(self.conv_out(out), size=(Hp, Wp), mode='bilinear')
        out = self.cross_attn(u, out)
        return out


##########################################################################
##---------- AdaIR -----------------------
class RBaIR(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 offset_range_factor=[2, 3, 4],
                 n_groups=[1, 2, 4],
                 ksizes=[7, 5, 3],
                 strides=[4, 2, 1],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 decoder=True):

        super(RBaIR, self).__init__()
        self.dim = dim
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.MoCE_in = nn.Sequential(*[MoCE(dim=dim) for _ in range(1)])
        self.MoCE_out = nn.Sequential(*[MoCE(dim=dim * 2) for _ in range(1)])

        self.decoder = decoder
        self.chnl_reduce3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)

        if self.decoder:
            k = 8
            self.DyRBF1 = DynamicRBFA(dim=dim * 2 ** 2, num_head=heads[2], hidden_dim=256,
                                      offset_range_factor=offset_range_factor[2], ksize=ksizes[2], stride=strides[2], n_groups=n_groups[2],
                                      min_value=-4., max_value=4., num_k=k*4)
            self.DyRBF2 = DynamicRBFA(dim=dim * 2 ** 2, num_head=heads[2], hidden_dim=128,
                                      offset_range_factor=offset_range_factor[1], ksize=ksizes[1], stride=strides[1], n_groups=n_groups[1],
                                      min_value=-3., max_value=3., num_k=k*2)
            self.DyRBF3 = DynamicRBFA(dim=dim * 2 ** 1, num_head=heads[1], hidden_dim=64,
                                      offset_range_factor=offset_range_factor[0], ksize=ksizes[0], stride=strides[0], n_groups=n_groups[0],
                                      min_value=-2., max_value=2., num_k=k)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2

        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3

        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4

        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 2))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**1) + 192, int(dim*2**2), kernel_size=1, bias=bias)

        self.decoder_level3 = nn.ModuleList([
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.ModuleList([
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.ModuleList([
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def replace_inpt(self):
        self.patch_embed = OverlapPatchEmbed(4, self.dim)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level1 = self.MoFE_in(inp_enc_level1)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)

        latent = self.latent(inp_enc_level4)
        latent = self.chnl_reduce3(latent)

        if self.decoder:
            latent = self.DyRBF1(inp_img, latent, False, False)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        if self.decoder:
            out_dec_level3 = self.DyRBF2(inp_img, out_dec_level3, False, False, False, False)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        if self.decoder:
            out_dec_level2 = self.DyRBF3(inp_img, out_dec_level2, False, False, False)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        return out_dec_level1
