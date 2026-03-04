import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
np.set_printoptions(threshold=1000)
from typing import Callable, Optional, Union, Tuple, List

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def heat_pos_embed(height=16, width=16, sigma=0.2): #0.6
    heatmap = np.zeros((1, height*width, height, width))
    factor = 1/(2*sigma*sigma)
    for y in range(height):
        for x in range(width):
            x_vec = ((np.arange(0, width) - x)/width) ** 2
            y_vec = ((np.arange(0, height) - y)/height) ** 2
            xv, yv = np.meshgrid(x_vec, y_vec)
            exponent = factor * (xv + yv)
            exp = np.exp(-exponent)
            heatmap[0, y*height + x, :, :] = exp
    return heatmap

def pos_grid(height=32, width=32):
    grid = np.zeros((1, height * width, height, width))
    for y in range(height):
        for x in range(width):
            x_vec = ((np.arange(0, width) - x) / width) ** 2
            y_vec = ((np.arange(0, height) - y) / height) ** 2
            yv, xv = np.meshgrid(x_vec, y_vec)
            disyx = (yv + xv)
            grid[0, y * width + x, :, :] = disyx
    return grid

def pos_grid_mask(height=32, width=32, thresh=0.25):
    grid = np.zeros((1, height * width, height, width))
    for y in range(height):
        for x in range(width):
            x_vec = ((np.arange(0, width) - x) / width) ** 2
            y_vec = ((np.arange(0, height) - y) / height) ** 2
            yv, xv = np.meshgrid(x_vec, y_vec)
            disyx = (yv + xv)
            disyx[disyx > thresh] = -1
            disyx[disyx >= 0] = 1
            disyx[disyx == -1] = 0
            grid[0, y * width + x, :, :] = disyx
    return grid

def relative_pos_index(height=32, weight=32):
    coords_h = torch.arange(height)
    coords_w = torch.arange(weight)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww # 0 is 32 * 32 for h, 1 is 32 * 32 for w
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += height - 1  # shift to start from 0 # the relative pos in y axial
    relative_coords[:, :, 1] += weight - 1  # the relative pos in x axial
    relative_coords[:, :, 0] *= 2*weight - 1 # the 1d pooling pos to recoard the pos
    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    return relative_position_index

def relative_pos_index_dis(height=32, weight=32):
    coords_h = torch.arange(height)
    coords_w = torch.arange(weight)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww # 0 is 32 * 32 for h, 1 is 32 * 32 for w
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    dis = (relative_coords[:, :, 0].float()/height) ** 2 + (relative_coords[:, :, 1].float()/weight) ** 2
    relative_coords[:, :, 0] += height - 1  # shift to start from 0 # the relative pos in y axial
    relative_coords[:, :, 1] += weight - 1  # the relative pos in x axial
    relative_coords[:, :, 0] *= 2*weight - 1 # the 1d pooling pos to recoard the pos
    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    return relative_position_index, dis

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNorm2pm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, prob, **kwargs):
        return self.fn(self.norm(x), prob, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), torch.cat((q, k, v), dim=-1), attn #attn

class Attention_RPEHP(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., height=16, width=16):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        relative_position_index, dis = relative_pos_index_dis(height, width)
        self.relative_position_index = relative_position_index # hw hw
        self.dis = dis.cuda()  # hw hw
        self.headsita = nn.Parameter(torch.zeros(heads) + torch.range(1, heads)*0.1, requires_grad=True)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * height - 1) * (2 * width - 1), heads), requires_grad=True)
        self.height = height
        self.weight = width

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots0 = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.height * self.weight, self.height * self.weight, -1)  # n n h
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        factor = 1 / (2 * self.headsita**2 + 1e-10)  # g
        exponent = factor[:, None, None] * self.dis[None, :, :] # g hw hw
        pos_embed = torch.exp(-exponent)  # g hw hw
        dots = dots0 + relative_position_bias.unsqueeze(0) + 0.01*pos_embed[None, :, :, :]

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), torch.cat((q, k, v), dim=-1), self.attend(dots0)
        #return self.to_out(out), self.attend(dots0)

def window_reverse(windows, window_size: List[int], img_size: List[int]):
    windows = windows.permute(0, 2, 3, 1)
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x.permute(0,3,1,2)

def grid_reverse(windows, grid_size: List[int], img_size: List[int]):
    windows = windows.permute(0, 2, 3, 1)
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, H, W, C)
    return x.permute(0, 3, 1, 2)

def window_partition(x, window_size: List[int]):
    x = x.permute(0, 2, 3, 1)
    B, H, W, C = x.shape
    assert H % window_size[0] == 0, f'height ({H}) must be divisible by window ({window_size[0]})'
    assert W % window_size[1] == 0, f'width ({W}) must be divisible by window ({window_size[1]})'
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows.permute(0, 3, 1, 2)

def grid_partition(x, grid_size: List[int]):
    x = x.permute(0, 2, 3, 1)
    B, H, W, C = x.shape
    assert H % grid_size[0] == 0, f'height {H} must be divisible by grid {grid_size[0]}'
    assert W % grid_size[1] == 0, f'width {W} must be divisible by grid {grid_size[1]}'
    x = x.view(B, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1], C)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return windows.permute(0, 3, 1, 2)

class AttentionPruneKV(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., num_patches=1024, use_mqa=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        height, width = int(np.sqrt(num_patches)), int(np.sqrt(num_patches))

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.use_mqa = use_mqa

        self.attend = nn.Softmax(dim=-1)
        if self.use_mqa:
            self.to_q = nn.Linear(dim, inner_dim, bias=False)
            self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)
        else:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # ------------------ relative position embedding -------------------
        relative_position_index, dis = relative_pos_index_dis(height, width)
        self.relative_position_index = relative_position_index  # hw hw
        self.dis = dis.cuda()  # hw hw
        self.headsita = nn.Parameter(torch.zeros(heads) + torch.arange(1, heads + 1) * 0.1, requires_grad=True)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * height - 1) * (2 * width - 1), heads),
                                                         requires_grad=True)
        self.height = height
        self.weight = width

        # -----------------add function-----------------
        self.gate = nn.Parameter(torch.tensor(-2.0), requires_grad=False)  #
        self.neg_thresh = 0.9
        self.thresh_for_kv = nn.Linear(dim_head, 1, bias=False)
        self.sig = nn.Sigmoid()

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.partition_size = (7,7)

    def forward(self, x, prob, rpe=True):
        if self.use_mqa:
            q = self.to_q(x)
            k, v = self.to_kv(x).chunk(2, dim=-1)
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
            k = rearrange(k, 'b n d -> b 1 n d') # Broadcast over heads
            v = rearrange(v, 'b n d -> b 1 n d')
        else:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
            
        dots0 = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.height * self.weight, self.height * self.weight, -1)  # n n h
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            factor = 1 / (2 * self.headsita ** 2 + 1e-6)  # g
            exponent = factor[:, None, None] * self.dis[None, :, :]  # g hw hw
            pos_embed = torch.exp(-exponent)  # g hw hw
            dots = dots0 + relative_position_bias.unsqueeze(0) + 0.01 * pos_embed[None, :, :, :]
            # dots = dots0 + relative_position_bias.unsqueeze(0)
        else:
            dots = dots0

        attn = self.attend(dots)  # b g n n
        # '''
        b, g, n, _ = attn.shape
        attn_max = torch.max(attn, dim=-1)[0]  # b g n
        attn_min = torch.min(attn, dim=-1)[0]  # b g n
        # q = rearrange(q, 'b g n d -> b n g d')
        # q[prob >= self.neg_thresh, :, :] = 0
        # q = rearrange(q, 'b n g d -> (b g) n d')
        q = rearrange(q, 'b g n d -> (b g) n d')
        thresh = self.sig(self.thresh_for_kv(q)) * self.sig(self.gate)  # bg n 1
        thresh = rearrange(thresh, '(b g) n d -> b g (n d)', b=b)
        thresh = attn_min + thresh * (attn_max - attn_min)

        record = attn - thresh[:, :, :, None]  # b g n n
        record[record > 0] = 1
        record[record <= 0] = 0
        prob = prob[:, None, :].repeat(1, g, 1)
        record[prob >= self.neg_thresh, :] = 0

        deno = torch.einsum('bcik,bcik->bci', [attn, record])
        attn = torch.mul(attn, record) / (deno[:, :, :, None] + 1e-6)
        # '''
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), self.attend(dots0)


class AttentionPruneKV_inference(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., num_patches=1024):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        height, width = int(np.sqrt(num_patches)), int(np.sqrt(num_patches))

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)

        # ------------------ relative position embedding -------------------
        relative_position_index, dis = relative_pos_index_dis(height, width)
        self.relative_position_index = relative_position_index  # hw hw
        self.dis = dis.cuda()  # hw hw
        self.headsita = nn.Parameter(torch.zeros(heads) + torch.range(1, heads) * 0.1, requires_grad=True)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * height - 1) * (2 * width - 1), heads),
                                                         requires_grad=True)
        self.height = height
        self.weight = width
        self.dim_head = dim_head

        # -----------------add function-----------------
        self.gate = nn.Parameter(torch.tensor(-2.0), requires_grad=False)  #
        self.neg_thresh = 0.9
        self.thresh_for_kv = nn.Linear(dim_head, 1, bias=False)
        self.sig = nn.Sigmoid()

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, prob, rpe=True):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        b, n, _ = x.shape

        q = rearrange(q, 'b g n d -> b n g d')
        q = q[:, prob[0, :] < self.neg_thresh, :, :]
        q = rearrange(q, 'b n g d -> (b g) n d')

        dots0 = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.height * self.weight, self.height * self.weight, -1)  # n n h
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            relative_position_bias = relative_position_bias[:, prob[0, :] < self.neg_thresh, :]
            factor = 1 / (2 * self.headsita ** 2 + 1e-6)  # g
            exponent = factor[:, None, None] * self.dis[None, :, :]  # g hw hw
            exponent = exponent[:, prob[0, :] < self.neg_thresh, :]
            pos_embed = torch.exp(-exponent)  # g hw hw
            dots = dots0 + relative_position_bias.unsqueeze(0) + 0.01*pos_embed[None, :, :, :]
        else:
            dots = dots0

        attn = self.attend(dots)  # b g n n
        if q.shape[1] > 0:
            #'''
            attn_max = torch.max(attn, dim=-1)[0]  # b g n
            attn_min = torch.min(attn, dim=-1)[0]  # b g n

            thresh = self.sig(self.thresh_for_kv(q)) * self.sig(self.gate)  # bg n 1
            thresh = rearrange(thresh, '(b g) n d -> b g (n d)', b=b)
            thresh = attn_min + thresh * (attn_max - attn_min)

            record = attn - thresh[:, :, :, None]  # b g n n
            record[record > 0] = 1
            record[record <= 0] = 0
            #print(torch.sum(record) / (b * g * n * n + 0.0000001))

            deno = torch.einsum('bcik,bcik->bci', [attn, record])
            attn = torch.mul(attn, record) / (deno[:, :, :, None] + 1e-6)
        #'''
        out0 = torch.matmul(attn, v)
        out0 = rearrange(out0, 'b h n d -> b n (h d)')
        out0 = self.to_out(out0)
        out = x * 0
        out[:, prob[0, :] < self.neg_thresh, :] = out0

        return out, self.attend(dots0)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        qkvs, attns = [], []
        for attn, ff in self.layers:
            ax, qkv, attn = attn(x)
            qkvs.append(qkv)
            attns.append(attn)
            x = ax + x
            x = ff(x) + x
        return x, qkvs, attns

class Transformer_HP(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., height=16, width=16):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention_RPEHP(dim, heads=heads, dim_head=dim_head, dropout=dropout, height=height, width=width)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        qkvs, attns = [], []
        for attn, ff in self.layers:
            ax, qkv, attn = attn(x)
            qkvs.append(qkv)
            attns.append(attn)
            x = ax + x
            x = ff(x) + x
        return x, qkvs, attns

class TransformerSPrune(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128, use_mqa=False, share_layers=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.share_layers = share_layers
        self.depth = depth
        
        # If share_layers is True, we only create 1 layer and reuse it
        num_layers_to_create = 1 if share_layers else depth
        
        for _ in range(num_layers_to_create):
            self.layers.append(nn.ModuleList([
                PreNorm2pm(dim, AttentionPruneKV(dim, heads=heads, dim_head=dim_head, dropout=dropout, num_patches=num_patches, use_mqa=use_mqa)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
            
    def forward(self, x, prob):
        attns = []
        for i in range(self.depth):
            if self.share_layers:
                attn, ff = self.layers[0]
            else:
                attn, ff = self.layers[i]
                
            ax, attn = attn(x, prob)
            attns.append(attn)
            x = ax + x
            x = ff(x) + x
        return x, attns

class TransformerSPrune_Test(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm2pm(dim, AttentionPruneKV_inference(dim, heads=heads, dim_head=dim_head, dropout=dropout, num_patches=num_patches)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x, prob):
        attns = []
        for attn, ff in self.layers:
            ax, attn = attn(x, prob)
            attns.append(attn)
            x = ax + x
            x = ff(x) + x
        return x, attns

class TransformerDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, image_size, depth=2, dmodel=1024, mlp_dim=2048, patch_size=2, heads=6, dim_head=128, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = in_channels * patch_height * patch_width
        self.dmodel = out_channels
        self.mlp_dim = self.dmodel*4

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.patch_dim, self.dmodel),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout, num_patches)

        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height//patch_height),
        )


    def forward(self, x):
        x = self.to_patch_embedding(x)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        
        x = self.dropout(x)
        # transformer layer
        ax, qkvs, attns = self.transformer(x)
        out = self.recover_patch_embedding(ax)
        return out, qkvs, attns

class TransformerDown_HP(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, image_size, depth=2, dmodel=1024, mlp_dim=2048, patch_size=2, heads=6, dim_head=128, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = in_channels * patch_height * patch_width
        self.dmodel = out_channels
        self.mlp_dim = self.dmodel * 4

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.patch_dim, self.dmodel),
        )

        #self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer_HP(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout, image_height//patch_height, image_width//patch_width)

        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height//patch_height),
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        b, n, _ = x.shape
        #x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        # transformer layer
        ax, qkvs, attns = self.transformer(x)
        out = self.recover_patch_embedding(ax)
        return out, qkvs, attns

class PrunedPartitionAttention(nn.Module):
    def __init__(self, in_channels, out_channels, partition_type, depth, heads, dim_head, mlp_dim, dropout, emb_dropout, partition_size=7, patch_size=1, 
                 transformer=None, pred_class=None, to_patch_embedding=None, use_mqa=False, share_layers=True):
        super().__init__()
        self.partition_type = partition_type
        self.partition_size = (partition_size, partition_size)
        patch_height, patch_width = pair(patch_size)
        self.patch_dim = in_channels * patch_height * patch_width
        self.dmodel = out_channels
        self.mlp_dim = mlp_dim
        
        # Num patches per window/grid
        self.num_patches = (partition_size // patch_height) * (partition_size // patch_width)
        
        if to_patch_embedding is not None:
            self.to_patch_embedding = to_patch_embedding
        else:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
                nn.Linear(self.patch_dim, self.dmodel),
            )
        
        self.dropout = nn.Dropout(emb_dropout)
        
        # Use TransformerSPrune
        if transformer is not None:
            self.transformer = transformer
        else:
            self.transformer = TransformerSPrune(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout, self.num_patches, use_mqa=use_mqa, share_layers=share_layers)
        
        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=partition_size // patch_height),
        )
        
        if pred_class is not None:
            self.pred_class = pred_class
        else:
            self.pred_class = nn.Conv2d(in_channels, 2, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        ima_size = x.shape[2:4]
        
        if self.partition_type == 'window':
            partitioned = window_partition(x, self.partition_size)
        else:
            partitioned = grid_partition(x, self.partition_size)
        
        sx = self.pred_class(partitioned)
        sxp = self.softmax(sx)
        sxp_neg_1d = rearrange(sxp[:, 0, :, :], 'b h w -> b (h w)')
        
        x_emb = self.to_patch_embedding(partitioned)
        x_emb = self.dropout(x_emb)
        
        ax, attns = self.transformer(x_emb, sxp_neg_1d)
        
        x_rec = self.recover_patch_embedding(ax)
        
        if self.partition_type == 'window':
            out = window_reverse(x_rec, self.partition_size, ima_size)
            sx_full = window_reverse(sx, self.partition_size, ima_size)
        else:
            out = grid_reverse(x_rec, self.partition_size, ima_size)
            sx_full = grid_reverse(sx, self.partition_size, ima_size)
            
        return out, sx_full, attns

class TransformerDown_SPrune(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, image_size, depth=2, dmodel=1024, mlp_dim=2048, patch_size=2, heads=6,
                 dim_head=128, dropout=0.1, emb_dropout=0.1, partition_size=7, share_weights=True, mlp_ratio=4, use_mqa=False, share_layers=True):
        super().__init__()
        
        if mlp_ratio is not None:
             mlp_dim = dmodel * mlp_ratio

        # Components for sharing
        shared_transformer = None
        shared_pred_class = None
        shared_embedding = None
        
        if share_weights:
            patch_height, patch_width = pair(patch_size)
            patch_dim = in_channels * patch_height * patch_width # Assuming in_channels is same for both or handled
            num_patches = (partition_size // patch_height) * (partition_size // patch_width)
            
            shared_transformer = TransformerSPrune(out_channels, depth, heads, dim_head, mlp_dim, dropout, num_patches, use_mqa=use_mqa, share_layers=share_layers)
            
            if in_channels == out_channels:
                 shared_pred_class = nn.Conv2d(in_channels, 2, kernel_size=1)
                 shared_embedding = nn.Sequential(
                    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
                    nn.Linear(patch_dim, out_channels),
                )

        self.attn_block = PrunedPartitionAttention(
            in_channels, out_channels, 'window', depth, heads, dim_head, mlp_dim, dropout, emb_dropout, partition_size, patch_size,
            transformer=shared_transformer,
            pred_class=shared_pred_class,
            to_patch_embedding=shared_embedding,
            use_mqa=use_mqa,
            share_layers=share_layers
        )

        grid_pred_class = shared_pred_class
        grid_embedding = shared_embedding

        self.attn_grid = PrunedPartitionAttention(
            out_channels, out_channels, 'grid', depth, heads, dim_head, mlp_dim, dropout, emb_dropout, partition_size, patch_size,
            transformer=shared_transformer,
            pred_class=grid_pred_class, # If None, it creates new
            to_patch_embedding=grid_embedding, # If None, it creates new
            use_mqa=use_mqa,
            share_layers=share_layers
        )

    def forward(self, x):
        x, sx1, attns1 = self.attn_block(x)
        x, sx2, attns2 = self.attn_grid(x)
        return x, sx1+sx2, attns1+attns2

class StandardPartitionAttention(nn.Module):
    """Standard Partition Attention (Window/Grid) without Pruning"""
    def __init__(self, in_channels, out_channels, partition_type, depth, heads, dim_head, mlp_dim, dropout, emb_dropout, partition_size=7, patch_size=1, 
                 transformer=None, to_patch_embedding=None):
        super().__init__()
        self.partition_type = partition_type
        self.partition_size = (partition_size, partition_size)
        patch_height, patch_width = pair(patch_size)
        self.patch_dim = in_channels * patch_height * patch_width
        self.dmodel = out_channels
        self.mlp_dim = mlp_dim
        
        if to_patch_embedding is not None:
            self.to_patch_embedding = to_patch_embedding
        else:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
                nn.Linear(self.patch_dim, self.dmodel),
            )
        
        self.dropout = nn.Dropout(emb_dropout)
        
        if transformer is not None:
            self.transformer = transformer
        else:
            # Standard Transformer with Relative Position Embedding (RPE)
            self.transformer = Transformer_HP(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout, partition_size, partition_size)
        
        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=partition_size // patch_height),
        )

    def forward(self, x):
        ima_size = x.shape[2:4]
        
        if self.partition_type == 'window':
            partitioned = window_partition(x, self.partition_size)
        else:
            partitioned = grid_partition(x, self.partition_size)
        
        x_emb = self.to_patch_embedding(partitioned)
        x_emb = self.dropout(x_emb)
        
        ax, qkvs, attns = self.transformer(x_emb)
        
        x_rec = self.recover_patch_embedding(ax)
        
        if self.partition_type == 'window':
            out = window_reverse(x_rec, self.partition_size, ima_size)
        else:
            out = grid_reverse(x_rec, self.partition_size, ima_size)
            
        return out, attns

class TransformerDown_Standard(nn.Module):
    """Comparative module using standard attention but with Window-Grid structure"""
    def __init__(self, in_channels, out_channels, image_size, depth=2, dmodel=1024, mlp_dim=2048, patch_size=2, heads=6,
                 dim_head=128, dropout=0.1, emb_dropout=0.1, partition_size=7, share_weights=True, mlp_ratio=4):
        super().__init__()
        
        if mlp_ratio is not None:
             mlp_dim = dmodel * mlp_ratio

        shared_transformer = None
        shared_embedding = None
        
        if share_weights:
            patch_height, patch_width = pair(patch_size)
            patch_dim = in_channels * patch_height * patch_width
            shared_transformer = Transformer_HP(out_channels, depth, heads, dim_head, mlp_dim, dropout, partition_size, partition_size)
            if in_channels == out_channels:
                 shared_embedding = nn.Sequential(
                    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
                    nn.Linear(patch_dim, out_channels),
                )

        self.attn_block = StandardPartitionAttention(
            in_channels, out_channels, 'window', depth, heads, dim_head, mlp_dim, dropout, emb_dropout, partition_size, patch_size,
            transformer=shared_transformer,
            to_patch_embedding=shared_embedding
        )

        self.attn_grid = StandardPartitionAttention(
            out_channels, out_channels, 'grid', depth, heads, dim_head, mlp_dim, dropout, emb_dropout, partition_size, patch_size,
            transformer=shared_transformer,
            to_patch_embedding=shared_embedding
        )

    def forward(self, x):
        x, attns1 = self.attn_block(x)
        x, attns2 = self.attn_grid(x)
        # Return 0 for sx to maintain compatibility with training loops expecting 3 values
        return x, 0, attns1+attns2

class TransformerDown_SPrune_Test(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, image_size, depth=2, dmodel=1024, mlp_dim=2048, patch_size=2, heads=6,
                 dim_head=128, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = in_channels * patch_height * patch_width
        self.dmodel = out_channels
        self.mlp_dim = self.dmodel * 4

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.patch_dim, self.dmodel),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerSPrune_Test(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout, num_patches)

        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height // patch_height),
        )

        self.pred_class = nn.Conv2d(in_channels, 2, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        sx = self.pred_class(x)  # b n h w -> b 2 h w
        sxp = self.softmax(sx)
        sxp_neg_1d = rearrange(sxp[:, 0, :, :], 'b h w -> b (h w)')  # b n

        x = self.to_patch_embedding(x)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        b, n, _ = x.shape
        # x += self.pos_embedding[:, :n]

        x = self.dropout(x)
        # transformer layer
        ax, attns = self.transformer(x, sxp_neg_1d)
        out = self.recover_patch_embedding(ax)
        return out, sx, attns
