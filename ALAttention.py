import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import DropPath, trunc_normal_

class ALAttention(nn.Module): 
    def __init__(self, input_size, dim, num_heads=1, window_size=3, leg_size=5
                 , qkv_bias=True, qk_scale=None,):
        super().__init__()
        assert window_size % 2 == 1, 'odd'
        self.dim = dim
        self.window_size = window_size
        self.leg_size = leg_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.set_input_size(input_size)

    def forward(self, x):
        x = self.attention(x)
        x = self.proj(x)
        return x

    def attention(self, x):
        B, C, H, W = x.shape
        assert H >= self.window_size and W >= self.window_size, 'input size must not be smaller than window size'
        qkv = self.qkv(x).view(B, 3, self.num_heads, C // self.num_heads, H * W).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q.unsqueeze(3) @ (k[:, :, self.attn_idx]).transpose(-1, -2)  # B,nh,L,1,K^2
        attn = attn.softmax(dim=-1)
        x = (attn @ v[:, :, self.attn_idx]).squeeze(3).transpose(-1, -2).contiguous().view(B, C, H, W)
        return x


    def get_attn_idx(self, H, W):

        attn_idx = torch.arange(0, H * W, dtype=torch.float).view(1, 1, H, W)
        attn_idx = attn_idx.view(-1).type(torch.long)
        attn_idx = self.get_bunfold_idx(H, W)[attn_idx]
        return attn_idx

    def get_bunfold_idx(self, H, W):
        h_idx = torch.arange(W).repeat(H)#50*50 [ 0,  1,  2,  ..., 47, 48, 49]
        w_idx = torch.arange(H).repeat_interleave(W) * W #[0, 0, 0, ..., 49, 49, 49]*50 =[0,..., 2744] 值*50
        hw_idx = h_idx + w_idx
        unfold_idx = torch.zeros([H*W,(self.window_size**2)+(2*(self.leg_size-self.window_size))])

        for hw,k in enumerate(hw_idx):
            if (hw % W) < (self.leg_size//2) :
                kB_idx_1C = torch.arange(start=hw//W*W,end=hw//W*W+self.leg_size,step=1)
            elif (hw % W) >= (W - (self.leg_size//2)):
                kB_idx_1C = torch.arange(start=(hw//W+1)*W - self.leg_size, end=(hw//W+1)*W, step=1)
            else:
                kB_idx_1C = torch.arange(start=hw - (self.leg_size//2), end=hw+(self.leg_size//2)+1, step=1)

            if (hw // W) < (self.leg_size//2) :#
                kB_idx_2C = torch.arange(start= hw%W ,end=W * self.leg_size,step=W)
            elif (hw // W) >= (H - (self.leg_size//2)):
                kB_idx_2C = torch.arange(start=(H-self.leg_size)*W+(hw%W), end=(H-1)*W+(hw%W)+1, step=W)
            else:
                kB_idx_2C = torch.arange(start=(hw//W-(self.leg_size//2))*W+(hw%W), end=(hw//W+(self.leg_size//2))*W+(hw%W)+1, step=W)

            if (hw % W) < (self.window_size//2):
                kN_idx_1C = torch.arange(start=(hw//W)*W, end=(hw//W)*W + self.window_size,step=1).repeat(self.window_size)
            elif (hw % W) >= (W - (self.window_size//2)):
                kN_idx_1C = torch.arange(start=(hw//W+1) * W-self.window_size, end=(hw//W+1) * W,step=1).repeat(self.window_size)
            else:
                kN_idx_1C = torch.arange(start=hw - (self.window_size//2), end=hw + (self.window_size//2) + 1,step=1).repeat(self.window_size)

            if (hw // W) < (self.window_size//2) :
                kN_idx_2C = (torch.arange(start=-(hw//W), end=self.window_size-(hw//W),step=1)).repeat_interleave(self.window_size) * W
            elif (hw // W) >= (H - (self.window_size//2)):
                kN_idx_2C = (torch.arange(start=H-(hw//W)-self.window_size, end=H-(hw//W), step=1)).repeat_interleave(self.window_size) * W
            else:
                kN_idx_2C = (torch.arange(start=-(self.window_size//2), end=(self.window_size//2)+1, step=1)).repeat_interleave(self.window_size) * W
            kN_idx_C = kN_idx_1C + kN_idx_2C
            k_idx_C = torch.cat([kN_idx_C,kB_idx_1C, kB_idx_2C], dim=0)
            k_idx_C = torch.unique(k_idx_C, dim=0)
            unfold_idx[k,:] = k_idx_C
        return unfold_idx.type(torch.long) #

    def set_input_size(self, input_size):
        H, W = input_size
        self.H, self.W = H, W
        attn_idx = self.get_attn_idx(H, W)
        self.register_buffer("attn_idx", attn_idx)


