import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class _conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(_conv, self).__init__(in_channels = in_channels, out_channels = out_channels, 
                               kernel_size = kernel_size, stride = stride, padding = (kernel_size) // 2, bias = True)
        
        self.weight.data = torch.normal(torch.zeros((out_channels, in_channels, kernel_size, kernel_size)), 0.02)
        self.bias.data = torch.zeros((out_channels))
        
        for p in self.parameters():
            p.requires_grad = True
        

class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, BN = False, act = None, stride = 1, bias = True):
        super(conv, self).__init__()
        m = []
        m.append(_conv(in_channels = in_channel, out_channels = out_channel, 
                               kernel_size = kernel_size, stride = stride, padding = (kernel_size) // 2, bias = True))
        
        if BN:
            m.append(nn.BatchNorm2d(num_features = out_channel))
        
        if act is not None:
            m.append(act)
        
        self.body = nn.Sequential(*m)
        
    def forward(self, x):
        out = self.body(x)
        return out
        
class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, act = nn.ReLU(inplace = True), bias = True):
        super(ResBlock, self).__init__()
        m = []
        m.append(conv(channels, channels, kernel_size, BN = True, act = act))
        m.append(conv(channels, channels, kernel_size, BN = True, act = None))
        self.body = nn.Sequential(*m)
        
    def forward(self, x):
        res = self.body(x)
        res += x
        return res
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_res_block, act = nn.ReLU(inplace = True)):
        super(BasicBlock, self).__init__()
        m = []
        
        self.conv = conv(in_channels, out_channels, kernel_size, BN = False, act = act)
        for i in range(num_res_block):
            m.append(ResBlock(out_channels, kernel_size, act))
        
        m.append(conv(out_channels, out_channels, kernel_size, BN = True, act = None))
        
        self.body = nn.Sequential(*m)
        
    def forward(self, x):
        res = self.conv(x)
        out = self.body(res)
        out += res
        
        return out
        
class Upsampler(nn.Module):
    def __init__(self, channel, kernel_size, scale, act = nn.ReLU(inplace = True)):
        super(Upsampler, self).__init__()
        m = []
        m.append(conv(channel, channel * scale * scale, kernel_size))
        m.append(nn.PixelShuffle(scale))
    
        if act is not None:
            m.append(act)
        
        self.body = nn.Sequential(*m)
    
    def forward(self, x):
        out = self.body(x)
        return out

class discrim_block(nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size, act = nn.LeakyReLU(inplace = True)):
        super(discrim_block, self).__init__()
        m = []
        m.append(conv(in_feats, out_feats, kernel_size, BN = True, act = act))
        m.append(conv(out_feats, out_feats, kernel_size, BN = True, act = act, stride = 2))
        self.body = nn.Sequential(*m)
        
    def forward(self, x):
        out = self.body(x)
        return out


class SelfAttention(nn.Module):
    def __init__(self, channels, dim_k=64, dim_v=64):
        super().__init__()

        self.dim_k = dim_k
        self.dim_v = dim_v

        self.key = nn.Conv2d(channels, dim_k, kernel_size=1)
        self.query = nn.Conv2d(channels, dim_k, kernel_size=1)
        self.value = nn.Conv2d(channels, dim_v, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))  # stabilisasi residual

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        key = self.key(x).view(B, self.dim_k, N).permute(0, 2, 1)   # B, N, K
        query = self.query(x).view(B, self.dim_k, N)                 # B, K, N
        value = self.value(x).view(B, self.dim_v, N).permute(0, 2, 1)  # B, N, V

        print(key.is_cuda)
        print(query.is_cuda)
        print(key.dtype)
        print(query.dtype)
        print(key.shape)
        print(query.shape)

        print(torch.cuda.memory_allocated() / 1024**2)
        print(torch.cuda.memory_reserved() / 1024**2)
        print(torch.cuda.mem_get_info())
        
        attn = torch.bmm(key, query) / math.sqrt(self.dim_k)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, value)  # B, N, V
        out = out.permute(0, 2, 1).contiguous().view(B, self.dim_v, H, W)

        return x + self.gamma * out
    
