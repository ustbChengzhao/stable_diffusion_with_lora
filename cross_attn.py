import torch
import torch.nn as nn
from config import *
import math

class CrossAttention(nn.Module):
    def __init__(self, channel, qsize, vsize, fsize, cls_emb_size):
        super().__init__()
        self.w_q = nn.Linear(channel, qsize)
        self.w_k = nn.Linear(cls_emb_size, qsize)
        self.w_v = nn.Linear(cls_emb_size, vsize)
        self.softmax = nn.Softmax(dim=-1)
        self.z_linear = nn.Linear(vsize, channel)
        self.norm1 = nn.LayerNorm(channel)
        
        # feed-forward
        self.feedforward = nn.Sequential(
            nn.Linear(channel, fsize),
            nn.ReLU(),
            nn.Linear(fsize, channel),
        )
        
        self.norm2 = nn.LayerNorm(channel)
        
    def forward(self, x, cls_emb):
        """
        x: [batch_size, channel, w, h]
        cls_emb: [batch_size, cls_emb_size]
        """
        x = x.permute(0, 2, 3, 1)   # [batch_size, w, h, channel]
        Q = self.w_q(x) # [batch_size, w, h, qsize]
        K = self.w_k(cls_emb) # [batch_size, vsize]
        V = self.w_v(cls_emb) # [batch_size, vsize]
        Q = Q.view(Q.size(0), Q.size(1) * Q.size(2), Q.size(3)) # [batch_size, w*h, qsize]
        K = K.view(K.size(0), K.size(1), 1) # [batch_size, qsize, 1]
        V = V.view(V.size(0), 1, V.size(1)) # [batch_size, 1, vsize]
        
        # 计算attention
        attn =  torch.matmul(Q, K)/ math.sqrt(Q.size(2)) # [batch_size, w*h, 1]
        attn = self.softmax(attn)   # [batch_size, w*h, 1]

                    
        # 注意力层输出
        Z = torch.matmul(attn, V) # [batch_size, w*h, vsize]
        Z = self.z_linear(Z)    # [batch_size, w*h, fsize]
        Z = Z.view(Z.size(0), x.size(1), x.size(2), x.size(3))  # [batch_size, w, h, channel]
        
        # 残差连接
        Z = self.norm1(Z + x)   # [batch_size, w, h, channel]
        
        # feedforward
        out = self.feedforward(Z)   # [batch_size, w, h, channel]
        out = self.norm2(out + Z)   # [batch_size, w, h, channel]
        return out.permute(0, 3, 1, 2) # [batch_size, channel, w, h]
    
if __name__=='__main__':
    batch_size=2
    channel=1
    qsize=256
    cls_emb_size=32
    
    cross_atn=CrossAttention(channel=1,qsize=256,vsize=128,fsize=512,cls_emb_size=32)
    
    x=torch.randn((batch_size,channel,Config.IMG_SIZE,Config.IMG_SIZE))
    cls_emb=torch.randn((batch_size,cls_emb_size)) # cls_emb_size=32

    Z=cross_atn(x,cls_emb)
    print(Z.size())     # Z: (2,1,48,48)