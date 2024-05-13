import torch
from config import *
from dataset import train_dataset, tensor_to_pil
import matplotlib.pyplot as plt

# 前向diffusion计算参数
betas = torch.linspace(0.00001, 0.02, Config.T)
alphas = 1 - betas

# 计算方差，形状都是[T, ]
alphas_cumprod = torch.cumprod(alphas, dim=0) # alpha_t累乘, (a1, a2, a3, ..., aT) -> (a1, a1*a2, a1*a2*a3, ..., a1*a2*...*aT
alphas_cumprod_prev = torch.cat((torch.tensor([1.0]), alphas_cumprod[:-1]), dim=-1) # alpha_t-1累乘, (a1, a2, a3, ..., aT) -> (1, a1, a1*a2, ..., a1*a2*...*aT-1)
variance = (1 - alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod) # 方差

# 前向加噪过程
def forward_diffusion(batch_x, batch_t):
    '''
    执行前向加噪过程
    输入：
        batch_x: [batch, channel, width, height]
        batch_t: [batch, ]
    输出：
        batch_x_t:
        batch_noise_t:
    '''
    batch_noise_t = torch.rand_like(batch_x)
    batch_alpha_cumprod = alphas_cumprod.to(Config.DEVICE)[batch_t].view(batch_x.size(0), 1, 1, 1)
    batch_x_t = torch.sqrt(batch_alpha_cumprod) * batch_x + torch.sqrt(1 - batch_alpha_cumprod) * batch_noise_t
    return batch_x_t, batch_noise_t

if __name__ == '__main__':
    batch_x = torch.stack((train_dataset[0][0], train_dataset[1][0]), dim=0).to(Config.DEVICE)
    
    # 加噪前的样子
    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.imshow(tensor_to_pil(batch_x[0]))
    plt.subplot(2,2,2)
    plt.imshow(tensor_to_pil(batch_x[1]))
    # plt.show()
    
    batch_x=batch_x*2-1 # [0,1]像素值调整到[-1,1]之间,以便与高斯噪音值范围匹配
    batch_t=torch.randint(0, Config.T, size=(batch_x.size(0),)).to(Config.DEVICE)  # 每张图片随机生成diffusion步数
    print('batch_t:',batch_t)
    
    batch_x_t,batch_noise_t=forward_diffusion(batch_x,batch_t)
    print('batch_x_t:',batch_x_t.size())
    print('batch_noise_t:',batch_noise_t.size())

    # 加噪后的样子
    plt.subplot(2,2,3)
    plt.imshow(tensor_to_pil((batch_x_t[0]+1)/2))   
    plt.subplot(2,2,4)
    plt.imshow(tensor_to_pil((batch_x_t[1]+1)/2))
    plt.show()
