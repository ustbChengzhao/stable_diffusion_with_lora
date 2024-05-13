import torch
from config import *
from diffusion import *
import matplotlib.pyplot as plt
from dataset import tensor_to_pil
from torch import nn
from unet import UNet

def backward_denoise(model, batch_x_t, batch_cls):
    steps = [batch_x_t, ]
    
    global alphas, alphas_cumprod, variance
    
    # model = model.to(Config.DEVICE)
    batch_x_t = batch_x_t.to(Config.DEVICE)
    alphas = alphas.to(Config.DEVICE)
    alphas_cumprod = alphas_cumprod.to(Config.DEVICE)
    variance = variance.to(Config.DEVICE)
    batch_cls = batch_cls.to(Config.DEVICE)
    
    model.eval()
    with torch.no_grad():
        for t in range(Config.T - 1, -1, -1):
            batch_t = torch.full((batch_x_t.size(0),), t).to(Config.DEVICE)
            # 预测x_t时刻的噪音
            batch_predict_noise_t = model(batch_x_t, batch_t, batch_cls)
            # 生成t-1时刻的图像
            shape = (batch_x_t.size(0), 1, 1, 1)
            batch_mean_t=1/torch.sqrt(alphas[batch_t].view(*shape))*  \
                (
                    batch_x_t-
                    (1-alphas[batch_t].view(*shape))/torch.sqrt(1-alphas_cumprod[batch_t].view(*shape))*batch_predict_noise_t
                )
                
            if t != 0:
                batch_x_t = batch_mean_t + torch.rand_like(batch_mean_t) * torch.sqrt(variance[batch_t].view(*shape))
            else:
                batch_x_t = batch_mean_t
            batch_x_t = torch.clamp(batch_x_t, -1, 1).detach()
            steps.append(batch_x_t)
    return steps

if __name__ == '__main__':
    # 加载模型
    model = UNet(1)
    model.load_state_dict(torch.load('model/199.pth'))
    model.eval()
    model = model.to(Config.DEVICE)
    # 生成噪音图
    batch_size=10
    batch_x_t=torch.randn(size=(batch_size,1,Config.IMG_SIZE,Config.IMG_SIZE))  # (5,1,48,48)
    batch_cls=torch.arange(start=0,end=10,dtype=torch.long)   # 引导词promot
    # 逐步去噪得到原图
    steps=backward_denoise(model,batch_x_t,batch_cls)
    # 绘制数量
    num_imgs=20
    # 绘制还原过程
    plt.figure(figsize=(15,15))
    for b in range(batch_size):
        for i in range(0,num_imgs):
            idx=int(Config.T/num_imgs)*(i+1)
            # 像素值还原到[0,1]
            final_img=(steps[idx][b].to('cpu')+1)/2
            # tensor转回PIL图
            final_img=tensor_to_pil(final_img)
            plt.subplot(batch_size,num_imgs,b*num_imgs+i+1)
            plt.imshow(final_img)
    plt.show()