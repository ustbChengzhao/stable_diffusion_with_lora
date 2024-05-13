from config import *
from torch.utils.data import DataLoader
from dataset import train_dataset
from unet import UNet
from diffusion import forward_diffusion
import torch
from torch import nn
import os

from torch.utils.tensorboard import SummaryWriter

EPOCH = 200
BATCH_SIZE = 32

dataloader = DataLoader(train_dataset,
                        batch_size=BATCH_SIZE,
                        num_workers=4,
                        persistent_workers=True,
                        shuffle=True)

model = UNet(1).to(Config.DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

writer = SummaryWriter()

if __name__ == "__main__":
    model.train()
    n_iter = 0
    for epoch in range(EPOCH):
        last_loss = 0
        for batch_x, batch_cls in dataloader:
            # 图像的像素转为[-1, 1]，和高斯分布对应
            batch_x = batch_x.to(Config.DEVICE) * 2 - 1
            # 引导分类ID
            batch_cls = batch_cls.to(Config.DEVICE)
            # 为每张图片生成随机t时刻
            batch_t = torch.randint(0, Config.T, size=(batch_x.size(0),)).to(Config.DEVICE)
            # 生成t时刻的加噪图片和对应噪音
            batch_x_t, batch_noise_t = forward_diffusion(batch_x, batch_t)
            # 模型预测t时刻的噪音
            batch_predict_t = model(batch_x_t, batch_t, batch_cls)
            # 求损失
            loss = loss_fn(batch_predict_t, batch_noise_t)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录损失
            last_loss = loss.item()
            writer.add_scalar("loss", last_loss, n_iter)
            n_iter += 1
        print(f"epoch:{epoch}, loss:{last_loss}")
        # 保存模型
        if not os.path.exists("model"):
            os.mkdir("model")
        torch.save(model.state_dict(), f"model/{epoch}.pth")
    writer.close()  

