import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from config import *

# PIL图像转为tensor
pil_to_tensor = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)), # 重置图像大小
    transforms.ToTensor()   # 转为tensor
])

# tensor转为PIL图像
tensor_to_pil = transforms.Compose([
    transforms.Lambda(lambda t: t * 255),   # 像素值还原
    transforms.Lambda(lambda t: t.type(torch.uint8)),   # 像素值取整
    transforms.ToPILImage() # tensor转为PIL图像，(C, H, W) -> (H, W, C)
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=pil_to_tensor)

if __name__ == '__main__':
    # 随机获取一个训练样本的tensor
    img, label = train_dataset[2]
    img = tensor_to_pil(img)
    plt.imshow(img, cmap='gray')
    plt.show()

