import torch

class Config:
    IMG_SIZE = 48
    T=1000
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")