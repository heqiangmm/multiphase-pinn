import torch
from pina.model import FeedForward

class SigmoidNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = FeedForward(*args, **kwargs)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.model(x))