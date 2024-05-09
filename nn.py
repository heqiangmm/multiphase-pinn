import torch
from pina.model import FeedForward

class SigmoidNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = FeedForward(*args, **kwargs)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        output = self.model(x)
        sigmoid_output = torch.cat((self.sigmoid(output[:, :1]), output[:, 1:]), dim=1)
        return sigmoid_output
