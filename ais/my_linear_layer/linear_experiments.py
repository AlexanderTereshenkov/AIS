import torch.nn as nn
import torch

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bies = True):
        super().__init__()

        self.weights = nn.Parameter(torch.rand(in_features, out_features))
        if bies:
            self.bies = nn.Parameter(torch.rand(out_features))
        else:
            self.bies = None
        
    def forward(self, x):
        x = x @ self.weights
        if self.bies != None:
            x += self.bies

        return x
    