import torch
import torch.nn as nn
import math
import numpy as np


class SenderAggregation(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SenderAggregation, self).__init__()
        
        self.fc1 = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        residual = x
        x = self.fc1(x)
        B = x.mean(dim=2, keepdim=True)
        C = B.mean(dim=1, keepdim=True)
        result = 0.5*residual + 0.5*C
        
        return result
