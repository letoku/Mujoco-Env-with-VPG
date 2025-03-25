import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical


class Policy(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, units: int = 16):
        super().__init__()
        self.l1 = nn.Linear(input_dim, units)
        self.l2 = nn.Linear(units, units)
        self.l3 = nn.Linear(units, output_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return Categorical(logits=x)
