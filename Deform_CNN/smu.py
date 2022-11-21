# coding=utf-8

import torch
from torch import nn


class SMU(nn.Module):
    """
    Implementation of SMU activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha: hyper parameter
    References:
        - See related paper:
        https://arxiv.org/abs/2111.04682
    """

    def __init__(self, alpha=0.01, mu=2.5):
        """
        Initialization.
        INPUT:
            - alpha: hyper parameter
            aplha is initialized with zero value by default
        """
        super(SMU, self).__init__()
        self.alpha = alpha
        # initialize mu
        self.mu = torch.nn.Parameter(torch.tensor(mu))

    def forward(self, x):
        return ((1 + self.alpha) * x + (1 - self.alpha) * x * torch.erf(self.mu * (1 - self.alpha) * x)) / 2


class SMU1(nn.Module):
    """
    Implementation of SMU-1 activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha: hyper parameter
    References:
        - See related paper:
        https://arxiv.org/abs/2111.04682
    """

    def __init__(self, alpha=0.01, mu=4.332461424154261e-9):
        """
        Initialization.
        INPUT:
            - alpha: hyper parameter
            aplha is initialized with zero value by default
        """
        super(SMU1, self).__init__()
        self.alpha = alpha
        # initialize mu
        self.mu = torch.nn.Parameter(torch.tensor(mu))

    def forward(self, x):
        return ((1 + self.alpha) * x + torch.sqrt(torch.square(x - self.alpha * x) + torch.square(self.mu))) / 2
