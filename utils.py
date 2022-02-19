import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def bpr_loss(pos: torch.Tensor, neg: torch.Tensor)-> torch.Tensor:
    """Bayesian Personalized Ranking Loss

    Parameters
    ----------
    pos : torch.Tensor
        Ranking score (0..1)
    neg : torch.Tensor
        Ranking score (0..1)
    
    Return
    ------
    loss scalar
    """
    diff = pos - neg
    return -F.logsigmoid(diff).mean()
    