import os

from matplotlib import pyplot as plt
import numpy as np
import torch

use_cuda = torch.cuda.is_available()
print('Using cuda:{}'.format(use_cuda))
torch_cuda = torch.cuda if use_cuda else torch
device = torch.device('cuda' if use_cuda else 'cpu')

