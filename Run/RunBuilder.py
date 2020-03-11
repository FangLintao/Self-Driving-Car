#!/usr/bin/env python
# coding: utf-8


import torch
import torchvision 
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from itertools import product
import time
import pandas as pd
from collections import OrderedDict
from collections import namedtuple
from IPython.display import clear_output


class RunBuilder():
    # build sets of parameters that define our runs
    @staticmethod
    def get_runs(params):
        # params is the dictionary
        Run  = namedtuple('Run',params.keys())
        # Run as this tuple name and param.keys() is the content insides this tuple
        runs=[]
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs
