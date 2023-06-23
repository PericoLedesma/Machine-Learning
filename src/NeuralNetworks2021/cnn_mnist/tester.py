"""
Experimental environment script
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets
from torchvision.transforms import ToTensor

# Local imports
from CNN import *
from readingMNIST import *
from load_eval import *

def tester():
    model = load("/home/goncalo/Documents/RUG/MSc_1st_year/block_1b/machineLearning/NeuralNetworks2021/cnn_mnist/models/model_cnn_id0001.pt")
    loaders = load_loaders(load_train_std_MNIST(), load_test_std_MNIST())
    eval(model,loaders)

if __name__ == '__main__':
    tester()