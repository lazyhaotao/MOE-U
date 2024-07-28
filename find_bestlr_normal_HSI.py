import torch
import torch.nn as nn 
import glob
import numpy as np
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
import multiprocessing
import os.path
import copy
import joblib
from torchvision import datasets
import torchvision
from torch.utils.data import DataLoader
from utils import  w, detach_var
from time import *
from utils import HSIUtil
from model import *

from parameters import args
torch.manual_seed(args.random_seed)
####################
DATASET = 'SalinasA'#
####################
def fit_normal(optimizee_objective_function, optimizee_network, optimizer_name,
               n_tests = 10, epochs = 50, **kwargs):
    results = []
    
    for i in range(n_tests):
        objective_function = optimizee_objective_function(DATASET=DATASET, BATCH_SIZE=32, TRAINING_OPTIMIZER=False, run=i)
        optimizee = w(optimizee_network())
        # Get the parameters of the optimizee in order to optimize them
        # Initialize the optimizer object
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(optimizee.parameters(), **kwargs)
        elif optimizer_name == 'AdaGrad':
            optimizer = optim.Adagrad(optimizee.parameters(), **kwargs)
        elif optimizer_name == 'RMSprop':
            optimizer = optim.RMSprop(optimizee.parameters(), **kwargs)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(optimizee.parameters(), **kwargs)
        else:
            optimizer = optim.SGD(optimizee.parameters(), **kwargs)
        total_loss = []
        for _ in range(epochs):
            current_loss = optimizee(objective_function)
            #print(current_loss.data.cpu().numpy())
            total_loss.append(current_loss.data.cpu().numpy())
            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()
        results.append(total_loss)
        #print(total_loss)
    return results

def find_best_lr_normal(optimizee_objective_function, optimizee_network, optimizer, **extra_kwargs):
    best_loss = 10000000000000000.0
    best_lr = 0.0
    for lr in [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]:
        loss = best_loss + 1.0
        #选择接近收敛的时候的loss进行平均
        loss = np.mean([np.sum(s[30:50]) for s in fit_normal(optimizee_objective_function, optimizee_network, optimizer,
                                                      lr = lr, **extra_kwargs)])
        if loss < best_loss:
            best_loss = loss
            best_lr = lr
    return best_loss, best_lr
if __name__ == '__main__':
    OPTIMIZER_NAMES = ['AdaGrad','Adam', 'RMSprop', 'SGD', 'SGD + Momentum']
    OPTIMIZER_PARAMETERS = [{},{}, {}, {}, {'nesterov': True, 'momentum': 0.9}]
    for optimizer_name, kwargs in zip(OPTIMIZER_NAMES, OPTIMIZER_PARAMETERS):
        print(optimizer_name)
        ############################################################
        print(find_best_lr_normal(HSIUtil, GenericNeuralNetforSalinasA, optimizer_name, **kwargs, n_tests=10))

