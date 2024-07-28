import torch
import math
import torch.nn as nn 
import glob
import numpy as np
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import random
#from tqdm import tqdm_notebook as tqdm
import tqdm
import multiprocessing
import os.path
import csv
import copy
import joblib
from torchvision import datasets
import torchvision
from torch.utils.data import DataLoader
from tools import *
import torch.utils.data as Torchdata
def gradient_process(x):
    #Input is (N), output is (N, 2)
    p = 10
    eps = 1e-6  
    indicator = (x.abs() > math.exp(-p))
    gradient_after_process = torch.zeros(indicator.shape[0], 2)
    for i in range(0, indicator.shape[0]):
            if indicator[i] == 0:
                gradient_after_process[i, 0] = -1
                gradient_after_process[i, 1] = float(math.exp(p)) * x[i]
            else:
                gradient_after_process[i, 0] = torch.log(torch.abs(x[i]) + 1e-8) / p
                gradient_after_process[i, 1] = torch.sign(x[i])
    return gradient_after_process

def toGPU(gpu_use_glag, gpu_num, data):
    if gpu_use_glag:
        return data.cuda(gpu_num)
    else:
        return data

#HSIdata for training the meta-optimizers
class HSIUtil:
    def __init__(self, DATASET, PATCH_SIZE=1, BATCH_SIZE=128, TRAINING_OPTIMIZER = True, SAMPLE_SIZE=10, run = 0):
        img, gt, LABEL_VALUES, _, _, _ = get_dataset(DATASET, target_folder="../Data/Datasets/")
        if not TRAINING_OPTIMIZER:
            train_gt, test_gt = get_sample(DATASET, SAMPLE_SIZE, run)
            gt = train_gt
        dataset = HyperX(img, gt, DATASET, PATCH_SIZE, False, False)

        indices = list(range(len(dataset)))
        self.loader = Torchdata.DataLoader(dataset,
                                           batch_size=BATCH_SIZE,
                                           sampler=Torchdata.sampler.SubsetRandomSampler(indices))
        self.batches = []
        self.cur_batch =0
    def sample(self):
        if self.cur_batch >= len(self.batches):
            self.batches = []
            self.cur_batch = 0
            for b in self.loader:
                self.batches.append(b)
        batch = self.batches[self.cur_batch]
        self.cur_batch += 1
        return batch

USE_CUDA = True
def w(v):
    if USE_CUDA:
        return v.cuda()
    return v
# This function is responsible for disabling the propagation of certain variable gradients
def detach_var(v):
    var = w(Variable(v.data, requires_grad=True))
    var.retain_grad()
    return var
