import torch
import torch.nn as nn 
import glob
import numpy as np
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import random
import os.path
import copy
from torchvision import datasets
import torchvision
from torch.utils.data import DataLoader
from utils import  w, detach_var
from utils import HSIUtil
from model import *
from tools import *
from parameters import args
torch.manual_seed(args.random_seed)
# target dataset name
####################
DATASET = 'KSC'#####
####################
channel_dict = {"PaviaU":[103, 9], "PaviaC":[102, 9], "Salinas":[204, 16],
                "SalinasA":[204, 6], "KSC":[176, 13], "IndianPines":[200, 16]}
#####################################################
# GenericNeuralNetforxxx xxx is DATASET name
class GeneralFullyCTest(GenericNeuralNetforKSC):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, loss):
        inp, out = loss.sample()
        inp = w(Variable(inp.view(inp.size()[0], channel_dict[DATASET][0])))
        out = w(Variable(out))
        cur_layer = 0
        while 'weight_' + str(cur_layer) in self.params:
            inp = self.activation(torch.matmul(inp, self.params['weight_' + str(cur_layer)]) +
                                  self.params['bias_' + str(cur_layer)])
            cur_layer += 1
        inp = torch.matmul(inp, self.params['final_weight']) + self.params['final_bias']
        l = self.loss(inp, out)
        inp = self.softmax(inp)
        return l, inp

# Package the individual test data into batch data that can be sampled
class Data_Pack():
    def __init__(self):
        self.data = 0
        self.label = 0
    def get_sample(self, data, label):
        self.data = data
        self.label = label
    def sample(self):
        return self.data.unsqueeze(0), self.label.unsqueeze(0)

def fit_normal(optimizee_objective_function, optimizee_network, optimizer_name,
               n_tests = 10, epochs = 50, SAMPLE_SIZE=10, **kwargs):
    results = []
    datapack = Data_Pack()
    for i in range(n_tests):
        objective_function = optimizee_objective_function(DATASET=DATASET, BATCH_SIZE=32, TRAINING_OPTIMIZER=False, run=i)
        optimizee = w(optimizee_network())
        # Get the parameters of the optimizee in order to optimize them
        optimizee.train()
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
            current_loss, _ = optimizee(objective_function)
            
            total_loss.append(current_loss.data.cpu().numpy())
            
            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()
        results.append(total_loss)

        # Get the final classification result
        optimizee.eval()
        img, gt, LABEL_VALUES, _, _, _ = get_dataset(DATASET, target_folder="../Data/Datasets/")
        train_gt, test_gt = get_sample(DATASET, SAMPLE_SIZE, i)
        indices_test = np.nonzero(test_gt)

        prdict_labels = np.zeros_like(gt)
        # Coordinates of all test labels
        xy_list = list(zip(*indices_test))
        num_test = 0
        num_right = 0

        for x in range(gt.shape[0]):
            for y in range(gt.shape[1]):
                #get the data
                x1, y1 = x, y
                x2, y2 = x+1, y+1
                data = img[x1:x2, y1:y2]
                label = gt[x,y]
                #print(data.shape)
                data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
                label = np.asarray(np.copy(label), dtype='int64')
                # Load the data into PyTorch tensors
                data = torch.from_numpy(data)
                # Now we need to predict the label with gt=0, but the label input to the neural network cannot be less than 0 or greater than the number of classes.
                if label != 0:
                    label = torch.from_numpy(label)-1
                else:
                    label = torch.from_numpy(label)
                datapack.get_sample(data, label)
                _, out = optimizee(datapack)
                #label = torch.from_numpy(label) - 1
                #print(out.shape)
                #print(out.device)
                # to an integer number
                out = out.argmax(dim=1).detach().cpu().numpy().squeeze()
                prdict_labels[x, y] = out + 1
                # Although all predictions are made, only test_gt is counted.
                if test_gt[x, y] != 0:
                    num_test += 1
                    if (out +1) == test_gt[x, y]:
                        num_right += 1
        print("accuracy:", num_right/num_test)
        path = "../NormResults/"+DATASET+"/"
        if not os.path.isdir(path):
            os.makedirs(path)
        np.save(path+"prejected_labels_"+str(SAMPLE_SIZE)+optimizer_name+"_run_"+str(i)+"_acc_"+str(num_right/num_test)+".npy", prdict_labels)
    results = np.array(results)
    np.save(path+"loss_"+str(SAMPLE_SIZE)+"_"+optimizer_name+".npy", results)
if __name__ == '__main__':
    OPTIMIZER_NAMES = ['AdaGrad', 'Adam', 'RMSprop', 'SGD', 'SGDMomentum']
    OPTIMIZER_PARAMETERS = [{}, {},{}, {}, {'nesterov': True, 'momentum': 0.9}]
    # Select the optimal learning rate value through multiple experiments
    if DATASET == 'PaviaU':
        LEARNING_RATES = [0.1, 0.05, 0.01, 0.5, 0.5]
    elif DATASET == 'PaviaC':
        LEARNING_RATES = [0.1, 0.05, 0.01, 0.5, 0.5]
    elif DATASET == 'Salinas':
        LEARNING_RATES = [0.05, 0.05, 0.005, 0.5, 0.5]
    elif DATASET == 'SalinasA':
        LEARNING_RATES = [0.05, 0.05, 0.005, 0.5, 0.1]
    elif DATASET == 'KSC':
        LEARNING_RATES = [0.05, 0.05, 0.01, 0.5, 0.1]
    elif DATASET == 'IndianPines':
        LEARNING_RATES = [0.05, 0.05, 0.005, 0.05, 0.5]

    for optimizer_name, kwargs, lr in zip(OPTIMIZER_NAMES, OPTIMIZER_PARAMETERS, LEARNING_RATES):
        print(optimizer_name)
        fit_normal(HSIUtil, GeneralFullyCTest, optimizer_name,  n_tests=10, lr =lr, **kwargs)

