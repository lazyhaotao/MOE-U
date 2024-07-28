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
import csv
import copy
import joblib
from torchvision import datasets
import torchvision
from torch.utils.data import DataLoader
from utils import w

class GenericNeuralNet(nn.Module):
    def __init__(self, layer_size = 20, n_layers = 1, **kwargs):
        super().__init__()
        if kwargs != {}:
            input_size = 28*28
            self.params = kwargs
        else:
            input_size = 28*28
            self.params = {}
            # Initialize parameter values for weight and bias
            for i in range(n_layers):
                weight_name = 'weight_' + str(i)
                bias_name = 'bias_' + str(i)
                #
                self.params[weight_name] = nn.Parameter(torch.randn(input_size, layer_size) * 0.001)
                self.params[bias_name] = nn.Parameter(torch.zeros(layer_size))
                input_size = layer_size
                
            self.params['final_weight'] = nn.Parameter(torch.randn(input_size, 10) * 0.001)
            self.params['final_bias'] = nn.Parameter(torch.zeros(10))
            
            # Put it all in a module list so that ordinary optimizers can find them.
            self.mods = nn.ParameterList()
            for v in self.params.values():
                self.mods.append(v)
                
        self.activation = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss()
        
    def all_named_parameters(self):
        return [(k, v) for k, v in self.params.items()]
    
    def forward(self, loss):
        inp, out = loss.sample()
        inp = w(Variable(inp.view(inp.size()[0], 28*28)))
        out = w(Variable(out))
        
        cur_layer = 0
        while 'weight_' + str(cur_layer) in self.params:
            inp = self.activation(torch.matmul(inp, self.params['weight_' + str(cur_layer)]) + 
                                               self.params['bias_' + str(cur_layer)])
            cur_layer += 1
            
        inp = torch.matmul(inp, self.params['final_weight']) + self.params['final_bias']
        #CrossEntropy 已经包含softmax了
        l = self.loss(inp, out)
        
        return l


class GenericNeuralNetforPaviaU(nn.Module):
    def __init__(self, layer_size=20, n_layers=3, **kwargs):
        super().__init__()
        #PaviaU has 103 bands
        if kwargs != {}:
            input_size = 103
            self.params = kwargs
        else:
            input_size = 103
            self.params = {}
            # Initialize parameter values for weight and bias
            for i in range(n_layers):
                weight_name = 'weight_' + str(i)
                bias_name = 'bias_' + str(i)
                self.params[weight_name] = nn.Parameter(torch.randn(input_size, layer_size))
                self.params[bias_name] = nn.Parameter(torch.zeros(layer_size))
                input_size = layer_size
            #PaviaU has 9 classes
            self.params['final_weight'] = nn.Parameter(torch.randn(input_size, 9))
            self.params['final_bias'] = nn.Parameter(torch.zeros(9))

            # Put it all in a module list so that ordinary optimizers can find them.
            self.mods = nn.ParameterList()
            for v in self.params.values():
                self.mods.append(v)

        self.activation = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss()

    def all_named_parameters(self):
        return [(k, v) for k, v in self.params.items()]

    def forward(self, loss):
        inp, out = loss.sample()
        inp = w(Variable(inp.view(inp.size()[0], 103)))
        out = w(Variable(out))

        cur_layer = 0
        while 'weight_' + str(cur_layer) in self.params:
            inp = self.activation(torch.matmul(inp, self.params['weight_' + str(cur_layer)]) +
                                  self.params['bias_' + str(cur_layer)])
            cur_layer += 1

        inp = torch.matmul(inp, self.params['final_weight']) + self.params['final_bias']
        l = self.loss(inp, out)

        return l

class GenericNeuralNetforPaviaC(nn.Module):
    def __init__(self, layer_size=20, n_layers=3, **kwargs):
        super().__init__()
        #PaviaU has 103 bands
        if kwargs != {}:
            input_size = 102
            self.params = kwargs
        else:
            input_size = 102
            self.params = {}
            # Initialize parameter values for weight and bias
            for i in range(n_layers):
                weight_name = 'weight_' + str(i)
                bias_name = 'bias_' + str(i)
                self.params[weight_name] = nn.Parameter(torch.randn(input_size, layer_size))
                self.params[bias_name] = nn.Parameter(torch.zeros(layer_size))
                input_size = layer_size
            #PaviaU has 9 classes
            self.params['final_weight'] = nn.Parameter(torch.randn(input_size, 9))
            self.params['final_bias'] = nn.Parameter(torch.zeros(9))

            # Put it all in a module list so that ordinary optimizers can find them.
            self.mods = nn.ParameterList()
            for v in self.params.values():
                self.mods.append(v)

        self.activation = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss()

    def all_named_parameters(self):
        return [(k, v) for k, v in self.params.items()]

    def forward(self, loss):
        inp, out = loss.sample()
        inp = w(Variable(inp.view(inp.size()[0], 102)))
        out = w(Variable(out))

        cur_layer = 0
        while 'weight_' + str(cur_layer) in self.params:
            inp = self.activation(torch.matmul(inp, self.params['weight_' + str(cur_layer)]) +
                                  self.params['bias_' + str(cur_layer)])
            cur_layer += 1

        inp = torch.matmul(inp, self.params['final_weight']) + self.params['final_bias']
        l = self.loss(inp, out)

        return l
class GenericNeuralNetforSalinas(nn.Module):
    def __init__(self, layer_size=20, n_layers=3, **kwargs):
        super().__init__()
        #PaviaU has 103 bands
        if kwargs != {}:
            input_size = 204
            self.params = kwargs
        else:
            input_size = 204
            self.params = {}
            # Initialize parameter values for weight and bias
            for i in range(n_layers):
                weight_name = 'weight_' + str(i)
                bias_name = 'bias_' + str(i)
                self.params[weight_name] = nn.Parameter(torch.randn(input_size, layer_size))
                self.params[bias_name] = nn.Parameter(torch.zeros(layer_size))
                input_size = layer_size
            #PaviaU has 9 classes
            self.params['final_weight'] = nn.Parameter(torch.randn(input_size, 16))
            self.params['final_bias'] = nn.Parameter(torch.zeros(16))

            # Put it all in a module list so that ordinary optimizers can find them.
            self.mods = nn.ParameterList()
            for v in self.params.values():
                self.mods.append(v)

        self.activation = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss()

    def all_named_parameters(self):
        return [(k, v) for k, v in self.params.items()]

    def forward(self, loss):
        inp, out = loss.sample()
        inp = w(Variable(inp.view(inp.size()[0], 204)))
        out = w(Variable(out))

        cur_layer = 0
        while 'weight_' + str(cur_layer) in self.params:
            inp = self.activation(torch.matmul(inp, self.params['weight_' + str(cur_layer)]) +
                                  self.params['bias_' + str(cur_layer)])
            cur_layer += 1

        inp = torch.matmul(inp, self.params['final_weight']) + self.params['final_bias']
        l = self.loss(inp, out)

        return l

class GenericNeuralNetforSalinasA(nn.Module):
    def __init__(self, layer_size=20, n_layers=3, **kwargs):
        super().__init__()
        #PaviaU has 103 bands
        if kwargs != {}:
            input_size = 204
            self.params = kwargs
        else:
            input_size = 204
            self.params = {}
            # Initialize parameter values for weight and bias
            for i in range(n_layers):
                weight_name = 'weight_' + str(i)
                bias_name = 'bias_' + str(i)
                self.params[weight_name] = nn.Parameter(torch.randn(input_size, layer_size) )
                self.params[bias_name] = nn.Parameter(torch.zeros(layer_size))
                input_size = layer_size
            #SalinasA has 6 classes
            self.params['final_weight'] = nn.Parameter(torch.randn(input_size, 6))
            self.params['final_bias'] = nn.Parameter(torch.zeros(6))

            # Put it all in a module list so that ordinary optimizers can find them.
            self.mods = nn.ParameterList()
            for v in self.params.values():
                self.mods.append(v)

        self.activation = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss()

    def all_named_parameters(self):
        return [(k, v) for k, v in self.params.items()]

    def forward(self, loss):
        inp, out = loss.sample()
        inp = w(Variable(inp.view(inp.size()[0], 204)))
        out = w(Variable(out))

        cur_layer = 0
        while 'weight_' + str(cur_layer) in self.params:
            inp = self.activation(torch.matmul(inp, self.params['weight_' + str(cur_layer)]) +
                                  self.params['bias_' + str(cur_layer)])
            cur_layer += 1

        inp = torch.matmul(inp, self.params['final_weight']) + self.params['final_bias']
        l = self.loss(inp, out)

        return l
class GenericNeuralNetforKSC(nn.Module):
    def __init__(self, layer_size=20, n_layers=3, **kwargs):
        super().__init__()
        #PaviaU has 103 bands
        if kwargs != {}:
            input_size = 176
            self.params = kwargs
        else:
            input_size = 176
            self.params = {}
            # Initialize parameter values for weight and bias
            for i in range(n_layers):
                weight_name = 'weight_' + str(i)
                bias_name = 'bias_' + str(i)
                self.params[weight_name] = nn.Parameter(torch.randn(input_size, layer_size) )
                self.params[bias_name] = nn.Parameter(torch.zeros(layer_size))
                input_size = layer_size
            #SalinasA has 6 classes
            self.params['final_weight'] = nn.Parameter(torch.randn(input_size, 13))
            self.params['final_bias'] = nn.Parameter(torch.zeros(13))

            # Put it all in a module list so that ordinary optimizers can find them.
            self.mods = nn.ParameterList()
            for v in self.params.values():
                self.mods.append(v)

        self.activation = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss()

    def all_named_parameters(self):
        return [(k, v) for k, v in self.params.items()]

    def forward(self, loss):
        inp, out = loss.sample()
        inp = w(Variable(inp.view(inp.size()[0], 176)))
        out = w(Variable(out))

        cur_layer = 0
        while 'weight_' + str(cur_layer) in self.params:
            inp = self.activation(torch.matmul(inp, self.params['weight_' + str(cur_layer)]) +
                                  self.params['bias_' + str(cur_layer)])
            cur_layer += 1

        inp = torch.matmul(inp, self.params['final_weight']) + self.params['final_bias']
        l = self.loss(inp, out)

        return l

