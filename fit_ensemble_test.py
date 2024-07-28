import torch
import torch.nn as nn 
import glob
import numpy as np
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import random
import multiprocessing
import os.path
import csv
import copy
import joblib
from torchvision import datasets
import torchvision
from utils import  w, detach_var
from optimizer2 import OptimizerNetwork
from target_func import RandomQuadraticLoss
from target_func import QuadraticOptimizee
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from parameters import args
torch.manual_seed(args.random_seed)

def average_all(update_list):
    return torch.mean(torch.cat(update_list, dim=1),dim=1).unsqueeze(1)

class ensemble_strategy_next():
    def __init__(self, beta, ensemble_num):
        self.beta = beta
        self.last_loss = None
        self.momentum_list = np.zeros(ensemble_num)

    def look_next(self, param_list, optimizee_obj_function, optimizee_network):
        for i in range(len(param_list)):
            optimizee = w(optimizee_network(**param_list[i]))
            current_loss = optimizee(optimizee_obj_function).data.cpu().numpy()
            self.momentum_list[i] = self.momentum_list[i] * self.beta + ((current_loss - self.last_loss))*(1 - self.beta) 
            optimizee_obj_function.cur_batch-=1
        optimizee_obj_function.cur_batch+=1
        return param_list[np.argmin(self.momentum_list)]
    def set_last_loss(self, loss):
            self.last_loss = loss

def fit_ensemble_test(optimizer_network, meta_optimizer, optimizee_obj_function, optimizee_network, iterations_to_optimize, iterations_to_unroll, out_mul, ensemble_strategy,
        seed, should_train = False, ensemble_num = 3, pth_list = [], ensemble_beta = 0.5):
    """
    Arguments: 
    - optimizer_network (the optimizer network we use, here the LSTM, Class instead of instance)
    - meta_optimizer (the optimizer of the optimizer network, e.g. Adam, SGD + nesterov, RMSprop, etc.)
    - optimizee_obj_function (the optimizee's objective function)
    - optimizee_network (the optimizee network)
    - epochs (total epochs for training)
    - iterations_to_optimize (iterations in every epoch)
    - should_train (if should_train is True, then we just train the optimizer, else we evaluate)
    - ensemble_num the num of models used in ensembling phrase
    - pth_list the parameter path list
    """
    #print('-----------------start----------------------')
    optimizer_list = []
    for i in range(ensemble_num):
        optimizer = w(optimizer_network())
        optimizer.load_state_dict(torch.load(pth_list[i], map_location='cuda:0'))
        optimizer.eval()
        optimizer_list.append(optimizer)

    if should_train:
        pass
        #optimizer_network.train()
    else:  
        unroll = 1
    torch.manual_seed(seed)
    optimizee_obj_function2 = optimizee_obj_function(training=should_train)
    #print(optimizee_obj_function2.y)
    optimizee = w(optimizee_network())
    ensemble_strategy = ensemble_strategy(beta = ensemble_beta, ensemble_num = ensemble_num)
    # Counting the parameters of the optimizee
    n_params = 0
    for param in optimizee.parameters():
        n_params += int(np.prod(param.size()))
    hidden_list = []
    cell_list = []
    for i in range(ensemble_num):    
        hidden_states = [w(Variable(torch.zeros(n_params, optimizer_list[i].hidden_size))) for _ in range(2)]
        cell_states = [w(Variable(torch.zeros(n_params, optimizer_list[i].hidden_size))) for _ in range(2)]
        hidden_list.append(hidden_states)
        cell_list.append(cell_states)
    losses_list = []
    hidden_list2 = []
    cell_list2 = []
    for i in range(ensemble_num):    
        hidden_states2 = [w(Variable(torch.zeros(n_params, optimizer_list[i].hidden_size))) for _ in range(2)]
        cell_states2 = [w(Variable(torch.zeros(n_params, optimizer_list[i].hidden_size))) for _ in range(2)]
        hidden_list2.append(hidden_states2)
        cell_list2.append(cell_states2)
    #ensemble方法类实例化
    
    if should_train:
        meta_optimizer.zero_grad()
    for iteration in range(iterations_to_optimize):
        print('iteration',iteration)
        # The loss of the current iteration
        current_loss = optimizee(optimizee_obj_function2)
        
        # Since the objective function of the optimizer is equal to the sum of the optimizee's losses
        # we want to measure the loss of every iteration and add it to the total sum of losses
        # Here dloss/dx is computed for every parameter x that has requires_grad = True
        # These are accumulated into x.grad for every parameter x
        # This is equal to x.grad += dloss/dx
        # We get the optimizee's gradients but we also retain the graph because
        # we need to run backpropagation again when we optimize the optimizer
        current_loss.backward(retain_graph = False)
        ensemble_strategy.set_last_loss(current_loss.detach().cpu().numpy())
        losses_list.append(current_loss.detach().cpu().numpy())
        offset = 0
        param_list = []
        for i in range(ensemble_num):
            result_params = {}
            param_list.append(result_params)
        with torch.no_grad():
            for name, param in optimizee.all_named_parameters():
                current_size = int(np.prod(param.size()))
                # We want to disconnect the gradients of some variables but not all, each time.
                # We do this in order to disconnect the gradients of the offset:offset+current_size
                # parameters but still get the gradients of the rest.
                gradients = detach_var(param.grad.view(current_size, 1))
                #print('gradients',gradients)
                # Call the optimizer and compute the new parameters
                update_list = []
                for i in range(ensemble_num):
                    updates, new_hidden, new_cell = optimizer_list[i](
                        gradients,
                        [h[offset:offset+current_size] for h in hidden_list[i]],
                        [c[offset:offset+current_size] for c in cell_list[i]]
                    )
                    #update_list.append(updates)
                    param_list[i][name] = param + updates.view(*param.size()) * out_mul
                # Here we replace the old parameters with the new values
                    for j in range(len(new_hidden)):
                        hidden_list2[i][j][offset:offset+current_size] = new_hidden[j]
                        cell_list2[i][j][offset:offset+current_size] = new_cell[j]
#                         print(new_hidden[j])
#                         print(hidden_list2[i][j])
                #updates_final = detach_var(ensemble_strategy(update_list))
                #result_params[name] = param + updates_final.view(*param.size()) * out_mul
                #result_params[name].retain_grad()
                offset += current_size
            final_result_params = ensemble_strategy.look_next(param_list, optimizee_obj_function2, optimizee_network)
                #print(final_result_params)
            optimizee = w(optimizee_network(**{k: detach_var(v) for k, v in final_result_params.items()}))
            #optimizee = w(optimizee_network(**final_result_params))
            hidden_list = hidden_list2[:]
            cell_list = cell_list2[:] 
    #print(losses_list) 
    return losses_list

def fit_ensemble_avg(optimizer_network, meta_optimizer, optimizee_obj_function, optimizee_network, iterations_to_optimize, iterations_to_unroll, out_mul, 
        seed, should_train = False,  ensemble_num = 3, pth_list = []):
    """
    Arguments: 
    - optimizer_network (the optimizer network we use, here the LSTM, Class instead of instance)
    - meta_optimizer (the optimizer of the optimizer network, e.g. Adam, SGD + nesterov, RMSprop, etc.)
    - optimizee_obj_function (the optimizee's objective function)
    - optimizee_network (the optimizee network)
    - epochs (total epochs for training)
    - iterations_to_optimize (iterations in every epoch)
    - should_train (if should_train is True, then we just train the optimizer, else we evaluate)
    - ensemble_num the num of models used in ensembling phrase
    - pth_list the parameter path list
    """
    #print('-----------------start----------------------')
    optimizer_list = []
    for i in range(ensemble_num):
        optimizer = w(optimizer_network())
        optimizer.load_state_dict(torch.load(pth_list[i], map_location='cuda:0'))
        optimizer.eval()
        optimizer_list.append(optimizer)
    if should_train:
        pass
        #optimizer_network.train()
    else:  
        unroll = 1
    torch.manual_seed(seed)
    optimizee_obj_function2 = optimizee_obj_function(training=should_train)
    #print(optimizee_obj_function2.y)
    optimizee = w(optimizee_network())
    # Counting the parameters of the optimizee
    n_params = 0
    for param in optimizee.parameters():
        n_params += int(np.prod(param.size()))
    hidden_list = []
    cell_list = []
    for i in range(ensemble_num):    
        hidden_states = [w(Variable(torch.zeros(n_params, optimizer_list[i].hidden_size))) for _ in range(2)]
        cell_states = [w(Variable(torch.zeros(n_params, optimizer_list[i].hidden_size))) for _ in range(2)]
        hidden_list.append(hidden_states)
        cell_list.append(cell_states)
    
    losses_list = []
    #ensemble方法类实例化
    if should_train:
        meta_optimizer.zero_grad()
    for iteration in range(iterations_to_optimize):
        # The loss of the current iteration
        current_loss = optimizee(optimizee_obj_function2)
        # Since the objective function of the optimizer is equal to the sum of the optimizee's losses
        # we want to measure the loss of every iteration and add it to the total sum of losses
        # Here dloss/dx is computed for every parameter x that has requires_grad = True
        # These are accumulated into x.grad for every parameter x
        # This is equal to x.grad += dloss/dx
        # We get the optimizee's gradients but we also retain the graph because
        # we need to run backpropagation again when we optimize the optimizer
        current_loss.backward(retain_graph = False)
        losses_list.append(current_loss.data.cpu().numpy()
        offset = 0
        param_list = []
        for i in range(ensemble_num):
            result_params = {}
            param_list.append(result_params)
        # These will be the new parameters. We will update all the parameters, cell and hidden states
        # by iterating through the optimizee's "all_named parameters"
        hidden_list2 = []
        cell_list2 = []
        for i in range(ensemble_num):    
            hidden_states2 = [w(Variable(torch.zeros(n_params, optimizer_list[i].hidden_size))) for _ in range(2)]
            cell_states2 = [w(Variable(torch.zeros(n_params, optimizer_list[i].hidden_size))) for _ in range(2)]
            hidden_list2.append(hidden_states2)
            cell_list2.append(cell_states2)
        with torch.no_grad():
            for name, param in optimizee.all_named_parameters():
                current_size = int(np.prod(param.size()))
                #print(param)
                # We want to disconnect the gradients of some variables but not all, each time.
                # We do this in order to disconnect the gradients of the offset:offset+current_size
                # parameters but still get the gradients of the rest.
                gradients = detach_var(param.grad.view(current_size, 1))
                #print('gradients',gradients)
                # Call the optimizer and compute the new parameters
                update_list = []
                for i in range(ensemble_num):
                    updates, new_hidden, new_cell = optimizer_list[i](
                        gradients,
                        [h[offset:offset+current_size] for h in hidden_list[i]],
                        [c[offset:offset+current_size] for c in cell_list[i]]
                    )
                    update_list.append(updates)
                    for j in range(len(new_hidden)):
                        hidden_list2[i][j][offset:offset+current_size] = new_hidden[j]
                        cell_list2[i][j][offset:offset+current_size] = new_cell[j]
                updates_final = average_all(update_list)
                # Here we replace the old parameters with the new values
                #updates_final = detach_var(ensemble_strategy(update_list))
                result_params[name] = param + updates_final.view(*param.size()) * out_mul
                offset += current_size
            optimizee = w(optimizee_network(**{k: detach_var(v) for k, v in result_params.items()}))
            hidden_list = hidden_list2
            cell_list = cell_list2
    return losses_list