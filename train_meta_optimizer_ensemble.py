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
from utils import  w, detach_var
from optimizer2 import OptimizerNetwork
from utils import HSIUtil

from model import GenericNeuralNetforPaviaC
from parameters import args
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
torch.manual_seed(args.random_seed)
def fit(optimizer_network, meta_optimizer, optimizee_obj_function, optimizee_network,
        iterations_to_optimize, iterations_to_unroll, out_mul,
        should_train = True):
    """
    Arguments: 
    - optimizer_network (the optimizer network we use, here the LSTM)
    - meta_optimizer (the optimizer of the optimizer network, e.g. Adam, SGD + nesterov, RMSprop, etc.)
    - optimizee_obj_function (the optimizee's objective function)
    - optimizee_network (the optimizee network)
    - epochs (total epochs for training)
    - iterations_to_optimize (iterations in every epoch)
    - should_train (if should_train is True, then we just train the optimizer, else we evaluate)
    """
    if should_train:
        optimizer_network.train()
    else:
        optimizer_network.eval()
        unroll = 1
    ############################################################################################
    optimizee_obj_function = optimizee_obj_function(DATASET='PaviaC')
    optimizee = w(optimizee_network())
    # Counting the parameters of the optimizee
    n_params = 0
    for param in optimizee.parameters():
        n_params += int(np.prod(param.size()))
    hidden_states = [w(Variable(torch.zeros(n_params, optimizer_network.hidden_size))) for _ in range(2)]
    cell_states = [w(Variable(torch.zeros(n_params, optimizer_network.hidden_size))) for _ in range(2)]
    losses_list = []
    if should_train:
        meta_optimizer.zero_grad()
    total_losses = None
    for iteration in range(iterations_to_optimize):
        # The loss of the current iteration
        current_loss = optimizee(optimizee_obj_function)
        # Since the objective function of the optimizer is equal to the sum of the optimizee's losses
        # we want to measure the loss of every iteration and add it to the total sum of losses
        if total_losses is None:
            total_losses = current_loss
        else:
            total_losses += current_loss
        losses_list.append(current_loss.data.cpu().numpy())
        # Here dloss/dx is computed for every parameter x that has requires_grad = True
        # These are accumulated into x.grad for every parameter x
        # This is equal to x.grad += dloss/dx
        # We get the optimizee's gradients but we also retain the graph because
        # we need to run backpropagation again when we optimize the optimizer
        current_loss.backward(retain_graph = True)
        offset = 0
        result_params = {}
        # These will be the new parameters. We will update all the parameters, cell and hidden states
        # by iterating through the optimizee's "all_named parameters"
        hidden_states2 = [w(Variable(torch.zeros(n_params, optimizer_network.hidden_size))) for _ in range(2)]
        cell_states2 = [w(Variable(torch.zeros(n_params, optimizer_network.hidden_size))) for _ in range(2)]
        for name, param in optimizee.all_named_parameters():
            current_size = int(np.prod(param.size()))
            # We want to disconnect the gradients of some variables but not all, each time.
            # We do this in order to disconnect the gradients of the offset:offset+current_size
            # parameters but still get the gradients of the rest.
            gradients = detach_var(param.grad.view(current_size, 1))
            # Call the optimizer and compute the new parameters
            updates, new_hidden, new_cell = optimizer_network(
                gradients,
                [h[offset:offset+current_size] for h in hidden_states],
                [c[offset:offset+current_size] for c in cell_states]
            )
            # Here we replace the old parameters with the new values
            for i in range(len(new_hidden)):
                hidden_states2[i][offset:offset+current_size] = new_hidden[i]
                cell_states2[i][offset:offset+current_size] = new_cell[i]
            result_params[name] = param + updates.view(*param.size()) * out_mul
            result_params[name].retain_grad()
            offset += current_size
        # If we have reached the number of iterations needed to update the optimizer
        # we run backprop on the optimizer network
        if iteration % iterations_to_unroll == 0:
            if should_train:
                # zero_grad() clears the gradients of all optimized tensors
                meta_optimizer.zero_grad()
                # we compute the gradient of the total losses  (i.e. the optimizer's loss function)
                # with respect to the optimizer's parameters
                total_losses.backward()
                # we finally perform the optimization step, i.e. the updates
                meta_optimizer.step()
            # Since we did the update on the optimizer network
            # we overwrite the total_losses
            total_losses = None
            # Here we detach the state variables because they are not propagated
            # to the graph (see Figure 2 of the paper for details)
            optimizee = w(optimizee_network(**{k: detach_var(v) for k, v in result_params.items()}))
            hidden_states = [detach_var(v) for v in hidden_states2]
            cell_states = [detach_var(v) for v in cell_states2]
        else:
            # Otherwise, we just create the next optimizee objective funtion
            optimizee = w(optimizee_network(**result_params))
            hidden_states = hidden_states2
            cell_states = cell_states2     
    return losses_list
def main_loop(OptimizerNetwork, optimizee_obj_function, optimizee_network, preprocessing = True,
        epochs = 500, iterations_to_optimize = 100, iterations_to_unroll = 20,
        n_tests = 5, lr = 0.001, out_mul = 0.1, T_0=100, T_mult=1, lr_min = 1e-5):
    
    optimizer_network = w(OptimizerNetwork(preprocessing = preprocessing))
    # To construct an Optimizer you need to give it an iterable containing the parameters to optimize
    meta_optimizer = optim.Adam(optimizer_network.parameters(), lr = lr)
    scheduler = CosineAnnealingWarmRestarts(meta_optimizer, T_0=T_0, T_mult=T_mult)
    # Initialize dummy variables for the best_net object and the best_loss
    for eop_epo in range(epochs):
        loss_list = fit(optimizer_network, meta_optimizer, optimizee_obj_function, optimizee_network,
            iterations_to_optimize, iterations_to_unroll, out_mul,
            should_train = True)
        cur_loss = np.mean(np.sum(loss_list))
        print(cur_loss)
        #print(eop_epo,str(datetime.datetime.now()))
        lr_present = meta_optimizer.param_groups[0]['lr']
        scheduler.step(eop_epo+1)
        #meta_optimizer.param_groups[0]['lr'] = lr - meta_optimizer.param_groups[0]['lr']
        lr_next = meta_optimizer.param_groups[0]['lr']
        #学习率突然升高标志
        if (lr_next - lr_present)>0.5*lr:
            #####################################################################################################################################
            torch.save(optimizer_network.state_dict(), '../EnsembleOptimizerPth/Ensemble_LSTMOptimizer_PaviaC_lr'+str(lr)+'episode'+str(eop_epo)+'.pth')
    ##最后再保存一次模型###############################################################################################################################
    torch.save(optimizer_network.state_dict(), '../EnsembleOptimizerPth/Ensemble_LSTMOptimizer_PaviaC_lr'+str(lr)+'episode'+str(eop_epo)+'.pth')

if __name__ == '__main__':
    USE_CUDA  = True
    ##########################################################################################################################################
    for lr in [0.01]:
        main_loop(OptimizerNetwork, HSIUtil, GenericNeuralNetforPaviaC, lr = lr, preprocessing=True, n_tests=5, epochs=700, out_mul = 0.1,T_0=100, T_mult=1)
