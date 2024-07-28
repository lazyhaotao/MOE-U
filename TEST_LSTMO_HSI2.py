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
from utils import  w, detach_var
from optimizer2 import OptimizerNetwork
from utils import HSIUtil
from model import *
from tools import *
from parameters import args
torch.manual_seed(args.random_seed)
# target dataset name
####################
DATASET = 'PaviaC'##
####################
channel_dict = {"PaviaU":[103, 9], "PaviaC":[102, 9], "Salinas":[204, 16],
                "SalinasA":[204, 6], "KSC":[176, 13], "IndianPines":[200, 16]}
###############################################################################
# GenericNeuralNetforxxx xxx is DATASET name
class GeneralFullyCTest(GenericNeuralNetforPaviaC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
class Data_Pack():
    def __init__(self):
        self.data = 0
        self.label = 0
    def get_sample(self, data, label):
        self.data = data
        self.label = label
    def sample(self):
        return self.data, self.label
def fit(optimizer_network, meta_optimizer, optimizee_obj_function, optimizee_network,
        iterations_to_optimize, iterations_to_unroll, out_mul, n_tests=10, SAMPLE_SIZE=10,
        should_train = False):
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
    loss_all = []
    datapack = Data_Pack()
    for run in range(n_tests):
        optimizee_obj = optimizee_obj_function(DATASET=DATASET, BATCH_SIZE=32, TRAINING_OPTIMIZER=False, run=run)
        optimizee = w(optimizee_network())
        optimizee.train()
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
            current_loss, _ = optimizee(optimizee_obj)
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
                #print(result_params[name].detach().cpu().numpy())
                result_params[name].retain_grad()
                offset += current_size
            # If we have reached the number of iterations needed to update the optimizer
            # we run backprop on the optimizer network
            # we are now in testing. We must set: iterations_to_unroll > iterations_to_optimize
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
                #print(optimizee.all_named_parameters())
                hidden_states = [detach_var(v) for v in hidden_states2]
                cell_states = [detach_var(v) for v in cell_states2]
            else:
                # Otherwise, we just create the next optimizee objective funtion
                optimizee = w(optimizee_network(**result_params))
                hidden_states = hidden_states2
                cell_states = cell_states2
        # Training the optimizee has been finished
        # Then, Testing
        loss_all.append(losses_list)
        optimizee.eval()
        img, gt, LABEL_VALUES, _, _, _ = get_dataset(DATASET, target_folder="../Data/Datasets/")
        train_gt, test_gt = get_sample(DATASET, SAMPLE_SIZE, run)
        indices_test = np.nonzero(test_gt)
        prdict_labels = np.zeros_like(gt)
        data = np.array(np.copy(img),dtype=np.float32)
        data = data.reshape((img.shape[0]*img.shape[1],img.shape[2],1,1))
        label = np.asarray(np.copy(gt), dtype='int64')
        label = label.reshape((label.shape[0]*label.shape[1]))
        label[label!=0] = label[label!=0] -1
        datapack.get_sample(torch.from_numpy(data), torch.from_numpy(label))
        _, out = optimizee(datapack)
        #print(out.shape)
        out = out.argmax(dim=1).detach().cpu().numpy()
        #print(out.shape)
        out = out.reshape((img.shape[0],img.shape[1])) + 1
        prdict_labels = out
        #print(out.shape)
        acc = np.sum(out[test_gt!=0] == test_gt[test_gt!=0])/np.sum(test_gt!=0)
        print("accuracy:", acc)
        np.save(path + "prejected_labels_sample"+str(SAMPLE_SIZE)+"_run_" + str(run) +"_acc_"+str(acc)+".npy", prdict_labels)
    loss_all = np.array(loss_all)
    np.save(path + "sample"+str(SAMPLE_SIZE)+"loss_.npy", loss_all)

if __name__ == '__main__':
    USE_CUDA = True
    LSTMOptimizer = w(OptimizerNetwork())
    # Fill in the address of the LSTM optimizer weights file
    LSTMOptimizer.load_state_dict(torch.load(' '))
    # Experimental results storage address
    path = " "
    if not os.path.isdir(path):
        os.makedirs(path)
    fit(LSTMOptimizer, None, HSIUtil, GeneralFullyCTest, 50, 100, out_mul = 0.1, n_tests=10, should_train = False)
