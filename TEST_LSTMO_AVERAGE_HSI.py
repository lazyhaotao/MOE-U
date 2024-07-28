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
# the averate method
def average_all(update_list):
    return torch.mean(torch.cat(update_list, dim=1),dim=1).unsqueeze(1)

def fit_ensemble_test(optimizer_network, meta_optimizer, optimizee_obj_function, optimizee_network,
                      iterations_to_optimize, iterations_to_unroll, out_mul,
                      seed, n_tests=10, SAMPLE_SIZE=10, should_train = False, ensemble_num=3, pth_list=[]):
    optimizer_list = []
    for i in range(ensemble_num):
        optimizer = w(optimizer_network())
        optimizer.load_state_dict(torch.load(pth_list[i], map_location='cuda:0'))
        optimizer.eval()
        optimizer_list.append(optimizer)

    if should_train:
        pass
        # optimizer_network.train()
    else:
        iterations_to_unroll = 1
    torch.manual_seed(seed)
    loss_all = []
    datapack = Data_Pack()
    for run in range(n_tests):
        optimizee_obj = optimizee_obj_function(DATASET=DATASET, BATCH_SIZE=32, TRAINING_OPTIMIZER=False, run=run)
        # print(optimizee_obj_function2.y)
        optimizee = w(optimizee_network())
        optimizee.train()
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
        if should_train:
            meta_optimizer.zero_grad()
        for iteration in range(iterations_to_optimize):
            current_loss, _ = optimizee(optimizee_obj)
            current_loss.backward(retain_graph=False)

            losses_list.append(current_loss.detach().cpu().numpy())
            offset = 0
            param_list = []
            for i in range(ensemble_num):
                result_params = {}
                param_list.append(result_params)
            with torch.no_grad():
                for name, param in optimizee.all_named_parameters():
                    current_size = int(np.prod(param.size()))
                    gradients = detach_var(param.grad.view(current_size, 1))
                    update_list = []
                    for i in range(ensemble_num):
                        updates, new_hidden, new_cell = optimizer_list[i](
                            gradients,
                            [h[offset:offset + current_size] for h in hidden_list[i]],
                            [c[offset:offset + current_size] for c in cell_list[i]]
                        )
                        update_list.append(updates)
                        # param_list[i][name] = param + updates.view(*param.size()) * out_mul
                        # Here we replace the old parameters with the new values
                        for j in range(len(new_hidden)):
                            hidden_list2[i][j][offset:offset + current_size] = new_hidden[j]
                            cell_list2[i][j][offset:offset + current_size] = new_cell[j]
                    updates_final = average_all(update_list)
                    result_params[name] = param + updates_final.view(*param.size()) * out_mul
                    # result_params[name].retain_grad()
                    offset += current_size
                optimizee = w(optimizee_network(**{k: detach_var(v) for k, v in result_params.items()}))
                # optimizee = w(optimizee_network(**final_result_params))
                hidden_list = hidden_list2[:]
                cell_list = cell_list2[:]
        loss_all.append(losses_list)
        optimizee.eval()
        img, gt, LABEL_VALUES, _, _, _ = get_dataset(DATASET, target_folder="../Data/Datasets/")
        train_gt, test_gt = get_sample(DATASET, SAMPLE_SIZE, run)
        indices_test = np.nonzero(test_gt)
        prdict_labels = np.zeros_like(gt)
        data = np.array(np.copy(img), dtype=np.float32)
        data = data.reshape((img.shape[0] * img.shape[1], img.shape[2], 1, 1))
        label = np.asarray(np.copy(gt), dtype='int64')
        label = label.reshape((label.shape[0] * label.shape[1]))
        label[label != 0] = label[label != 0] - 1
        datapack.get_sample(torch.from_numpy(data), torch.from_numpy(label))
        _, out = optimizee(datapack)
        # print(out.shape)
        out = out.argmax(dim=1).detach().cpu().numpy()
        # print(out.shape)
        out = out.reshape((img.shape[0], img.shape[1])) + 1
        prdict_labels = out
        # print(out.shape)
        acc = np.sum(out[test_gt != 0] == test_gt[test_gt != 0]) / np.sum(test_gt != 0)
        print("accuracy:", acc)
        np.save(path + "prejected_labels_sample" + str(SAMPLE_SIZE) + "_run_" + str(run) + "_acc_" + str(acc) + ".npy", prdict_labels)
    loss_all = np.array(loss_all)
    np.save(path + "sample" + str(SAMPLE_SIZE) + "loss_.npy", loss_all)


if __name__ == '__main__':
    USE_CUDA  = True
    # Paths of all meta-optimizers in the ensemble
    ensemble_pth = [
                    ' ',
                    ' ',
                    ' '
                    ]
    # Experimental results storage address
    path = " "
    if not os.path.isdir(path):
        os.makedirs(path)
    fit_ensemble_test(OptimizerNetwork, None, HSIUtil, GeneralFullyCTest, 50, 1, out_mul = 0.1,
                      seed = args.random_seed, should_train = False, n_tests=10,
                      ensemble_num=len(ensemble_pth), pth_list=ensemble_pth)
