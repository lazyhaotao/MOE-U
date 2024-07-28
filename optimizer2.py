import torch
import torch.nn as nn 
from torch.nn import Parameter
from torch.autograd import Variable
import numpy as np
from utils import w

class OptimizerNetwork(nn.Module):
    def __init__(self, preprocessing = True, hidden_size = 20, preprocessing_factor = 10.0):
        super(OptimizerNetwork, self).__init__()
        self.hidden_size = hidden_size
        if preprocessing:
            # Since we have the preprocessing flag enabled, we want the neural network
            # to have two arguments and not just the gradient (see the forward function)
            self.recurs = nn.LSTMCell(2, hidden_size)
        else:
            self.recurs = nn.LSTMCell(1, hidden_size)
        self.recurs2 = nn.LSTMCell(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.preprocessing = preprocessing
        self.preprocessing_factor = preprocessing_factor
        self.preprocessing_threshold = np.exp(-preprocessing_factor)
        
    def forward(self, inp, hidden, cell):
        if self.preprocessing:
            inp = inp.data
            inp2 = w(torch.zeros(inp.size()[0], 2))
            keep_grads = torch.abs(inp) >= self.preprocessing_threshold
            
            # If the absolute value is greater or equal than the preprocessing threshold
            # (see the condition in the first part of the gradient winged formula) we pass
            # the log of the absolute value of the gradient divided by the preprocessing factor
            # as the first parameter, and we pass the sign of the gradient as the second parameter.

            keep_grads_numpy = keep_grads.data.cpu().numpy()
            
            for i in range(0, keep_grads_numpy.shape[0]):
                
                if keep_grads_numpy[i][0] == 0:
                    inp2[i, 0] = -1
                    inp2[i, 1] = float(np.exp(self.preprocessing_factor)) * inp[i]
                else:
                    inp2[i, 0] = torch.log(torch.abs(inp[i]) + 1e-8) / self.preprocessing_factor
                    inp2[i, 1] = torch.sign(inp[i])

            inp = w(Variable(inp2))
        
        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        
        return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)