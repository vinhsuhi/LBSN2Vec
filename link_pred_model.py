import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from random import shuffle, randint
import torch.nn.functional as F
from itertools import combinations, combinations_with_replacement
from sklearn.metrics import f1_score, accuracy_score


class StructMLP(nn.Module):
    def __init__(self, input_rep, num_neurons):
        """
        input_rep: the dimension of input embedding
        num_neurons: hidden dimension of StructMLP
        """
        super(StructMLP, self).__init__()
        #Deepsets MLP
        self.num_neurons = num_neurons
        self.input_rep = input_rep
        self.ds_layer_1 = nn.Linear(input_rep, num_neurons)
        self.ds_layer_2 = nn.Linear(num_neurons, num_neurons)
        self.rho_layer_1 = nn.Linear(num_neurons, num_neurons)
        self.rho_layer_2 = nn.Linear(num_neurons, num_neurons)

        #One Hidden Layer
        self.layer1 = nn.Linear(num_neurons, num_neurons)
        self.layer2 = nn.Linear(num_neurons, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.device = torch.device("cuda")

    def forward(self, input_tensor, samples):
        """
        TODO: what is input_tensor
        TODO: what is samples
        """
        import pdb
        pdb.set_trace()
        #Deepsets initially on each of the samples
        sum_tensor = torch.zeros(samples.shape[0], self.num_neurons).to(self.device)
        print("1")
        print(sum_tensor)
        for i in range(input_tensor.shape[0]):
            #Process the input tensor to form n choose k combinations and create a zero tensor
            set_init_rep = input_tensor[i].view(-1, self.input_rep)
            x = self.ds_layer_1(set_init_rep)
            x = self.relu(x)
            x = self.ds_layer_2(x)
            x = x[samples]
            x = torch.sum(x, dim=1)
            import pdb
            pdb.set_trace()
            x = self.rho_layer_1(x)
            sum_tensor += x

        x = sum_tensor / input_tensor.shape[0]

        #One Hidden Layer for predictor
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        # x = self.sigmoid(x)
        return x

    def compute_loss(self, input_tensor, samples, target):
        """
        input_tensor: 
        samples: 
        target: label of edges and non-edges 
        """
        pred = self.forward(input_tensor, samples)
        return F.cross_entropy(pred, target)
