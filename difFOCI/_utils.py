import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors


def convert_to_numeric(data: pd.DataFrame):
    data = data.copy(True)
    for c in data:
        if data[c].dtype not in ["int", "float"]:
            codes, _ = pd.factorize(data[c])
            if torch.min(codes) < 0:
                codes = codes.astype("float")
                codes[codes == -1] = torch.nan
            data[c] = codes

    return data

def validate_and_prepare_for_conditional_dependence(y, x):
    y_shape = y.shape
    x_shape = x.shape

    if not (y.ndim == 1 and y_shape[0] > 1):
        raise ValueError("y must be a 1D array with at least one sample")
    if not (1 <= x.ndim <= 2 and x_shape[0] >= 1):
        raise ValueError("x must be a 1D or 2D array with at least one sample")

    return y, x

def softmaxb(x, beta=1e-5):
    e_x = torch.exp((x - torch.max(x))/beta)
    return e_x / e_x.sum()


class KNeighbors:
    def __init__(self, k=2, device='cpu'):
        self.device = device

    def fit(self, X, beta=1e-5):
        output = torch.cdist(X, X)
        return torch.vmap(lambda x: softmaxb(x, beta))(-output-1e10*torch.eye(output.shape[0]).to(self.device))
        

class one_layer_net(torch.nn.Module):    
    # Constructor
    def __init__(self, input_size, hidden_neurons, output_size):
        super(one_layer_net, self).__init__()
        # hidden layer 
        self.linear_one = torch.nn.Linear(input_size, hidden_neurons)
        self.linear_two = torch.nn.Linear(hidden_neurons, output_size) 
        # defining layers as attributes
        self.layer_in = None
        self.act = None
        self.layer_out = None
    # prediction function
    def forward(self, x):
        self.layer_in = self.linear_one(x)
        self.act = torch.relu(self.layer_in)
        self.layer_out = self.linear_two(self.act)
        y_pred = torch.relu(self.linear_two(self.act))
        return y_pred
    
class two_layer_net(torch.nn.Module):
    # Constructor
    def __init__(self, input_size, hidden_neurons1, hidden_neurons2, output_size):
        super(two_layer_net, self).__init__()
        # First hidden layer
        self.linear_one = torch.nn.Linear(input_size, hidden_neurons1)
        # Second hidden layer
        self.linear_two = torch.nn.Linear(hidden_neurons1, hidden_neurons2)
        # Output layer
        self.linear_three = torch.nn.Linear(hidden_neurons2, output_size)
        
        # Defining layers as attributes
        self.layer_in = None
        self.act1 = None
        self.act2 = None
        self.layer_out = None
    # Prediction function
    def forward(self, x):
        self.layer_in = self.linear_one(x)
        self.act1 = torch.relu(self.layer_in)
        self.layer_in = self.linear_two(self.act1)
        self.act2 = torch.relu(self.layer_in)
        y_pred = self.linear_three(self.act2)
        return y_pred