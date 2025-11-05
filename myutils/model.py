"""
-*- coding:utf-8 -*-
@Time      :2025/11/1 下午4:07
@Author    :Chen Junpeng

"""
import logging
import os.path
import warnings
from typing import List
from .file_process import *
from torch import nn
from torch.nn import init
from .utils_waring import *


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, dropout, norm, activation, num_hidden_layers: int = None, hidden_size: int | List[int] = None, hidden_size_delay_rate: float = 0.5, logger: logging.Logger = None):
        super(FeedForwardNeuralNetwork, self).__init__()

        self.log = logger.info if logger else print

        if isinstance(hidden_size, list):
            warnings.warn(UtilsWarning('The hidden size list is given, parameter num_hidden_layers is ignored.'))

            layers = []

            for i, size in enumerate(hidden_size):
                if i == 0:
                    layers.append(nn.Linear(input_size, size))
                else:
                    layers.append(nn.Linear(hidden_size[i-1], size))

                self.initialize_weights(layers[-1], activation)
                self.add_norm_layer(norm, layers, size)
                self.add_activation(activation, layers)
                layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(hidden_size[-1], output_size))
            self.initialize_weights(layers[-1], None)

        elif isinstance(hidden_size, int):
            assert num_hidden_layers is not None, "num_hidden_layers should be specified when hidden_size is an integer"
            assert num_hidden_layers > 0, "num_hidden_layers should be greater than 0"
            assert hidden_size_delay_rate > 0, "hidden_size_delay_rate should be greater than 0"

            layers = []

            for i in range(num_hidden_layers):
                if i == 0:
                    layers.append(nn.Linear(input_size, hidden_size))
                    _size = hidden_size
                else:
                    now_size = max(int(_size * hidden_size_delay_rate), output_size)
                    layers.append(nn.Linear(_size, now_size))
                    _size = now_size

                self.initialize_weights(layers[-1], activation)
                self.add_norm_layer(norm, layers, _size)
                self.add_activation(activation, layers)
                layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(_size, output_size))
            self.initialize_weights(layers[-1], None)

        elif hidden_size is None:
            layers = [nn.Linear(input_size, output_size)]
            self.initialize_weights(layers[-1], None)

        else:
            raise ValueError(f"Invalid hidden size: {hidden_size}, with type of {type(hidden_size)}")

        self.network = nn.Sequential(*layers)

        self.log(self.network)

    def forward(self, x):
        return self.network(x)

    def add_activation(self, activation, layers):
        activation_dict = {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'elu': nn.ELU(),
            'prelu': nn.PReLU()
        }
        if activation not in activation_dict.keys():
            raise ValueError(f"Invalid activation type: {activation}, choices are {activation_dict.keys()}")
        layers.append(activation_dict[activation])

    def initialize_weights(self, module, activation):
        if isinstance(module, nn.Linear):
            if activation == 'relu' or activation is None:
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                self.log(f'Initialize {module} with ReLU')
            elif activation == 'leakyrelu':
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                self.log(f'Initialize {module} with LeakyReLU')
            else:
                init.xavier_normal_(module.weight)
                self.log(f'Initialize {module} with {activation}')
            if module.bias is not None:
                init.constant_(module.bias, 0)
                self.log(f'Initialize {module} bias with 0')
        else:
            self.log(f'Ignore {module} initialization')

    def add_norm_layer(self, norm, layers, hidden_size):
        norm_dict = {
            'layernorm': nn.LayerNorm,
            'batchnorm': nn.BatchNorm1d,
            'instancenorm': nn.InstanceNorm1d
        }
        if norm not in norm_dict.keys():
            raise ValueError(f"Invalid norm type: {norm}, choices are {norm_dict.keys()}")
        layers.append(norm_dict[norm](hidden_size))


