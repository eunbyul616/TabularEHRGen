import torch
import torch.nn as nn

from Utils.model import set_activation


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
        super(Discriminator, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation

        dim = input_dim
        seq = []
        for item in list(hidden_dims):
            seq += [nn.Linear(dim, item), set_activation(activation), nn.Dropout(0.5)]
            dim = item

        seq.append(nn.Linear(dim, output_dim))
        self.layers = nn.Sequential(*seq)

    def forward(self, x):
        out = self.layers(x)

        return out
