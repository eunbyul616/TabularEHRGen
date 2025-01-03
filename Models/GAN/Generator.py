import torch.nn as nn

from Models.Common.ResidualBlock import ResidualBlock


class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
        super(Generator, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation

        dim = input_dim
        seq = []
        for item in list(hidden_dims):
            seq += [ResidualBlock(dim, item, activation)]
            dim = item
        seq += [nn.Linear(dim, output_dim), nn.Sigmoid()]
        self.layers = nn.Sequential(*seq)

    def forward(self, x):
        out = self.layers(x)

        return out
