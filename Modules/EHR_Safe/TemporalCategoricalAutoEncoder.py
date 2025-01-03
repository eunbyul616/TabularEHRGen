import torch
import torch.nn as nn

from Models.AutoEncoder import Encoder, MultiHeadDecoder
from Utils.namespace import _namespace_to_dict


class TemporalCategoricalAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(TemporalCategoricalAutoencoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        batch_size, seq_len, n_features = x.size()
        # flatten
        x = x.view(-1, seq_len * n_features)
        rep = self.encoder(x)
        x_hat = self.decoder(rep)

        return rep, x_hat


def build_model(model_config, device=torch.device('cpu')):
    print("Building Temporal Categorical Autoencoder model")
    encoder = Encoder(**_namespace_to_dict(model_config.encoder))
    decoder = MultiHeadDecoder(**_namespace_to_dict(model_config.decoder))

    return TemporalCategoricalAutoencoder(encoder=encoder, decoder=decoder).to(device)
