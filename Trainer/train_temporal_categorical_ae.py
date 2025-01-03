import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List

import torch
from torch.utils.data import DataLoader

from Utils.namespace import set_cfg
from Utils.train import save_ckpt
from Visualization.barplot import countplot_categorical_feature
from Trainer.train_categorical_ae import CategoricalAETrainer


class TemporalCategoricalAETrainer(CategoricalAETrainer):
    def __init__(self, config, rank, **kwargs):
        super(TemporalCategoricalAETrainer, self).__init__(config=config, rank=rank, data_type='temporal', **kwargs)

    def _calculate_loss(self, x, x_hat):
        batch_size, seq_len, feature_dim = x.size()
        pad_mask = torch.sum(x == self.cfg.dataloader.pad_value, dim=-1) == 0.

        loss = 0
        s_idx = 0

        for i in range(len(x_hat)):
            _x_hat = x_hat[i].view(batch_size, seq_len, -1)
            dim = _x_hat.shape[-1]
            loss += self.loss_fn(_x_hat, x[:, :, s_idx:s_idx + dim], mask=pad_mask)
            s_idx += dim

        return loss

    def _evaluate(self, transformer, feature_info, x, x_hat, epoch):
        batch_size, seq_len, feature_dim = x.size()

        pad_mask = x != self.cfg.dataloader.pad_value
        pad_mask = pad_mask.view(-1, feature_dim)
        pad_mask = pad_mask[:, 0].view(-1, 1).repeat(1, feature_dim)
        pad_mask = pad_mask.detach().cpu().numpy()

        x = transformer.inverse_transform(x.view(-1, feature_dim).detach().cpu().numpy(), feature_info)
        x_hat = transformer.inverse_transform(x_hat.view(-1, feature_dim).detach().cpu().numpy(), feature_info)
        x = pd.DataFrame(np.where(pad_mask, x, np.nan), columns=x.columns)
        x_hat = pd.DataFrame(np.where(pad_mask, x_hat, np.nan), columns=x_hat.columns)

        for col in x.columns:
            countplot_categorical_feature(data1=x,
                                          data2=x_hat,
                                          col=col,
                                          stat='percent',
                                          label1='Real',
                                          label2='Reconstructed',
                                          title=col,
                                          save_path=self.cfg.path.plot_file_path + f'countplot_{col}_epoch_{epoch}.png')


def temporal_categorical_ae_trainer_main(dataset: str, cols: List[str]=None):
    import config_manager
    from Utils.reproducibility import lock_seed
    from Utils.train import set_loss_fn, set_optimizer

    from Modules.EHR_Safe.TemporalCategoricalAutoEncoder import build_model

    if dataset == 'INSPIRE':
        from Datasets.dataset_inspire import INSPIREDataset as CustomDataset
    else:
        raise ValueError(f'Invalid dataset: {dataset}')

    config_manager.load_config()
    cfg = config_manager.config

    lock_seed(seed=cfg.seed)

    train_dataset = CustomDataset(cfg=cfg, mode='train', static_cols=cols)
    validation_dataset = CustomDataset(cfg=cfg, mode='val', static_cols=cols)
    test_dataset = CustomDataset(cfg=cfg, mode='test', static_cols=cols)

    # update model config following the dataset
    categorical_feature_info = [info for info in train_dataset.temporal_transformer._data_manipulation_info_list
                                if (info.column_type in ['Categorical', 'Binary']) and ('_mask' not in info.column_name)]
    categorical_feature_out_dims = [info.output_dimensions for info in categorical_feature_info]
    set_cfg(cfg, 'model.temporal_categorical_autoencoder.decoder.output_dims', categorical_feature_out_dims)
    model = build_model(cfg.model.temporal_categorical_autoencoder,
                        device=torch.device(f'cuda:{cfg.device_num}'))

    loss_fn = set_loss_fn(cfg.train.loss)
    optimizer = set_optimizer(model.parameters(), cfg.train.optimizer)

    trainer = TemporalCategoricalAETrainer(config=cfg,
                                           rank=cfg.device_num,
                                           model=model,
                                           collate_fn=None,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           train_dataset=train_dataset,
                                           validation_dataset=validation_dataset,
                                           test_dataset=test_dataset)
    trainer.run_epochs()


if __name__ == '__main__':
    dataset = 'INSPIRE'
    temporal_categorical_ae_trainer_main(dataset=dataset)