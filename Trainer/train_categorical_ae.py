import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List

import torch
from torch.utils.data import DataLoader

from Utils.train import save_ckpt
from Visualization.barplot import countplot_categorical_feature


class CategoricalAETrainer:
    def __init__(self, config, rank, data_type, **kwargs):
        self.cfg = config

        self.rank = rank
        self.device = torch.device(f'cuda:{self.rank}')

        self.model_name = self.cfg.train.model_name
        self.model_save_path = self.cfg.path.ckpt_path
        self.save_name = self.cfg.log.time

        self.model = kwargs['model']
        self.loss_fn = kwargs['loss_fn']
        self.optimizer = kwargs['optimizer']

        self.train_dataset = kwargs['train_dataset']
        self.valid_dataset = kwargs['validation_dataset']
        self.test_dataset = kwargs['test_dataset']

        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=self.cfg.dataloader.batch_size,
                                       pin_memory=self.cfg.dataloader.pin_memory,
                                       sampler=kwargs['train_sampler'] if 'train_sampler' in kwargs else None,
                                       generator=None,
                                       collate_fn=kwargs['collate_fn'] if 'collate_fn' in kwargs else None,
                                       drop_last=self.cfg.dataloader.drop_last)
        self.valid_loader = DataLoader(dataset=self.valid_dataset,
                                       batch_size=self.cfg.dataloader.batch_size,
                                       pin_memory=self.cfg.dataloader.pin_memory,
                                       sampler=kwargs['valid_sampler'] if 'valid_sampler' in kwargs else None,
                                       generator=None,
                                       collate_fn=kwargs['collate_fn'] if 'collate_fn' in kwargs else None,
                                       drop_last=self.cfg.dataloader.drop_last)
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=self.cfg.dataloader.batch_size,
                                      pin_memory=self.cfg.dataloader.pin_memory,
                                      sampler=kwargs['test_sampler'] if 'test_sampler' in kwargs else None,
                                      generator=None,
                                      collate_fn=kwargs['collate_fn'] if 'collate_fn' in kwargs else None,
                                      drop_last=self.cfg.dataloader.drop_last)

        self.start_epoch = 0
        self.total_epochs = self.cfg.train.general.num_epochs

        self.loss_keys = self.cfg.train.general.keys
        self.train_loss = {key: [] for key in self.loss_keys}
        self.validation_loss = {key: [] for key in self.loss_keys}
        self.test_loss = {key: [] for key in self.loss_keys}

        self.logit_threshold = kwargs['logit_threshold'] if 'logit_threshold' in kwargs else 0.5
        self.data_type = data_type

    def run_epochs(self):
        for epoch in range(self.start_epoch, self.total_epochs):
            self.model.train()
            self.train_one_epoch(epoch)
            self.model.eval()
            self.validate_one_epoch(epoch)

            if epoch % self.cfg.train.general.eval_freq == 0:
                self.eval_one_epoch(epoch)

            save_ckpt(
                cfg=self.cfg,
                epoch=epoch,
                validation_loss=self.validation_loss['Total_Loss'],
                states={'epoch': epoch,
                        'arch': self.cfg.log.time,
                        'model': self.model,
                        'optimizer': self.optimizer.state_dict()}
            )

    def _set_iterator(self, loader, epoch, mode='Train'):
        if (self.rank == 0) or (self.rank == self.cfg.device_num):
            return tqdm(loader, desc=f'{mode} Epoch: {epoch}', dynamic_ncols=True)
        else:
            return tqdm(loader, disable=True)

    def _set_iterator_postfix(self, iterator, loss):
        iterator.set_postfix(loss=loss / (iterator.n + 1),
                             lr=self.optimizer.param_groups[0]['lr'])

    def _calculate_loss(self, x, x_hat):
        loss = 0
        s_idx = 0

        for i in range(len(x_hat)):
            dim = x_hat[i].shape[-1]
            loss += self.loss_fn(x_hat[i], x[:, s_idx:s_idx + dim])
            s_idx += dim

        return loss

    def train_one_epoch(self, epoch):
        total_losses = {key: 0.0 for key in self.loss_keys}

        train_iterator = self._set_iterator(self.train_loader, epoch, mode='Train')

        for batch in train_iterator:
            self.optimizer.zero_grad()

            x = batch[0] if self.data_type == 'static' else batch[1]
            x = x.to(self.device)

            rep, x_hat = self.model(x)

            loss = self._calculate_loss(x, x_hat)
            total_losses['Total_Loss'] += loss.item()
            loss.backward()
            self.optimizer.step()

            self._set_iterator_postfix(train_iterator, total_losses['Total_Loss'])

        for key in self.loss_keys:
            self.train_loss[key].append(total_losses[key] / len(self.train_loader))

    def validate_one_epoch(self, epoch):
        total_losses = {key: 0.0 for key in self.loss_keys}

        validation_iterator = self._set_iterator(self.valid_loader, epoch, mode='val')

        with torch.no_grad():
            for batch in validation_iterator:
                x = batch[0] if self.data_type == 'static' else batch[1]
                x = x.to(self.device)
                rep, x_hat = self.model(x)

                loss = self._calculate_loss(x, x_hat)
                total_losses['Total_Loss'] += loss.item()
                self._set_iterator_postfix(validation_iterator, total_losses['Total_Loss'])

            for key in self.loss_keys:
                self.validation_loss[key].append(total_losses[key] / len(self.valid_loader))

    def eval_one_epoch(self, epoch):
        total_losses = {key: 0.0 for key in self.loss_keys}
        test_iterator = self._set_iterator(self.test_loader, epoch, mode='test')

        transformer = self.test_dataset.static_transformer if self.data_type == 'static' else self.test_dataset.temporal_transformer
        categorical_feature_info = [info for info in transformer._data_manipulation_info_list
                                    if (info.column_type in ['Categorical', 'Binary']) and
                                    ('_mask' not in info.column_name)]

        if not os.path.exists(self.cfg.path.plot_file_path):
            os.makedirs(self.cfg.path.plot_file_path, exist_ok=True)

        with torch.no_grad():
            data = []
            data_hat = []
            for batch in test_iterator:
                x = batch[0] if self.data_type == 'static' else batch[1]
                x = x.to(self.device)
                rep, x_hat = self.model(x)

                loss = self._calculate_loss(x, x_hat)
                total_losses['Total_Loss'] += loss.item()
                self._set_iterator_postfix(test_iterator, total_losses['Total_Loss'])

                if self.data_type == 'static':
                    act_x_hat = [self._apply_activation_fn(
                        categorical_feature_info[i],
                        x_hat[i]
                    ) for i in range(len(x_hat))]
                    act_x_hat = torch.concatenate(act_x_hat, dim=-1)
                elif self.data_type == 'temporal':
                    batch_size, seq_len, feature_dim = x.size()
                    act_x_hat = [self._apply_activation_fn(
                        categorical_feature_info[i],
                        x_hat[i].view(batch_size, seq_len, -1)
                    ) for i in range(len(x_hat))]
                    act_x_hat = torch.concatenate(act_x_hat, dim=-1).view(-1, seq_len, feature_dim)
                else:
                    raise ValueError(f'Invalid data type: {self.data_type}')

                data.append(x)
                data_hat.append(act_x_hat)
            data = torch.concatenate(data, dim=0)
            data_hat = torch.concatenate(data_hat, dim=0)
            self._evaluate(transformer, categorical_feature_info, data, data_hat, epoch)

            for key in self.loss_keys:
                self.test_loss[key].append(total_losses[key] / len(self.test_loader))

    def _apply_activation_fn(self, feature_info, x):
        if feature_info.column_type == 'Binary':
            return (torch.sigmoid(x) >= self.logit_threshold).float()
        elif feature_info.column_type == 'Categorical':
            return torch.softmax(x, dim=-1)
        else:
            return x

    def _evaluate(self, transformer, feature_info, x, x_hat, epoch):
        x = transformer.inverse_transform(x.detach().cpu().numpy(), feature_info)
        x_hat = transformer.inverse_transform(x_hat.detach().cpu().numpy(), feature_info)

        for col in x.columns:
            countplot_categorical_feature(data1=x,
                                          data2=x_hat,
                                          col=col,
                                          stat='percent',
                                          label1='Real',
                                          label2='Reconstructed',
                                          title=col,
                                          save_path=self.cfg.path.plot_file_path + f'countplot_{col}_epoch_{epoch}.png')
