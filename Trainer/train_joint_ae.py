import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List

import torch
from torch.utils.data import DataLoader

from Utils.namespace import set_cfg
from Utils.train import save_ckpt
import Loss.reconstruction as reconstruction_loss

from Visualization.barplot import countplot_categorical_feature
from Visualization.distribution import vis_cdf

from Trainer.train import Trainer


class JointAETrainer(Trainer):
    def __init__(self, config, rank, **kwargs):
        super(JointAETrainer, self).__init__(config=config, rank=rank, **kwargs)

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

        self.static_categorical_ae = kwargs['static_categorical_ae']
        self.temporal_categorical_ae = kwargs['temporal_categorical_ae']

        self.temporal_loss_w = 1.0
        self.static_loss_w = 1.0
        self.mask_loss_w = 1.0
        self.time_loss_w = 0.1

        self.mse = reconstruction_loss.MSELoss()
        self.bce = reconstruction_loss.BCELoss()

    def run_epochs(self):
        # freeze the weights of the static and temporal categorical autoencoders
        for param in self.static_categorical_ae.parameters():
            param.requires_grad = False
        for param in self.temporal_categorical_ae.parameters():
            param.requires_grad = False

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

    def _calculate_loss(self, x, sc_rep_hat, tc_rep_hat, sn_hat, tn_hat, sm_hat, tm_hat, time_hat, pad_mask):
        batch_size = self.cfg.dataloader.batch_size
        seq_len = self.cfg.dataloader.seq_len

        s_idx = 0

        dim = sc_rep_hat.shape[-1]
        sc_rep = x[:, s_idx:s_idx + dim]
        s_idx += dim
        dim = sn_hat.shape[-1]
        sn = x[:, s_idx:s_idx + dim]
        s_idx += dim
        s_loss = self.mse(sc_rep_hat, sc_rep) + self.mse(torch.sigmoid(sn_hat), sn)

        _pad_mask = pad_mask.view(batch_size, seq_len, 1).repeat(1, 1, tn_hat.view(batch_size, seq_len, -1).size(-1)).view(batch_size, -1)
        dim = tc_rep_hat.shape[-1]
        tc_rep = x[:, s_idx:s_idx + dim]
        s_idx += dim
        dim = tn_hat.shape[-1]
        tn = x[:, s_idx:s_idx + dim]
        s_idx += dim
        t_loss = self.mse(tc_rep_hat, tc_rep) + self.mse(torch.sigmoid(tn_hat), tn, mask=_pad_mask)

        _pad_mask = pad_mask.view(batch_size, seq_len, 1).repeat(1, 1, tm_hat.view(batch_size, seq_len, -1).size(-1)).view(batch_size, -1)
        dim = sm_hat.shape[-1]
        sm = x[:, s_idx:s_idx + dim]
        s_idx += dim
        dim = tm_hat.shape[-1]
        tm = x[:, s_idx:s_idx + dim]
        s_idx += dim
        m_loss = self.bce(sm_hat, sm) + self.bce(tm_hat, tm, mask=_pad_mask)

        _pad_mask = pad_mask.view(batch_size, seq_len, 1).repeat(1, 1, time_hat.view(batch_size, seq_len, -1).size(-1)).view(batch_size, -1)
        dim = time_hat.shape[-1]
        time = x[:, s_idx:s_idx + dim]
        time_loss = self.mse(time_hat, time, mask=_pad_mask)

        loss = (self.static_loss_w * s_loss +
                self.temporal_loss_w * t_loss +
                self.mask_loss_w * m_loss +
                self.time_loss_w * time_loss)

        return loss, s_loss, t_loss, m_loss, time_loss

    def train_one_epoch(self, epoch):
        total_losses = {key: 0.0 for key in self.loss_keys}

        train_iterator = self._set_iterator(self.train_loader, epoch, mode='Train')

        for batch in train_iterator:
            self.optimizer.zero_grad()

            sc, tc, sn, tn, sm, tm, time = self._set_data(batch)

            # static categorical autoencoder
            sc_rep, sc_hat = self.static_categorical_ae(sc)
            # temporal categorical autoencoder
            tc_rep, tc_hat = self.temporal_categorical_ae(tc)

            x = torch.cat([sc_rep, sn, tc_rep, tn, sm, tm, time], dim=-1)
            # joint autoencoder
            rep, x_hat = self.model(x)

            sc_rep_hat, tc_rep_hat, sn_hat, tn_hat, sm_hat, tm_hat, time_hat = self._set_data_hat(x_hat, sn, tn,
                                                                                                  sm, tm, time)
            # calculate loss
            pad_mask = torch.sum(tc == self.cfg.dataloader.pad_value, dim=-1) == 0.
            loss, s_loss, t_loss, m_loss, time_loss = self._calculate_loss(x, sc_rep_hat, tc_rep_hat, sn_hat, tn_hat,
                                                                           sm_hat, tm_hat, time_hat, pad_mask)

            total_losses['Total_Loss'] += loss.item()
            total_losses['BCE_Loss'] += m_loss.item()
            total_losses['MSE_Loss'] += (s_loss.item() + t_loss.item() + time_loss.item())
            total_losses['Static_Loss'] += s_loss.item()
            total_losses['Temporal_Loss'] += t_loss.item()
            total_losses['Mask_Loss'] += m_loss.item()
            total_losses['Time_Loss'] += time_loss.item()

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
                sc, tc, sn, tn, sm, tm, time = self._set_data(batch)

                # static categorical autoencoder
                sc_rep, sc_hat = self.static_categorical_ae(sc)
                # temporal categorical autoencoder
                tc_rep, tc_hat = self.temporal_categorical_ae(tc)

                x = torch.cat([sc_rep, sn, tc_rep, tn, sm, tm, time], dim=-1)
                # joint autoencoder
                rep, x_hat = self.model(x)

                sc_rep_hat, tc_rep_hat, sn_hat, tn_hat, sm_hat, tm_hat, time_hat = self._set_data_hat(x_hat, sn, tn,
                                                                                                      sm, tm, time)
                # calculate loss
                pad_mask = torch.sum(tc == self.cfg.dataloader.pad_value, dim=-1) == 0.
                loss, s_loss, t_loss, m_loss, time_loss = self._calculate_loss(x, sc_rep_hat, tc_rep_hat, sn_hat,
                                                                               tn_hat,
                                                                               sm_hat, tm_hat, time_hat, pad_mask)

                total_losses['Total_Loss'] += loss.item()
                total_losses['BCE_Loss'] += m_loss.item()
                total_losses['MSE_Loss'] += (s_loss.item() + t_loss.item() + time_loss.item())
                total_losses['Static_Loss'] += s_loss.item()
                total_losses['Temporal_Loss'] += t_loss.item()
                total_losses['Mask_Loss'] += m_loss.item()
                total_losses['Time_Loss'] += time_loss.item()

                self._set_iterator_postfix(validation_iterator, total_losses['Total_Loss'])

            for key in self.loss_keys:
                self.validation_loss[key].append(total_losses[key] / len(self.valid_loader))

    def eval_one_epoch(self, epoch):
        total_losses = {key: 0.0 for key in self.loss_keys}
        test_iterator = self._set_iterator(self.test_loader, epoch, mode='test')

        static_transformer = self.test_dataset.static_transformer
        temporal_transformer = self.test_dataset.temporal_transformer
        time_transformer = self.test_dataset.time_transformer

        (sc_feature_info, sn_feature_info, sm_feature_info,
         tc_feature_info, tn_feature_info, tm_feature_info,
         time_feature_info) = self._get_feature_info(static_transformer, temporal_transformer, time_transformer)

        if not os.path.exists(self.cfg.path.plot_file_path):
            os.makedirs(self.cfg.path.plot_file_path, exist_ok=True)

        data = {'static_data': [], 'temporal_data': [], 'time_data': []}
        data_hat = {'static_data': [], 'temporal_data': [], 'time_data': []}

        with torch.no_grad():
            for batch in test_iterator:
                sc, tc, sn, tn, sm, tm, time = self._set_data(batch)

                # static categorical autoencoder
                sc_rep, sc_hat = self.static_categorical_ae(sc)
                # temporal categorical autoencoder
                tc_rep, tc_hat = self.temporal_categorical_ae(tc)

                x = torch.cat([sc_rep, sn, tc_rep, tn, sm, tm, time], dim=-1)
                # joint autoencoder
                rep, x_hat = self.model(x)

                sc_rep_hat, tc_rep_hat, sn_hat, tn_hat, sm_hat, tm_hat, time_hat = self._set_data_hat(x_hat, sn, tn,
                                                                                                      sm, tm, time)
                # calculate loss
                pad_mask = torch.sum(tc == self.cfg.dataloader.pad_value, dim=-1) == 0.
                loss, s_loss, t_loss, m_loss, time_loss = self._calculate_loss(x, sc_rep_hat, tc_rep_hat, sn_hat,
                                                                               tn_hat,
                                                                               sm_hat, tm_hat, time_hat, pad_mask)

                total_losses['Total_Loss'] += loss.item()
                total_losses['BCE_Loss'] += m_loss.item()
                total_losses['MSE_Loss'] += (s_loss.item() + t_loss.item() + time_loss.item())
                total_losses['Static_Loss'] += s_loss.item()
                total_losses['Temporal_Loss'] += t_loss.item()
                total_losses['Mask_Loss'] += m_loss.item()
                total_losses['Time_Loss'] += time_loss.item()

                self._set_iterator_postfix(test_iterator, total_losses['Total_Loss'])

                static_x = torch.cat([sc, sn, sm], dim=-1)
                data['static_data'].append(static_x)

                batch_size = self.cfg.dataloader.batch_size
                seq_len = self.cfg.dataloader.seq_len
                tn = tn.view(batch_size, seq_len, -1)
                tm = tm.view(batch_size, seq_len, -1)
                temporal_x = torch.cat([tc, tn, tm], dim=-1)
                data['temporal_data'].append(temporal_x)

                time = time.view(batch_size, seq_len, -1)
                data['time_data'].append(time)

                sc_hat = self.static_categorical_ae.decoder(sc_rep_hat)
                act_sc_hat = self._apply_activation_fn(sc_hat, sc_feature_info)
                act_sn_hat = self._apply_activation_fn(sn_hat, sn_feature_info)
                act_sm_hat = self._apply_activation_fn(sm_hat, sm_feature_info)
                static_x_hat = torch.cat([act_sc_hat, act_sn_hat, act_sm_hat], dim=-1)
                data_hat['static_data'].append(static_x_hat)

                tc_hat = self.temporal_categorical_ae.decoder(tc_rep_hat)
                tc_hat = [tc_hat[i].view(tc_hat[i].shape[0], seq_len, -1) for i in range(len(tc_hat))]
                tn_hat = tn_hat.view(batch_size, seq_len, -1)
                tm_hat = tm_hat.view(batch_size, seq_len, -1)

                act_tc_hat = self._apply_activation_fn(tc_hat, tc_feature_info)
                act_tn_hat = self._apply_activation_fn(tn_hat, tn_feature_info)
                act_tm_hat = self._apply_activation_fn(tm_hat, tm_feature_info)
                temporal_x_hat = torch.cat([act_tc_hat, act_tn_hat, act_tm_hat], dim=-1)
                data_hat['temporal_data'].append(temporal_x_hat)

                time_hat = time_hat.view(batch_size, seq_len, -1)
                act_time_hat = self._apply_activation_fn(time_hat, time_feature_info)
                data_hat['time_data'].append(act_time_hat)

            data['static_data'] = torch.concatenate(data['static_data'], dim=0)
            data['temporal_data'] = torch.concatenate(data['temporal_data'], dim=0)
            data['time_data'] = torch.concatenate(data['time_data'], dim=0)
            data_hat['static_data'] = torch.concatenate(data_hat['static_data'], dim=0)
            data_hat['temporal_data'] = torch.concatenate(data_hat['temporal_data'], dim=0)
            data_hat['time_data'] = torch.concatenate(data_hat['time_data'], dim=0)

            static_feature_info = sc_feature_info + sn_feature_info + sm_feature_info
            temporal_feature_info = tc_feature_info + tn_feature_info + tm_feature_info

            self._evaluate(static_transformer, temporal_transformer, time_transformer,
                           static_feature_info, temporal_feature_info, time_feature_info,
                           data, data_hat, epoch)

            for key in self.loss_keys:
                self.test_loss[key].append(total_losses[key] / len(self.test_loader))


def joint_ae_trainer_main(dataset: str, cols: List[str]=None):
    import config_manager
    from Utils.reproducibility import lock_seed
    from Utils.train import set_loss_fn, set_optimizer

    from Modules.EHR_Safe.JointAutoEncoder import build_model

    from Trainer.utils import load_embedding_model

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

    # load static categorical autoencoder
    static_categorical_ae = load_embedding_model(cfg, model_type='static')
    # load temporal categorical autoencoder
    temporal_categorical_ae = load_embedding_model(cfg, model_type='temporal')

    # set parameters for joint autoencoder
    sn_feature_dim = sum([info.output_dimensions
                          for info in train_dataset.static_transformer._data_manipulation_info_list
                          if (info.column_type == 'Numerical') or ('_mask' in info.column_name)])
    tn_feature_dim = sum([info.output_dimensions
                          for info in train_dataset.temporal_transformer._data_manipulation_info_list
                          if (info.column_type == 'Numerical') or ('_mask' in info.column_name)])

    static_feature_dim = static_categorical_ae.encoder.embedding_dim + sn_feature_dim
    temporal_feature_dim = temporal_categorical_ae.encoder.embedding_dim + tn_feature_dim * cfg.dataloader.seq_len
    time_feature_dim = 1 * cfg.dataloader.seq_len
    feature_dim = static_feature_dim + temporal_feature_dim + time_feature_dim
    set_cfg(cfg, 'model.joint_autoencoder.encoder.input_dim', feature_dim)
    set_cfg(cfg, 'model.joint_autoencoder.decoder.output_dim', feature_dim)

    model = build_model(cfg.model.joint_autoencoder, device=torch.device(f'cuda:{cfg.device_num}'))

    loss_fn = set_loss_fn(cfg.train.loss)
    optimizer = set_optimizer(model.parameters(), cfg.train.optimizer)

    trainer = JointAETrainer(config=cfg,
                             rank=cfg.device_num,
                             model=model,
                             loss_fn=loss_fn,
                             optimizer=optimizer,
                             train_dataset=train_dataset,
                             validation_dataset=validation_dataset,
                             test_dataset=test_dataset,
                             static_categorical_ae=static_categorical_ae,
                             temporal_categorical_ae=temporal_categorical_ae)

    trainer.run_epochs()


if __name__ == '__main__':
    dataset = 'INSPIRE'
    joint_ae_trainer_main(dataset=dataset)
