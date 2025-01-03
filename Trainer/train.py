import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List

import torch
from torch.utils.data import DataLoader

from Visualization.barplot import countplot_categorical_feature
from Visualization.distribution import vis_cdf


class Trainer:
    def __init__(self, config, rank, **kwargs):
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

    def _set_iterator(self, loader, epoch, mode='Train'):
        if (self.rank == 0) or (self.rank == self.cfg.device_num):
            return tqdm(loader, desc=f'{mode} Epoch: {epoch}', dynamic_ncols=True)
        else:
            return tqdm(loader, disable=True)

    def _set_iterator_postfix(self, iterator, loss):
        iterator.set_postfix(loss=loss / (iterator.n + 1),
                             lr=self.optimizer.param_groups[0]['lr'])

    def _set_data(self, batch):
        sc, tc, sn, tn, sm, tm, time = batch
        sc = sc.to(self.device)
        tc = tc.to(self.device)
        sn = sn.to(self.device)
        tn = tn.to(self.device)
        sm = sm.to(self.device)
        tm = tm.to(self.device)
        time = time.to(self.device)

        batch_size, seq_len, _ = tc.size()
        tn = tn.view(batch_size, -1)
        tm = tm.view(batch_size, -1)
        time = time.view(batch_size, -1)

        return sc, tc, sn, tn, sm, tm, time

    def _set_data_hat(self, x_hat, sn, tn, sm, tm, time):
        sn_dim = sn.size(-1)
        tn_dim = tn.size(-1)
        sm_dim = sm.size(-1)
        tm_dim = tm.size(-1)
        time_dim = time.size(-1)

        s_idx = 0
        rep_dim = self.static_categorical_ae.encoder.embedding_dim
        sc_rep_hat = x_hat[:, s_idx:s_idx + rep_dim]
        s_idx += rep_dim
        sn_hat = x_hat[:, s_idx:s_idx + sn_dim]
        s_idx += sn_dim

        rep_dim = self.temporal_categorical_ae.encoder.embedding_dim
        tc_rep_hat = x_hat[:, s_idx:s_idx + rep_dim]
        s_idx += rep_dim
        tn_hat = x_hat[:, s_idx:s_idx + tn_dim]
        s_idx += tn_dim

        sm_hat = x_hat[:, s_idx:s_idx + sm_dim]
        s_idx += sm_dim
        tm_hat = x_hat[:, s_idx:s_idx + tm_dim]
        s_idx += tm_dim

        time_hat = x_hat[:, s_idx:s_idx + time_dim]

        return sc_rep_hat, tc_rep_hat, sn_hat, tn_hat, sm_hat, tm_hat, time_hat

    def _get_feature_info(self, static_transformer, temporal_transformer, time_transformer):
        sc_feature_info = [info for info in static_transformer._data_manipulation_info_list
                           if (info.column_type in ['Categorical', 'Binary']) and
                           ('_mask' not in info.column_name)]
        sn_feature_info = [info for info in static_transformer._data_manipulation_info_list
                           if info.column_type == 'Numerical']
        sm_feature_info = []
        mask_info = [info for info in static_transformer._data_manipulation_info_list if
                     '_mask' in info.column_name]
        for s_info in (sc_feature_info + sn_feature_info):
            for info in mask_info:
                if f'{s_info.column_name}_mask' == info.column_name:
                    sm_feature_info.append(info)
                    break

        tc_feature_info = [info for info in temporal_transformer._data_manipulation_info_list
                           if (info.column_type in ['Categorical', 'Binary']) and ('_mask' not in info.column_name)]
        tn_feature_info = [info for info in temporal_transformer._data_manipulation_info_list
                           if info.column_type == 'Numerical']
        tm_feature_info = []
        mask_info = [info for info in temporal_transformer._data_manipulation_info_list
                     if '_mask' in info.column_name]
        for tc_info in (tc_feature_info + tn_feature_info):
            for info in mask_info:
                if f'{tc_info.column_name}_mask' == info.column_name:
                    tm_feature_info.append(info)
                    break

        time_feature_info = [info for info in time_transformer._data_manipulation_info_list]

        return (sc_feature_info, sn_feature_info, sm_feature_info,
                tc_feature_info, tn_feature_info, tm_feature_info,
                time_feature_info)

    def _apply_activation_fn(self,
                             x_hat: List[torch.Tensor] or torch.Tensor,
                             feature_info: list,
                             logit_threshold: float = 0.5) -> torch.Tensor:
        if isinstance(x_hat, list):
            act_x_hat = []
            for i in range(len(x_hat)):
                if feature_info[i].column_type == 'Binary':
                    _act_x_hat = torch.sigmoid(x_hat[i])
                    _act_x_hat = (_act_x_hat >= logit_threshold).float()
                    act_x_hat.append(_act_x_hat)
                elif feature_info[i].column_type == 'Categorical':
                    act_x_hat.append(torch.softmax(x_hat[i], dim=-1))
                else:
                    act_x_hat.append(torch.sigmoid(x_hat[i]))
            act_x_hat = torch.concatenate(act_x_hat, dim=-1)
        else:
            if feature_info[0].column_type == 'Binary':
                act_x_hat = torch.sigmoid(x_hat)
                act_x_hat = (act_x_hat >= logit_threshold).float()
            elif feature_info[0].column_type == 'Categorical':
                act_x_hat = torch.softmax(x_hat, dim=-1)
            else:
                act_x_hat = torch.sigmoid(x_hat)

        return act_x_hat

    def _inverse_transform(self,
                           real: np.array,
                           synthetic: np.array,
                           transformer,
                           feature_info=None,
                           mask: np.array=None) -> (pd.DataFrame, pd.DataFrame):
        if mask is None:
            real = transformer.inverse_transform(real, feature_info)
            synthetic = transformer.inverse_transform(synthetic, feature_info)
        else:
            real = np.where(mask, real, np.nan)
            synthetic = np.where(mask, synthetic, np.nan)

            real = transformer.inverse_transform(real, feature_info)
            synthetic = transformer.inverse_transform(synthetic, feature_info)

            # real = pd.DataFrame(np.where(mask, real, np.nan), columns=real.columns)
            # synthetic = pd.DataFrame(np.where(mask, synthetic, np.nan), columns=synthetic.columns)

        return real, synthetic

    def _evaluate(self, static_transformer, temporal_transformer, time_transformer,
                  static_feature_info, temporal_feature_info, time_feature_info,
                  data, data_hat, epoch):

        static_data = data['static_data'].detach().cpu().numpy()
        static_data_hat = data_hat['static_data'].detach().cpu().numpy()
        static_data, static_data_hat = self._inverse_transform(real=static_data,
                                                               synthetic=static_data_hat,
                                                               transformer=static_transformer,
                                                               feature_info=static_feature_info)

        mask = data['temporal_data'] != self.cfg.dataloader.pad_value
        mask = mask.view(-1, data['temporal_data'].shape[-1])
        mask = mask[:, 0].view(-1, 1).repeat(1, data['temporal_data'].shape[-1])
        mask = mask.detach().cpu().numpy()

        temporal_data = data['temporal_data'].view(-1, data['temporal_data'].shape[-1]).detach().cpu().numpy()
        temporal_data_hat = data_hat['temporal_data'].view(-1, data_hat['temporal_data'].shape[-1]).detach().cpu().numpy()
        temporal_data, temporal_data_hat = self._inverse_transform(real=temporal_data,
                                                                   synthetic=temporal_data_hat,
                                                                   transformer=temporal_transformer,
                                                                   feature_info=temporal_feature_info, mask=mask)

        mask = data['time_data'] != self.cfg.dataloader.pad_value
        mask = mask.view(-1, data['time_data'].shape[-1]).detach().cpu().numpy()

        time_data = data['time_data'].view(-1, data['time_data'].shape[-1]).detach().cpu().numpy()
        time_data_hat = data_hat['time_data'].view(-1, data_hat['time_data'].shape[-1]).detach().cpu().numpy()
        time_data, time_data_hat = self._inverse_transform(real=time_data,
                                                           synthetic=time_data_hat,
                                                           transformer=time_transformer,
                                                           feature_info=time_feature_info, mask=mask)

        # static features
        self._eval_numerical_features(cols=self.test_dataset.sn_cols, real=static_data, synthetic=static_data_hat,
                                      labels=['Real', 'Reconstructed'],
                                      epoch=epoch, save_path=self.cfg.path.plot_file_path)
        self._eval_categorical_features(cols=self.test_dataset.sc_cols, real=static_data, synthetic=static_data_hat,
                                        labels=['Real', 'Reconstructed'],
                                        epoch=epoch, save_path=self.cfg.path.plot_file_path)
        # temporal features
        self._eval_numerical_features(cols=self.test_dataset.tn_cols, real=temporal_data, synthetic=temporal_data_hat,
                                      labels=['Real', 'Reconstructed'],
                                      epoch=epoch, save_path=self.cfg.path.plot_file_path)
        self._eval_categorical_features(cols=self.test_dataset.tc_cols, real=temporal_data, synthetic=temporal_data_hat,
                                        labels=['Real', 'Reconstructed'],
                                        epoch=epoch, save_path=self.cfg.path.plot_file_path)
        # time features
        self._eval_numerical_features(cols=self.cfg.data.time_cols, real=time_data, synthetic=time_data_hat,
                                      labels=['Real', 'Reconstructed'],
                                      epoch=epoch, save_path=self.cfg.path.plot_file_path)

    def _eval_numerical_features(self,
                                 cols: List[str],
                                 real: pd.DataFrame,
                                 synthetic: pd.DataFrame,
                                 labels: List[str] = ['Real', 'Synthetic'],
                                 save_path: str = None,
                                 epoch: int = None) -> None:
        for col in cols:
            fname = f'cdf_{col}_epoch_{epoch}.png' if epoch is not None else f'cdf_{col}.png'
            vis_cdf(data=[real[col], synthetic[col]],
                    label=labels,
                    title=col,
                    save_path=os.path.join(save_path, fname) if save_path is not None else None)

    def _eval_categorical_features(self,
                                   cols: List[str],
                                   real: pd.DataFrame,
                                   synthetic: pd.DataFrame,
                                   labels: List[str] = ['Real', 'Synthetic'],
                                   stat: str = 'percent',
                                   save_path: str = None,
                                   epoch: int = None) -> None:
        for col in cols:
            fname = f'countplot_{col}_epoch_{epoch}.png' if epoch is not None else f'countplot_{col}.png'
            countplot_categorical_feature(data1=real,
                                          data2=synthetic,
                                          col=col,
                                          stat=stat,
                                          label1=labels[0],
                                          label2=labels[1],
                                          title=col,
                                          save_path=os.path.join(save_path, fname) if save_path is not None else None)