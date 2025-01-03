import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

import torch

from Trainer.utils import load_embedding_model, load_gan_model


def apply_mask(data: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.where(mask, data, np.nan), columns=data.columns, index=data.index)


def generate_random_noise_vector(sample_size: int,
                                 dim: int,
                                 device: torch.device):
    mean = torch.zeros(sample_size, dim)
    std = torch.ones(sample_size, dim)
    z = torch.normal(mean=mean, std=std).type(torch.float32)

    return z.to(device)


def split_generated_data(x_hat, dims):
    out = []
    s_idx = 0
    for dim in dims:
        out.append(x_hat[:, s_idx:s_idx + dim])
        s_idx += dim

    return out


def apply_activation(x_hat: List[torch.Tensor] or torch.Tensor,
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


def get_feature_info(static_transformer, temporal_transformer, time_transformer):
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


def generate_synthetic_data(cfg, model_name, checkpoint, sample_size,
                            sn_dim, tn_dim, sm_dim, tm_dim,
                            static_transformer, temporal_transformer, time_transformer,
                            pad_mask, logit_threshold=0.5, device=torch.device('cpu')):
    (sc_feature_info, sn_feature_info, sm_feature_info,
     tc_feature_info, tn_feature_info, tm_feature_info,
     time_feature_info) = get_feature_info(static_transformer, temporal_transformer, time_transformer)

    # load static categorical autoencoder
    static_categorical_ae = load_embedding_model(cfg, model_type='static')
    # load temporal categorical autoencoder
    temporal_categorical_ae = load_embedding_model(cfg, model_type='temporal')
    # load joint categorical autoencoder
    joint_categorical_ae = load_embedding_model(cfg, model_type='joint')
    # load GAN
    gan = load_gan_model(cfg, model_name, checkpoint)

    static_categorical_ae.eval()
    temporal_categorical_ae.eval()
    joint_categorical_ae.eval()
    gan.eval()

    # generate synthetic data
    z_dim = gan.generator.input_dim
    z = generate_random_noise_vector(sample_size, z_dim, device)
    fake = gan.generator(z)

    # decode synthetic data
    decoded_fake = joint_categorical_ae.decoder(fake)
    sc_rep_dim = static_categorical_ae.encoder.embedding_dim
    tc_rep_dim = temporal_categorical_ae.encoder.embedding_dim

    seq_len = cfg.dataloader.seq_len
    tn_dim = seq_len * tn_dim
    tm_dim = seq_len * tm_dim
    time_dim = seq_len

    dims = [sc_rep_dim, sn_dim, tc_rep_dim, tn_dim, sm_dim, tm_dim, time_dim]
    out = split_generated_data(decoded_fake, dims)
    sc_rep_hat, sn_hat, tc_rep_hat, tn_hat, static_mask_hat, temporal_mask_hat, time_hat = out

    sc_hat = static_categorical_ae.decoder(sc_rep_hat)
    tc_hat = temporal_categorical_ae.decoder(tc_rep_hat)

    # apply activation function
    act_sc_hat = apply_activation(sc_hat, sc_feature_info, logit_threshold=logit_threshold)
    act_sn_hat = apply_activation(sn_hat, sn_feature_info, logit_threshold=logit_threshold)
    act_sm_hat = apply_activation(static_mask_hat, sm_feature_info, logit_threshold=logit_threshold)
    static_data_hat = torch.cat([act_sc_hat, act_sn_hat, act_sm_hat], dim=-1)

    tc_hat = [tc_hat[i].view(tc_hat[i].shape[0], seq_len, -1) for i in range(len(tc_hat))]
    act_tc_hat = apply_activation(tc_hat, tc_feature_info, logit_threshold=logit_threshold)
    tn_hat = tn_hat.view(sample_size, seq_len, -1)
    act_tn_hat = apply_activation(tn_hat, tn_feature_info, logit_threshold=logit_threshold)
    temporal_mask_hat = temporal_mask_hat.view(sample_size, seq_len, -1)
    act_tm_hat = apply_activation(temporal_mask_hat, tm_feature_info, logit_threshold=logit_threshold)
    temporal_data_hat = torch.cat([act_tc_hat, act_tn_hat, act_tm_hat], dim=-1)

    time_hat = time_hat.view(sample_size, seq_len, -1)
    act_time_hat = apply_activation(time_hat, time_feature_info)

    static_data_hat = static_data_hat.detach().cpu().numpy()
    temporal_data_hat = temporal_data_hat.detach().cpu().numpy()
    time_hat = act_time_hat.detach().cpu().numpy()

    # inverse transform
    static_feature_info = sc_feature_info + sn_feature_info + sm_feature_info
    static_data_hat = static_transformer.inverse_transform(static_data_hat, static_feature_info)

    # temporal data
    temporal_feature_info = tc_feature_info + tn_feature_info + tm_feature_info
    feature_dim = sum([info.output_dimensions for info in temporal_feature_info])
    temporal_data_hat = temporal_data_hat.reshape(-1, feature_dim)

    _pad_mask = np.expand_dims(pad_mask, axis=-1).repeat(feature_dim, axis=-1)
    _pad_mask = _pad_mask.reshape(-1, feature_dim)
    temporal_data_hat = np.where(_pad_mask, temporal_data_hat, np.nan)
    temporal_data_hat = temporal_transformer.inverse_transform(temporal_data_hat, temporal_feature_info)

    feature_dim =  1
    time_hat = time_hat.reshape(-1, feature_dim)
    _pad_mask = np.expand_dims(pad_mask, axis=-1).repeat(feature_dim, axis=-1)
    _pad_mask = _pad_mask.reshape(-1, feature_dim)
    time_hat = np.where(_pad_mask, time_hat, np.nan)
    time_hat = time_transformer.inverse_transform(time_hat, time_feature_info)

    temporal_data_hat = pd.concat([time_hat, temporal_data_hat], axis=1)

    return static_data_hat, temporal_data_hat


if __name__ == '__main__':
    import config_manager
    from Utils.reproducibility import lock_seed

    from Datasets.dataset_inspire import INSPIREDataset as CustomDataset

    config_manager.load_config()
    cfg = config_manager.config

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device_num)
    lock_seed(seed=cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path = cfg.path.csv_file_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    static_cols = None
    temporal_cols = None
    sample_size = cfg.sample_size

    test_dataset = CustomDataset(cfg=cfg, mode='test', static_cols=static_cols, temporal_cols=temporal_cols)
    static_transformer = test_dataset.static_transformer
    temporal_transformer = test_dataset.temporal_transformer
    time_transformer = test_dataset.time_transformer

    sc = test_dataset.sc_data
    sn = test_dataset.sn_data
    tc = test_dataset.tc_data
    tn = test_dataset.tn_data
    sm = torch.concatenate([test_dataset.sc_mask_data, test_dataset.sn_mask_data], dim=-1)
    tm = torch.concatenate([test_dataset.tc_mask_data, test_dataset.tn_mask_data], dim=-1)
    time_data = test_dataset.time_data

    temporal_real = torch.cat([tc, tn, tm], dim=-1)

    # sampling real data
    sample_idx = np.random.choice(len(temporal_real), sample_size, replace=False)
    temporal_real = temporal_real[sample_idx]
    pad_mask = torch.sum(temporal_real == cfg.dataloader.pad_value, dim=-1) == 0.
    pad_mask = pad_mask.detach().cpu().numpy()

    static_syn, temporal_syn = generate_synthetic_data(cfg=cfg,
                                                       model_name='GAN',
                                                       checkpoint='2025-01-03/15-31',
                                                       sample_size=sample_size,
                                                       sn_dim=sn.shape[-1],
                                                       tn_dim=tn.shape[-1],
                                                       sm_dim=sm.shape[-1],
                                                       tm_dim=tm.shape[-1],
                                                       static_transformer=static_transformer,
                                                       temporal_transformer=temporal_transformer,
                                                       time_transformer=time_transformer,
                                                       pad_mask=pad_mask,
                                                       device=device)
    static_syn.to_csv(os.path.join(save_path, 'synthetic_static_data.csv'), index=False)
    temporal_syn.to_csv(os.path.join(save_path, 'synthetic_temporal_data.csv'), index=False)

