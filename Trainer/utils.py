import os
from pathlib import Path

import torch

from Utils.namespace import _load_yaml


def load_embedding_model(cfg, model_type):
    if model_type == 'static':
        from Modules.EHR_Safe import StaticCategoricalAutoEncoder as Model
        train_cfg = cfg.train.static_categorical_ae

    elif model_type == 'temporal':
        from Modules.EHR_Safe import TemporalCategoricalAutoEncoder as Model
        train_cfg = cfg.train.temporal_categorical_ae
    elif model_type == 'joint':
        from Modules.EHR_Safe import JointAutoEncoder as Model
        train_cfg = cfg.train.joint_ae
    else:
        raise ValueError(f'Invalid data_type: {model_type}')

    checkpoint_saved_root = '/'.join(Path(cfg.path.ckpt_path).parts[:-1])

    checkpoint_path = os.path.join(checkpoint_saved_root,
                                   train_cfg.name,
                                   train_cfg.checkpoint)
    config = _load_yaml(os.path.join(checkpoint_path, 'config.yaml'))
    if model_type == 'static':
        model_config = config.model.static_categorical_autoencoder
    elif model_type == 'temporal':
        model_config = config.model.temporal_categorical_autoencoder
    elif model_type == 'joint':
        model_config = config.model.joint_autoencoder
    else:
        raise ValueError(f'Invalid data_type: {model_type}')

    model = Model.build_model(model_config, device=torch.device(f'cuda:{cfg.device_num}'))
    model_checkpoint = torch.load(os.path.join(checkpoint_path, 'checkpoint_best.pth.tar'),
                                  map_location=f'cuda:{cfg.device_num}')
    model.load_state_dict(model_checkpoint['state_dict'])

    return model


def load_gan_model(cfg, model_name, checkpoint):
    from Models.GAN import GAN

    checkpoint_saved_root = '/'.join(Path(cfg.path.ckpt_path).parts[:-1])
    model_checkpoint_path = os.path.join(checkpoint_saved_root, model_name, checkpoint)
    cfg = _load_yaml(os.path.join(model_checkpoint_path, 'config.yaml'))

    model = GAN.build_model(model_config=cfg.model.gan, device=torch.device(f'cuda:{cfg.device_num}'))
    model_checkpoint = torch.load(os.path.join(model_checkpoint_path, 'checkpoint_best.pth.tar'),
                                  map_location=f'cuda:{cfg.device_num}')
    model.load_state_dict(model_checkpoint['state_dict'])

    return model



