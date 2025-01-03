import os
import shutil
from types import SimpleNamespace

import torch

import Loss.reconstruction as reconstruction_loss
import Loss.generation as generation_loss
from Utils.namespace import save_config


def set_loss_fn(config):
    name = config.name

    if name == 'category':
        loss_fn = reconstruction_loss.CategoryLoss()
    elif name == 'wgan_discriminator':
        loss_fn = generation_loss.WGANDiscriminatorLoss()
    elif name == 'wgan_generator':
        loss_fn = generation_loss.WGANGeneratorLoss()
    else:
        raise ValueError("Invalid loss function")

    return loss_fn


def set_optimizer(model_params, config):
    name = config.name
    lr = float(config.lr)

    if name == 'adam':
        optimizer = torch.optim.Adam(model_params, lr=lr)
    elif name == 'adamw':
        optimizer = torch.optim.AdamW(model_params,
                                      lr=lr,
                                      weight_decay=float(config.weight_decay),
                                      betas=(config.beta1, config.beta2))
    else:
        raise ValueError("Invalid optimizer")

    return optimizer


def save_ckpt(cfg: SimpleNamespace,
              epoch: int,
              validation_loss: list,
              states: dict) -> None:
    def _select_save_condition():
        try:
            save_condition = getattr(cfg.metrics, cfg.train.loss.name).is_better
        except AttributeError:
            print(f"Cannot find '{cfg.train.loss.name}' in metrics\n Using 'last' condition for saving checkpoint")
            save_condition = 'last'

        if save_condition == 'lower':
            return validation_loss[-1] == min(validation_loss)
        elif save_condition == 'higher':
            return validation_loss[-1] == max(validation_loss)
        elif save_condition == 'last':
            return validation_loss[-2] > validation_loss[-1]

    if epoch != 0:
        target_path = cfg.path.ckpt_config_file_path
        if not os.path.exists(target_path):
            os.makedirs(target_path, exist_ok=True)

        save_config(cfg)
        # copy config.yaml from hydra output directory to model_save_path
        shutil.copyfile(cfg.path.base_config_file_path,
                        os.path.join(target_path, 'config.yaml'))

        # handle DataParallel model save case
        if hasattr(states['model'], 'module'):
            states['state_dict'] = states['model'].module.state_dict()
        else:
            states['state_dict'] = states['model'].state_dict()

        # move model state_dict to cpu before saving
        states['state_dict'] = {k: v.cpu() for k, v in states['state_dict'].items()}

        # save current state of model
        print(f"Saving current model checkpoint")
        current_ckpt_path = os.path.join(target_path, 'checkpoint.pth.tar')
        best_ckpt_path = os.path.join(target_path, 'checkpoint_best.pth.tar')

        torch.save(states, current_ckpt_path)

        if _select_save_condition():
            print(f"*** Saving best model  at epoch {epoch} ***")
            shutil.copyfile(current_ckpt_path, best_ckpt_path)


