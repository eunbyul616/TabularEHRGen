import os
import os.path
from types import SimpleNamespace
import yaml
from typing import Any, List
from copy import deepcopy
from omegaconf import OmegaConf


def _load_yaml(cfg_path):
    with open(cfg_path, 'r') as f:
        return _dict_to_namespace(yaml.safe_load(f))


def _dict_to_namespace(d):
    return SimpleNamespace(**{k: _dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()})


def _namespace_to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: _namespace_to_dict(v) for k, v in ns.__dict__.items()}
    elif isinstance(ns, list):
        return [_namespace_to_dict(v) for v in ns]
    else:
        return ns


def set_cfg(cfg: SimpleNamespace, key: str, value: Any):
    keys = key.split('.')

    for k in keys[:-1]:
        cfg = getattr(cfg, k)

    setattr(cfg, keys[-1], value)


def save_config(cfg: SimpleNamespace):
    save_path = cfg.path.base_config_file_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Saving config to {save_path}")
    with open(save_path, 'w') as f:
        OmegaConf.save(config=_namespace_to_dict(cfg), f=f)

