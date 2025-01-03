import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List
from types import SimpleNamespace

from Utils.file import *
from Manipulation.manipulation import Manipulation


def manipulate_data(cfg: SimpleNamespace,
                    transformer_path: str,
                    transformed_data_path: str,
                    transformer: Manipulation,
                    data: pd.DataFrame,
                    feature_type: dict):
    if cfg.manipulation.load:
        if os.path.exists(transformer_path):
            print('Transformer found. Load transformer.')
            transformer = transformer.load(transformer_path)

            if os.path.exists(transformed_data_path):
                print('Transformed data found. Load transformed data.')
                transformed_data = load_pkl(transformed_data_path)
            else:
                print('Transformed data not found. Create new transformed data.')
                transformed_data = transformer.transform(data)
                if cfg.manipulation.save:
                    save_pkl(transformed_data, transformed_data_path)
        else:
            print('Transformer not found. Create new transformer.')
            transformer.fit(data,
                            numerical_transform=cfg.manipulation.numerical_transform,
                            numerical_transform_feature_range=cfg.manipulation.feature_range,
                            feature_type=feature_type)
            transformed_data = transformer.transform(data)
            if cfg.manipulation.save:
                save_pkl(transformer, transformer_path)
                save_pkl(transformed_data, transformed_data_path)

    else:
        print('Load flag is False. Create new transformer and transformed data.')
        transformer.fit(data,
                        numerical_transform=cfg.manipulation.numerical_transform,
                        numerical_transform_feature_range=cfg.manipulation.feature_range,
                        feature_type=feature_type)
        transformed_data = transformer.transform(data)
        if cfg.manipulation.save:
            save_pkl(transformer, transformer_path)
            save_pkl(transformed_data, transformed_data_path)

    return transformer, transformed_data


def split_data_feature_type(data: pd.DataFrame,
                            feature_type: dict,
                            transformer: Manipulation):
    cat_data, cat_cols = [], []
    num_data, num_cols = [], []
    cat_mask_data, num_mask_data = [], []

    st = 0
    data_manipulation_info_list = transformer._data_manipulation_info_list
    for data_manipulation_info in data_manipulation_info_list:
        dim = data_manipulation_info.output_dimensions
        _data = data[:, st:st + dim]

        if '_mask' not in data_manipulation_info.column_name:
            _key = data_manipulation_info.column_name
            if feature_type[_key] == 'Numerical':
                num_data.append(_data)
                num_cols.append(data_manipulation_info.column_name)
            elif feature_type[_key] in ['Categorical', 'Binary']:
                cat_data.append(_data)
                cat_cols.append(data_manipulation_info.column_name)
            else:
                raise ValueError('Invalid type')
        else:
            _key = data_manipulation_info.column_name.replace('_mask', '')
            if feature_type[_key] == 'Numerical':
                num_mask_data.append(_data)
            elif feature_type[_key] in ['Categorical', 'Binary']:
                cat_mask_data.append(_data)
            else:
                raise ValueError('Invalid type')

        st += dim

    cat_data = np.concatenate(cat_data, axis=1)
    num_data = np.concatenate(num_data, axis=1)
    cat_mask_data = np.concatenate(cat_mask_data, axis=1)
    num_mask_data = np.concatenate(num_mask_data, axis=1)

    return cat_data, num_data, cat_mask_data, num_mask_data, cat_cols, num_cols
