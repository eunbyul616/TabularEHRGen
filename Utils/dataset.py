import pandas as pd
import numpy as np
import h5py


def load_data(data_path: str, hdf_key: str, mode: str='train'):
    data = pd.read_hdf(data_path, key=f'{hdf_key}_{mode}')
    feature_type = load_feature_type(data_path, hdf_key, mode)

    return data, feature_type


def load_feature_type(data_path: str, hdf_key: str, mode: str='train'):
    feature_type = dict()
    with h5py.File(data_path, 'r') as f:
        dataset = f[f'{hdf_key}_{mode}']

        for attr_name, attr_value in dataset.attrs.items():
            if attr_name.startswith('feature_type_'):
                feature_type[attr_name.replace('feature_type_', '')] = attr_value

    return feature_type


def load_column_dtypes(data_path: str, hdf_key: str, mode: str='train'):
    column_dtype = dict()
    with h5py.File(data_path, 'r') as f:
        dataset = f[f'{hdf_key}_{mode}']

        for attr_name, attr_value in dataset.attrs.items():
            if attr_name.startswith('column_dtype_'):
                column_dtype[attr_name.replace('column_dtype_', '')] = attr_value

    return column_dtype


def explore_data_structure(name, obj):
    if hasattr(obj, "attrs"):
        print(f"Path: {name}")

        for attr_name, attr_value in obj.attrs.items():
            if attr_name.startswith('dtype_'):
                print(f"  - Attribute: {attr_name} = {attr_value}")
