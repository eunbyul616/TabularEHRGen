import os
import pandas as pd
import numpy as np
from collections import namedtuple
from typing import List, Tuple
from tqdm import tqdm
import pickle

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from Utils.normalization import StochasticNormalization, MinMaxNormalization


DataManipulationInfo = namedtuple('DataManipulationInfo', [
    'column_name', 'column_type', 'transform', 'output_dimensions', 'activation_fn', 'unique_categories'
])


class Manipulation:
    def __init__(self,
                 verbose: bool = True,
                 numerical_activation_fn: str='sigmoid',
                 categorical_activation_fn: str='softmax',
                 binary_activation_fn: str='sigmoid',
                 min_max_values: dict=None,
                 drop_first: bool=True):

        self.verbose = verbose
        self.numerical_activation_fn = numerical_activation_fn
        self.categorical_activation_fn = categorical_activation_fn
        self.binary_activation_fn = binary_activation_fn
        self.min_max_values = min_max_values
        self.drop_first = drop_first

    def _fit_categorical(self, data: pd.DataFrame):
        col_name = data.columns[0]
        unique_values = data[col_name].unique()
        if self.drop_first and (len(unique_values) > 1):
            ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            ohe.fit(data)
            num_components = len(ohe.categories_[0]) - 1
        else:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe.fit(data)
            num_components = len(ohe.categories_[0])

        num_categories = len(ohe.categories_[0])
        unique_categories = ohe.categories_[0]

        return DataManipulationInfo(
            column_name=col_name,
            column_type='Categorical',
            transform=ohe,
            output_dimensions=num_components,
            activation_fn=self.categorical_activation_fn,
            unique_categories=unique_categories
        )

    def _fit_binary(self, data: pd.DataFrame):
        col_name = data.columns[0]
        unique_values = data[col_name].unique()

        if len(unique_values) > 1:
            ohe = OneHotEncoder(drop='if_binary', sparse_output=False)
            ohe.fit(data)
            num_components = len(ohe.categories_[0]) - 1
        else:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe.fit(data)
            num_components = len(ohe.categories_[0])

        num_categories = len(ohe.categories_[0])
        unique_categories = ohe.categories_[0]

        return DataManipulationInfo(
            column_name=col_name,
            column_type='Binary',
            transform=ohe,
            output_dimensions=num_components,
            activation_fn=self.binary_activation_fn,
            unique_categories=unique_categories
        )

    def _fit_numerical(self, data: pd.DataFrame):
        col_name = data.columns[0]

        if self.numerical_activation_fn == 'minmax':
            norm = MinMaxNormalization(feature_range=self.numerical_transform_feature_range)
            norm.fit(data, min_max_values=self.min_max_values)
            num_components = 1

            return DataManipulationInfo(
                column_name=col_name,
                column_type='Numerical',
                transform=norm,
                output_dimensions=num_components,
                activation_fn=self.numerical_activation_fn,
                unique_categories=None
            )

        elif self.numerical_activation_fn == 'stochastic':
            norm = StochasticNormalization()
            norm.fit(data)
            num_components = 1

            return DataManipulationInfo(
                column_name=col_name,
                column_type='Numerical',
                transform=norm,
                output_dimensions=num_components,
                activation_fn=self.numerical_activation_fn,
                unique_categories=None
            )

    def fit(self,
            data: pd.DataFrame,
            numerical_transform: str='minmax',
            numerical_transform_feature_range: Tuple[int, int]=(0, 1),
            feature_type: dict=None):
        assert numerical_transform in ['minmax', 'stochastic'], 'numerical_transform should be either minmax or stochastic'
        assert feature_type is not None, 'feature_type should be provided'

        self.numerical_activation_fn = numerical_transform
        self.numerical_transform_feature_range = numerical_transform_feature_range

        self._data_manipulation_info_list = []
        col_iterator = tqdm(data.columns, desc='Fit Data Transformation', disable=not self.verbose)
        for col_name in col_iterator:
            not_null = data[col_name].notnull()
            if feature_type[col_name] == 'Numerical':
                data_manipulation_info = self._fit_numerical(data[not_null][[col_name]])
            elif feature_type[col_name] == 'Categorical':
                data_manipulation_info = self._fit_categorical(data[not_null][[col_name]])
            elif feature_type[col_name] == 'Binary':
                data_manipulation_info = self._fit_binary(data[not_null][[col_name]])
            else:
                raise ValueError(f'Unknown feature type: {feature_type[col_name]}')
            self._data_manipulation_info_list.append(data_manipulation_info)

            col_iterator.set_postfix(column_name=col_name)

    def _transform_categorical(self, data: pd.DataFrame, norm: OneHotEncoder):
        if norm is None:
            return data.to_numpy()
        else:
            data_values = data.to_numpy()
            not_nan_mask = ~pd.isnull(data_values)

            if not_nan_mask.any().any():
                reshaped_data = data_values[not_nan_mask].reshape(-1, 1)
                reshaped_data = pd.DataFrame(reshaped_data, columns=data.columns)
                encoded_data = norm.transform(reshaped_data)
                transformed_data = np.full((data.shape[0], encoded_data.shape[1]), np.nan, dtype=float)
                transformed_data[not_nan_mask.ravel()] = encoded_data
            else:
                transformed_data = np.full((data.shape[0], 1), np.nan, dtype=float)

            return transformed_data

    def _transform_binary(self, data: pd.DataFrame, norm: OneHotEncoder):
        if norm is None:
            return data.to_numpy()
        else:
            data_values = data.to_numpy()
            not_nan_mask = ~pd.isnull(data_values)
            transformed_data = np.full(data.shape, np.nan, dtype=float)

            if not_nan_mask.any().any():
                reshaped_data = data_values[not_nan_mask].reshape(-1, 1)
                reshaped_data = pd.DataFrame(reshaped_data, columns=data.columns)
                transformed_data[not_nan_mask] = norm.transform(reshaped_data).flatten()
                return transformed_data
            else:
                return np.full(data.shape, np.nan, dtype=float)

    def _transform_numerical(self, data: pd.DataFrame, norm: MinMaxNormalization or StochasticNormalization):
        if norm is None:
            return data.to_numpy()
        else:
            column_name = data.columns[0]
            flatten_col = data[column_name].to_numpy().flatten()
            data = data.assign(**{column_name: flatten_col})
            transformed_data = norm.transform(data)

            return np.expand_dims(transformed_data, axis=1)

    def transform(self, data: pd.DataFrame):
        transformed_data_list = []
        col_iterator = tqdm(self._data_manipulation_info_list, desc='Transform Data', disable=not self.verbose)
        for data_manipulation_info in col_iterator:
            col_name = data_manipulation_info.column_name
            _data = data[[col_name]]

            if data_manipulation_info.column_type == 'Numerical':
                transformed_data = self._transform_numerical(_data, data_manipulation_info.transform)
            elif data_manipulation_info.column_type == 'Categorical':
                transformed_data = self._transform_categorical(_data, data_manipulation_info.transform)
            elif data_manipulation_info.column_type == 'Binary':
                transformed_data = self._transform_binary(_data, data_manipulation_info.transform)
            else:
                raise ValueError(f'Unknown column type: {data_manipulation_info.column_type}')
            transformed_data_list.append(transformed_data)

            col_iterator.set_postfix(column_name=col_name)

        return np.concatenate(transformed_data_list, axis=1)

    def _inverse_transform_categorical(self, data: np.ndarray, norm: OneHotEncoder):
        unique_categories = norm.categories_[0]
        if self.drop_first and (len(unique_categories) > 1):
            data = pd.DataFrame(data, columns=unique_categories[1:])
        else:
            data = pd.DataFrame(data, columns=unique_categories)

        nan_mask = pd.isna(data).any(axis=1)
        transformed_data = norm.inverse_transform(data.fillna(0))

        if nan_mask.sum() > 0:
            transformed_data[nan_mask] = np.nan

        return transformed_data

    def _inverse_transform_binary(self, data: np.array, norm: OneHotEncoder):
        unique_categories = norm.categories_[0]
        if len(unique_categories) > 1:
            data = pd.DataFrame(data, columns=unique_categories[1:])
        else:
            data = pd.DataFrame(data, columns=unique_categories)

        nan_mask = pd.isna(data).any(axis=1)
        transformed_data = norm.inverse_transform(data.fillna(0))

        if nan_mask.sum() > 0:
            transformed_data[nan_mask] = np.nan

        return transformed_data

    def _inverse_transform_numerical(self, data: np.array, norm: MinMaxNormalization or StochasticNormalization):
        data = pd.DataFrame(data, columns=[norm.get_output_sdtypes()])
        transformed_data = norm.inverse_transform(data)
        return transformed_data

    def inverse_transform(self, data: np.array, data_manipulation_info_list: List[DataManipulationInfo]=None):
        assert len(data.shape) == 2, 'data should be 2D array'

        if data_manipulation_info_list is None:
            data_manipulation_info_list = self._data_manipulation_info_list

        st = 0
        transformed_data_list = []
        col_names = []
        col_iterator = tqdm(data_manipulation_info_list, desc='Inverse Transform Data', disable=not self.verbose)
        for data_manipulation_info in col_iterator:
            dim = data_manipulation_info.output_dimensions
            _data = data[:, st:st+dim]

            if data_manipulation_info.column_type == 'Numerical':
                transformed_data = self._inverse_transform_numerical(_data, data_manipulation_info.transform)
            elif data_manipulation_info.column_type == 'Categorical':
                transformed_data = self._inverse_transform_categorical(_data, data_manipulation_info.transform)
            elif data_manipulation_info.column_type == 'Binary':
                transformed_data = self._inverse_transform_binary(_data, data_manipulation_info.transform)
            else:
                raise ValueError(f'Unknown column type: {data_manipulation_info.column_type}')
            transformed_data_list.append(transformed_data)
            col_names.append(data_manipulation_info.column_name)
            st += dim

            col_iterator.set_postfix(column_name=data_manipulation_info.column_name)

        transformed_data_list = np.column_stack(transformed_data_list)
        return pd.DataFrame(transformed_data_list, columns=col_names)

    def load(self, save_path: str):
        with open(save_path, 'rb') as f:
            parmas = pickle.load(f)
            self.__dict__.update(parmas.__dict__)

        return self

    def save(self, save_path: str):
        with open(save_path, 'wb') as f:
            pickle.dump(self.__dict__, f)


if __name__ == "__main__":
    feature_types = {
        'mortality': 'Binary',
        'gender': 'Categorical',
        'age': 'Numerical',
        'race': 'Categorical'
    }

    data = pd.DataFrame({'mortality': [0, 1, 0, 0], 'gender': ['M', 'F', 'M', np.NaN], 'age': [23, 35, 40, 57], 'race': ['Asian', 'Asian', 'Asian', 'Asian']})
    manipulation = Manipulation()
    manipulation.fit(data, numerical_transform='stochastic',  feature_type=feature_types)
    transformed_data = manipulation.transform(data)
    recovered_data = manipulation.inverse_transform(transformed_data)
    breakpoint()