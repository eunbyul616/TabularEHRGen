import numpy as np
import pandas as pd
from collections import OrderedDict
from bisect import bisect_left


class StochasticNormalization:
    def __init__(self):
        self.info = OrderedDict()

    def fit(self, x: pd.DataFrame):
        lower_bound = 0.0
        upper_bound = 0.0

        col_name = x.columns[0]
        self.params = dict()
        self.bounds = []
        self.values = []

        value_counts = x[col_name].dropna().value_counts(normalize=True).sort_index()
        self.N = len(x[col_name].dropna())

        for val, val_ratio in value_counts.items():
            upper_bound = lower_bound + val_ratio
            self.params[float(val)] = [lower_bound, upper_bound]
            self.bounds.append(upper_bound)
            self.values.append(val)
            lower_bound = upper_bound

        self.info[col_name] = self.params

    def transform(self, x: pd.DataFrame):
        col_name = x.columns[0]
        x_col = x[col_name].values
        x_norm = np.zeros_like(x_col, dtype=float)

        # Process seen values
        for val, (lower_bound, upper_bound) in self.params.items():
            val_idxs = (x_col == val)
            x_norm[val_idxs] = np.random.uniform(lower_bound, upper_bound, val_idxs.sum())

        # Handle NaN values
        nan_idxs = np.isnan(x_col)
        x_norm[nan_idxs] = np.nan

        # Process unseen values
        unseen_values_mask = ~np.isin(x_col, list(self.params.keys())) & ~nan_idxs
        unseen_values = x_col[unseen_values_mask]

        for idx, input_val in zip(np.where(unseen_values_mask)[0], unseen_values):
            closest_idx = np.argmin([abs(input_val - v) for v in self.values])
            closest_val = self.values[closest_idx]

            lower_bound, upper_bound = self.params[closest_val]
            x_norm[idx] = np.random.uniform(lower_bound, upper_bound)

        return x_norm

    def inverse_transform(self, x: np.array or pd.DataFrame):
        out = np.zeros_like(x, dtype=float)

        for i, val in enumerate(x.values):
            if np.isnan(val):
                out[i] = np.nan
                continue
            idx = bisect_left(self.bounds, val)
            out[i] = self.values[idx] if idx < len(self.values) else np.nan

        return out

    def get_output_sdtypes(self):
        return self.info.keys()


class MinMaxNormalization:
    def __init__(self,
                 feature_range=(0, 1)):
        self.info = OrderedDict()
        self.range = feature_range

    def fit(self, x, min_max_values=None):
        col_name = x.columns[0]

        if min_max_values is not None:
            self.info[col_name] = min_max_values[col_name]
        else:
            self.info[col_name] = {'min': x.min().values[0], 'max': x.max().values[0]}

    def transform(self, x):
        col_name = x.columns[0]
        a, b = self.range[0], self.range[1]
        _x = x[col_name]

        _min, _max = self.info[col_name]['min'], self.info[col_name]['max']
        x_norm = ((_x - _min) / (_max - _min)) * (b - a) + a

        return x_norm.values

    def inverse_transform(self, x):
        col_name = x.columns[0]
        a, b = self.range[0], self.range[1]
        _min, _max = self.info[col_name]['min'], self.info[col_name]['max']
        x = x.map(lambda x: x if np.isnan(x) or (x >= a) else a)
        x = x.map(lambda x: x if np.isnan(x) or (x <= b) else b)
        out = ((x - a) / (b - a) * (_max - _min) + _min)
        return out

    def get_output_sdtypes(self):
        return self.info.keys()


class Standardization:
    def __init__(self):
        self.info = OrderedDict()

    def fit(self, x):
        mean = np.mean(x)
        std = np.std(x)
        col_name = x.columns[0]
        self.info[col_name] = [mean, std]

    def transform(self, x):
        col_name = x.columns[0]
        mean, std = self.info[col_name]
        x_norm = (x - mean) / std
        return x_norm.values.flatten()

    def inverse_transform(self, x):
        col_name = x.columns[0]
        mean, std = self.info[col_name]
        out = (x * std) + mean
        return out

    def get_output_sdtypes(self):
        return self.info.keys()


if __name__ == "__main__":
    x = pd.DataFrame([1, 2, 2, 2, 2, 2, 2, 10, 3, 3, np.nan, None], columns=['dbp'])
    # min_max_info = {'dbp': {'max': 10.0, 'min': 0.0}}
    # mm = MinMaxNormalization(value_range=(-1, 1))
    # mm.fit(x=x, min_max_info=min_max_info)
    # x_norm = mm.transform(x=x)
    # print(x_norm)
    #
    # x_norm = pd.DataFrame(x_norm, columns=['dbp'])
    # x = mm.inverse_transform(x_norm)
    # print(x)

    sn = StochasticNormalization()
    sn.fit(x=x)
    x_norm = sn.transform(x=x)
    print(x_norm)

    x = pd.DataFrame([1, 2, 2, 2, 2, 2, 2, 11, 3, 3, np.nan, None], columns=['dbp'])
    x_norm = sn.transform(x=x)
    print(x_norm)

    x_norm = pd.DataFrame(x_norm)
    x = sn.inverse_transform(x_norm)
    print(x)
    breakpoint()
