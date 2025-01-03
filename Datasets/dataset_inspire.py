import numpy as np
from omegaconf import DictConfig
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader

from Utils.file import *
from Utils.dataset import *
from Datasets.utils import *
from Manipulation.manipulation import Manipulation


class INSPIREDataset(Dataset):
    def __init__(self,
                 cfg: DictConfig,
                 dataset_fname: str=None,
                 static_hdf5_key: str='static_clip',
                 temporal_hdf5_key: str='temporal_clip',
                 mode: str='train',
                 static_cols: List[str]=None,
                 temporal_cols: List[str]=None):
        assert mode in ['train', 'val', 'test'], 'Invalid mode'

        self.cfg = cfg
        self.verbose = cfg.dataset.verbose
        self.dataset_name = 'INSPIRE'

        key_cols = cfg.data.key_cols
        time_cols = cfg.data.time_cols
        static_exclude_cols = cfg.data.excluded_cols.static
        temporal_exclude_cols = cfg.data.excluded_cols.temporal
        seq_len = cfg.dataloader.seq_len

        data_path = os.path.join(cfg.path.preprocessed_data_path, self.dataset_name)
        if dataset_fname is None:
            # dataset_fname = 'INSPIRE_260K_tot_clip.h5'
            dataset_fname = 'INSPIRE_260K_30000.h5'


        os.makedirs(os.path.join(cfg.path.transformer_path, self.dataset_name), exist_ok=True)
        static_transformer_fpath = os.path.join(cfg.path.transformer_path,
                                                self.dataset_name,
                                                f'{cfg.manipulation.transformer_fname}_static_transformer.pkl')
        static_transformed_data_fpath = os.path.join(cfg.path.transformer_path,
                                                     self.dataset_name,
                                                     f'{cfg.manipulation.transformer_fname}_static_{mode}.pkl')
        static_min_max_fpath = os.path.join(cfg.path.transformer_path,
                                            self.dataset_name,
                                            f'{cfg.manipulation.transformer_fname}_static_min_max_values.pkl')
        temporal_transformer_fpath = os.path.join(cfg.path.transformer_path,
                                                  self.dataset_name,
                                                  f'{cfg.manipulation.transformer_fname}_temporal_transformer.pkl')
        temporal_transformed_data_fpath = os.path.join(cfg.path.transformer_path,
                                                       self.dataset_name,
                                                       f'{cfg.manipulation.transformer_fname}_temporal_{mode}.pkl')
        temporal_min_max_fpath = os.path.join(cfg.path.transformer_path,
                                              self.dataset_name,
                                              f'{cfg.manipulation.transformer_fname}_temporal_min_max_values.pkl')

        static_data, static_type = load_data(os.path.join(data_path, dataset_fname), static_hdf5_key, mode)
        temporal_data, temporal_type = load_data(os.path.join(data_path, dataset_fname), temporal_hdf5_key, mode)
        static_data_key_cols = [col for col in key_cols if col in static_data.columns]
        temporal_data_key_cols = [col for col in key_cols if col in temporal_data.columns]

        static_data = static_data.set_index(static_data_key_cols)
        temporal_data = temporal_data.set_index(temporal_data_key_cols + time_cols)

        if mode == 'train':
            static_exclude_cols += [col for col in static_data.columns if static_data[col].nunique() == 1]
            if len(static_exclude_cols) > 0:
                static_data = static_data.drop(columns=static_exclude_cols)

            num_cols = [col for col in static_data.columns if static_type[col] == 'Numerical']
            min_max_values = static_data[num_cols].agg(['min', 'max']).to_dict()
            save_pkl(min_max_values, static_min_max_fpath)

            static_transformer = Manipulation(
                verbose=self.verbose,
                numerical_activation_fn=cfg.manipulation.activation_fn.numerical,
                categorical_activation_fn=cfg.manipulation.activation_fn.categorical,
                binary_activation_fn=cfg.manipulation.activation_fn.binary,
                min_max_values=min_max_values,
                drop_first=cfg.manipulation.drop_first,
            )

            temporal_exclude_cols += [col for col in temporal_data.columns if temporal_data[col].nunique() == 1]
            if len(temporal_exclude_cols) > 0:
                temporal_data = temporal_data.drop(columns=temporal_exclude_cols)

            num_cols = [col for col in temporal_data.columns if temporal_type[col] == 'Numerical']
            min_max_values = temporal_data[num_cols].agg(['min', 'max']).to_dict()
            save_pkl(min_max_values, temporal_min_max_fpath)

            temporal_transformer = Manipulation(
                verbose=self.verbose,
                numerical_activation_fn=cfg.manipulation.activation_fn.numerical,
                categorical_activation_fn=cfg.manipulation.activation_fn.categorical,
                binary_activation_fn=cfg.manipulation.activation_fn.binary,
                min_max_values=min_max_values,
                drop_first=cfg.manipulation.drop_first,
            )

        else:
            min_max_values = load_pkl(static_min_max_fpath)
            static_transformer = Manipulation(
                verbose=self.verbose,
                numerical_activation_fn=cfg.manipulation.activation_fn.numerical,
                categorical_activation_fn=cfg.manipulation.activation_fn.categorical,
                binary_activation_fn=cfg.manipulation.activation_fn.binary,
                min_max_values=min_max_values,
                drop_first=cfg.manipulation.drop_first,
            )
            if os.path.exists(static_transformer_fpath):
                print('Transformer found. Load transformer.')
                static_transformer = static_transformer.load(static_transformer_fpath)

            include_cols = [col.column_name for col in static_transformer._data_manipulation_info_list if '_mask' not in col.column_name]
            static_data = static_data[include_cols]

            min_max_values = load_pkl(temporal_min_max_fpath)
            temporal_transformer = Manipulation(
                verbose=self.verbose,
                numerical_activation_fn=cfg.manipulation.activation_fn.numerical,
                categorical_activation_fn=cfg.manipulation.activation_fn.categorical,
                binary_activation_fn=cfg.manipulation.activation_fn.binary,
                min_max_values=min_max_values,
                drop_first=cfg.manipulation.drop_first,
            )
            if os.path.exists(temporal_transformer_fpath):
                print('Transformer found. Load transformer.')
                temporal_transformer = temporal_transformer.load(temporal_transformer_fpath)

            include_cols = [col.column_name for col in temporal_transformer._data_manipulation_info_list if '_mask' not in col.column_name]
            temporal_data = temporal_data[include_cols]

        if cfg.dataloader.mask:
            static_mask_data, _ = load_data(os.path.join(data_path, dataset_fname), f'{static_hdf5_key}_mask', mode)
            temporal_mask_data, _ = load_data(os.path.join(data_path, dataset_fname), f'{temporal_hdf5_key}_mask', mode)
            static_mask_data = static_mask_data.set_index(static_data_key_cols)
            temporal_mask_data = temporal_mask_data.set_index(temporal_data_key_cols+time_cols)

            if mode == 'train':
                if len(static_exclude_cols) > 0:
                    static_mask_data = static_mask_data.drop(columns=static_exclude_cols)
                if len(temporal_exclude_cols) > 0:
                    temporal_mask_data = temporal_mask_data.drop(columns=temporal_exclude_cols)
            else:
                include_cols = [col.column_name for col in static_transformer._data_manipulation_info_list if '_mask' not in col.column_name]
                static_mask_data = static_mask_data[include_cols]

                include_cols = [col.column_name for col in temporal_transformer._data_manipulation_info_list if '_mask' not in col.column_name]
                temporal_mask_data = temporal_mask_data[include_cols]

            # if cfg.data.remove_timepoints:
            #     # remove time point without measurement
            #     temporal_data, temporal_mask_data = remove_timepoints_with_no_measurement(data=temporal_data,
            #                                                                               mask_data=temporal_mask_data)

        if static_cols is None:
            static_type = {col: static_type[col] for col in static_data.columns
                           if col not in static_exclude_cols}
        else:
            static_type = {col: static_type[col] for col in static_data.columns
                           if (col in static_cols) and (col not in static_exclude_cols)}
        if temporal_cols is None:
            temporal_type = {col: temporal_type[col] for col in temporal_data.columns
                             if col not in temporal_exclude_cols}
        else:
            temporal_type = {col: temporal_type[col] for col in temporal_data.columns
                             if (col in temporal_cols) and (col not in temporal_exclude_cols)}
        s_cols = static_data.columns
        t_cols = temporal_data.columns

        if cfg.dataloader.mask:
            if static_cols is None:
                static_mask_type = {f'{col}_mask': 'Binary' for col in static_data.columns
                                    if col not in static_exclude_cols}
            else:
                static_mask_type = {f'{col}_mask': 'Binary' for col in static_data.columns
                                    if (col in static_cols) and (col not in static_exclude_cols)}
            if temporal_cols is None:
                temporal_mask_type = {f'{col}_mask': 'Binary' for col in temporal_data.columns
                                      if col not in temporal_exclude_cols}
            else:

                temporal_mask_type = {f'{col}_mask': 'Binary' for col in temporal_data.columns
                                      if (col in temporal_cols) and (col not in temporal_exclude_cols)}

            static_type.update(static_mask_type)
            temporal_type.update(temporal_mask_type)

            sm_cols = [f'{col}_mask' for col in s_cols]
            tm_cols = [f'{col}_mask' for col in t_cols]
            static_mask_data.columns = sm_cols
            temporal_mask_data.columns = tm_cols

            static_data = pd.concat([static_data, static_mask_data], axis=1)
            temporal_data = pd.concat([temporal_data, temporal_mask_data], axis=1)

        temporal_data = temporal_data.reset_index()
        padding_idx = temporal_data[time_cols].isna().any(axis=1)
        # temporal_data = clip_data_by_timepoints(temporal_data,
        #                                         timepoints=seq_len,
        #                                         padding=cfg.data.padding,
        #                                         group_cols=key_cols,
        #                                         parallel=cfg.data.parallel)

        # forward fill
        if not cfg.dataloader.pad_flag:
            temporal_data[t_cols] = temporal_data[t_cols].fillna(method='ffill')
            temporal_data[tm_cols] = temporal_data[tm_cols].fillna(0)
            temporal_data[time_cols] = temporal_data[time_cols].fillna(method='ffill')
        temporal_data = temporal_data.set_index(temporal_data_key_cols+time_cols)

        static_transformer, static_transformed_data = manipulate_data(cfg=cfg,
                                                                      transformer_path=static_transformer_fpath,
                                                                      transformed_data_path=static_transformed_data_fpath,
                                                                      transformer=static_transformer,
                                                                      data=static_data,
                                                                      feature_type=static_type)
        sc_data, sn_data, sc_mask_data, sn_mask_data, sc_cols, sn_cols = split_data_feature_type(
            static_transformed_data,
            static_type,
            static_transformer
        )

        temporal_transformer, temporal_transformed_data = manipulate_data(cfg=cfg,
                                                                          transformer_path=temporal_transformer_fpath,
                                                                          transformed_data_path=temporal_transformed_data_fpath,
                                                                          transformer=temporal_transformer,
                                                                          data=temporal_data,
                                                                          feature_type=temporal_type)
        if cfg.dataloader.pad_flag:
            temporal_transformed_data[padding_idx] = cfg.dataloader.pad_value
        # temporal_transformed_data = np.nan_to_num(temporal_transformed_data, nan=cfg.dataloader.pad_value)

        tc_data, tn_data, tc_mask_data, tn_mask_data, tc_cols, tn_cols = split_data_feature_type(
            temporal_transformed_data,
            temporal_type,
            temporal_transformer
        )

        tc_data = tc_data.reshape(-1, seq_len, tc_data.shape[-1])
        tn_data = tn_data.reshape(-1, seq_len, tn_data.shape[-1])
        tc_mask_data = tc_mask_data.reshape(-1, seq_len, tc_mask_data.shape[-1])
        tn_mask_data = tn_mask_data.reshape(-1, seq_len, tn_mask_data.shape[-1])

        time_data = pd.DataFrame(temporal_data.index.get_level_values(len(key_cols)), columns=time_cols)
        time_type = {time_cols[0]: 'Numerical'}
        time_transformer_fpath = os.path.join(cfg.path.transformer_path,
                                              self.dataset_name,
                                              f'{cfg.manipulation.transformer_fname}_time_transformer.pkl')
        time_transformed_data_fpath = os.path.join(cfg.path.transformer_path,
                                                   self.dataset_name,
                                                   f'{cfg.manipulation.transformer_fname}_time_{mode}.pkl')
        time_min_max_fpath = os.path.join(cfg.path.transformer_path,
                                          self.dataset_name,
                                          f'{cfg.manipulation.transformer_fname}_time_min_max_values.pkl')
        if mode == 'train':
            num_cols = [col for col in time_data.columns if time_type[col] == 'Numerical']
            min_max_values = time_data[num_cols].agg(['min', 'max']).to_dict()
            save_pkl(min_max_values, time_min_max_fpath)
        else:
            min_max_values = load_pkl(time_min_max_fpath)

        time_transformer = Manipulation(
            verbose=self.verbose,
            numerical_activation_fn=cfg.manipulation.activation_fn.numerical,
            categorical_activation_fn=cfg.manipulation.activation_fn.categorical,
            binary_activation_fn=cfg.manipulation.activation_fn.binary,
            min_max_values=min_max_values,
            drop_first=cfg.manipulation.drop_first,
        )
        time_transformer, time_transformed_data = manipulate_data(cfg=cfg,
                                                                  transformer_path=time_transformer_fpath,
                                                                  transformed_data_path=time_transformed_data_fpath,
                                                                  transformer=time_transformer,
                                                                  data=time_data,
                                                                  feature_type=time_type)
        if cfg.dataloader.pad_flag:
            time_transformed_data[padding_idx] = np.nan_to_num(time_transformed_data[padding_idx], nan=cfg.dataloader.pad_value)
        time_transformed_data = time_transformed_data.reshape(-1, seq_len, 1)

        self.static_data = static_data
        self.temporal_data = temporal_data
        self.static_type = static_type
        self.temporal_type = temporal_type

        self.static_transformer = static_transformer
        self.temporal_transformer = temporal_transformer
        self.time_transformer = time_transformer

        # for static categorical autoencoder
        self.sc_data = torch.tensor(sc_data, dtype=torch.float32)
        self.sc_cols = sc_cols
        # for temporal categorical autoencoder
        self.tc_data = torch.tensor(tc_data, dtype=torch.float32)
        self.tc_cols = tc_cols
        # for ehr autoencoder
        self.sn_data = torch.tensor(sn_data, dtype=torch.float32)
        self.sn_cols = sn_cols
        self.tn_data = torch.tensor(tn_data, dtype=torch.float32)
        self.tn_cols = tn_cols
        self.sc_mask_data = torch.tensor(sc_mask_data, dtype=torch.float32)
        self.tc_mask_data = torch.tensor(tc_mask_data, dtype=torch.float32)
        self.sn_mask_data = torch.tensor(sn_mask_data, dtype=torch.float32)
        self.tn_mask_data = torch.tensor(tn_mask_data, dtype=torch.float32)
        self.time_data = torch.tensor(time_transformed_data, dtype=torch.float32)

        del static_data, temporal_data, static_mask_data, temporal_mask_data
        del sc_data, tc_data, sc_mask_data, tc_mask_data
        del sn_data, tn_data, sn_mask_data, tn_mask_data
        del static_transformed_data, temporal_transformed_data
        del time_data, time_transformed_data
        del static_transformer, temporal_transformer
        del time_transformer

    def __getitem__(self, idx):
        static_mask_data = torch.concatenate([self.sc_mask_data[idx], self.sn_mask_data[idx]], dim=-1)
        temporal_mask_data = torch.concatenate([self.tc_mask_data[idx], self.tn_mask_data[idx]], dim=-1)

        return (self.sc_data[idx], self.tc_data[idx],
                self.sn_data[idx], self.tn_data[idx],
                static_mask_data, temporal_mask_data,
                self.time_data[idx])

    def __len__(self):
        return len(self.sc_data)

    @property
    def static_key_df(self):
        return pd.DataFrame(list(self.static_data.index), columns=self.cfg.data.key_cols)

    @property
    def temporal_key_df(self):
        return pd.DataFrame(list(self.temporal_data.index), columns=self.cfg.data.key_cols+self.cfg.data.time_cols)


if __name__ == "__main__":
    import config_manager
    
    config_manager.load_config()
    cfg = config_manager.config

    train_dataset = INSPIREDataset(cfg, mode='train')
    breakpoint()
