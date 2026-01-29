import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Ensure project root is on the path so we can import from src/*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_preprocessing.synthetic_data import generate_synthetic_trajectories
from data_preprocessing.helpers_data import split_train_test
from src.helpers import get_model, get_flow_matching_inputs, get_data_stats
from src.interpolants import estimate_sigma_from_loader


class IrregularDataset(Dataset):
    def __init__(self, trajectories, window_size, fill_method='linear', stride=None):
        self.window_size = window_size
        self.stride = stride or window_size
        self.items = []
        self.fill_method = fill_method

        for traj in trajectories:
            times = np.asarray(traj["times"], dtype=np.float32)
            values = np.asarray(traj["values"], dtype=np.float32)
            if 'params' in traj:
                params = traj["params"]
            else:
                params = None
            if values.ndim == 1:
                values = values[:, None]

            mask = np.asarray(traj.get("mask", ~np.isnan(values)), dtype=bool)
            if mask.ndim == 1:
                mask = mask[:, None]

            original_values= values.copy()
            values[~mask] = np.nan
            if self.fill_method == 'linear':
                filled = linear_interpolant(times, values, mask)
            elif self.fill_method == 'zero':
                filled = zero_interpolant(times, values, mask)
            elif self.fill_method == 'bspline':
                filled = bspline_interpolant(times, values, mask)
            elif self.fill_method == 'cubic':
                filled = cubic_interpolant(times, values, mask)
            elif self.fill_method == 'nofill':
                filled = values
            else:
                raise ValueError(f"Invalid fill method: {self.fill_method}")

            start = 0
            while start < len(times):
                end = min(start + window_size, len(times))
                self.items.append(
                    (
                        torch.from_numpy(filled[start:end]),
                        torch.from_numpy(mask[start:end].astype(np.float32)),
                        torch.from_numpy(times[start:end]),
                        torch.from_numpy(original_values[start:end]),
                        params
                    )
                )
                if end == len(times):
                    break
                start += self.stride

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        values, mask, times, original_values, params = self.items[idx]
        rel_times = times[1:] - times[:-1]
        return {
            "values": values,
            "original_values": original_values,
            "mask": mask,
            "times": times,
            "relative_times": rel_times,
            "params": [x for x in params.values()]
        }


class ProcessedDataset(Dataset):
    def __init__(self, irregular_loader, train=False, interpolant_kind='linear', degree=3, subsample_per_interval=8, device=None, stats=None, dynamics_kind='ode', sigma=None):
        self.items= []
        self.dynamics_kind= dynamics_kind
        if sigma is None:
            self.sigma= estimate_sigma_from_loader(irregular_loader, device=device)
        else:
            self.sigma= sigma

        for batch in irregular_loader:

            values = batch["values"].to(device).float()          # [B, W, D]
            times = batch["times"].to(device).float()   # [B, W]
            mask = batch["mask"].to(device)
            original_values= batch["original_values"].to(device).float()
            original_times= times.clone().to(device).float()

            # inputs, times, targets, mask, score, lambda_t, eps_xt, eps_score = get_flow_matching_inputs(values, times, mask, interpolant_kind, degree, subsample_per_interval=subsample_per_interval, dynamics_kind=dynamics_kind, sigma= sigma)
            # self.items.append((inputs, times, targets, mask, original_times, original_values, score, lambda_t, eps_xt, eps_score))

            inputs, times, targets, mask, sigma_t, lambda_t, der_sigma_t = get_flow_matching_inputs(values, times, mask, interpolant_kind, degree, subsample_per_interval=subsample_per_interval, dynamics_kind=dynamics_kind, sigma= self.sigma)
            self.items.append((inputs, times, targets, mask, original_times, original_values, sigma_t, lambda_t, der_sigma_t))
        
        if train==True and stats==None:
            self.stats= get_data_stats(self.items)
        elif train!=True and stats!=None:
            self.stats=stats
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        # inputs, times, targets, mask, original_times, original_values, score, lambda_t, eps_xt, eps_score= self.items[idx]
        inputs, times, targets, mask, original_times, original_values, sigma_t, lambda_t, der_sigma_t= self.items[idx]

        """
        important: these normalizations are over interpolated data, not just available one
                    so it may pop up some unstability if the interpolants are crazy, like lagrangians/hermite
        """
        

        inputs= (inputs-self.stats['values_mean'])/(self.stats['values_std'])
        times= (times-self.stats['times_mean'])/(self.stats['times_std'])
        targets= (targets-self.stats['targets_mean'])/(self.stats['targets_std'])

        original_times_normalized= (original_times- self.stats['times_mean'])/(self.stats['times_std'])
        original_values_normalized= (original_values-self.stats['values_mean'])/(self.stats['values_std'])

        # return {
        # 'inputs': inputs,
        # 'times': times,
        # 'targets': targets,
        # 'mask': mask,
        # 'score': score,
        # 'lambda_t': lambda_t,
        # 'eps_xt': eps_xt,
        # 'eps_score': eps_score,

        # 'original_times_normalized': original_times_normalized,
        # 'original_values_normalized': original_values_normalized
        # }

        return {
        'inputs': inputs,
        'times': times,
        'targets': targets,
        'mask': mask,
        'sigma_t': sigma_t,
        'lambda_t': lambda_t,
        'der_sigma_t': der_sigma_t,
        'original_times_normalized': original_times_normalized,
        'original_values_normalized': original_values_normalized
        }






def get_dataset(data_config, reqs, times=[0.0, 10.0, 0.1], missing_prob=0.0, seed=None):

    datafile = f'/home/c02sara/CISPA-az6/dgdlm-2024/higher_interpolant_matching/data/{data_config}.pkl'
    data_path = Path(datafile)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if data_path.exists():
        with data_path.open("rb") as fh:
            trajectories = pickle.load(fh)
    else:
        trajectories = generate_synthetic_trajectories(reqs, times=times, missing_prob=missing_prob, seed=seed)
        with data_path.open("wb") as fh:
            pickle.dump(trajectories, fh)
    
    return trajectories


def get_dataloaders(config_name, batch_size, data_sample_complexity=1):

    data_config = json.load(open(f"/home/c02sara/CISPA-az6/dgdlm-2024/higher_interpolant_matching/configs/data_configs/{config_name}.json"))
    if data_config["synthetic"] == True:
        reqs = data_config["reqs"]
        data_name = list(reqs.keys())[0]
        missing_prob = data_config["missing_prob"] if 'missing_prob' in data_config else 0
        seed = data_config["seed"] if 'seed' in data_config else 42
        window_size = data_config["window_size"] if 'window_size' in data_config else reqs[data_name]["trajectories_per_config"]
        stride = data_config["stride"] if 'stride' in data_config else 1
        fill_method = data_config["fill_method"] if 'fill_method' in data_config else 'linear'
        times = data_config["times"] if 'times' in data_config else [0.0, 10.0, 0.1]

        trajectories = get_dataset(config_name, reqs, times, missing_prob, seed)

        data_until= int(data_sample_complexity*len(trajectories))
        trajectories= trajectories[:data_until]
    
    else:
        raise ValueError(f"Invalid data type: {data_config['type']}")

    train_traj, test_traj = split_train_test(trajectories, train_frac=0.8, seed=seed)
    train_dataset = IrregularDataset(train_traj, window_size=window_size, fill_method=fill_method, stride=stride)
    test_dataset = IrregularDataset(test_traj, window_size=window_size, fill_method=fill_method, stride=stride)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

    return train_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--create_data", type=bool, required=True)
    args = parser.parse_args()

    if args.create_data:
        config=json.load(open(f"/home/c02sara/CISPA-az6/dgdlm-2024/higher_interpolant_matching/configs/data_configs/{args.config_name}.json"))
        reqs = config["reqs"]
        data_name = list(reqs.keys())[0]
        missing_prob = config["missing_prob"] if 'missing_prob' in config else 0
        seed = config["seed"] if 'seed' in config else 42
        window_size = config["window_size"] if 'window_size' in config else reqs[data_name]["trajectories_per_config"]
        stride = config["stride"] if 'stride' in config else 1
        fill_method = config["fill_method"] if 'fill_method' in config else 'linear'
        times = config["times"] if 'times' in config else [0.0, 10.0, 0.1]
        print(f"Creating dataset for {data_name} with missing probability {missing_prob}")
        _ = get_dataset(args.config_name, reqs, times=times, missing_prob=missing_prob, seed=seed)
    



# train_loader, test_loader = get_dataloaders("lotka_volterra", batch_size=5)
# for batch in train_loader:
#     print('heres a subset of the values: ', batch['params'])
# for batch in test_loader:
#     print('heres a subset of the values: ', batch['params'])
# import pdb; pdb.set_trace()








################################################################################ # example usage ################################################################################
# # example usage
# trajectories1d = [
#     {
#         "times": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#         "values": [0.1, 0.2, 0.3, np.nan, 0.5, 0.6, 0.7, 0.8, np.nan, 1.0],
#         "mask": [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
#     },

#     {
#         "times": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#         "values": [1, np.nan, 3, np.nan, 5, np.nan, 7, np.nan, 9, np.nan],
#         "mask": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
#     },

#     {
#         "times": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#         "values": [0.4, 0.5, 0.6, 0.7, np.nan, 0.9, 1.0, 1.1, 1.2, 1.3],
#         "mask": [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
#     },

#     {
#         "times": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#         "values": [1.4, 1.5, 1.6, 1.7, np.nan, 1.9, 2.0, 2.1, 2.2, 2.3],
#         "mask": [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
#     }
# ]

# trajectoriesnd = [
#     {
#         "times": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#         "values": [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [np.nan, np.nan], [0.5, 0.5], [0.6, 0.6], [0.7, 0.7], [0.8, 0.8], [np.nan, np.nan], [1.0, 1.0]],
#         "mask": [[1, 1], [1, 1], [1, 1], [0, 0], [1, 1], [1, 1], [1, 1], [1, 1], [0, 0], [1, 1]],
#     },

#     {
#         "times": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#         "values": [[1, 1], [np.nan, np.nan], [3, 3], [np.nan, np.nan], [5, 5], [np.nan, np.nan], [7, 7], [np.nan, np.nan], [9, 9], [np.nan, np.nan]],
#         "mask": [[1, 1], [0, 0], [1, 1], [0, 0], [1, 1], [0, 0], [1, 1], [0, 0], [1, 1], [0, 0]],
#     },

#     {
#         "times": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#         "values": [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6], [0.7, 0.7], [np.nan, np.nan], [0.9, 0.9], [1.0, 1.0], [1.1, 1.1], [1.2, 1.2], [1.3, 1.3]],
#         "mask": [[1, 1], [1, 1], [1, 1], [1, 1], [0, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
#     },

#     {
#         "times": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#         "values": [[1.4, 1.4], [1.5, 1.5], [1.6, 1.6], [1.7, 1.7], [np.nan, np.nan], [1.9, 1.9], [2.0, 2.0], [2.1, 2.1], [2.2, 2.2], [2.3, 2.3]],
#         "mask": [[1, 1], [1, 1], [1, 1], [1, 1], [0, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
#     }
# ]



# # dataset = IrregularDataset(trajectories1d, window_size=10, fill_method='zero')
# dataset = IrregularDataset(trajectoriesnd, window_size=10, fill_method='linear')
# loader = DataLoader(dataset, batch_size=2)

# for batch in loader:
#     print('heres a subset of the values: ', batch['values'][:,:5])
    

# import pdb; pdb.set_trace()
