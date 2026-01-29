import numpy as np
import torch
import os
import json
import pandas as pd

exp_name= 'standard'

data_types= ['lotka_volterra', 'damped_harmonic', 'harmonic_oscillator', 'lorenz', 'exp_decay', 'hopperphysics']
sparsity_list= ['', '_sparse', '_v_sparse', '_vv_sparse']
interpolant_types= ['_bspline', '_linear']

base_path= '/Users/rssantanu/Downloads/higher_interpolant_matching_corrected/results/synthetic_new'

metric_dict= {'data_type':[], 'sparsity':[], 'interpolant_type':[], 'train_loss_mean':[], 'train_loss_std':[], 'test_loss_mean':[], 'test_loss_std':[]}
for data_type in data_types:
    for sparsity in sparsity_list:
        for interpolant_type in interpolant_types:
            base_name= f'{exp_name}_{data_type}{sparsity}{interpolant_type}'
            try:
                dirs= [x if base_name in x else '' for x in os.listdir(os.path.join(base_path, base_name))]
                train_= []
                test_= []
                validation_metrics_mse= []
                for d in dirs:
                    metrics= json.load(open(os.path.join(base_path, base_name, d, 'final_mse_metrics.json')))
                    metrics_validation= json.load(open(os.path.join(base_path, base_name, d, 'test_validation_metrics.json')))
                    train_.append(metrics['train'][0][1])
                    test_.append(metrics['test'][0][1])
                    validation_metrics_mse.append(metrics_validation['mse'])
            

                metric_dict['sparsity'].append(sparsity)
                metric_dict['interpolant_type'].append(interpolant_type)
                metric_dict['data_type'].append(data_type)
                metric_dict['train_loss_mean'].append(np.mean(train_))
                metric_dict['train_loss_std'].append(np.std(train_))
                metric_dict['test_loss_mean'].append(np.mean(test_))
                metric_dict['test_loss_std'].append(np.std(test_))
                metric_dict['test_validation_metrics_mean'].append(np.mean(validation_metrics_mse))
                metric_dict['test_validation_metrics_std'].append(np.std(validation_metrics_mse))
                print(base_name)
            except:
                # pass
                print(f'{base_name} not found')
            
            

metric_df= pd.DataFrame(metric_dict)
metric_df.to_csv(f'./result_metrics/{exp_name}.csv')

import pdb; pdb.set_trace()