# python data_preprocessing/dataloader.py --config_name exp_decay_sde --create_data True
# python data_preprocessing/dataloader.py --config_name damped_harmonic_sde --create_data True
# python data_preprocessing/dataloader.py --config_name lorenz_sde --create_data True
# python data_preprocessing/dataloader.py --config_name lotka_volterra_sde --create_data True

python data_preprocessing/dataloader.py --config_name exp_decay_sde_sparse --create_data True
python data_preprocessing/dataloader.py --config_name damped_harmonic_sde_sparse --create_data True
python data_preprocessing/dataloader.py --config_name lorenz_sde_sparse --create_data True
python data_preprocessing/dataloader.py --config_name lotka_volterra_sde_sparse --create_data True

python data_preprocessing/dataloader.py --config_name exp_decay_sde_v_sparse --create_data True
python data_preprocessing/dataloader.py --config_name damped_harmonic_sde_v_sparse --create_data True
python data_preprocessing/dataloader.py --config_name lorenz_sde_v_sparse --create_data True
python data_preprocessing/dataloader.py --config_name lotka_volterra_sde_v_sparse --create_data True

python data_preprocessing/dataloader.py --config_name exp_decay_sde_vv_sparse --create_data True
python data_preprocessing/dataloader.py --config_name damped_harmonic_sde_vv_sparse --create_data True
python data_preprocessing/dataloader.py --config_name lorenz_sde_vv_sparse --create_data True
python data_preprocessing/dataloader.py --config_name lotka_volterra_sde_vv_sparse --create_data True