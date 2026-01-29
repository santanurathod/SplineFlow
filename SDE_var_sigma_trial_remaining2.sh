python main.py --interpolant_kind linear --exp_name trial_quadratic_sigma_known --data_config lorenz_sde --dynamics_kind sde_quadratic_sigma
python main.py --interpolant_kind linear --exp_name trial_quadratic_sigma_known --data_config damped_harmonic_sde --dynamics_kind sde_quadratic_sigma
python main.py --interpolant_kind linear --exp_name trial_quadratic_sigma_known --data_config exp_decay_sde --dynamics_kind sde_quadratic_sigma
python main.py --interpolant_kind linear --exp_name trial_quadratic_sigma_known --data_config lotka_volterra_sde --dynamics_kind sde_quadratic_sigma
python main.py --interpolant_kind linear --exp_name trial_quadratic_sigma_known_longer --data_config lorenz_sde --dynamics_kind sde_quadratic_sigma --epochs 100000
python main.py --interpolant_kind linear --exp_name trial_quadratic_sigma_known_longer --data_config damped_harmonic_sde --dynamics_kind sde_quadratic_sigma --epochs 100000
python main.py --interpolant_kind linear --exp_name trial_quadratic_sigma_known_longer --data_config exp_decay_sde --dynamics_kind sde_quadratic_sigma --epochs 100000
python main.py --interpolant_kind linear --exp_name trial_quadratic_sigma_known_longer --data_config lotka_volterra_sde --dynamics_kind sde_quadratic_sigma --epochs 100000