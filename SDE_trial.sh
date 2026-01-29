python main.py --interpolant_kind bspline --exp_name trial_known_sigma --data_config lorenz_sde --dynamics_kind sde_constant_sigma
python main.py --interpolant_kind bspline --exp_name trial_known_sigma --data_config damped_harmonic_sde --dynamics_kind sde_constant_sigma
python main.py --interpolant_kind bspline --exp_name trial_known_sigma --data_config exp_decay_sde --dynamics_kind sde_constant_sigma
python main.py --interpolant_kind bspline --exp_name trial_known_sigma --data_config lotka_volterra_sde --dynamics_kind sde_constant_sigma

python main.py --interpolant_kind bspline --exp_name trial_known_sigma_longer --data_config lorenz_sde --dynamics_kind sde_constant_sigma --epochs 100000
python main.py --interpolant_kind bspline --exp_name trial_known_sigma_longer --data_config damped_harmonic_sde --dynamics_kind sde_constant_sigma --epochs 100000
python main.py --interpolant_kind bspline --exp_name trial_known_sigma_longer --data_config exp_decay_sde --dynamics_kind sde_constant_sigma --epochs 100000
python main.py --interpolant_kind bspline --exp_name trial_known_sigma_longer --data_config lotka_volterra_sde --dynamics_kind sde_constant_sigma --epochs 100000

python main.py --interpolant_kind bspline --exp_name trial_known_sigma_longer_d4 --data_config lorenz_sde --dynamics_kind sde_constant_sigma --epochs 100000 --degree 4
python main.py --interpolant_kind bspline --exp_name trial_known_sigma_longer_d4 --data_config damped_harmonic_sde --dynamics_kind sde_constant_sigma --epochs 100000 --degree 4
python main.py --interpolant_kind bspline --exp_name trial_known_sigma_longer_d4 --data_config exp_decay_sde --dynamics_kind sde_constant_sigma --epochs 100000 --degree 4
python main.py --interpolant_kind bspline --exp_name trial_known_sigma_longer_d4 --data_config lotka_volterra_sde --dynamics_kind sde_constant_sigma --epochs 100000 --degree 4

python main.py --interpolant_kind bspline --exp_name trial_known_sigma_longer_d5 --data_config lorenz_sde --dynamics_kind sde_constant_sigma --epochs 100000 --degree 5
python main.py --interpolant_kind bspline --exp_name trial_known_sigma_longer_d5 --data_config damped_harmonic_sde --dynamics_kind sde_constant_sigma --epochs 100000 --degree 5
python main.py --interpolant_kind bspline --exp_name trial_known_sigma_longer_d5 --data_config exp_decay_sde --dynamics_kind sde_constant_sigma --epochs 100000 --degree 5
python main.py --interpolant_kind bspline --exp_name trial_known_sigma_longer_d5 --data_config lotka_volterra_sde --dynamics_kind sde_constant_sigma --epochs 100000 --degree 5

python main.py --interpolant_kind bspline --exp_name trial_known_sigma_longer_bigger --data_config lorenz_sde --dynamics_kind sde_constant_sigma --epochs 100000 --model_config MLP_wide
python main.py --interpolant_kind bspline --exp_name trial_known_sigma_longer_bigger --data_config damped_harmonic_sde --dynamics_kind sde_constant_sigma --epochs 100000 --model_config MLP_wide
python main.py --interpolant_kind bspline --exp_name trial_known_sigma_longer_bigger --data_config exp_decay_sde --dynamics_kind sde_constant_sigma --epochs 100000 --model_config MLP_wide
python main.py --interpolant_kind bspline --exp_name trial_known_sigma_longer_bigger --data_config lotka_volterra_sde --dynamics_kind sde_constant_sigma --epochs 100000 --model_config MLP_wide