# hopperphysics
python main.py --data_config hopperphysics --interpolant_kind bspline --exp_name widenetworks_d1 --degree 1 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001
python main.py --data_config hopperphysics_sparse --interpolant_kind bspline --exp_name widenetworks_d1 --degree 1 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001
python main.py --data_config hopperphysics_v_sparse --interpolant_kind bspline --exp_name widenetworks_d1 --degree 1 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001
python main.py --data_config hopperphysics_vv_sparse --interpolant_kind bspline --exp_name widenetworks_d1 --degree 1 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001


python main.py --data_config hopperphysics --interpolant_kind bspline --exp_name widenetworks_d2 --degree 2 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001
python main.py --data_config hopperphysics_sparse --interpolant_kind bspline --exp_name widenetworks_d2 --degree 2 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001
python main.py --data_config hopperphysics_v_sparse --interpolant_kind bspline --exp_name widenetworks_d2 --degree 2 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001
python main.py --data_config hopperphysics_vv_sparse --interpolant_kind bspline --exp_name widenetworks_d2 --degree 2 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001

python main.py --data_config hopperphysics --interpolant_kind linear --exp_name widenetworks_d1 --degree 1 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001
python main.py --data_config hopperphysics_sparse --interpolant_kind linear --exp_name widenetworks_d1 --degree 1 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001
python main.py --data_config hopperphysics_v_sparse --interpolant_kind linear --exp_name widenetworks_d1 --degree 1 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001
python main.py --data_config hopperphysics_vv_sparse --interpolant_kind linear --exp_name widenetworks_d1 --degree 1 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001


# lorenz

python main.py --data_config lorenz --interpolant_kind bspline --exp_name widenetworks_d3 --degree 3 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001
python main.py --data_config lorenz_sparse --interpolant_kind bspline --exp_name widenetworks_d3 --degree 3 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001
python main.py --data_config lorenz_v_sparse --interpolant_kind bspline --exp_name widenetworks_d3 --degree 3 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001
python main.py --data_config lorenz_vv_sparse --interpolant_kind bspline --exp_name widenetworks_d3 --degree 3 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001


python main.py --data_config lorenz --interpolant_kind linear --exp_name widenetworks_d3 --degree 3 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001
python main.py --data_config lorenz_sparse --interpolant_kind linear --exp_name widenetworks_d3 --degree 3 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001
python main.py --data_config lorenz_v_sparse --interpolant_kind linear --exp_name widenetworks_d3 --degree 3 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001
python main.py --data_config lorenz_vv_sparse --interpolant_kind linear --exp_name widenetworks_d3 --degree 3 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001


python main.py --data_config lorenz --interpolant_kind bspline --exp_name widenetworks_d4 --degree 4 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001
python main.py --data_config lorenz_sparse --interpolant_kind bspline --exp_name widenetworks_d4 --degree 4 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001
python main.py --data_config lorenz_v_sparse --interpolant_kind bspline --exp_name widenetworks_d4 --degree 4 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001
python main.py --data_config lorenz_vv_sparse --interpolant_kind bspline --exp_name widenetworks_d4 --degree 4 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001

python main.py --data_config lorenz --interpolant_kind bspline --exp_name widenetworks_d5 --degree 5 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001
python main.py --data_config lorenz_sparse --interpolant_kind bspline --exp_name widenetworks_d5 --degree 5 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001
python main.py --data_config lorenz_v_sparse --interpolant_kind bspline --exp_name widenetworks_d5 --degree 5 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001
python main.py --data_config lorenz_vv_sparse --interpolant_kind bspline --exp_name widenetworks_d5 --degree 5 --model_config MLP_wide --lr_scheduler cosine --epochs 100000 --lr 0.001
