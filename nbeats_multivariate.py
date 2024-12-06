import os
import pickle
import time
import argparse
import pandas as pd

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from src.losses.numpy import mae, mse
from src.experiments.utils import hyperopt_tunning

# Define hyperparameter search space
def get_experiment_space(args):
    space = {
        # Architecture parameters
        'model': 'nbeats',
        'mode': 'simple',
        'n_time_in': hp.choice('n_time_in', [5 * args.horizon]),
        'n_time_out': hp.choice('n_time_out', [args.horizon]),
        'n_x_hidden': hp.choice('n_x_hidden', [0]),
        'n_s_hidden': hp.choice('n_s_hidden', [0]),
        'shared_weights': hp.choice('shared_weights', [True]),
        'activation': hp.choice('activation', ['ReLU']),
        'initialization': hp.choice('initialization', ['lecun_normal']),
        'stack_types': hp.choice('stack_types', [3 * ['identity']]),
        'n_blocks': hp.choice('n_blocks', [3 * [1]]),
        'n_layers': hp.choice('n_layers', [3 * [2]]),
        'n_hidden': hp.choice('n_hidden', [512]),
        'batch_normalization': hp.choice('batch_normalization', [False]),
        # Regularization and optimization parameters
        'dropout_prob_theta': hp.choice('dropout_prob_theta', [0]),
        'learning_rate': hp.choice('learning_rate', [0.001]),
        'lr_decay': hp.choice('lr_decay', [0.5]),
        'n_lr_decays': hp.choice('n_lr_decays', [3]),
        'weight_decay': hp.choice('weight_decay', [0]),
        'max_epochs': hp.choice('max_epochs', [100]),
        'max_steps': hp.choice('max_steps', [1000]),
        'early_stop_patience': hp.choice('early_stop_patience', [10]),
        'eval_freq': hp.choice('eval_freq', [50]),
        'loss_train': hp.choice('loss', ['MAE']),
        'loss_hypar': hp.choice('loss_hypar', [0.5]),
        'loss_valid': hp.choice('loss_valid', ['MAE']),
        'random_seed': hp.quniform('random_seed', 1, 10, 1),
        # Data parameters
        'normalizer_y': hp.choice('normalizer_y', [None]),
        'normalizer_x': hp.choice('normalizer_x', [None]),
        'batch_size': hp.choice('batch_size', [4]),
        'complete_windows': hp.choice('complete_windows', [True]),
        'frequency': hp.choice('frequency', ['H']),
        'seasonality': hp.choice('seasonality', [24]),
        'idx_to_sample_freq': hp.choice('idx_to_sample_freq', [1]),
        'val_idx_to_sample_freq': hp.choice('val_idx_to_sample_freq', [1]),
        'n_windows': hp.choice('n_windows', [256]),
        'n_harmonics': hp.choice('n_harmonics', [1, 2, 4, 8, 16]),
        'n_polynomials': hp.choice('n_polynomials', [1, 2, 4, 8])

    }
    return space


def main(args):
    # Load data
    Y_df = pd.read_csv(f'./data/{args.dataset}/M/df_y.csv')
    X_df = None
    S_df = None

    print("Y_df: ", Y_df.head())

    # Train: 60%, Test: 20%, Val: 20%
    if args.dataset == 'ETTm2':
        len_val = 11520
        len_test = 11520

    # Define hyperparameter space
    space = get_experiment_space(args)

    # Create output directory
    output_dir = f'./results/multivariate/{args.dataset}_{args.horizon}/NBEATS/'
    os.makedirs(output_dir, exist_ok=True)
    assert os.path.exists(output_dir), f"Output dir {output_dir} does not exist"

    # Hyperparameter tuning
    hyperopt_file = output_dir + f'hyperopt_{args.experiment_id}.p'

    if not os.path.isfile(hyperopt_file):
        print("Hyperparameter optimization")
        trials = hyperopt_tunning(
            space=space,
            hyperopt_max_evals=args.hyperopt_max_evals,
            loss_function_val=mae,
            loss_functions_test={"mae": mae, "mse": mse},
            Y_df=Y_df,
            X_df=X_df,
            S_df=S_df,
            f_cols=[],
            evaluate_train=True,
            ds_in_val=len_val,
            ds_in_test=len_test,
            return_forecasts=False,
            results_file=hyperopt_file,
            save_progress=True,
            loss_kwargs={},
        )
        with open(hyperopt_file, "wb") as f:
            pickle.dump(trials, f)
    else:
        print("Hyperparameter optimization already done!")


def parse_args():
    desc = "NBEATS Hyperparameter Tuning"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--hyperopt_max_evals', type=int, help='Max evaluations for Hyperopt')
    parser.add_argument('--experiment_id', type=str, default=None, help='Experiment ID for tracking results')
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    if args is None:
        exit()

    horizons = [96, 192, 336, 720]
    datasets = ['ETTm2']  # You can add more datasets as needed

    for dataset in datasets:
        for horizon in horizons:
            print(50 * "-", dataset, 50 * "-")
            print(50 * "-", horizon, 50 * "-")
            start = time.time()
            args.dataset = dataset
            args.horizon = horizon
            main(args)
            print("Time: ", time.time() - start)

    main(args)
