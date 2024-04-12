import argparse
import os
import torch
import numpy as np
import subprocess
import copy
from datasets import __all__ as dataset_list, fairness_dataset

model_pool = ["MLP", "resnet18", "resnet18small"]
optimizer_pool = ["sgd", "adam"]
algorithm_pool = ["optimization", "greedy"]
metric_pool = ["std", "EO"]
method_pool = ['FSW', "FSS", 'joint', 'finetune', 'AGEM', "GSS"]

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--dataset', type=str, default='MNIST', choices=dataset_list)
    parser.add_argument('--model', type=str, default='MLP', choices=model_pool)
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for torch and np.')
    parser.add_argument('--method', type=str, default="FSW", choices=method_pool,
                        help='Seed for torch and np.')

    # benchmark
    parser.add_argument('--num_tasks', type=int, default=5, metavar='N',
                        help='Number of total tasks.')
    parser.add_argument('--epochs_per_task', type=int, default=1, metavar='N',
                        help='Epochs for each task.')
    parser.add_argument('--per_task_examples', type=int, default=np.inf, metavar='N',
                        help='Number of samples for each task.')
    parser.add_argument('--per_task_memory_examples', type=int, default=64, metavar='N',
                        help='Number of samples stored in memory for each task.')
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='train_batch_size',
                        help='Size of train batch.')
    parser.add_argument('--batch_size_memory', type=int, default=64, metavar='train_batch_size',
                        help='Size of memory batch.')
    parser.add_argument('--batch_size_validation', type=int, default=256, metavar='train_batch_size',
                        help='Size of validation batch.')
    parser.add_argument('--random_class_idx', action='store_true',
                        help='Randomize class order')
    parser.add_argument('--tau', type=float, default=5,
                        help='Hyperparameter of update weight on memory.')

    # optimizing
    parser.add_argument('--optimizer', type=str, default="sgd", choices=optimizer_pool,
                        help='Type of optimizer to use.')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='momentum.')
    parser.add_argument('--learning_rate_decay', type=float, default=1.0, 
                        help='lr decay.')
    
    # sample selection/weighting
    parser.add_argument('--metric', type=str, default="std", choices=metric_pool,
                        help='Target metric for sample selection problem.')
    parser.add_argument('--fairness_agg', type=str, default="mean", choices=['mean', 'max'],
                        help='Aggregation measure for fairness metric.')
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='Update rate for sample selection problem.')
    parser.add_argument('--lambda', type=float, default=0.0,
                        help='Hyperparameter of loss for sample selection problem.')
    parser.add_argument('--lambda_old', type=float, default=0.0,
                        help='Hyperparameter of old class loss for sample selection problem.')
    
    # etc
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda device.')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2, 3],
                        help='0 -> 1 -> 2 -> 3')

    opt = parser.parse_args()

    # raise NotImplemented
    if opt.metric == "EO":
        if opt.dataset not in fairness_dataset:
            raise ValueError(f"Wrong dataset({opt.dataset}) for corresponding metric({opt.metric}).")
        
    return opt

def make_params(args) -> dict:
    from pathlib import Path

    params = dict()
    for arg in vars(args):
        params[arg] = getattr(args, arg)
        if args.verbose > 1:
            print(f"\t\"{arg}\" : \"{params[arg]}\", \\")

    params['device'] = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    params['criterion'] = torch.nn.CrossEntropyLoss()
    trial_id = f"dataset={params['dataset']}"

    if params['random_class_idx']:
        trial_id+="_randidx"

    # define method
    if params['num_tasks'] == 1:
        params['method'] = 'joint'
    elif params['tau'] == 0:
        params['method'] = 'finetune'
    elif params['alpha'] == 0:
        params['method'] = 'vanilla'

    if params['method'] == 'joint':
        params['num_tasks'] == 1
    elif params['method'] == 'finetune':
        params['tau'] == 0
    elif params['method'] == 'vanilla':
        params['alpha'] == 0
    trial_id = os.path.join(trial_id, f"{params['method']}")

    # add hyperparameters
    trial_id = os.path.join(trial_id, \
                            f"seed={params['seed']}_model={params['model']}_epoch={params['epochs_per_task']}_lr={params['learning_rate']}")
    # contain tau (buffer parameter) if buffer is used
    if params['method'] in ["FSS", "FSW", "GSS"]: 
        trial_id += f"_tau={params['tau']}"
    # contain alpha for current data selection
    if params['method'] in ["FSS", "FSW"]: 
        trial_id += f"_alpha={params['alpha']}"
    if params['lambda'] != 0:
        trial_id+=f"_lmbd={params['lambda']}_lmbdold={params['lambda_old']}"
    params['trial_id'] = trial_id
    params['output_dir'] = os.path.join("./outputs/{}".format(trial_id))
    if args.verbose > 1:
        print(f"output_dir={params['output_dir']}")
    Path(params['output_dir']).mkdir(parents=True, exist_ok=True)
    
    return params