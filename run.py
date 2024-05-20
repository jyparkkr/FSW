import torch
import numpy as np

import random
import copy
import os
import importlib
import pickle

import cl_gym as cl
from configs import parse_option, make_params, datasets, fairness_datasets

def main():
    opt = parse_option()
    seed = opt.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) 
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(6)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


    if opt.cuda is not None:
        torch.cuda.set_device(opt.cuda)
        if opt.verbose > 0:
            print(f"torch.cuda.set_device({opt.cuda})")

    params = make_params(opt)
    params['num_dataloader_workers'] = torch.get_num_threads()

    # dataset
    from datasets import MNIST, FashionMNIST, BiasedMNIST, CIFAR10, CIFAR100, Drug
    if params['dataset'] == 'MNIST':
        benchmark = MNIST(num_tasks=params['num_tasks'],
                          per_task_memory_examples=params['per_task_memory_examples'],
                          per_task_examples = params['per_task_examples'],
                          joint = (params['method'] == "joint"),
                          random_class_idx = params['random_class_idx'])
        input_dim = (28, 28)
    elif params['dataset'] == 'FashionMNIST':
        benchmark = FashionMNIST(num_tasks=params['num_tasks'],
                                 per_task_memory_examples=params['per_task_memory_examples'],
                                 per_task_examples = params['per_task_examples'],
                                 joint = (params['method'] == "joint"),
                                 random_class_idx = params['random_class_idx'])
        input_dim = (28, 28)
    elif params['dataset'] == 'CIFAR10':
        benchmark = CIFAR10(num_tasks=params['num_tasks'],
                            per_task_memory_examples=params['per_task_memory_examples'],
                            per_task_examples = params['per_task_examples'],
                            joint = (params['method'] == "joint"),
                            random_class_idx = params['random_class_idx'])
        input_dim = (3, 32, 32)
    elif params['dataset'] == 'CIFAR100':        
        benchmark = CIFAR100(num_tasks=params['num_tasks'],
                             per_task_memory_examples=params['per_task_memory_examples'],
                             per_task_examples = params['per_task_examples'],
                             joint = (params['method'] == "joint"),
                             random_class_idx = params['random_class_idx'])
        input_dim = (3, 32, 32)
    elif params['dataset'] in ["BiasedMNIST"]:
        benchmark = BiasedMNIST(num_tasks=params['num_tasks'],
                                per_task_memory_examples=params['per_task_memory_examples'],
                                per_task_examples = params['per_task_examples'],
                                joint = (params['method'] == "joint"),
                                random_class_idx = params['random_class_idx'])
        input_dim = (3, 28, 28)
    elif params['dataset'] in ["Drug"]:
        benchmark = Drug(num_tasks=params['num_tasks'],
                         per_task_memory_examples=params['per_task_memory_examples'],
                         per_task_examples = params['per_task_examples'],
                         joint = (params['method'] == "joint"),
                         random_class_idx = params['random_class_idx'])
        input_dim = (12)
    else:
        raise NotImplementedError
    class_idx = benchmark.class_idx
    num_classes = len(class_idx)

    # load backbone, 
    if params['model'] == "MLP": 
        from backbones import MLP2Layers2
        backbone = MLP2Layers2(
            input_dim=input_dim, 
            hidden_dim_1=256, 
            hidden_dim_2=256, 
            output_dim=num_classes,
            class_idx=class_idx,
            config=params
            ).to(params['device'])
    # elif params['model'] == "resnet18small": 
    #     from backbones import ResNet18Small2
    #     backbone = ResNet18Small2(
    #         input_dim=input_dim, 
    #         output_dim=num_classes,
    #         class_idx=class_idx,
    #         config=params
    #         ).to(params['device'])
    elif params['model'] == "resnet18": 
        from backbones import ResNet18
        backbone = ResNet18(
            input_dim=input_dim, 
            output_dim=num_classes,
            class_idx=class_idx,
            config=params
            ).to(params['device'])
    else:
        raise NotImplementedError

    # load metric, trainer
    fairness_metrics = ["std", "EER", "EO", "DP"]
    if params['dataset'] not in fairness_datasets:
        from metrics import MetricCollector2 as MetricCollector
        from trainers.imbalance_trainer import ImbalanceContinualTrainer1 as ContinualTrainer
    else:
        from metrics import FairMetricCollector as MetricCollector
        from trainers.fair_trainer import FairContinualTrainer2 as ContinualTrainer

    # load algorithm & trainer for other baselines
    if params['method'] in ['vanilla', "FSW", 'joint', 'finetune']:
        if params['dataset'] not in fairness_datasets:
            from algorithms.imbalance import Heuristic2 as Algorithm
        elif params['dataset'] in fairness_datasets:
            from algorithms.sensitive import Heuristic3 as Algorithm
    elif params['method'] in ["FSS"]:
        from algorithms.imbalance_greedy import Heuristic1 as Algorithm
    elif params['method'] in ["AGEM"]:
        from algorithms.agem import AGEM as Algorithm
    elif params['method'] in ["GSS"]:
        from algorithms.gss import GSSGreedy as Algorithm
        from trainers.baselines import BaseMemoryContinualTrainer as ContinualTrainer
    elif params['method'] in ["FaIRL"]:
        from algorithms.fairl import FaIRL as Algorithm
        from trainers.fair_trainer import FairContinualTrainer1 as ContinualTrainer
    elif params['method'] in ["OCS"]:
        from algorithms.ocs import OCS as Algorithm
        from trainers.baselines import BaseMemoryContinualTrainer3 as ContinualTrainer
    elif params['method'] in ["WA"]:
        from algorithms.wa import WA as Algorithm
        from trainers.baselines import BaseContinualTrainer as ContinualTrainer
    elif params['method'] in ["iCaRL"]:
        from algorithms.icarl import iCaRL as Algorithm
        from trainers.baselines import BaseContinualTrainer as ContinualTrainer
        # for GSS, batch size should be smaller than per_task_memory size
    else:
        print(f"{params['method']=}")
        raise NotImplementedError


    algorithm = Algorithm(backbone, benchmark, params, requires_memory=True)
    metric_manager_callback = MetricCollector(num_tasks=params['num_tasks'],
                                                eval_interval='epoch',
                                                epochs_per_task=params['epochs_per_task'])
    trainer = ContinualTrainer(algorithm, params, callbacks=[metric_manager_callback])


    # optimization parameter fix
    if params['fairness_agg'] == "mean":
        agg = np.mean
    elif params['fairness_agg'] == "max":
        agg = np.max
    else:
        raise NotImplementedError
    for metric in metric_manager_callback.meters:
        if metric in fairness_metrics:
            metric_manager_callback.meters[metric].agg = agg
            
    # run & save & log metrics
    trainer.run()
    print(f"accuracy:{np.mean(metric_manager_callback.meters['accuracy'].compute_overall())}")
    for metric in metric_manager_callback.meters:
        if metric in fairness_metrics:
            print(f"{metric}:\n{np.mean(metric_manager_callback.meters[metric].compute_overall())}")        

    with open(os.path.join(params['output_dir'], 'metrics', 'metrics.pickle'), "wb") as f:
        pickle.dump(metric_manager_callback, f)

    with open(os.path.join(params['output_dir'], 'plots', 'output.txt'), "w") as f:
        for metric in metric_manager_callback.meters:
            if metric == "accuracy":
                print(f"accuracy matrix:\n{metric_manager_callback.meters['accuracy'].get_data()}", file=f)
                print(f"avg. acc:\n{np.mean(metric_manager_callback.meters['accuracy'].compute_overall())}", file=f)
            elif metric in fairness_metrics:
                print(f"avg. {metric}:\n{np.mean(metric_manager_callback.meters[metric].compute_overall())}", file=f)        

if __name__ == '__main__':
    main()
