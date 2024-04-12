import torch
import numpy as np

import random
import copy
import os
import importlib
import pickle

import cl_gym as cl
from configs import parse_option, make_params, dataset_list

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
    # torch.backends.cudnn.enabled = False

    if opt.cuda is not None:
        torch.cuda.set_device(opt.cuda)
        if opt.verbose > 0:
            print(f"torch.cuda.set_device({opt.cuda})")

    params = make_params(opt)

    # dataset
    from datasets import MNIST, FashionMNIST, BiasedMNIST, CIFAR10, CIFAR100
    if params['dataset'] == 'MNIST':
        benchmark = MNIST(num_tasks=params['num_tasks'],
                        per_task_memory_examples=params['per_task_memory_examples'],
                        per_task_examples = params['per_task_examples'],
                        random_class_idx = params['random_class_idx'])
        input_dim = (28, 28)
    elif params['dataset'] == 'FashionMNIST':
        benchmark = FashionMNIST(num_tasks=params['num_tasks'],
                                per_task_memory_examples=params['per_task_memory_examples'],
                                per_task_examples = params['per_task_examples'],
                                random_class_idx = params['random_class_idx'])
        input_dim = (28, 28)
    elif params['dataset'] == 'CIFAR10':
        benchmark = CIFAR10(num_tasks=params['num_tasks'],
                            per_task_memory_examples=params['per_task_memory_examples'],
                            per_task_examples = params['per_task_examples'],
                            random_class_idx = params['random_class_idx'])
        input_dim = (3, 32, 32)
    elif params['dataset'] == 'CIFAR100':        
        benchmark = CIFAR100(num_tasks=params['num_tasks'],
                            per_task_memory_examples=params['per_task_memory_examples'],
                            per_task_examples = params['per_task_examples'],
                            random_class_idx = params['random_class_idx'])
        input_dim = (3, 32, 32)
    elif params['dataset'] in ["BiasedMNIST"]:
        benchmark = BiasedMNIST(num_tasks=params['num_tasks'],
                                per_task_memory_examples=params['per_task_memory_examples'],
                                per_task_examples = params['per_task_examples'],
                                random_class_idx = params['random_class_idx'])
        input_dim = (3, 28, 28)
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
    elif params['model'] == "resnet18small": 
        from backbones import ResNet18Small2
        backbone = ResNet18Small2(
            input_dim=input_dim, 
            output_dim=num_classes,
            class_idx=class_idx,
            config=params
            ).to(params['device'])
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

    # load algorithm, metric, trainer
    fairness_metrics = ['EO']
    
    if params['metric'] == "std":
        fair_metric = 'std'
        from metrics import MetricCollector2
        from trainers import ContinualTrainer
        MetricCollector = MetricCollector2
        Trainer = ContinualTrainer

        if params['method'] in ["FSW", 'joint', 'finetune']:
            from algorithms.imbalance import Heuristic2
            Algorithm = Heuristic2
        elif params['method'] in ["FSS"]:
            from algorithms.imbalance_greedy import Heuristic1
            Algorithm = Heuristic1
        elif params['method'] in ["AGEM"]:
            from algorithms.agem import AGEM
            Algorithm = AGEM
        elif params['method'] in ["GSS"]:
            from algorithms.gss import GSSGreedy
            from trainers.baselines import BaseMemoryContinualTrainer
            Algorithm = GSSGreedy
            Trainer = BaseMemoryContinualTrainer
        else:
            raise NotImplementedError
        
    elif params['metric'] == "EO":
        fair_metric = 'multiclass_eo'
        from metrics import FairMetricCollector
        from trainers.fair_trainer import FairContinualTrainer2
        MetricCollector = FairMetricCollector
        Trainer = FairContinualTrainer2

        if params['method'] == "FSW":
            from algorithms.sensitive import Heuristic3
            Algorithm = Heuristic3
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    algorithm = Algorithm(backbone, benchmark, params, requires_memory=True)
    metric_manager_callback = MetricCollector(num_tasks=params['num_tasks'],
                                                eval_interval='epoch',
                                                epochs_per_task=params['epochs_per_task'])
    trainer = Trainer(algorithm, params, callbacks=[metric_manager_callback])


    # optimization parameter fix
    if params['metric'] in ["EO"]:
        if params['fairness_agg'] == "mean":
            agg = np.mean
        elif params['fairness_agg'] == "max":
            agg = np.max
        else:
            raise NotImplementedError
        metric_manager_callback.meters[fair_metric].agg = agg

    # run & save & log metrics
    trainer.run()
    print(f"accuracy:{np.mean(metric_manager_callback.meters['accuracy'].compute_overall())}")
    print(f"fairness:{np.mean(metric_manager_callback.meters[fair_metric].compute_overall())}")

    with open(os.path.join(params['output_dir'], 'metrics', 'metrics.pickle'), "wb") as f:
        pickle.dump(metric_manager_callback, f)

    with open(os.path.join(params['output_dir'], 'plots', 'output.txt'), "w") as f:
        print(f"accuracy matrix:\n{metric_manager_callback.meters['accuracy'].get_data()}", file=f)
        print(f"avg. acc:\n{np.mean(metric_manager_callback.meters['accuracy'].compute_overall())}", file=f)
        print(f"avg. fairness:\n{np.mean(metric_manager_callback.meters[fair_metric].compute_overall())}", file=f)

        if params['metric'] in fairness_metrics:
            print(f"=======classwise {params['metric']}========",end=" ",file=f)
            for eos in metric_manager_callback.meters[fair_metric].get_data():
                for eo in eos:
                    print(round(eo, 3),end=" ",file=f)
                print(file=f)
        

if __name__ == '__main__':
    main()
