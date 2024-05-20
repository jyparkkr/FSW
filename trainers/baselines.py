from typing import Iterable, Optional
from cl_gym.algorithms import ContinualAlgorithm
from cl_gym.utils.callbacks import ContinualCallback
from cl_gym.utils.loggers import Logger
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Iterable, Optional
import cl_gym as cl
from .fair_trainer import process_for_biasedmnist
from .base import get_avg, avg_

class BaseContinualTrainer(cl.trainer.ContinualTrainer):
    def __init__(self,
                 algorithm: ContinualAlgorithm,
                 params: dict,
                 callbacks=Iterable[ContinualCallback],
                 logger: Optional[Logger] = None):
        super().__init__(algorithm, params, callbacks, logger)

    def on_before_training_task(self):
        super().on_before_training_task()
        if hasattr(self.algorithm, "before_training_task"):
            self.algorithm.before_training_task()

    def on_before_training_epoch(self):
        super().on_before_training_epoch()
        if hasattr(self.algorithm, "before_training_epoch"):
            self.algorithm.before_training_epoch()

    def train_algorithm_on_task(self, task: int):
        train_loader = self.algorithm.prepare_train_loader(task)
        optimizer = self.algorithm.prepare_optimizer(task)
        criterion = self.algorithm.prepare_criterion(task)
        device = self.params['device']
        for epoch in range(1, self.params['epochs_per_task']+1):
            self.on_before_training_epoch()
            self.tick('epoch')
            self.algorithm.backbone.train()
            self.algorithm.backbone = self.algorithm.backbone.to(device)
            for batch_idx, items in enumerate(train_loader):
                item_to_devices = [item.to(device) if isinstance(item, torch.Tensor) else item for item in items]
                inp, targ, task_ids, *_ = item_to_devices
                # if batch_idx == 0:
                #     print(f"{sample_weight.to(device)=}")
                self.on_before_training_step()
                self.tick('step')
                if epoch in self.params.get('learning_rate_decay_epoch', []): # decay
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr'] / 10
                self.algorithm.training_step(task_ids, inp, targ, \
                                             optimizer, criterion)
                self.algorithm.training_step_end()
                self.on_after_training_step()
            self.algorithm.training_epoch_end()
            self.on_after_training_epoch()
        self.algorithm.training_task_end()

    def validate_algorithm_on_task(self, task: int, validate_on_train: bool = False) -> Dict[str, float]:
        self.algorithm.backbone.eval()
        device = self.params['device']
        self.algorithm.backbone = self.algorithm.backbone.to(device)
        sen_dataset = (self.params['dataset'] not in ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"])
        test_loss = 0
        total = 0
        class_acc = dict()
        class_acc_s0 = dict()
        class_acc_s1 = dict()

        class_pred_count = dict()
        class_pred_count_s0 = dict()
        class_pred_count_s1 = dict()
        count, count_s0, count_s1 = 0, 0, 0

        if validate_on_train:
            eval_loader = self.algorithm.prepare_train_loader(task)
        else:
            eval_loader = self.algorithm.prepare_validation_loader(task)
        criterion = self.algorithm.prepare_criterion(task)
        with torch.no_grad():
            for items in eval_loader:
                item_to_devices = [item.to(device) if isinstance(item, torch.Tensor) else item for item in items]
                inp, targ, task_ids, _, _, sensitive_label, *_ = item_to_devices
                if criterion._get_name() != "BCEWithLogitsLoss":
                    pred = self.algorithm.backbone(inp)
                    total += len(targ)
                    test_loss += criterion(pred, targ).item()
                    pred = pred.data.max(1, keepdim=True)[1]
                    same = pred.eq(targ.data.view_as(pred))

                elif criterion._get_name() == "BCEWithLogitsLoss":
                    pred = self.algorithm.prototype_classifier(inp)
                    total += len(targ)
                    same = pred.eq(targ.data.view_as(pred))

                if self.algorithm.benchmark.__class__.__name__ == "BiasedMNIST":
                    sensitive_label = process_for_biasedmnist(sensitive_label, targ)

                for p, t, s, sen in zip(pred, targ, same, sensitive_label):
                    p = p.cpu().item()
                    t = t.cpu().item()
                    s = s.cpu().item()
                    sen = sen.cpu().item()
                    class_acc[t] = class_acc.get(t, np.array([0, 0])) + np.array([s, 1])
                    class_pred_count[p] = class_pred_count.get(p, 0) + 1
                    count+=1
                    if sen_dataset:
                        if sen == 0:
                            class_acc_s0[t] = class_acc_s0.get(t, np.array([0, 0])) + np.array([s, 1])
                            class_pred_count_s0[p] = class_pred_count_s0.get(p, 0) + 1
                            count_s0+=1
                        elif sen == 1:
                            class_acc_s1[t] = class_acc_s1.get(t, np.array([0, 0])) + np.array([s, 1])
                            class_pred_count_s1[p] = class_pred_count_s1.get(p, 0) + 1
                            count_s1+=1
                        else:
                            raise NotImplementedError


        test_loss /= total
        avg = np.mean([cor/count for cor, count in class_acc.values()])
        # cor, tot = np.array(list(class_acc.values())).sum(axis=0)
        std = np.std([cor/count for cor, count in class_acc.values()])
        # EER is updated by accuracy matrix, just update dummy value
        if sen_dataset:
            multiclass_eo = [abs(avg_(class_acc_s0[c]) - avg_(class_acc_s1[c])) for c in class_acc.keys()]
            # DP calculation required overall model prediction (data from other task can have prediction on current task)
            DP_ingredients = {"class_pred_count_s0":class_pred_count_s0, "class_pred_count_s1":class_pred_count_s1,\
                            "class_pred_count":class_pred_count, "count_s0":count_s0, "count_s1":count_s1, "count":count}
            accuracy_s0 = get_avg(class_acc_s0)
            accuracy_s1 = get_avg(class_acc_s1)
        else:
            multiclass_eo = -1
            DP_ingredients = dict()
            accuracy_s0 = -1
            accuracy_s1 = -1

        return {'accuracy': avg, 'loss': test_loss, "std": std, "EER": -1, 
                'EO': multiclass_eo, 'DP': -1, 
                'accuracy_s0': accuracy_s0, 'accuracy_s1': accuracy_s1, 
                'classwise_accuracy': class_acc, "DP_ingredients":DP_ingredients}

# training_step requires sample index
class BaseMemoryContinualTrainer(BaseContinualTrainer):
    def train_algorithm_on_task(self, task: int):
        train_loader = self.algorithm.prepare_train_loader(task)
        optimizer = self.algorithm.prepare_optimizer(task)
        criterion = self.algorithm.prepare_criterion(task)
        device = self.params['device']
        for epoch in range(1, self.params['epochs_per_task']+1):
            self.on_before_training_epoch()
            self.tick('epoch')
            self.algorithm.backbone.train()
            self.algorithm.backbone = self.algorithm.backbone.to(device)
            for batch_idx, items in enumerate(train_loader):
                item_to_devices = [item.to(device) if isinstance(item, torch.Tensor) else item for item in items]
                inp, targ, task_ids, indices, *_ = item_to_devices

                # if batch_idx == 0:
                #     print(f"{sample_weight.to(device)=}")
                self.on_before_training_step()
                self.tick('step')
                if epoch in self.params.get('learning_rate_decay_epoch', []): # decay
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr'] / 10
                self.algorithm.training_step(task_ids, inp, targ, \
                                             indices, optimizer, criterion)
                self.algorithm.training_step_end()
                self.on_after_training_step()
            self.algorithm.training_epoch_end()
            self.on_after_training_epoch()
        self.algorithm.training_task_end()

# while training_step also requires batch_idx
class BaseMemoryContinualTrainer2(BaseMemoryContinualTrainer):
    def train_algorithm_on_task(self, task: int):
        train_loader = self.algorithm.prepare_train_loader(task)
        optimizer = self.algorithm.prepare_optimizer(task)
        criterion = self.algorithm.prepare_criterion(task)
        device = self.params['device']
        for epoch in range(1, self.params['epochs_per_task']+1):
            self.on_before_training_epoch()
            self.tick('epoch')
            self.algorithm.backbone.train()
            self.algorithm.backbone = self.algorithm.backbone.to(device)
            for batch_idx, items in enumerate(train_loader):
                item_to_devices = [item.to(device) if isinstance(item, torch.Tensor) else item for item in items]
                inp, targ, task_ids, indices, *_ = item_to_devices
                # if batch_idx == 0:
                #     print(f"{sample_weight.to(device)=}")
                self.on_before_training_step()
                self.tick('step')
                if epoch in self.params.get('learning_rate_decay_epoch', []): # decay
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr'] / 10
                self.algorithm.training_step(task_ids, inp, targ, \
                                             indices, optimizer, criterion, batch_idx)
                self.algorithm.training_step_end()
                self.on_after_training_step()
            self.algorithm.training_epoch_end()
            self.on_after_training_epoch()
        self.algorithm.training_task_end()

# for OCS - online training 
class BaseMemoryContinualTrainer3(BaseMemoryContinualTrainer):
    def train_algorithm_on_task(self, task: int):
        train_loader = self.algorithm.prepare_train_loader(task)
        optimizer = self.algorithm.prepare_optimizer(task)
        criterion = self.algorithm.prepare_criterion(task)
        # config['n_substeps'] = int(config['seq_epochs'] * (config['stream_size'] / config['batch_size']))
        stream_size_divided_by_batch_size = 2
        # if "mnist" in self.params['dataset'].lower():
        #     stream_size_divided_by_batch_size = 10
        n_substeps = int(self.params['epochs_per_task'] * stream_size_divided_by_batch_size)
        for _step in range(1, n_substeps+1):
            # if config['coreset_base'] and task > 1:
            self.on_before_training_epoch()
            if (_step-1) % stream_size_divided_by_batch_size == 0:                
                self.tick('epoch')
            if False and task > 1:
                pass
                # model = train_coreset_single_step(model, optimizer, train_loader, task, _step, config)
            # elif task == 1 or (config['ocspick'] == False):
            elif task == 1 or (True == False):
                self.algorithm.train_single_step(optimizer, criterion, train_loader, task, _step, n_substeps)
                # model = train_single_step(model, optimizer, train_loader, task, _step, config)
            else:
                self.algorithm.train_ocs_single_step(optimizer, criterion, train_loader, task, _step, n_substeps)
                # model = train_ocs_single_step(model, optimizer, train_loader, task, _step, config)
            self.algorithm.training_epoch_end()
            self.on_after_training_epoch()
            # metrics = eval_single_epoch(model, train_loader['sequential'][task]['val'], config)
            # print('Epoch {} >> (per-task accuracy): {}'.format(_step/config['n_substeps'], np.mean(metrics['accuracy'])))
            # print('Epoch {} >> (class accuracy): {}'.format(_step/config['n_substeps'], metrics['per_class_accuracy']))
        self.algorithm.training_task_end()
        
