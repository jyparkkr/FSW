import torch
from typing import Dict, Iterable, Optional
import cl_gym as cl
from .base import ContinualTrainer1
import numpy as np

class FairContinualTrainer1(ContinualTrainer1):
    # Basic frame of FairContinualTrainer
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
                inp, targ, task_ids, _, sample_weight, sensitive_label, *_ = item_to_devices
                # if batch_idx == 0:
                #     print(f"{sample_weight.to(device)=}")
                self.on_before_training_step()
                self.tick('step')
                self.algorithm.training_step(task_ids, inp, targ, optimizer, criterion, \
                                             sample_weight=sample_weight, sensitive_label=sensitive_label) #ADDED
                self.algorithm.training_step_end()
                self.on_after_training_step()
            self.algorithm.training_epoch_end()
            self.on_after_training_epoch()
        self.algorithm.training_task_end()

    def validate_algorithm_on_task(self, task: int, validate_on_train: bool = False) -> Dict[str, float]:
        # TODO: 일반적인 fairness dataset일 때에 대해서 사용할 수 있도록 수정하기 (바꾸긴 했는데 검증 필요)
        self.algorithm.backbone.eval()
        device = self.params['device']
        self.algorithm.backbone = self.algorithm.backbone.to(device)
        test_loss = 0
        total = 0
        class_acc = dict()
        class_acc_s0 = dict()
        class_acc_s1 = dict()

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
                    pred = self.algorithm.backbone(inp, task_ids)
                    total += len(targ)
                    test_loss += criterion(pred, targ).item()
                    pred = pred.data.max(1, keepdim=True)[1]
                    same = pred.eq(targ.data.view_as(pred))
                elif criterion._get_name() == "BCEWithLogitsLoss":
                    pred = self.algorithm.prototype_classifier(inp)
                    total += len(targ)
                    same = pred.eq(targ.data.view_as(pred))

                for t, s, sen in zip(targ, same, sensitive_label):
                    t = t.cpu().item()
                    s = s.cpu().item()
                    sen = sen.cpu().item()
                    class_acc[t] = class_acc.get(t, np.array([0, 0])) + np.array([s, 1])
                    if sen == 0:
                        class_acc_s0[t] = class_acc_s0.get(t, np.array([0, 0])) + np.array([s, 1])
                    elif sen == 1:
                        class_acc_s1[t] = class_acc_s1.get(t, np.array([0, 0])) + np.array([s, 1])
                    else:
                        raise NotImplementedError

        test_loss /= total
        get_avg = lambda cor_count: np.mean([cor/count for cor, count in cor_count.values()])
        avg_ = lambda cor_count: cor_count[0]/cor_count[1]
        multiclass_eo = [abs(avg_(class_acc_s0[c]) - avg_(class_acc_s1[c])) for c in class_acc.keys()]
        # TODO: implement DP
        multiclass_dp = None
        return {'accuracy': get_avg(class_acc), 'loss': test_loss, 'EO': multiclass_eo, 'DP': multiclass_dp, \
                'accuracy_s0': get_avg(class_acc_s0), 'accuracy_s1': get_avg(class_acc_s1), \
                'classwise_accuracy': class_acc}
        
class FairContinualTrainer2(FairContinualTrainer1):
    def train_algorithm_on_task(self, task: int):
        # train_loader = self.algorithm.prepare_train_loader(task)
        optimizer = self.algorithm.prepare_optimizer(task)
        criterion = self.algorithm.prepare_criterion(task)
        device = self.params['device']
        for epoch in range(1, self.params['epochs_per_task']+1):
            self.on_before_training_epoch()
            self.tick('epoch')
            train_loader = self.algorithm.prepare_train_loader(task, epoch=epoch) # modified
            self.algorithm.backbone.train()
            self.algorithm.backbone = self.algorithm.backbone.to(device)
            for batch_idx, items in enumerate(train_loader):
                item_to_devices = [item.to(device) if isinstance(item, torch.Tensor) else item for item in items]
                inp, targ, task_ids, indices, sample_weight, sensitive_label, *_ = item_to_devices
                # if batch_idx == 0:
                #     print(f"{sample_weight.to(device)=}")
                self.on_before_training_step()
                self.tick('step')
                if epoch in self.params.get('learning_rate_decay_epoch', []): # decay
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr'] / 10
                self.algorithm.training_step(task_ids, inp, targ, optimizer, criterion, \
                                             sample_weight=sample_weight, sensitive_label=sensitive_label) #ADDED
                self.algorithm.training_step_end()
                self.on_after_training_step()
            self.algorithm.training_epoch_end()
            self.on_after_training_epoch()
        self.algorithm.training_task_end()

class FairContinualTrainer3(FairContinualTrainer2): # Trainer for BiasedMNIST set
    def validate_algorithm_on_task(self, task: int, validate_on_train: bool = False) -> Dict[str, float]:
        self.algorithm.backbone.eval()
        device = self.params['device']
        self.algorithm.backbone = self.algorithm.backbone.to(device)
        test_loss = 0
        total = 0
        class_acc = dict()
        class_acc_s0 = dict()
        class_acc_s1 = dict()

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
                    pred = self.algorithm.backbone(inp, task_ids)
                    total += len(targ)
                    test_loss += criterion(pred, targ).item()
                    pred = pred.data.max(1, keepdim=True)[1]
                    same = pred.eq(targ.data.view_as(pred))
                elif criterion._get_name() == "BCEWithLogitsLoss":
                    pred = self.algorithm.prototype_classifier(inp)
                    total += len(targ)
                    same = pred.eq(targ.data.view_as(pred))

                sensitive = torch.ne(targ, sensitive_label)
                sensitive = sensitive.long()
                for t, s, sen in zip(targ, same, sensitive):
                    t = t.cpu().item()
                    s = s.cpu().item()
                    sen = sen.cpu().item()
                    class_acc[t] = class_acc.get(t, np.array([0, 0])) + np.array([s, 1])
                    if sen == 0:
                        class_acc_s0[t] = class_acc_s0.get(t, np.array([0, 0])) + np.array([s, 1])
                    elif sen == 1:
                        class_acc_s1[t] = class_acc_s1.get(t, np.array([0, 0])) + np.array([s, 1])
                    else:
                        raise NotImplementedError

        test_loss /= total
        # def get_avg(cor_count):
        #     return 100.0 * np.mean([cor/count for cor, count in cor_count.values()])
        get_avg = lambda cor_count: np.mean([cor/count for cor, count in cor_count.values()])
        # avg = get_avg(class_acc)
        # cor, tot = np.array(list(class_acc.values())).sum(axis=0)
        avg_ = lambda cor_count: cor_count[0]/cor_count[1]
        # multiclass_eo = max([abs(avg_(class_acc_s0[c]) - avg_(class_acc_s1[c])) for c in class_acc.keys()])
        multiclass_eo = [abs(avg_(class_acc_s0[c]) - avg_(class_acc_s1[c])) for c in class_acc.keys()]
        # TODO
        multiclass_dp = None
        return {'accuracy': get_avg(class_acc), 'loss': test_loss, 'EO': multiclass_eo, 'DP': multiclass_dp, \
                'accuracy_s0': get_avg(class_acc_s0), 'accuracy_s1': get_avg(class_acc_s1), \
                'classwise_accuracy': class_acc}
