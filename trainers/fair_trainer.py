from typing import Iterable, Optional
from cl_gym.algorithms import ContinualAlgorithm
from cl_gym.utils.callbacks import ContinualCallback
from cl_gym.utils.loggers import Logger
import torch
from typing import Dict, Iterable, Optional
import cl_gym as cl
from .trainer import ContinualTrainer3
import numpy as np

class FairContinualTrainer(ContinualTrainer3):
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
                inp, targ, task_ids, sample_weight, sensitive_label, *_ = items
                inp, targ, task_ids, sensitive_label = inp.to(device), targ.to(device), task_ids.to(device), sensitive_label.to(device)
                pred = self.algorithm.backbone(inp, task_ids)
                total += len(targ)
                test_loss += criterion(pred, targ).item()
                pred = pred.data.max(1, keepdim=True)[1]
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
        get_avg = lambda cor_count: 100.0 * np.mean([cor/count for cor, count in cor_count.values()])
        # avg = get_avg(class_acc)
        # cor, tot = np.array(list(class_acc.values())).sum(axis=0)
        avg_ = lambda cor_count: cor_count[0]/cor_count[1]
        # multiclass_eo = max([abs(avg_(class_acc_s0[c]) - avg_(class_acc_s1[c])) for c in class_acc.keys()])
        multiclass_eo = [abs(avg_(class_acc_s0[c]) - avg_(class_acc_s1[c])) for c in class_acc.keys()]

        return {'accuracy': get_avg(class_acc), 'loss': test_loss, 'multiclass_eo': multiclass_eo, \
                'accuracy_s0': get_avg(class_acc_s0), 'accuracy_s1': get_avg(class_acc_s1), \
                'classwise_accuracy': class_acc}
        
