from typing import Iterable, Optional
from cl_gym.algorithms import ContinualAlgorithm
from cl_gym.utils.callbacks import ContinualCallback
from cl_gym.utils.loggers import Logger
import torch
import numpy as np
from typing import Dict, Iterable, Optional
import cl_gym as cl


class BaseContinualTrainer(cl.trainer.ContinualTrainer):
    def __init__(self,
                 algorithm: ContinualAlgorithm,
                 params: dict,
                 callbacks=Iterable[ContinualCallback],
                 logger: Optional[Logger] = None):
        super().__init__(algorithm, params, callbacks, logger)

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
                inp, targ, task_ids, *_ = items
                # if batch_idx == 0:
                #     print(f"{sample_weight.to(device)=}")
                self.on_before_training_step()
                self.tick('step')
                self.algorithm.training_step(task_ids.to(device), inp.to(device), targ.to(device), \
                                             optimizer, criterion)
                self.algorithm.training_step_end()
                self.on_after_training_step()
            self.algorithm.training_epoch_end()
            self.on_after_training_epoch()
        self.algorithm.training_task_end()

    def validate_algorithm_on_task(self, task: int, validate_on_train: bool = False) -> Dict[str, float]:
        self.algorithm.backbone.eval()
        device = self.params['device']
        num_classes_per_split = self.algorithm.benchmark.num_classes_per_split
        self.algorithm.backbone = self.algorithm.backbone.to(device)
        test_loss = 0
        total = 0
        class_acc = dict()
        if validate_on_train:
            eval_loader = self.algorithm.prepare_train_loader(task)
            classes = self.algorithm.benchmark.class_idx[task*(num_classes_per_split-1):task*num_classes_per_split]
        else:
            eval_loader = self.algorithm.prepare_validation_loader(task)
            classes = self.algorithm.benchmark.class_idx[:task*num_classes_per_split]
        criterion = self.algorithm.prepare_criterion(task)
        with torch.no_grad():
            for items in eval_loader:
                inp, targ, task_ids, *_ = items
                inp, targ, task_ids = inp.to(device), targ.to(device), task_ids.to(device)
                # inp, targ, task_ids, sample_weight = inp.to(device), targ.to(device), task_ids.to(device), sample_weight.to(device)
                pred = self.algorithm.backbone(inp, task_ids)
                total += len(targ)
                test_loss += criterion(pred, targ).item()
                pred = pred.data.max(1, keepdim=True)[1]
                same = pred.eq(targ.data.view_as(pred))
                for t, s in zip(targ, same):
                    t = t.cpu().item()
                    s = s.cpu().item()
                    class_acc[t] = class_acc.get(t, np.array([0, 0])) + np.array([s, 1])

        test_loss /= total
        avg = 100.0 * np.mean([cor/count for cor, count in class_acc.values()])
        # cor, tot = np.array(list(class_acc.values())).sum(axis=0)
        std = 100.0 * np.std([cor/count for cor, count in class_acc.values()])
        return {'accuracy': avg, 'loss': test_loss, "std": std}