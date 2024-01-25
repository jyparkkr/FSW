from typing import Iterable, Optional
from cl_gym.algorithms import ContinualAlgorithm
from cl_gym.utils.callbacks import ContinualCallback
from cl_gym.utils.loggers import Logger
import torch
import numpy as np
from typing import Dict, Iterable, Optional
import cl_gym as cl


class ContinualTrainer_modified(cl.trainer.ContinualTrainer):
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
            for batch_idx, (inp, targ, task_ids, sample_weight) in enumerate(train_loader):
                if batch_idx == 0:
                    print(f"{sample_weight=}")
                self.on_before_training_step()
                self.tick('step')
                self.algorithm.training_step(task_ids.to(device), inp.to(device), targ.to(device), optimizer, criterion, sample_weight=sample_weight.to(device))
                self.algorithm.training_step_end()
                self.on_after_training_step()
            self.algorithm.training_epoch_end()
            self.on_after_training_epoch()
        self.algorithm.training_task_end()


    def validate_algorithm_on_task(self, task: int, validate_on_train: bool = False) -> Dict[str, float]:
        self.algorithm.backbone.eval()
        device = self.params['device']
        self.algorithm.backbone = self.algorithm.backbone.to(device)
        test_loss = 0
        correct = 0
        total = 0
        class_acc = dict()
        if validate_on_train:
            eval_loader = self.algorithm.prepare_train_loader(task)
        else:
            eval_loader = self.algorithm.prepare_validation_loader(task)
        criterion = self.algorithm.prepare_criterion(task)
        with torch.no_grad():
            for (inp, targ, task_ids, sample_weight) in eval_loader:
                inp, targ, task_ids = inp.to(device), targ.to(device), task_ids.to(device)
                # inp, targ, task_ids, sample_weight = inp.to(device), targ.to(device), task_ids.to(device), sample_weight.to(device)
                pred = self.algorithm.backbone(inp, task_ids)
                total += len(targ)
                test_loss += criterion(pred, targ).item()
                pred = pred.data.max(1, keepdim=True)[1]
                same = pred.eq(targ.data.view_as(pred))
                correct += same.sum()
                for t, s in zip(targ, same):
                    t = t.cpu().item()
                    s = s.cpu().item()
                    class_acc[t] = class_acc.get(t, np.array([0, 0])) + np.array([s, 1])

            test_loss /= total
            correct = correct.cpu()
            avg_acc = 100.0 * float(correct.numpy()) / total
            std = 100.0 * np.std([cor/count for cor, count in class_acc.values()])
            return {'accuracy': avg_acc, 'loss': test_loss, "std": std}

    # def validate_algorithm_on_task(self, task: int, validate_on_train: bool = False) -> Dict[str, float]:
    #     self.algorithm.backbone.eval()
    #     device = self.params['device']
    #     self.algorithm.backbone = self.algorithm.backbone.to(device)
    #     test_loss = 0
    #     correct = 0
    #     correct_0 = 0
    #     correct_1 = 0
    #     total = 0
    #     total_0 = 0
    #     total_1 = 0

    #     if validate_on_train:
    #         eval_loader = self.algorithm.prepare_train_loader(task)
    #     else:
    #         eval_loader = self.algorithm.prepare_validation_loader(task)
    #     criterion = self.algorithm.prepare_criterion(task)
    #     with torch.no_grad():
    #         for (inp, targ, task_ids, sample_weight) in eval_loader:
    #             inp, targ, task_ids, sample_weight = inp.to(device), targ.to(device), task_ids.to(device), sample_weight.to(device)
    #             pred = self.algorithm.backbone(inp, task_ids)
    #             total += len(targ)
    #             total_0 += torch.sum(sensitive == 0).cpu().item()
    #             total_1 += torch.sum(sensitive == 1).cpu().item()
    #             # print(f"{total=}, {total_0=}, {total_1=}")

                
    #             test_loss += criterion(pred, targ).item()
    #             pred = pred.data.max(1, keepdim=True)[1]
    #             correct_all = pred.eq(targ.data.view_as(pred)).reshape(-1)
    #             correct += correct_all.sum()
    #             correct_0 += correct_all.masked_fill(sensitive != 0, 0).sum()
    #             correct_1 += correct_all.masked_fill(sensitive != 1, 0).sum()
    #             # print(f"{correct=}, {correct_0=}, {correct_1=}")

    #         test_loss /= total
    #         correct = correct.cpu()
    #         correct_0 = correct_0.cpu()
    #         correct_1 = correct_1.cpu()
    #         avg_acc = 100.0 * float(correct.numpy()) / total
    #         avg_acc_0 = 100.0 * float(correct_0.numpy()) / total_0
    #         avg_acc_1 = 100.0 * float(correct_1.numpy()) / total_1
    #         return {'accuracy': avg_acc, 'accuracy_s0': avg_acc_0, 'accuracy_s1': avg_acc_1, 'loss': test_loss}
        
