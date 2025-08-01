from typing import Iterable, Optional
from cl_gym.algorithms import ContinualAlgorithm
from cl_gym.utils.callbacks import ContinualCallback
from cl_gym.utils.loggers import Logger
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Iterable, Optional
import cl_gym as cl
import os
import pickle
from pathlib import Path
from .fair_trainer import process_for_biasedmnist
from .base import ContinualTrainer1
from .base import get_avg, avg_

class BaseContinualTrainer(cl.trainer.ContinualTrainer):
    def __init__(self,
                 algorithm: ContinualAlgorithm,
                 params: dict,
                 callbacks=Iterable[ContinualCallback],
                 logger: Optional[Logger] = None):
        super().__init__(algorithm, params, callbacks, logger)
        self.post_processing = params.get("post_processing", False)
        if self.post_processing:
            print(f"{self.post_processing=}")

    def on_before_training_task(self):
        super().on_before_training_task()
        if hasattr(self.algorithm, "before_training_task"):
            self.algorithm.before_training_task()

    def on_before_training_epoch(self):
        super().on_before_training_epoch()
        if hasattr(self.algorithm, "before_training_epoch"):
            self.algorithm.before_training_epoch()

    def on_after_training_epoch(self):
        super().on_after_training_epoch()
        # save the model
        if self.params.get("save_model", False):
            if self.current_epoch % self.params['epochs_per_task'] == 0: # task ends
                model_path = os.path.join(self.params['output_dir'], 'models')
                Path(model_path).mkdir(parents=True, exist_ok=True)
                model_path = os.path.join(model_path, f"epoch={self.current_epoch}.pth")
                torch.save(self.algorithm.backbone.state_dict(), model_path)

        if hasattr(self.algorithm, "after_training_epoch"):
            self.algorithm.after_training_epoch()

    def train_algorithm_on_task(self, task: int):
        train_loader = self.algorithm.prepare_train_loader(task)
        optimizer = self.algorithm.prepare_optimizer(task)
        criterion = self.algorithm.prepare_criterion(task)
        device = self.params['device']
        for epoch in range(1, self.params['epochs_per_task']+1):
            self.on_before_training_epoch()
            self.tick('epoch')
            self.epoch = epoch
            self.algorithm.backbone.train()
            self.algorithm.backbone = self.algorithm.backbone.to(device)
            for batch_idx, items in enumerate(train_loader):
                item_to_devices = [item.to(device) if isinstance(item, torch.Tensor) else item for item in items]
                inp, targ, task_ids, *_ = item_to_devices
                if isinstance(inp, list):
                    inp = [x.to(device) for x in inp]
                # if batch_idx == 0:
                #     print(f"{sample_weight.to(device)=}")
                self.on_before_training_step()
                self.tick('step')
                if epoch in self.params.get('learning_rate_decay_epoch', []): # decay
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr'] / 10
                self.algorithm.training_step(task_ids, inp, targ, \
                                             optimizer, criterion)
                if isinstance(inp, list):
                    del inp[:]
                del inp, targ

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


        if self.post_processing and self.epoch == self.params['epochs_per_task']:
            print(f"{self.post_processing} start")
            train_loader, _ = self.algorithm.benchmark.load(task, 256, shuffle=False)
            with torch.no_grad():
                X_train, targets, sens, probs = list(), list(), list(), list()
                for items in train_loader:
                    item_to_devices = [item.to(device) if isinstance(item, torch.Tensor) else item for item in items]
                    # TODO: For datasets above CIFAR, test_transform should be applied
                    inp, targ, task_ids, _, _, sensitive_label, *_ = item_to_devices
                    if isinstance(inp, list):
                        inp = [x.to(device) for x in inp]
                        inp = torch.stack(inp, 1)
                    if self.algorithm.benchmark.__class__.__name__ == "BiasedMNIST":
                        sensitive_label = process_for_biasedmnist(sensitive_label, targ)

                    if criterion._get_name() != "BCEWithLogitsLoss":
                        prob = self.algorithm.backbone(inp)

                    elif criterion._get_name() == "BCEWithLogitsLoss":
                        prob = self.algorithm.prototype_classifier_prob(inp)
                    
                    X_train.append(inp.reshape(inp.shape[0], -1))
                    targets.append(targ)
                    sens.append(sensitive_label)
                    probs.append(prob)
            
            X_train = torch.concat(X_train).to("cpu").numpy()
            targets = torch.concat(targets).to("cpu").numpy()
            sens_train = torch.concat(sens).to("cpu").numpy()
            probs_train = torch.concat(probs).to("cpu").numpy()

        with torch.no_grad():
            for items in eval_loader:
                item_to_devices = [item.to(device) if isinstance(item, torch.Tensor) else item for item in items]
                inp, targ, task_ids, _, _, sensitive_label, *_ = item_to_devices
                if isinstance(inp, list):
                    inp = [x.to(device) for x in inp]
                    # inp = torch.stack(inp, 1)

                if criterion._get_name() != "BCEWithLogitsLoss":
                    prob = self.algorithm.backbone(inp)
                    total += len(targ)
                    test_loss += criterion(prob, targ).item()
                    pred = prob.data.max(1, keepdim=True)[1]
                    same = pred.eq(targ.data.view_as(pred))

                elif criterion._get_name() == "BCEWithLogitsLoss":
                    pred = self.algorithm.prototype_classifier(inp)
                    prob = self.algorithm.prototype_classifier_prob(inp)

                    total += len(targ)
                    same = pred.eq(targ.data.view_as(pred))

                if self.algorithm.benchmark.__class__.__name__ == "BiasedMNIST":
                    sensitive_label = process_for_biasedmnist(sensitive_label, targ)

                if self.post_processing and self.epoch == self.params['epochs_per_task']:
                    if isinstance(inp, list):
                        inp = torch.stack(inp, 1)
                    X_test = inp.reshape(inp.shape[0], -1).to("cpu").numpy()
                    prob_test = prob.to("cpu").numpy()
                    sen_test = sensitive_label.to("cpu").numpy()

                    if self.post_processing == "eps_fairness":
                        # n_classes = task * self.algorithm.benchmark.num_classes_per_split
                        n_classes = len(self.algorithm.benchmark.class_idx)
                        from algorithms.postprocessing.epsilon_fairness.multiclass_fairness import run_fairness_experimentation
                        eps = self.params.get('pp_eps', None)
                        pred_pp = run_fairness_experimentation(None, X_test, epsilon_fair=eps, Xtrain=X_train, prob_train=probs_train, \
                                                            sen_train=sens_train, prob=prob_test, sen=sen_test, n_classes=n_classes)
                        pred = torch.from_numpy(pred_pp).view_as(pred)
                        targ = targ.cpu()
                        same = pred.eq(targ.data.view_as(pred))

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

class BaseContinualTrainerPP(BaseContinualTrainer):
    def train_algorithm_on_task(self, task: int):
        # load trained model
        if not self.params.get("load_model", False):
            return super().train_algorithm_on_task(task)
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        print(f"Load saved model")
        target_epoch = int(np.ceil((self.current_epoch + 1) / self.params['epochs_per_task']) * self.params['epochs_per_task'])
        output_root = self.params['output_dir']
        if self.params['output_dir'][9:14] == "/demo":
            output_root = self.params['output_dir'][:9]+self.params['output_dir'][14:]
        if "DP" in output_root and self.params['method'] != "FSW":
            idx = output_root.find("DP")
            output_root = output_root[:idx]+"no_metrics"+output_root[idx+2:]
        eps="_eps_fairness"
        if eps in output_root:
            idx = output_root.find(eps)
            output_root = output_root[:idx]+output_root[idx+len(eps):]
        eps="_eps="
        if eps in output_root:
            idx = output_root.find(eps)
            output_root = output_root[:idx]
        model_path = os.path.join(output_root, 'models', f"epoch={target_epoch}.pth")
        self.algorithm.backbone.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.params['device']))
        self.algorithm.backbone = self.algorithm.backbone.to(self.params['device'])
        for epoch in range(1, self.params['epochs_per_task']+1):
            self.on_before_training_epoch()
            self.tick('epoch')
            self.epoch = epoch
            self.algorithm.training_epoch_end()
            if epoch == self.params['epochs_per_task']:
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

        if self.post_processing and self.epoch == self.params['epochs_per_task']:
            print(f"{self.post_processing} start")
            train_loader, _ = self.algorithm.benchmark.load(task, 256, shuffle=False)
            with torch.no_grad():
                X_train, targets, sens, probs = list(), list(), list(), list()
                for items in train_loader:
                    item_to_devices = [item.to(device) if isinstance(item, torch.Tensor) else item for item in items]
                    # TODO: Need test-transform to CIFAR or more complex image datasets
                    inp, targ, task_ids, _, _, sensitive_label, *_ = item_to_devices
                    if isinstance(inp, list):
                        inp = [x.to(device) for x in inp]
                        inp = torch.stack(inp, 1)
                    if self.algorithm.benchmark.__class__.__name__ == "BiasedMNIST":
                        sensitive_label = process_for_biasedmnist(sensitive_label, targ)

                    if criterion._get_name() != "BCEWithLogitsLoss":
                        prob = self.algorithm.backbone(inp)

                    elif criterion._get_name() == "BCEWithLogitsLoss":
                        prob = self.algorithm.prototype_classifier_prob(inp)
                    
                    X_train.append(inp.reshape(inp.shape[0], -1))
                    targets.append(targ)
                    sens.append(sensitive_label)
                    probs.append(prob)
            
            X_train = torch.concat(X_train).to("cpu").numpy()
            targets = torch.concat(targets).to("cpu").numpy()
            sens_train = torch.concat(sens).to("cpu").numpy()
            probs_train = torch.concat(probs).to("cpu").numpy()

        with torch.no_grad():
            for items in eval_loader:
                item_to_devices = [item.to(device) if isinstance(item, torch.Tensor) else item for item in items]
                inp, targ, task_ids, _, _, sensitive_label, *_ = item_to_devices
                if isinstance(inp, list):
                    inp = [x.to(device) for x in inp]
                    # inp = torch.stack(inp, 1)

                if criterion._get_name() != "BCEWithLogitsLoss":
                    prob = self.algorithm.backbone(inp)
                    total += len(targ)
                    test_loss += criterion(prob, targ).item()
                    pred = prob.data.max(1, keepdim=True)[1]
                    same = pred.eq(targ.data.view_as(pred))

                elif criterion._get_name() == "BCEWithLogitsLoss":
                    pred = self.algorithm.prototype_classifier(inp)
                    prob = self.algorithm.prototype_classifier_prob(inp)

                    total += len(targ)
                    same = pred.eq(targ.data.view_as(pred))

                if self.algorithm.benchmark.__class__.__name__ == "BiasedMNIST":
                    sensitive_label = process_for_biasedmnist(sensitive_label, targ)

                if self.post_processing and self.epoch == self.params['epochs_per_task']:
                    if isinstance(inp, list):
                        inp = torch.stack(inp, 1)
                    X_test = inp.reshape(inp.shape[0], -1).to("cpu").numpy()
                    prob_test = prob.to("cpu").numpy()
                    sen_test = sensitive_label.to("cpu").numpy()

                    if self.post_processing == "eps_fairness":
                        # n_classes = task * self.algorithm.benchmark.num_classes_per_split
                        n_classes = len(self.algorithm.benchmark.class_idx)
                        from algorithms.postprocessing.epsilon_fairness.multiclass_fairness import run_fairness_experimentation
                        eps = self.params.get('pp_eps', None)
                        pred_pp = run_fairness_experimentation(None, X_test, epsilon_fair=eps, Xtrain=X_train, prob_train=probs_train, \
                                                            sen_train=sens_train, prob=prob_test, sen=sen_test, n_classes=n_classes)
                        pred = torch.from_numpy(pred_pp).view_as(pred)
                        targ = targ.cpu()
                        same = pred.eq(targ.data.view_as(pred))

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
            self.epoch = epoch
            self.algorithm.backbone.train()
            self.algorithm.backbone = self.algorithm.backbone.to(device)
            for batch_idx, items in enumerate(train_loader):
                item_to_devices = [item.to(device) if isinstance(item, torch.Tensor) else item for item in items]
                inp, targ, task_ids, indices, *_ = item_to_devices
                if isinstance(inp, list):
                    inp = [x.to(device) for x in inp]
                # if batch_idx == 0:
                #     print(f"{sample_weight.to(device)=}")
                self.on_before_training_step()
                self.tick('step')
                self.epoch = epoch
                if epoch in self.params.get('learning_rate_decay_epoch', []): # decay
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr'] / 10
                self.algorithm.training_step(task_ids, inp, targ, indices, optimizer, criterion)
                self.algorithm.training_step_end()
                self.on_after_training_step()
            self.algorithm.training_epoch_end()
            self.on_after_training_epoch()
        self.algorithm.training_task_end()

class BaseMemoryContinualTrainerPP(BaseMemoryContinualTrainer):
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

        if self.post_processing and self.epoch == self.params['epochs_per_task']:
            print(f"{self.post_processing} start")
            train_loader, _ = self.algorithm.benchmark.load(task, 256, shuffle=False)
            with torch.no_grad():
                X_train, targets, sens, probs = list(), list(), list(), list()
                for items in train_loader:
                    item_to_devices = [item.to(device) if isinstance(item, torch.Tensor) else item for item in items]
                    # TODO: Need test-transform to CIFAR or more complex image datasets
                    inp, targ, task_ids, _, _, sensitive_label, *_ = item_to_devices
                    if isinstance(inp, list):
                        inp = [x.to(device) for x in inp]
                        inp = torch.stack(inp, 1)
                    if self.algorithm.benchmark.__class__.__name__ == "BiasedMNIST":
                        sensitive_label = process_for_biasedmnist(sensitive_label, targ)

                    if criterion._get_name() != "BCEWithLogitsLoss":
                        prob = self.algorithm.backbone(inp)

                    elif criterion._get_name() == "BCEWithLogitsLoss":
                        prob = self.algorithm.prototype_classifier_prob(inp)
                    
                    X_train.append(inp.reshape(inp.shape[0], -1))
                    targets.append(targ)
                    sens.append(sensitive_label)
                    probs.append(prob)
            
            X_train = torch.concat(X_train).to("cpu").numpy()
            targets = torch.concat(targets).to("cpu").numpy()
            sens_train = torch.concat(sens).to("cpu").numpy()
            probs_train = torch.concat(probs).to("cpu").numpy()

        with torch.no_grad():
            for items in eval_loader:
                item_to_devices = [item.to(device) if isinstance(item, torch.Tensor) else item for item in items]
                inp, targ, task_ids, _, _, sensitive_label, *_ = item_to_devices
                if isinstance(inp, list):
                    inp = [x.to(device) for x in inp]
                    # inp = torch.stack(inp, 1)

                if criterion._get_name() != "BCEWithLogitsLoss":
                    prob = self.algorithm.backbone(inp)
                    total += len(targ)
                    test_loss += criterion(prob, targ).item()
                    pred = prob.data.max(1, keepdim=True)[1]
                    same = pred.eq(targ.data.view_as(pred))

                elif criterion._get_name() == "BCEWithLogitsLoss":
                    pred = self.algorithm.prototype_classifier(inp)
                    prob = self.algorithm.prototype_classifier_prob(inp)

                    total += len(targ)
                    same = pred.eq(targ.data.view_as(pred))

                if self.algorithm.benchmark.__class__.__name__ == "BiasedMNIST":
                    sensitive_label = process_for_biasedmnist(sensitive_label, targ)

                if self.post_processing and self.epoch == self.params['epochs_per_task']:
                    if isinstance(inp, list):
                        inp = torch.stack(inp, 1)
                    X_test = inp.reshape(inp.shape[0], -1).to("cpu").numpy()
                    prob_test = prob.to("cpu").numpy()
                    sen_test = sensitive_label.to("cpu").numpy()

                    if self.post_processing == "eps_fairness":
                        # n_classes = task * self.algorithm.benchmark.num_classes_per_split
                        n_classes = len(self.algorithm.benchmark.class_idx)
                        from algorithms.postprocessing.epsilon_fairness.multiclass_fairness import run_fairness_experimentation
                        eps = self.params.get('pp_eps', None)
                        pred_pp = run_fairness_experimentation(None, X_test, epsilon_fair=eps, Xtrain=X_train, prob_train=probs_train, \
                                                            sen_train=sens_train, prob=prob_test, sen=sen_test, n_classes=n_classes)
                        pred = torch.from_numpy(pred_pp).view_as(pred)
                        targ = targ.cpu()
                        same = pred.eq(targ.data.view_as(pred))

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
            self.epoch = epoch
            self.algorithm.backbone.train()
            self.algorithm.backbone = self.algorithm.backbone.to(device)
            for batch_idx, items in enumerate(train_loader):
                item_to_devices = [item.to(device) if isinstance(item, torch.Tensor) else item for item in items]
                inp, targ, task_ids, indices, *_ = item_to_devices
                if isinstance(inp, list):
                    inp = [x.to(device) for x in inp]
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
        device = self.params['device']
        # config['n_substeps'] = int(config['seq_epochs'] * (config['stream_size'] / config['batch_size']))
        stream_size_divided_by_batch_size = 2
        n_substeps = int(self.params['epochs_per_task'] * stream_size_divided_by_batch_size)
        for _step in range(1, n_substeps+1):
            # if config['coreset_base'] and task > 1:
            self.on_before_training_epoch()
            self.epoch = _step // stream_size_divided_by_batch_size
            self.algorithm.backbone = self.algorithm.backbone.to(device)
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
        self.algorithm.training_task_end()
        
