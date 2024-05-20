# code imported and modified from https://github.com/XxidroxX/Incremental-Learning-iCarl
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from cl_gym.algorithms.utils import flatten_grads, assign_grads
from PIL import Image

import copy

from .baselines import BaseContinualAlgoritm

class iCaRL(BaseContinualAlgoritm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"iCaRL")
        # the number of gradient vectors to estimate new samples similarity, line 5 in alg.2
        self.device = self.params['device']
        self.mem_size = self.params['per_task_memory_examples'] * self.params['num_tasks']
        self.old_backbone = None
        self.exemplar_dict = {cls:list() for cls in self.benchmark.class_idx}
        self._modify_benchmark()

    def prepare_train_loader(self, task_id, epoch=0):
        return super().prepare_train_loader(task_id)

    def _modify_benchmark(self):
        num_tasks = self.benchmark.num_tasks
        self.memory_current_index = dict()
        for task in range(1, num_tasks+1):
            self.benchmark.memory_indices_train[task] = list()
            self.memory_current_index[task] = 0

    def _compute_loss(self, inp, target, criterion):
        pred = self.backbone(inp)
        num_classes = len(self.benchmark.class_idx)
        target = F.one_hot(target, num_classes).float()
        if self.old_backbone is None:
            return criterion(pred, target)
        else:
            with torch.no_grad():
                old_target = torch.sigmoid(self.old_backbone(inp))
            n_c = self.benchmark.class_idx[:num_classes//self.params['num_tasks']*(self.current_task-1)]
            target[:, n_c] = old_target[:, n_c]
            return criterion(pred, target)

    def training_step(self, task_ids, inp, targ, optimizer, criterion, sample_weight=None, sensitive_label=None):
        optimizer.zero_grad()
        loss = self._compute_loss(inp, targ, criterion)
        loss.backward()
        if (task_ids[0] > 1) and self.params['tau']:
            grad_batch = flatten_grads(self.backbone).detach().clone()
            optimizer.zero_grad()

            # get grad_ref
            inp_ref, targ_ref, task_ids_ref = self.sample_batch_from_memory()
            loss = self._compute_loss(inp_ref, targ_ref, criterion)
            # pred_ref = self.backbone(inp_ref, task_ids_ref)
            # loss = criterion(pred_ref, targ_ref.reshape(len(targ_ref)))
            loss.backward()
            grad_ref = flatten_grads(self.backbone).detach().clone()
            grad_batch += self.params['tau']*grad_ref

            optimizer.zero_grad()
            self.backbone = assign_grads(self.backbone, grad_batch)
        optimizer.step()

    def get_task_classes(self, task):
        if task < 1:
            raise AssertionError
        return self.benchmark.class_idx[\
            (task-1)*self.benchmark.num_classes_per_split:task*self.benchmark.num_classes_per_split]

    def get_current_classes(self):
        return self.get_task_classes(self.current_task)

    def update_memory_after_train(self):
        # update indices from self.exemplar_dict to benchmark.memory_indices_train[task]
        print(f"update_memory_after_train")
        for task in range(1, self.current_task+1):
            indices_task = list()
            target_classes = self.get_task_classes(task)
            for cls in target_classes:
                indices_task+=self.exemplar_dict[cls]
            
            # print(f"{task=}, len(memory) for task: {len(indices_task)=}")
            self.benchmark.memory_indices_train[task] = indices_task

    def training_epoch_end(self):
        self._compute_exemplar_class_mean()
        return super().training_epoch_end()

    def training_task_end(self):
        # iCaRL task
        self.backbone.eval()
        current_memory_per_class = self.mem_size // (self.current_task*self.benchmark.num_classes_per_split)
        # print(f"{current_memory_per_class=}")
        self._reduce_exemplar_dict(current_memory_per_class)
        # print(f"{self.get_current_classes()=}")
        current_dataloader = self.benchmark.trains[self.current_task]
        for cls in self.get_current_classes():
            print(f'construct class {cls} examplar:')
            target = (current_dataloader.targets == cls)
            target_indices = np.where(target == 1)[0]
            img, _ = current_dataloader.getitem_test_transform_list(target_indices)
            if np.sum(_ != cls) > 0:
                print(f"{cls=}")
                print(f"{current_dataloader.targets[self.exemplar_dict[cls]]=}")
                raise AssertionError

            self._construct_exemplar_set(img, cls, target_indices, current_memory_per_class)
        
        self.old_backbone = copy.deepcopy(self.backbone)
        self.old_backbone.eval()
        # update to cl_gym framework
        self.update_memory_after_train()
        super().training_task_end()

    def _compute_exemplar_class_mean(self):
        """
        Compute the mean of all the exemplars.
        :return: None
        """
        self.class_mean_dict = {k:None for k in self.exemplar_dict}
        self.backbone.eval()
        # update prev. samples
        for task in range(1, self.current_task):
            task_dataloader = self.benchmark.trains[task]
            for cls in self.get_task_classes(task):
                if np.sum(task_dataloader.targets[self.exemplar_dict[cls]] != cls) > 0:
                    print(f"{cls=}")
                    print(f"{task_dataloader.targets[self.exemplar_dict[cls]]=}")
                    raise AssertionError
                
                target_indices = self.exemplar_dict[cls]
                img, _ = task_dataloader.getitem_test_transform_list(target_indices)
                if np.sum(_ != cls) > 0:
                    print(f"{cls=}")
                    print(f"{task_dataloader.targets[self.exemplar_dict[cls]]=}")
                    raise AssertionError

                class_mean, _ = self.compute_class_mean(img)
                class_mean = class_mean.data / class_mean.norm() # why?
                self.class_mean_dict[cls] = class_mean
        for cls in self.get_current_classes():
            current_dataloader = self.benchmark.trains[self.current_task]
            target = (current_dataloader.targets == cls)
            target_indices = np.where(target == 1)[0]
            img, _ = current_dataloader.getitem_test_transform_list(target_indices)
            if np.sum(_ != cls) > 0:
                print(f"{cls=}")
                print(f"{current_dataloader.targets[self.exemplar_dict[cls]]=}")
                raise AssertionError

            class_mean, _ = self.compute_class_mean(img)
            class_mean = class_mean.data / class_mean.norm()
            self.class_mean_dict[cls] = class_mean

    def compute_class_mean(self, images):
        """
        Passo tutte le immagini di una determinata classe e faccio la media.
        :param special_transform:
        :param images: tutte le immagini della classe x
        :return: media della classe e features extractor.
        """
        self.backbone.eval()
        if isinstance(images[0], np.ndarray):
            images = [torch.from_numpy(img) for img in images]
        images = torch.stack(images).to(self.device)  # 500x3x32x32  #stack vs cat.
        with torch.no_grad():
            phi_X = torch.nn.functional.normalize(self.backbone.forward_embeds(images)[1])

        # phi_X.shape = 500x64
        mean = phi_X.mean(dim=0)
        mean.data = mean.data / mean.data.norm()
        return mean, phi_X

    def prototype_classifier(self, images):
        # batch_sizex3x32x32

        result = []
        self.backbone.eval()
        with torch.no_grad():
            phi_X = F.normalize(self.backbone.forward_embeds(images)[1])

        # 10x64 (di ogni classe mi salvo la media di ogni features)
        for x in phi_X:
            dist_class = dict()
            for cls in self.class_mean_dict:
                if self.class_mean_dict[cls] is not None:
                    dist_class[cls] = (self.class_mean_dict[cls] - x).norm()
            y = min(dist_class, key=dist_class.get)
            result.append(y)
        return torch.tensor(result).to(self.device)

    def _reduce_exemplar_dict(self, images_per_class):
        for cls in self.exemplar_dict:
            self.exemplar_dict[cls] = self.exemplar_dict[cls][:images_per_class]
            print(f"Reduce size of class {cls} to {len(self.exemplar_dict[cls])} examplar")

    def _construct_exemplar_set(self, images, cls, ind, mem_per_class):
        """
        Costruisco il set degli exemplar basato sugli indici.
        :param images: tutte le immagini di quella classe
        :param m: numero di immagini da salvare
        :return:
        """
        self.backbone.eval()
        if isinstance(images[0], np.ndarray):
            images = [torch.from_numpy(img) for img in images]
        images = torch.stack(images).to(self.device)
        with torch.no_grad():
            phi_X = torch.nn.functional.normalize(self.backbone.forward_embeds(images)[1]).cpu()

        mu_y = phi_X.mean(dim=0)  # vettore di 64 colonne
        mu_y.data = mu_y.data / mu_y.data.norm()

        Py = []
        # Accumulates sum of exemplars
        # sum_taken_exemplars = torch.zeros(1, 64)
        sum_taken_exemplars = torch.zeros(1, *phi_X.shape[1:])

        indexes = list()
        for k in range(1, int(mem_per_class + 1)):
            asd = F.normalize((1 / k) * (phi_X + sum_taken_exemplars))
            mean_distances = (mu_y - asd).norm(dim=1)
            used = -1
            sorted, _ = torch.sort(mean_distances)
            for item in sorted:
                mins = (mean_distances == item).nonzero()
                for j in mins: # in case of multiple same distance items
                    if j not in indexes:
                        indexes.append(j)
                        Py.append(ind[j])
                        used = j
                        sum_taken_exemplars += phi_X[j]
                        break
                if used != -1:
                    break
        self.exemplar_dict[cls] = Py
        print(f"{len(self.exemplar_dict[cls])=}")
