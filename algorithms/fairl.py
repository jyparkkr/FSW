# code imported and modified from https://github.com/brcsomnath/FaIRL

"""
adversarial debiasing for removing demographic information by controlling the number of bits (rate-distortion)
required to encode the learned representations
    - ΔR(Z, 𝛱): diff between #of bits to encode when membership information is provided or not.
    - encoder: discriminative for the target attribute y and not informative about protected attribute
    * max ΔR(Z, 𝛱y) - 𝛽ΔR(Z', 𝛱g)
    - discriminator: input as representations preduced by feature encoder and generates protected attribute(g)
    * max ΔR(Z', 𝛱g)
    - how: loss (include buffer loss)
    * buffer: ΔR(Z_old, Z^bar_old) = ΣΔR(Zi, Zi) = ΣR(Z∪Z) - 1/2 [R(Z) + R(Z)]
    * meaning: minimize information difference for the exemplars
    * Zold: representation for the exemplars
    * Z^bar: examplar representations of previous training stage
    * Zi: examplar representation of ith class
    - 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.nn.functional import relu, avg_pool2d
from typing import Optional, Dict, Iterable

import copy
from cl_gym.algorithms.utils import flatten_grads, assign_grads

from .base import Heuristic
from .sensitive import Heuristic3

class IdentityNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        return x

class Net(nn.Module):
    """
    Dynamic MLP network with ReLU non-linearity
    """
    def __init__(self, embedding_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(embedding_size, embedding_size))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(embedding_size, embedding_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MCR(nn.Module):
    def __init__(self,
                 gam1=1.,
                 gam2=1.,
                 gam3=1.,
                 eps=0.5,
                 numclasses=1000,
                 mode=1,
                 num_protected_class=2):
        super(MCR, self).__init__()

        self.num_class = numclasses
        self.num_protected_class = num_protected_class
        self.train_mode = mode
        self.faster_logdet = False
        self.gam1 = gam1
        self.gam2 = gam2
        self.gam3 = gam3
        self.eps = eps

    def logdet(self, X):
        if self.faster_logdet:
            return 2 * torch.sum(
                torch.log(torch.diag(torch.linalg.cholesky(X, upper=True))))
        else:
            return torch.logdet(X)

    def compute_discrimn_loss(self, Z):
        """Theoretical Discriminative Loss."""
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        scalar = d / (n * self.eps)
        logdet = self.logdet(I + scalar * Z @ Z.T)
        return logdet / 2.

    def compute_compress_loss(self, Z, Pi):
        """Theoretical Compressive Loss."""
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        compress_loss = []
        scalars = []
        for j in range(Pi.shape[1]):
            Z_ = Z[:, Pi[:, j] == 1]
            trPi = Pi[:, j].sum() + 1e-8
            scalar = d / (trPi * self.eps)
            log_det = 1. if Pi[:, j].sum() == 0 else self.logdet(I + scalar *
                                                                 Z_ @ Z_.T)
            compress_loss.append(log_det)
            scalars.append(trPi / (2 * n))
        return compress_loss, scalars

    def deltaR(self, Z, Y, num_classes):
        Y_new = torch.zeros_like(Y)
        for i, e in enumerate(Y.unique()):
            Y_new[Y==e] = i
        
        Pi = F.one_hot(Y_new, num_classes).to(Z.device)
        discrimn_loss = self.compute_discrimn_loss(Z.T)
        compress_loss, scalars = self.compute_compress_loss(Z.T, Pi)

        compress_term = 0.
        for z, s in zip(compress_loss, scalars):
            compress_term += s * z
        total_loss = discrimn_loss - compress_term

        return -total_loss, (discrimn_loss, compress_term, compress_loss, scalars)
    

class FaIRL(Heuristic3):
    def __init__(self, backbone, benchmark, params, **kwargs):
        self.backbone = backbone
        self.benchmark = benchmark
        self.params = params
        self.device = self.params['device']
        self.mem_size = self.params['per_task_memory_examples']
        super().__init__(backbone, benchmark, params, **kwargs)

    def prepare_train_loader(self, task_id):
        # num_workers = self.params.get('num_dataloader_workers', torch.get_num_threads())
        # return self.benchmark.load(task_id, self.params['batch_size_train'],
        #                            num_workers=num_workers, pin_memory=True)[0]
        return super(Heuristic, self).prepare_train_loader(task_id)

    def before_training_task(self):
        """
        called before loader, optimizer, criterion initialized
        """
        # init
        if not hasattr(self, "embedding_size"):
            if self.params['model'] == "MLP":
                self.embedding_size = 256
            elif self.params['model'] == "resnet18small":
                self.embedding_size = 160
            elif self.params['model'] == "resnet18":
                self.embedding_size = 1024
            else:
                raise NotImplementedError
            
            if "MNIST" in self.params['dataset']:
                self.num_target_class = 10
                self.num_protected_class = 10
            elif "10" in self.params['dataset']:
                self.num_target_class = 10
            elif "100" in self.params['dataset']:
                self.num_target_class = 100
            else:
                raise NotImplementedError

            # etc hyperparameters
            self.num_layers = 3 # #layers of Discriminator, Generator
            self.beta = 0.01
            self.gamma = 1.
            self.eta = 0.01
            self.LN = nn.LayerNorm(self.embedding_size, elementwise_affine=False)

            self.netG = Net(self.embedding_size, self.num_layers)
            # self.netG = IdentityNet()
            self.netG.to(self.device)

            def forward_with_netG(inp: torch.Tensor, head_ids: Optional[Iterable] = None):
                _, embeds = self.backbone.forward_embeds(inp, head_ids=head_ids)
                Z = self.netG(embeds)
                out = self.backbone.forward_classifier(Z, head_ids=head_ids)
                return out
            self.backbone.forward = forward_with_netG


        # do for each task
        self.netD = Net(self.embedding_size, self.num_layers)
        self.netD.to(self.device)

        self.mcr_loss = MCR(numclasses=self.num_target_class)

    def before_training_epoch(self):
        self.netD.train()
        self.netG.train()

    def prepare_optimizer(self, task_id):
        if self.params['model'] == "MLP":
            last_layer_name = "blocks.2.layers"
        elif self.params['model'] == "resnet18":
            last_layer_name = "linear"
        else:
            raise NotImplementedError
        
        if task_id >= 0:
            self.current_task = task_id
        backbone_parmas = []
        classifier_params = []
        for name, param in self.backbone.named_parameters():
            if last_layer_name not in name:
                backbone_parmas.append(param)
            else:
                classifier_params.append(param)

        optim_G = torch.optim.Adam([{"params": backbone_parmas},
                                    {"params": self.netG.parameters()}],
                                    lr=2e-5,
                                    betas=(0.5, 0.999))
                                #    lr=0.001,
                                #    betas=(0.9, 0.999),
                                #    eps=1e-8)

        optim_D = torch.optim.Adam([{"params": self.netD.parameters()}],
                                   lr=2e-5,
                                   betas=(0.5, 0.999))
                                #    lr=0.001,
                                #    betas=(0.9, 0.999),
                                #    eps=1e-8)
        optim_C = torch.optim.Adam([{"params": classifier_params}],
                                   lr=0.001,
                                   betas=(0.9, 0.999),
                                   eps=1e-8)
        return {"optim_G":optim_G, "optim_D":optim_D, "optim_C":optim_C}

    def training_step(self, task_ids, inp, targ, optimizer, criterion, sample_weight=None, sensitive_label=None):
        optim_D = optimizer['optim_D']
        optim_G = optimizer['optim_G']

        # update discriminator first
        optim_D.zero_grad()
        optim_G.zero_grad()

        _, embeds = self.backbone.forward_embeds(inp, task_ids)
        Z = self.netG(embeds)
        Z_bar = self.netD(Z.detach())

        disc_loss, comp = self.mcr_loss.deltaR(self.LN(Z_bar), sensitive_label, self.num_protected_class)
        disc_loss.backward()
        optim_D.step()

        # update generator
        optim_D.zero_grad()
        optim_G.zero_grad()

        _, embeds = self.backbone.forward_embeds(inp, task_ids)
        Z = self.netG(embeds)
        Z_bar = self.netD(Z.detach())

        task_loss, _ = self.mcr_loss.deltaR(self.LN(Z), targ, task_ids[0].item() * self.benchmark.num_classes_per_split)
        bias_loss, _ = self.mcr_loss.deltaR(self.LN(Z_bar), sensitive_label, self.num_protected_class)
        loss = task_loss - self.beta * bias_loss

        old_loss = 0.
        old_bias_loss = 0.
        if (task_ids[0] > 1):
            inp_ref, targ_ref, task_ids_ref, *_, sensitive_label_ref = self.sample_batch_from_memory()
            unique_ref = torch.unique(targ_ref)
            _, prev_embeds_ref = self.previous_backbone.forward_embeds(inp_ref, task_ids_ref)
            Z_old = self.previous_netG(prev_embeds_ref).detach()
            _, (_, _, z_old_losses, _) = self.mcr_loss.deltaR(self.LN(Z_old), targ_ref, len(unique_ref))

            _, embeds_ref = self.backbone.forward_embeds(inp_ref, task_ids_ref)
            Z = self.netG(embeds_ref)
            _, (R_z, _, z_losses, _) = self.mcr_loss.deltaR(self.LN(Z), targ_ref, len(unique_ref))

            R_zjzjold = 0.
            for j in unique_ref:
                new_z = torch.cat((Z[targ_ref == j], Z_old[targ_ref == j]), 0)
                R_zjzjold += self.mcr_loss.compute_discrimn_loss(self.LN(new_z).T)

            old_loss = R_zjzjold - 0.25 * sum(z_losses) - 0.25 * sum(z_old_losses)
            loss += self.gamma * old_loss

            Z_bar = self.netD(Z)
            old_bias_loss, _ = self.mcr_loss.deltaR(self.LN(Z_bar), sensitive_label_ref, self.num_protected_class)
            loss = loss - self.eta * old_bias_loss

        loss.backward()
        optim_G.step()

    def training_epoch_end(self):
        super().training_epoch_end()
        task = self.current_task
        optimizer = self.prepare_optimizer(task) 
        train_loader = self.prepare_train_loader(task)
        criterion = self.prepare_criterion(task)
        optim_C = optimizer['optim_C']
        for _ in range(5): # 5 iter train for fc classifier
            for items in train_loader:
                inp, targ, task_ids, *_ = items
                inp,targ, task_ids = inp.to(self.device), targ.to(self.device), task_ids.to(self.device)
                super().training_step(task_ids, inp, targ, optim_C, criterion)
        
    def training_task_end(self):
        super().training_task_end()
        self.previous_netG = self.netG
        self.previous_backbone = copy.deepcopy(self.backbone)