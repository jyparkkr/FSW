import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import random


class NormalNN(nn.Module):
    """NormalNN.
    
    Attributes:
        input_features: Number of input features.
        n_class: Number of classes.
        seed: Random seed number.
    """
    def __init__(self, input_features, n_class, seed):
        """Initialize NormalNN."""
        super(NormalNN, self).__init__()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        self.input_features = input_features
        
        self.linear1 = nn.Linear(self.input_features, 256)
        self.linear2 = nn.Linear(256, n_class)        

    def forward(self, x):
        """Forward pass of the model.
        
        Args:
            x: Input data.
        Returns:
            Model output.
        """
        x1 = x.view(-1,self.input_features)
        x2 = F.relu(self.linear1(x1))
        x3 = self.linear2(x2)
        return x3, x2
        
        
class EarlyStopping:
    """EarlyStopping.
    
    Attributes:
        patience: Number of possible epochs with no improvement.
        delta: Minimum change to qualify as an improvement.
        path: Path to save model checkpoint.
    """
    def __init__(self, patience=10, delta=0, path='checkpoint.pt'):
        """Initialize EarlyStopping."""
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss, model):
        """Monitor the improvement of the model.
        
        Args:
            val_loss: Validation loss.
            model: Model.
        Returns:
            None.
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        """Save the model checkpoint.
        
        Args:
            val_loss: Validation loss.
            model: Trained model.
        Returns:
            None.
        """
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        

class NNClassifier:
    """NNClassifier.
    
    Attributes:
        model: Train model.
        criterion: Train criterion.
        optimizer: Train optimizer.
        optimizer_config: Train optimizer config.
    """
    def __init__(self, model, criterion, optimizer, optimizer_config):
        """Initialize NNClassifier."""
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), **optimizer_config)
        self.criterion = criterion
        self.LOSS = {'train': [], 'val': []}
        
    def fit(self, loader, epochs, earlystop_path):
        """Train the model with evaluation using validation set.
        
        Args:
            loader: Data loader.
            epochs: Train epochs.
            earlystop_path: Earlystop model checkpoint path.
        Returns:
            Index of minimum validation loss.
        """
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10)
        early_stopping = EarlyStopping(patience=10, delta=0.001, path=earlystop_path)
        
        for epoch in range(epochs):
            total = 0.0
            tloss = 0
            self.model.train()
            for x, y in loader["train"]:
                total += y.shape[0]
                self.optimizer.zero_grad()
                outputs, _ = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                
                self.optimizer.step()
                
                tloss += loss.item()
            
            self.LOSS['train'].append(tloss/total)
            
            with torch.no_grad():
                val_correct = 0.0
                val_total = 0.0
                vloss = 0
                self.model.eval()
                for x_val, y_val in loader["val"]:
                    val_total += y_val.shape[0]
                    val_output, _ = self.model(x_val)
                    val_loss = self.criterion(val_output, y_val)
                    vloss += val_loss.item()*y_val.shape[0]

                self.LOSS['val'].append(vloss/val_total)
                
                _, val_pred = val_output.max(1)
                val_true = y_val.reshape(-1,1)
                val_correct += (val_pred == val_true).sum().item()
                        
            scheduler.step(self.LOSS['val'][-1])
            early_stopping(self.LOSS['val'][-1], self.model)
            
            if early_stopping.early_stop:
                break
        
        self.model.load_state_dict(torch.load(earlystop_path))
        
        return np.argmin(self.LOSS['val'])
        
    def evaluate(self, loader):
        """Evaluate the trained model.
        
        Args:
            loader: Data loader.
        Returns:
            Evaluation result.
        """
        eval_loss = 0.0
        output_dict = {'x': [], 'output': [], 'true_y': []}
        
        self.model.eval()
        with torch.no_grad():
            total = 0.0
            for x, y in loader:
                total += y.shape[0]
                outputs, _ = self.model(x)
                loss = self.criterion(outputs, y)
                eval_loss += loss.item()*y.shape[0]
                _, predicted = outputs.max(1)
                true = y.reshape(-1,1)
               
                output_dict['x'].append(x.detach().cpu().numpy().squeeze())
                output_dict['output'] = output_dict['output'] + [element.item() for element in predicted.flatten()]
                output_dict['true_y'] = output_dict['true_y'] + [element.item() for element in y.flatten()]
            
        return output_dict, float(eval_loss/total)
    

class NNClassifier_CL:
    """NNClassifier_CL.
    
    Attributes:
        model: Train model.
        criterion: Train criterion.
        optimizer: Train optimizer.
        optimizer_config: Train optimizer config.
    """
    def __init__(self, model, criterion, optimizer, optimizer_config):
        """Initialize NNClassifier."""
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), **optimizer_config)
        self.criterion = criterion
        self.LOSS = {'train': [], 'val': []}
        
    def fit(self, loader, epochs, sample_size, lamb, device, earlystop_path, seed):
        """Train the model with evaluation using validation set.
        
        Args:
            loader: Data loader.
            epochs: Train epochs.
            earlystop_path: Earlystop model checkpoint path.
        Returns:
            Index of minimum validation loss.
        """
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10)
        early_stopping = EarlyStopping(patience=10, delta=0.001, path=earlystop_path)
        
        np.random.seed(seed)
        
        for epoch in range(epochs):
            total = 0
            tloss = 0
            self.model.train()
            for x1, y1 in loader["train"]:
                total += y1.shape[0]
                self.optimizer.zero_grad()
                outputs1, _ = self.model(x1)
                loss1 = self.criterion(outputs1, y1)
                
                x_buffer = loader["buffer"][0]
                y_buffer = loader["buffer"][1]
                
                sample_ind = np.random.choice(len(x_buffer), sample_size, replace=False)
                x_buffer_sample = x_buffer[sample_ind]
                y_buffer_sample = y_buffer[sample_ind]
                
                x_buffer_sample = torch.Tensor(x_buffer_sample).to(device, dtype=torch.float32)
                y_buffer_sample = torch.Tensor(y_buffer_sample).to(device, dtype=torch.int64)
        
                buffer_ds = TensorDataset(x_buffer_sample, y_buffer_sample)
                buffer_loader = DataLoader(dataset=buffer_ds, batch_size=sample_size, shuffle=True)
                
                for x2, y2 in buffer_loader:
                    total += y2.shape[0]
                    outputs2, _ = self.model(x2)
                    loss2 = self.criterion(outputs2, y2)
                    
                loss = loss1 + lamb * loss2
                loss.backward()
                
                self.optimizer.step()
                
                tloss += loss.item()
            
            self.LOSS['train'].append(tloss/total)
            
            with torch.no_grad():
                val_correct = 0.0
                val_total = 0.0
                vloss = 0
                self.model.eval()
                for x_val, y_val in loader["val"]:
                    val_total += y_val.shape[0]
                    val_output, _ = self.model(x_val)
                    val_loss = self.criterion(val_output, y_val)
                    vloss += val_loss.item()*y_val.shape[0]

                self.LOSS['val'].append(vloss/val_total)
                
                _, val_pred = val_output.max(1)
                val_true = y_val.reshape(-1,1)
                val_correct += (val_pred == val_true).sum().item()
                        
            scheduler.step(self.LOSS['val'][-1])
            early_stopping(self.LOSS['val'][-1], self.model)
            
            if early_stopping.early_stop:
                break
        
        self.model.load_state_dict(torch.load(earlystop_path))
        
        return np.argmin(self.LOSS['val'])
        
    def evaluate(self, loader):
        """Evaluate the trained model.
        
        Args:
            loader: Data loader.
        Returns:
            Evaluation result.
        """
        eval_loss = 0.0
        output_dict = {'x': [], 'output': [], 'true_y': []}
        
        self.model.eval()
        with torch.no_grad():
            total = 0.0
            for x, y in loader:
                total += y.shape[0]
                outputs, _ = self.model(x)
                loss = self.criterion(outputs, y)
                eval_loss += loss.item()*y.shape[0]
                _, predicted = outputs.max(1)
                true = y.reshape(-1,1)
               
                output_dict['x'].append(x.detach().cpu().numpy().squeeze())
                output_dict['output'] = output_dict['output'] + [element.item() for element in predicted.flatten()]
                output_dict['true_y'] = output_dict['true_y'] + [element.item() for element in y.flatten()]
            
        return output_dict, float(eval_loss/total)
    
    
class NNClassifier_AGEM:
    """NNClassifier_AGEM.
    
    Attributes:
        model: Train model.
        criterion: Train criterion.
        optimizer: Train optimizer.
        optimizer_config: Train optimizer config.
    """
    def __init__(self, model, criterion, optimizer, optimizer_config):
        """Initialize NNClassifier."""
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), **optimizer_config)
        self.criterion = criterion
        self.LOSS = {'train': [], 'val': []}
        
    def fit(self, loader, epochs, earlystop_path, x_prev_buffer, y_prev_buffer, buffer_size, i, device, n_class):
        """Train the model with evaluation using validation set.
        
        Args:
            loader: Data loader.
            epochs: Train epochs.
            earlystop_path: Earlystop model checkpoint path.
        Returns:
            Index of minimum validation loss.
        """
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10)
        early_stopping = EarlyStopping(patience=10, delta=0.001, path=earlystop_path)
        
        for epoch in range(epochs):
            total = 0.0
            tloss = 0
            
            # buffer gradient computation
            loss_sample = nn.CrossEntropyLoss(reduction='none')

#             for n in range(i*2):
                
#                 x_buffer = x_prev_buffer[n*buffer_size:(n+1)*buffer_size]
#                 y_buffer = y_prev_buffer[n*buffer_size:(n+1)*buffer_size]
#                 x_buffer = torch.Tensor(x_buffer).to(device, dtype=torch.float32)
#                 y_buffer = torch.Tensor(y_buffer).to(device, dtype=torch.int64)

#                 if n == 0:    
#                     out, emb = self.model(x_buffer)
#                     loss = loss_sample(out, y_buffer).sum()
# #                     buffer_l0_grads = torch.autograd.grad(loss, out)[0]
# #                     buffer_l0_expand = torch.repeat_interleave(buffer_l0_grads, 256, dim=1)
# #                     buffer_l1_grads = buffer_l0_expand * emb.repeat(1, n_class)
                    
# #                     buffer_l0_grads = buffer_l0_grads.mean(dim=0).view(1, -1)
# #                     buffer_l1_grads = buffer_l1_grads.mean(dim=0).view(1, -1)

#                     self.optimizer.zero_grad()
                    
#                     loss.backward()
#                     buffer_grads = []
#                     for name, param in self.model.named_parameters():
#                         buffer_grads.append(param.grad.view(-1))
#                     buffer_grads = torch.cat(buffer_grads).detach().clone().view(1,-1)
                    

#                 else:   
#                     out, emb = self.model(x_buffer)
#                     loss = loss_sample(out, y_buffer).sum()
# #                     batch_l0_grads = torch.autograd.grad(loss, out)[0]
# #                     batch_l0_expand = torch.repeat_interleave(batch_l0_grads, 256, dim=1)
# #                     batch_l1_grads = batch_l0_expand * emb.repeat(1, n_class)
                    
# #                     batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
# #                     batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)

#                     self.optimizer.zero_grad()

#                     loss.backward()
#                     batch_grads = []
#                     for name, param in self.model.named_parameters():
#                         batch_grads.append(param.grad.view(-1))
#                     batch_grads = torch.cat(batch_grads).detach().clone().view(1,-1)

# #                     buffer_l0_grads = torch.cat((buffer_l0_grads, batch_l0_grads), dim=0)
# #                     buffer_l1_grads = torch.cat((buffer_l1_grads, batch_l1_grads), dim=0)

#                     buffer_grads = torch.cat((buffer_grads, batch_grads), dim=0)

# #             buffer_grads = torch.cat((buffer_l0_grads, buffer_l1_grads), dim=1)

# #             print('buffer_grads:', buffer_grads)
# #             print('buffer_grads_size:', buffer_grads.size())
            
#             avg_buffer_grads = torch.mean(buffer_grads, dim=0).view(-1)

            for n in range(i*2):
                if n == 0:
                    x_buffer = x_prev_buffer[n*buffer_size:(n+1)*buffer_size]
                    y_buffer = y_prev_buffer[n*buffer_size:(n+1)*buffer_size]
                    x_buffer = torch.Tensor(x_buffer).to(device, dtype=torch.float32)
                    y_buffer = torch.Tensor(y_buffer).to(device, dtype=torch.int64)
                    
                else:
                    next_x_buffer = x_prev_buffer[n*buffer_size:(n+1)*buffer_size]
                    next_y_buffer = y_prev_buffer[n*buffer_size:(n+1)*buffer_size]
                    next_x_buffer = torch.Tensor(next_x_buffer).to(device, dtype=torch.float32)
                    next_y_buffer = torch.Tensor(next_y_buffer).to(device, dtype=torch.int64)
                    
                    x_buffer = torch.cat((x_buffer, next_x_buffer), dim=0)
                    y_buffer = torch.cat((y_buffer, next_y_buffer), dim=0)
                    
#             print(x_buffer, x_buffer.size())
#             print(y_buffer, y_buffer.size())

#             self.optimizer.zero_grad()
                    
#             out, emb = self.model(x_buffer)
#             loss = self.criterion(out, y_buffer)
#             loss.backward()
#             buffer_grads = []
#             for name, param in self.model.named_parameters():
#                 buffer_grads.append(param.grad.view(-1))
#             buffer_grads = torch.cat(buffer_grads).detach().clone().view(1,-1)
            
#             avg_buffer_grads = buffer_grads.view(-1)
            
#             print('buffer:', avg_buffer_grads)
                
            
#             print('buffer_grads:', avg_buffer_grads)
#             print('buffer_grads_size:', avg_buffer_grads.size())
            
            self.model.train()
        
            num_neg = 0
            num_iter = 0
            for x, y in loader["train"]:
                num_iter +=1
                total += y.shape[0]
#                 self.optimizer.zero_grad()
                outputs, _ = self.model(x)
#                 loss = self.criterion(outputs, y)*y.shape[0]
                self.optimizer.zero_grad()
                loss = self.criterion(outputs, y)
        
#                 self.optimizer.zero_grad()
                loss.backward()
                
#                 new_l0_grads = self.model.linear2.bias.grad.view(1, -1)
#                 new_l1_grads = self.model.linear2.weight.grad.view(1, -1)
                
#                 new_grads = torch.cat((new_l0_grads, new_l1_grads), dim=1).view(-1)

                new_grads = []
                for name, param in self.model.named_parameters():
                    new_grads.append(param.grad.view(-1))
                new_grads = torch.cat(new_grads).detach().clone().view(-1)
                
#                 self.optimizer.zero_grad()
                    
                out, emb = self.model(x_buffer)
                loss = self.criterion(out, y_buffer)
                loss.backward()
                buffer_grads = []
                for name, param in self.model.named_parameters():
                    buffer_grads.append(param.grad.view(-1))
                buffer_grads = torch.cat(buffer_grads).detach().clone().view(1,-1)

                avg_buffer_grads = buffer_grads.view(-1)
                
#                 print('new:', new_grads)
                
        
#                 grad_dims = []
#                 for param in self.model.parameters():
#                     grad_dims.append(param.data.numel())
                
#                 grads = torch.Tensor(sum(grad_dims), 1)
                
#                 grads[:, 0].fill_(0.0)
#                 cnt = 0
#                 for param in self.model.parameters():
#                     if param.grad is not None:
#                         beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
#                         en = sum(grad_dims[:cnt + 1])
#                         grads[beg: en, 0].copy_(param.grad.data.view(-1))
#                     cnt += 1

                if torch.dot(new_grads, avg_buffer_grads) < 0:
#                     print('neg')
                    num_neg += 1
                    correct_grads = new_grads - (torch.dot(new_grads, avg_buffer_grads)/torch.dot(avg_buffer_grads, avg_buffer_grads)) * avg_buffer_grads
        
#                     print('correct:', correct_grads)
                
#                     self.optimizer.zero_grad()
                    
#                     print('correct_grads:', correct_grads)
#                     print('correct_grads_size:', correct_grads.size())

#                     print(correct_grads[:n_class].size())
#                     print(correct_grads[n_class:].size())

                    state_dict = self.model.state_dict(keep_vars=True)
                    index = 0
                    for param in state_dict.keys():
                        # ignore batchnorm params
                        if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param:
                            continue
                        param_count = state_dict[param].numel()
                        param_shape = state_dict[param].shape
                        state_dict[param].grad = correct_grads[index:index+param_count].view(param_shape).clone()
                        index += param_count
                    self.model.load_state_dict(state_dict)
                    
                else:
                    state_dict = self.model.state_dict(keep_vars=True)
                    index = 0
                    for param in state_dict.keys():
                        # ignore batchnorm params
                        if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param:
                            continue
                        param_count = state_dict[param].numel()
                        param_shape = state_dict[param].shape
                        state_dict[param].grad = avg_buffer_grads[index:index+param_count].view(param_shape).clone()
#                         state_dict[param].grad = new_grads[index:index+param_count].view(param_shape).clone()
                        index += param_count
                    self.model.load_state_dict(state_dict)
                    
                    
#                     self.model.linear2.bias.grad = correct_grads[:n_class].detach()
#                     self.model.linear2.weight.grad = correct_grads[n_class:].reshape(n_class,256).detach()
                    
#                     print(self.model.linear2.bias.grad.size())
#                     print(self.model.linear2.weight.grad.size())
                
#                 print(self.model.linear1.weight.grad)
                
#                 self.model.linear1.weight.grad = self.model.linear1.weight.grad + 2
        
#                     for name, param in self.model.named_parameters():
#                         print(name, param.grad)
                
#                 print(self.model.linear1.weight.grad)
                
#                 print('l1 weight grad:', self.model.linear1.weight.grad.size())
#                 print('l1 bias grad:', self.model.linear1.bias.grad.size())
                
#                 print('l2 weight grad:', self.model.linear2.weight.grad.size())
#                 print('l2 bias grad:', self.model.linear2.bias.grad.size())
                
                self.optimizer.step()
                
                tloss += loss.item()
                
            print('num neg:', num_neg)
            print('num iter:', num_iter)
            
            self.LOSS['train'].append(tloss/total)
            
            with torch.no_grad():
                val_correct = 0.0
                val_total = 0.0
                vloss = 0
                self.model.eval()
                for x_val, y_val in loader["val"]:
                    val_total += y_val.shape[0]
                    val_output, _ = self.model(x_val)
                    val_loss = self.criterion(val_output, y_val)
                    vloss += val_loss.item()*y_val.shape[0]

                self.LOSS['val'].append(vloss/val_total)
                
                _, val_pred = val_output.max(1)
                val_true = y_val.reshape(-1,1)
                val_correct += (val_pred == val_true).sum().item()
                        
#             scheduler.step(self.LOSS['val'][-1])
            early_stopping(self.LOSS['val'][-1], self.model)
            
            if early_stopping.early_stop:
                break
        
        self.model.load_state_dict(torch.load(earlystop_path))
        
        return np.argmin(self.LOSS['val'])
        
    def evaluate(self, loader):
        """Evaluate the trained model.
        
        Args:
            loader: Data loader.
        Returns:
            Evaluation result.
        """
        eval_loss = 0.0
        output_dict = {'x': [], 'output': [], 'true_y': []}
        
        self.model.eval()
        with torch.no_grad():
            total = 0.0
            for x, y in loader:
                total += y.shape[0]
                outputs, _ = self.model(x)
                loss = self.criterion(outputs, y)
                eval_loss += loss.item()*y.shape[0]
                _, predicted = outputs.max(1)
                true = y.reshape(-1,1)
               
                output_dict['x'].append(x.detach().cpu().numpy().squeeze())
                output_dict['output'] = output_dict['output'] + [element.item() for element in predicted.flatten()]
                output_dict['true_y'] = output_dict['true_y'] + [element.item() for element in y.flatten()]
            
        return output_dict, float(eval_loss/total)
    

# class NNRegressor:
#     """NNRegressor.
    
#     Attributes:
#         model: Train model.
#         criterion: Train criterion.
#         optimizer: Train optimizer.
#         optimizer_config: Train optimizer config.
#     """
#     def __init__(self, model, criterion, optimizer, optimizer_config):
#         """Initialize NNRegressor."""
#         self.model = model
#         self.optimizer = optimizer(self.model.parameters(), **optimizer_config)
#         self.criterion = criterion
#         self.LOSS = {'train': [], 'val': []}
        
#     def fit(self, loader, epochs, earlystop_path):
#         """Train the model with evaluation using validation set.
        
#         Args:
#             loader: Data loader.
#             epochs: Train epochs.
#             earlystop_path: Earlystop model checkpoint path.
#         Returns:
#             Index of minimum validation loss.
#         """
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10)
#         early_stopping = EarlyStopping(patience=10, delta=0.0001, path=earlystop_path)
        
#         for epoch in range(epochs):
#             total = 0.0
#             tloss = 0
#             self.model.train()
#             for x, y in loader["train"]:
#                 total += y.shape[0]
#                 self.optimizer.zero_grad()
#                 outputs, _ = self.model(x)
# #                 print(outputs.size())
# #                 print(y.unsqueeze(dim=1).size())
#                 loss = self.criterion(outputs, y.unsqueeze(dim=1))*y.shape[0]
#                 loss.backward()
#                 self.optimizer.step()
                
#                 tloss += loss.item()
            
#             self.LOSS['train'].append(tloss/total)
            
#             with torch.no_grad():
#                 val_correct = 0.0
#                 val_total = 0.0
#                 vloss = 0
#                 self.model.eval()
#                 for x_val, y_val in loader["val"]:
#                     val_total += y_val.shape[0]
#                     val_output, _ = self.model(x_val)
#                     val_loss = self.criterion(val_output, y_val.unsqueeze(dim=1))
#                     vloss += val_loss.item()*y_val.shape[0]

#                 self.LOSS['val'].append(vloss/val_total)
                        
#             scheduler.step(self.LOSS['val'][-1])
#             early_stopping(self.LOSS['val'][-1], self.model)
            
#             if early_stopping.early_stop:
#                 break
        
#         self.model.load_state_dict(torch.load(earlystop_path))
        
#         return np.argmin(self.LOSS['val'])
        
#     def evaluate(self, loader):
#         """Evaluate the trained model.
        
#         Args:
#             loader: Data loader.
#         Returns:
#             Evaluation result.
#         """
#         eval_loss = 0.0
#         output_dict = {'x': [], 'output': [], 'true_y': []}
        
#         self.model.eval()
#         with torch.no_grad():
#             total = 0.0
#             for x, y in loader:
#                 total += y.shape[0]
#                 outputs, _ = self.model(x)
#                 loss = self.criterion(outputs, y.unsqueeze(dim=1))
#                 eval_loss += loss.item()*y.shape[0]
#                 predicted = outputs
#                 true = y.reshape(-1,1)
               
#                 output_dict['x'].append(x.detach().cpu().numpy().squeeze())
#                 output_dict['output'] = output_dict['output'] + [element.item() for element in predicted.flatten()]
#                 output_dict['true_y'] = output_dict['true_y'] + [element.item() for element in y.flatten()]
            
#         return output_dict, float(eval_loss/total)