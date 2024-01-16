import numpy as np
import torch

class EarlyStopping:
    """EarlyStopping.
    
    Attributes:
        patience: Number of possible epochs with no improvement.
        delta: Minimum change to qualify as an improvement.
        path: Path to save model checkpoint.
    """
    def __init__(self, patience=10, delta=0, path=''):
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