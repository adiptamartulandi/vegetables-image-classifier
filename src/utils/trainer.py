import os
import torch
import torch.nn as nn

from typing import Dict
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: str = 'cpu',
        checkpoint_dir: str = 'checkpoints'
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': total_loss/len(train_loader), 'acc': 100.*correct/total})
        
        return {
            'loss': total_loss/len(train_loader),
            'accuracy': 100.*correct/total
        }
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(val_loader, desc='Evaluating')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return {
            'loss': total_loss/len(val_loader),
            'accuracy': 100.*correct/total
        }
    
    def save_checkpoint(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        is_best: bool = False
    ) -> None:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        # Save latest checkpoint
        torch.save(
            checkpoint,
            os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        )
        
        # Save best checkpoint
        if is_best:
            torch.save(
                checkpoint,
                os.path.join(self.checkpoint_dir, 'best_checkpoint.pth')
            )
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint
