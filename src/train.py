import warnings
import torch.nn as nn

from torch.optim import Adam
from data.dataset import create_dataloaders
from models.cnn import create_model
from utils.trainer import Trainer

warnings.filterwarnings('ignore')


def main():
    # Configuration
    data_dir = 'data'
    device = 'cpu'
    num_epochs = 30
    batch_size = 32
    learning_rate = 0.001
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        train_split=0.8,
        num_workers=0  # Set to 0 for CPU training
    )
    
    # Create model, loss function, and optimizer
    model = create_model(num_classes=2, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )
    
    # Training loop
    best_val_acc = 0
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        print(f'Train Loss: {train_metrics["loss"]:.4f}, Train Acc: {train_metrics["accuracy"]:.2f}%')
        
        # Validate
        val_metrics = trainer.evaluate(val_loader)
        print(f'Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["accuracy"]:.2f}%')
        
        # Save checkpoint
        is_best = val_metrics["accuracy"] > best_val_acc
        if is_best:
            print(f'Saving best model in epoch {epoch+1} with:')
            print(f'train accuracy {train_metrics["accuracy"]:.2f}%')
            print(f'val accuracy {val_metrics["accuracy"]:.2f}%')
            best_val_acc = val_metrics["accuracy"]
        
        trainer.save_checkpoint(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            is_best=is_best
        )

if __name__ == '__main__':
    main()