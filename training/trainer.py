"""
Training Module
===============

Training utilities and trainer class for VTM-enhanced models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Optional

from configs import TrainingConfig, DataConfig
from training.metrics import calculate_psnr, calculate_ssim
from training.losses import MultiComponentLoss


class VTMTrainer:
    """Trainer for VTM-enhanced video enhancement models"""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        data_config: DataConfig,
        device: torch.device,
        save_dir: str = "results"
    ):
        self.model = model
        self.config = config
        self.data_config = data_config
        self.device = device
        self.save_dir = Path(save_dir)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup loss function
        self.criterion = MultiComponentLoss(config.loss_weights)
        
        # Create dataloaders
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        
    def _create_optimizer(self):
        """Create optimizer based on configuration"""
        if self.config.optimizer.value == "adam":
            return torch.optim.Adam(
                self.model.parameters(), 
                lr=self.config.learning_rate,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.value == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9
            )
        elif self.config.optimizer.value == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_dataloaders(self):
        """Create training and validation dataloaders"""
        from data.dataset import VTMDataset, preprocess_vtm_data
        
        # Preprocess data if needed
        data_file = Path(self.data_config.processed_data_dir) / "preprocessed_vtm_data.pkl"
        if not data_file.exists():
            print("Preprocessing data...")
            preprocess_vtm_data(self.data_config)
        
        # Load dataset
        dataset = VTMDataset(str(data_file))
        
        # Split into train/val (90/10 split)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            decoded = batch["decoded"].to(self.device)
            features = batch["features"].to(self.device)
            target = batch["target"].to(self.device)
            
            # Forward pass
            enhanced = self.model(decoded, features)
            loss = self.criterion(enhanced, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # Log progress
            if batch_idx % self.config.log_frequency == 0:
                print(f"  Batch {batch_idx:3d}: Loss = {loss.item():.6f}")
        
        return epoch_loss / len(self.train_loader)
    
    def validate_epoch(self) -> dict:
        """Validate for one epoch"""
        self.model.eval()
        val_loss = 0
        psnr_total = 0
        ssim_total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                decoded = batch["decoded"].to(self.device)
                features = batch["features"].to(self.device)
                target = batch["target"].to(self.device)
                
                enhanced = self.model(decoded, features)
                loss = self.criterion(enhanced, target)
                
                val_loss += loss.item()
                
                # Calculate metrics
                psnr_total += calculate_psnr(enhanced, target).item()
                ssim_total += calculate_ssim(enhanced, target).item()
        
        return {
            "loss": val_loss / len(self.val_loader),
            "psnr": psnr_total / len(self.val_loader),
            "ssim": ssim_total / len(self.val_loader)
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config.dict()
        }
        
        # Save latest checkpoint
        checkpoint_path = self.save_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  New best model saved with loss {self.best_loss:.6f}")
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config.epochs} epochs...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            val_loss = val_metrics["loss"]
            
            # Check if this is the best model
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            # Save checkpoint
            if self.config.save_best and is_best:
                self.save_checkpoint(is_best=True)
            elif epoch % self.config.eval_frequency == 0:
                self.save_checkpoint(is_best=False)
            
            # Print epoch results
            print(f"Epoch {epoch:3d}:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Val PSNR:   {val_metrics['psnr']:.3f} dB")
            print(f"  Val SSIM:   {val_metrics['ssim']:.4f}")
            print(f"  Best Loss:  {self.best_loss:.6f}")
            
            # Early stopping
            if self.config.early_stopping and val_loss < self.config.early_stopping:
                print(f"Early stopping at epoch {epoch} (loss < {self.config.early_stopping})")
                break
        
        # Training completed
        end_time = time.time()
        training_time = end_time - start_time
        
        print("\nTraining completed!")
        print(f"Total time: {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
        print(f"Best validation loss: {self.best_loss:.6f}")
        
        # Save final checkpoint
        self.save_checkpoint(is_best=(val_loss <= self.best_loss))