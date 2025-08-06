import os
import torch
import date
import time
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from vis_model import HisToGene
from utils import *
from predict import model_predict, sr_predict
from dataset import ViT_HEST1K

class EpochTimeCallback(pl.Callback):
    """Callback to track epoch training time"""
    
    def __init__(self):
        self.epoch_start_time = None
        self.epoch_times = []
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
    
    def on_train_epoch_end(self, trainer, pl_module):
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            
            # Log epoch time
            pl_module.log('epoch_time', epoch_time, on_epoch=True, prog_bar=True)
            
            # Print timing info
            avg_time = sum(self.epoch_times) / len(self.epoch_times)
            print(f"Epoch {trainer.current_epoch} completed in {epoch_time:.2f}s "
                  f"(avg: {avg_time:.2f}s)")

def train(n_genes = 2977,
          lr = 1e-5,
          max_epochs = 100,
          genes = '3CA',
          num_workers = 1):
    today = date.today().strftime("%Y%m%d")
    tag = genes + '-HEST-' + today
    
    # Create datasets
    trainset = ViT_HEST1K(gene_list=genes, mode='train')
    valset = ViT_HEST1K(gene_list=genes, mode='val')
    testset = ViT_HEST1K(gene_list=genes, mode='test')
    
    # Create data loaders
    train_loader = DataLoader(trainset, batch_size=1, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(valset, batch_size=1, num_workers=num_workers, shuffle=False)
    
    # Create model
    model = HisToGene(n_layers=8, n_genes=n_genes, learning_rate=lr)
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="model/",
        filename=f"best_{tag}" + "-{epoch:02d}-{val_loss:.4f}",
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True
    )
    
    # Add epoch timing callback
    epoch_timer = EpochTimeCallback()
    
    # Create trainer with validation and callbacks
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping, epoch_timer],
        check_val_every_n_epoch=1,
        log_every_n_steps=10
    )
    
    # Track total training time
    training_start_time = time.time()
    
    # Train with validation
    trainer.fit(model, train_loader, val_loader)
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    
    # Save the final model as well
    trainer.save_checkpoint(f"model/last_train_{tag}.ckpt")
    
    # Print timing summary
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")
    if epoch_timer.epoch_times:
        avg_epoch_time = sum(epoch_timer.epoch_times) / len(epoch_timer.epoch_times)
        print(f"Average time per epoch: {avg_epoch_time:.2f}s")
        print(f"Fastest epoch: {min(epoch_timer.epoch_times):.2f}s")
        print(f"Slowest epoch: {max(epoch_timer.epoch_times):.2f}s")
    
    # Print best model path
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score}")
    
    return trainer, checkpoint_callback.best_model_path

if __name__ == "__main__":
    trainer, best_model_path = train(genes='3CA-copy')