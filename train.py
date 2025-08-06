import os
import torch
import date
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from vis_model import HisToGene
from utils import *
from predict import model_predict, sr_predict
from dataset import ViT_HEST1K

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
        filename=f"best_{tag}" + "-{epoch:02d}-{valid_loss:.4f}",
        monitor='valid_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor='valid_loss',
        patience=10,
        mode='min',
        verbose=True
    )
    
    # Create trainer with validation and callbacks
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        check_val_every_n_epoch=1,
        log_every_n_steps=10
    )
    
    # Train with validation
    trainer.fit(model, train_loader, val_loader)
    
    # Save the final model as well
    trainer.save_checkpoint(f"model/last_train_{tag}.ckpt")
    
    # Print best model path
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score}")
    
    return trainer, checkpoint_callback.best_model_path

if __name__ == "__main__":
    trainer, best_model_path = train(genes='3CA-copy')