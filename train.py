import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from vis_model import HisToGene
from utils import *
from predict import model_predict, sr_predict
from dataset import ViT_HEST1K

def train(n_genes = 2977,
          lr = 1e-5,
          max_epochs = 100,
          genes = '3CA',
          tag = genes + '-HEST'):
    trainset = ViT_HEST1K(gene_list=genes)
    testset = ViT_HEST1K(gene_list=genes, mode='test')
    train_loader = DataLoader(trainset, batch_size=1, num_workers=4, shuffle=True)
    model = HisToGene(n_layers=8, n_genes=n_genes, learning_rate=lr)
    trainer = pl.Trainer('gpu', max_epochs=max_epochs)
    trainer.fit(model, train_loader)
    trainer.save_checkpoint("model/last_train_"+tag+".ckpt")
    
if __name__ == "__main__":
    train()