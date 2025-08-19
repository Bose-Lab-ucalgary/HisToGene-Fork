import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils import *
import anndata as ann
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

MODEL_PATH = ''

# device = 'cpu'
def model_predict(model, test_loader, adata=None, attention=True, device = torch.device('cpu')): 
    model.eval()
    model = model.to(device)
    preds = None
    with torch.no_grad():
        for patch, position, exp, center in tqdm(test_loader):

            patch, position = patch.to(device), position.to(device)
            
            pred = model(patch, position)


            if preds is None:
                preds = pred.squeeze()
                ct = center
                gt = exp
            else:
                preds = torch.cat((preds,pred),dim=0)
                ct = torch.cat((ct,center),dim=0)
                gt = torch.cat((gt,exp),dim=0)
                
    preds = preds.cpu().squeeze().numpy()
    ct = ct.cpu().squeeze().numpy()
    gt = gt.cpu().squeeze().numpy()
    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct

    adata_gt = ann.AnnData(gt)
    adata_gt.obsm['spatial'] = ct

    return adata, adata_gt

def sr_predict(model, test_loader, attention=True,device = torch.device('cpu')):
    model.eval()
    model = model.to(device)
    preds = None
    with torch.no_grad():
        for patch, position, center in tqdm(test_loader):
            
            patch, position = patch.to(device), position.to(device)
            pred = model(patch, position)
            
            if preds is None:
                preds = pred.squeeze()
                ct = center
            else:
                preds = torch.cat((preds,pred),dim=0)
                ct = torch.cat((ct,center),dim=0)
    preds = preds.cpu().squeeze().numpy()
    ct = ct.cpu().squeeze().numpy()
    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct


    return adata

def model_predict_lightning(model, test_loader, device=torch.device('cpu')):
    """PyTorch Lightning-based prediction function"""
    
    # Create a Lightning trainer for prediction
    trainer = pl.Trainer(
        accelerator='gpu' if device.type == 'cuda' else 'cpu',
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True
    )
    
    # Run predictions
    results = trainer.predict(model, test_loader)
    
    # Concatenate all results
    all_preds = []
    all_gt = []
    all_centers = []
    
    for batch_result in results:
        all_preds.append(batch_result['predictions'])
        if batch_result['ground_truth'] is not None:
            all_gt.append(batch_result['ground_truth'])
        if batch_result['centers'] is not None:
            all_centers.append(batch_result['centers'])
    
    # Concatenate tensors
    preds = torch.cat(all_preds, dim=0).squeeze().numpy()
    
    if all_centers:
        centers = torch.cat(all_centers, dim=0).squeeze().numpy()
    else:
        centers = None
        
    if all_gt:
        gt = torch.cat(all_gt, dim=0).squeeze().numpy()
    else:
        gt = None
    
    # Create AnnData objects
    adata_pred = ann.AnnData(preds)
    if centers is not None:
        adata_pred.obsm['spatial'] = centers
    
    adata_gt = None
    if gt is not None:
        adata_gt = ann.AnnData(gt)
        if centers is not None:
            adata_gt.obsm['spatial'] = centers
    
    return adata_pred, adata_gt

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # for fold in [5,11,17,26]:
    for fold in range(12):
        # fold=30
        # tag = '-vit_skin_aug'
        # tag = '-cnn_her2st_785_32_cv'
        tag = '-vit_her2st_785_32_cv'
        # tag = '-cnn_skin_134_cv'
        # tag = '-vit_skin_134_cv'
        ds = 'HER2'
        # ds = 'Skin'

        print('Loading model ...')
        # model = STModel.load_from_checkpoint('model/last_train_'+tag+'.ckpt')
        # model = VitModel.load_from_checkpoint('model/last_train_'+tag+'.ckpt')
        # model = STModel.load_from_checkpoint("model/last_train_"+tag+'_'+str(fold)+".ckpt") 
        model = SpatialTransformer.load_from_checkpoint("model/last_train_"+tag+'_'+str(fold)+".ckpt")
        model = model.to(device)
        # model = torch.nn.DataParallel(model)
        print('Loading data ...')

        # g = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
        g = list(np.load('data/skin_hvg_cut_1000.npy',allow_pickle=True))

        # dataset = SKIN(train=False,ds=ds,fold=fold)
        dataset = ViT_HER2ST(train=False,mt=False,sr=True,fold=fold)
        # dataset = ViT_SKIN(train=False,mt=False,sr=False,fold=fold)
        # dataset = VitDataset(diameter=112,sr=True)

        test_loader = DataLoader(dataset, batch_size=16, num_workers=4)
        print('Making prediction ...')

        adata_pred, adata = model_predict(model, test_loader, attention=False)
        # adata_pred = sr_predict(model,test_loader,attention=True)

        adata_pred.var_names = g
        print('Saving files ...')
        adata_pred = comp_tsne_km(adata_pred,4)
        # adata_pred = comp_umap(adata_pred)
        print(fold)
        print(adata_pred)

        adata_pred.write('processed/test_pred_'+ds+'_'+str(fold)+tag+'.h5ad')
        # adata_pred.write('processed/test_pred_sr_'+ds+'_'+str(fold)+tag+'.h5ad')

        # quit()

if __name__ == '__main__':
    main()

