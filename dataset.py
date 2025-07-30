import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import read_tiff
import numpy as np
import torchvision
import torchvision.transforms as transforms
import scanpy as sc
from utils import get_data
import os
import glob
from PIL import Image
import pandas as pd 
import scprep as scp
from PIL import ImageFile
import seaborn as sns
import matplotlib.pyplot as plt
import h5py
import anndata as ad
from pathlib import Path
from collections import defaultdict as dfd
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

BCELL = ['CD19', 'CD79A', 'CD79B', 'MS4A1']
TUMOR = ['FASN']
CD4T = ['CD4']
CD8T = ['CD8A', 'CD8B']
DC = ['CLIC2', 'CLEC10A', 'CD1B', 'CD1A', 'CD1E']
MDC = ['LAMP3']
CMM = ['BRAF', 'KRAS']
IG = {'B_cell':BCELL, 'Tumor':TUMOR, 'CD4+T_cell':CD4T, 'CD8+T_cell':CD8T, 'Dendritic_cells':DC, 
        'Mature_dendritic_cells':MDC, 'Cutaneous_Malignant_Melanoma':CMM}
MARKERS = []
for i in IG.values():
    MARKERS+=i
LYM = {'B_cell':BCELL, 'CD4+T_cell':CD4T, 'CD8+T_cell':CD8T}

class STDataset(torch.utils.data.Dataset):
    """Some Information about STDataset"""
    def __init__(self, adata, img_path, diameter=177.5, train=True):
        super(STDataset, self).__init__()

        self.exp = adata.X.toarray()
        self.im = read_tiff(img_path)
        self.r = np.ceil(diameter/2).astype(int)
        self.train = train
        # self.d_spot = self.d_spot if self.d_spot%2==0 else self.d_spot+1
        self.transforms = transforms.Comloce([
            transforms.ColorJitter(0.5,0.5,0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])
        self.centers = adata.obsm['spatial']
        self.loc = adata.obsm['locition_norm']
    def __getitem__(self, index):
        exp = self.exp[index]
        center = self.centers[index]
        x, y = center
        patch = self.im.crop((x-self.r, y-self.r, x+self.r, y+self.r))
        exp = torch.Tensor(exp)
        mask = exp!=0
        mask = mask.float()
        if self.train:
            patch = self.transforms(patch)
        loc = torch.Tensor(self.loc[index])
        return patch, loc, exp, mask

    def __len__(self):
        return len(self.centers)



class HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""
    def __init__(self,train=True,gene_list=None,ds=None,fold=0):
        super(HER2ST, self).__init__()
        self.cnt_dir = 'data/her2st/data/ST-cnts'
        self.img_dir = 'data/her2st/data/ST-imgs'
        self.loc_dir = 'data/her2st/data/ST-spotfiles'
        self.lbl_dir = 'data/her2st/data/ST-pat/lbl'
        self.r = 224//2
        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]
        self.train = train
  
        # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        samples = names[1:33]
        te_names = [samples[fold]]
        tr_names = list(set(samples)-set(te_names))
        if train:
            # names = names[1:33]
            # names = names[1:33] if self.cls==False else ['A1','B1','C1','D1','E1','F1','G2']
            names = tr_names
        else:
            # names = [names[33]]
            # names = ['A1']
            # names = [ds] if ds else ['H1']
            names = te_names
        print('Loading imgs...')
        self.img_dict = {i:self.get_img(i) for i in names}
        # print('Loading metadata...')
        # self.meta_dict = {i:self.get_meta(i) for i in names}

        # self.gene_set = self.get_overlap(self.meta_dict,gene_list)
        # print(len(self.gene_set))
        # np.save('data/her_hvg',self.gene_set)
        # quit()
        # self.gene_set = list(gene_list)
        # self.exp_dict = {i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i,m in self.meta_dict.items()}
        # self.center_dict = {i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) for i,m in self.meta_dict.items()}
        # self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}


        # self.lengths = [len(i) for i in self.meta_dict.values()]
        # self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.transforms = transforms.Comloce([
            transforms.ColorJitter(0.5,0.5,0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])
    def __getitem__(self, index):
        i = 0
        while index>=self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i-1]
        
        exp = self.exp_dict[self.id2name[i]][idx]
        center = self.center_dict[self.id2name[i]][idx]
        loc = self.loc_dict[self.id2name[i]][idx]

        # if self.cls or self.train==False:

        exp = torch.Tensor(exp)
        loc = torch.Tensor(loc)

        x, y = center
        patch = self.img_dict[self.id2name[i]].crop((x-self.r, y-self.r, x+self.r, y+self.r))
        if self.train:
            patch = self.transforms(patch)
        else:
            patch = transforms.ToTensor()(patch)

        if self.train:
            return patch, loc, exp
        else: 
            return patch, loc, exp, torch.Tensor(center)

    def __len__(self):
        return self.cumlen[-1]

    def get_img(self,name):
        pre = self.img_dir+'/'+name[0]+'/'+name
        fig_name = os.listdir(pre)[0]
        path = pre+'/'+fig_name
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = self.cnt_dir+'/'+name+'.tsv'
        df = pd.read_csv(path,sep='\t',index_col=0)
        return df

    def get_loc(self,name):
        path = self.loc_dir+'/'+name+'_selection.tsv'
        # path = self.loc_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_lbl(self,name):
        # path = self.loc_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        loc = self.get_loc(name)
        meta = cnt.join((loc.set_index('id')))
        self.max_x = 0
        self.max_y = 0
        loc = meta[['x','y']].values
        self.max_x = max(self.max_x, loc[:,0].max())
        self.max_y = max(self.max_y, loc[:,1].max())
        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)

class ViT_HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""
    def __init__(self,train=True,gene_list=None,ds=None,sr=False,fold=0):
        super(ViT_HER2ST, self).__init__()

        self.cnt_dir = 'data/her2st/data/ST-cnts'
        self.img_dir = 'data/her2st/data/ST-imgs'
        self.loc_dir = 'data/her2st/data/ST-spotfiles'
        self.lbl_dir = 'data/her2st/data/ST-pat/lbl'
        self.r = 224//4

        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]
        self.train = train
        self.sr = sr
        
        # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        samples = names[1:33]

        te_names = [samples[fold]]
        tr_names = list(set(samples)-set(te_names))

        if train:
            # names = names[1:33]
            # names = names[1:33] if self.cls==False else ['A1','B1','C1','D1','E1','F1','G2']
            names = tr_names
        else:
            # names = [names[33]]
            # names = ['A1']
            # names = [ds] if ds else ['H1']
            names = te_names

        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in names}


        self.gene_set = list(gene_list)
        self.exp_dict = {i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i,m in self.meta_dict.items()}
        self.center_dict = {i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) for i,m in self.meta_dict.items()}
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}


        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.transforms = transforms.Comloce([
            transforms.ColorJitter(0.5,0.5,0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i,exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp>0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:,j])


    def __getitem__(self, index):
        i = index
        im = self.img_dict[self.id2name[i]]
        im = im.permute(1,0,2)
        # im = torch.Tensor(np.array(self.im))
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        loc = self.loc_dict[self.id2name[i]]
        locitions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4

        if self.sr:
            centers = torch.LongTensor(centers)
            max_x = centers[:,0].max().item()
            max_y = centers[:,1].max().item()
            min_x = centers[:,0].min().item()
            min_y = centers[:,1].min().item()
            r_x = (max_x - min_x)//30
            r_y = (max_y - min_y)//30

            centers = torch.LongTensor([min_x,min_y]).view(1,-1)
            locitions = torch.LongTensor([0,0]).view(1,-1)
            x = min_x
            y = min_y

            while y < max_y:  
                x = min_x            
                while x < max_x:
                    centers = torch.cat((centers,torch.LongTensor([x,y]).view(1,-1)),dim=0)
                    locitions = torch.cat((locitions,torch.LongTensor([x//r_x,y//r_y]).view(1,-1)),dim=0)
                    x += 56                
                y += 56
            
            centers = centers[1:,:]
            locitions = locitions[1:,:]

            n_patches = len(centers)
            patches = torch.zeros((n_patches,patch_dim))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                patches[i] = patch.flatten()


            return patches, locitions, centers

        else:    
            n_patches = len(centers)
            
            patches = torch.zeros((n_patches,patch_dim))
            exps = torch.Tensor(exps)


            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                patches[i] = patch.flatten()

            if self.train:
                return patches, locitions, exps
            else: 
                return patches, locitions, exps, torch.Tensor(centers)
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        pre = self.img_dir+'/'+name[0]+'/'+name
        fig_name = os.listdir(pre)[0]
        path = pre+'/'+fig_name
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = self.cnt_dir+'/'+name+'.tsv'
        df = pd.read_csv(path,sep='\t',index_col=0)

        return df

    def get_loc(self,name):
        path = self.loc_dir+'/'+name+'_selection.tsv'
        # path = self.loc_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_lbl(self,name):
        # path = self.loc_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        loc = self.get_loc(name)
        meta = cnt.join((loc.set_index('id')))

        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)



class ViT_SKIN(torch.utils.data.Dataset):
    """Some Information about ViT_SKIN"""
    def __init__(self,train=True,gene_list=None,ds=None,sr=False,aug=False,norm=False,fold=0):
        super(ViT_SKIN, self).__init__()

        self.dir = '/ibex/scratch/pangm0a/spatial/data/GSE144240_RAW/'
        self.r = 224//4

        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i+'_ST_'+j)
        test_names = ['P2_ST_rep2']

        # gene_list = list(np.load('data/skin_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('data/skin_hvg_cut_1000.npy',allow_pickle=True))
        # gene_list = list(np.load('figures/mse_2000-vit_skin_a.npy',allow_pickle=True))

        self.gene_list = gene_list

        self.train = train
        self.sr = sr
        self.aug = aug
        self.transforms = transforms.Comloce([
            transforms.ColorJitter(0.5,0.5,0.5),
            transforms.ToTensor()
        ])
        self.norm = norm

        samples = names
        te_names = [samples[fold]]
        tr_names = list(set(samples)-set(te_names))

        if train:
            # names = names
            # names = names[3:]
            # names = test_names
            names = tr_names
        else:
            # names = [names[33]]
            # names = ['A1']
            # names = test_names
            names = te_names

        print('Loading imgs...')
        if self.aug:
            self.img_dict = {i: self.get_img(i) for i in names}
        else:
            self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in names}

        self.gene_set = list(gene_list)
        if self.norm:
            self.exp_dict = {i:sc.pp.scale(scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))) for i,m in self.meta_dict.items()}
        else:
            self.exp_dict = {i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i,m in self.meta_dict.items()}
        self.center_dict = {i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) for i,m in self.meta_dict.items()}
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))


    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i,exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp>0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:,j])


    def __getitem__(self, index):
        i = index
        im = self.img_dict[self.id2name[i]]
        if self.aug:
            im = self.transforms(im)
            # im = im.permute(1,2,0)
            im = im.permute(2,1,0)
        else:
            im = im.permute(1,0,2)
            # im = im

        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        loc = self.loc_dict[self.id2name[i]]
        locitions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4

        if self.sr:
            centers = torch.LongTensor(centers)
            max_x = centers[:,0].max().item()
            max_y = centers[:,1].max().item()
            min_x = centers[:,0].min().item()
            min_y = centers[:,1].min().item()
            r_x = (max_x - min_x)//30
            r_y = (max_y - min_y)//30

            centers = torch.LongTensor([min_x,min_y]).view(1,-1)
            locitions = torch.LongTensor([0,0]).view(1,-1)
            x = min_x
            y = min_y

            while y < max_y:  
                x = min_x            
                while x < max_x:
                    centers = torch.cat((centers,torch.LongTensor([x,y]).view(1,-1)),dim=0)
                    locitions = torch.cat((locitions,torch.LongTensor([x//r_x,y//r_y]).view(1,-1)),dim=0)
                    x += 56                
                y += 56
            
            centers = centers[1:,:]
            locitions = locitions[1:,:]

            n_patches = len(centers)
            patches = torch.zeros((n_patches,patch_dim))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                patches[i] = patch.flatten()


            return patches, locitions, centers

        else:    
            n_patches = len(centers)
            
            patches = torch.zeros((n_patches,patch_dim))
            exps = torch.Tensor(exps)

            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                patches[i] = patch.flatten()

            
                if self.train:
                    return patches, locitions, exps
                else: 
                    return patches, locitions, exps, torch.Tensor(centers)
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        path = glob.glob(self.dir+'*'+name+'.jpg')[0]
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = glob.glob(self.dir+'*'+name+'_stdata.tsv')[0]
        df = pd.read_csv(path,sep='\t',index_col=0)
        return df

    def get_loc(self,name):
        path = glob.glob(self.dir+'*spot*'+name+'.tsv')[0]
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        loc = self.get_loc(name)
        meta = cnt.join(loc.set_index('id'),how='inner')

        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)


class SKIN(torch.utils.data.Dataset):
    """Some Information about ViT_SKIN"""
    def __init__(self,train=True,gene_list=None,ds=None,sr=False,fold=0):
        super(SKIN, self).__init__()

        self.dir = '/ibex/scratch/pangm0a/spatial/data/GSE144240_RAW/'
        self.r = 224//2

        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i+'_ST_'+j)
        test_names = ['P2_ST_rep2']

        gene_list = list(np.load('data/skin_hvg_cut_1000.npy',allow_pickle=True))
        self.gene_list = gene_list

        self.train = train
        self.sr = sr

        samples = names
        te_names = [samples[fold]]
        tr_names = list(set(samples)-set(te_names))

        if train:
            # names = names
            # names = names[3:]
            # names = test_names
            names = tr_names
        else:
            # names = [names[33]]
            # names = ['A1']
            # names = test_names
            names = te_names

        print('Loading imgs...')
        self.img_dict = {i:self.get_img(i) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in names}

        self.gene_set = list(gene_list)
        self.exp_dict = {i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i,m in self.meta_dict.items()}
        self.center_dict = {i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) for i,m in self.meta_dict.items()}
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.transforms = transforms.Comloce([
            transforms.ColorJitter(0.5,0.5,0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        i = 0
        while index>=self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i-1]
        
        exp = self.exp_dict[self.id2name[i]][idx]
        center = self.center_dict[self.id2name[i]][idx]
        loc = self.loc_dict[self.id2name[i]][idx]

        exp = torch.Tensor(exp)
        loc = torch.Tensor(loc)

        x, y = center
        patch = self.img_dict[self.id2name[i]].crop((x-self.r, y-self.r, x+self.r, y+self.r))
        if self.train:
            patch = self.transforms(patch)
        else:
            patch = transforms.ToTensor()(patch)

        if self.train:
            return patch, loc, exp
        else: 
            return patch, loc, exp, torch.Tensor(center)

    def __len__(self):
        return self.cumlen[-1]

    def get_img(self,name):
        path = glob.glob(self.dir+'*'+name+'.jpg')[0]
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = glob.glob(self.dir+'*'+name+'_stdata.tsv')[0]
        df = pd.read_csv(path,sep='\t',index_col=0)
        return df

    def get_loc(self,name):
        path = glob.glob(self.dir+'*spot*'+name+'.tsv')[0]
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        loc = self.get_loc(name)
        meta = cnt.join(loc.set_index('id'),how='inner')

        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)


def calcADJ(loc, neighs=4, pruneTag='Grid'):
    """Calculate adjacency matrix for spatial locitions"""
    n_spots = len(loc)
    adj = np.zeros((n_spots, n_spots))
    
    for i in range(n_spots):
        distances = np.sqrt(np.sum((loc - loc[i])**2, axis=1))
        # Get nearest neighbors
        nearest_indices = np.argsort(distances)[1:neighs+1]  # Exclude self (distance=0)
        adj[i, nearest_indices] = 1
    
    return adj


class Symbol_Converter:
    """Dummy symbol converter class"""
    def __init__(self):
        pass


class ViT_HEST1K(torch.utils.data.Dataset):
    """ViT Dataset for HEST1K data - simplified to match ViT_HER2ST format"""
    def __init__(self, mode='train', gene_list=None, sr=False):
        super(ViT_HEST1K, self).__init__()

        self.hest_path = Path("/work/bose_lab/tahsin/data/HEST")
        self.patch_path = Path("../../../data/HERST_preprocess/patches_112x112") # Same patch size as her2st?
        self.processed_path = Path("../../data/HERST_preprocess")
        self.r = 224//4  # Same as ViT_HER2ST        
        self.gene_list = gene_list
        self.mode = mode
        self.sr = sr
        
        if gene_list == "HER2ST":
            self.processed_path = self.processed_path / 'HER2ST'
        elif gene_list == "3CA":
            self.processed_path = self.processed_path / '3CA_genes'
        elif gene_list == "Hallmark":
            self.processed_path = self.processed_path / 'Hallmark_genes'

        if mode == 'train':
            self.processed_path = self.processed_path / 'train'
        elif mode == 'val':
            self.processed_path = self.processed_path / 'val'
        else:
            self.processed_path = self.processed_path / 'test'
        
        print(f"Looking for HEST1K data in: {self.processed_path}")
        
        # Get sample IDs from available files
        if self.processed_path.exists():
            sample_files = list(self.processed_path.glob("*.h5ad"))
            self.sample_ids = [file.stem.split('_')[0] for file in sample_files]
            print(f"Found {len(self.sample_ids)} samples.")
        else:
            print(f"Warning: Path {self.processed_path} does not exist. Using dummy data.")
            self.sample_ids = ['dummy_sample']
        
        print('Loading HEST1K data...')
        # self.img_dict = {i: self.get_img(i) for i in self.sample_ids}
        # print('Loading metadata...')
        # self.meta_dict = {i: self.get_meta(i) for i in self.sample_ids}

        # self.gene_set = list(gene_list)
        # self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) 
        #                 for i, m in self.meta_dict.items()}
        # self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) 
        #                    for i, m in self.meta_dict.items()}
        # self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}

        # self.lengths = [len(i) for i in self.meta_dict.values()]
        # self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.sample_ids))

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        adata_path = os.path.join(self.processed_path, f"{sample_id}_preprocessed.h5ad")
        adata = ad.read_h5ad(adata_path)
        
        print(f"\nProcessing sample {sample_id}")
        
        # Make var_names unique before any indexing
        if not adata.var_names.is_unique:
            print(f"Found {sum(adata.var_names.duplicated())} duplicate gene names, making them unique")
            # Method 1: Make unique by appending _1, _2, etc. to duplicates
            adata.var_names_make_unique()
    
        exps = adata.X
        
        # Get array coordinates
        if 'array_row' in adata.obs and 'array_col' in adata.obs:
            loc = adata.obs[['array_row', 'array_col']].values.astype(int)
        elif 'spatial' in adata.obsm:
            loc = adata.obsm['spatial'].copy()
        else:
            print(f"Error: Sample {sample_id} does not have spatial coordinates.")
            loc = np.zeros((adata.n_obs, 2), dtype=int)  
            for i in range(adata.n_obs):
                loc[i] = [i // 64, i % 64]  
        

        # Normalize positions to [0, 63] range like HER2ST
        pos_min = loc.min(axis=0)
        pos_max = loc.max(axis=0)
        
        # Normalize to [0, 1] then scale to [0, 63]
        loc_normalized = (loc - pos_min) / (pos_max - pos_min + 1e-8)
        loc_scaled = (loc_normalized * 63).astype(int)
        loc_scaled = np.clip(loc_scaled, 0, 63)  # Ensure within bounds
        
        loc = torch.LongTensor(loc_scaled)
    
        # Get pixel coordinates
        if 'spatial' in adata.obsm:
            centers = adata.obsm['spatial']
        elif 'spatial' in adata.uns:
            centers = adata.uns['spatial']
        else:
            # print(f"Error: Sample {sample_id} does not have spatial coordinates in obsm['spatial'].")
            print(adata)
            raise ValueError(f"Sample {sample_id} does not have spatial coordinates in obsm['spatial'] or uns['spatial'].")
        # centers = adata.obsm['spatial']
        
        # Load Patches
        patch_path = os.path.join(self.patch_path, f"{sample_id}.h5")
        if os.path.exists(patch_path):
            patches = self._load_patches(sample_id, adata.obs_names)
        else:
            patches = np.random.randn(len(adata), 3, 112, 112)
            
        # In your __getitem__ method, add:
        print(f"Patches shape after loading: {patches.shape}")
        patches = torch.FloatTensor(patches)
        print(f"Patches shape as tensor: {patches.shape}")
        num_patches = patches.shape[0]
        patch_dim = 3 * 112 * 112  # 37632
        patches_flat = patches.view(num_patches, patch_dim)  # Shape: (num_spots, 37632)
    
        
        # patch_dim = 3 * self.r * self.r * 4
        # if self.sr:
        #     # Same super-resolution logic as ViT_HER2ST
        #     centers = torch.LongTensor(centers)
        #     max_x = centers[:, 0].max().item()
        #     max_y = centers[:, 1].max().item()
        #     min_x = centers[:, 0].min().item()
        #     min_y = centers[:, 1].min().item()
        #     r_x = (max_x - min_x) // 30
        #     r_y = (max_y - min_y) // 30

        #     centers = torch.LongTensor([min_x, min_y]).view(1, -1)
        #     locitions = torch.LongTensor([0, 0]).view(1, -1)
        #     x = min_x
        #     y = min_y

        #     while y < max_y:
        #         x = min_x
        #         while x < max_x:
        #             centers = torch.cat((centers, torch.LongTensor([x, y]).view(1, -1)), dim=0)
        #             locitions = torch.cat((locitions, torch.LongTensor([x // r_x, y // r_y]).view(1, -1)), dim=0)
        #             x += 56
        #         y += 56

        #     centers = centers[1:, :]
        #     locitions = locitions[1:, :]

        #     n_patches = len(centers)
        #     patches = torch.zeros((n_patches, patch_dim))
        #     for i in range(n_patches):
        #         center = centers[i]
        #         x, y = center
        #         patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
        #         patches[i] = patch.flatten()

        #     return patches, locitions, centers
            # n_patches = len(centers)
        exps = torch.Tensor(exps)
        centers = torch.FloatTensor(centers)
        loc = torch.LongTensor(loc)
        # patches = torch.FloatTensor(patches)

        if self.mode == 'train':
            return patches_flat, loc, exps
        else:
            return patches_flat, loc, exps, centers

    def __len__(self):
        return len(self.sample_ids)

    def _load_patches(self, sample_id, spot_names):
        patches = []
        path = os.path.join(self.patch_path, f"{sample_id}.h5")
        
        with h5py.File(path, 'r') as f:
            images = f['img'][:]
            barcodes = [bc[0].decode('utf-8') if isinstance(bc[0], bytes) else bc[0] for bc in f['barcode'][:]]
            
            barcode_to_idx = {bc: i for i, bc in enumerate(barcodes)}
            
            
            for spot in spot_names:
                if spot in barcode_to_idx:
                    idx = barcode_to_idx[spot]
                    img = images[idx]
                    # patches.append(images[idx])

                    # Why convert to tensor and normalize??
                    if len(img.shape) == 2:
                        img = np.stack([img, img, img], axis =0) # Convert grayscale to RGB
                    else:
                        img = img.transpose(2, 0, 1) # Convert HxWxC to CxHxW
                    patches.append(img)
                else:
                    patches.append(np.zeros((3, 112, 112)))
                    
        return np.array(patches)
    
    def get_gene_names(self):
        """Get gene names from the first sample"""
        if len(self.sample_ids) > 0:
            adata_path = os.path.join(self.processed_path, f"{self.sample_ids[0]}_preprocessed.h5ad")
            adata = ad.read_h5ad(adata_path)
            return adata.var_names.tolist()
        else:
            return []

if __name__ == '__main__':
    # dataset = VitDataset(diameter=112,sr=True)
    # dataset = ViT_HER2ST(train=True,mt=False)
    # dataset = ViT_SKIN(train=True,mt=False,sr=False,aug=False)
    
    # Test ViT_HEST1K dataset with same interface as ViT_HER2ST
    print("Testing ViT_HEST1K dataset...")
    dataset = ViT_HEST1K(train=True, fold=0)

    print(f"Dataset length: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Number of return values: {len(sample)}")
        print(f"Patches shape: {sample[0].shape}")
        print(f"locitions shape: {sample[1].shape}")
        print(f"Expression shape: {sample[2].shape}")
    
    # Test test mode
    print("\nTesting test mode...")
    dataset_test = ViT_HEST1K(train=False, fold=0)
    if len(dataset_test) > 0:
        sample_test = dataset_test[0]
        print(f"Number of return values (test): {len(sample_test)}")
        if len(sample_test) == 4:
            print(f"Centers shape: {sample_test[3].shape}")
    
    print("ViT_HEST1K testing completed.")