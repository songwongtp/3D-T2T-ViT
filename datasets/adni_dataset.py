import sys
from os import path
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

### DIMENSION OF 3D VOLUME
dX = 105
dY = 126
dZ = 105

class ADNIDataset(Dataset):
    def __init__(self, root_dir, split, trainsplit=0.8, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.load_data(trainsplit)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input = np.take(self.x, idx, axis=0)
        target = np.take(self.y, idx, axis=0)
        
        if self.transform:
            input = self.transform(input)
        
        return input, target
        
    def load_data(self, trainsplit):
        # metadata = pd.read_csv(path.join(self.root_dir, 'metadata.csv'), sep=',')
        
        # x = self.load_X_data(metadata["FileName"].values).astype(np.float32)
        # y = metadata["Label"].values.astype(np.float32)

        metadata = pd.read_csv(path.join(self.root_dir, 'ADNI_MPRAGE_mr2pet_2_20_2020.csv'), sep=',')
        metadata = metadata.loc[metadata['Group'].isin(['AD', 'CN'])]
        metadata.loc[metadata['Group'] == 'AD', 'Label'] = 1.
        print ('AD', len(metadata.loc[metadata['Group'] == 'AD']))
        metadata.loc[metadata['Group'] == 'CN', 'Label'] = 0.
        print ('CN', len(metadata.loc[metadata['Group'] == 'CN']))
        x = self.load_X_data(metadata["Subject"].values).astype(np.float32)
        y = metadata["Label"].values.astype(np.float32)
        

        split_point = int(round(float(x.shape[0]) * trainsplit))
        
        if self.split == 'train':
            self.x = x[:split_point]
            self.y = y[:split_point]
        elif self.split == 'val':
            self.x = x[split_point:]
            self.y = y[split_point:]
        else:
            print('Invalid split')
            sys.exit(0)
    
    def load_X_data(self, fnames):
        dat = np.empty((len(fnames), 1, dX, dY, dZ), dtype=np.float32)
        print ('len', len(fnames))
        for f,i in zip(fnames, range(0,len(fnames))):
            # tmp = np.load(path.join(self.root_dir, f))
            tmp = np.load(path.join(self.root_dir, 'npys', f + '.npy'))
            dat[i,0,:,:,:] = tmp[:,:-1,:]
        return dat
