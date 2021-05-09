# for generating images from images

import numpy as np
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
from random import seed, shuffle

### DIMENSION OF 3D VOLUME
dX = 105
dY = 127
dZ = 105

# vectorized is return, check reshape lines for changing this
class ADNIProblemDataset():

    def __init__(self, batch_size, trainsplit=0.8):

        self.datapath = ''
        self.metapath = 'metadata.csv'
        if Path(self.datapath).exists() and Path(self.metapath).exists():
            pass
        else:
            print('Paths specified do not exist.')
            sys.exit(0)

        self.batch_size = batch_size
        self.trainsplit = trainsplit

        metadata = pd.read_csv(self.metapath, sep=',')

        self.total_samples = metadata.shape[0]

        # loads all image data into memory!!!
        print('Loading all ADNI images (~400) into RAM...')
        X = self.load_X_data(metadata["FileName"].values).astype(np.float32)
        print('Done.\n')

        y = metadata["Label"].values.astype(np.float32)
        x_control = metadata["Manufacturer"].values.astype(np.float32)

        """permute the data randomly"""
        self.perm = list(range(0,self.total_samples))
        shuffle(self.perm)

        X = X[self.perm]
        y = y[self.perm]
        x_control = x_control[self.perm]

        self.split_into_train_valid(X, y, x_control)

        self.num_train = self.x_train.shape[0]
        self.num_valid = self.x_valid.shape[0]

        self.num_train_batches = int(np.ceil(self.num_train/self.batch_size))
        self.num_valid_batches = int(np.ceil(self.num_valid/self.batch_size))

        self.num_batches = {}
        self.num_batches['train'] = self.num_train_batches
        self.num_batches['valid'] = self.num_valid_batches

    def get_batch(self, batch_idx, train=True):
        start_idx = batch_idx * self.batch_size
        end_idx = start_idx + self.batch_size
        sidx = start_idx
        eidx = end_idx

        if train=='train':
            return self.x_train[sidx:eidx], self.x_control_train[sidx:eidx], self.y_train[sidx:eidx]
        elif train=='valid':
            return self.x_valid[sidx:eidx], self.x_control_valid[sidx:eidx], self.y_valid[sidx:eidx]
        else:
            error("Unknown dataset split type, choices are train and test.")


    def get_train_batch(self, batch_idx):
        start_idx = batch_idx * self.batch_size
        end_idx = start_idx + self.batch_size
        sidx = start_idx
        eidx = end_idx
        return self.x_train[sidx:eidx], self.x_control_train[sidx,eidx], self.y_train[sidx:eidx]

    def get_valid_batch(self, batch_idx):
        start_idx = batch_idx * self.batch_size
        end_idx = start_idx + self.batch_size
        sidx = start_idx
        eidx = end_idx
        return self.x_valid[sidx:eidx], self.x_control_valid[sidx,eidx], self.y_valid[sidx:eidx]

    def load_X_data(self, fnames):
        dat = np.empty((len(fnames), dX, dY, dZ), dtype=np.float32)
        for f,i in zip(fnames, range(0,len(fnames))):
            tmp = np.load(self.datapath+f)
            dat[i,:,:,:] = tmp
        return dat

    def split_into_train_valid(self, x, y, x_control):
        split_point = int(round(float(x.shape[0]) * self.trainsplit))
        self.x_train = x[:split_point]
        self.x_valid = x[split_point:]
        self.y_train = y[:split_point]
        self.y_valid = y[split_point:]
        self.x_control_train = x_control[:split_point]
        self.x_control_valid = x_control[split_point:]
                                                            

    def shuffle_train(self):
        perm= list(range(0,self.num_train))
        shuffle(self.perm)

        self.x_train = self.x_train[perm]
        self.x_control_train = self.x_control_train[perm]
        self.y_train = self.y_train[perm]
