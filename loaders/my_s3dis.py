import os
import h5py
import numpy as np
import torch.utils.data as data
from tqdm import tqdm
import pdb

class MYS3DIS(data.Dataset):
    def __init__(self, root, training=True):
        self.root = root
        self.split = 'train.txt' if training else 'my_test.txt'
        self.flist = os.path.join(self.root, 'metadata', self.split)
        self.rooms = [line.strip() for line in open(self.flist)]
        # Load all data into memory
        self.coords = []
        self.points = []
        print('> Loading h5 files...')
        for fname in tqdm(self.rooms, ascii=True):
            fin = h5py.File(os.path.join(self.root, 'my_h5', fname))
            pdb.set_trace()
            self.coords.append(fin['coords'][:])
            self.points.append(fin['points'][:])
            fin.close()
        self.coords = np.concatenate(self.coords, axis=0)
        self.points = np.concatenate(self.points, axis=0)
        # Post-processing
        self.dataset_size = self.points.shape[0]
        self.num_points = self.points.shape[1]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        return {
            'coords': self.coords[i],
            'points': self.points[i],
        }