import os
import csv
from glob import glob

import torch
import pandas as pd

class TabularTrans:
    
    def __init__(self):
        self.x_max, self.x_min = 0, 0
        self.y_max, self.y_min = 0, 0
    
    def transform(self, origin, is_x=True):
        """
        original 2D torch data
        """
        col_max = torch.max(origin, dim=0)[0]
        col_min = torch.min(origin, dim=0)[0]
        transed = (origin - col_min) / (col_max - col_min)
        if is_x:
            self.x_max = col_max
            self.x_min = col_min
        else:
            self.y_max = col_max
            self.y_min = col_min
        return transed
        
    def inverse(self, transed, is_x=True):
        if not isinstance(transed, torch.Tensor):
            transed = torch.Tensor(transed)
        if is_x:
            inversed = transed * (self.x_max - self.x_min) + self.x_min
        else:
            inversed = transed * (self.y_max - self.y_min) + self.y_min
        return inversed
        

class CatalogDataset(torch.utils.data.Dataset):
    
    def __init__(self, src_dir="../catalog_data", feature_col=["coat"], targets=["V"], files=["d5_soku_lbyD_less3.csv"]):
        self.features = []
        self.targets = []
        for f in files:
            df = pd.read_csv(os.path.join(src_dir, f))
            df = df.dropna()
            self.features.extend(df[feature_col].values)
            self.targets.extend(df[targets].values)
        
        #print(self.features[:2])
        #print(self.targets[:2])
        self.features = torch.Tensor(self.features)
        self.targets = torch.Tensor(self.targets)
        #print(self.features[:2])
        #print(self.targets[:2])
        
        self.trans = TabularTrans()
        self.features = self.trans.transform(self.features)
        self.targets = self.trans.transform(self.targets, is_x=False)
        print("dataset shape x: {}, y: {}".format(self.features.shape, self.targets.shape))
        #print(self.features[:2])
        #print(self.targets[:2])
            
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        return x, y