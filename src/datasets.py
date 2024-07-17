import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint

from torch.utils.data import Dataset
from collections import defaultdict
from sklearn.utils import resample

from statsmodels.nonparametric.smoothers_lowess import lowess


# ベースライン補正の例
def baseline_correction(data):
    # 各サンプルの平均値を計算
    baseline = data.mean(dim=1, keepdim=True)
    
    # 平均値を各サンプルから引く
    corrected_data = data - baseline
    
    return corrected_data


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", resample_type: str = None) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        #self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))

        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt")).float()  # データをfloat型に変換
        self.X = baseline_correction(self.X)  # ベースライン補正を適用
        self.X = self.normalize(self.X)

        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
            
            if resample_type == 'oversample':
                self.X, self.y, self.subject_idxs = self._oversample()
            elif resample_type == 'undersample':
                self.X, self.y, self.subject_idxs = self._undersample()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    def normalize(self, X):
        mean = X.mean(dim=(0, 2), keepdim=True)
        std = X.std(dim=(0, 2), keepdim=True)
        return (X - mean) / (std + 1e-8)
            
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
    
    def _oversample(self):
        class_counts = defaultdict(int)
        for label in self.y:
            class_counts[label.item()] += 1
        
        max_count = max(class_counts.values())
        new_X, new_y, new_subject_idxs = [], [], []
        
        for label in range(self.num_classes):
            indices = torch.where(self.y == label)[0]
            if len(indices) > 0:
                sampled_indices = resample(indices, replace=True, n_samples=max_count)
                augmented_X = [self._augment(self.X[idx]) for idx in sampled_indices]
                new_X.extend(augmented_X)
                new_y.extend(self.y[sampled_indices])
                new_subject_idxs.extend(self.subject_idxs[sampled_indices])
        
        return torch.stack(new_X), torch.stack(new_y), torch.stack(new_subject_idxs)
    
    def _undersample(self):
        class_counts = defaultdict(int)
        for label in self.y:
            class_counts[label.item()] += 1
        
        min_count = min(class_counts.values())
        new_X, new_y, new_subject_idxs = [], [], []
        
        for label in range(self.num_classes):
            indices = torch.where(self.y == label)[0]
            if len(indices) > 0:
                sampled_indices = resample(indices, replace=False, n_samples=min_count)
                new_X.extend(self.X[sampled_indices])
                new_y.extend(self.y[sampled_indices])
                new_subject_idxs.extend(self.subject_idxs[sampled_indices])
        
        return torch.stack(new_X), torch.stack(new_y), torch.stack(new_subject_idxs)

    def _augment(self, x):
        aug_type = np.random.choice(['noise', 'time_shift', 'scale'])
        if aug_type == 'noise':
            noise = torch.randn_like(x) * 0.01
            return x + noise
        elif aug_type == 'time_shift':
            shift = np.random.randint(-10, 11)
            return torch.roll(x, shifts=shift, dims=-1)
        elif aug_type == 'scale':
            scale = np.random.uniform(0.9, 1.1)
            return x * scale
        