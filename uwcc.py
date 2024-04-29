import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class uwcc(Dataset):
    def __init__(self, ori_dirs, ucc_dirs, train=True):
        self.ori_dirs = ori_dirs
        self.ucc_dirs = ucc_dirs
        self.train = train
        self.ori_paths = []
        self.ucc_paths = []

        self._make_paths()
        if len(self.ori_paths) == 0 or len(self.ucc_paths) == 0:
            raise RuntimeError('Found 0 image pairs in given directories.')

    def __len__(self):
        return len(self.ori_paths)

    def __getitem__(self, idx):
        ori_array = np.array(Image.open(self.ori_paths[idx]))
        ucc_array = np.array(Image.open(self.ucc_paths[idx]))

        # Convert images to PyTorch tensors and normalize
        ori_tensor = torch.from_numpy(ori_array).permute(2, 0, 1).float()
        ucc_tensor = torch.from_numpy(ucc_array).permute(2, 0, 1).float()

        return ori_tensor, ucc_tensor

    def _make_paths(self):
        for ori_dir, ucc_dir in zip(self.ori_dirs, self.ucc_dirs):
            ori_paths = [os.path.join(ori_dir, img) for img in os.listdir(ori_dir) if img.endswith('.png')]
            ucc_paths = [os.path.join(ucc_dir, img) for img in os.listdir(ucc_dir) if img.endswith('.png')]
            self.ori_paths.extend(ori_paths)
            self.ucc_paths.extend(ucc_paths)
