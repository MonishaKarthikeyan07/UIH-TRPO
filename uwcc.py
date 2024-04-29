import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class uwcc(Dataset):
    def __init__(self, ori_dirs, ucc_dirs, train=True):
        self.ori_dirs = ori_dirs
        self.ucc_dirs = ucc_dirs
        self.train = train
        self.image_pairs = self.find_image_pairs()

        if len(self.image_pairs) == 0:
            print("Warning: Found 0 image pairs in given directories.")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        ori_path, ucc_path = self.image_pairs[index]
        ori_image = Image.open(ori_path)
        ucc_image = Image.open(ucc_path)

        # Convert images to numpy arrays and normalize
        ori_array = np.array(ori_image) / 255.0
        ucc_array = np.array(ucc_image) / 255.0

        # Convert arrays to tensors
        ori_tensor = torch.from_numpy(ori_array).permute(2, 0, 1).float()
        ucc_tensor = torch.from_numpy(ucc_array).permute(2, 0, 1).float()

        return ori_tensor, ucc_tensor

    def find_image_pairs(self):
        image_pairs = []

        for ori_dir, ucc_dir in zip(self.ori_dirs, self.ucc_dirs):
            ori_files = os.listdir(ori_dir)
            ucc_files = os.listdir(ucc_dir)

            for ori_file in ori_files:
                ori_path = os.path.join(ori_dir, ori_file)
                ucc_file = ori_file.replace('ori', 'ucc')
                ucc_path = os.path.join(ucc_dir, ucc_file)

                if ucc_file in ucc_files:
                    image_pairs.append((ori_path, ucc_path))

        return image_pairs
