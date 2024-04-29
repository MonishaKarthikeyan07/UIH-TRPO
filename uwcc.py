import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def img_loader(path):
    img = Image.open(path)
    return img

def get_imgs_list(ori_dirs, ucc_dirs):
    img_list = []
    for ori_imgdir in ori_dirs:
        for img_file in os.listdir(ori_imgdir):
            if img_file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_name = os.path.splitext(img_file)[0]
                ucc_imgfile = os.path.join(os.path.dirname(ucc_dirs[0]), img_name + '.png')

                if ucc_imgfile in ucc_dirs:
                    ori_imgpath = os.path.join(ori_imgdir, img_file)
                    img_list.append((ori_imgpath, ucc_imgfile))

    return img_list



class UWCCDataset(Dataset):
    def __init__(self, ori_dirs, ucc_dirs, train=True, loader=img_loader):
        super(UWCCDataset, self).__init__()

        print("Original image directories:")
        print(ori_dirs)
        print("UCC image directories:")
        print(ucc_dirs)

        self.img_list = get_imgs_list(ori_dirs, ucc_dirs)
        if len(self.img_list) == 0:
            raise RuntimeError('Found 0 image pairs in given directories. Please ensure that each original image has a corresponding UCC image.')

        self.train = train
        self.loader = loader

        if self.train:
            print(f'Found {len(self.img_list)} pairs of training images')
        else:
            print(f'Found {len(self.img_list)} pairs of testing images')
            
    def __getitem__(self, index):
        img_paths = self.img_list[index]
        sample = [self.loader(img_paths[i]) for i in range(len(img_paths))]

        if self.train:
            oritransform = transforms.Compose([
                transforms.ToTensor(),
            ])
            ucctransform = transforms.Compose([
                transforms.ToTensor(),
            ])
            sample[0] = oritransform(sample[0])
            sample[1] = ucctransform(sample[1])
        else:
            oritransform = transforms.Compose([
                transforms.ToTensor(),
            ])
            ucctransform = transforms.Compose([
                transforms.ToTensor(),
            ])
            sample[0] = oritransform(sample[0])
            sample[1] = ucctransform(sample[1])

        return sample

    def __len__(self):
        return len(self.img_list)
