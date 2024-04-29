import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def img_loader(path):
    img = Image.open(path)
    return img

def enhance_image(image):
    # Add image enhancement logic here
    # This could involve algorithms like histogram equalization, gamma correction, etc.
    # For demonstration, we'll use a simple brightness adjustment
    enhanced_image = transforms.functional.adjust_brightness(image, 1.2)
    return enhanced_image

def get_imgs_list(ori_dirs, ucc_dirs):
    img_list = []
    for ori_imgdir in ori_dirs:
        img_name = os.path.splitext(os.path.basename(ori_imgdir))[0]
        ucc_imgdir = os.path.join(os.path.dirname(ucc_dirs[0]), img_name + '.png')

        if ucc_imgdir in ucc_dirs:
            img_list.append((ori_imgdir, ucc_imgdir))

    return img_list

class UWCCDataset(Dataset):
    def __init__(self, ori_dirs, ucc_dirs, train=True, loader=img_loader):
        super(UWCCDataset, self).__init__()

        self.img_list = get_imgs_list(ori_dirs, ucc_dirs)
        if len(self.img_list) == 0:
            raise RuntimeError('Found 0 image pairs in given directories.')

        self.train = train
        self.loader = loader

        if self.train:
            print(f'Found {len(self.img_list)} pairs of training images')
        else:
            print(f'Found {len(self.img_list)} pairs of testing images')
            
    def __getitem__(self, index):
        img_paths = self.img_list[index]
        sample = [self.loader(img_paths[i]) for i in range(len(img_paths))]

        # Image enhancement
        if self.train:
            sample[0] = enhance_image(sample[0])

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
