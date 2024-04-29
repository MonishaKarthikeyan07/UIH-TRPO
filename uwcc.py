import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def img_loader(path):
    img = Image.open(path)
    return img

def enhance_image(image):
    # Your image enhancement process goes here
    # For example, you can apply a transformation pipeline using torchvision.transforms
    # Here's a simple example:
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    enhanced_image = transform(image)
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

        if self.train:
            # Apply image enhancement to the first image (original image)
            sample[0] = enhance_image(sample[0])

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        sample[1] = transform(sample[1])  # Transform the UCC image to a tensor

        return sample

    def __len__(self):
        return len(self.img_list)
