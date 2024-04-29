import os
from PIL import Image
import torch.utils.data as data
from torchvision import transforms

def img_loader(path):
    img = Image.open(path)
    return img

def get_img_pairs(ori_dirs, ucc_dirs):
    img_pairs = []
    for ori_imgdir in ori_dirs:
        img_name = os.path.splitext(os.path.basename(ori_imgdir))[0]
        ucc_imgdir = os.path.join(os.path.dirname(ucc_dirs[0]), img_name + '.png')

        if os.path.exists(ucc_imgdir):
            img_pairs.append((ori_imgdir, ucc_imgdir))

    return img_pairs

class uwcc(data.Dataset):
    def __init__(self, ori_dirs, ucc_dirs, train=True, loader=img_loader):
        super(uwcc, self).__init__()

        self.img_pairs = get_img_pairs(ori_dirs, ucc_dirs)
        if len(self.img_pairs) == 0:
            raise RuntimeError('Found 0 image pairs in given directories.')

        self.train = train
        self.loader = loader

        if self.train:
            print(f'Found {len(self.img_pairs)} pairs of training images')
        else:
            print(f'Found {len(self.img_pairs)} pairs of testing images')
            
    def __getitem__(self, index):
        ori_path, ucc_path = self.img_pairs[index]
        ori_img = self.loader(ori_path)
        ucc_img = self.loader(ucc_path)

        if self.train:
            ori_transform = transforms.Compose([
                # Add your training transformations here
                transforms.ToTensor(),
            ])
            ucc_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            ori_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            ucc_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        ori_img = ori_transform(ori_img)
        ucc_img = ucc_transform(ucc_img)

        return ori_img, ucc_img

    def __len__(self):
        return len(self.img_pairs)
