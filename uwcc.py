import torch.utils.data as data
import os
from PIL import Image
from torchvision import transforms

def img_loader(path):
    img = Image.open(path)
    return img

def get_imgs_list(ori_dirs, ucc_dirs):
    img_list = []
    for ori_imgdir in ori_dirs:
        img_name = os.path.splitext(os.path.basename(ori_imgdir))[0]
        ucc_imgdir = os.path.join(os.path.dirname(ucc_dirs[0]), img_name + '.png')

        if os.path.exists(ucc_imgdir):
            img_list.append((ori_imgdir, ucc_imgdir))
        else:
            print(f"UCC image '{ucc_imgdir}' not found for '{ori_imgdir}'")

    return img_list

class UWCCDataset(data.Dataset):
    def __init__(self, ori_dirs, ucc_dirs, train=True, loader=img_loader):
        super(UWCCDataset, self).__init__()

        self.img_list = get_imgs_list(ori_dirs, ucc_dirs)
        if len(self.img_list) == 0:
            raise RuntimeError('Found 0 image pairs in given directories.')
        else:
            print(f'Found {len(self.img_list)} image pairs in given directories.')

        self.train = train
        self.loader = loader

        if self.train:
            print('Training mode')
        else:
            print('Testing mode')
            
    def __getitem__(self, index):
        img_paths = self.img_list[index]
        sample = [self.loader(img_paths[i]) for i in range(len(img_paths))]

        # Apply transformations based on the mode
        if self.train:
            oritransform = transforms.Compose([
                # Add your training transformations here
                transforms.ToTensor(),
            ])
            ucctransform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            oritransform = transforms.Compose([
                transforms.ToTensor(),
            ])
            ucctransform = transforms.Compose([
                transforms.ToTensor(),
            ])
        
        # Apply transformations to each image
        sample[0] = oritransform(sample[0])
        sample[1] = ucctransform(sample[1])

        return sample

    def __len__(self):
        return len(self.img_list)
