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
        for root, _, files in os.walk(ori_imgdir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_name = os.path.splitext(file)[0]
                    ucc_img_path = os.path.join(ucc_dirs[0], f"{img_name}.png")
                    if os.path.exists(ucc_img_path):
                        ori_img_path = os.path.join(root, file)
                        img_list.append((ori_img_path, ucc_img_path))
                        break  # Break loop after finding the matching image pair
    return img_list

class uwcc(data.Dataset):
    def __init__(self, ori_dirs, ucc_dirs, train=True, loader=img_loader):
        super(uwcc, self).__init__()

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
            oritransform = transforms.Compose([
                # Data augmentation transforms for training
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
