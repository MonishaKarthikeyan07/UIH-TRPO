import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def img_loader(path):
    img = Image.open(path)
    return img

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
            print('Found {} pairs of training images'.format(len(self.img_list)))
        else:
            print('Found {} pairs of testing images'.format(len(self.img_list)))

    def __getitem__(self, index):
        img_paths = self.img_list[index]
        sample = [self.loader(img_paths[i]) for i in range(len(img_paths))]

        if self.train:
            oritransform = transforms.Compose([
                # Data augmentation transforms for training (if needed)
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

class TRPOAgent:
    def __init__(self):
        self.dataset = None
        self.dataloader = None

    def prepare_data(self, ori_dirs, ucc_dirs, batch_size, n_workers, train=True):
        self.dataset = UWCCDataset(ori_dirs, ucc_dirs, train=train)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)

    def train(self, ori_dirs, ucc_dirs, batch_size, n_workers, epochs):
        self.prepare_data(ori_dirs, ucc_dirs, batch_size, n_workers, train=True)
        
        for epoch in range(epochs):
            for batch in self.dataloader:
                # Training loop
                pass

    def test(self, ori_dirs, ucc_dirs, batch_size, n_workers):
        self.prepare_data(ori_dirs, ucc_dirs, batch_size, n_workers, train=False)
        
        for batch in self.dataloader:
            # Testing loop
            pass

# Example usage:
# trpo_agent = TRPOAgent()
# trpo_agent.train(ori_dirs, ucc_dirs, batch_size, n_workers, epochs)
# trpo_agent.test(ori_dirs, ucc_dirs, batch_size, n_workers)
