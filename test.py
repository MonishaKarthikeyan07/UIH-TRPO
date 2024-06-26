import os
import torch
import numpy as np
from PIL import Image
from model import PhysicalNN
import argparse
from torchvision import transforms
import datetime


def main(checkpoint, imgs_path, result_path):

    ori_dirs = []
    for image in os.listdir(imgs_path):
        ori_dirs.append(os.path.join(imgs_path, image))

    # Load model
    model = PhysicalNN()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['state_dict'])
    print("=> loading trained model")
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model at epoch {}".format(checkpoint['epoch']))
    model.eval()

    testtransform = transforms.Compose([
                transforms.ToTensor(),
            ])
    unloader = transforms.ToPILImage()

    starttime = datetime.datetime.now()
    for imgdir in ori_dirs:
        img_name = (imgdir.split('/')[-1]).split('.')[0]
        img = Image.open(imgdir)
        inp = testtransform(img).unsqueeze(0)
        out = model(inp)

        corrected = unloader(out.cpu().squeeze(0))
        dir = '{}/results_{}'.format(result_path, checkpoint['epoch'])
        if not os.path.exists(dir):
            os.makedirs(dir)
        corrected.save(dir+'/{}corrected.png'.format(img_name))
    endtime = datetime.datetime.now()
    print(endtime-starttime)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='checkpoints path', required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint

    # Default paths for images and results folders
    imgs_path = './test_img/'
    result_path = './results/'

    main(checkpoint=checkpoint, imgs_path=imgs_path, result_path=result_path)
