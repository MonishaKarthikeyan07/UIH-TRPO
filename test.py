import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import datetime
from trpo import TRPO
from model import PhysicalNN

def main(checkpoint_path):

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = PhysicalNN()
    model = torch.nn.DataParallel(model).to(device)
    print("=> loading trained model")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model at epoch {}".format(checkpoint['epoch']))
    model = model.module
    model.eval()

    # TRPO instance
    trpo = TRPO(model)

    # Test image path
    imgs_path = '/content/drive/MyDrive/Dataset/test_images/'

    # Result path
    result_path = '/content/drive/MyDrive/Dataset/results/'

    # Get list of test images
    ori_dirs = [os.path.join(imgs_path, image) for image in os.listdir(imgs_path)]

    testtransform = transforms.Compose([
                transforms.ToTensor(),
            ])
    unloader = transforms.ToPILImage()

    starttime = datetime.datetime.now()
    for imgdir in ori_dirs:
        img_name = os.path.splitext(os.path.basename(imgdir))[0]
        img = Image.open(imgdir)
        inp = testtransform(img).unsqueeze(0)
        inp = inp.to(device)
        out = trpo.predict(inp)

        corrected = unloader(out.cpu().squeeze(0))
        dir = os.path.join(result_path, 'results_{}'.format(checkpoint['epoch']))
        if not os.path.exists(dir):
            os.makedirs(dir)
        corrected.save(os.path.join(dir, '{}_corrected.png'.format(img_name)))
    endtime = datetime.datetime.now()
    print("Total time taken:", endtime - starttime)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python test.py CHECKPOINT_PATH")
    else:
        _, checkpoint_path = sys.argv
        main(checkpoint_path)
