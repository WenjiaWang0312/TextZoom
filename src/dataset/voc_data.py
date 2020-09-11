import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import cv2
import sys
import os
import bisect
import warnings
from PIL import Image
import numpy as np
import string

sys.path.append('../')
from utils import str_filt
from utils.labelmaps import get_vocabulary, labels2strs
from IPython import embed
random.seed(0)


def rand_crop(im):
    w, h = im.size
    scale = 0.95
    p1 = (random.uniform(0, w * (1 - scale)), random.uniform(0, h * (1 - scale)))
    p2 = (p1[0] + scale * w, p1[1] + scale * h)
    return im.crop(p1 + p2)


def sp_noise(image, prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


class load_voc(Dataset):
    def __init__(self, root):
        super(load_voc, self).__init__()
        self.root = root
        im_names = os.listdir(self.root)
        for name in im_names:
            if '.jpg' not in name:
                im_names.remove(name)
        self.im_path = [os.path.join(self.root, im_name) for im_name in im_names]

    def __len__(self):
        return self.im_path.__len__()

    def __getitem__(self, index):
        im_input = Image.open(self.im_path[index])
        im_label = rand_crop(im_input)
        return im_input, im_label


class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img, noise=False):
        img = img.resize(self.size, self.interpolation)
        if noise:
            img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
            img = sp_noise(img, 0.5)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class alignCollate(object):

    def __init__(self, imgH=256, imgW=256, down_sample_scale=1):
        self.imgH = imgH
        self.imgW = imgW
        self.down_sample_scale = down_sample_scale

    def __call__(self, batch):
        images_lr, images_hr = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        transform = resizeNormalize((imgW, imgH))
        transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), True)
        images_hr = [transform(image) for image in images_hr]
        images_hr = torch.cat([t.unsqueeze(0) for t in images_hr], 0)

        images_lr = [image.resize((image.size[0]//self.down_sample_scale, image.size[1]//self.down_sample_scale), Image.BICUBIC) for image in images_lr]
        images_lr = [transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        return images_lr, images_hr