#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
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

scale = 0.90


def rand_crop(im):
    w, h = im.size
    p1 = (random.uniform(0, w*(1-scale)), random.uniform(0, h*(1-scale)))
    p2 = (p1[0] + scale*w, p1[1] + scale*h)
    return im.crop(p1 + p2)


def central_crop(im):
    w, h = im.size
    p1 = (((1-scale)*w/2), (1-scale)*h/2)
    p2 = ((1+scale)*w/2, (1+scale)*h/2)
    return im.crop(p1 + p2)


def buf2PIL(txn, key, type='RGB'):
    imgbuf = txn.get(key)
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    im = Image.open(buf).convert(type)
    return im


class lmdbDataset(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=31, test=True):
        super(lmdbDataset, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples

        self.max_len = max_len
        self.voc_type = voc_type

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)

        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())

        try:
            img = buf2PIL(txn, b'image_hr-%09d' % index, 'RGB')
        except TypeError:
            img = buf2PIL(txn, b'image-%09d' % index, 'RGB')
        except IOError or len(label) > self.max_len:
            return self[index + 1]

        label_str = str_filt(word, self.voc_type)
        return img, label_str


class lmdbDataset_real(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False):
        super(lmdbDataset_real, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        img_HR_key = b'image_hr-%09d' % index  # 128*32
        img_lr_key = b'image_lr-%09d' % index  # 64*16
        try:
            img_HR = buf2PIL(txn, img_HR_key, 'RGB')
            img_lr = buf2PIL(txn, img_lr_key, 'RGB')
        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)
        return img_HR, img_lr, label_str


class resizeNormalize(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mask = mask

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img_tensor = self.toTensor(img)
        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = self.toTensor(mask)
            img_tensor = torch.cat((img_tensor, mask), 0)
        return img_tensor


class lmdbDataset_mix(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False):
        super(lmdbDataset_mix, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        if self.test:
            try:
                img_HR = buf2PIL(txn, b'image_hr-%09d' % index, 'RGB')
                img_lr = buf2PIL(txn, b'image_lr-%09d' % index, 'RGB')
            except:
                img_HR = buf2PIL(txn, b'image-%09d' % index, 'RGB')
                img_lr = img_HR

        else:
            img_HR = buf2PIL(txn, b'image_hr-%09d' % index, 'RGB')
            if random.uniform(0, 1) < 0.5:
                img_lr = buf2PIL(txn, b'image_lr-%09d' % index, 'RGB')
            else:
                img_lr = img_HR

        label_str = str_filt(word, self.voc_type)
        return img_HR, img_lr, label_str


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate_syn(object):
    def __init__(self, imgH=64, imgW=256, down_sample_scale=4, keep_ratio=False, min_ratio=1, mask=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.down_sample_scale = down_sample_scale
        self.mask = mask

    def __call__(self, batch):
        images, label_strs = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        transform = resizeNormalize((imgW, imgH), self.mask)
        transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)

        images_hr = [transform(image) for image in images]
        images_hr = torch.cat([t.unsqueeze(0) for t in images_hr], 0)

        images_lr = [image.resize((image.size[0]//self.down_sample_scale, image.size[1]//self.down_sample_scale), Image.BICUBIC) for image in images]
        images_lr = [transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        return images_hr, images_lr, label_strs


class alignCollate_real(alignCollate_syn):
    def __call__(self, batch):
        images_HR, images_lr, label_strs = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        transform = resizeNormalize((imgW, imgH), self.mask)
        transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)
        images_HR = [transform(image) for image in images_HR]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        images_lr = [transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        return images_HR, images_lr, label_strs


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


if __name__ == '__main__':
    embed(header='dataset.py')
