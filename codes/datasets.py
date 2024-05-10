import torch
import numpy as np

import pandas as pd

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils import data
from torch.utils.data import Subset, TensorDataset


from datasets import load_dataset
from transformers import BertTokenizer
from transformers import AutoTokenizer


from enum import auto
from codes import DictEnum


import itertools 



from collections import defaultdict
import random as random

from collections import defaultdict
from torch.utils.data import Subset
import random as random


class Dataset(DictEnum):
    Emotion = auto()
    CIFAR10 = auto()
    


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def CIFAR10(cfg, train):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    train_transform = transforms.Compose([])
    
    data_augmentation = True
    if data_augmentation:
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)
    
    cutout = True
    nholes = 1
    length = 16
    if cutout:
        train_transform.transforms.append(Cutout(n_holes=nholes, length=length))


    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    
    root = '/tmp'
    num_classes = 10
    train_dataset = datasets.CIFAR10(root=root,
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(root=root,
                                    train=False,
                                    transform=test_transform,
                                    download=True)

    if train is False:
#         nclasses = 10
#         mdnsamples = 400
#         val_inds, test_inds = list(), list()
#         for i in range(nclasses):
#             indices = [j for j, x in enumerate(dataset.targets) if x == i]
#             random.shuffle(indices)

#             val_inds += indices[:mdnsamples]
#             test_inds += indices[mdnsamples:]

#         return Subset(dataset, test_inds), Subset(dataset, val_inds)
        return test_dataset, None
    
    return train_dataset, (None, num_classes)
    
    

# def CIFAR10(cfg, train):
#     root = '/tmp'
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#     transform = transforms.Compose([transforms.ToTensor(), normalize])
#     dataset = datasets.CIFAR10(root, train=train, transform=transform, download=True)
#     if train is False:
# #         nclasses = 10
# #         mdnsamples = 400
# #         val_inds, test_inds = list(), list()
# #         for i in range(nclasses):
# #             indices = [j for j, x in enumerate(dataset.targets) if x == i]
# #             random.shuffle(indices)

# #             val_inds += indices[:mdnsamples]
# #             test_inds += indices[mdnsamples:]

# #         return Subset(dataset, test_inds), Subset(dataset, val_inds)
#         return dataset, None
    
#     return dataset, (None, 10)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]



class _Emotion(metaclass=Singleton):
# class _GoEmotions(metaclass=Singleton):
    def __init__(self, cfg):
        self.cfg = cfg
        self.classes = True

        self.dset = load_dataset('emotion')
        classes = self.dset["train"].features['label'].names

        self.dset.set_format(type="pandas")
        train_df = self.dset['train'][:]
        valid_df = self.dset['validation'][:]
        test_df = self.dset['test'][:]
        
        print(f"{len(train_df)=}")
        print(f"{len(valid_df)=}")
        print(f"{len(test_df)=}")

        # train_df = train_df.groupby('label').apply(lambda x: x.sample(200, random_state=cfg.seed)).reset_index(drop=True)
        # valid_df = valid_df.groupby('label').apply(lambda x: x.sample(70, random_state=cfg.seed)).reset_index(drop=True)
        # test_df = test_df.groupby('label').apply(lambda x: x.sample(20, random_state=cfg.seed)).reset_index(drop=True)
        
        # print(f"{train_df=}")

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        train_input_ids, train_att_masks = self.encode(train_df['text'].values.tolist())
        valid_input_ids, valid_att_masks = self.encode(valid_df['text'].values.tolist())
        test_input_ids, test_att_masks = self.encode(test_df['text'].values.tolist())

        train_y = torch.LongTensor(train_df['label'].values.tolist())
        valid_y = torch.LongTensor(valid_df['label'].values.tolist())
        test_y = torch.LongTensor(test_df['label'].values.tolist())

        self.train = TensorDataset(train_input_ids, train_att_masks, train_y)
        self.val = TensorDataset(valid_input_ids, valid_att_masks, valid_y)
        self.test = TensorDataset(test_input_ids, test_att_masks, test_y)


    # @staticmethod
    def encode(self, docs):
        '''
        This function takes list of texts and returns input_ids and attention_mask of texts
        '''
        # encoded_dict = self.tokenizer.batch_encode_plus(docs, add_special_tokens=True, max_length=128, padding='max_length', return_attention_mask=True, truncation=True, return_tensors='pt')
        encoded_dict = self.tokenizer(docs, add_special_tokens=True, max_length=128, padding='max_length', return_attention_mask=True, truncation=True, return_tensors='pt')

        input_ids = encoded_dict['input_ids']
        # print(f"{type(input_ids)=}")
        attention_masks = encoded_dict['attention_mask']
        # print(f"{type(attention_masks)=}")
        return input_ids, attention_masks


def Emotion(cfg, train):
    if train is True:
        return _Emotion(cfg).train, (None, 7)
    else:
        return _Emotion(cfg).test, _Emotion(cfg, train).val
