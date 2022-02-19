# -*- coding: utf-8 -*-

"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.

Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

import os
import sys
import cv2
import json
import uuid
import tqdm
import math
import torch
import random
# import h5py
import numpy as np
import os.path as osp
import scipy.io as sio
import torch.utils.data as data
from collections import OrderedDict
sys.path.append('.')
import operator
import utils
from utils import Corpus

import argparse
import collections
import logging
import json
import re

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
# from transformers import BertTokenizer,BertModel
from utils.transforms import letterbox, random_affine

import albumentations

sys.modules['utils'] = utils

cv2.setNumThreads(0)

def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples

## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features

class DatasetNotFoundError(Exception):
    pass

class MyAugment:
    def __init__(self) -> None:
        self.transform = albumentations.Compose([
                albumentations.Blur(p=0.01),
                albumentations.MedianBlur(p=0.01),
                albumentations.ToGray(p=0.01),
                albumentations.CLAHE(p=0.01),
                albumentations.RandomBrightnessContrast(p=0.0),
                albumentations.RandomGamma(p=0.0),
                albumentations.ImageCompression(quality_lower=75, p=0.0)])
    
    def augment_hsv(self, im, hgain=0.5, sgain=0.5, vgain=0.5):
        # HSV color-space augmentation
        if hgain or sgain or vgain:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_RGB2HSV))
            dtype = im.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB, dst=im)  # no return needed

    def __call__(self, img, bbox, phrase):
        imgh,imgw, _ = img.shape
        x, y, w, h = (bbox[0]+bbox[2])/2/imgw, (bbox[1]+bbox[3])/2/imgh, (bbox[2]-bbox[0])/imgw, (bbox[3]-bbox[1])/imgh
        img = self.transform(image=img)['image']
        #self.augment_hsv(img)
        # Flip up-down
        if random.random() < 0.5:
            img = np.flipud(img)
            #labels[:, 2] = 1 - labels[:, 2]
            y = 1-y
            phrase = phrase.replace('south','*&^special^&*').replace('north','south').replace('*&^special^&*','north')

        # Flip left-right
        if random.random() < 0.5:
            img = np.fliplr(img)
            #labels[:, 1] = 1 - labels[:, 1]
            x = 1-x
            phrase = phrase.replace('west','*&^special^&*').replace('east','west').replace('*&^special^&*','east')
        #
        new_imgh, new_imgw, _ = img.shape
        assert new_imgh==imgh, new_imgw==imgw
        x, y, w, h = x*imgw, y*imgh, w*imgw, h*imgh

        # Crop image
        iscropped=False
        if random.random() < 0.5:
            left, top, right, bottom = x-w/2, y-h/2, x+w/2, y+h/2
            if left >= new_imgw/2:
                start_cropped_x = random.randint(0, int(0.15*new_imgw))
                img = img[:, start_cropped_x:, :]
                left, right = left - start_cropped_x, right - start_cropped_x
            if right <= new_imgw/2:
                start_cropped_x = random.randint(int(0.85*new_imgw), new_imgw)
                img = img[:, 0:start_cropped_x, :]
            if top >= new_imgh/2:
                start_cropped_y = random.randint(0, int(0.15*new_imgh))
                img = img[start_cropped_y:, :, :]
                top, bottom = top - start_cropped_y, bottom - start_cropped_y
            if bottom <= new_imgh/2:
                start_cropped_y = random.randint(int(0.85*new_imgh), new_imgh)
                img = img[0:start_cropped_y, :, :]
            cropped_imgh, cropped_imgw, _ = img.shape
            left, top, right, bottom = left/cropped_imgw, top/cropped_imgh, right/cropped_imgw, bottom/cropped_imgh
            if cropped_imgh != new_imgh or cropped_imgw != new_imgw:
                img = cv2.resize(img, (new_imgh, new_imgw))
            new_cropped_imgh, new_cropped_imgw, _ = img.shape
            left, top, right, bottom = left*new_cropped_imgw, top*new_cropped_imgh, right*new_cropped_imgw, bottom*new_cropped_imgh 
            x, y, w, h = (left+right)/2, (top+bottom)/2, right-left, bottom-top
            iscropped=True
        #if iscropped:
        #    print((new_imgw, new_imgh))
        #    print((cropped_imgw, cropped_imgh), flush=True)
        #    print('============')
        #print(type(img))
        #draw_bbox = np.array([x-w/2, y-h/2, x+w/2, y+h/2], dtype=int)
        #img_new=bbv.draw_rectangle(img, draw_bbox)
        #cv2.imwrite('tmp/'+str(random.randint(0,5000))+"_"+str(iscropped)+".jpg", img_new)

        new_bbox = [(x-w/2), y-h/2, x+w/2, y+h/2]
        #print(bbox)
        #print(new_bbox)
        #print('---end---')
        return img, np.array(new_bbox, dtype=int), phrase

class ReferDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'referit': {'splits': ('train', 'val', 'trainval', 'test')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'gref_umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
        },
        'flickr': {
            'splits': ('train', 'val', 'test')},
        'rsvg': {
            'splits': ('train', 'val', 'test')
        },
    }

    def __init__(self, data_root, split_root='data', dataset='referit', imsize=256,
                 transform=None, augment=False, return_idx=False, testmode=False,
                 split='train', max_query_len=128, lstm=False, bert_model='bert-base-uncased'):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.imsize = imsize
        self.query_len = max_query_len
        self.lstm = lstm
        self.transform = transform
        self.testmode = testmode
        self.split = split
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.augment=augment
        self.return_idx=return_idx
        self.myaugment = MyAugment()

        if self.dataset == 'referit':
            self.dataset_root = osp.join(self.data_root, 'referit')
            self.im_dir = osp.join(self.dataset_root, 'images')
            self.split_dir = osp.join(self.dataset_root, 'splits')
        elif  self.dataset == 'flickr':
            self.dataset_root = osp.join(self.data_root, 'Flickr30k')
            self.im_dir = osp.join(self.dataset_root, 'flickr30k_images')
        elif self.dataset == 'rsvg':
            self.dataset_root = osp.join(self.data_root, 'rsvg')
            self.im_dir = osp.join(self.dataset_root, 'images')
        else:   ## refcoco, etc.
            self.dataset_root = osp.join(self.data_root, 'other')
            self.im_dir = osp.join(
                self.dataset_root, 'images', 'mscoco', 'images', 'train2014')
            self.split_dir = osp.join(self.dataset_root, 'splits')

        if not self.exists_dataset():
            # self.process_dataset()
            print('Please download index cache to data folder: \n \
                https://drive.google.com/open?id=1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ')
            exit(0)

        dataset_path = osp.join(self.split_root, self.dataset)
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        if self.lstm:
            self.corpus = Corpus()
            corpus_path = osp.join(dataset_path, 'corpus.pth')
            self.corpus = torch.load(corpus_path)

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        splits = [split]
        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(dataset_path, imgset_file)
            self.images += torch.load(imgset_path)
        #self.images = self.images[0:100]

    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))

    def pull_item(self, idx):
        if self.dataset == 'flickr':
            img_file, bbox, phrase = self.images[idx]
        elif self.dataset == 'rsvg':
            img_file, bbox, phrase, _ = self.images[idx]
        else:
            img_file, _, bbox, phrase, attri = self.images[idx]
        ## box format: to x1y1x2y2
        if self.dataset == 'rsvg':
            bbox = np.array(bbox, dtype=int)
        else:
            assert False
        #if not (self.dataset == 'referit' or self.dataset == 'flickr'):
        #    bbox = np.array(bbox, dtype=int)
        #    bbox[2], bbox[3] = bbox[0]+bbox[2], bbox[1]+bbox[3]
        #else:
        #    bbox = np.array(bbox, dtype=int)

        img_path = osp.join(self.im_dir, img_file)
        img = cv2.imread(img_path)
        #print(img)
        #print(img_path, flush=True)
        
        ## duplicate channel if gray image
        if img.shape[-1] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.stack([img] * 3)
        return img, phrase, bbox

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, phrase, bbox = self.pull_item(idx)
        # phrase = phrase.decode("utf-8").encode().lower()
        #phrase = "find a water tower that is about 170 square meters. there is a ground track field, located approximately 34 meters southeast of the water tower. there is a tennis court that is about 2281 square meters, located approximately 275 meters southwest of the water tower."
        phrase = phrase.lower()
        if self.augment:
            #img, bbox, phrase = self.myaugment(img, bbox, phrase)
            augment_flip, augment_hsv, augment_affine = True,True,True

        ## seems a bug in torch transformation resize, so separate in advance
        h,w = img.shape[0], img.shape[1]
        if self.augment:
        #if False:
            ## random horizontal flip
            if augment_flip and random.random() > 0.5:
                img = cv2.flip(img, 1)
                bbox[0], bbox[2] = w-bbox[2]-1, w-bbox[0]-1
                #phrase = phrase.replace('right','*&^special^&*').replace('left','right').replace('*&^special^&*','left')
                phrase = phrase.replace('west','*&^special^&*').replace('east','west').replace('*&^special^&*','east')
            ## random intensity, saturation change
            if augment_hsv:
                fraction = 0.50
                img_hsv = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
                S = img_hsv[:, :, 1].astype(np.float32)
                V = img_hsv[:, :, 2].astype(np.float32)
                a = (random.random() * 2 - 1) * fraction + 1
                if a > 1:
                    np.clip(S, a_min=0, a_max=255, out=S)
                a = (random.random() * 2 - 1) * fraction + 1
                V *= a
                if a > 1:
                    np.clip(V, a_min=0, a_max=255, out=V)

                img_hsv[:, :, 1] = S.astype(np.uint8)
                img_hsv[:, :, 2] = V.astype(np.uint8)
                img = cv2.cvtColor(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB)
            img, _, ratio, dw, dh = letterbox(img, None, self.imsize)
            bbox[0], bbox[2] = bbox[0]*ratio+dw, bbox[2]*ratio+dw
            bbox[1], bbox[3] = bbox[1]*ratio+dh, bbox[3]*ratio+dh
            ## random affine transformation
            if augment_affine:
                img, _, bbox, M = random_affine(img, None, bbox, \
                    degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))
        else:   ## should be inference, or specified training
            img, _, ratio, dw, dh = letterbox(img, None, self.imsize)
            bbox[0], bbox[2] = bbox[0]*ratio+dw, bbox[2]*ratio+dw
            bbox[1], bbox[3] = bbox[1]*ratio+dh, bbox[3]*ratio+dh

        ## Norm, to tensor
        if self.transform is not None:
            img = self.transform(img)
        if self.lstm:
            phrase = self.tokenize_phrase(phrase)
            word_id = phrase
            # word_mask = np.zeros(word_id.shape)
            word_mask = np.array(word_id>0,dtype=int)
        else:
            ## encode phrase to bert input
            examples = read_examples(phrase, idx)
            features = convert_examples_to_features(
                examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
            word_id = features[0].input_ids
            word_mask = features[0].input_mask
        if self.testmode:
            return img, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
                np.array(bbox, dtype=np.float32), np.array(ratio, dtype=np.float32), \
                np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0]
        else:
            return img, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
            np.array(bbox, dtype=np.float32)