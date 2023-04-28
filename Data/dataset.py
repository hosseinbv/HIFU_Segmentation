import random
from skimage.io import imread, imsave

import torch
from torch.utils import data
import torchvision.transforms.functional as TF
import glob, os
import json
import numpy as np
from skimage.io import imsave
import cv2

class SegDataset(data.Dataset):
    def __init__(
        self,
        input_paths_b: list,
        input_paths_a: list,

        target_paths: list,
        transform_input=None,
        transform_target=None,
        hflip=False,
        vflip=False,
        affine=False,
        margin = 25,
    ):
        self.input_paths_b = input_paths_b
        self.input_paths_a = input_paths_a
        self.target_paths = target_paths
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.hflip = hflip
        self.vflip = vflip
        self.affine = affine
        self.margin = margin

    def __len__(self):
        return len(self.input_paths_b)

    def resize2SquareKeepingAspectRation(img, size, interpolation=cv2.INTER_AREA):
        h, w = img.shape[:2]
        c = None if len(img.shape) < 3 else img.shape[2]
        if h == w: return cv2.resize(img, (size, size), interpolation)
        if h > w: dif = h
        else:     dif = w
        x_pos = int((dif - w)/2.)
        y_pos = int((dif - h)/2.)
        if c is None:
            mask = np.zeros((dif, dif), dtype=img.dtype)
            mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
        else:
            mask = np.zeros((dif, dif, c), dtype=img.dtype)
            mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
        return cv2.resize(mask, (size, size), interpolation)

    def __getitem__(self, index: int):
        input_ID_b = self.input_paths_b[index]
        input_ID_a = os.path.join(self.input_paths_a, os.path.split(input_ID_b)[1].replace('b', 'a'))
        target_ID = os.path.join(self.target_paths, os.path.split(input_ID_b)[1].replace('b', 'a').split('.')[0]+'.png')
        json_path = os.path.join(self.target_paths, os.path.split(input_ID_b)[1].replace('b', 'a').split('.')[0]+'.json')
        # print(os.path.split(input_ID_b)[-1], ' - ', os.path.split(input_ID_a)[-1], ' - ', os.path.split(json_path)[-1])
        xb = imread(input_ID_b)
        xa = imread(input_ID_a) 
        try:
            y = imread(target_ID)
        except:
            y = []

        try:
            f = open(json_path)
        except:
            raise("No json file is found for " + input_ID_a)
        
        data = json.load(f)
        points = data['shapes'][0].get('points')
        points = np.asarray(points)
        mins = np.min(points, axis=0).astype('int32') - self.margin
        maxs = np.max(points, axis=0).astype('int32') + self.margin
        xb = xb[mins[1]:maxs[1], mins[0]:maxs[0], :]
        xa = xa[mins[1]:maxs[1], mins[0]:maxs[0], :]
        y = y[mins[1]:maxs[1], mins[0]:maxs[0], :]

        y = (y[:,:,0]/y[:,:,0].max())*255
        y[y<=128] = 0
        y[y>128] = 1

        # if not os.path.exists(os.path.join('Data','HIFU_data','cropped', os.path.split(input_ID_b)[-1])):
        #     p1 = os.path.join('Data','HIFU_data','cropped', 'Before_'+os.path.split(input_ID_b)[-1])
        #     imsave(p1, xb)
        # if not os.path.exists(os.path.join('Data','HIFU_data','cropped', os.path.split(input_ID_a)[-1])):
        #     p2 = os.path.join('Data','HIFU_data','cropped', 'after_'+os.path.split(input_ID_a)[-1])
        #     imsave(p2, xa)
        #     p3 = os.path.join('Data','HIFU_data','cropped', 'Diff_'+os.path.split(input_ID_a)[-1])
        #     S = xa.astype('int32')-xb.astype('int32')
        #     S[S<50] = 0
        #     imsave(p3, S.astype('uint8'))
        # if not os.path.exists(os.path.join('Data','HIFU_data','cropped', os.path.split(target_ID)[-1])):
        #     p1 = os.path.join('Data','HIFU_data','cropped', 'mask_'+os.path.split(target_ID)[-1])
        #     imsave(p1, y*255)
        xb=xb+100
        xb[xb>255] = 255
        xa=xa+100
        xa[xa>255] = 255
        xb = self.transform_input(xb/255)
        xa = self.transform_input(xa/255)
        y = self.transform_target(y)

        if self.hflip:
            if random.uniform(0.0, 1.0) > 0.5:
                xb = TF.hflip(xb)
                xa = TF.hflip(xa)
                y = TF.hflip(y)

        if self.vflip:
            if random.uniform(0.0, 1.0) > 0.5:
                xb = TF.vflip(xb)
                xa = TF.vflip(xa)
                y = TF.vflip(y)

        if self.affine:
            angle = random.uniform(-180.0, 180.0)
            h_trans = random.uniform(-352 / 8, 352 / 8)
            v_trans = random.uniform(-352 / 8, 352 / 8)
            scale = random.uniform(0.5, 1.5)
            shear = random.uniform(-22.5, 22.5)
            xb = TF.affine(xb, angle, (h_trans, v_trans), scale, shear, fill=-1.0)
            xa = TF.affine(xa, angle, (h_trans, v_trans), scale, shear, fill=-1.0)
            y = TF.affine(y, angle, (h_trans, v_trans), scale, shear, fill=0.0)
        diff = xa-xb
        diff[diff<0] = 0
        x = torch.cat((xb, xa, diff), dim=0)
        return x.float(), y.float(), input_ID_a, target_ID, json_path

