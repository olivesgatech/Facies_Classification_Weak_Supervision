# Adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py

import math
import numbers
import random
import numpy as np

from PIL import Image, ImageOps, ImageChops

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask, conf):
        img, mask, conf = Image.fromarray(img, mode=None), Image.fromarray(mask, mode='L'), conf#Image.fromarray(conf, mode=None)
        assert img.size == mask.size

        for a in self.augmentations:
            img, mask, conf = a(img, mask, conf)
        return np.array(img), np.array(mask, dtype=np.uint8), np.array(conf)

class AddNoise(object):
    def __call__(self, img, mask, conf):
        noise = np.random.normal(loc=0,scale=0.02,size=(img.size[1], img.size[0]))
        return img + noise, mask, conf

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, conf):
        if random.random() < 0.5:
            for i in range(conf.shape[-1]):
                conf1 = conf[:, :, i].squeeze()
                Image.fromarray(conf1, mode='F').transpose(Image.FLIP_TOP_BOTTOM)
                conf[:, :, i] = conf1
            return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM), conf
        return img, mask, conf
    
class RandomVerticallyFlip(object):
    def __call__(self, img, mask, conf):
        if random.random() < 0.5:
            for i in range(conf.shape[-1]):
                conf1 = conf[:, :, i].squeeze()
                Image.fromarray(conf1, mode='F').transpose(Image.FLIP_LEFT_RIGHT)
                conf[:, :, i] = conf1
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), conf
        return img, mask, conf
   


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask, conf):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)
        for i in range(conf.shape[-1]):
            conf1 = conf[:,:,i].squeeze()
            conf1 = Image.fromarray(conf1).rotate(rotate_degree, Image.BILINEAR)
            conf[:,:,i]=np.array(conf1)
        return img, mask, conf

    # def __call__(self, img, mask, conf):
    #     '''
    #     PIL automatically adds zeros to the borders of images that rotated. To fix this 
    #     issue, the code in the botton sets anywhere in the labels (mask) that is zero to 
    #     255 (the value used for ignore_index).
    #     '''
    #     rotate_degree = random.random() * 2 * self.degree - self.degree

    #     img = img.rotate(rotate_degree, Image.BILINEAR)
    #     mask =  mask.rotate(rotate_degree, Image.NEAREST)

    #     for i in range(conf.shape[-1]):
    #         conf1=conf[:,:,i].squeeze()
    #         Image.fromarray(conf1, mode='F').rotate(rotate_degree, Image.BILINEAR)
    #         conf[:,:,i]=conf1

    #     binary_mask = Image.fromarray(np.ones([mask.size[1], mask.size[0]]))
    #     binary_mask = binary_mask.rotate(rotate_degree, Image.NEAREST)
    #     binary_mask = np.array(binary_mask)

    #     mask_arr = np.array(mask)
    #     mask_arr[binary_mask==0] = 255
    #     mask = Image.fromarray(mask_arr)

    #     return img, mask, conf
