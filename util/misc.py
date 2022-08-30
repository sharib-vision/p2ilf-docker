#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 20:45:48 2022

@author: sharib
"""

import torch
from model import UNet
import os
import argparse
import numpy as np
import torch.nn as nn
import cv2


def loadMHAfile(image_data):
    img = np.zeros((image_data.shape[1], image_data.shape[2], 3), dtype=np.float32)
    img[:,:,0] = image_data[0,:,:]
    img[:,:,1] = image_data[1,:,:]
    img[:,:,2] = image_data[2,:,:]
    ImgOrig = cv2.rotate(img.astype('uint8'), cv2.ROTATE_90_CLOCKWISE)
    ImgOrig = np.fliplr(ImgOrig).astype('uint8')
    
    return ImgOrig

def find_rgb(img, r_query, g_query, b_query):
    coordinates= []
    for x in range(0,img.shape[0]-1):
        for y in range(0,img.shape[1]-1):
            r, g, b = img[x,y]
            if r == r_query and g == g_query and b == b_query:
                # print("{},{} contains {}-{}-{} ".format(x, y, r, g, b))
                coordinates.append((x, y))
    return(coordinates)

def getColors():
    """
    List of RGB colors for representing liver classes
    """
    # For MICCAI challenge:
    colors = [
    torch.tensor([0,0,0],dtype=torch.uint8),
    torch.tensor([255,0,0],dtype=torch.uint8),
    torch.tensor([0,0,255],dtype=torch.uint8),
    torch.tensor([255,255,0],dtype=torch.uint8),
    ]
    return colors

def convertFromOneHot(T_one_hot):
    """ T_one_hot [b,c,h,w] with values in {0,1}--> T[b,h,w] with values in {0,...,c-1}"""
    if T_one_hot.dim() == 4:
        return torch.argmax(T_one_hot,dim=1)
    elif T_one_hot.dim() == 3:
        return torch.argmax(T_one_hot,dim=0)
    else:
        return torch.argmax(T_one_hot, dim=0)
    
def createVisibleLabel(label):
    """
    Goal: match each class index with 'real' colors

    Label: [H,W] with values in {0, ..., n_class-1}
    return [H,W,C] image tensor uint8
    """

    if label.dim() == 3:
        # case contours is one_hot:
        label = convertFromOneHot(label)

    M = int(torch.max(label.view(-1)))
    image = torch.zeros((label.shape[0], label.shape[1],3), dtype=torch.uint8)
    if M == 0:
        return image.byte()
    colors = getColors()
    for i in range(M+1):
        mask = label == i

        # in some case: label has empty class
        if mask.sum() != 0:
            if image[mask].numel() != 0:
                image[mask] = colors[i]

    return image.byte()

def convertContoursToImage(label, tensor=True):
    """
    Label: [H,W] with values in {0, ..., n_class-1}
    return [H,W,C] image
    """
    image = createVisibleLabel(label)
    return image.numpy() 
