#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 20:31:49 2022

@author: sharib
"""

import SimpleITK
from pathlib import Path

from pandas import DataFrame
import torch
import torchvision

import  skimage.transform 
from evalutils import SegmentationAlgorithm

from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
import evalutils
import numpy as np
import json
from model import UNet
from util.misc import convertContoursToImage, find_rgb
from util.data import ContoursDetectionTransform
import cv2

execute_in_docker = True

class P2ILFChallenge(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path = Path("/input/images/laparoscopic-image/") if execute_in_docker else Path("./test/images/laparoscopic-image/"),
            output_file = Path("/output/2d-liver-contours.json") if execute_in_docker else Path("./output/2d-liver-contours.json")
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if execute_in_docker:
            path_model = "/opt/algorithm/ckpt/CP50_criteria.pth"
        else:
            path_model = "./ckpt/CP50_criteria.pth"
            
        self.num_classes = 4
        self.model = UNet(n_channels=3, n_classes=self.num_classes , upsample=False)
        # model_path = torch.load(opts.model)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(path_model))
        print("XXXXX   model weight loaded")
        

    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f)

    def process_case(self, *, idx, case):
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)

        # Detect and score candidates
        results = self.predict(input_image=input_image)

        # Write resulting candidates to result.json for this case
        return results
    
    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:
        from util.misc_findContours import findCountures
        
        image = SimpleITK.GetArrayFromImage(input_image)
        image = np.array(image)
        shape = image.shape
        # Resize/normalise what you want to do according to your method
        I = skimage.transform.resize(image,(270,480,3))
        I = ContoursDetectionTransform(3)({'image':I,'label':np.zeros_like(I)})
        I['image'] = I['image'].unsqueeze(0)    
        Image_resized = I['image'].to(self.device)
        # predict:
        activation = 'None'
        with torch.no_grad():
            output = self.model(Image_resized)
            
        label_image = cv2.resize(convertContoursToImage(output.squeeze()), (image.shape[1], image.shape[0]))
        # Array to build the contour dictionary:
        contoursArray = []
        contourCounter = 0;
        
        # Extract biggest components from each label:
        # Extract ridge masks:
        imageRidge = np.zeros(shape=(label_image.shape[0],label_image.shape[1],3),dtype=np.uint8)
        coordsRidge = find_rgb(label_image, 255,0,0)
        for c in coordsRidge:
            imageRidge[c] = label_image[c]


        filteredImage = np.zeros(shape=(label_image.shape[0],label_image.shape[1],3),dtype=np.uint8)

        # imageSilhouetteGray = cv2.cvtColor(imageSilhouette, cv2.COLOR_BGR2GRAY)
        cType = 'Ridge'
        contoursArray, contourCounter, filteredImage = findCountures([255,0,0], cType, contoursArray, contourCounter, imageRidge , coordsRidge, label_image, filteredImage)

        # Extract ligament masks:
        imageLigament = np.zeros(shape=(label_image.shape[0],label_image.shape[1],3),dtype=np.uint8)
        coordsLigament = find_rgb(label_image,0,0,255)
        # from misc_findContours import findCountures
        cType = 'Ligament'
        contoursArray, contourCounter, filteredImage = findCountures([0,0,255], cType, contoursArray, contourCounter, imageLigament , coordsLigament, label_image, filteredImage)
        # Extract silhouette masks:
        imageSilhouette = np.zeros(shape=(label_image.shape[0],label_image.shape[1],3),dtype=np.uint8)
        coordsSilhouette = find_rgb(label_image,255,255,0)
        # from misc_findContours import findCountures
        cType = 'Silhouette'
        contoursArray, contourCounter, filteredImage = findCountures([255,255,0], cType, contoursArray, contourCounter, imageSilhouette , coordsSilhouette, label_image, filteredImage)

        
        """
        3: Save your Output : /output/2d-liver-contours.json
        """
        # Step 3) Writing you json file to  ===> /output/2d-liver-contours.json
        my_dictionary = {"numOfContours": int(contourCounter), "contour": contoursArray }
        
        # check
        # import matplotlib.pyplot as plt
        # plt.imshow(filteredImage)
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plt.show()
        return my_dictionary
        
        
if __name__ == "__main__":
    P2ILFChallenge().process()        
        
