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
import shutil


def cameraIntrinsicMat(data_params):
    Knew=[]
    Knew.append([float(data_params['fx']), float(data_params['skew']), float(data_params['cx'])])
    Knew.append([ 0, float(data_params['fy']), float(data_params['cy'])])
    Knew.append([float(data_params['p1']), float(data_params['p2']), 1])
    
    return np.asmatrix(Knew)

execute_in_docker = True
useOnly2DSeg = 1 # Set flag for 2D segmentation only --> set 0 if you are doing both 2D and 3D contour segmentation

# If you are doing task 1: set only useOnly3DSeg (2D and 3D contour) i.e. set useOnly2DSeg = 0
# if you are doing task 2: set both useOnly3DSeg and useReg to 1  i.e. set useOnly2DSeg = 0

useOnly3DSeg = 0 # set:1 for all three (2D seg, 3D seg and/or 2D-3D registration outputs)
useReg = 0 # set 1 if you are doing 2d-3d registeration --> for all three (2D seg, 3D seg and 2D-3D registration outputs)

class P2ILFChallenge(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            # Reading all input files 
            input_path = Path("/input/images/laparoscopic-image/") if execute_in_docker else Path("./test/images/laparoscopic-image/"),
            # input_path_1 = Path("/input/images/laparoscopic-image/") if execute_in_docker else [Path("./test/images/laparoscopic-image/"), Path("./test/transformed-3d-liver-model.obj"), Path("./test/acquisition-camera-metadata.json")],
            output_file = [Path("/output/2d-liver-contours.json"), Path("/output/3d-liver-contours.json"), Path("/output/transformed-3d-liver-model.obj")]if execute_in_docker else [Path("./output/2d-liver-contours.json"), Path("./output/3d-liver-contours.json"), Path("./output/transformed-3d-liver-model.obj")]
            # output_file = Path("/output/transformed-3d-liver-model.obj") if useOnlySeg else shutil.copyfile('./dummy/transformed-3d-liver-model.obj', "./output/transformed-3d-liver-model.obj")
                
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
        
        
    '''Instruction 1: YOU will need to work here for writing your results as per your task'''
    def save(self):
        if (useOnly2DSeg == 1):
            # logging first output file  /output/2d-liver-contours.json
            with open(str(self._output_file[0]), "w") as f:
                json.dump(self._case_results[0][0], f)  
                
            #create dummy files 
            shutil.copyfile('./dummy/3d-liver-contours.json', self._output_file[1])
            # shutil.copyfile('./dummy/transformed-3d-liver-model.obj', self._output_file[2])

            
        if useOnly3DSeg==1 or useReg ==1:
            # Hint you can append the results for 2D and 3D segmentation contour dictionaries
            print('\n 3D seg or registration flag on ---> Could not save the contours as your are returning only one result - try appending your results')
            for i in range(0,2):
                # print(len(self._case_results[i]))
                with open(str(self._output_file[i]), "w") as f:
                    json.dump(self._case_results[0][i], f)
        if useReg ==0:
            # This is because you need to write all files 
            shutil.copyfile('./dummy/transformed-3d-liver-model.obj', self._output_file[2])
        

        if useReg:
            print('\n writing our transformed-3d-liver-model.obj')    
                    
            #TODO: you can write the registration file to the idx - [2] instead of dummy file
            # Let us know if you will need help to figure this out
        
        print('\n All files written successfully')


    '''Instruction 2: YOU will need to work here for appending your 2D and 3D contour results'''
    def process_case(self, *, idx, case):
        results_append=[]
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)

        # Detect and score candidates
        results1 = self.predict(input_image=input_image)
        results_append.append(results1)
        # you can call your 3D predict function and log in your result
        # results2 = self.predict_3D(input_image=input_image) 
        
        
        

        

        
        if useOnly3DSeg:
            # Write resulting candidates to result.json for this case
            results2 = self.predict2()
            results_append.append(results2)
        
        return results_append
  
    
  # Sample provided - write your 3D contour prediction ehre
    def predict2(self):
        import meshio
        import json
       
        # Hard coded paths
        if execute_in_docker:
            input_path_mesh = Path('/input/3D-liver-model.obj')
            input_path_K = Path('/input/acquisition-camera-metadata.json')

            
        else:
            input_path_mesh = Path('./test/3D-liver-model.obj')
            input_path_K = Path('./test/acquisition-camera-metadata.json')
          
        # Reading 3D mesh    
        textured_mesh_input = meshio.read(input_path_mesh) # 3D mesh input
       
        # Reading camera parameters
        f = open(input_path_K) # camera parameters
        data_params = json.load(f)
        K = cameraIntrinsicMat(data_params)
        # Your code and return 3D contours or mesh to write
        
        
        # Replace data_params with Your 3D contour in dictionary format
        # my_dictionary = {"numOfContours": 0, "contour": [0.0 0.0] }
        
        
        # Return your 3D contour (remove data_params from here)
        return  data_params
        
        
    ''' Instruction 3: YOU will need to write similar functins for your 2D, 3D and registration - 
    these predict functions can be called by process_case for logging in results -> Tuple[SimpleITK.Image, Path]:'''
    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:
    # def predict(self):
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
        
