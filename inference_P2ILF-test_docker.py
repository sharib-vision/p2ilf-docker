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

"""
These are important settings:
Case I: Task 1 only (segmentation of 2D only): Set useOnly2DSeg = 1 and others 0
Case II: Task 1 with both 2D and 3D segmentation: Set use3DSeg = 1 and others 0
Case III: Both Task 1 and Task 2 both: Set useReg = 1 and others 0

An examples with useOnly2DSeg = 1 (2D contour segmentation is provided)

Note: All three outputs need to be written in each run. 
/output/2d-liver-contours.json and  /output/3d-liver-contours.json and  /output/transformed-3d-liver-model.obj

To adapt that: we have provided you with dummy /output/3d-liver-contours.json and transformed-3d-liver-model.obj in case you want to do only Case I.
"""

execute_in_docker = True
useOnly2DSeg = 1 # Set flag for 2D segmentation only
use3DSeg = 0 # state:1 for all three (2D seg, 3D seg and 2D-3D registration outputs)
useReg = 0 #for all three (2D seg, 3D seg and 2D-3D registration outputs)

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
            output_file = [Path("/output/2d-liver-contours.json"), Path("/output/3d-liver-contours.json"), Path("/output/transformed-3d-liver-model.obj")]if execute_in_docker else [Path("./output/2d-liver-contours.json"), Path("./output/3d-liver-contours.json"), Path("./output/transformed-3d-liver-model.obj")]     
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #--->  1) Load your model/models - you can use different names if you have more than one model
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
        

     #--->  2) YOU will need to work here for writing your results as per your task   
    def save(self):
        if (useOnly2DSeg == 1):
            # logging first output file  /output/2d-liver-contours.json
            with open(str(self._output_file[0]), "w") as f:
                json.dump(self._case_results[0][0], f)  
                
            #create dummy files 
            shutil.copyfile('./dummy/3d-liver-contours.json', self._output_file[1])
            shutil.copyfile('./dummy/transformed-3d-liver-model.obj', self._output_file[2])
            print('\n All files written successfully')

        # Allows you to write both 2D and 3D landmark contours
        if use3DSeg==1 or useReg ==1:
            print('\n 3D seg or registration flag on ---> Could not save the contours as your are returning only one result - try appending your results')
            for i in range(0,2):
                # print(len(self._case_results[i]))
                with open(str(self._output_file[i]), "w") as f:
                    json.dump(self._case_results[0][i], f)
       
        # Hint you can append the results for 2D and 3D segmentation contour dictionaries

        if useReg ==1:   
            print('\n writing 3D registered mesh --> You need to put the library needed in dockerfile for writing your mesh')  

            #TODO: you can write the registration file to the idx - [2] instead of dummy file
            # Let us know if you will need help to figure this out
        
        
    #--->  3)  YOU will need to work here for appending your 2D and 3D contour results 
    def process_case(self, *, idx, case):
        results_append=[]
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)

        # Detect and score candidates
        results1 = self.predict(input_image=input_image)
        results_append.append(results1)

        # e.g., you can call your 3D predict function and log in your result
        # results2 = self.predict_3D(input_image=input_image) 
        if use3DSeg==1 or useReg ==1:
            # Write resulting candidates to result.json for this case
            results2 = self.predict2() # That is the function that runs for my 3D segmentation model
            results_append.append(results2)

        # Similarly you can do for your transformed mesh model and append it 
        




        return results_append
  
  #--->  4): YOU will need to write similar functins for your 2D, 3D and registration - 
    # these predict functions can be called by process_case for logging in results -> Tuple[SimpleITK.Image, Path]
    
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
        
        return my_dictionary  



  #--->  5) Your code for 3D contour prediction here ----> contact us if you will need help! You can also use this function to do your registration and return both 
  #  3d-liver-contours.json and  transformed-3d-liver-model.obj

    def predict2(self):
        import meshio
        import json
       
        # Hard coded paths -- to the inputs - this will be always true so do not worry about this
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
        


        
        # Replace the return with your 3D contour (remove data_params from here)
        return  data_params
        
        
    
if __name__ == "__main__":
    P2ILFChallenge().process()        
        
