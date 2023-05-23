# -*- coding: utf-8 -*-
"""
Created on Fri May 12 09:06:14 2023

@author: phili
"""

import os
import numpy as np
from PIL import Image
import pandas as pd

Training = "Grey_Basis"

image_width = 1280
image_height = 720
# Loop through all files in Training_folder folder
def image_matrix(Training_folder):
    image_arrays = []
    for filename in os.listdir(Training_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # Open the image file as a NumPy array
            image_path = os.path.join(Training_folder, filename)
            image_array = np.array(Image.open(image_path))

            # Flatten the array into a column vector
            column_vector = image_array.flatten('F').reshape(-1, 1)
            
            # Append the column vector to the list of image arrays
            image_arrays.append(column_vector)

    # Stack the column vectors together to form a matrix
    image_matrix = np.hstack(image_arrays)
    return image_matrix, image_arrays

def eigenfaces(image_arrays,image_matrix):
    mean = np.mean(image_arrays,axis=0)
    meansub = image_matrix-mean
    U,S,V=np.linalg.svd(meansub,full_matrices=0)
    return U

def eigensave(U,image_width,image_height):
    os.makedirs(f"Eigenfaces\{Training}",exist_ok=True)
    Path = os.path.join(os.getcwd(),f"Eigenfaces\{Training}")
    N=image_matrix.shape[1]
    for n in range(N):
        image_array = (U[:,n]).reshape((image_height, image_width))
        image_array=(255*(image_array - np.min(image_array))/np.ptp(image_array)).astype(np.int8)
        image = Image.fromarray(np.transpose(image_array)).convert("RGB")
        image.save(os.path.join(Path,f"{n}.jpg"))

Training_folder = os.path.join(os.getcwd(),"GreyPics",Training)

image_matrix, image_arrays = image_matrix(Training_folder)

U=eigenfaces(image_arrays, image_matrix)

eigensave(U,image_height,image_width)