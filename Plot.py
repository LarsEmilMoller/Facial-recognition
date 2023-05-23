# -*- coding: utf-8 -*-
"""
Created on Fri May 12 09:21:39 2023

@author: phili
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

Training = "Grey_Basis"
Test = "Grey_Test"
Names=['Anna','Lars','Mads','Philip','Signe C','Signe H']
pc1 = 1
pc2 = 3

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

def baseproj(U,matrix, N):
    L=[]
    for n in range(N):
        Ub=(np.matmul(np.transpose(U),matrix[:,n])).reshape(-1, 1)
        L.append(Ub)
    proj=np.hstack(L)
    return proj

def prinplot(A,P1,P2):
    # tager input matrix A og 2 principal components
    for n in range(6):
        plt.scatter(A[P1][n*5-5:n*5], A[P2][n*5-5:n*5], label=Names[n])
    #for loop skifter bare farve
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
    plt.show()
    
Training_folder = os.path.join(os.getcwd(),"GreyPics",Training)
Test_folder = os.path.join(os.getcwd(),"GreyPics",Test)

image_matrix, image_arrays = image_matrix(Training_folder)

U=eigenfaces(image_arrays, image_matrix)

N=image_matrix.shape[1]

Baseproj=baseproj(U,image_matrix,N)

prinplot(Baseproj,pc1-1,pc2-1,)