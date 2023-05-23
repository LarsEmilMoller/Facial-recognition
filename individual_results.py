# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 20:58:36 2023

@author: Phili
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#print(os.path.abspath(os.getcwd()))
Training = "Grey_Basis"
Test = "Grey_Test"
Names=['Anna','Lars','Mads','Philip','Signe C','Signe H']

Training_folder = os.path.join(os.getcwd(),"GreyPics",Training)
Test_folder = os.path.join(os.getcwd(),"GreyPics",Test)

def image_matrix(Training_folder):
    """
    Takes the training set as input, and reshapes the images into columns, 
    adding these to the list image_arrays. This list is horsontally stacked
    to the image matrix, returning the matrix and the list. 

    Parameters
    ----------
    Training_folder : The folder containing the images for the training set.

    Returns
    -------
    image_matrix : An array containing all the vectors in image_array.
    image_arrays : A list of each image in the training set represented as a vector.

    """
    image_arrays = []
    for filename in os.listdir(Training_folder): 
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            image_path = os.path.join(Training_folder, filename)
            image_array = np.array(Image.open(image_path))

            column_vector = image_array.flatten('F').reshape(-1, 1)
            
            image_arrays.append(column_vector)

    image_matrix = np.hstack(image_arrays)
    return image_matrix, image_arrays

def eigenfaces(image_arrays,image_matrix):
    """
    Takes the list of images reprecented as vectors, and calculates the mean of the list.
    Then subtracting the mean from the image matrix. Then calculating the SVD of the
    mean subtracted matrix. Returns the U matrix of the SVD.

    Parameters
    ----------
    image_arrays : A list of each image in the training set reprecented as a vector.
    image_matrix : An array containing all the vectors in image_array.

    Returns
    -------
    U : The matrix U containing the principal components of image_matrix.

    """
    mean = np.mean(image_arrays,axis=0)
    meansub = image_matrix-mean
    U,S,V=np.linalg.svd(meansub,full_matrices=0)
    return U

def baseproj(U,matrix, N):
    """
    Takes the columns of the matrix and projects them onto the subspace spanned by 
    the columns of matrix U. These projections are converted to a matrix and returned.

    Parameters
    ----------
    U : The matrix U containing the principal components of image_matrix.
    matrix : An array with indexes each representing an image of the faces
                wished to be recognized.
    N : The number of images in the matrix.

    Returns
    -------
    proj : A matrix consisting of the images represented by the columns of U.

    """
    L=[]
    for n in range(N):
        Ub=(np.matmul(np.transpose(U),matrix[:,n])).reshape(-1, 1)
        L.append(Ub)
    proj=np.hstack(L)
    return proj

def dist(U,Baseproj,path):
    """
    Opens the test image, and then makes a projection onto the span
    of principal componets of matrix U. Then calculating the Euclidean distance
    from the test image to all the images in the Baseproj. 
    Parameters
    ----------
    U : The matrix U containing the principal components of image_matrix.
    Baseproj : A matrix consisting of the images of the faces wished to be recognized as 
                represented by the columns of U.
    path : The path to the test image.

    Returns
    -------
    dist : A list containing the Euclidian distance between the test image and Baseproj.

    """
    X=Image.open(path)
    Xa=(np.array(X)).flatten('F').reshape(-1, 1)
    Xb=np.hstack(np.matmul(np.transpose(U),Xa))
    dist=[]
    for n in range(len(Baseproj)):
        dist.append(np.linalg.norm(Baseproj[:,n]-Xb))
    return dist

image_matrix, image_arrays = image_matrix(Training_folder)
U = eigenfaces(image_arrays,image_matrix)
N=image_matrix.shape[1]

Baseproj=baseproj(U,image_matrix,N)

def who_is_it(baseproj):
    """
    Takes each image in the test folder, finds the shortest distance by the function
    dist(). The index corresponding to the shortest distance is printed. The indexes 
    corresponding to each subsject is printed. Lastly the success rate is printed. 

    Parameters
    ----------
    baseproj : A matrix consisting of the images represented by the span of principal components.

    Returns
    -------
    None.

    """
    i=0
    k=1
    Z=int(image_matrix.shape[1]/6)
    Rate=0
    for filename in os.listdir(Test_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # Open the image file as a NumPy array
            image_path = os.path.join(Test_folder, filename)
            if i% (len(os.listdir(Test_folder))/6)==0:
                k+=1
                if i==0:
                    k-=1  
                print(f'Person {Names[k-1]} is index {k*Z-Z} to {k*Z-1}:')
            if i==-1:
                i+=1
            i+=1
            if k*Z-Z <= dist(U,Baseproj,image_path).index(min(dist(U,Baseproj,image_path))) <= k*Z-1:
                Rate+=1
            #print(image_path)
            #print(f'Intervallet er {k*Z-Z} til {k*Z-1}')
    
            print(dist(U,Baseproj,image_path).index(min(dist(U,Baseproj,image_path))))
    print(f'Success rate: {(Rate/i)*100}')

who_is_it(Baseproj)
