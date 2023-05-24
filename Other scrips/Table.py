# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 20:58:36 2023

@author: Phili
"""
import os
import numpy as np
from PIL import Image
import pandas as pd

Training = "Grey_Basis"
Test = "Grey_Test"
Names=['Anna','Lars','Mads','Philip','Signe C','Signe H']

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

def dist(Baseproj,path):
    X=Image.open(path)
    Xa=(np.array(X)).flatten('F').reshape(-1, 1)
    Xb=np.hstack(np.matmul(np.transpose(U),Xa))
    E=[]
    for n in range(len(Baseproj)):
        E.append(np.linalg.norm(Baseproj[:,n]-Xb))
    return E

def Recognition_Table(Test_folder, Names):
    Table = np.array(len(Names)*[np.zeros(len(Names),dtype=int)])
    i=0
    k=1
    for filename in os.listdir(Test_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # Open the image file as a NumPy array
            image_path = os.path.join(Test_folder, filename)
            if i% (len(os.listdir(Test_folder))/6)==0:
                k+=1
                if i==0:
                    k-=1  
            if i==-1:
                i+=1
            i+=1
            c = dist(Baseproj,image_path).index(min(dist(Baseproj,image_path)))
            if c<=4:
                Table[0,k-1]+=1
            elif c<=9:
                Table[1,k-1]+=1
            elif c<=14:
                Table[2,k-1]+=1
            elif c<=19:
                Table[3,k-1]+=1
            elif c<=24:
                Table[4,k-1]+=1
            else:
                Table[5,k-1]+=1
    return pd.DataFrame(Table,index=Names, columns=Names)

Training_folder = os.path.join(os.getcwd(),"GreyPics",Training)
Test_folder = os.path.join(os.getcwd(),"GreyPics",Test)

image_matrix, image_arrays = image_matrix(Training_folder)

N=image_matrix.shape[1]

U=eigenfaces(image_arrays, image_matrix)

Baseproj=baseproj(U,image_matrix,N)

Table=Recognition_Table(Test_folder, Names)      

print((Table.style.format_index(lambda v: f"\\hline {v}")).to_latex(column_format="|c|c|c|c|c|c|c|"))