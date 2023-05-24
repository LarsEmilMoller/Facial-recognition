# Facial-recognition
To use the facial recognition algorithm open Individual_results.py. Then specify which folder should be the test set and training set.  The folders are distributed such that every training set has its own folder. The different training sets are named:

• Grey_control_set 

• Grey_control_1_each

• Grey_control_2_each

• Grey_control_3_each

• Grey_control_4_each

• Grey_facialexpressions_set

• Grey_distance_set

• Grey_orientation_set

• Grey_position_set

• Grey_control_Halfcropped

• Grey_control_Fullcropped

The folders containing the images of the test set of three different sizes are named:

• Grey_Test_set

• Grey_Test_Halfcropped

• Grey_Test_Fullcropped

The script will print which indexes would be he right guesses for a specefic person and then the 10 guesses of the training images corresponding to that person. This is done for each of the 6 subject, and then followed by the success rate of the experiment. 


In the directory Facial-recognintion, there are three additional python scrips. These can be used to further explore the algorithm. The python files are the following: 

Eigenface.py: This script creates images of the eigenfaces/principal components from a training set. The Images will be saved in the folder called Eigenfaces_{Training_set} in this folder Eigenface.py is located. 

Plot.py: This script can plot the coordinate vectors for a training set with two arbitray principal components on the x- and y axis. The training set have to be specified along with which principal componens sould be on each axis.

Table.py: This script can plot confusion matricies with latex syntax for a training set and test set.
