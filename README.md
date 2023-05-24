# Facial-recognition
The folders are distributed such that every training set has its own folder. The different training sets are named:

• Grey_control_set 

• Grey_control_1_each

• Grey_control_2_each

• Grey_control_3_each

• Grey_control_4_each

• Grey_facialexpressions_2_set

• Grey_facialexpressions_3_set

• Grey_facialexpressions_4_set

• Grey_facialexpressions_5_set

• Grey_distance_set

• Grey_rotation_set

• Grey_position_set

• Grey_control_Halfcropped

• Grey_control_Fullcropped

The folders containing the images of the test set of three different sizes are named:

• Grey_Test_set

• Grey_Test_Halfcropped

• Grey_Test_Fullcropped

In the directory **Facial-recognition**, there are four python scripts. The python files are the following: 

### Individual_results.py
This script is the facial recognition algorithm. To use the algorithm specify which folder should be the test set and training set.
The script will print which indexes would be the right guesses for a specific person, and afterwards the guesses of the 10 test images corresponding to that person. This is done for each of the 6 subjects, and then followed by the success rate of the experiment. 


### Eigenface.py 
This script creates images of the eigenfaces/principal components from a training set. The Images will be saved in a folder named Eigenfaces_{Training_set} in the folder in which Eigenface.py is located in. 

### Plot.py
This script plots the coordinate vectors for a training set with two arbitray principal components on the x- and y axis. The training set have to be specified along with which principal componens should be on each axis.

### Table.py
This script prints confusion matricies with latex syntax for a training set and test set.
