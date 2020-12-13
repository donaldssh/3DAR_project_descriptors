# Project B.2 
### Local feature compression using autoencoders - SfM tests

Design a compression strategy for local SURF descriptors using autoencoders. 

Training data can be generated using the images of dataset Portello <http://www.dei.unipd.it/~sim1mil/materiale/3Drecon/portelloData.zip> and Castle.

Testing must be done on dataset FountainP-11 and Tiso (available at <https://github.com/openMVG/SfM_quality_evaluation/tree/master/Benchmarking_Camera_Calibration_2008> and <http://www.dei.unipd.it/~sim1mil/materiale/3Drecon/>). 


### Testing on 3D reconstruction using SfM 

The reconstructed descriptors (only for the test set) are used to perform a SfM reconstruction using COLMAP <https://colmap.github.io/>, (using the two test dataset).
