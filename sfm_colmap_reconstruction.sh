#!/bin/bash

# ./sfm_colmap_reconstruction.sh IMAGE_PATH DESCRIPTORS_PATH OUTPUT_DIR
# The ouput of the program will be saved in ~/colmap_reconstructions/OUTPUT_DIR
#
# ./sfm_colmap_reconstruction.sh IMAGE_PATH DESCRIPTORS_PATH OUPUT_BASE_PATH OUTPUT_DIR
# The ouput of the program will be saved in OUPUT_BASE_PATH/OUTPUT_DIR
#
# EXAMPLES FOR THE TEST DATASET:
#
# FOUNTAIN 
# ./sfm_colmap_reconstruction.sh ~/data/fountain-P11/images ~/data/original/sfm_data_fountain fountain
# ./sfm_colmap_reconstruction.sh ~/data/fountain-P11/images ~/data/compressed/sfm_data_fountain fountain_compressed
#
# TISO
# ./sfm_colmap_reconstruction.sh ~/data/tisoDataset ~/data/original/sfm_data_tiso tiso
# ./sfm_colmap_reconstruction.sh ~/data/tisoDataset ~/data/compressed/sfm_data_tiso tiso_compressed 

args=("$@")

DATASET_PATH=${args[0]}  
DESCRIPTORS_BASE_PATH=${args[1]} 

if [ $# = 3 ]; then
    OUTPUT_BASE_PATH=~/colmap_reconstructions
    mkdir -p $OUTPUT_BASE_PATH
    OUTPUT_PATH=$OUTPUT_BASE_PATH/${args[2]}
else
    OUTPUT_BASE_PATH=${args[2]}
    mkdir -p $OUTPUT_BASE_PATH
    OUTPUT_PATH=$OUTPUT_BASE_PATH/${args[3]}
fi 

mkdir -p $OUTPUT_PATH

DATABASE_PATH=$OUTPUT_PATH/database.db

colmap database_creator \
        --database_path $DATABASE_PATH

colmap feature_importer \
        --database_path $DATABASE_PATH \
        --image_path $DATASET_PATH/ \
        --import_path $DESCRIPTORS_BASE_PATH/desc \

colmap matches_importer \
        --database_path $DATABASE_PATH \
        --match_type raw \
        --match_list_path $DESCRIPTORS_BASE_PATH/mmm_file.txt
        
colmap mapper \
        --image_path $DATASET_PATH \
        --database_path $DATABASE_PATH \
        --output_path $OUTPUT_PATH
        
echo Sfm dense reconstruction generated at $OUTPUT_PATH
