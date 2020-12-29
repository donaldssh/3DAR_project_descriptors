import cv2
import numpy as np
import argparse
import os

def main(paths, out):
    
    descriptors = []

    surf = cv2.xfeatures2d.SURF_create()
    #surf.setExtended(True)
    
    for path in paths:
        for entry in os.scandir(path):
            if (entry.path.endswith(".jpg") or entry.path.endswith(".png")) and entry.is_file():
                print(entry.path)
            
                img = cv2.imread(entry.path)

                # compute the keypoint and feature descriptors
                kp, des = surf.detectAndCompute(img, None)
                print(f"descriptors shape: {des.shape}")
                
                descriptors.append(des)

    descriptors = np.concatenate(descriptors, axis=0)
    print(f"Total descriptors shape: {descriptors.shape}")
    
    descriptors.tofile(out)

"""
Providing in input the paths to n images directories it creates a unique dataset set 
in binary composed by the 128 values of every surf descriptors in every image.

Example:

train_set:
    python generate_dataset_autoencoder.py --paths ~/data/portelloDataset/ ~/data/castle-P30/images  --out train_set_64.bin
    
    Total descriptors shape: (723252, 64)
    
test_set:
    python generate_dataset_autoencoder.py --paths ~/data/fountain-P11/images/ ~/data/tisoDataset --out test_set_64.bin

    Total descriptors shape: (1174700, 64)
    
"""
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
            description="Create a bin dataset containing the surf descriptors of the given set of images")
    
    parser.add_argument("--paths", nargs='+', required=True, help="paths to the datasets")
    parser.add_argument("--out", help="name of output dataset")
    
    args = parser.parse_args()
    
    main(paths=args.paths, out=args.out)
