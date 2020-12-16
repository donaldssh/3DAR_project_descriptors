import cv2
import numpy as np
import argparse
import os

def main(paths, out):
    
    descriptors = []

    # Create SURF object with Hessian Threshold=600, 128 values (extended) 
    surf = cv2.xfeatures2d.SURF_create(600)
    surf.setExtended(True)
    
    for path in paths:
        for entry in os.scandir(path):
            if (entry.path.endswith(".jpg") or entry.path.endswith(".png")) and entry.is_file():
                print(entry.path)
            
                img = cv2.imread(entry.path)

                # compute the keypoint and feature descriptors
                kp, des = surf.detectAndCompute(img, None)
                print(f"descriptors shape: {des.shape}")
                
                # keypoints.append(kp)
                descriptors.append(des)

    descriptors = np.concatenate(descriptors, axis=0)
    print(f"Total descriptors shape: {descriptors.shape}")
    
    descriptors.tofile(out)

"""
Providing in input the paths to n images directories it creates a unique dataset set 
in binary composed by the 128 values of every surf descriptors in every image.

Example:

train_set:
    python generate_dataset.py --paths ~/data/portelloDataset/ ~/data/castle-P19/ --out train_set.bin
    
test_set:
    python generate_dataset.py --paths ~/data/fountain-P11/ ~/data/tisoDataset/ --out test_set.bin
    
"""
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
            description="Create a csv dataset containing the surf descriptors of the given set of images")
    
    parser.add_argument("--paths", nargs='+', required=True, help="paths to the datasets")
    parser.add_argument("--out", help="name of output dataset")
    
    args = parser.parse_args()
    
    main(paths=args.paths, out=args.out)
