import cv2
import numpy as np
import argparse
import os

def main(path, out):
    
    descriptors = []
    #keypoints = []
    
    for entry in os.scandir(path):
        if (entry.path.endswith(".jpg") or entry.path.endswith(".png")) and entry.is_file():
            print(entry.path)
        
            img = cv2.imread(entry.path)

            #Create SURF object with Hessian Threshold = 400
            surf = cv2.xfeatures2d.SURF_create(400)

            #compute the keypoint and feature descriptors
            kp, des = surf.detectAndCompute(img, None)
            print(f"descriptors shape: {des.shape}")
            
            #keypoints.append(kp)
            descriptors.append(des)

    descriptors = np.concatenate(descriptors, axis=0)
    print(f"Total descriptors shape: {descriptors.shape}")
    
    #np.savetxt(out+".cv", descriptors, delimiter=",")
    descriptors.tofile(out)

# running example:
# python generate_dataset.py --path ~/data/portelloDataset/ --out descriptors_portello_dataset
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
            description="Create a csv dataset containing the surf descriptors of the given set of images")
    
    parser.add_argument("--path", required=True, help="path to the dataset")
    parser.add_argument("--out", help="name of output dataset")
    
    args = parser.parse_args()
    
    main(path=args.path, out=args.out)
