import cv2
import numpy as np
import argparse
import os
from  itertools import combinations
from tqdm import tqdm

"""
###################################################################################################

For every image create a text file i.e. "01.jpg" -> "01.jpg.txt" with the following format
given N descriptors:


N 128 (number of features of the descriptor)
u_keypoint v_keypoint scale orientation 0 0 0 .... 0 (128 zeros)
u_keypoint v_keypoint scale orientation 0 0 0 .... 0 (128 zeros)
.
.
.
u_keypoint v_keypoint scale orientation 0 0 0 .... 0 (128 zeros)


I set all zeros because I dont want that colmap does the matching from this.
The matches will be provided to colmap by the following file.

###################################################################################################

The format of mmm_file.txt is the following:
0001.jpg 0002.jpg
i j                         (where i and j are the index of the 2 keypoints that matches)
k l
m n
.
.
.
p q
  
0002.jpg 0003.jpg           (next couple of images)
i j
k l
.
.
.

it will be done the combination of each image with all the others
###################################################################################################
"""

def main(path, out_dir):
    
    descriptors_dir = "/desc"
    os.makedirs(out_dir+descriptors_dir, exist_ok=True)

    image_names = []
    descriptors = []
    keypoints = []
    
    # Create SURF object with Hessian Threshold=300, 128 values (extended) 
    surf = cv2.xfeatures2d.SURF_create(300)
    surf.setExtended(True)
    
    for entry in os.scandir(path):
        if (entry.path.endswith(".jpg") or entry.path.endswith(".png")) and entry.is_file():
            
            image_name = entry.path.replace(path+"/", "")
            
            image_names.append(image_name)
        
            img = cv2.imread(entry.path)

            kps, des = surf.detectAndCompute(img, None)
            print(f"descriptors shape: {des.shape}")
            
            keypoints.append(kps)
            descriptors.append(des)

            f = open(out_dir+descriptors_dir+"/"+image_name+".txt", "w")
            f.write(str(len(kps))+" 128\n")
            for kp in kps:
                coords = [kp.pt[0], kp.pt[1], kp.size, kp.angle]
                f.write(" ".join(str(i) for i in coords))
                f.write(" "+" ".join(str(0) for i in range(128)))
                f.write("\n")
            f.close()
    
    
    # find matches between each combinations of 2 images in the dataset

    f = open(out_dir+"/mmm_file.txt", "w")
    for ids in tqdm(combinations(range(len(image_names)), 2)):
        i = ids[0]
        j = ids[1]
        f.write(image_names[i]+" "+image_names[j]+"\n")
        
        #feature matching
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

        matches = bf.match(descriptors[i], descriptors[j])
        
        f.write("\n".join(str(match.queryIdx)+" "+str(match.trainIdx) for match in matches))
        f.write("\n\n")

    f.close()

# running example:
# python generate_sfm_data.py --path ~/data/fountain-P11/images --out sfm_data
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
            description="Create the descriptors and matches file for sfm reconstruction with colmap")
    
    parser.add_argument("--path", required=True, help="path to the dataset")
    parser.add_argument("--out", help="path of the output folder")
    
    args = parser.parse_args()
    
    main(path=args.path, out_dir=args.out)
