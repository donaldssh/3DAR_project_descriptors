import cv2
import numpy as np
import argparse

def main(image_path):
    
    img = cv2.imread(image_path)

    # Create SURF object with Hessian Threshold = 400
    surf = cv2.xfeatures2d.SURF_create(400)

    # compute the keypoint and feature descriptors
    kp, des = surf.detectAndCompute(img, None)

    # display the results
    img2 = cv2.drawKeypoints(img, kp, None, (255,0,0), 4)
    cv2.imshow("image", img2)
    
    # save the descriptors in a csv file
    np.savetxt("surf_descriptors.csv", des, delimiter=",")

    # close when a key is pressed
    cv2.waitKey(0)


# running example:
# python surf.py --image lena.png
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
            description="Save to csv the surf descriptors of the given image"
            )
    
    parser.add_argument("--image", required=True, help="path to the image")
    args = parser.parse_args()
    
    main(image_path=args.image)
