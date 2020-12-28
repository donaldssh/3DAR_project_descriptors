import cv2
import numpy as np
import argparse
import os
from  itertools import combinations
from tqdm import tqdm
from utils import *

"""
Format of the output.txt file:

image_name_k number_of_keypoints_image_k
kp_x kp_y kp_scale kp_orientation encoded_surf descriptors
....
image_name_{k+1} number_of_keypoints_image_{K+1}
kp_x kp_y kp_scale kp_orientation encoded_surf descriptors
"""

def main(path, out):
    
    # Create SURF object with Hessian Threshold=600, 128 values (extended) 
    surf = cv2.xfeatures2d.SURF_create(600)
    surf.setExtended(True)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # load the best model for the encoder
    encoder = Encoder(encoded_space_dim=16, conv1_ch=32, conv2_ch=64, conv3_ch=64, fc_ch=64)
    encoder.load_state_dict(torch.load('best_encoder16.torch'))
    encoder.to(device)
    encoder.eval()
    
    f = open(out, "w")
    
    for entry in tqdm(os.scandir(path)):
        if (entry.path.endswith(".jpg") or entry.path.endswith(".png")) and entry.is_file():
            
            image_name = entry.path.replace(path+"/", "")
        
            img = cv2.imread(entry.path)

            kps, des = surf.detectAndCompute(img, None)
            print(f"descriptors shape: {des.shape}")
 
            composed_transform = transforms.Compose([Surf3DReshape(), NpToTensor()])
            surf3d = SurfDataset(pd.DataFrame(des), transform=composed_transform)
            surf3d_dataloader = DataLoader(surf3d, batch_size=len(des), shuffle=True)
            
            with torch.no_grad():
                des_enc = encoder(next(iter(surf3d_dataloader)).to(device)).cpu().numpy()

            f.write(image_name+" "+str(des.shape[0])+"\n")
            for i in range(len(kps)):
                coords = [kps[i].pt[0], kps[i].pt[1], kps[i].size, kps[i].angle]
                f.write(" ".join(str(i) for i in coords))
                f.write(" "+" ".join(str(i) for i in des_enc[i]))
                f.write("\n")
        
    f.close()
    
"""   
Example:
python generate_sfm_data_encoder.py --path ~/data/fountain-P11/images --out foutain_encoded_descr.txt
"""
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Send the encoded descriptors")
    
    parser.add_argument("--path", required=True, help="path to the dataset")
    parser.add_argument("--out", required=True, help="name of output file")
    
    args = parser.parse_args()
    
    main(path=args.path, out=args.out)
