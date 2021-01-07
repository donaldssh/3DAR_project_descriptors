import cv2
import numpy as np
import argparse
import os
from  itertools import combinations
from tqdm import tqdm
from utils import *
import pandas as pd

"""
generate colmap SFM data from encoded surf descriptors:

output_dir/
├─ desc/
│  ├─ image_name_0.txt
│  ..
│  ├─ image_name_k.txt
├─ mmm_file.txt


image_name_i.txt contains the keypoints with the following format:
N 128 (number of features of the descriptor)
u_keypoint v_keypoint scale orientation 0 0 0 .... 0 (128 zeros)
u_keypoint v_keypoint scale orientation 0 0 0 .... 0 (128 zeros)
.
.
u_keypoint v_keypoint scale orientation 0 0 0 .... 0 (128 zeros)

I set all zeros because I dont want that colmap does the matching from this.
The matches will be provided to colmap by mmm_file.txt


mmm_file.txt contains the matching for each combination of images with the following format:

0001.jpg 0002.jpg
i j                         (where i and j are the index of the 2 keypoints that matches)
k l
m n
.
.
p q
  
0002.jpg 0003.jpg           (next couple of images)
i j
k l
.
.
"""

def main(enc, out_dir):
    
    descriptors_dir = "/desc"
    os.makedirs(out_dir+descriptors_dir, exist_ok=True)
    
    image_names = []
    descriptors = []
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # load the best model for the decoder
    decoder = DecoderConv(encoded_space_dim=16, conv1_ch=126, conv2_ch=99, conv3_ch=106, fc_ch=59)
    decoder.load_state_dict(torch.load('best_decoderCNN16_final.torch'))
    decoder.to(device)
    decoder.eval()

    cumrow = 1
    header = pd.read_csv(enc, sep=' ', header=None, skiprows=0, nrows=1)
    
    while(not header.empty):
        image_name, kps_len = header.iloc[0].values
        image_names.append(image_name)
        
        body = pd.read_csv(enc, sep=' ', header=None, skiprows=cumrow, nrows=kps_len)
        cumrow = cumrow + kps_len
        
        f = open(out_dir+descriptors_dir+"/"+image_name+".txt", "w")
        f.write(str(kps_len)+" 128\n")
        for i, row in body.iterrows():
            f.write(" ".join(str(i) for i in row.iloc[0:4]))
            f.write(" "+" ".join(str(0) for i in range(128)))
            f.write("\n")
        f.close()

        des_enc = body.loc[:, 4:].to_numpy()
        with torch.no_grad():
            des = decoder(torch.from_numpy(des_enc).float().to(device)).cpu().numpy().reshape(-1, 64)
            
            composed_transform = transforms.Compose([Surf3DInverseReshape(), NpToTensor()])
            
            des_1d = SurfDataset(pd.DataFrame(des), transform=composed_transform)
            dl = DataLoader(des_1d, batch_size=len(des), shuffle=False)
            des = next(iter(dl)).to(device).cpu().numpy()
                        
        descriptors.append(des)

        try:
            header = pd.read_csv(enc, sep=' ', header=None, skiprows=cumrow, nrows=1)
            cumrow += 1
        except:
            header = pd.DataFrame()
    
    
    f = open(out_dir+"/mmm_file.txt", "w")
    for ids in tqdm(combinations(range(len(image_names)), 2)):
        i = ids[0]
        j = ids[1]
        f.write(image_names[i]+" "+image_names[j]+"\n")
        
        #feature matching
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        matches = bf.match(descriptors[i], descriptors[j])
        
        f.write("\n".join(str(match.queryIdx)+" "+str(match.trainIdx) for match in matches))
        f.write("\n\n")

    f.close()
    
"""   
Example:

python generate_sfm_data_decoder.py --enc foutain_encoded_descr.txt --out ~/data/compressed/sfm_data_fountain

python generate_sfm_data_decoder.py --enc tiso_encoded_descr.txt --out sfm_data_tiso
"""
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Receive the encoded descriptors, decode and generate SFM data for colmap")
    
    parser.add_argument("--enc", required=True, help="path to the input file received with the encoded descriptors")
    parser.add_argument("--out", required=True, help="path to the output folder")
    
    args = parser.parse_args()
    
    main(enc=args.enc, out_dir=args.out)
    
