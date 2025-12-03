import os 
import json
import pandas as pd
import shutil
from tqdm import tqdm
# This script splits the image directory into train, test_seen, test_unseen and validation sets

base_img_path = './data/image/FB'

gt_path = './data/gt/FB'
def split_img(base_img_path, gt_path):
    all_img_path = base_img_path + '/All'
    def do_data_split(split):
        # read json file 
        split_df = pd.read_json(os.path.join(
                gt_path, '{}.jsonl'.format(split)), lines=True)
        split_path = os.path.join(base_img_path, split)
        os.makedirs(split_path, exist_ok=True)
        
        for img_file in tqdm(split_df["img"], desc="Copying images for {}".format(split)):
            # Copy the image id from all_images_path to dev_seen 
            shutil.copy(os.path.join(all_img_path, img_file[-9:]), split_path)
 
    for split in ['train', 'dev_seen', "dev_unseen", "test_seen", "test_unseen"]:
        do_data_split(split)
        
    

if __name__ == '__main__':
    split_img(base_img_path, gt_path)