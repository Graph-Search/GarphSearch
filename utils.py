from GroundingDINO import crop_image_by_groundingdino
from config import *
import cv2
import os
import sys
import tqdm
import shutil
sys.path.append('/root/GraphSearch/GroundingDINO')
def crop_and_save(orig_path, save_path):
    image_files = os.listdir(orig_path)
    img_num = 0
    for image_file in tqdm.tqdm(image_files):
        file_path = os.path.join(orig_path, image_file)
        image, boxes, labels = crop_image_by_groundingdino(file_path, TEXT_PROMPT)
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2) 
            center_y = int((y1 + y2) / 2)
            width = x2 - x1
            height = y2 - y1 
            expanded_width = int(width * EXPAND_RATIO) 
            expanded_height = int(height * EXPAND_RATIO) 
            expanded_x1 = max(center_x - expanded_width // 2, 0)
            expanded_y1 = max(center_y - expanded_height // 2, 0)
            expanded_x2 = min(center_x + expanded_width // 2, image.shape[1])
            expanded_y2 = min(center_y + expanded_height // 2, image.shape[0])
            cropped_image = image[expanded_y1:expanded_y2, expanded_x1:expanded_x2][:,:,::-1]
            cv2.imwrite(os.path.join(save_path, str(img_num) + '.jpg'), cropped_image)
            img_num += 1
    return img_num
        
def filter_and_save(image_dir, new_image_dir):
    filtered_files = os.listdir(image_dir)
    os.makedirs(new_image_dir, exist_ok=True)
    img_num = 0
    for image_file in tqdm.tqdm(filtered_files):
        img = cv2.imread(f'{image_dir}/{image_file}')
        H, W, C = img.shape
        if H > 150 and W > 150:
            shutil.move(f'{image_dir}/{image_file}', f'{new_image_dir}/{image_file}')
            img_num += 1
    return img_num
if __name__ == '__main__':
    # crop_and_save(ORIG_PATH, IMAGES_PATH)
    filter_and_save(IMAGES_PATH, NEW_IMAGES_PATH)