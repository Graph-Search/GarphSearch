import torch
from CLIP import clip
from PIL import Image
import numpy as np
import json
import glob
import gc
import argparse
from config import *

class Image2Embedding:
    def __init__(self, device, read_path, out_path1, out_path2, app):
        self.device = device
        self.read_path = read_path
        self.out_path1 = out_path1
        self.out_path2 = out_path2
        self.batch_size = 500
        self.img_files = glob.glob(self.read_path + '/*.jpg')# .jpg
        self.file_nums = len(self.img_files)
        self.model, self.preprocess = clip.load("/root/.cache/clip/ViT-L-14-336px.pt", device=self.device)
        self.images_embedding = []
        self.scale = torch.ones([]) * np.log(1 / 0.07)
        self.scale = self.scale.exp()
        self.app = app

        
    def preprocess_img_batches(self):
        # including calling preprocess
        # self.images_embedding - List[List]
        
        batch_num = (self.file_nums // self.batch_size) + 1
        for i in range(batch_num):
            print(f'{i} / {batch_num}')
            batch_files = self.img_files[i * self.batch_size: (i + 1) * self.batch_size]
            images = []
            for filename in batch_files:
                images.append(self.preprocess(Image.open(filename)))

            image_features = self.cal_embedding(images).astype(np.float32)
            image_features = image_features.tolist()
            self.images_embedding.extend(image_features)
    

    def forward_one_image(self, one_image_path, top_k):
        if self.app:
            # loaded directly as a image from the website instead of img_path
            one_image = self.preprocess(one_image_path).unsqueeze(0)
        else:
            one_image = self.preprocess(Image.open(one_image_path)).unsqueeze(0)
        one_image_features = self.cal_embedding(one_image).astype(np.float32) # numpy
        
        image_features = np.load(self.out_path2) # numpy # Notice: loaded as np.float64 even though casted as np.float32 when saving
        image_probs = self.cal_similarity(one_image_features, image_features)
        
        top_probs, top_index = image_probs.topk(top_k, dim=-1)
        top_probs = top_probs.numpy().squeeze(0)

        top_index = top_index.numpy().squeeze(0)
        found_files = np.array(self.img_files)[top_index] 

        
        return found_files, top_probs, image_probs



    def cal_embedding(self, images):
        # input - numpy
        # to_torch and then implement model.encode_image
        # return - numpy
        image_input = torch.tensor(np.stack(images)).to(self.device)
        # breakpoint()
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            # image_features = self.model.encode_image(image_input).float() # 如果这里转float的话就不需要在forward里numpy.astype(np.float32)了
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
        image_features_cpu = image_features.cpu().numpy()
        del image_input # 删除显存
        del image_features
        torch.cuda.empty_cache()
        # image_input = None # gc删除内存
        # image_features = None
        gc.collect()
        return image_features_cpu

    def cal_similarity(self, one_image_features, image_features):
        # input - numpy, numpy
        # return - torch.tensor [1, image_features.shape[0]] in cpu()
        one_image_features = torch.tensor(one_image_features).to(self.device).float()
        image_features = torch.tensor(image_features).to(self.device).float()

        # if do softmax(dim=-1), variables are required as 'torch_tensor' format
        image_probs = (self.scale * one_image_features @ image_features.T).softmax(dim=-1)  # return as (1,4)
        # image_probs = (self.scale * image_features @ one_image_features.T).softmax(dim=0) # return as (4,1)
        image_probs = image_probs.cpu()

        return image_probs

    def save_files(self):
        # load self.images_embedding into output_path2

        np.save(self.out_path2, np.array(self.images_embedding))

        # load self.images_embedding and self.img_files into output_path2
        dic = {}
        # breakpoint()
        for i in range(self.file_nums):
            dic[i] = {'img_path': self.img_files[i], 'img_embedding': self.images_embedding[i]}
        with open(self.out_path1, 'w') as f:
            json.dump(dic, f)
        
    @torch.no_grad()
    def pre_embedding_process(self):
        self.preprocess_img_batches()
        self.save_files()

    
def parser():
    parser = argparse.ArgumentParser(description='parameters of GraphSearch')
    parser.add_argument('--pre_process', default=False,  type=bool, help='whether generate embeddings for dataset images or perform one_image inference')
    parser.add_argument('--top_k', default=200, type=int, help='the first top_k images that most similar to input of one_image')
    parser.add_argument('--app', default=False,  type=bool, help='whether for App, default to be False')

    return parser.parse_args()


if __name__ == "__main__":
    device = "cuda:1"
    read_dir = NEW_IMAGES_PATH
    output_path1 = EMBEDDING_JSON
    output_path2 = EMBEDDING_NPY
    one_img_path = '/root/CLIP/car.jpg'

    args = parser()
    image_processor = Image2Embedding(device, read_dir, output_path1, output_path2, args.app)
    if args.pre_process:
        image_processor.pre_embedding_process()
    else:
        found_files, top_probs, image_probs = image_processor.forward_one_image(one_img_path, args.top_k)

        print("all_prob:", image_probs)
        print("find top {}:".format(args.top_k), found_files, top_probs)