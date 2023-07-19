import torch
import clip
from PIL import Image

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
# model, preprocess = clip.load("ViT-L-14-336px.pt", device=device)

model, preprocess = clip.load("/root/.cache/clip/ViT-L-14-336px.pt", device=device)




# image = preprocess(Image.open("cat1.png")).unsqueeze(0).to(device)
import numpy as np

images = []
# preprocess 之后 - 单个image size: torch.Size([3, 336, 336])
images.append(preprocess(Image.open("cat1.png")))
images.append(preprocess(Image.open("cat2.png")))
images.append(preprocess(Image.open("CLIP.png")))
image = torch.tensor(np.stack(images)).cuda()

text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():

    image_features = model.encode_image(image)


image_features /= image_features.norm(dim=-1, keepdim=True)
image_features = image_features.cpu().numpy()
similarity = image_features @ image_features.T

    cat = image_features @ image_features.T 
    test = cat.detach().cpu().numpy()
    test = test.astype(float)



    # text_features = model.encode_text(text)
    
    # logits_per_image, logits_per_text = model(image, text)
    
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
