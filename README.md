# GarphSearch
The project designed for image-to-image retrieval

## Usage

### 1. download cropped images and model weight files

* with new collected and uncropped images, run `python utils.py` to perform detections and cropping

### 2. generate image (embedding) library

~~~shell
python clip_search.py --pre_process=True --app=False
~~~

* or download the embeddings from the website

### 3. run GraphSearch

* find the top_k images that similar to the input image

* modify the input image path (named as one_img_path) in clip_search.py

~~~shell
python clip_search.py --pre_process=False --top_k=200 --app=False
~~~

## Results

![GarphSearch](Graph_top10.png)
