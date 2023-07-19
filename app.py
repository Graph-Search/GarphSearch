
from flask import Flask, request, jsonify
from PIL import Image 
import io
app = Flask(__name__)

from clip_search import Image2Embedding
from config import *

i2e = Image2Embedding(DEVICE, NEW_IMAGES_PATH, EMBEDDING_JSON, EMBEDDING_NPY)


@app.route('/process', methods=['POST'])
def process_image():
    # 在这里调用 process_img 函数处理接收到的图片
    # 获取上传的图片数据
    image_file = request.files['image']
    # 将上传的图片转换成 PIL.Image 对象
    image = Image.open(io.BytesIO(image_file.read()))
    found_files, top_probs, image_probs = image_processor.forward_one_image(image)
    return jsonify({"urls": found_files})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
