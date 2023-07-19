TEXT_PROMPT = 'person . bicycle . car . motorcycle . airplane . bus . train . truck . boat . traffic light . fire hydrant . bird . cat . dog . horse . sheep . cow . elephant . bear . zebra . giraffe . bed . chair . sofa . dining table . toilet . television . laptop . microwave. oven . toaster . apple . orange . banana . carrot . sandwich . pizza . donut .'
# groundingdino 切割图片完成后进行放大的因子
EXPAND_RATIO = 1.2
ORIG_PATH = '/root/autolabel_dataset/cfsdata/niecong/train2017'
IMAGES_PATH = '/root/autolabel_dataset/cfsdata/niecong/clip_images'
NEW_IMAGES_PATH = '/root/autolabel_dataset/cfsdata/niecong/clip_images_300px'
DEVICE = 'cuda:1'
EMBEDDING_JSON = './embedding.json'
EMBEDDING_NPY = './embedding.npy'

PORT = 8001