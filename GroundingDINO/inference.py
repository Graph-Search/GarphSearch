import sys
sys.path.append('/root/GraphSearch/GroundingDINO')
from groundingdino.util.inference import load_model, load_image, predict, annotate, get_annotate
import cv2
import numpy as np
from typing import Tuple, List, Optional
model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "GroundingDINO/weights/groundingdino_swint_ogc.pth", device = 'cuda:6')
# IMAGE_PATH = '/root/GroundingDINO-main/images/1688620950646756.jpg'
# TEXT_PROMPT = "cans"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

# image_source, image = load_image(IMAGE_PATH)

# boxes, logits, phrases = predict(
#     model=model,
#     image=image,
#     caption=TEXT_PROMPT,
#     box_threshold=BOX_TRESHOLD,
#     text_threshold=TEXT_TRESHOLD
# )

# annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
# cv2.imwrite("annotated_1688620950646756.jpg", annotated_frame)

def crop_image_by_groundingdino(image_path: str, text_prompt: str) -> Tuple[np.ndarray, List[str], List[str]]:
    image_source, image = load_image(image_path)
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    boxes, labels = get_annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    return image_source, boxes, labels
