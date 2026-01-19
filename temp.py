from groundingdino.util.inference import load_model, load_image, predict, annotate

import cv2

model = load_model("/workspace/labeling-agent/agent/model_config/GroundingDINO_SwinT_OGC.py", "/workspace/labeling-agent/weights/groundingdino_swint_ogc.pth")
IMAGE_PATH = "../images/05RJPvJXAr5b5EolJzx9zVx-1.fit_lim.size_625x365.v1740431187.jpg"
TEXT_PROMPT = "phone"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)