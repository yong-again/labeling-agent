from groundingdino.util.inference import load_model, predict, annotate
from segment_anything import sam_model_registry, SamPredictor 
import torchvision
import torch
from agent.utils.box_transforms import cxcywh_to_xyxy
from agent.utils.image_loader import load_image
import cv2

model = load_model("/workspace/labeling-agent/agent/model_config/GroundingDINO_SwinT_OGC.py", "/workspace/labeling-agent/weights/groundingdino_swint_ogc.pth")
IMAGE_PATH = "/workspace/labeling-agent/uploads/a52cbc65-01ac-4a76-b964-19cc6c166680.png"
TEXT_PROMPT = "cat"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image_transformed, pil_image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image_transformed,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

# annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
# cv2.imwrite("annotated_image.jpg", annotated_frame)

image_width, image_height = pil_image.size

transformed_boxes = cxcywh_to_xyxy(boxes, image_width, image_height)

sam_checkpoint = "/workspace/labeling-agent/weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image_source)
transformed_boxes = torch.tensor(transformed_boxes, device=device)

masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)


