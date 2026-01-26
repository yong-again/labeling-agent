from groundingdino.util.inference import load_model, predict, annotate
from segment_anything import sam_model_registry, SamPredictor 
import torchvision
import torch
from agent.utils.box_transforms import cxcywh_to_xyxy
from agent.utils.image_loader import load_image
from agent.pipeline import LabelingPipeline
from agent.config import Config
import cv2
import numpy as np
import os

# def get_dino_boxes(image, model, prompt, box_threshold, text_threshold):
#     boxes, logits, phrases = predict(
#         model=model,
#         image=image,
#         caption=prompt,
#         box_threshold=box_threshold,
#         text_threshold=text_threshold
#     )
#     return boxes, logits, phrases

def get_sam_lib_masks(boxes, image):
    sam_checkpoint = "/workspace/labeling-agent/weights/sam_vit_l_0b3195.pth"
    model_type = "vit_l"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(image)
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
    transformed_boxes = transformed_boxes.to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return masks

def get_mask_cordinates(masks, boxes):
        if len(masks) > 0:
            import numpy as np
            # 마지막 마스크 추출
            masks_array = masks.cpu().numpy()  # (N, 1, H, W)
            last_mask = masks_array[-1, 0]  # (H, W) - 마지막 마스크의 2D 형태
            
            # 좌표 추출
            coords = np.where(last_mask > 0)
            
            print(f"\n마지막 마스크 정보:")
            print(f"  Shape: {last_mask.shape}")
            print(f"  True 픽셀 개수: {np.sum(last_mask > 0)}")
            print(f"  좌표 배열 개수: {len(coords[0])}")
            print(f"\n좌표 샘플 (처음 10개):")
            print(f"  Y 좌표: {coords[0][:10]}")
            print(f"  X 좌표: {coords[1][:10]}")
            print(f"\n좌표 샘플 (마지막 10개):")
            print(f"  Y 좌표: {coords[0][-10:]}")
            print(f"  X 좌표: {coords[1][-10:]}")
            
            # 전체 인덱스 결과 출력
            print(f"\n전체 인덱스 결과:")
            print(f"  (array(shape=({len(coords[0])},)), array(shape=({len(coords[1])},)))")
            print(f"  Y range: [{coords[0].min()}, {coords[0].max()}]")
            print(f"  X range: [{coords[1].min()}, {coords[1].max()}]")
            print("=" * 80)

            # 박스 좌표 확인
            print(f"\n변환된 박스 좌표 (첫 번째):")
            print(f"  {boxes[0]}")
            print(f"\n변환된 박스 좌표 (마지막):")
            print(f"  {boxes[-1]}")

def save_masks(masks, output_dir="./output_masks", prefix="mask"):
    """
    Save mask images to disk
    
    Args:
        masks: torch.Tensor of shape (N, 1, H, W) - the masks to save
        output_dir: str - directory to save the masks
        prefix: str - prefix for the mask filenames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert masks to numpy array
    masks_array = masks.cpu().numpy()  # (N, 1, H, W)
    
    print(f"\nSaving {len(masks_array)} masks to {output_dir}...")
    
    for i, mask in enumerate(masks_array):
        # Extract 2D mask (H, W)
        mask_2d = mask[0]  # Remove the channel dimension
        
        # Convert boolean mask to uint8 (0 or 255)
        mask_image = (mask_2d * 255).astype(np.uint8)
        
        # Save mask image
        filename = f"{prefix}_{i:03d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, mask_image)
        print(f"  Saved: {filepath}")
    
    print(f"Successfully saved {len(masks_array)} masks!\n")


if __name__ == "__main__":
    IMAGE_PATH = "/workspace/labeling-agent/uploads/a52cbc65-01ac-4a76-b964-19cc6c166680.png"
    TEXT_PROMPT = "cat"
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    image_source, image_transformed, pil_image = load_image(IMAGE_PATH)
    
    # pipeline results
    print('-' * 50)
    print('pipeline results')
    print('-' * 50)
    config = Config()
    pipeline = LabelingPipeline(config)
    result = pipeline.process_image(image_path=IMAGE_PATH, text_prompt=TEXT_PROMPT)
    pipeline_masks = result.masks
    transformed_boxes = cxcywh_to_xyxy(result.boxes, image_height=image_source.shape[0], image_width=image_source.shape[1])
    get_mask_cordinates(pipeline_masks, transformed_boxes)
    print('-' * 50)

    # sam lib resutls
    print('sam lib results')
    print('-' * 50)
    sam_masks = get_sam_lib_masks(transformed_boxes, image_source)
    get_mask_cordinates(sam_masks, transformed_boxes)
    
    # Save masks
    save_masks(pipeline_masks, output_dir="./output_masks/pipeline", prefix="pipeline_mask")
    save_masks(sam_masks, output_dir="./output_masks/sam_lib", prefix="sam_mask")

