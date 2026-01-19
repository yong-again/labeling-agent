#!/bin/bash

# Create weights directory
echo "Creating weights directory..."
mkdir -p weights

# Download GroundingDINO weights
echo "Downloading GroundingDINO weights..."
cd weights

# GroundingDINO SWIN-T model
GROUNDINGDINO_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
GROUNDINGDINO_FILE="groundingdino_swint_ogc.pth"

if [ ! -f "$GROUNDINGDINO_FILE" ]; then
    echo "Downloading $GROUNDINGDINO_FILE..."
    wget $GROUNDINGDINO_URL -O $GROUNDINGDINO_FILE
else
    echo "$GROUNDINGDINO_FILE already exists, skipping..."
fi

# Download SAM (Segment Anything Model) weights
echo "Downloading SAM weights..."

# SAM ViT-H model (default model)
SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_FILE="sam_vit_h_4b8939.pth"

if [ ! -f "$SAM_FILE" ]; then
    echo "Downloading $SAM_FILE..."
    wget $SAM_URL -O $SAM_FILE
else
    echo "$SAM_FILE already exists, skipping..."
fi

# Optional: Download other SAM model variants (uncomment if needed)
# SAM ViT-L model
# SAM_VIT_L_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
# wget $SAM_VIT_L_URL -O sam_vit_l_0b3195.pth

# SAM ViT-B model
# SAM_VIT_B_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
# wget $SAM_VIT_B_URL -O sam_vit_b_01ec64.pth

cd ..

echo "Download complete! Model weights are stored in the 'weights' directory."
