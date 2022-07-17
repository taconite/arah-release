#!/bin/bash
# Download pretrained model
echo "Downloading pretrained model..."
gdown https://drive.google.com/drive/folders/16YSPoN4AQnBPjodfcv9vGjxswCXn-_af -O out/arah-zju/ --folder
# Download and extract pose data
echo "Downloading and extracting pose data..."
gdown https://drive.google.com/uc?id=19D9zZIfZaKukxShsXXIia5WVsA2_7H19
unzip demo_poses.zip
rm -f demo_poses.zip
