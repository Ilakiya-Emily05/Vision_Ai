import gdown
import os
import torch

checkpoint_path = "model.pth"

if not os.path.exists(checkpoint_path):
    url = "https://drive.google.com/uc?id=1UKF-vg3I-csqeNzOmvf0Z-daEKi-o84h"
    gdown.download(url, checkpoint_path, quiet=False)

model = torch.load(checkpoint_path, map_location=torch.device("cpu"))
model.eval()
