import gdown
import os
import torch

checkpoint_path = "deeplabv3_resumed_epoch30.pth"
if not os.path.exists(checkpoint_path):
    url = "https://drive.google.com/uc?id=1UKF-vg3I-csqeNzOmvf0Z-daEKi-o84h"
    gdown.download(url, checkpoint_path, quiet=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(checkpoint_path, map_location=device)
model.eval()

