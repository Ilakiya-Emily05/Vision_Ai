import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import torch, torchvision
from torchvision import transforms
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.morphology import remove_small_objects, remove_small_holes, closing, disk
from skimage.filters import threshold_otsu
from skimage import measure
import io, gdown, os
from random import randint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes, image_size = 2, 256

model_file_id = "1UKF-vg3I-csqeNzOmvf0Z-daEKi-o84h"
model_path = "deeplabv3_resumed_epoch30.pth"
demo_file_id = "1t_gh8qPnjwpu7WwPQBz9YNp16ARvL8M8"
demo_path = "demo_image.png"
how_it_works_file_id = "1RMd3LiX84ZgDQUWQqG5jfWPBqGoiDPzJ"
how_it_works_path = "how_it_works.png"

for path, file_id in [(model_path, model_file_id),(demo_path,demo_file_id),(how_it_works_path,how_it_works_file_id)]:
    if not os.path.exists(path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)

@st.cache_resource(show_spinner=True)
def load_model():
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, aux_loss=True, num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = {k:v for k,v in checkpoint.items() if "aux_classifier" not in k}
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((image_size,image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
])

def refine_mask(prob_mask, min_size=500, dilate_size=3):
    smoothed = gaussian_filter(prob_mask, sigma=1.0)
    smoothed = gaussian_filter(smoothed, sigma=2.0)
    try: thresh_val = threshold_otsu(smoothed)
    except: thresh_val = 0.3*smoothed.max()
    mask = (smoothed>thresh_val).astype(np.uint8)
    mask = remove_small_objects(mask.astype(bool), min_size=min_size)
    mask = remove_small_holes(mask, area_threshold=min_size)
    mask = closing(mask,disk(3))
    mask = binary_dilation(mask, structure=disk(dilate_size))
    return mask.astype(np.uint8)

def tta_inference(model,img_tensor,scales=[0.75,1.0,1.25],flips=[None,'h','v']):
    _,C,H,W = img_tensor.shape
    agg_output = torch.zeros((1,num_classes,H,W),device=img_tensor.device)
    for scale in scales:
        scaled_img = img_tensor if scale==1.0 else torch.nn.functional.interpolate(img_tensor,size=(int(H*scale),int(W*scale)),mode='bilinear',align_corners=False)
        for f in flips:
            inp = scaled_img
            if f=='h': inp = torch.flip(inp,[3])
            if f=='v': inp = torch.flip(inp,[2])
            with torch.no_grad(): out = model(inp)["out"]
            if f=='h': out = torch.flip(out,[3])
            if f=='v': out = torch.flip(out,[2])
            if scale!=1.0: out = torch.nn.functional.interpolate(out, size=(H,W), mode='bilinear', align_corners=False)
            agg_output += out
    agg_output /= (len(scales)*len(flips))
    return agg_output

st.set_page_config(page_title="The Pixel Wizard", layout="wide")

# ------------------- Kawaii CSS -------------------
st.markdown("""
<style>
/* Full page background */
[data-testid="stAppViewContainer"] {background: #ffc5d3;}
[data-testid="stSidebar"] {background: #ffe0e9;}

/* Titles */
h1,h2 {font-family: 'Comic Sans MS', cursive, sans-serif; color: #FF69B4; text-align:center;}

/* Gradient Buttons */
div.stButton>button {background: linear-gradient(90deg,#FFB6C1,#FF69B4); color:white; font-weight:bold; border-radius:12px;}

/* Sliders kawaii */
.css-1aumxhk input[type=range] {accent-color: #FF69B4; height:8px; border-radius:10px;}

/* Add sparkles effect */
body::after {content:'';position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;background-image:radial-gradient(white 1px, transparent 1px);background-size:20px 20px;opacity:0.1;}
</style>
""", unsafe_allow_html=True)

# ------------------- Title -------------------
st.markdown("<h1>The Pixel Wizard</h1>", unsafe_allow_html=True)
st.markdown("<h2>Transforming Images with Precision and Magic</h2>", unsafe_allow_html=True)

# ------------------- How it Works -------------------
how_img = Image.open(how_it_works_path).convert("RGBA")
sparkle = Image.new("RGBA", how_img.size)
draw = ImageDraw.Draw(sparkle)
for _ in range(100):
    x,y = randint(0,how_img.width-1), randint(0,how_img.height-1)
    draw.ellipse((x,y,x+4,y+4),fill=(255,255,255,150))
how_img = Image.alpha_composite(how_img,sparkle)
st.image(how_img, use_container_width=True)

st.markdown("""
<h3 style='text-align:center; color:#FF69B4;'>How the Tool Works:</h3>
<ul>
<li>Upload any image or try the demo</li>
<li>Automatically segment objects with AI precision</li>
<li>Remove or replace backgrounds easily</li>
<li>Highlight edges with cool overlays</li>
</ul>
""", unsafe_allow_html=True)

# ------------------- Demo / Upload -------------------
col1,col2 = st.columns(2)
use_demo = col1.button("Try Demo Image")
uploaded_file = col2.file_uploader("Or Upload Your Own Image", type=["jpg","jpeg","png"])

if use_demo:
    image = Image.open(demo_path).convert("RGB")
elif uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
else:
    st.stop()

img_tensor = transform(image).unsqueeze(0).to(device)

# ------------------- Sidebar Controls -------------------
st.sidebar.subheader("Mask Morphology Controls")
min_size = st.sidebar.slider("Min Object Size", 100, 5000, 500, 50)
dilate_size = st.sidebar.slider("Dilation Size", 1, 15, 3)

st.sidebar.subheader("Edge Overlay Settings")
edge_color = st.sidebar.color_picker("Edge Color", "#FF69B4")
edge_thick = st.sidebar.slider("Edge Thickness", 1, 10, 2)

st.sidebar.subheader("Background Removal / Replacement")
bg_option = st.sidebar.selectbox("Background", ["Transparent", "Black", "White", "Custom Color"])
if bg_option == "Custom Color":
    bg_color = st.sidebar.color_picker("Pick BG Color", "#FFC0CB")
else:
    bg_color = {"Black":"#000000","White":"#FFFFFF","Transparent":None}[bg_option]

# ------------------- Mask & Overlay -------------------
output = tta_inference(model,img_tensor)
prob_mask = torch.softmax(output,dim=1)[0,1].cpu().numpy()
final_mask = refine_mask(prob_mask, min_size=min_size, dilate_size=dilate_size)
mask_resized = Image.fromarray((final_mask*255).astype(np.uint8)).resize(image.size,resample=Image.NEAREST)
mask_bool = np.array(mask_resized).astype(bool)

contours = measure.find_contours(np.array(mask_resized)//255,0.5)
overlay_edges = image.copy()
draw = ImageDraw.Draw(overlay_edges)
for contour in contours:
    contour = contour * (image.size[0]/mask_resized.width)
    contour = [tuple(p[::-1]) for p in contour]
    if len(contour)>1:
        draw.line(contour, fill=edge_color, width=edge_thick)

img_np = np.array(image)
if bg_color is None:
    seg_out = np.zeros((image.size[1],image.size[0],4),dtype=np.uint8)
    seg_out[...,:3] = img_np * mask_bool[...,None]
    seg_out[...,3] = mask_bool.astype(np.uint8)*255
else:
    bg_rgb = tuple(int(bg_color.lstrip("#")[i:i+2],16) for i in (0,2,4))
    seg_out = np.zeros_like(img_np)
    seg_out[mask_bool] = img_np[mask_bool]
    seg_out[~mask_bool] = bg_rgb
segmented_output = Image.fromarray(seg_out)

# ------------------- Display -------------------
st.subheader("Results")
col1,col2,col3 = st.columns(3)
with col1: st.image(image, caption="Original Image", use_container_width=True)
with col2: st.image(segmented_output, caption="Segmented / BG Removed", use_container_width=True)
with col3: st.image(overlay_edges, caption="Edges Overlay", use_container_width=True)

# ------------------- Download -------------------
st.subheader("Download Options")
for img, name, fname in [(image,"Original","original.png"),(segmented_output,"Segmented Object","segmented_object.png"),(overlay_edges,"Edge Overlay","edge_overlay.png")]:
    buf = io.BytesIO()
    img.save(buf,format="PNG")
    st.download_button(f"Download {name}", buf.getvalue(), file_name=fname, mime="image/png")
