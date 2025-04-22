import streamlit as st
from PIL import Image
import torch
import numpy as np
import tempfile
import cv2

st.set_page_config(layout="wide")
st.title("🛩️ Drone Forensic Detection (YOLOv5s + Optical Flow)")

# Load pretrained YOLOv5s once
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

model = load_model()

# YOLO Inference Function
def infer_image(img):
    results = model(img)
    results.render()
    return Image.fromarray(results.ims[0])

# Optical Flow Function
def draw_optical_flow(prev_img, next_img):
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    if p0 is None:
        return next_img

    lk_params = dict(winSize=(15,15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p1, stt, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None, **lk_params)

    for i, (pt1, pt2) in enumerate(zip(p0, p1)):
        if stt[i]:
            x0, y0 = pt1.ravel()
            x1, y1 = pt2.ravel()
            cv2.arrowedLine(next_img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2, tipLength=0.3)

    return next_img

# Upload & Process
uploaded_imgs = st.file_uploader("Upload two consecutive images (for optical flow)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_imgs and len(uploaded_imgs) == 2:
    img_paths = []
    for uploaded in uploaded_imgs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded.read())
            img_paths.append(tmp.name)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_paths[0], caption="Previous Frame", use_column_width=True)
        pred = infer_image(img_paths[0])
        st.image(pred, caption="YOLO Prediction 1", use_column_width=True)
    
    with col2:
        st.image(img_paths[1], caption="Next Frame", use_column_width=True)
        pred2 = infer_image(img_paths[1])
        st.image(pred2, caption="YOLO Prediction 2", use_column_width=True)

    # Optical Flow Visualization
    st.subheader("Optical Flow Quiver Overlay")
    prev = cv2.imread(img_paths[0])
    nextf = cv2.imread(img_paths[1])
    result = draw_optical_flow(prev, nextf)
    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Optical Flow Overlay", use_column_width=True)

elif uploaded_imgs and len(uploaded_imgs) != 2:
    st.warning("Please upload **exactly two images** to visualize optical flow.", icon="⚠️")
