"""
Streamlit DETR Fire Detection — Auto-download & Local Cache + Alerts
- Fixed deprecated use_column_width → use_container_width
- Fire alert added when fire/smoke/flame detected
"""

import streamlit as st
import torch
import cv2
import numpy as np
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import tempfile
import os
import time

st.set_page_config(page_title="DETR Fire Detection", layout="wide")
st.title("🔥 Fire / Smoke Detection — DETR (Auto-Download + Alerts)")

# ------------------ CONFIG ------------------
LOCAL_MODEL_DIR = r"D:\manikya tech\projects\Fire detection"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "detr-fire-model")
DEFAULT_HF_MODEL = "facebook/detr-resnet-50"

st.sidebar.header("Model & Settings")
model_hf_name = st.sidebar.text_input("HuggingFace model (auto-download if missing)", value=DEFAULT_HF_MODEL)
st.sidebar.write("Local model cache:")
st.sidebar.code(LOCAL_MODEL_PATH)
conf_thres = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)

# ------------------ MODEL LOADING ------------------
@st.cache_resource
def load_or_download_model(local_path: str, hf_name: str):
    if os.path.exists(local_path) and os.path.isdir(local_path):
        try:
            processor = DetrImageProcessor.from_pretrained(local_path)
            model = DetrForObjectDetection.from_pretrained(local_path)
            return processor, model, True
        except Exception:
            pass

    os.makedirs(local_path, exist_ok=True)
    processor = DetrImageProcessor.from_pretrained(hf_name)
    model = DetrForObjectDetection.from_pretrained(hf_name)

    try:
        processor.save_pretrained(local_path)
        model.save_pretrained(local_path)
    except Exception:
        pass

    return processor, model, False

with st.spinner("Loading DETR model... (downloads if missing)"):
    try:
        processor, detr_model, was_local = load_or_download_model(LOCAL_MODEL_PATH, model_hf_name)
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

if was_local:
    st.sidebar.success("Loaded model from local cache")
else:
    st.sidebar.success("Downloaded and cached model locally")

# fire keywords
TARGET_KEYWORDS = ["fire", "flame", "smoke"]

# ----------------- DRAW BOXES -------------------
def draw_boxes(image_bgr: np.ndarray, outputs, target_size):
    processed = processor.post_process_object_detection(outputs, target_sizes=[target_size])[0]
    detected_fire = False

    for score, label, box in zip(processed["scores"], processed["labels"], processed["boxes"]):
        if float(score) < conf_thres:
            continue

        cls_name = detr_model.config.id2label[int(label.item())].lower()
        if not any(k in cls_name for k in TARGET_KEYWORDS):
            continue

        detected_fire = True
        x1, y1, x2, y2 = map(int, box.tolist())
        label_text = f"{cls_name} {float(score):.2f}"

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        ly = max(0, y1 - th - 6)
        cv2.rectangle(image_bgr, (x1, ly), (x1 + tw, ly + th + 6), (0, 0, 255), -1)
        cv2.putText(image_bgr, label_text, (x1, ly + th + -2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return image_bgr, detected_fire

# ------------------ APP INPUT ------------------
mode = st.selectbox("Choose Input", ["Image", "Video"]) 

# ------------------ IMAGE MODE ------------------
if mode == "Image":
    img_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if img_file:
        image = Image.open(img_file).convert("RGB")
        img_np = np.array(image)

        with st.spinner("Analyzing image..."):
            inputs = processor(images=image, return_tensors="pt")
            outputs = detr_model(**inputs)
            out_bgr, detected_fire = draw_boxes(cv2.cvtColor(img_np.copy(), cv2.COLOR_RGB2BGR), outputs, target_size=torch.tensor([image.size[1], image.size[0]]))

        if detected_fire:
            st.error("🔥 ALERT: Fire detected in the image!")
        else:
            st.success("No fire detected.")

        st.image(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB), caption="Detections", use_container_width=True)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        cv2.imwrite(tmp.name, out_bgr)
        with open(tmp.name, "rb") as f:
            st.download_button("Download Processed Image", f, file_name="detr_fire.png")

# ------------------ VIDEO MODE ------------------
else:
    vid_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"]) 
    if vid_file:
        temp_in = tempfile.NamedTemporaryFile(delete=False)
        temp_in.write(vid_file.read())

        cap = cv2.VideoCapture(temp_in.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        writer = cv2.VideoWriter(temp_out.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        progress = st.progress(0)
        frame_no = 0
        fire_detected_any_frame = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_no += 1
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            inputs = processor(images=pil_img, return_tensors="pt")
            outputs = detr_model(**inputs)

            out_frame, detected_fire = draw_boxes(cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR), outputs, target_size=torch.tensor([h, w]))
            if detected_fire:
                fire_detected_any_frame = True

            writer.write(out_frame)

            if total:
                progress.progress(min(100, int(frame_no / total * 100)))

        cap.release()
        writer.release()

        if fire_detected_any_frame:
            st.error("🔥 ALERT: Fire detected in the video!")
        else:
            st.success("No fire detected in the video.")

        prev = cv2.VideoCapture(temp_out.name)
        ret, frame = prev.read()
        if ret:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Preview", use_container_width=True)
        prev.release()

        with open(temp_out.name, "rb") as f:
            st.download_button("Download Processed Video", f, file_name="detr_fire_video.mp4")

st.caption("🔥 DETR-based Fire Detection — Alerts enabled + HF auto-download cache")
