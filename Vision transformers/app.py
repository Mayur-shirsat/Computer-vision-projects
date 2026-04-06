import os
import cv2
import torch
import tempfile
import numpy as np
import streamlit as st
from decord import VideoReader, cpu
from ultralytics import YOLO
from transformers import TimesformerForVideoClassification, AutoImageProcessor

# ---------------------- Streamlit Setup ----------------------
st.set_page_config(page_title="Conflict Detection (TimeSformer + YOLOv11s)", layout="wide")
st.title("🎥 Conflict Detection with Person Detection (ViT + YOLOv11s)")
st.markdown("""
Upload a CCTV clip or use live feed to detect **conflict** (fights) and **persons** accurately.
Model: Vision Transformer (TimeSformer) + YOLOv11s
""")

# ---------------------- Model Setup ----------------------
MODEL_DIR = r"D:\manikya tech\projects\Vision transformers"
os.environ["TRANSFORMERS_NO_TF"] = "1"

# --- Load TimeSformer ---
@st.cache_resource
def load_timesformer():
    st.info("⏳ Loading Vision Transformer (TimeSformer)...")
    processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400", cache_dir=MODEL_DIR)
    model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400", cache_dir=MODEL_DIR)
    model.eval()
    return processor, model

# --- Load YOLOv11s (person detection) ---
@st.cache_resource
def load_yolo():
    st.info("⏳ Loading YOLOv11s model...")
    model = YOLO("yolo11s.pt")  # Must be present or downloaded automatically
    return model

processor, vit_model = load_timesformer()
yolo_model = load_yolo()

# Conflict-related classes for ViT
CONFLICT_KEYWORDS = ["boxing", "punching_person", "slapping", "wrestling", "kicking", "fighting", "fight", "arm_wrestling"]

# ---------------------- Conflict Detection Function ----------------------
def analyze_conflict(video_path):
    """Analyzes video frames with TimeSformer to detect conflict-like actions."""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    frame_indices = np.linspace(0, total_frames - 1, 16, dtype=int)
    frames = [vr[i].asnumpy() for i in frame_indices]

    inputs = processor(list(frames), return_tensors="pt")
    with torch.no_grad():
        outputs = vit_model(**inputs)
        logits = outputs.logits
        pred_idx = logits.argmax(-1).item()
        label = vit_model.config.id2label[pred_idx].lower()

    conflict = any(k in label for k in CONFLICT_KEYWORDS)
    return label, conflict

# ---------------------- YOLO Person Detection Function ----------------------
def detect_persons(frame, model):
    """Detect persons using YOLOv11s with red bounding boxes and class accuracy."""
    results = model(frame, verbose=False)
    annotated_frame = frame.copy()

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if label.lower() == "person":
            conf = float(box.conf[0]) * 100
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)  # thin red box
            text = f"person: {conf:.1f}%"
            cv2.putText(annotated_frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return annotated_frame

# ---------------------- Video Processing Function ----------------------
def process_video(video_path):
    """Processes video for both person detection and conflict detection."""
    label, conflict = analyze_conflict(video_path)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20

    output_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect persons
        frame = detect_persons(frame, yolo_model)

        # Add conflict label
        color = (0, 0, 255) if conflict else (0, 255, 0)
        text = f"⚠️ Conflict: {label}" if conflict else f"✅ Normal: {label}"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    return output_path, label, conflict

# ---------------------- Streamlit UI ----------------------
uploaded_video = st.file_uploader("📂 Upload a CCTV video (5–10 sec)", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    st.video(video_path)
    st.write("🔍 Processing video for person and conflict detection...")

    processed_path, label, conflict = process_video(video_path)

    # Display results
    st.subheader("🧠 Model Prediction")
    st.write(f"**Predicted Action:** {label}")
    if conflict:
        st.error(f"⚠️ Conflict Detected — Action: {label}")
    else:
        st.success(f"✅ Normal Scene — Action: {label}")

    # Show processed video
    st.video(processed_path)

    # Provide download option
    with open(processed_path, "rb") as f:
        st.download_button(
            label="⬇️ Download Processed Video",
            data=f,
            file_name="processed_conflict_detection.mp4",
            mime="video/mp4"
        )

    # Safe cleanup (avoid PermissionError)
    try:
        f.close()
        cap = None
        os.remove(video_path)
    except Exception:
        pass
