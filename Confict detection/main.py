import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import tempfile
import os
import time

# ======================================================
# CNN + LSTM MODEL
# ======================================================
class CNN_LSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=1):
        super().__init__()

        resnet = models.resnet18(weights="IMAGENET1K_V1")
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn.eval()

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)

        with torch.no_grad():
            features = self.cnn(x).reshape(b, t, 512)

        lstm_out, _ = self.lstm(features)
        out = self.fc(lstm_out[:, -1, :])
        return out


# ======================================================
# LOAD MODEL
# ======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = r"E:\Rajyug working\Conflict detection\20.pt"

model = CNN_LSTM().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ======================================================
# TRANSFORMS
# ======================================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ======================================================
# PREDICT VIDEO (FIRST 20 FRAMES ONLY)
# ======================================================
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    interval = max(total_frames // 20, 1)

    for i in range(20):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(transform(frame))

    cap.release()

    frames = torch.stack(frames).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(frames)
        pred = torch.argmax(output).item()

    return pred  # 1 = VIOLENCE, 0 = NON-VIOLENCE


# ======================================================
# PROCESS VIDEO (ADD LABEL)
# ======================================================
def process_video(input_path, output_path, violence_detected):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if violence_detected:
            cv2.putText(frame, "ALERT: VIOLENCE", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        out.write(frame)

    cap.release()
    out.release()


# ======================================================
# RTSP LIVE CAMERA STREAM INFERENCE
# ======================================================
def predict_rtsp(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)

    st_frame = st.empty()

    frame_list = []  # store only last 20 frames

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Unable to access RTSP stream")
            break

        # Convert + append for inference
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_list.append(transform(rgb))

        # Keep max 20 frames
        if len(frame_list) > 20:
            frame_list.pop(0)

        # Real-time display
        st_frame.image(rgb, channels="RGB")

        # Run detection if enough frames collected
        if len(frame_list) == 20:
            with torch.no_grad():
                batch = torch.stack(frame_list).unsqueeze(0).to(device)
                output = model(batch)
                pred = torch.argmax(output).item()

            if pred == 1:
                st.error("⚠️ VIOLENCE DETECTED — LIVE")
            else:
                st.success("✔ Normal — No Violence Detected")

        time.sleep(0.15)

    cap.release()


# ======================================================
# STREAMLIT UI
# ======================================================
st.set_page_config(page_title="Violence Detection", layout="wide")
st.title("⚠️ Violence Detection — CNN + LSTM Model")

mode = st.radio("Choose Input Type", ["Upload Video", "RTSP Live Camera"])

# ---------------------- VIDEO UPLOAD MODE ----------------------
if mode == "Upload Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_input.write(uploaded_video.read())
        temp_input_path = temp_input.name

        st.video(temp_input_path)

        if st.button("Process Video"):
            with st.spinner("Analyzing video..."):

                prediction = predict_video(temp_input_path)
                violence = (prediction == 1)

                temp_output_path = temp_input_path.replace(".mp4", "_processed.mp4")
                process_video(temp_input_path, temp_output_path, violence)

            if violence:
                st.error("⚠️ VIOLENCE DETECTED")
            else:
                st.success("✔ No Violence Detected")

            st.video(temp_output_path)

            with open(temp_output_path, "rb") as f:
                st.download_button(
                    label="Download Processed Video",
                    data=f,
                    file_name="processed_output.mp4",
                    mime="video/mp4"
                )

# ---------------------- RTSP LIVE CAMERA MODE ----------------------
else:
    rtsp_url = st.text_input("Enter RTSP URL", placeholder="rtsp://username:password@ip:port/stream")

    if st.button("Start RTSP Stream"):
        if rtsp_url.strip() == "":
            st.warning("Please enter a valid RTSP link.")
        else:
            predict_rtsp(rtsp_url)
