import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
from datetime import datetime
import base64
from io import BytesIO
import pygame
from ultralytics import YOLO
import os
from collections import defaultdict
from scipy.spatial import distance

# Initialize pygame for audio alerts
pygame.mixer.init()

class IntruderTracker:
    def __init__(self):
        self.tracked_intruders = {}  # id: {bbox, last_seen, confidence, status}
        self.next_id = 0
        self.max_disappeared = 30  # frames before considering intruder gone
        self.distance_threshold = 100  # pixels for matching
        
    def register_intruder(self, bbox, confidence):
        """Register a new intruder"""
        intruder_id = self.next_id
        self.tracked_intruders[intruder_id] = {
            'bbox': bbox,
            'last_seen': 0,
            'disappeared': 0,
            'confidence': confidence,
            'status': 'climbing',
            'first_detected': datetime.now()
        }
        self.next_id += 1
        return intruder_id
    
    def update(self, current_detections, climbing_detections):
        """
        Update tracked intruders with current frame detections
        current_detections: all person detections
        climbing_detections: persons currently climbing
        """
        # If no detections, increment disappeared counter
        if len(current_detections) == 0:
            for intruder_id in list(self.tracked_intruders.keys()):
                self.tracked_intruders[intruder_id]['disappeared'] += 1
                if self.tracked_intruders[intruder_id]['disappeared'] > self.max_disappeared:
                    del self.tracked_intruders[intruder_id]
            return []
        
        # Get centroids of current detections
        current_centroids = []
        for det in current_detections:
            x1, y1, x2, y2 = det['bbox']
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            current_centroids.append(centroid)
        
        # If no existing intruders, check for new climbing detections
        if len(self.tracked_intruders) == 0:
            for det in climbing_detections:
                self.register_intruder(det['bbox'], det['confidence'])
        else:
            # Get centroids of existing intruders
            intruder_ids = list(self.tracked_intruders.keys())
            intruder_centroids = []
            for intruder_id in intruder_ids:
                bbox = self.tracked_intruders[intruder_id]['bbox']
                x1, y1, x2, y2 = bbox
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                intruder_centroids.append(centroid)
            
            # Compute distance matrix
            if len(intruder_centroids) > 0 and len(current_centroids) > 0:
                D = distance.cdist(np.array(intruder_centroids), np.array(current_centroids))
                
                # Match existing intruders with current detections
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]
                
                used_rows = set()
                used_cols = set()
                
                # Update matched intruders
                for (row, col) in zip(rows, cols):
                    if row in used_rows or col in used_cols:
                        continue
                    
                    if D[row, col] > self.distance_threshold:
                        continue
                    
                    intruder_id = intruder_ids[row]
                    self.tracked_intruders[intruder_id]['bbox'] = current_detections[col]['bbox']
                    self.tracked_intruders[intruder_id]['confidence'] = current_detections[col]['confidence']
                    self.tracked_intruders[intruder_id]['disappeared'] = 0
                    
                    # Update status (once marked as intruder, stays intruder)
                    # But track if currently climbing or running
                    if current_detections[col]['is_climbing']:
                        self.tracked_intruders[intruder_id]['status'] = 'climbing'
                    else:
                        self.tracked_intruders[intruder_id]['status'] = 'intruder_running'
                    
                    used_rows.add(row)
                    used_cols.add(col)
                
                # Check for disappeared intruders
                unused_rows = set(range(0, D.shape[0])).difference(used_rows)
                for row in unused_rows:
                    intruder_id = intruder_ids[row]
                    self.tracked_intruders[intruder_id]['disappeared'] += 1
                    
                    if self.tracked_intruders[intruder_id]['disappeared'] > self.max_disappeared:
                        del self.tracked_intruders[intruder_id]
                
                # Register new climbing detections as intruders
                unused_cols = set(range(0, D.shape[1])).difference(used_cols)
                for col in unused_cols:
                    if current_detections[col]['is_climbing']:
                        self.register_intruder(
                            current_detections[col]['bbox'],
                            current_detections[col]['confidence']
                        )
        
        return list(self.tracked_intruders.keys())
    
    def get_intruder_info(self, intruder_id):
        """Get information about a specific intruder"""
        if intruder_id in self.tracked_intruders:
            return self.tracked_intruders[intruder_id]
        return None
    
    def get_all_intruders(self):
        """Get all tracked intruders"""
        return self.tracked_intruders


class AdvancedPerimeterDetection:
    def __init__(self):
        # Load YOLO models
        self.model = YOLO('yolov8n.pt')
        try:
            self.pose_model = YOLO('yolov8n-pose.pt')
            self.pose_enabled = True
        except:
            st.warning("Pose model not available. Using position-based detection only.")
            self.pose_enabled = False
        
        # Initialize intruder tracker
        self.intruder_tracker = IntruderTracker()
        
        # Alert settings
        self.alert_threshold = 0.5
        self.alert_cooldown = 3
        self.last_alert_time = {}
        
        # Climbing detection parameters
        self.climbing_threshold = 0.35
        self.height_ratio_threshold = 1.5
        
        # Frame processing
        self.process_every_n_frames = 1
        
    def create_alert_sound(self):
        """Generate an alert sound"""
        try:
            duration = 0.5
            frequency = 800
            sample_rate = 22050
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            wave = np.sin(2 * np.pi * frequency * t)
            audio = (wave * 32767).astype(np.int16)
            stereo_audio = np.column_stack([audio, audio])
            
            sound = pygame.sndarray.make_sound(stereo_audio)
            return sound
        except Exception as e:
            st.warning(f"Could not create audio alert: {e}")
            return None
    
    def is_climbing(self, bbox, keypoints, frame_height, frame_width):
        """Determine if person is climbing"""
        x1, y1, x2, y2 = bbox
        
        person_center_y = (y1 + y2) / 2
        person_height = y2 - y1
        person_width = x2 - x1
        
        relative_position = person_center_y / frame_height
        is_elevated = relative_position < self.climbing_threshold
        
        if person_width > 0:
            height_ratio = person_height / person_width
            is_vertical_pose = height_ratio > self.height_ratio_threshold
        else:
            is_vertical_pose = False
        
        is_climbing_pose = False
        if keypoints is not None and len(keypoints) > 0:
            is_climbing_pose = self.analyze_climbing_pose(keypoints, person_center_y)
        
        climbing = is_elevated and (is_vertical_pose or is_climbing_pose)
        
        return climbing, relative_position, height_ratio
    
    def analyze_climbing_pose(self, keypoints, person_center_y):
        """Analyze pose keypoints to detect climbing behavior"""
        try:
            if len(keypoints.shape) == 3:
                keypoints = keypoints[0]
            
            nose = keypoints[0][:2] if len(keypoints) > 0 else None
            left_shoulder = keypoints[5][:2] if len(keypoints) > 5 else None
            right_shoulder = keypoints[6][:2] if len(keypoints) > 6 else None
            left_wrist = keypoints[9][:2] if len(keypoints) > 9 else None
            right_wrist = keypoints[10][:2] if len(keypoints) > 10 else None
            left_hip = keypoints[11][:2] if len(keypoints) > 11 else None
            right_hip = keypoints[12][:2] if len(keypoints) > 12 else None
            
            climbing_indicators = 0
            
            if (left_wrist is not None and left_shoulder is not None and 
                left_wrist[1] < left_shoulder[1] - 20):
                climbing_indicators += 1
            
            if (right_wrist is not None and right_shoulder is not None and 
                right_wrist[1] < right_shoulder[1] - 20):
                climbing_indicators += 1
            
            if (left_shoulder is not None and left_hip is not None and
                right_shoulder is not None and right_hip is not None):
                shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
                hip_center_y = (left_hip[1] + right_hip[1]) / 2
                body_stretch = hip_center_y - shoulder_center_y
                
                if body_stretch > 80:
                    climbing_indicators += 1
            
            return climbing_indicators >= 2
            
        except Exception as e:
            return False
    
    def detect_intrusion(self, frame, conf_threshold=0.5):
        """Detect persons and determine if they're climbing"""
        frame_height, frame_width = frame.shape[:2]
        
        results = self.model(frame, conf=conf_threshold, verbose=False)
        
        pose_results = None
        if self.pose_enabled:
            try:
                pose_results = self.pose_model(frame, conf=conf_threshold, verbose=False)
            except:
                pass
        
        detections = []
        for idx, result in enumerate(results):
            boxes = result.boxes
            for box_idx, box in enumerate(boxes):
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                
                if class_name == 'person':
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    keypoints = None
                    if pose_results is not None:
                        try:
                            keypoints = pose_results[idx].keypoints.data.cpu().numpy()
                            if len(keypoints) > box_idx:
                                keypoints = keypoints[box_idx]
                        except:
                            pass
                    
                    is_climbing, vertical_pos, height_ratio = self.is_climbing(
                        (x1, y1, x2, y2), keypoints, frame_height, frame_width
                    )
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2),
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'is_climbing': is_climbing,
                        'vertical_position': vertical_pos,
                        'height_ratio': height_ratio,
                        'keypoints': keypoints
                    })
        
        return detections
    
    def draw_detections(self, frame, detections, intruder_ids):
        """Draw bounding boxes for all detections and tracked intruders"""
        annotated_frame = frame.copy()
        
        # First, draw all regular detections (persons not yet marked as intruders)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            center = det['center']
            
            # Check if this detection matches a tracked intruder
            is_tracked_intruder = False
            intruder_info = None
            
            for intruder_id in intruder_ids:
                intruder_data = self.intruder_tracker.get_intruder_info(intruder_id)
                if intruder_data:
                    ix1, iy1, ix2, iy2 = intruder_data['bbox']
                    icenter = ((ix1 + ix2) // 2, (iy1 + iy2) // 2)
                    
                    # Check if centers are close (same person)
                    dist = np.sqrt((center[0] - icenter[0])**2 + (center[1] - icenter[1])**2)
                    if dist < 50:
                        is_tracked_intruder = True
                        intruder_info = (intruder_id, intruder_data)
                        break
            
            if not is_tracked_intruder:
                # Regular detection (not an intruder)
                is_climbing = det['is_climbing']
                confidence = det['confidence']
                
                color = (255, 165, 0)  # Orange for monitoring
                thickness = 2
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                label = f"Person: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                cv2.rectangle(annotated_frame, 
                             (x1, y1 - label_size[1] - 10),
                             (x1 + label_size[0], y1),
                             color, -1)
                
                cv2.putText(annotated_frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.circle(annotated_frame, center, 5, (0, 255, 0), -1)
        
        # Draw all tracked intruders with RED boxes
        for intruder_id in intruder_ids:
            intruder_data = self.intruder_tracker.get_intruder_info(intruder_id)
            if intruder_data:
                x1, y1, x2, y2 = intruder_data['bbox']
                confidence = intruder_data['confidence']
                status = intruder_data['status']
                
                # ALWAYS RED for intruders
                color = (0, 0, 255)
                thickness = 4
                
                # Draw thick red box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Status label
                if status == 'climbing':
                    label = f"INTRUDER #{intruder_id}: CLIMBING!"
                else:
                    label = f"INTRUDER #{intruder_id}: {status.upper()}"
                
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                # Red background
                cv2.rectangle(annotated_frame, 
                             (x1, y1 - label_size[1] - 15),
                             (x1 + label_size[0] + 10, y1),
                             color, -1)
                
                # White text
                cv2.putText(annotated_frame, label, (x1 + 5, y1 - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw warning symbol
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                cv2.circle(annotated_frame, center, 8, (0, 0, 255), -1)
                cv2.circle(annotated_frame, center, 12, (255, 255, 255), 2)
                
                # Add time since detection
                time_elapsed = (datetime.now() - intruder_data['first_detected']).seconds
                time_text = f"Time: {time_elapsed}s"
                cv2.putText(annotated_frame, time_text, (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return annotated_frame
    
    def add_alert_banner(self, frame, num_intruders, climbing_count, total_detections):
        """Add visual alert banner to frame"""
        h, w = frame.shape[:2]
        
        if num_intruders > 0:
            # Red alert for tracked intruders
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 255), -1)
            frame = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)
            
            alert_text = f"🚨 INTRUDER ALERT! Active Intruders: {num_intruders}"
            cv2.putText(frame, alert_text, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
            
            status_text = f"Climbing: {climbing_count} | Running: {num_intruders - climbing_count}"
            cv2.putText(frame, status_text, (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            
            threat_text = "DISPATCH SECURITY IMMEDIATELY"
            cv2.putText(frame, threat_text, (20, 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        elif total_detections > 0:
            # Yellow monitoring banner
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 60), (0, 165, 255), -1)
            frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
            
            monitor_text = f"👁️ Monitoring: {total_detections} person(s) detected"
            cv2.putText(frame, monitor_text, (20, 38),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (w - 300, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def process_frame(self, frame, alert_sound, frame_number):
        """Process a single frame"""
        # Only process every N frames for performance
        if frame_number % self.process_every_n_frames != 0:
            # For skipped frames, just draw existing intruders
            intruder_ids = list(self.intruder_tracker.tracked_intruders.keys())
            annotated_frame = self.draw_detections(frame, [], intruder_ids)
            
            num_intruders = len(intruder_ids)
            climbing_count = sum(1 for iid in intruder_ids 
                               if self.intruder_tracker.get_intruder_info(iid)['status'] == 'climbing')
            
            annotated_frame = self.add_alert_banner(annotated_frame, num_intruders, climbing_count, 0)
            
            return annotated_frame, [], [], False
        
        # Detect all persons
        detections = self.detect_intrusion(frame, self.alert_threshold)
        
        # Separate climbing detections
        climbing_detections = [d for d in detections if d['is_climbing']]
        
        # Update intruder tracker
        intruder_ids = self.intruder_tracker.update(detections, climbing_detections)
        
        # Draw all detections and tracked intruders
        annotated_frame = self.draw_detections(frame, detections, intruder_ids)
        
        # Count statuses
        num_intruders = len(intruder_ids)
        climbing_count = sum(1 for iid in intruder_ids 
                           if self.intruder_tracker.get_intruder_info(iid)['status'] == 'climbing')
        
        # Generate alerts
        alert_triggered = False
        if num_intruders > 0:
            current_time = time.time()
            alert_key = 'intruder'
            
            if alert_key not in self.last_alert_time:
                self.last_alert_time[alert_key] = 0
            
            if current_time - self.last_alert_time[alert_key] > self.alert_cooldown:
                alert_triggered = True
                self.last_alert_time[alert_key] = current_time
                
                if alert_sound:
                    try:
                        alert_sound.play()
                    except:
                        pass
        
        # Add visual banner
        annotated_frame = self.add_alert_banner(annotated_frame, num_intruders, 
                                               climbing_count, len(detections))
        
        return annotated_frame, detections, climbing_detections, alert_triggered


def main():
    st.set_page_config(
        page_title="AI Perimeter Intrusion Detection",
        page_icon="🛡️",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            background-color: #0e1117;
        }
        .stAlert {
            padding: 20px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
        }
        .intruder-alert {
            background-color: #ff0000;
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🛡️ Advanced Perimeter Intrusion Detection System")
    st.markdown("**AI-Powered Intruder Tracking with Persistent Identification**")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    input_type = st.sidebar.selectbox(
        "Select Input Type",
        ["Upload Image", "Upload Video", "Live Webcam"]
    )
    
    confidence_threshold = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    climbing_sensitivity = st.sidebar.slider(
        "Climbing Sensitivity",
        min_value=0.2,
        max_value=0.5,
        value=0.35,
        step=0.05,
        help="Lower = More Sensitive"
    )
    
    # Frame processing option
    frame_skip = st.sidebar.selectbox(
        "Process Every N Frames",
        options=[1, 2, 3, 5, 10],
        index=0,
        help="Process every Nth frame (higher = faster but less accurate)"
    )
    
    # Tracking settings
    st.sidebar.markdown("### 🎯 Tracking Settings")
    tracking_persistence = st.sidebar.slider(
        "Intruder Tracking Persistence (frames)",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
        help="How many frames to keep tracking after intruder disappears"
    )
    
    audio_enabled = st.sidebar.checkbox("Enable Audio Alerts", value=True)
    show_metrics = st.sidebar.checkbox("Show Detection Metrics", value=True)
    
    # Initialize system
    if 'detector' not in st.session_state:
        with st.spinner("Loading AI models..."):
            st.session_state.detector = AdvancedPerimeterDetection()
    
    detector = st.session_state.detector
    detector.alert_threshold = confidence_threshold
    detector.climbing_threshold = climbing_sensitivity
    detector.process_every_n_frames = frame_skip
    detector.intruder_tracker.max_disappeared = tracking_persistence
    
    # Create alert sound
    alert_sound = None
    if audio_enabled:
        alert_sound = detector.create_alert_sound()
    
    # Initialize session state
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Video Feed")
        video_placeholder = st.empty()
    
    with col2:
        st.subheader("🚨 Alert Log")
        alert_placeholder = st.empty()
        
        if show_metrics:
            st.subheader("📊 Detection Metrics")
            metrics_placeholder = st.empty()
        
        st.subheader("👤 Tracked Intruders")
        intruder_list_placeholder = st.empty()
    
    # Process based on input type
    if input_type == "Upload Image":
        uploaded_file = st.sidebar.file_uploader(
            "Upload an image",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            # Reset tracker for new image
            detector.intruder_tracker = IntruderTracker()
            
            image = Image.open(uploaded_file)
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            with st.spinner("Analyzing image..."):
                annotated_frame, detections, climbing_detections, alert_triggered = detector.process_frame(
                    frame, alert_sound, 1
                )
            
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_frame_rgb, use_container_width=True)
            
            # Download button
            result_image = Image.fromarray(annotated_frame_rgb)
            buf = BytesIO()
            result_image.save(buf, format='PNG')
            st.download_button(
                label="⬇️ Download Processed Image",
                data=buf.getvalue(),
                file_name=f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )
            
            # Show alerts
            intruders = detector.intruder_tracker.get_all_intruders()
            if len(intruders) > 0:
                timestamp = datetime.now().strftime("%H:%M:%S")
                for iid, idata in intruders.items():
                    alert_msg = f"[{timestamp}] 🚨 INTRUDER #{iid} DETECTED! Status: {idata['status'].upper()}"
                    st.session_state.alerts.insert(0, alert_msg)
                st.error("⚠️ INTRUDER DETECTED IN IMAGE!")
            
            with alert_placeholder.container():
                for alert in st.session_state.alerts[:10]:
                    st.warning(alert)
            
            # Intruder list
            with intruder_list_placeholder.container():
                if len(intruders) > 0:
                    for iid, idata in intruders.items():
                        st.error(f"**Intruder #{iid}** - Status: {idata['status'].upper()} | Confidence: {idata['confidence']:.2%}")
                else:
                    st.info("No intruders tracked")
            
            if show_metrics:
                with metrics_placeholder.container():
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Persons", len(detections))
                    m2.metric("Climbing", len(climbing_detections))
                    m3.metric("Tracked Intruders", len(intruders))
    
    elif input_type == "Upload Video":
        uploaded_file = st.sidebar.file_uploader(
            "Upload a video",
            type=['mp4', 'avi', 'mov', 'mkv']
        )
        
        if uploaded_file is not None:
            # Reset tracker for new video
            detector.intruder_tracker = IntruderTracker()
            
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()
            
            cap = cv2.VideoCapture(tfile.name)
            
            # Video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Output video
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            stop_button = st.sidebar.button("Stop Processing")
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames}")
                
                # Process frame
                annotated_frame, detections, climbing_detections, alert_triggered = detector.process_frame(
                    frame, alert_sound, frame_count
                )
                
                # Write to output
                out.write(annotated_frame)
                
                # Display every 3rd frame
                if frame_count % 3 == 0:
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(annotated_frame_rgb, use_container_width=True)
                    
                    # Get current intruders
                    intruders = detector.intruder_tracker.get_all_intruders()
                    
                    if alert_triggered and len(intruders) > 0:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        for iid, idata in intruders.items():
                            alert_msg = f"[{timestamp}] Frame {frame_count}: INTRUDER #{iid} - {idata['status'].upper()}"
                            st.session_state.alerts.insert(0, alert_msg)
                    
                    # Update alerts
                    with alert_placeholder.container():
                        if len(intruders) > 0:
                            st.error(f"🚨 {len(intruders)} ACTIVE INTRUDER(S)!")
                        for alert in st.session_state.alerts[:10]:
                            st.warning(alert)
                    
                    # Update intruder list
                    with intruder_list_placeholder.container():
                        if len(intruders) > 0:
                            for iid, idata in intruders.items():
                                time_tracked = (datetime.now() - idata['first_detected']).seconds
                                st.error(f"**Intruder #{iid}** | {idata['status'].upper()} | Tracked: {time_tracked}s | Conf: {idata['confidence']:.2%}")
                        else:
                            st.info("No intruders currently tracked")
                    
                    # Metrics
                    if show_metrics:
                        with metrics_placeholder.container():
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Frame", frame_count)
                            m2.metric("Persons", len(detections))
                            m3.metric("Climbing", len(climbing_detections))
                            m4.metric("Intruders", len(intruders))
            
            cap.release()
            out.release()
            progress_bar.empty()
            status_text.empty()
            
            st.success("✅ Video processing completed!")
            
            # Download button
            with open(output_path, 'rb') as f:
                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=f,
                    file_name=f"intruder_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                    mime="video/mp4"
                )
            
            # Final summary
            final_intruders = detector.intruder_tracker.get_all_intruders()
            if len(final_intruders) > 0:
                st.error(f"**SECURITY SUMMARY: {len(final_intruders)} intruder(s) detected in video**")
                for iid, idata in final_intruders.items():
                    st.warning(f"Intruder #{iid}: First seen at frame ~{iid * 30}, Status: {idata['status']}")
            
            # Cleanup
            os.unlink(tfile.name)
    
    elif input_type == "Live Webcam":
        st.sidebar.info("Live webcam monitoring with persistent intruder tracking")
        
        col_start, col_stop, col_reset = st.sidebar.columns(3)
        start_webcam = col_start.button("▶️ Start")
        stop_webcam = col_stop.button("⏹️ Stop")
        reset_tracking = col_reset.button("🔄 Reset")
        
        if reset_tracking:
            detector.intruder_tracker = IntruderTracker()
            st.session_state.alerts = []
            st.sidebar.success("Tracking reset!")
        
        if start_webcam:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Failed to access webcam")
            else:
                frame_count = 0
                
                while cap.isOpened() and not stop_webcam:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read from webcam")
                        break
                    
                    frame_count += 1
                    
                    # Process frame
                    annotated_frame, detections, climbing_detections, alert_triggered = detector.process_frame(
                        frame, alert_sound, frame_count
                    )
                    
                    # Display
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(annotated_frame_rgb, use_container_width=True)
                    
                    # Get intruders
                    intruders = detector.intruder_tracker.get_all_intruders()
                    
                    if alert_triggered and len(intruders) > 0:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        for iid, idata in intruders.items():
                            alert_msg = f"[{timestamp}] INTRUDER #{iid} - {idata['status'].upper()}"
                            st.session_state.alerts.insert(0, alert_msg)
                    
                    # Update alerts
                    with alert_placeholder.container():
                        if len(intruders) > 0:
                            st.error(f"🚨 {len(intruders)} ACTIVE INTRUDER(S) - TAKE ACTION!")
                        for alert in st.session_state.alerts[:10]:
                            st.warning(alert)
                    
                    # Update intruder list
                    with intruder_list_placeholder.container():
                        if len(intruders) > 0:
                            for iid, idata in intruders.items():
                                time_tracked = (datetime.now() - idata['first_detected']).seconds
                                status_emoji = "🧗" if idata['status'] == 'climbing' else "🏃"
                                st.error(f"{status_emoji} **Intruder #{iid}** | {idata['status'].upper()} | Tracked: {time_tracked}s")
                        else:
                            st.success("✅ No intruders detected")
                    
                    # Metrics
                    if show_metrics:
                        with metrics_placeholder.container():
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Frame", frame_count)
                            m2.metric("Persons", len(detections))
                            m3.metric("Climbing", len(climbing_detections))
                            m4.metric("Intruders", len(intruders))
                    
                    time.sleep(0.03)
                
                cap.release()
                st.info("Webcam stopped")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 System Features")
    st.sidebar.success(
        "**Persistent Intruder Tracking:**\n"
        "✅ Once marked as intruder, ALWAYS tracked\n"
        "✅ RED boxes follow intruders even when running\n"
        "✅ Unique ID assigned to each intruder\n"
        "✅ Time tracking for each intruder\n"
        "✅ Status updates (climbing → running)\n"
        "✅ Adjustable frame processing rate\n"
        "✅ Download processed videos/images"
    )
    
    st.sidebar.markdown("### 🔍 Detection Logic")
    st.sidebar.info(
        "**Initial Detection (Orange Box):**\n"
        "• Person detected near perimeter\n"
        "• Monitoring mode - no alert\n\n"
        "**Intruder Marking (Red Box):**\n"
        "• Person detected climbing\n"
        "• Immediately marked as INTRUDER\n"
        "• Assigned unique ID\n"
        "• RED box persists forever\n\n"
        "**Tracking:**\n"
        "• Intruder tracked across frames\n"
        "• Status updated: climbing → running\n"
        "• Maintains RED box identification\n"
        "• Alerts continue while tracked"
    )
    
    st.sidebar.markdown("### ⚙️ Frame Processing")
    st.sidebar.info(
        f"**Current Setting:** Process every {frame_skip} frame(s)\n\n"
        "• 1 frame: Maximum accuracy, slower\n"
        "• 2-3 frames: Balanced performance\n"
        "• 5-10 frames: Faster, less accurate\n\n"
        "Intruders remain tracked even on skipped frames."
    )


if __name__ == "__main__":
    main()