import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
from datetime import datetime
import pygame
from ultralytics import YOLO
import os
from collections import deque
import torch
import json

# Initialize pygame for audio alerts
pygame.mixer.init()

class ZoneDrawer:
    """Interactive zone drawing tool"""
    def __init__(self):
        self.zones = []
        self.current_zone = []
        self.drawing = False
        
    def add_point(self, x, y):
        """Add point to current zone"""
        self.current_zone.append([int(x), int(y)])
    
    def complete_zone(self, name="Restricted Zone"):
        """Complete current zone"""
        if len(self.current_zone) >= 3:
            self.zones.append({
                'name': name,
                'points': self.current_zone.copy(),
                'color': (0, 0, 255)
            })
            self.current_zone = []
            return True
        return False
    
    def clear_current(self):
        """Clear current zone being drawn"""
        self.current_zone = []
    
    def clear_all(self):
        """Clear all zones"""
        self.zones = []
        self.current_zone = []
    
    def draw_zones(self, frame):
        """Draw all zones on frame"""
        overlay = frame.copy()
        
        # Draw completed zones
        for zone in self.zones:
            points = np.array(zone['points'], dtype=np.int32)
            cv2.fillPoly(overlay, [points], zone['color'])
            cv2.polylines(frame, [points], True, (0, 255, 0), 2)
            
            # Draw zone name
            if len(points) > 0:
                cx = int(np.mean(points[:, 0]))
                cy = int(np.mean(points[:, 1]))
                cv2.putText(frame, zone['name'], (cx, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw current zone being created
        if len(self.current_zone) > 0:
            points = np.array(self.current_zone, dtype=np.int32)
            cv2.polylines(frame, [points], False, (255, 255, 0), 2)
            for point in self.current_zone:
                cv2.circle(frame, tuple(point), 5, (0, 255, 255), -1)
        
        # Blend overlay with frame
        frame = cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)
        return frame
    
    def save_zones(self, filename='zones.json'):
        """Save zones to file"""
        with open(filename, 'w') as f:
            json.dump(self.zones, f)
    
    def load_zones(self, filename='zones.json'):
        """Load zones from file"""
        try:
            with open(filename, 'r') as f:
                self.zones = json.load(f)
            return True
        except:
            return False


class PerimeterDetectionSystem:
    def __init__(self, model_size='m'):
        # Load YOLO model
        model_options = {
            'n': 'yolov8n.pt',
            's': 'yolov8s.pt',
            'm': 'yolov8m.pt',
            'l': 'yolov8l.pt',
            'x': 'yolov8x.pt'
        }
        
        self.model = YOLO(model_options.get(model_size, 'yolov8m.pt'))
        
        # Use GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Enhanced detection parameters
        self.iou_threshold = 0.4
        self.conf_threshold = 0.5
        self.max_det = 300
        
        # Tracking system
        self.tracker = {}
        self.next_id = 0
        self.max_disappeared = 30
        
        # Motion detection
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
        
        # Detection history
        self.detection_history = deque(maxlen=10)
        
        # Behavior analysis
        self.dwell_time_threshold = 3
        self.velocity_threshold = 5
        
        # Alert settings
        self.alert_cooldown = 2
        self.last_alert_time = 0
        
        # Target classes with priorities
        self.target_classes = {
            'person': 1.0,
            'bicycle': 0.7,
            'motorcycle': 0.8,
            'car': 0.6,
            'truck': 0.6,
            'backpack': 0.5,
            'handbag': 0.5
        }
        
        # Zone drawer
        self.zone_drawer = ZoneDrawer()
        
    def create_alert_sound(self):
        """Generate alert sound"""
        try:
            duration = 0.3
            sample_rate = 44100
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            wave1 = np.sin(2 * np.pi * 800 * t)
            wave2 = np.sin(2 * np.pi * 1000 * t)
            wave = (wave1 + wave2) / 2
            
            fade_samples = int(sample_rate * 0.05)
            wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
            wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            
            audio = (wave * 32767).astype(np.int16)
            stereo_audio = np.column_stack([audio, audio])
            
            sound = pygame.sndarray.make_sound(stereo_audio)
            return sound
        except Exception as e:
            return None
    
    def detect_motion(self, frame):
        """Detect motion in frame"""
        fg_mask = self.background_subtractor.apply(frame)
        fg_mask[fg_mask == 127] = 0
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        motion_pixels = np.sum(fg_mask == 255)
        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        motion_ratio = motion_pixels / total_pixels
        
        return fg_mask, motion_ratio
    
    def is_in_restricted_zone(self, center_point):
        """Check if point is in any restricted zone"""
        if not self.zone_drawer.zones:
            return True
        
        x, y = center_point
        for zone in self.zone_drawer.zones:
            points = np.array(zone['points'], dtype=np.int32)
            if cv2.pointPolygonTest(points, (float(x), float(y)), False) >= 0:
                return True
        return False
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def track_objects(self, detections):
        """Track objects across frames"""
        current_objects = {}
        
        for det in detections:
            best_match = None
            best_iou = 0
            
            for obj_id, tracked_obj in self.tracker.items():
                iou = self.calculate_iou(det['bbox'], tracked_obj['bbox'])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_match = obj_id
            
            if best_match:
                current_objects[best_match] = {
                    'bbox': det['bbox'],
                    'class': det['class'],
                    'confidence': det['confidence'],
                    'center': det['center'],
                    'disappeared': 0,
                    'first_seen': self.tracker[best_match]['first_seen'],
                    'track_history': self.tracker[best_match]['track_history'] + [det['center']]
                }
            else:
                current_objects[self.next_id] = {
                    'bbox': det['bbox'],
                    'class': det['class'],
                    'confidence': det['confidence'],
                    'center': det['center'],
                    'disappeared': 0,
                    'first_seen': time.time(),
                    'track_history': [det['center']]
                }
                self.next_id += 1
        
        for obj_id in list(self.tracker.keys()):
            if obj_id not in current_objects:
                self.tracker[obj_id]['disappeared'] += 1
                if self.tracker[obj_id]['disappeared'] < self.max_disappeared:
                    current_objects[obj_id] = self.tracker[obj_id]
        
        self.tracker = current_objects
        return self.tracker
    
    def analyze_behavior(self, tracked_objects):
        """Analyze suspicious behaviors"""
        suspicious = []
        
        for obj_id, obj in tracked_objects.items():
            threat_level = 0
            reasons = []
            
            dwell_time = time.time() - obj['first_seen']
            if dwell_time > self.dwell_time_threshold:
                threat_level += 0.2
                reasons.append("Loitering")
            
            if len(obj['track_history']) > 5:
                recent_points = obj['track_history'][-5:]
                distances = [
                    np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                    for p1, p2 in zip(recent_points[:-1], recent_points[1:])
                ]
                avg_velocity = np.mean(distances)
                
                if avg_velocity > self.velocity_threshold:
                    threat_level += 0.3
                    reasons.append("Rapid movement")
            
            if self.is_in_restricted_zone(obj['center']):
                threat_level += 0.5
                reasons.append("Restricted zone")
            
            if threat_level > 0.3:
                suspicious.append({
                    'id': obj_id,
                    'threat_level': min(threat_level, 1.0),
                    'reasons': reasons,
                    'object': obj
                })
        
        return suspicious
    
    def detect_intrusion(self, frame, motion_mask=None, use_motion_filter=True):
        """Detect intrusions"""
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
            agnostic_nms=True
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                
                if class_name in self.target_classes:
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    motion_validated = True
                    if use_motion_filter and motion_mask is not None:
                        roi = motion_mask[y1:y2, x1:x2]
                        if roi.size > 0:
                            motion_in_roi = np.sum(roi == 255) / roi.size
                            motion_validated = motion_in_roi > 0.1
                    
                    if motion_validated:
                        priority = self.target_classes[class_name]
                        adjusted_confidence = confidence * priority
                        
                        detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'adjusted_confidence': adjusted_confidence,
                            'bbox': (x1, y1, x2, y2),
                            'center': center,
                            'priority': priority
                        })
        
        detections.sort(key=lambda x: x['adjusted_confidence'], reverse=True)
        
        self.detection_history.append(len(detections))
        avg_detections = np.mean(self.detection_history)
        
        if len(detections) > avg_detections * 2 and len(self.detection_history) > 5:
            detections = detections[:int(avg_detections * 1.5)]
        
        return detections
    
    def draw_detections(self, frame, detections, tracked_objects=None, suspicious_objects=None):
        """Draw detections with enhancements"""
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw zones
        annotated_frame = self.zone_drawer.draw_zones(annotated_frame)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            priority = det['priority']
            
            if priority >= 0.9 and confidence > 0.7:
                color = (0, 0, 255)
                thickness = 4
            elif priority >= 0.7:
                color = (0, 165, 255)
                thickness = 3
            else:
                color = (0, 255, 255)
                thickness = 2
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            if tracked_objects:
                for obj_id, obj in tracked_objects.items():
                    if len(obj['track_history']) > 1:
                        points = obj['track_history'][-10:]
                        for i in range(len(points) - 1):
                            cv2.line(annotated_frame, points[i], points[i+1], (255, 0, 255), 2)
            
            label_lines = [
                f"{det['class'].upper()}: {confidence:.2%}",
                f"Priority: {priority:.1f}"
            ]
            
            if suspicious_objects:
                for susp in suspicious_objects:
                    if susp['object']['center'] == det['center']:
                        label_lines.append(f"Threat: {susp['threat_level']:.1%}")
                        label_lines.extend([f"- {r}" for r in susp['reasons']])
            
            y_offset = y1 - 10
            for line in label_lines:
                label_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                cv2.rectangle(annotated_frame,
                            (x1, y_offset - label_size[1] - 5),
                            (x1 + label_size[0], y_offset),
                            color, -1)
                
                cv2.putText(annotated_frame, line, (x1, y_offset - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                y_offset -= label_size[1] + 5
            
            cv2.circle(annotated_frame, det['center'], 5, (0, 255, 0), -1)
            
            bar_width = x2 - x1
            bar_height = 5
            filled_width = int(bar_width * confidence)
            cv2.rectangle(annotated_frame, (x1, y2 + 5), (x1 + filled_width, y2 + 5 + bar_height), color, -1)
            cv2.rectangle(annotated_frame, (x1, y2 + 5), (x2, y2 + 5 + bar_height), color, 2)
        
        return annotated_frame
    
    def add_alert_banner(self, frame, num_detections, max_confidence, suspicious_count=0):
        """Add alert banner"""
        if num_detections > 0:
            h, w = frame.shape[:2]
            
            if suspicious_count > 0 or max_confidence > 0.9:
                alert_color = (0, 0, 255)
                alert_text = "🚨 HIGH THREAT DETECTED"
            elif num_detections > 2 or max_confidence > 0.7:
                alert_color = (0, 165, 255)
                alert_text = "⚠️ INTRUSION DETECTED"
            else:
                alert_color = (0, 255, 255)
                alert_text = "⚡ ACTIVITY DETECTED"
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 100), alert_color, -1)
            frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
            
            cv2.putText(frame, alert_text, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
            
            details = f"Detections: {num_detections} | Confidence: {max_confidence:.1%}"
            if suspicious_count > 0:
                details += f" | Suspicious: {suspicious_count}"
            
            cv2.putText(frame, details, (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
    
    def process_frame(self, frame, alert_sound, use_tracking=True, use_motion=True, use_behavior=True):
        """Process frame with all enhancements"""
        motion_mask = None
        motion_ratio = 0
        if use_motion:
            motion_mask, motion_ratio = self.detect_motion(frame)
        
        detections = self.detect_intrusion(frame, motion_mask, use_motion)
        
        tracked_objects = None
        if use_tracking and detections:
            tracked_objects = self.track_objects(detections)
        
        suspicious_objects = []
        if use_behavior and tracked_objects:
            suspicious_objects = self.analyze_behavior(tracked_objects)
        
        annotated_frame = self.draw_detections(frame, detections, tracked_objects, suspicious_objects)
        
        alert_triggered = False
        if len(detections) > 0 or len(suspicious_objects) > 0:
            current_time = time.time()
            if current_time - self.last_alert_time > self.alert_cooldown:
                alert_triggered = True
                self.last_alert_time = current_time
                
                if alert_sound:
                    try:
                        alert_sound.play()
                    except:
                        pass
        
        max_conf = max([d['adjusted_confidence'] for d in detections], default=0)
        annotated_frame = self.add_alert_banner(
            annotated_frame, 
            len(detections), 
            max_conf,
            len(suspicious_objects)
        )
        
        return annotated_frame, detections, alert_triggered, tracked_objects, suspicious_objects, motion_ratio


def main():
    st.set_page_config(
        page_title="Enhanced AI Perimeter Detection",
        page_icon="🛡️",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {background-color: #0e1117;}
        .stAlert {
            background-color: #ff4b4b;
            color: white;
            padding: 20px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🛡️ Enhanced AI Perimeter Intrusion Detection")
        st.markdown("**Multi-Layer Video Analytics with Zone Configuration**")
    with col2:
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            st.success("🚀 GPU Acceleration ON")
        else:
            st.info("💻 Running on CPU")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Advanced Configuration")
    
    # Model selection
    st.sidebar.subheader("🤖 Model Settings")
    model_size = st.sidebar.selectbox(
        "YOLO Model Size",
        options=['n', 's', 'm', 'l', 'x'],
        index=2,
        help="n=fastest, x=most accurate"
    )
    
    confidence_threshold = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.1,
        max_value=0.95,
        value=0.5,
        step=0.05
    )
    
    iou_threshold = st.sidebar.slider(
        "IOU Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.4,
        step=0.05
    )
    
    # Enhancement features
    st.sidebar.subheader("🔧 Enhancement Features")
    use_tracking = st.sidebar.checkbox("Object Tracking", value=True)
    use_motion_filter = st.sidebar.checkbox("Motion Detection", value=True)
    use_behavior_analysis = st.sidebar.checkbox("Behavior Analysis", value=True)
    
    # Alert settings
    st.sidebar.subheader("🚨 Alert Settings")
    audio_enabled = st.sidebar.checkbox("Audio Alerts", value=True)
    show_metrics = st.sidebar.checkbox("Show Metrics", value=True)
    
    # Input type
    st.sidebar.subheader("📥 Input Source")
    input_type = st.sidebar.selectbox(
        "Select Input Type",
        ["Upload Image", "Upload Video", "Live Webcam", "Zone Configuration"]
    )
    
    # Initialize system
    if 'detector' not in st.session_state or st.session_state.get('model_size') != model_size:
        with st.spinner(f"Loading YOLOv8-{model_size} model..."):
            st.session_state.detector = PerimeterDetectionSystem(model_size)
            st.session_state.model_size = model_size
            st.session_state.alerts = []
    
    detector = st.session_state.detector
    detector.conf_threshold = confidence_threshold
    detector.iou_threshold = iou_threshold
    
    # Create alert sound
    alert_sound = None
    if audio_enabled:
        alert_sound = detector.create_alert_sound()
    
    # Zone Configuration Mode
    if input_type == "Zone Configuration":
        st.header("🎨 Interactive Zone Configuration")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Draw Restricted Zones")
            
            uploaded_frame = st.file_uploader(
                "Upload reference image to draw zones",
                type=['jpg', 'jpeg', 'png']
            )
            
            if uploaded_frame is not None:
                image = Image.open(uploaded_frame)
                frame = np.array(image)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Initialize zone points in session state
                if 'zone_points' not in st.session_state:
                    st.session_state.zone_points = []
                
                # Display frame with zones
                display_frame = detector.zone_drawer.draw_zones(frame.copy())
                display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                st.image(display_frame_rgb, use_container_width=True)
                
                st.info("💡 Click on the image above to add zone points. Add at least 3 points to create a zone.")
                
        with col2:
            st.subheader("Zone Controls")
            
            zone_name = st.text_input("Zone Name", "Restricted Zone 1")
            
            st.markdown("**Manual Point Entry:**")
            point_col1, point_col2 = st.columns(2)
            with point_col1:
                point_x = st.number_input("X coordinate", min_value=0, step=1)
            with point_col2:
                point_y = st.number_input("Y coordinate", min_value=0, step=1)
            
            if st.button("➕ Add Point"):
                detector.zone_drawer.add_point(point_x, point_y)
                st.success(f"Point added: ({point_x}, {point_y})")
                st.rerun()
            
            st.markdown("---")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ Complete Zone"):
                    if detector.zone_drawer.complete_zone(zone_name):
                        st.success(f"Zone '{zone_name}' created!")
                        st.rerun()
                    else:
                        st.error("Need at least 3 points!")
            
            with col_b:
                if st.button("🗑️ Clear Current"):
                    detector.zone_drawer.clear_current()
                    st.rerun()
            
            if st.button("🗑️ Clear All Zones"):
                detector.zone_drawer.clear_all()
                st.success("All zones cleared!")
                st.rerun()
            
            st.markdown("---")
            
            if st.button("💾 Save Zones"):
                detector.zone_drawer.save_zones()
                st.success("Zones saved to zones.json!")
            
            if st.button("📂 Load Zones"):
                if detector.zone_drawer.load_zones():
                    st.success("Zones loaded!")
                    st.rerun()
                else:
                    st.error("No zones file found!")
            
            st.markdown("---")
            st.subheader("Current Zones")
            
            if detector.zone_drawer.zones:
                for i, zone in enumerate(detector.zone_drawer.zones):
                    with st.expander(f"Zone {i+1}: {zone['name']}"):
                        st.write(f"Points: {len(zone['points'])}")
                        for j, point in enumerate(zone['points']):
                            st.text(f"  Point {j+1}: ({point[0]}, {point[1]})")
            else:
                st.info("No zones configured yet")
            
            if detector.zone_drawer.current_zone:
                st.markdown("**Current Zone (in progress):**")
                st.write(f"Points: {len(detector.zone_drawer.current_zone)}")
                for j, point in enumerate(detector.zone_drawer.current_zone):
                    st.text(f"  Point {j+1}: ({point[0]}, {point[1]})")
    
    # Image Processing Mode
    elif input_type == "Upload Image":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📹 Video Feed")
            video_placeholder = st.empty()
        
        with col2:
            st.subheader("🚨 Alert Log")
            alert_placeholder = st.empty()
            
            if show_metrics:
                st.subheader("📊 Metrics")
                metrics_placeholder = st.empty()
        
        uploaded_file = st.sidebar.file_uploader(
            "Upload an image",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            with st.spinner("🔍 Analyzing image..."):
                annotated_frame, detections, alert_triggered, tracked, suspicious, motion = detector.process_frame(
                    frame, alert_sound, use_tracking, use_motion_filter, use_behavior_analysis
                )
            
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_frame_rgb, use_container_width=True)
            
            if alert_triggered:
                timestamp = datetime.now().strftime("%H:%M:%S")
                for det in detections:
                    threat_level = "HIGH" if det['adjusted_confidence'] > 0.7 else "MEDIUM"
                    alert_msg = f"[{timestamp}] {threat_level} - {det['class'].upper()}: {det['confidence']:.2%}"
                    st.session_state.alerts.insert(0, alert_msg)
                st.error("⚠️ INTRUSION DETECTED!")
            
            with alert_placeholder.container():
                for alert in st.session_state.alerts[:10]:
                    st.warning(alert)
            
            if show_metrics and detections:
                with metrics_placeholder.container():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Detections", len(detections))
                    avg_conf = np.mean([d['confidence'] for d in detections])
                    m2.metric("Avg Confidence", f"{avg_conf:.1%}")
                    m3.metric("Suspicious", len(suspicious) if suspicious else 0)
                    m4.metric("Motion", f"{motion*100:.1f}%")
    
    # Video Processing Mode
    elif input_type == "Upload Video":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📹 Video Feed")
            video_placeholder = st.empty()
        
        with col2:
            st.subheader("🚨 Alert Log")
            alert_placeholder = st.empty()
            
            if show_metrics:
                st.subheader("📊 Metrics")
                metrics_placeholder = st.empty()
        
        uploaded_file = st.sidebar.file_uploader(
            "Upload a video",
            type=['mp4', 'avi', 'mov', 'mkv']
        )
        
        if uploaded_file is not None:
            # Save to temporary file with delete=False to control deletion
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()  # Close the file before opening with cv2
            
            # Open video
            cap = cv2.VideoCapture(tfile.name)
            
            if not cap.isOpened():
                st.error("Failed to open video file")
                try:
                    os.unlink(tfile.name)
                except:
                    pass
            else:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                st.sidebar.info(f"📹 Video: {total_frames} frames @ {fps} FPS")
                
                process_every_n = st.sidebar.slider("Process every N frames", 1, 10, 2)
                
                stop_button = st.sidebar.button("⏹️ Stop Processing")
                
                frame_count = 0
                processed_count = 0
                total_detections = 0
                detection_history = []
                
                progress_bar = st.sidebar.progress(0)
                
                try:
                    while cap.isOpened() and not stop_button:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_count += 1
                        progress = frame_count / total_frames
                        progress_bar.progress(min(progress, 1.0))
                        
                        if frame_count % process_every_n == 0:
                            processed_count += 1
                            
                            annotated_frame, detections, alert_triggered, tracked, suspicious, motion = detector.process_frame(
                                frame, alert_sound, use_tracking, use_motion_filter, use_behavior_analysis
                            )
                            
                            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            video_placeholder.image(annotated_frame_rgb, use_container_width=True)
                            
                            detection_history.append(len(detections))
                            
                            if alert_triggered:
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                for det in detections:
                                    threat_level = "HIGH" if det['adjusted_confidence'] > 0.7 else "MEDIUM"
                                    alert_msg = f"[{timestamp}] {threat_level} - {det['class'].upper()}: {det['confidence']:.2%}"
                                    st.session_state.alerts.insert(0, alert_msg)
                                    total_detections += 1
                            
                            with alert_placeholder.container():
                                if len(detections) > 0:
                                    st.error("🚨 ACTIVE DETECTION")
                                for alert in st.session_state.alerts[:10]:
                                    st.warning(alert)
                            
                            if show_metrics:
                                with metrics_placeholder.container():
                                    m1, m2, m3, m4 = st.columns(4)
                                    m1.metric("Frame", f"{frame_count}/{total_frames}")
                                    m2.metric("Detections", len(detections))
                                    m3.metric("Total Alerts", total_detections)
                                    m4.metric("Tracked", len(tracked) if tracked else 0)
                            
                            time.sleep(0.01)
                    
                except Exception as e:
                    st.error(f"Error processing video: {e}")
                
                finally:
                    # Ensure resources are released
                    cap.release()
                    progress_bar.progress(1.0)
                    
                    # Delete temporary file after releasing video capture
                    try:
                        time.sleep(0.1)  # Small delay to ensure file is released
                        os.unlink(tfile.name)
                    except PermissionError:
                        st.warning("Temporary file will be cleaned up on next run")
                    except Exception as e:
                        pass
                    
                    st.success("✅ Video processing completed!")
                    
                    # Show final statistics
                    if detection_history:
                        st.subheader("📈 Processing Summary")
                        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
                        sum_col1.metric("Total Frames", processed_count)
                        sum_col2.metric("Total Detections", sum(detection_history))
                        sum_col3.metric("Total Alerts", total_detections)
                        avg_det = np.mean(detection_history)
                        sum_col4.metric("Avg Detections", f"{avg_det:.2f}")
    
    # Webcam Mode
    elif input_type == "Live Webcam":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📹 Live Feed")
            video_placeholder = st.empty()
        
        with col2:
            st.subheader("🚨 Alert Log")
            alert_placeholder = st.empty()
            
            if show_metrics:
                st.subheader("📊 Metrics")
                metrics_placeholder = st.empty()
        
        st.sidebar.info("📷 Click 'Start Webcam' to begin")
        
        webcam_col1, webcam_col2 = st.sidebar.columns(2)
        start_webcam = webcam_col1.button("▶️ Start")
        stop_webcam = webcam_col2.button("⏹️ Stop")
        
        if start_webcam:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Failed to access webcam")
            else:
                frame_count = 0
                total_detections = 0
                detection_timeline = deque(maxlen=100)
                
                st.sidebar.success("🟢 Webcam active")
                
                try:
                    while cap.isOpened() and not stop_webcam:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("❌ Failed to read frame")
                            break
                        
                        frame_count += 1
                        
                        annotated_frame, detections, alert_triggered, tracked, suspicious, motion = detector.process_frame(
                            frame, alert_sound, use_tracking, use_motion_filter, use_behavior_analysis
                        )
                        
                        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(annotated_frame_rgb, use_container_width=True)
                        
                        detection_timeline.append(len(detections))
                        
                        if alert_triggered:
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            for det in detections:
                                threat_level = "HIGH" if det['adjusted_confidence'] > 0.7 else "MEDIUM"
                                alert_msg = f"[{timestamp}] {threat_level} - {det['class'].upper()}: {det['confidence']:.2%}"
                                st.session_state.alerts.insert(0, alert_msg)
                                total_detections += 1
                        
                        with alert_placeholder.container():
                            if len(detections) > 0:
                                st.error("🚨 LIVE INTRUSION!")
                            if suspicious:
                                st.warning(f"⚠️ {len(suspicious)} Suspicious Behavior(s)")
                            
                            for alert in st.session_state.alerts[:10]:
                                st.warning(alert)
                        
                        if show_metrics:
                            with metrics_placeholder.container():
                                m1, m2, m3, m4 = st.columns(4)
                                m1.metric("Frame", frame_count)
                                m2.metric("Detections", len(detections))
                                m3.metric("Total Alerts", total_detections)
                                m4.metric("Motion", f"{motion*100:.1f}%")
                                
                                if tracked:
                                    st.markdown("**Active Tracks:**")
                                    for obj_id, obj in list(tracked.items())[:5]:
                                        dwell = time.time() - obj['first_seen']
                                        st.text(f"ID {obj_id}: {obj['class']} ({dwell:.1f}s)")
                        
                        time.sleep(0.03)
                
                except Exception as e:
                    st.error(f"Error: {e}")
                
                finally:
                    cap.release()
                    st.sidebar.info("🔴 Webcam stopped")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Performance Tips")
    st.sidebar.markdown(f"""
    **Current Setup:**
    - Model: YOLOv8-{model_size}
    - Device: {'GPU' if torch.cuda.is_available() else 'CPU'}
    - Tracking: {'ON' if use_tracking else 'OFF'}
    - Motion: {'ON' if use_motion_filter else 'OFF'}
    - Behavior: {'ON' if use_behavior_analysis else 'OFF'}
    - Zones: {len(detector.zone_drawer.zones)} configured
    
    **For Best Accuracy:**
    - Use YOLOv8-l or x
    - Enable all features
    - Confidence: 0.4-0.6
    - Configure restricted zones
    
    **For Speed:**
    - Use YOLOv8-n or s
    - Process every 2-3 frames
    - Disable behavior analysis
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Features")
    st.sidebar.info("""
    ✅ Multi-layer AI detection
    ✅ Object tracking
    ✅ Motion filtering
    ✅ Behavior analysis
    ✅ Interactive zone drawing
    ✅ Real-time alerts
    ✅ Audio notifications
    ✅ Threat level scoring
    """)


if __name__ == "__main__":
    main()