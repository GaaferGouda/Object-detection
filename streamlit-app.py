import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import os

# -----------------------------
# Sidebar Configuration
# -----------------------------
st.sidebar.title("Settings 🛠️")

model_choice = st.sidebar.selectbox(
    "Select YOLO Model",
    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.1, 1.0, 0.25, 0.05
)

upload_option = st.sidebar.radio(
    "Choose Input Type",
    ["Upload Image/Video", "Use Webcam"]
)

st.sidebar.markdown("— Developed by **Gaafer Gouda**")

# -----------------------------
# Load YOLO model
# -----------------------------
@st.cache_resource
def load_yolo(model_path):
    return YOLO(model_path)

model = load_yolo(model_choice)

# -----------------------------
# Helper: Run YOLO on Frames
# -----------------------------
def process_frame(frame, conf):
    results = model(frame, conf=conf)
    annotated = results[0].plot()
    classes = results[0].names
    detected_objects = [classes[int(cls)] for cls in results[0].boxes.cls]
    return annotated, detected_objects

# -----------------------------
# Webcam Transformer
# -----------------------------
class YOLOTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.conf = confidence_threshold

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb, conf=self.conf)
        annotated = results[0].plot()
        return cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

# -----------------------------
# App UI
# -----------------------------
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>YOLO Object Detection App</h1>", unsafe_allow_html=True)
st.write("Upload an image/video or use your webcam for real-time object detection using Ultralytics YOLO.")
st.markdown("---")

# -----------------------------
# Tabs for Upload / Webcam
# -----------------------------
tab1, tab2 = st.tabs(["Upload", "Webcam"])

# -----------------------------
# Upload Tab
# -----------------------------
with tab1:
    uploaded_file = st.file_uploader(
        "Upload an image or video file",
        type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv"]
    )

    if uploaded_file is not None:
        file_type = uploaded_file.type

        # -----------------------------
        # Image
        # -----------------------------
        if file_type.startswith("image"):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                processed, detected_objects = process_frame(img_rgb, confidence_threshold)

                st.subheader("Processed Image")
                st.image(processed, channels="RGB")
                st.info(f"Detected objects: {detected_objects}")
            else:
                st.error("Failed to read the image. Please try another file.")

        # -----------------------------
        # Video
        # -----------------------------
        elif file_type.startswith("video"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            vid_path = tfile.name

            st.video(vid_path)
            run = st.button("Run Detection")

            if run:
                st.subheader("Processing Video...")
                cap = cv2.VideoCapture(vid_path)

                # Get video properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Temporary file for processed video
                processed_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(processed_path, fourcc, fps, (width, height))

                frame_container = st.empty()
                progress_bar = st.progress(0)

                with st.spinner("Processing video..."):
                    for i in range(frame_count):
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        processed, detected_objects = process_frame(frame_rgb, confidence_threshold)
                        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

                        col1, col2 = st.columns(2)
                        col1.image(frame, caption="Original", channels="BGR")
                        col2.image(processed_bgr, caption=f"Processed | Objects: {detected_objects}", channels="BGR")

                        out.write(processed_bgr)
                        progress_bar.progress((i+1)/frame_count)

                cap.release()
                out.release()
                st.success("Video processing completed.")

                # Download link
                with open(processed_path, "rb") as f:
                    st.download_button(
                        label="Download Processed Video",
                        data=f,
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )
                os.remove(vid_path)

# -----------------------------
# Webcam Tab
# -----------------------------
with tab2:
    st.subheader("Webcam Detection")
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="yolo-webcam",
        rtc_configuration=RTC_CONFIGURATION,
        video_transformer_factory=YOLOTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True
    )
