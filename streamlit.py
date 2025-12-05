import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import tempfile
import imageio

# Set page config
st.set_page_config(
    page_title="Card Classifier",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #1f77b4;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🃏 YOLO Card Classifier")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    # model_path = "runs/detect/train8/weights/best.pt"
    model_path = 'best.pt'
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}")
        st.info("Please train the model first using: `python card_classifier.py`")
        return None
    return YOLO(model_path)

model = load_model()


if model is None:
    st.stop()

# Sidebar
st.sidebar.title("⚙️ Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Lower values detect more objects but with lower confidence"
)

# Use CPU for all inference
device_id = "cpu"

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📷 Image", "🎥 Video", "📁 Folder", "📹 Webcam"])

# Tab 1: Image Upload
with tab1:
    st.header("Test on Image")
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Supported formats: JPG, JPEG, PNG, BMP"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        # Run inference automatically
        with st.spinner("Running inference..."):
            # Convert PIL to numpy array
            image_np = np.array(image)
            
            # Run prediction
            results = model.predict(
                source=image_np,
                conf=conf_threshold,
                device=device_id,
                verbose=False
            )
            
            # Get annotated image
            annotated_image = results[0].plot()
            annotated_pil = Image.fromarray(annotated_image)
            
            with col2:
                st.subheader("Detection Results")
                st.image(annotated_pil, use_column_width=True)
            
            # Display detections
            st.subheader("📊 Detections")
            if len(results[0].boxes) > 0:
                detections_data = []
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    class_name = results[0].names[class_id]
                    confidence = float(box.conf[0])
                    detections_data.append({
                        "Class": class_name,
                        "Confidence": f"{confidence:.2%}",
                        "Box": f"({int(box.xyxy[0][0])}, {int(box.xyxy[0][1])}, {int(box.xyxy[0][2])}, {int(box.xyxy[0][3])})"
                    })
                st.dataframe(detections_data, use_container_width=True)
            else:
                st.info("No objects detected")

# Tab 2: Video Upload

with tab2:
    st.header("Test on Video")
    video_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if video_file is not None:
        st.video(video_file)

        with st.spinner("Processing video..."):
            # Save uploaded video temporarily
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_input.write(video_file.getbuffer())
            temp_input.close()

            # Read input video
            cap = cv2.VideoCapture(temp_input.name)
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Collect processed frames
            processed_frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.predict(frame, device=device_id, conf=conf_threshold, verbose=False)
                processed_frame = results[0].plot()
                # Convert BGR to RGB for imageio
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                processed_frames.append(processed_frame_rgb)

            cap.release()

            # Write video using imageio (includes ffmpeg)
            output_dir = "temp_outputs"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "processed_video_output.mp4")
            
            imageio.mimsave(
                output_path,
                processed_frames,
                fps=fps,
                codec='libx264',
                pixelformat='yuv420p'
            )

            st.subheader("🎬 Processed Video")
            col1, col2, col3 = st.columns([1, 2, 1])
 
            with col2:
                st.video(output_path)
            
            # Download button
            with col2:
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=f.read(),
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )
            
            # Clean up
            os.remove(temp_input.name)



# Tab 3: Folder of Images

with tab3:
    st.header("Test on Folder")
    folder_path = st.text_input(
        "Enter folder path",
        value="roboflow_dataset/test/images",
        help="Path to folder containing images"
    )
    
    if st.button("🔍 Run Detection on Folder", key="folder_detect"):
        if os.path.isdir(folder_path):
            with st.spinner("Processing folder... This may take a while"):
                # Run prediction
                results = model.predict(
                    source=folder_path,
                    conf=conf_threshold,
                    save=True,
                    device=device_id,
                    verbose=False
                )
                
                st.success(f"✅ Processed {len(results)} images!")
                st.info("Results saved to `runs/detect/predict`")
                
                # Display summary
                total_detections = sum(len(r.boxes) for r in results)
                st.metric("Total Detections", total_detections)
        else:
            st.error(f"❌ Folder not found: {folder_path}")

# Tab 4: Webcam
with tab4:
    st.header("Live Webcam Detection")
    
    # Check if running on Streamlit Cloud
    is_cloud = os.getenv('STREAMLIT_SERVER_HEADLESS') == 'true'
    
    if is_cloud:
        st.warning("⚠️ Webcam is not supported on Streamlit Cloud")
        st.info("""
        Webcam detection only works when running locally. 
        
        To use webcam locally:
        1. Clone the repository
        2. Install dependencies: `pip install -r requirements.txt`
        3. Run: `streamlit run streamlit.py`
        4. Open the Webcam tab
        """)
    else:
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
            import av
            
            rtc_configuration = RTCConfiguration(
                {
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]},
                        {"urls": ["stun:stun2.l.google.com:19302"]},
                        {"urls": ["stun:stun3.l.google.com:19302"]},
                        {"urls": ["stun:stun4.l.google.com:19302"]},
                    ]
                }
            )
            
            class VideoProcessor:
                def __init__(self):
                    self.conf = conf_threshold
                
                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    
                    # Resize frame to reduce latency (416x416 is standard YOLO input)
                    h, w = img.shape[:2]
                    target_size = 416
                    scale = min(target_size / h, target_size / w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    img_resized = cv2.resize(img, (new_w, new_h))
                    
                    # Run inference on resized frame
                    results = model(img_resized, conf=self.conf, device=device_id, verbose=False)
                    annotated_frame = results[0].plot()
                    
                    # Resize back to original size for display
                    annotated_frame = cv2.resize(annotated_frame, (w, h))
                    
                    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
            
            webrtc_ctx = webrtc_streamer(
                key="card-classifier-webcam",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_configuration,
                media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
                async_processing=True,
                video_processor_factory=VideoProcessor,
            )
            
            if webrtc_ctx.state.playing:
                st.info("📹 Webcam is active. Detection running in real-time.")
            else:
                st.info("Click 'Start' to begin webcam detection")
                
        except ImportError:
            st.warning("⚠️ streamlit-webrtc not installed")
            st.info("Install it with: `pip install streamlit-webrtc`")
            
            # Fallback: Simple OpenCV approach
            st.subheader("Alternative: Local Webcam Testing")
            st.code("""
# Run this in terminal instead:
python test.py
# Then select option 3 for webcam
            """)
# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
    <p>🃏 YOLO Card Classifier | Powered by Ultralytics YOLOv11</p>
    </div>
    """, unsafe_allow_html=True)
