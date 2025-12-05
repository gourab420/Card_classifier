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
    st.header("Live Camera Detection")
    
    detection_mode = st.radio(
        "Choose detection mode:",
        ["📸 Camera Capture (Recommended)", "🎥 Live Webcam (Advanced)"],
        horizontal=True
    )
    
    if detection_mode == "📸 Camera Capture (Recommended)":
        st.info("Take photos with your camera for instant detection - works on all devices!")
        
        camera_photo = st.camera_input("Take a photo")
        
        if camera_photo is not None:
            image = Image.open(camera_photo)
            img_array = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original")
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Detection")
                with st.spinner("Analyzing..."):
                    results = model.predict(img_array, device=device_id, conf=conf_threshold, verbose=False)
                    annotated_img = results[0].plot()
                    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    
                    st.image(annotated_img_rgb, use_container_width=True)
                    
                    if len(results[0].boxes) > 0:
                        st.success(f"Found {len(results[0].boxes)} object(s):")
                        for box in results[0].boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            st.write(f"• **{model.names[cls]}** - {conf:.1%} confidence")
                    else:
                        st.info("No objects detected. Try another angle!")
    
    else:  # Live Webcam
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
            import av
            
            st.warning("⚠️ Requires good internet connection and modern browser (Chrome/Edge)")
            
            rtc_configuration = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
            
            class VideoProcessor:
                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    results = model(img, conf=conf_threshold, device=device_id, verbose=False)
                    return av.VideoFrame.from_ndarray(results[0].plot(), format="bgr24")
            
            webrtc_streamer(
                key="detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_configuration,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=VideoProcessor,
                async_processing=True,
            )
            
        except ImportError:
            st.error("Install: `pip install streamlit-webrtc aiortc`")
        
# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
    <p>🃏 YOLO Card Classifier | Powered by Ultralytics YOLOv11</p>
    </div>
    """, unsafe_allow_html=True)




