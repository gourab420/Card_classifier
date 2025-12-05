import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import tempfile
import imageio

#set page config
st.set_page_config(
    page_title="Card Classifier",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded"
)

#custom CSS
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

#title
st.title("YOLO Card Classifier")
st.markdown("---")

#load the model
@st.cache_resource #->for faster load model
def load_model():
    model_path = 'best.pt'
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        st.info("train the model first ")
        return None
    return YOLO(model_path)

model=load_model() #call the function and store the cache model in model veriable

if model is None:
    st.stop()

#sidebar
st.sidebar.title("Settings")
conf_threshold = st.sidebar.slider(  #-> make a slide bar that take confidence threshold input from user to revel full potential of the model accuracy
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="lower values detect more objects but with lower confidence (increase false positives)"
)

#use CPU because in local i initially use gpu but when i try to deploy it in cloud then findout in cloud(streamlit.io) it only support cpu not gpu
device_id="cpu"

#main tabs
tab1,tab2= st.tabs(["📷 Image", "🎥 Video"])

#Tab 1: single to multiple image upload + live take photo using camera
with tab1:
    st.header("Test on Image(s)")

    # =============================
    # ROW 1 → IMAGE UPLOAD SECTION
    # =============================
    st.subheader("📤 Upload Image(s)")

    uploaded_files = st.file_uploader( #-> choose image from local drive 
        "Upload one or more images",
        type=["jpg","jpeg","png","bmp"],
        accept_multiple_files=True,
        key="upload_img_multiple_tab1"
    )

    if uploaded_files:
        st.info(f"📦 {len(uploaded_files)} file(s) uploaded")

        #loop for all uploaded images to get access one by one
        for idx, uploaded_file in enumerate(uploaded_files):
            st.write(f"### 🖼 Image {idx + 1}: {uploaded_file.name}")

            image=Image.open(uploaded_file)#->store single image
            col1,col2 = st.columns(2)#-> two colum one for orginal image and another for predicted image

            #original image
            with col1:
                st.subheader("Original Image")
                st.image(image,use_column_width=True)

            #detection though the trained model 
            with st.spinner("Running inference..."):
                image_np = np.array(image) #->convert the image to numpy array for prediction
                results = model.predict( #-> run the pretained model for the image
                    source=image_np, #-> here image source is numpy array not raw image
                    conf=conf_threshold,
                    device=device_id,
                    verbose=False
                )
                annotated = results[0].plot() #->detect the object from the image and draw the bounding box 
                annotated_pil = Image.fromarray(annotated) #->convert anoteated numpy array to real image

            #predicted image
            with col2:
                st.subheader("Detection Result")
                st.image(annotated_pil, use_column_width=True)

            #show detection table
            st.subheader("📊 Detections") #-> show the value get from the images
            if len(results[0].boxes) > 0:
                det_list = []
                for box in results[0].boxes: 
                    cls = int(box.cls[0]) #->get the class id(heart,dimond etc) from the predicted image
                    conf = float(box.conf[0]) #->get the confidence score
                    det_list.append({
                        "Class": results[0].names[cls],
                        "Confidence": f"{conf:.2%}",
                        "Box": f"({int(box.xyxy[0][0])}, {int(box.xyxy[0][1])}," #->get the actual 4 points(coordinates) of the bounding box
                               f"{int(box.xyxy[0][2])}, {int(box.xyxy[0][3])})"
                    })
                st.dataframe(det_list,use_container_width=True)#-> display the detected object in a table
                
            else:#-> if detection box less than 1 then the image does not contain any detected object
                st.info("No objects detected")
            st.write("---")

    st.write("")

    # ROW 2 → LIVE CAMERA SECTION
    st.subheader("📸 Live Camera Detection")

    #camera session state
    if "cam_active" not in st.session_state:
        st.session_state.cam_active = False

    #start/stop buttons for camera
    colA,colB = st.columns(2) #-> in column-1 start button and column-2 stop button
    with colA:
        if st.button("▶️ Start Camera", key="start_cam_button_tab1"):
            st.session_state.cam_active = True

    with colB:
        if st.button("⛔ Stop Camera", key="stop_cam_button_tab1"):
            st.session_state.cam_active = False

    st.write("")

    #show camera if active or already granted the permission
    if st.session_state.cam_active:
        camera_photo = st.camera_input("Take a photo", key="live_cam_input_tab1") #-> take photo from the user

        if camera_photo is not None:
            image = Image.open(camera_photo)
            
            '''same as row-1 the main difference is in the row-1 user give image from local stroage and here user
               give the input though the camera'''
               
            img_array = np.array(image)
            col1, col2 = st.columns(2)
            #show original live image
            with col1:
                st.subheader("Original (Live Photo)")
                st.image(image, use_column_width=True)

            #detection though the trained model 
            with col2:
                st.subheader("Detection Result")
                with st.spinner("Detecting..."):
                    results = model.predict(
                        img_array,
                        device=device_id,
                        conf=conf_threshold,
                        verbose=False
                    )
                    annotated_img = results[0].plot()
                    annotated_pil_img = Image.fromarray(annotated_img)
                    st.image(annotated_pil_img, use_column_width=True)

            #show detection list
            st.subheader("📊 Live Camera Detections")
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    st.write(f"- {model.names[cls]} → {conf:.2%}")
            else:
                st.info("No objects detected")
    else:
        st.warning("Live Camera is Off. Click **Start Camera** to begin.")


#Tab 2: Video Upload
with tab2:
    st.header("Test on Video")
    video_file = st.file_uploader( #-> upload video from local stroage
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if video_file is not None:
        st.video(video_file) #-> display the orginal video which is uploaded

        with st.spinner("Processing video..."):
            #save uploaded video temporarily
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_input.write(video_file.getbuffer())
            temp_input.close()

            #read the input video
            cap = cv2.VideoCapture(temp_input.name) #create a videocapture object to read frames from the video file 
            fps = cap.get(cv2.CAP_PROP_FPS) #get the frame per secound of the uploaded video

            #collect processed frames
            processed_frames=[]
            
            while True:
                ret,frame = cap.read()
                if not ret:
                    break
                results = model.predict(frame, device=device_id, conf=conf_threshold, verbose=False) #->run the model on current frame
                processed_frame = results[0].plot()#draw the bounding box and lebel
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)#->convert BGR to RGB for proper display
                processed_frames.append(processed_frame_rgb)#->add processed frame to the list

            cap.release() #release the video captur object

            #save processed frames as a new video using imageio
            output_dir = "temp_outputs"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "processed_video_output.mp4")
            
            imageio.mimsave(
                output_path,
                processed_frames,
                fps=fps,           #->keep same orginal video fps
                codec='libx264',
                pixelformat='yuv420p'
            )
            #display the predicted video
            st.subheader("🎬 Processed Video")
            col1, col2, col3 = st.columns([1, 2, 1])
 
            with col2:
                st.video(output_path)
            
            #download button for new predicted video
            with col2:
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=f.read(),
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )
            
            #clean up
            os.remove(temp_input.name)
            
   
# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
    <p>🃏 YOLO Card Classifier | Powered by Ultralytics YOLOv11</p>
    </div>
    """, unsafe_allow_html=True)





