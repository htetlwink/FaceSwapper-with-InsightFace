import streamlit as st
import numpy as np
import os
import tempfile
import cv2
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis

st.markdown("<h1 style='text-align: center;'>Face Swapper</h1>", unsafe_allow_html=True)
st.markdown("---")

# Initialize session state for face numbers
if 'face1' not in st.session_state:
    st.session_state.face1 = 0
if 'face2' not in st.session_state:
    st.session_state.face2 = 0
if 'faces_detected' not in st.session_state:
    st.session_state.faces_detected = False

# File upload columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h5 style='text-align: center;'>Upload Source Photo</h5>", unsafe_allow_html=True)
    SourcePhoto = st.file_uploader("Upload Source Photo", type=["jpg", "png", "jpeg"])
    if SourcePhoto is not None:
        st.image(SourcePhoto)
st.markdown("---")

with col2:
    st.markdown("<h5 style='text-align: center;'>Upload Your Photo</h5>", unsafe_allow_html=True)
    YourPhoto = st.file_uploader("Upload Your Photo", type=["jpg", "png", "jpeg"])
    if YourPhoto is not None:
        st.image(YourPhoto)
st.markdown("---")

if SourcePhoto is not None and YourPhoto is not None:
    detectBtm = st.button("Detect Faces")

    if detectBtm:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(SourcePhoto.read())
            SourceTemp = temp_file.name

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(YourPhoto.read())
            YourTemp = temp_file.name

        app = FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=0, det_size=(640, 640))

        # Detect faces in Source Photo
        img = cv2.imread(SourceTemp)
        st.session_state.faces = app.get(img)
        st.session_state.num_faces = len(st.session_state.faces)

        # Display detected faces
        if st.session_state.num_faces > 0:
            fig, axs = plt.subplots(1, st.session_state.num_faces, figsize=(12, 5))
            if st.session_state.num_faces == 1:
                axs = [axs]  # Convert single Axes to list

            for i, face in enumerate(st.session_state.faces):
                bbox = face["bbox"].astype(int)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                axs[i].imshow(img[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
                axs[i].axis("off")
                axs[i].set_title(f"Face {i+1}")

            st.pyplot(fig)

        # Detect faces in Your Photo
        simg = cv2.imread(YourTemp)
        st.session_state.sfaces = app.get(simg)
        st.session_state.num_sfaces = len(st.session_state.sfaces)

        # Display detected faces
        if st.session_state.num_sfaces > 0:
            fig, axs = plt.subplots(1, st.session_state.num_sfaces, figsize=(12, 5))
            if st.session_state.num_sfaces == 1:
                axs = [axs]  # Convert to a list for consistency

            for i, face in enumerate(st.session_state.sfaces):
                bbox = face["bbox"].astype(int)
                cv2.rectangle(simg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                axs[i].imshow(simg[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
                axs[i].axis("off")
                axs[i].set_title(f"Face {i+1}")

            st.pyplot(fig)


        st.session_state.faces_detected = True  # Mark faces as detected

if st.session_state.faces_detected:
    st.markdown("---")
    with col1:
        st.session_state.face1 = st.number_input("Choose Face Number to swap", min_value=0, max_value=st.session_state.num_faces-1, value=st.session_state.face1)
    
    with col2:
        st.session_state.face2 = st.number_input("Choose Face Number to swap", min_value=0, max_value=st.session_state.num_sfaces-1, value=st.session_state.face2)

    swapper = insightface.model_zoo.get_model("inswapper_128.onnx", download=False)

    if st.button("FaceSwap"):
        face1final = int(st.session_state.face1)
        face2final = int(st.session_state.face2)
        photofinal = swapper.get(SourceTemp, st.session_state.faces[face1final], st.session_state.sfaces[face2final], paste_back=True)
        photofinal = photofinal[:, :, ::-1]
        st.image(photofinal, use_column_width=True)
