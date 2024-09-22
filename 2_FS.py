import streamlit as st
import numpy as np
import os
import glob
import cv2
import tempfile
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis
from insightface.app import ins_get_image
print("Libaries Import Successful !")

#st.set_page_config(page_title="FaceSwapper",page_icon="💋")
st.markdown("<h1 style='text-align: center;'>Face Swapper</h1>", unsafe_allow_html=True)
st.markdown("---")

#st.write(number)
#print(type(number))
col1,col2 = st.columns(2)

with col1:
  st.markdown("<h5 style='text-align: center;'>Upload Source Photo</h5>", unsafe_allow_html=True)
  SourcePhoto = st.file_uploader("Upload Source Photo", type=["jpg", "png", "jpeg"])
  if SourcePhoto is not None:
     st.image(SourcePhoto)
  st.markdown("---")


with col2:
   
  st.markdown("<h5 style='text-align: center;'>Upload Your Photo</h5>", unsafe_allow_html=True)
  YourPhoto = st.file_uploader("Upload Your Photo", type=["jpg","png","jpeg"])
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
    print("Face Detection Model Loaded...")

    with col1:
      #For First Photo
      img = cv2.imread(SourceTemp)
      faces = app.get(img)
      num_faces = len(faces)
      print(num_faces)
      fig, axs = plt.subplots(1, num_faces, figsize=(12, 5))

      # Example labels for each face (customize as needed)
      labels = [f"Face {i+1}" for i in range(num_faces)]

      for i, face in enumerate(faces):
          bbox = face["bbox"]
          bbox = [int(b) for b in bbox]

          cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

          # Plot the face
          if num_faces == 1:
              axs.imshow(img[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
              axs.axis("off")
              axs.text(0.5, -0.1, labels[i], size=12, ha="center", transform=axs.transAxes)  # Add label
          else:
              axs[i].imshow(img[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
              axs[i].axis("off")
              axs[i].text(0.5, -0.1, labels[i], size=12, ha="center", transform=axs[i].transAxes)  # Add label

      st.pyplot(fig)

    with col2:
      #For Second Photo
      simg = cv2.imread(YourTemp)
      sfaces = app.get(simg)
      num_sfaces = len(sfaces)

      # Create subplots for displaying detected faces
      fig, axs = plt.subplots(1, num_sfaces if num_sfaces > 1 else 1, figsize=(12, 5))

      # Example labels for each face (customize as needed)
      labels = [f"Face {i+1}" for i in range(num_sfaces)]

      for i, face in enumerate(sfaces):
          bbox = face["bbox"]
          bbox = [int(b) for b in bbox]

          cv2.rectangle(simg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

          # Crop and display the detected face
          if num_sfaces == 1:
              axs.imshow(simg[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
              axs.axis("off")
              axs.text(0.5, -0.1, labels[i], size=12, ha="center", transform=axs.transAxes)  # Add label
          else:
              axs[i].imshow(simg[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
              axs[i].axis("off")
              axs[i].text(0.5, -0.1, labels[i], size=12, ha="center", transform=axs[i].transAxes)  # Add label

      st.pyplot(fig)    
    #Ask Your to Choose the Photo
    st.markdown("---")
    user_input = st.text_input("Type your string here:")
    face1, face2 = user_input.split(",")  
    
    SwapBtn = st.button("FaceSwap")

    if SwapBtn:
      swapper = insightface.model_zoo.get_model("inswapper_128.onnx",
                                          download=False,
                                          download_zip=False)
      print("Face Swapper Model Loaded...")
      face1final = int(face1)
      face2final = int(face2)
      photofinal = SourceTemp.copy()
      photofinal = swapper.get(photofinal,faces[face1final],sfaces[face2final], paste_back=True)
      photofinal = photofinal[:,:,::-1]
      st.image(photofinal, use_column_width=True)

