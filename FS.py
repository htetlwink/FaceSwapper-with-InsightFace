import streamlit as st
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import tempfile
import insightface
from insightface.app import FaceAnalysis
from insightface.app import ins_get_image

print("Libaries Import Successful !")

def FaceSwap1212(img1_fn, img2_fn, app, swapper, plot_before=True, plot_after=True):
  img1 = cv2.imread(img1_fn)
  img2 = cv2.imread(img2_fn)

  if plot_before:
    fig, axs = plt.subplots(1,2, figsize=(2, 2))
    axs[0].imshow(img1[:,:,::-1])
    axs[0].axis("off")
    axs[1].imshow(img2[:,:,::-1])
    axs[1].axis("off")
    plt.show()

  face1 = app.get(img1)[0]
  face2 = app.get(img2)[0]

  img1_ = img1.copy()
  img2_ = img2.copy()

  if plot_after:
    img1_ = swapper.get(img1_, face1, face2, paste_back=True)
    img2_ = swapper.get(img2_, face2, face1, paste_back=True)
    img1_rgb = cv2.cvtColor(img1_, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2_, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(2, 2))
    plt.imshow(img1_rgb[:, :, ::-1])
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(20, 20))
    plt.imshow(img2_rgb[:, :, ::-1])
    plt.axis("off")
    plt.show()

  return img1_rgb, img2_rgb

st.set_page_config(page_title="FaceSwapper",page_icon="💋")
st.markdown("<h1 style='text-align: center;'>Face Swapper</h1>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<h5 style='text-align: center;'>Upload Source Photo</h5>", unsafe_allow_html=True)
SourcePhoto = st.file_uploader("Upload Source Photo", type=["jpg", "png", "jpeg"])
st.markdown("---")
st.markdown("<h5 style='text-align: center;'>Upload Your Photo</h5>", unsafe_allow_html=True)
YourPhoto = st.file_uploader("Upload Your Photo", type=["jpg","png","jpeg"])
st.markdown("---")

swapbtm = st.button("Swap Faces")

if swapbtm:

  with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    temp_file.write(SourcePhoto.read())
    SourceTemp = temp_file.name

  with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    temp_file.write(YourPhoto.read())
    YourTemp = temp_file.name

  app = FaceAnalysis(name='buffalo_l')
  app.prepare(ctx_id=0, det_size=(640, 640))
  print("Face Detection Model Loaded...")

  swapper = insightface.model_zoo.get_model("/content/drive/MyDrive/InsightFace/inswapper_128.onnx",
                                          download=False,
                                          download_zip=False)
  print("Face Swapper Moel Loaded...")
  swapped = FaceSwap1212(SourceTemp,YourTemp, app, swapper)
  st.image(swapped)