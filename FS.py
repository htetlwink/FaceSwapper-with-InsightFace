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
import io
from PIL import Image

print("Libaries Import Successful !")

import onnxruntime
#print(onnxruntime.__version__)

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

    image = Image.fromarray(img1_rgb)
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="JPEG")
    img_bytes = img_buffer.getvalue()

    st.download_button(
    label="Download Processed Image1",
    data=img_bytes,
    file_name="processed_image1.jpg",  # Change file name and extension as needed
    mime="image/jpeg"  # Change MIME type if you use a different image format
)

    image2 = Image.fromarray(img2_rgb)
    img_buffer2 = io.BytesIO()
    image2.save(img_buffer2, format="JPEG")
    img_bytes2 = img_buffer2.getvalue()

    st.download_button(
    label="Download Processed Image2",
    data=img_bytes2,
    file_name="processed_image2.jpg",  # Change file name and extension as needed
    mime="image/jpeg"  # Change MIME type if you use a different image format
)
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

if SourcePhoto is not None and YourPhoto is not None:
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

    swapper = insightface.model_zoo.get_model("inswapper_128.onnx",
                                                download=False,
                                                download_zip=False)


    swappedp1, swappedp2 = FaceSwap1212(SourceTemp,YourTemp, app, swapper)
    st.image(swappedp1)
    st.image(swappedp2)