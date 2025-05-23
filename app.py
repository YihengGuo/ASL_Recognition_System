import streamlit as st
from PIL import Image, ImageOps, ExifTags
import numpy as np
import tensorflow as tf
import os

# ✅ 必须最前面设置页面
st.set_page_config(page_title="Sign Language Recognition System", page_icon="🤟")

# ✅ 修正上传图像的方向
def correct_image_orientation(image):
    try:
        for orientation in ExifTags.TAGS:
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except Exception:
        pass
    return image

# ✅ 图像预处理，匹配模型输入 (64, 64, 3)
def preprocess_image(image: Image.Image):
    image = correct_image_orientation(image)
    image = image.convert("RGB")
    image = ImageOps.fit(image, (64, 64))
    img_array = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# ✅ 加载模型，只加载一次
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("asl_cnn_model.h5")

model = load_model()

# ✅ 类别标签
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

# ✅ 页面展示
st.title("✋ Sign Language Recognition System")
st.markdown("Please upload a sign language picture and the system will predict the letter it represents.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Upload an image", use_container_width=True)

    with st.spinner("Identifying..."):
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)[0]
        top_index = np.argmax(prediction)
        pred_class = class_names[top_index]
        confidence = prediction[top_index] * 100

    st.success(f"✅ Identification results：**{pred_class}**（Confidence：{confidence:.2f}%）")
