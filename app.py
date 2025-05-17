import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import os

# ✅ 这句必须放在所有 st.xxx() 之前
st.set_page_config(page_title="手语识别系统", page_icon="🤟")

# ===== 模型加载函数 =====
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("asl_cnn_model.h5")

model = load_model()

# 识别类别
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

# 图片预处理函数
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")                            # 保证是彩色图像
    image = ImageOps.fit(image, (64, 64))                   # 改为 64×64（不是 224×224）
    img_array = np.array(image).astype(np.float32) / 255.0  # 归一化
    return np.expand_dims(img_array, axis=0)                # 变成 (1, 64, 64, 3)

# ✅ 页面 UI
st.title("✋ 美国手语识别系统")
st.markdown("请上传一张手语图像，系统将预测代表的字符。")

uploaded_file = st.file_uploader("📤 上传图像", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="上传图像", use_column_width=True)

    with st.spinner("识别中..."):
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)[0]
        pred_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    st.success(f"✅ 识别结果：**{pred_class}**（置信度：{confidence:.2f}%）")
