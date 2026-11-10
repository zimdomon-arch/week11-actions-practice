"""수업에서 읽는 핵심 코드. 실행: python -m streamlit run app_minimal.py"""
from pathlib import Path
from threading import Lock

from PIL import Image, ImageOps
from ultralytics import YOLO
import streamlit as st

MODEL_PATH = Path(__file__).parent / "models" / "yolo11n.pt"
CONFIDENCE = 0.25  # 실습: 0.80으로 바꾸고 같은 이미지를 비교하세요.

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH)), Lock()

st.set_page_config(page_title="AI 이미지 객체 탐지", layout="wide")
st.title("AI 이미지 객체 탐지")
st.write("이미지를 선택한 뒤 탐지 시작을 눌러 결과를 확인하세요.")
uploaded = st.file_uploader("이미지 선택", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
    left, right = st.columns(2)
    left.image(image, caption="원본 이미지", width="stretch")

    if st.button("탐지 시작")
        if not MODEL_PATH.exists():
            st.error("터미널에서 python prepare_model.py를 먼저 실행하세요.")
            st.stop()
        model, model_lock = load_model()
        with st.spinner("CPU에서 객체를 찾고 있습니다..."):
            with model_lock:
                result = model.predict(
                    image, conf=CONFIDENCE, device="cpu", verbose=False
                )[0]
        annotated_rgb = result.plot()[:, :, ::-1]
        right.image(annotated_rgb, caption="탐지 결과", width="stretch")
        st.caption("사각형은 위치, 숫자는 모델의 confidence 점수입니다.")
else:
    st.info("assets/astronaut.png 또는 직접 준비한 이미지를 올려보세요.")
    st.stop()