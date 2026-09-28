import os
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import tensorflow as tf


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Lung Cancer CT Classification",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown("""
<style>

    /* Main page */
    .main {
        padding-top: 1rem;
    }

    /* Header */
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #101827 0%,
            #172554 100%
        );
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: white;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        line-height: 1.6;
    }

    /* Section headings */
    .section-title {
        font-size: 1.5rem;
        font-weight: 750;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    /* Cards */
    .info-card {
        padding: 1.2rem;
        border-radius: 14px;
        background: rgba(30,41,59,0.65);
        border: 1px solid rgba(148,163,184,0.15);
        min-height: 120px;
    }

    .info-card-title {
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .info-card-text {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Prediction card */
    .prediction-card {
        padding: 1.5rem;
        border-radius: 16px;
        background: linear-gradient(
            135deg,
            rgba(15,118,110,0.18),
            rgba(30,41,59,0.75)
        );
        border: 1px solid rgba(45,212,191,0.25);
        text-align: center;
        margin: 1rem 0;
    }

    .prediction-label {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
    }

    .prediction-class {
        font-size: 2rem;
        font-weight: 800;
        color: #5eead4;
    }

    /* Methodology */
    .pipeline {
        padding: 1rem;
        border-radius: 14px;
        background: rgba(15,23,42,0.7);
        border: 1px solid rgba(148,163,184,0.15);
        text-align: center;
        font-weight: 650;
        line-height: 2.2;
    }

    .pipeline-arrow {
        color: #64748b;
        margin: 0 0.4rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        padding: 2rem 0 1rem;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

CLASS_NAMES = [
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Normal",
    "Squamous Cell Carcinoma"
]

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Hybrid_CNN_CBAM_Transformer_SavedModel"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = tf.saved_model.load(MODEL_PATH)
    return model.signatures["serve"]


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    # Convert image to RGB
    image = np.array(image.convert("RGB"))

    # Resize
    image = cv2.resize(image, (224, 224))

    # RGB → grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # Median filtering
    median = cv2.medianBlur(
        gray,
        3
    )

    # Histogram equalization
    equalized = cv2.equalizeHist(
        median
    )

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    clahe_img = clahe.apply(
        equalized
    )

    # Morphological closing
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    morph = cv2.morphologyEx(
        clahe_img,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Grayscale → RGB
    final_image = cv2.cvtColor(
        morph,
        cv2.COLOR_GRAY2RGB
    )

    # Normalize 0–1
    normalized = (
        final_image.astype(np.float32) / 255.0
    )

    return final_image, normalized


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_image(normalized_image):

    batch = np.expand_dims(
        normalized_image,
        axis=0
    ).astype(np.float32)

    model = load_model()

    result = model(
        input_ct_image=tf.constant(batch)
    )

    probabilities = list(
        result.values()
    )[0].numpy()[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    return (
        predicted_class,
        confidence,
        probabilities
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🫁 Project Information")

    st.markdown(
        """
        **Hybrid CNN–CBAM–Transformer**

        Multi-class CT image classification
        research prototype.
        """
    )

    st.divider()

    st.markdown("### Model")

    st.write("**Backbone:** EfficientNetB0")
    st.write("**Attention:** CBAM")
    st.write("**Context Modeling:** Transformer")
    st.write("**Input:** 224 × 224 × 3")
    st.write("**Classes:** 4")

    st.divider()

    st.markdown("### Classification Classes")

    for class_name in CLASS_NAMES:
        st.write(f"• {class_name}")

    st.divider()

    st.caption(
        "For educational and research demonstration purposes."
    )


# ============================================================
# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
<div class="hero">
<div class="hero-title">🫁 Lung Cancer CT Image Classification</div>
<div class="hero-subtitle">
A hybrid deep learning system combining EfficientNetB0,
CBAM attention and Transformer encoders for multi-class
lung CT image classification.
</div>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# DISCLAIMER
# ============================================================

st.warning(
    "Research prototype only. The predicted class is a model "
    "output for educational and project demonstration purposes "
    "and is not a clinical diagnosis."
)


# ============================================================
# PROJECT OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">Project Overview</div>',
    unsafe_allow_html=True
)

overview_col1, overview_col2, overview_col3 = st.columns(3)

with overview_col1:
    st.markdown("""
    <div class="info-card">
        <div class="info-card-title">🔬 Image Enhancement</div>
        <div class="info-card-text">
            Median filtering, histogram equalization,
            CLAHE and morphological closing are applied
            before classification.
        </div>
    </div>
    """, unsafe_allow_html=True)

with overview_col2:
    st.markdown("""
    <div class="info-card">
        <div class="info-card-title">🧠 Hybrid Architecture</div>
        <div class="info-card-text">
            EfficientNetB0 extracts spatial features,
            CBAM emphasizes relevant features and
            Transformer encoders model contextual relationships.
        </div>
    </div>
    """, unsafe_allow_html=True)

with overview_col3:
    st.markdown("""
    <div class="info-card">
        <div class="info-card-title">🎯 Multi-Class Output</div>
        <div class="info-card-text">
            The model classifies CT images into four
            categories: three lung cancer types and
            normal lung tissue.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">Upload CT Scan</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a JPG, JPEG or PNG CT image",
    type=["jpg", "jpeg", "png"],
    help="Upload one CT scan image for classification."
)


# ============================================================
# IMAGE PROCESSING
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    processed_image, normalized_image = (
        preprocess_image(image)
    )

    st.markdown(
        '<div class="section-title">Image Processing</div>',
        unsafe_allow_html=True
    )

    image_col1, image_col2 = st.columns(2)

    with image_col1:

        st.markdown("#### Original CT Image")

        st.image(
            image,
            width="stretch"
        )

    with image_col2:

        st.markdown("#### Enhanced Preprocessed Image")

        st.image(
            processed_image,
            width="stretch"
        )

    st.caption(
        "Preprocessing pipeline: RGB → Resize (224×224) → "
        "Grayscale → Median Filtering → Histogram Equalization → "
        "CLAHE → Morphological Closing → 3-channel normalization"
    )

    st.divider()

    # ========================================================
    # PREDICTION
    # ========================================================

    st.markdown(
        '<div class="section-title">Model Prediction</div>',
        unsafe_allow_html=True
    )

    predict_button = st.button(
        "🔍 Classify CT Scan",
        type="primary",
        width="stretch"
    )

    if predict_button:

        with st.spinner(
            "Running hybrid CNN–CBAM–Transformer model..."
        ):

            (
                predicted_class,
                confidence,
                probabilities
            ) = predict_image(
                normalized_image
            )

        # ----------------------------------------------------
        # MAIN PREDICTION CARD
        # ----------------------------------------------------

        st.markdown(
            f"""
<div class="prediction-card">
    <div class="prediction-label">Predicted Class</div>
    <div class="prediction-class">{predicted_class}</div>
</div>
""",
            unsafe_allow_html=True
        )

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

        with metric_col2:
            st.metric(
                "Number of Classes",
                "4"
            )

        with metric_col3:
            st.metric(
                "Input Resolution",
                "224 × 224"
            )

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        st.markdown(
            "#### Class Probability Distribution"
        )

        for i, class_name in enumerate(
            CLASS_NAMES
        ):

            probability = float(
                probabilities[i]
            )

            st.write(
                f"**{class_name}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                probability
            )


# ============================================================
# METHODOLOGY
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">Model Methodology</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="pipeline">

Input CT Image
<span class="pipeline-arrow">→</span>
Image Enhancement
<span class="pipeline-arrow">→</span>
EfficientNetB0
<span class="pipeline-arrow">→</span>
CBAM Attention
<span class="pipeline-arrow">→</span>
Transformer Encoder
<span class="pipeline-arrow">→</span>
Classification
<span class="pipeline-arrow">→</span>
4 Classes

</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">Model Performance</div>',
    unsafe_allow_html=True
)

st.caption(
    "Performance measured on the independent held-out test set."
)

perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

with perf_col1:
    st.metric(
        "Accuracy",
        "71.75%"
    )

with perf_col2:
    st.metric(
        "Precision",
        "72.91%"
    )

with perf_col3:
    st.metric(
        "Recall",
        "71.75%"
    )

with perf_col4:
    st.metric(
        "F1-Score",
        "72.12%"
    )


# ============================================================
# ARCHITECTURE
# ============================================================

st.markdown(
    '<div class="section-title">Hybrid Architecture</div>',
    unsafe_allow_html=True
)

architecture_col1, architecture_col2 = st.columns(2)

with architecture_col1:

    st.markdown("""
    **CNN Feature Extraction**

    - EfficientNetB0 backbone
    - Frozen pretrained feature extractor
    - Extracts spatial representations
    - Output feature map: 7 × 7 × 1280
    """)

with architecture_col2:

    st.markdown("""
    **Attention + Global Context**

    - CBAM channel attention
    - CBAM spatial attention
    - Feature projection into tokens
    - Transformer encoder blocks
    - Global average pooling
    - Final 4-class softmax classifier
    """)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    Lung Cancer CT Image Classification |
    Hybrid CNN–CBAM–Transformer Research Prototype

</div>
""", unsafe_allow_html=True)
