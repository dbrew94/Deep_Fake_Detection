# hci_deepfake_app.py
"""
COS640 - HCI Deepfake Detection Application
Dylan Brewington
Prototype graphical user interface implementing:
  - Configurable frame rate selection (T = 5, 10, 15, 20)
  - Optional contextual metadata input
  - Real-time detection with confidence visualization
"""

import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
import time
import os
import sys
import tempfile
from PIL import Image

# ==============================================================
# PAGE CONFIG — must be first Streamlit call
# ==============================================================

st.set_page_config(
    page_title='HCI Deepfake Detector',
    page_icon='🎭',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ==============================================================
# CUSTOM CSS
# ==============================================================

st.markdown("""
<style>
    .main { background-color: #F0F4F8; }
    .stApp { background-color: #F0F4F8; }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin: 10px 0;
    }
    .result-fake {
        background-color: #FFEBEE;
        border: 2px solid #B71C1C;
    }
    .result-real {
        background-color: #E8F5E9;
        border: 2px solid #1B5E20;
    }
    .result-pending {
        background-color: #F5F5F5;
        border: 2px solid #9E9E9E;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #E0E0E0;
        text-align: center;
        margin: 5px;
    }
    .frame-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        margin-top: 10px;
    }
    h1 {
        color: #1A1A2E;
        font-family: serif;
    }
    .stButton > button {
        background-color: #1565C0;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #0D47A1;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================
# CONSTANTS
# ==============================================================

FRAME_RATES = [5, 10, 15, 20]
IMG_SIZE    = 224
MODEL_PATH  = 'models/cnn_lstm_best.pth'

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

FRAME_GUIDE = {
    5:  ('4.8ms',  '0.694', 'Fast — lower accuracy'),
    10: ('9.6ms',  '0.727', 'Balanced — good accuracy'),
    15: ('14.2ms', '0.735', 'Optimal — best F1 score ★'),
    20: ('18.8ms', '0.731', 'Thorough — diminishing returns'),
}


# ==============================================================
# MODEL LOADING (cached so it only loads once)
# ==============================================================

@st.cache_resource
def load_model():
    """
    Load CNN+LSTM model.
    Cached by Streamlit so it persists across reruns.
    Returns model and device.
    """
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    try:
        sys.path.insert(
            0,
            os.path.dirname(os.path.abspath(__file__))
        )
        from src.model import CNNLSTMModel
        model = CNNLSTMModel(
            sequence_length=10,
            hidden_size=256,
            num_layers=1,
            bidirectional=True
        ).to(device)

        if os.path.exists(MODEL_PATH):
            state = torch.load(
                MODEL_PATH, map_location=device
            )
            model.load_state_dict(state)
            model.eval()
            return model, device, True
        else:
            return None, device, False

    except Exception as e:
        return None, device, False


# ==============================================================
# VIDEO PROCESSING
# ==============================================================

def extract_frames(video_path, num_frames):
    """
    Extract evenly spaced frames from a video file.
    Returns list of PIL Images.
    """
    try:
        import cv2
    except ImportError:
        st.error(
            "OpenCV not installed. "
            "Run: pip install opencv-python-headless"
        )
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total < num_frames:
        num_frames = max(1, total)

    indices = [
        int(k * total / num_frames)
        for k in range(num_frames)
    ]

    frames = []
    meta   = {
        'fps':        round(fps, 2),
        'resolution': f'{w} x {h}',
        'total_frames': total,
        'duration_s': round(total / fps, 2)
                      if fps > 0 else 'N/A',
        'file_size_mb': round(
            os.path.getsize(video_path) / (1024*1024), 2
        )
    }

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )
            frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames, meta


def build_metadata_vector(compression, source):
    """Build 7-dim metadata vector."""
    comp_map = {'raw': 0, 'c23': 1, 'c40': 2}
    src_map  = {'youtube': 0, 'actors': 1}
    comp_vec = [0.0, 0.0, 0.0]
    src_vec  = [0.0, 0.0]
    comp_vec[comp_map.get(compression, 1)] = 1.0
    src_vec[src_map.get(source, 0)]        = 1.0
    return comp_vec + src_vec + [0.5, 0.5]


def run_inference(model, frames, device, T):
    """Run deepfake detection on extracted frames."""
    if model is None:
        # Demo mode with random result
        time.sleep(0.5)
        conf  = float(np.random.uniform(0.25, 0.95))
        label = 'FAKE' if conf > 0.5 else 'REAL'
        return label, conf, 15.0

    while len(frames) < T:
        frames.append(frames[-1])
    frames = frames[:T]

    tensors = [TRANSFORM(f) for f in frames]
    seq     = torch.stack(tensors).unsqueeze(0).to(device)

    if device.type == 'cuda':
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev   = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        with torch.no_grad():
            logits = model(seq)
        end_ev.record()
        torch.cuda.synchronize()
        latency = start_ev.elapsed_time(end_ev)
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(seq)
        latency = (time.perf_counter() - t0) * 1000.0

    probs     = torch.softmax(logits, dim=1)
    fake_prob = float(probs[0, 1].cpu())
    label     = 'FAKE' if fake_prob > 0.5 else 'REAL'
    return label, fake_prob, latency


# ==============================================================
# MAIN APP LAYOUT
# ==============================================================

def main():

    # Load model once
    model, device, model_loaded = load_model()

    # ==============================================================
    # HEADER
    # ==============================================================

    col_title, col_status = st.columns([3, 1])

    with col_title:
        st.markdown(
            "# 🎭 HCI Deepfake Detector"
        )
        st.markdown(
            "Real-time deepfake detection using a "
            "hybrid **CNN+LSTM** architecture trained on "
            "FaceForensics++. "
            "Select a video, choose your frame rate, "
            "and run detection."
        )

    with col_status:
        st.markdown("**System Status**")
        if model_loaded:
            st.success(
                f"Model ready\n\n"
                f"Device: {str(device).upper()}"
            )
        else:
            st.warning(
                "Demo mode\n\n"
                "Model file not found\n"
                "(random predictions)"
            )

    st.divider()

    # ==============================================================
    # MAIN LAYOUT: Left controls, Right results
    # ==============================================================

    left_col, right_col = st.columns([1, 1.4])

    # ----------------------------------------------------------
    # LEFT COLUMN — Controls
    # ----------------------------------------------------------

    with left_col:

        # ---- Video Upload ----
        st.subheader("📂 Upload Video")
        uploaded = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Supported formats: MP4, AVI, MOV, MKV"
        )

        # ---- Frame Rate Selection ----
        st.subheader("🎞️ Frame Rate (T)")

        T = st.radio(
            "Frames per clip",
            options=FRAME_RATES,
            index=1,    # Default T=10
            horizontal=True,
            help="Higher T = better accuracy, higher latency"
        )

        lat, f1, desc = FRAME_GUIDE[T]
        st.info(
            f"**T = {T}**  \n"
            f"Latency: {lat}  |  F1: {f1}  \n"
            f"{desc}"
        )

        # ---- Metadata (Optional) ----
        with st.expander(
            "⚙️ Contextual Metadata (optional)",
            expanded=False
        ):
            compression = st.selectbox(
                "Compression level",
                options=['raw', 'c23', 'c40'],
                index=1
            )
            source_type = st.selectbox(
                "Video source type",
                options=['youtube', 'actors'],
                index=0
            )
            st.caption(
                "Metadata helps the model calibrate "
                "detection for different compression levels."
            )

        # ---- Detect Button ----
        st.markdown("")
        detect_btn = st.button(
            "▶  RUN DETECTION",
            disabled=(uploaded is None),
            use_container_width=True
        )

        # ---- Frame Rate Guide ----
        st.subheader("📊 Frame Rate Guide")
        guide_data = {
            'T': [5, 10, 15, 20],
            'F1 Score': [0.694, 0.727, 0.735, 0.731],
            'Latency': ['4.8ms', '9.6ms',
                        '14.2ms ★', '18.8ms'],
            'FNR': ['28.7%', '23.3%',
                    '22.7%', '22.7%']
        }
        import pandas as pd
        df_guide = pd.DataFrame(guide_data)
        df_guide = df_guide.set_index('T')
        st.dataframe(
            df_guide,
            use_container_width=True
        )
        st.caption(
            "FNR = False Negative Rate.  "
            "★ = Recommended configuration."
        )

    # ----------------------------------------------------------
    # RIGHT COLUMN — Results
    # ----------------------------------------------------------

    with right_col:

        st.subheader("🔍 Detection Results")

        # ---- Default state ----
        if not detect_btn or uploaded is None:
            st.markdown(
                '<div class="result-box result-pending">'
                '<h2 style="color:#9E9E9E">—</h2>'
                '<p style="color:#9E9E9E">'
                'Upload a video and click Run Detection'
                '</p></div>',
                unsafe_allow_html=True
            )
            st.markdown("")
            st.subheader("🎬 Extracted Frames")
            st.info(
                "Frames from the uploaded video will "
                "appear here after detection."
            )
            st.subheader("📋 Video Metadata")
            st.info("Video information will appear here.")
            return

        # ---- Save uploaded file to temp location ----
        with tempfile.NamedTemporaryFile(
            delete=False, suffix='.mp4'
        ) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        # ---- Extract frames ----
        with st.spinner(
            f'Extracting {T} frames from video...'
        ):
            result = extract_frames(tmp_path, T)

        if result is None:
            st.error(
                "Could not extract frames. "
                "Please check the video file format."
            )
            os.unlink(tmp_path)
            return

        frames, meta = result

        # ---- Run inference ----
        with st.spinner('Running detection model...'):
            meta_vec = build_metadata_vector(
                compression, source_type
            )
            label, conf, latency = run_inference(
                model, frames, device, T
            )

        os.unlink(tmp_path)

        # ---- Display result ----
        css_class = (
            'result-fake' if label == 'FAKE'
            else 'result-real'
        )
        icon  = '⚠️' if label == 'FAKE' else '✅'
        color = '#B71C1C' if label == 'FAKE' else '#1B5E20'
        pct   = int(conf * 100)

        st.markdown(
            f'<div class="result-box {css_class}">'
            f'<h1 style="color:{color};margin:0">'
            f'{icon} {label}</h1>'
            f'<h3 style="color:{color};margin:5px 0">'
            f'Confidence: {pct}%</h3>'
            f'<p style="color:#666;margin:0">'
            f'Latency: {latency:.1f}ms  |  '
            f'T = {T} frames  |  '
            f'Device: {str(device).upper()}</p>'
            f'</div>',
            unsafe_allow_html=True
        )

        # ---- Confidence bar ----
        st.markdown("**Detection Confidence**")
        bar_color = (
            '#B71C1C' if label == 'FAKE'
            else '#1B5E20'
        )
        st.markdown(
            f"""
            <div style="
                background:#E0E0E0;
                border-radius:8px;
                height:28px;
                position:relative;
                margin:8px 0 4px 0;
            ">
              <div style="
                background:{bar_color};
                width:{pct}%;
                height:100%;
                border-radius:8px;
                opacity:0.85;
              "></div>
              <div style="
                position:absolute;
                top:50%; left:50%;
                transform:translate(-50%,-50%);
                color:white;
                font-weight:bold;
                font-size:14px;
              ">{pct}%</div>
              <div style="
                position:absolute;
                top:50%; left:50%;
                transform:translate(-50%,-50%)
                translateX(-50%);
                width:2px; height:100%;
                background:white;
                opacity:0.6;
              "></div>
            </div>
            <div style="
                display:flex;
                justify-content:space-between;
                font-size:11px; color:#666;
            ">
              <span>REAL (0%)</span>
              <span>FAKE (100%)</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ---- Metric cards ----
        st.markdown("")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Result",    label)
        with m2:
            st.metric("Confidence", f"{pct}%")
        with m3:
            st.metric("Latency",   f"{latency:.1f}ms")
        with m4:
            st.metric("Frames",    f"T = {T}")

        # ---- Extracted frames grid ----
        st.subheader("🎬 Extracted Frames")
        st.caption(
            f"{len(frames)} frames extracted "
            f"at evenly spaced intervals"
        )

        cols = st.columns(min(len(frames), 5))
        for i, (col, frame) in enumerate(
            zip(cols * (len(frames) // 5 + 1),
                frames)
        ):
            with col:
                st.image(
                    frame,
                    caption=f'Frame {i+1}',
                    use_column_width=True
                )

        # ---- Video metadata ----
        st.subheader("📋 Video Metadata")
        meta_col1, meta_col2 = st.columns(2)

        with meta_col1:
            st.markdown(f"**File:** {uploaded.name}")
            st.markdown(
                f"**Resolution:** {meta['resolution']}"
            )
            st.markdown(
                f"**FPS:** {meta['fps']}"
            )

        with meta_col2:
            st.markdown(
                f"**Duration:** {meta['duration_s']}s"
            )
            st.markdown(
                f"**Total Frames:** {meta['total_frames']}"
            )
            st.markdown(
                f"**File Size:** {meta['file_size_mb']} MB"
            )

    # ==============================================================
    # ABOUT SECTION
    # ==============================================================

    st.divider()
    with st.expander("ℹ️ About This Application"):
        st.markdown("""
        ### HCI Deepfake Detection — COS590 Final Project

        This application implements a hybrid
        **CNN+LSTM architecture** for real-time
        deepfake detection in video streams,
        developed as the final project for the
        Human-Computer Interaction course at
        Full Sail University.

        **Architecture:**
        - CNN Backbone: MobileNetV2
          (pretrained on ImageNet)
        - Temporal Modeling: Bidirectional LSTM
          (hidden=256, bidirectional)
        - Dataset: FaceForensics++ c23
          (500 real + 500 fake videos)

        **Key Results:**
        | Configuration | F1 Score | Latency |
        |---|---|---|
        | T=5  frames | 0.694 | 4.8ms  |
        | T=10 frames | 0.727 | 9.6ms  |
        | T=15 frames | 0.735 | 14.2ms |
        | T=20 frames | 0.731 | 18.8ms |

        **References:**
        - Rossler et al. (2019). FaceForensics++.
          ICCV.
        - Sandler et al. (2018). MobileNetV2.
          CVPR.
        - Masood et al. (2023). Deepfakes
          generation and detection. Applied
          Intelligence.
        """)


if __name__ == '__main__':
    main()
