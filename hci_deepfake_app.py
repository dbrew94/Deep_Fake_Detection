# hci_deepfake_app.py
"""
HCI Deepfake Detection — Streamlit Web Application
COS590 Final Project — Full Sail University

Deploy locally:   streamlit run hci_deepfake_app.py
Deploy to cloud:  push to GitHub then connect to share.streamlit.io

All imports are handled safely so the app runs in demo mode
even if PyTorch or OpenCV are not yet installed on the server.
"""

import streamlit as st
import numpy as np
import time
import os
import sys
import tempfile

# ==============================================================
# SAFE IMPORTS — never crash on missing packages
# ==============================================================

try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# ==============================================================
# PAGE CONFIG — must be the very first Streamlit call
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

    /* ---- Page background ---- */
    .main { background-color: #F0F4F8; }
    .stApp { background-color: #F0F4F8; }

    /* ---- Result boxes ---- */
    .result-fake {
        padding: 24px;
        border-radius: 14px;
        background: #FFEBEE;
        border: 2.5px solid #B71C1C;
        text-align: center;
        margin: 10px 0;
    }
    .result-real {
        padding: 24px;
        border-radius: 14px;
        background: #E8F5E9;
        border: 2.5px solid #1B5E20;
        text-align: center;
        margin: 10px 0;
    }
    .result-pending {
        padding: 24px;
        border-radius: 14px;
        background: #F5F5F5;
        border: 2px solid #BDBDBD;
        text-align: center;
        margin: 10px 0;
    }

    /* ---- Buttons ---- */
    .stButton > button {
        background-color: #1565C0;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 12px 28px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background-color: #0D47A1;
        color: white;
    }
    .stButton > button:disabled {
        background-color: #90A4AE;
        color: #ECEFF1;
    }

    /* ---- Metric cards ---- */
    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 12px;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background-color: #1A1A2E;
    }
    section[data-testid="stSidebar"] * {
        color: #ECF0F1 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #90CAF9 !important;
        font-weight: bold;
    }

    /* ---- Divider ---- */
    hr {
        border-color: #BDBDBD;
        margin: 16px 0;
    }

    /* ---- File uploader ---- */
    [data-testid="stFileUploader"] {
        border: 2px dashed #1565C0;
        border-radius: 10px;
        padding: 10px;
        background: white;
    }

    /* ---- Header ---- */
    h1 { color: #1A1A2E; }
    h2 { color: #1A237E; }
    h3 { color: #283593; }

</style>
""", unsafe_allow_html=True)

# ==============================================================
# CONSTANTS
# ==============================================================

FRAME_RATES = [5, 10, 15, 20]
IMG_SIZE    = 224
MODEL_PATH  = 'models/cnn_lstm_best.pth'

FRAME_GUIDE = {
    5:  {
        'latency': '4.8ms',
        'f1':      '0.694',
        'fnr':     '28.7%',
        'desc':    'Fast — lower accuracy',
        'color':   '#78909C'
    },
    10: {
        'latency': '9.6ms',
        'f1':      '0.727',
        'fnr':     '23.3%',
        'desc':    'Balanced — good accuracy',
        'color':   '#1565C0'
    },
    15: {
        'latency': '14.2ms',
        'f1':      '0.735',
        'fnr':     '22.7%',
        'desc':    'Optimal — best F1 score ★',
        'color':   '#1B5E20'
    },
    20: {
        'latency': '18.8ms',
        'f1':      '0.731',
        'fnr':     '22.7%',
        'desc':    'Thorough — diminishing returns',
        'color':   '#E65100'
    },
}

if TORCH_AVAILABLE:
    TRANSFORM = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
else:
    TRANSFORM = None


# ==============================================================
# MODEL LOADING — cached so it only loads once per session
# ==============================================================

@st.cache_resource
def load_model():
    """
    Load the CNN+LSTM deepfake detection model.

    Returns:
        model   : PyTorch model or None
        device  : torch.device or None
        loaded  : bool
        msg     : status string
    """
    if not TORCH_AVAILABLE:
        return (
            None, None, False,
            "PyTorch not installed — running in demo mode"
        )

    device = torch.device('cpu')

    # Try to import the model class
    try:
        sys.path.insert(
            0,
            os.path.dirname(os.path.abspath(__file__))
        )
        from src.model import CNNLSTMModel
    except ImportError as e:
        return (
            None, device, False,
            f"Could not import model class: {e}"
        )

    # Download from Hugging Face Hub if not present locally
    if not os.path.exists(MODEL_PATH):
        try:
            from huggingface_hub import hf_hub_download
            os.makedirs('models', exist_ok=True)
            hf_hub_download(
                repo_id='YOUR_HF_USERNAME/deepfake-detection',
                filename='cnn_lstm_best.pth',
                local_dir='models'
            )
        except Exception as e:
            return (
                None, device, False,
                f"Model file not found and download failed: {e}"
            )

    # Load weights
    try:
        model = CNNLSTMModel(
            sequence_length=10,
            hidden_size=256,
            num_layers=1,
            bidirectional=True
        ).to(device)

        state = torch.load(
            MODEL_PATH,
            map_location=device
        )
        model.load_state_dict(state)
        model.eval()
        return (
            model, device, True,
            "Model loaded successfully — CPU inference"
        )

    except Exception as e:
        return (
            None, device, False,
            f"Model weight loading failed: {e}"
        )


# ==============================================================
# FRAME EXTRACTION
# ==============================================================

def extract_frames(uploaded_file, num_frames):
    """
    Extract evenly spaced frames from an uploaded video file.

    Args:
        uploaded_file : Streamlit UploadedFile object
        num_frames    : int — number of frames to extract

    Returns:
        frames : list of PIL Images or None
        meta   : dict of video metadata or None
    """
    if not CV2_AVAILABLE:
        return None, None
    if not PIL_AVAILABLE:
        return None, None

    # Determine file extension
    name   = getattr(uploaded_file, 'name', 'video.mp4')
    suffix = os.path.splitext(name)[-1] or '.mp4'

    # Write to a temporary file OpenCV can open
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
    except Exception as e:
        st.error(f"Could not write temp file: {e}")
        return None, None

    try:
        cap = cv2.VideoCapture(tmp_path)

        if not cap.isOpened():
            return None, None

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS)
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total < 1:
            return None, None

        # Clamp to available frames
        actual = min(num_frames, total)
        indices = [
            int(k * total / actual)
            for k in range(actual)
        ]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )
                frames.append(Image.fromarray(rgb))

        cap.release()

        meta = {
            'fps':          round(fps, 2),
            'resolution':   f'{w} × {h}',
            'total_frames': total,
            'duration_s':   (
                round(total / fps, 2)
                if fps > 0 else 'N/A'
            ),
            'file_size_mb': round(
                getattr(uploaded_file, 'size', 0)
                / (1024 * 1024),
                2
            )
        }

        return (frames if frames else None), meta

    except Exception as e:
        st.error(f"Frame extraction error: {e}")
        return None, None

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ==============================================================
# INFERENCE
# ==============================================================

def run_inference(model, device, frames, T):
    """
    Run deepfake detection on extracted frames.

    Returns:
        label      : 'REAL' or 'FAKE'
        confidence : float 0.0 to 1.0
        latency_ms : float
        is_demo    : bool (True if using random prediction)
    """
    # Demo mode — no model or torch
    if model is None or not TORCH_AVAILABLE:
        time.sleep(0.6)
        conf  = float(np.random.uniform(0.20, 0.95))
        label = 'FAKE' if conf > 0.5 else 'REAL'
        return label, conf, 0.0, True

    # No frames extracted
    if not frames:
        conf  = float(np.random.uniform(0.20, 0.95))
        label = 'FAKE' if conf > 0.5 else 'REAL'
        return label, conf, 0.0, True

    try:
        # Pad or trim to T frames
        seq_frames = list(frames)
        while len(seq_frames) < T:
            seq_frames.append(seq_frames[-1])
        seq_frames = seq_frames[:T]

        # Build tensor (1, T, 3, H, W)
        tensors = [TRANSFORM(f) for f in seq_frames]
        seq     = torch.stack(tensors).unsqueeze(0).to(device)

        # Timed inference
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(seq)
        latency = (time.perf_counter() - t0) * 1000.0

        probs     = torch.softmax(logits, dim=1)
        fake_prob = float(probs[0, 1].cpu())
        label     = 'FAKE' if fake_prob > 0.5 else 'REAL'

        return label, fake_prob, latency, False

    except Exception as e:
        st.error(f"Inference error: {e}")
        conf  = float(np.random.uniform(0.2, 0.9))
        label = 'FAKE' if conf > 0.5 else 'REAL'
        return label, conf, 0.0, True


# ==============================================================
# CONFIDENCE BAR HTML
# ==============================================================

def confidence_bar_html(pct, label):
    """Return styled HTML confidence bar."""
    bar_color = (
        '#B71C1C' if label == 'FAKE' else '#1B5E20'
    )
    return f"""
    <div style="
        background: #E0E0E0;
        border-radius: 10px;
        height: 34px;
        position: relative;
        margin: 10px 0 4px 0;
        overflow: hidden;
    ">
      <div style="
        background: {bar_color};
        width: {pct}%;
        height: 100%;
        border-radius: 10px;
        opacity: 0.85;
        transition: width 0.5s ease;
      "></div>
      <div style="
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: white;
        font-weight: bold;
        font-size: 15px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
      ">{pct}%</div>
      <div style="
        position: absolute;
        top: 0; left: 50%;
        width: 2px; height: 100%;
        background: white;
        opacity: 0.5;
      "></div>
    </div>
    <div style="
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        color: #757575;
        margin-bottom: 14px;
    ">
      <span>← REAL (0%)</span>
      <span>50% decision threshold</span>
      <span>FAKE (100%) →</span>
    </div>
    """


# ==============================================================
# SIDEBAR
# ==============================================================

def build_sidebar():
    """Build sidebar controls. Returns config dict."""
    with st.sidebar:
        st.markdown(
            "## 🎭 HCI Deepfake Detector"
        )
        st.markdown(
            "*COS590 — Full Sail University*"
        )
        st.divider()

        # ---- Frame Rate ----
        st.markdown("### 🎞️ Frame Rate (T)")
        T = st.select_slider(
            "Frames per clip",
            options=FRAME_RATES,
            value=10,
            label_visibility='collapsed'
        )

        g = FRAME_GUIDE[T]
        st.markdown(
            f"""
            <div style="
                background: {g['color']}22;
                border: 1px solid {g['color']};
                border-radius: 8px;
                padding: 10px 14px;
                margin: 6px 0 10px 0;
            ">
            <b style="color:{g['color']};font-size:15px;">
                T = {T} frames</b><br>
            <span style="font-size:12px;color:#CFD8DC;">
                Latency: <b>{g['latency']}</b>
                &nbsp;|&nbsp;
                F1: <b>{g['f1']}</b><br>
                FNR: <b>{g['fnr']}</b><br>
                {g['desc']}
            </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ---- Metadata ----
        st.markdown("### 📋 Metadata (optional)")
        compression = st.selectbox(
            "Compression level",
            ['raw', 'c23', 'c40'],
            index=1,
            help="Encoding level of the video"
        )
        source_type = st.selectbox(
            "Video source",
            ['youtube', 'actors'],
            index=0,
            help="Origin of the video content"
        )

        st.divider()

        # ---- Performance Guide Table ----
        st.markdown("### 📊 Performance Guide")
        if PANDAS_AVAILABLE:
            guide_df = pd.DataFrame({
                'T':       [5, 10, 15, 20],
                'F1':      ['0.694', '0.727',
                            '0.735★', '0.731'],
                'Latency': ['4.8ms', '9.6ms',
                            '14.2ms', '18.8ms'],
                'FNR':     ['28.7%', '23.3%',
                            '22.7%', '22.7%']
            }).set_index('T')
            st.dataframe(
                guide_df,
                use_container_width=True
            )
        else:
            st.code(
                "T= 5  F1=0.694  4.8ms\n"
                "T=10  F1=0.727  9.6ms\n"
                "T=15  F1=0.735 14.2ms ★\n"
                "T=20  F1=0.731 18.8ms"
            )

        st.caption(
            "★ Recommended  "
            "|  FNR = False Negative Rate  "
            "|  All latencies < 200ms target"
        )

        st.divider()

        # ---- About ----
        with st.expander("ℹ️ About"):
            st.markdown("""
**Architecture:**
MobileNetV2 CNN + Bidirectional LSTM

**Dataset:**
FaceForensics++ c23
500 real + 500 fake clips

**Hypothesis validated:**
CNN+LSTM outperforms CNN baseline
by 5.3 F1 points at sub-200ms latency

**References:**
- Rossler et al. 2019 ICCV
- Sandler et al. 2018 CVPR
- Masood et al. 2023
""")

    return {
        'T':           T,
        'compression': compression,
        'source_type': source_type
    }


# ==============================================================
# MAIN APPLICATION
# ==============================================================

def main():

    # Load model once
    model, device, model_loaded, model_msg = load_model()

    # Build sidebar
    config      = build_sidebar()
    T           = config['T']
    compression = config['compression']
    source_type = config['source_type']

    # ---- Page Header ----
    st.markdown("# 🎭 Real-Time Deepfake Detection")
    st.markdown(
        "Upload a video clip and click **Run Detection** "
        "to analyze it using a hybrid **CNN+LSTM** "
        "deep learning architecture. Adjust the frame "
        "rate slider in the sidebar to explore the "
        "accuracy-latency tradeoff."
    )

    # ---- System Status Banner ----
    if model_loaded:
        st.success(
            f"✅ {model_msg}  "
            f"|  Device: {str(device).upper()}"
        )
    else:
        st.info(
            "🔄 **Demo Mode** — "
            "Dependencies are loading or model file "
            "is not yet uploaded. "
            "The interface is fully functional and "
            "detection results are simulated. "
            "Upload a video to explore the application."
        )

    st.divider()

    # ---- Main layout ----
    left_col, right_col = st.columns([1, 1.5])

    # ================================================
    # LEFT COLUMN — Upload and controls
    # ================================================
    with left_col:

        st.subheader("📂 Upload Video")
        uploaded = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            help="Supported: MP4, AVI, MOV, MKV, WebM"
        )

        if uploaded:
            st.success(
                f"✅ {uploaded.name}  "
                f"({uploaded.size / (1024*1024):.1f} MB)"
            )

        st.markdown("")

        # Frame rate display
        g = FRAME_GUIDE[T]
        st.subheader(f"🎞️ Frame Rate: T = {T}")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("F1 Score", g['f1'])
        with col_b:
            st.metric("Latency",  g['latency'])
        with col_c:
            st.metric("FNR",      g['fnr'])
        st.caption(g['desc'])

        st.markdown("")

        # Metadata display
        st.subheader("📋 Metadata")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Compression", compression.upper())
        with col_m2:
            st.metric("Source", source_type.capitalize())

        st.markdown("")

        # Detect button
        detect = st.button(
            "▶  RUN DETECTION",
            disabled=(uploaded is None),
            use_container_width=True
        )

        if not uploaded:
            st.caption(
                "⬆️ Upload a video file above "
                "to enable detection."
            )

        st.divider()

        # Architecture summary
        st.subheader("🧠 Architecture")
        st.markdown("""
| Component | Details |
|---|---|
| CNN Backbone | MobileNetV2 |
| Temporal Model | Bidirectional LSTM |
| Hidden Size | 256 per direction |
| Input | T × 224×224×3 |
| Output | REAL / FAKE |
| Parameters | ~5.4M |
        """)

    # ================================================
    # RIGHT COLUMN — Results
    # ================================================
    with right_col:

        st.subheader("🔍 Detection Result")

        # ---- Waiting state ----
        if not detect or uploaded is None:
            st.markdown(
                '<div class="result-pending">'
                '<h2 style="color:#9E9E9E;margin:0">—</h2>'
                '<p style="color:#9E9E9E;margin:8px 0 0 0">'
                'Upload a video and click Run Detection'
                '</p></div>',
                unsafe_allow_html=True
            )
            st.markdown("")
            st.subheader("🎬 Extracted Frames")
            st.info(
                "Frames extracted from your video "
                "will appear here after detection."
            )
            st.subheader("📋 Video Information")
            st.info(
                "Video metadata will appear here "
                "after uploading a file."
            )
            return

        # ---- Extract frames ----
        uploaded.seek(0)
        with st.spinner(
            f"Extracting {T} frames from video..."
        ):
            if CV2_AVAILABLE and PIL_AVAILABLE:
                frames, meta = extract_frames(uploaded, T)
            else:
                frames, meta = None, None

        if frames is None and CV2_AVAILABLE:
            st.error(
                "Could not read this video file. "
                "Please try a different format or file."
            )
            return

        # ---- Run inference ----
        with st.spinner(
            "Running CNN+LSTM detection model..."
        ):
            label, conf, latency, is_demo = run_inference(
                model, device,
                frames if frames else [],
                T
            )

        pct       = int(conf * 100)
        css_class = (
            'result-fake' if label == 'FAKE'
            else 'result-real'
        )
        icon  = '⚠️' if label == 'FAKE' else '✅'
        color = '#B71C1C' if label == 'FAKE' else '#1B5E20'
        demo_note = ' (demo)' if is_demo else ''

        # Result box
        st.markdown(
            f'<div class="{css_class}">'
            f'<h1 style="color:{color};margin:0;'
            f'font-size:42px;">'
            f'{icon} {label}</h1>'
            f'<h3 style="color:{color};margin:8px 0 4px 0;">'
            f'Confidence: {pct}%{demo_note}</h3>'
            f'<p style="color:#616161;margin:0;">'
            f'T = {T} frames  |  '
            f'{"%.1f" % latency + "ms" if latency > 0 else "demo mode"}  '
            f'|  Compression: {compression}  '
            f'|  Source: {source_type}'
            f'</p></div>',
            unsafe_allow_html=True
        )

        # Confidence bar
        st.markdown("**Detection Confidence**")
        st.markdown(
            confidence_bar_html(pct, label),
            unsafe_allow_html=True
        )

        # Metric row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Result",     label)
        with m2:
            st.metric("Confidence", f"{pct}%")
        with m3:
            st.metric(
                "Latency",
                f"{latency:.1f}ms" if latency > 0
                else "demo"
            )
        with m4:
            st.metric("Frames",     f"T = {T}")

        # Risk interpretation
        if pct >= 75:
            st.error(
                "🔴 **High Risk** — Model is highly confident "
                "this video has been manipulated."
            )
        elif pct >= 50:
            st.warning(
                "🟡 **Moderate Risk** — Model flags this "
                "video as likely fake. Exercise caution."
            )
        elif pct >= 30:
            st.info(
                "🔵 **Low Risk** — Model leans toward real "
                "but confidence is moderate."
            )
        else:
            st.success(
                "🟢 **Very Low Risk** — Model is confident "
                "this video is authentic."
            )

        st.divider()

        # ---- Extracted Frames ----
        st.subheader("🎬 Extracted Frames")

        if frames and PIL_AVAILABLE:
            n_cols = min(len(frames), 5)
            for row_start in range(0, len(frames), n_cols):
                row_frames = frames[row_start:row_start + n_cols]
                cols = st.columns(len(row_frames))
                for col, frame, fi in zip(
                    cols,
                    row_frames,
                    range(row_start, row_start + len(row_frames))
                ):
                    with col:
                        st.image(
                            frame,
                            caption=f'Frame {fi + 1}',
                            use_column_width=True
                        )
            st.caption(
                f"{len(frames)} frames extracted at "
                f"evenly spaced intervals across the clip."
            )
        elif not CV2_AVAILABLE:
            st.warning(
                "OpenCV not available on this server. "
                "Frame display requires OpenCV. "
                "Detection ran in demo mode."
            )
        else:
            st.warning(
                "Could not extract frames from this file. "
                "Demo prediction was used."
            )

        st.divider()

        # ---- Video Metadata ----
        st.subheader("📋 Video Information")

        if meta:
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric(
                    "Resolution", meta['resolution']
                )
                st.metric(
                    "FPS", str(meta['fps'])
                )
            with mc2:
                st.metric(
                    "Duration", f"{meta['duration_s']}s"
                )
                st.metric(
                    "Total Frames",
                    str(meta['total_frames'])
                )
            with mc3:
                st.metric(
                    "File Size",
                    f"{meta['file_size_mb']} MB"
                )
                st.metric(
                    "Extracted", f"{T} frames"
                )
        else:
            st.info(
                "Video metadata not available in demo mode."
            )

    # ================================================
    # BOTTOM SECTION — Full Results Table
    # ================================================
    st.divider()
    st.subheader("📊 Frame Rate Tradeoff Summary")

    col_table, col_chart = st.columns([1, 1])

    with col_table:
        if PANDAS_AVAILABLE:
            results_df = pd.DataFrame({
                'T':            [5,    10,    15,    20   ],
                'F1 Score':     [0.694, 0.727, 0.735, 0.731],
                'Accuracy':     ['69.3%','72.7%','73.3%','73.3%'],
                'FNR':          ['28.7%','23.3%','22.7%','22.7%'],
                'Latency':      ['4.8ms','9.6ms','14.2ms','18.8ms'],
                'Recommended':  ['', '', '★ Yes', '']
            }).set_index('T')

            st.dataframe(
                results_df,
                use_container_width=True
            )
            st.caption(
                "FNR = False Negative Rate.  "
                "★ = Recommended configuration.  "
                "All latencies satisfy < 200ms target."
            )

    with col_chart:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor('#F0F4F8')
            ax.set_facecolor('#F8F9FA')

            ts  = [5, 10, 15, 20]
            f1s = [0.694, 0.727, 0.735, 0.731]
            lats = [4.8, 9.6, 14.2, 18.8]

            ax2 = ax.twinx()

            ax.plot(
                ts, f1s,
                color='#1B5E20', lw=2,
                marker='o', markersize=6,
                label='F1 Score'
            )
            ax2.plot(
                ts, lats,
                color='#1565C0', lw=2,
                marker='s', markersize=5,
                linestyle='--',
                label='Latency (ms)'
            )

            # Highlight T=15
            ax.scatter(
                [15], [0.735],
                color='#F59E0B', s=140,
                zorder=6, edgecolors='black',
                linewidth=1.2
            )
            ax.annotate(
                'Optimal',
                xy=(15, 0.735),
                xytext=(16, 0.730),
                fontsize=7.5, color='#1B5E20',
                fontweight='bold',
                arrowprops=dict(
                    arrowstyle='->',
                    color='#1B5E20', lw=1.0
                )
            )

            ax.set_xlabel('Sequence Length (T)', fontsize=8)
            ax.set_ylabel('F1 Score', fontsize=8,
                          color='#1B5E20')
            ax2.set_ylabel('Latency (ms)', fontsize=8,
                           color='#1565C0')
            ax.set_title(
                'F1 Score vs Latency by Frame Rate T',
                fontsize=8.5, fontweight='bold'
            )
            ax.set_xticks(ts)
            ax.set_ylim(0.66, 0.76)
            ax2.set_ylim(0, 25)
            ax.grid(alpha=0.25, linestyle='--')
            ax.set_axisbelow(True)
            for sp in ax.spines.values():
                sp.set_linewidth(0.7)

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(
                lines1 + lines2,
                labels1 + labels2,
                fontsize=7.5, loc='lower right'
            )

            plt.tight_layout(pad=0.5)
            st.pyplot(fig)
            plt.close()

        except ImportError:
            st.info(
                "Install matplotlib to see the "
                "performance chart."
            )

    # ================================================
    # FOOTER
    # ================================================
    st.divider()
    st.markdown(
        "<center>"
        "<small>"
        "HCI Deepfake Detector &nbsp;|&nbsp; "
        "COS590 Human-Computer Interaction &nbsp;|&nbsp; "
        "Full Sail University &nbsp;|&nbsp; "
        "Built with Streamlit + PyTorch + MobileNetV2"
        "</small>"
        "</center>",
        unsafe_allow_html=True
    )


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == '__main__':
    main()
