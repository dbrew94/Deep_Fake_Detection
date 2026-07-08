# hci_deepfake_app.py
"""
COS640 - HCI Deepfake Detection Application
Dylan Brewington
Prototype graphical user interface implementing:
  - Configurable frame rate selection (T = 5, 10, 15, 20)
  - Optional contextual metadata input
  - Real-time detection with confidence visualization
"""

# web_app.py
"""
HCI Deepfake Detection — Streamlit Web Application
Fully cloud-safe version with graceful fallbacks.

Deploy: streamlit run web_app.py
"""

import streamlit as st
import numpy as np
import time
import os
import sys
import tempfile

# ---- Safe imports ----
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
# PAGE CONFIG
# ==============================================================

st.set_page_config(
    page_title='HCI Deepfake Detector',
    page_icon='🎭',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ==============================================================
# CSS
# ==============================================================

st.markdown("""
<style>
    .result-fake {
        padding:20px; border-radius:12px;
        background:#FFEBEE; border:2px solid #B71C1C;
        text-align:center; margin:10px 0;
    }
    .result-real {
        padding:20px; border-radius:12px;
        background:#E8F5E9; border:2px solid #1B5E20;
        text-align:center; margin:10px 0;
    }
    .result-pending {
        padding:20px; border-radius:12px;
        background:#F5F5F5; border:2px solid #9E9E9E;
        text-align:center; margin:10px 0;
    }
    .stButton > button {
        background-color:#1565C0; color:white;
        border-radius:8px; border:none;
        padding:10px 24px; font-size:16px;
        font-weight:bold; width:100%;
    }
    .stButton > button:hover {
        background-color:#0D47A1;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================
# CONSTANTS
# ==============================================================

FRAME_RATES = [5, 10, 15, 20]
IMG_SIZE    = 224
MODEL_PATH  = 'models/cnn_lstm_best.pth'

FRAME_GUIDE = {
    5:  ('4.8ms',  '0.694', 'Fast — lower accuracy'),
    10: ('9.6ms',  '0.727', 'Balanced — good accuracy'),
    15: ('14.2ms', '0.735', 'Optimal — best F1 score ★'),
    20: ('18.8ms', '0.731', 'Thorough — diminishing returns'),
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
# MODEL LOADING
# ==============================================================

@st.cache_resource
def load_model():
    """
    Load CNN+LSTM model.
    Returns (model, device, loaded_bool, status_msg).
    """
    if not TORCH_AVAILABLE:
        return (None, None, False,
                "PyTorch not available — running in demo mode")

    device = torch.device('cpu')

    try:
        sys.path.insert(
            0, os.path.dirname(os.path.abspath(__file__))
        )
        from src.model import CNNLSTMModel
    except ImportError as e:
        return (None, device, False,
                f"Could not import model: {e}")

    # Download from Hugging Face if not present locally
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
            return (None, device, False,
                    f"Model not found and download failed: {e}\n"
                    f"Running in demo mode.")

    try:
        model = CNNLSTMModel(
            sequence_length=10,
            hidden_size=256,
            num_layers=1,
            bidirectional=True
        ).to(device)

        state = torch.load(
            MODEL_PATH, map_location=device
        )
        model.load_state_dict(state)
        model.eval()
        return (model, device, True, "Model loaded — CPU")

    except Exception as e:
        return (None, device, False,
                f"Model load error: {e}")


# ==============================================================
# FRAME EXTRACTION
# ==============================================================

def extract_frames_from_upload(uploaded_file, num_frames):
    """
    Extract frames from a Streamlit uploaded file.
    Saves to temp file then extracts with OpenCV.
    Returns (frames_list, metadata_dict) or (None, None).
    """
    if not CV2_AVAILABLE:
        return None, None

    if not PIL_AVAILABLE:
        return None, None

    # Write to temp file
    suffix = os.path.splitext(
        uploaded_file.name
    )[-1] or '.mp4'

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix
    ) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

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

        actual_frames = min(num_frames, total)
        indices = [
            int(k * total / actual_frames)
            for k in range(actual_frames)
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
            'resolution':   f'{w} x {h}',
            'total_frames': total,
            'duration_s':   round(total / fps, 2)
                            if fps > 0 else 'N/A',
            'file_size_mb': round(
                uploaded_file.size / (1024 * 1024), 2
            )
        }

        return frames if frames else None, meta

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
    Run detection. Falls back to demo mode
    if model is None or torch unavailable.
    """
    if model is None or not TORCH_AVAILABLE:
        time.sleep(0.8)
        conf  = float(np.random.uniform(0.20, 0.95))
        label = 'FAKE' if conf > 0.5 else 'REAL'
        return label, conf, 0.0, True

    try:
        # Pad or trim sequence
        while len(frames) < T:
            frames.append(frames[-1])
        frames = frames[:T]

        tensors = [TRANSFORM(f) for f in frames]
        seq     = torch.stack(tensors).unsqueeze(0).to(device)

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
# SIDEBAR
# ==============================================================

def build_sidebar():
    """Build the sidebar controls. Returns config dict."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.divider()

        # Frame rate
        st.markdown("### 🎞️ Frame Rate (T)")
        T = st.select_slider(
            "Frames per clip",
            options=FRAME_RATES,
            value=10,
            label_visibility='collapsed'
        )
        lat, f1, desc = FRAME_GUIDE[T]
        st.info(
            f"**T = {T} frames**\n\n"
            f"Latency: `{lat}`\n\n"
            f"F1 Score: `{f1}`\n\n"
            f"{desc}"
        )

        st.divider()

        # Metadata
        st.markdown("### 📋 Contextual Metadata")
        compression = st.selectbox(
            "Compression level",
            ['raw', 'c23', 'c40'],
            index=1
        )
        source_type = st.selectbox(
            "Video source type",
            ['youtube', 'actors'],
            index=0
        )

        st.divider()

        # Frame rate guide table
        st.markdown("### 📊 Performance Guide")
        if PANDAS_AVAILABLE:
            import pandas as pd
            guide_df = pd.DataFrame({
                'T':       [5, 10, 15, 20],
                'F1':      [0.694, 0.727, 0.735, 0.731],
                'ms':      ['4.8', '9.6', '14.2', '18.8'],
                'FNR':     ['28.7%', '23.3%', '22.7%', '22.7%']
            }).set_index('T')
            st.dataframe(
                guide_df,
                use_container_width=True
            )
        else:
            st.code(
                "T=5   F1=0.694  4.8ms\n"
                "T=10  F1=0.727  9.6ms\n"
                "T=15  F1=0.735 14.2ms ★\n"
                "T=20  F1=0.731 18.8ms"
            )

        st.caption(
            "★ = Recommended  |  FNR = False Negative Rate"
        )

    return {
        'T':           T,
        'compression': compression,
        'source_type': source_type
    }


# ==============================================================
# MAIN APP
# ==============================================================

def main():

    # ---- Load model ----
    model, device, model_loaded, model_msg = load_model()

    # ---- Sidebar ----
    config = build_sidebar()
    T           = config['T']
    compression = config['compression']
    source_type = config['source_type']

    # ---- Header ----
    st.markdown("# 🎭 HCI Deepfake Detector")
    st.markdown(
        "Real-time deepfake detection using a hybrid "
        "**CNN+LSTM** architecture.  "
        "Upload a video, choose a frame rate, "
        "and click **Run Detection**."
    )

    # System status banner
    if model_loaded:
        st.success(
            f"✅ {model_msg}  |  "
            f"Device: {str(device).upper()}"
        )
    else:
        st.warning(
            f"⚠️ Demo Mode — {model_msg}  "
            f"Results are randomized for demonstration."
        )

    # Dependency warnings
    if not CV2_AVAILABLE:
        st.error(
            "OpenCV not available. "
            "Frame extraction disabled. "
            "Check requirements.txt and packages.txt."
        )
    if not TORCH_AVAILABLE:
        st.error(
            "PyTorch not available. "
            "Check requirements.txt — ensure CPU build URL "
            "is included."
        )

    st.divider()

    # ---- File Upload ----
    uploaded = st.file_uploader(
        "📂 Upload Video File",
        type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
        help="Maximum file size: 200MB"
    )

    if uploaded is not None:
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("File", uploaded.name[:25])
        with col_info2:
            st.metric(
                "Size",
                f"{uploaded.size / (1024*1024):.1f} MB"
            )
        with col_info3:
            st.metric("Frame Rate (T)", T)

    # ---- Detect Button ----
    detect = st.button(
        "▶  RUN DETECTION",
        disabled=(uploaded is None),
        use_container_width=True
    )

    st.divider()

    # ---- Results Area ----
    col_result, col_frames = st.columns([1, 1.5])

    with col_result:
        st.subheader("🔍 Detection Result")

        if not detect or uploaded is None:
            st.markdown(
                '<div class="result-pending">'
                '<h2 style="color:#9E9E9E">—</h2>'
                '<p style="color:#9E9E9E">'
                'Awaiting video input</p></div>',
                unsafe_allow_html=True
            )
        else:
            # Extract frames
            with st.spinner(
                f'Extracting {T} frames...'
            ):
                uploaded.seek(0)
                if CV2_AVAILABLE and PIL_AVAILABLE:
                    frames, meta = extract_frames_from_upload(
                        uploaded, T
                    )
                else:
                    frames = None
                    meta   = None

            if frames is None and CV2_AVAILABLE:
                st.error(
                    "Could not extract frames from this file. "
                    "Try a different video format."
                )
                return

            # Run inference
            with st.spinner('Running detection model...'):
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
            color = (
                '#B71C1C' if label == 'FAKE'
                else '#1B5E20'
            )
            demo_note = ' (demo)' if is_demo else ''

            # Result box
            st.markdown(
                f'<div class="result-box {css_class}">'
                f'<h1 style="color:{color};margin:0">'
                f'{icon} {label}</h1>'
                f'<h3 style="color:{color};margin:5px 0">'
                f'Confidence: {pct}%{demo_note}</h3>'
                f'<p style="color:#666;margin:0">'
                f'T={T} frames  |  '
                f'{latency:.1f}ms latency</p>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Confidence bar
            st.markdown("**Confidence Visualization**")
            bar_color = (
                '#B71C1C' if label == 'FAKE'
                else '#1B5E20'
            )
            st.markdown(
                f"""
                <div style="
                    background:#E0E0E0;
                    border-radius:8px;
                    height:32px;
                    position:relative;
                    margin:8px 0 4px 0;
                ">
                  <div style="
                    background:{bar_color};
                    width:{pct}%;
                    height:100%;
                    border-radius:8px;
                    opacity:0.82;
                  "></div>
                  <div style="
                    position:absolute;
                    top:50%; left:50%;
                    transform:translate(-50%,-50%);
                    color:white;
                    font-weight:bold;
                    font-size:15px;
                  ">{pct}%</div>
                </div>
                <div style="
                    display:flex;
                    justify-content:space-between;
                    font-size:11px; color:#888;
                    margin-bottom:12px;
                ">
                  <span>REAL ← 0%</span>
                  <span>50% threshold</span>
                  <span>100% → FAKE</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Metric row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Result", label)
            with m2:
                st.metric("Confidence", f"{pct}%")
            with m3:
                if latency > 0:
                    st.metric("Latency", f"{latency:.1f}ms")
                else:
                    st.metric("Mode", "Demo")

            # Video metadata
            if meta:
                st.subheader("📋 Video Info")
                vm1, vm2 = st.columns(2)
                with vm1:
                    st.markdown(
                        f"**Resolution:** {meta['resolution']}"
                    )
                    st.markdown(
                        f"**FPS:** {meta['fps']}"
                    )
                    st.markdown(
                        f"**Duration:** {meta['duration_s']}s"
                    )
                with vm2:
                    st.markdown(
                        f"**Total frames:** {meta['total_frames']}"
                    )
                    st.markdown(
                        f"**File size:** {meta['file_size_mb']} MB"
                    )
                    st.markdown(
                        f"**Frames extracted:** {T}"
                    )

    # ---- Frames display ----
    with col_frames:
        st.subheader("🎬 Extracted Frames")

        if not detect or uploaded is None:
            st.info(
                "Extracted frames will appear here "
                "after running detection."
            )
        elif frames and PIL_AVAILABLE:
            n_per_row = 5
            rows = [
                frames[i:i + n_per_row]
                for i in range(0, len(frames), n_per_row)
            ]
            for row in rows:
                cols = st.columns(len(row))
                for col, frame, idx in zip(
                    cols,
                    row,
                    range(len(frames))
                ):
                    with col:
                        st.image(
                            frame,
                            caption=f'F{idx+1}',
                            use_column_width=True
                        )
        else:
            st.warning(
                "Frame display requires OpenCV and PIL. "
                "Detection still ran in demo mode."
            )

    # ---- About expander ----
    st.divider()
    with st.expander("ℹ️ About This Application"):
        st.markdown("""
        ### HCI Deepfake Detection — COS590 Final Project

        Developed for the Human-Computer Interaction
        course at Full Sail University.

        **Model Architecture:**
        - CNN Backbone: MobileNetV2 (ImageNet pretrained)
        - Temporal Model: Bidirectional LSTM
          (hidden=256)
        - Input: T evenly spaced frames per video clip
        - Output: Binary classification (REAL / FAKE)

        **Dataset:** FaceForensics++ c23
        (500 real + 500 fake video clips)

        **Key Results (test set):**

        | T | F1 | Latency | FNR |
        |---|---|---|---|
        | 5  | 0.694 | 4.8ms  | 28.7% |
        | 10 | 0.727 | 9.6ms  | 23.3% |
        | 15 | 0.735 | 14.2ms | 22.7% ★ |
        | 20 | 0.731 | 18.8ms | 22.7% |

        ★ Recommended configuration.

        **References:**
        - Rossler et al. (2019). FaceForensics++. ICCV.
        - Sandler et al. (2018). MobileNetV2. CVPR.
        - Masood et al. (2023). Applied Intelligence.
        """)


if __name__ == '__main__':
    main()
