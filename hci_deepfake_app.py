# hci_deepfake_app.py
"""
HCI Deepfake Detection — Streamlit Web Application
COS640 Final Project — Full Sail University

Run locally:   streamlit run hci_deepfake_app.py
"""
import subprocess
import sys

def _ensure_torch():
    try:
        import torch
        return
    except ImportError:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--quiet",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
            "torch==2.1.0+cpu",
            "torchvision==0.16.0+cpu",
        ])

_ensure_torch()
# ─────────────────────────────────────────────────────────────

import streamlit as st
import numpy as np
import time
import os
import tempfile

# ── Imports ──────────────────────────────────────────────────
TORCH_ERROR = None
try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    TORCH_ERROR = str(e)

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

try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="HCI Deepfake Detector",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }

    .box-fake {
        padding: 28px 20px;
        border-radius: 14px;
        background: #fff0f0;
        border: 2.5px solid #c62828;
        text-align: center;
        margin-bottom: 14px;
    }
    .box-real {
        padding: 28px 20px;
        border-radius: 14px;
        background: #f0fff4;
        border: 2.5px solid #1b5e20;
        text-align: center;
        margin-bottom: 14px;
    }
    .box-wait {
        padding: 40px 20px;
        border-radius: 14px;
        background: #fafafa;
        border: 2px dashed #bdbdbd;
        text-align: center;
        margin-bottom: 14px;
    }

    div[data-testid="stButton"] > button {
        background: #1565c0;
        color: #ffffff;
        font-weight: 700;
        font-size: 16px;
        border-radius: 8px;
        border: none;
        padding: 12px 0;
        width: 100%;
        transition: background 0.2s;
    }
    div[data-testid="stButton"] > button:hover {
        background: #0d47a1;
        color: #ffffff;
    }
    div[data-testid="stButton"] > button:disabled {
        background: #90a4ae;
        color: #eceff1;
    }

    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] li {
        color: #cfd8dc;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff;
    }
    [data-testid="stSidebar"] hr {
        border-color: #37474f;
    }

    .footer-text {
        text-align: center;
        color: #9e9e9e;
        font-size: 12px;
        padding: 12px 0 4px 0;
    }

    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────
FRAME_RATES = [5, 10, 15, 20]
IMG_SIZE    = 224
MODEL_DIR   = "models"
MODEL_FILE  = "cnn_lstm_best.pth"
MODEL_PATH  = os.path.join(MODEL_DIR, MODEL_FILE)

GUIDE = {
    5:  dict(latency="4.8ms",  f1="0.694",
             fnr="28.7%", tag="Fast",
             color="#546E7A"),
    10: dict(latency="9.6ms",  f1="0.727",
             fnr="23.3%", tag="Balanced",
             color="#1565C0"),
    15: dict(latency="14.2ms", f1="0.735",
             fnr="22.7%", tag="Optimal ★",
             color="#1B5E20"),
    20: dict(latency="18.8ms", f1="0.731",
             fnr="22.7%", tag="Thorough",
             color="#E65100"),
}

if TORCH_AVAILABLE:
    TRANSFORM = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
else:
    TRANSFORM = None


# ════════════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading detection model…")
def load_model():
    if not TORCH_AVAILABLE:
        import sys
        st.error("PyTorch failed to import.")
        st.code(
            f"Python version:  {sys.version}\n"
            f"Python path:     {sys.executable}\n"
            f"Import error:    {TORCH_ERROR}"
        )
        st.markdown(
            "**Paste the error above here so it "
            "can be diagnosed.**"
        )
        st.stop()

    device = torch.device("cpu")

    # Add project root to path so src.model can be found
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    try:
        from src.model import CNNLSTMModel
    except ImportError as e:
        st.error(
            f"Cannot import CNNLSTMModel: {e}  \n"
            "Make sure src/model.py is in the repository."
        )
        st.stop()

    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file not found: `{MODEL_PATH}`  \n"
            "Make sure `models/cnn_lstm_best.pth` "
            "was pushed to GitHub."
        )
        st.stop()

    try:
        model = CNNLSTMModel(
            sequence_length=10,
            hidden_size=256,
            num_layers=1,
            bidirectional=True,
        ).to(device)

        state = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=False
        )
        model.load_state_dict(state)
        model.eval()

        size_mb = (
            os.path.getsize(MODEL_PATH) / (1024 * 1024)
        )
        return model, device, size_mb

    except Exception as e:
        st.error(f"Failed to load model weights: {e}")
        st.stop()


# ════════════════════════════════════════════════════════════════
# FRAME EXTRACTION
# ════════════════════════════════════════════════════════════════

def extract_frames(uploaded_file, num_frames):
    """
    Extract num_frames evenly spaced frames from a video.
    Returns (list of PIL Images, metadata dict)
    or raises an error via st.error and returns (None, None).
    """
    if not CV2_AVAILABLE:
        st.error(
            "OpenCV not available. "
            "Check packages.txt and requirements.txt."
        )
        return None, None

    if not PIL_AVAILABLE:
        st.error("Pillow not installed.")
        return None, None

    ext = os.path.splitext(
        getattr(uploaded_file, "name", "video.mp4")
    )[-1] or ".mp4"

    # Write uploaded bytes to a temp file
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=ext
        ) as tmp:
            tmp.write(uploaded_file.read())
            path = tmp.name
    except Exception as e:
        st.error(f"Could not save uploaded file: {e}")
        return None, None

    try:
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            st.error(
                "Could not open this video file.  \n"
                "Supported formats: MP4, AVI, MOV, MKV, WebM."
            )
            return None, None

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS)
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total < 1:
            st.error(
                "This video appears to have no frames."
            )
            return None, None

        n       = min(num_frames, total)
        indices = [
            int(k * total / n) for k in range(n)
        ]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(
                    Image.fromarray(
                        cv2.cvtColor(
                            frame, cv2.COLOR_BGR2RGB
                        )
                    )
                )
        cap.release()

        if not frames:
            st.error(
                "No frames could be extracted "
                "from this video."
            )
            return None, None

        meta = {
            "Resolution":   f"{w} × {h}",
            "FPS":          f"{fps:.1f}",
            "Duration":     (
                f"{total / fps:.1f}s"
                if fps > 0 else "N/A"
            ),
            "Total Frames": str(total),
            "Extracted":    str(len(frames)),
            "File Size":    (
                f"{getattr(uploaded_file, 'size', 0) / 1048576:.1f} MB"
            ),
        }
        return frames, meta

    except Exception as e:
        st.error(f"Frame extraction failed: {e}")
        return None, None
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════
# INFERENCE
# ════════════════════════════════════════════════════════════════

def run_inference(model, device, frames, T):
    """
    Run CNN+LSTM deepfake detection.

    Returns:
        label      : 'REAL' or 'FAKE'
        confidence : float — probability of FAKE (0.0–1.0)
        latency_ms : float — inference time in milliseconds
    """
    # Pad or trim frames to exactly T
    seq = list(frames)
    while len(seq) < T:
        seq.append(seq[-1])
    seq = seq[:T]

    # Build tensor (1, T, 3, H, W)
    tensor = torch.stack(
        [TRANSFORM(f) for f in seq]
    ).unsqueeze(0).to(device)

    # Time the inference pass
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
    latency = (time.perf_counter() - t0) * 1000.0

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=1)
    conf  = float(probs[0, 1].cpu())   # P(FAKE)
    label = "FAKE" if conf > 0.5 else "REAL"

    return label, conf, latency


# ════════════════════════════════════════════════════════════════
# UI HELPERS
# ════════════════════════════════════════════════════════════════

def confidence_bar(pct, label):
    """Return styled HTML confidence bar."""
    color = "#c62828" if label == "FAKE" else "#1b5e20"
    return f"""
    <div style="
        background: #e0e0e0;
        border-radius: 8px;
        height: 32px;
        position: relative;
        overflow: hidden;
        margin: 10px 0 4px 0;
    ">
      <div style="
        background: {color};
        width: {pct}%;
        height: 100%;
        opacity: 0.85;
        border-radius: 8px;
      "></div>
      <span style="
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        font-weight: 700;
        font-size: 14px;
        color: #fff;
        text-shadow: 0 1px 3px rgba(0,0,0,.55);
      ">{pct}%</span>
      <div style="
        position: absolute;
        top: 0; left: 50%;
        width: 2px; height: 100%;
        background: rgba(255,255,255,.45);
      "></div>
    </div>
    <div style="
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        color: #757575;
        margin-bottom: 12px;
    ">
      <span>← REAL (0%)</span>
      <span>50% decision threshold</span>
      <span>FAKE (100%) →</span>
    </div>"""


def risk_badge(pct, label):
    """Show contextual risk interpretation."""
    if label == "FAKE":
        if pct >= 80:
            st.error(
                "**Very High Confidence** — "
                "Model is highly confident this video "
                "has been manipulated."
            )
        elif pct >= 65:
            st.warning(
                "**High Confidence** — "
                "Model flags this video as likely fake. "
                "Exercise caution before sharing."
            )
        else:
            st.warning(
                "**Moderate Confidence** — "
                "Model leans toward fake but confidence "
                "is below 65%. Verify with additional sources."
            )
    else:
        if pct <= 20:
            st.success(
                "**Very Low Risk** — "
                "Model is highly confident this video "
                "is authentic."
            )
        elif pct <= 35:
            st.success(
                "**Low Risk** — "
                "Model identifies this video as likely real."
            )
        else:
            st.info(
                "**Borderline** — "
                "Model leans real but confidence is moderate. "
                "Review the extracted frames carefully."
            )


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════

def build_sidebar(size_mb):
    with st.sidebar:
        st.markdown("## HCI Deepfake Detector")
        st.markdown("*COS640 — Full Sail University*")
        st.markdown("**Dylan Brewington**")
        st.divider()

        # Model status
        st.markdown("### Model Status")
        st.success(
            f"CNN+LSTM loaded  \n"
            f"Size: {size_mb:.1f} MB  \n"
            f"Device: CPU  \n"
            f"Backbone: MobileNetV2 + BiLSTM"
        )

        st.divider()

        # Frame rate
        st.markdown("### Frame Rate (T)")
        T = st.select_slider(
            "T",
            options=FRAME_RATES,
            value=15,
            label_visibility="collapsed",
        )
        g = GUIDE[T]
        st.markdown(
            f"<div style='"
            f"background:{g['color']}22;"
            f"border:1px solid {g['color']};"
            f"border-radius:8px;"
            f"padding:10px 14px;'>"
            f"<b style='color:{g['color']};font-size:14px;'>"
            f"T = {T} — {g['tag']}</b><br>"
            f"<span style='font-size:12px;'>"
            f"F1: <b>{g['f1']}</b>  |  "
            f"Latency: <b>{g['latency']}</b><br>"
            f"FNR: <b>{g['fnr']}</b>"
            f"</span></div>",
            unsafe_allow_html=True
        )

        st.divider()

        # Metadata
        st.markdown("### Metadata (optional)")
        compression = st.selectbox(
            "Compression level",
            ["raw", "c23", "c40"],
            index=1,
            help=(
                "The compression level of the input video. "
                "c23 is moderate, c40 is heavy compression."
            )
        )
        source = st.selectbox(
            "Video source",
            ["youtube", "actors"],
            index=0,
            help="The origin type of the video content."
        )

        st.divider()

        # Performance guide
        st.markdown("### Performance Guide")
        if PANDAS_AVAILABLE:
            df = pd.DataFrame(
                {
                    "F1":   ["0.694","0.727",
                              "0.735","0.731"],
                    "ms":   ["4.8","9.6","14.2","18.8"],
                    "FNR":  ["28.7%","23.3%",
                              "22.7%","22.7%"],
                    "":     ["","","★",""],
                },
                index=pd.Index(
                    [5, 10, 15, 20], name="T"
                ),
            )
            st.dataframe(df, use_container_width=True)
        st.caption(
            "★ = recommended configuration  \n"
            "FNR = False Negative Rate  \n"
            "All latencies satisfy < 200ms target"
        )

        st.divider()

        # Architecture summary
        st.markdown("### Architecture")
        st.markdown(
            "- **CNN:** MobileNetV2 (ImageNet)  \n"
            "- **RNN:** Bidirectional LSTM  \n"
            "- **Hidden size:** 256 per direction  \n"
            "- **Parameters:** ~5.4M  \n"
            "- **Dataset:** FaceForensics++ c23  \n"
            "- **Best F1:** 0.735 at T=15  \n"
            "- **Hypothesis:** VALIDATED ✓"
        )

    return T, compression, source


# ════════════════════════════════════════════════════════════════
# PERFORMANCE CHART
# ════════════════════════════════════════════════════════════════

def draw_performance_chart():
    """Matplotlib F1 vs Latency chart."""
    if not MPL_AVAILABLE:
        return

    fig, ax1 = plt.subplots(
        figsize=(6, 3.2),
        facecolor="#FAFAFA"
    )
    ax1.set_facecolor("#F8F9FA")
    ax2 = ax1.twinx()

    ts   = [5, 10, 15, 20]
    f1s  = [0.694, 0.727, 0.735, 0.731]
    lats = [4.8, 9.6, 14.2, 18.8]

    ax1.plot(
        ts, f1s, "o-",
        color="#1B5E20", lw=2.2,
        markersize=7, label="F1 Score", zorder=4
    )
    ax2.plot(
        ts, lats, "s--",
        color="#1565C0", lw=2.2,
        markersize=6, label="Latency (ms)", zorder=4
    )

    # Highlight optimal T=15
    ax1.scatter(
        [15], [0.735],
        color="#F59E0B", s=180, zorder=6,
        edgecolors="black", linewidth=1.2,
        label="Optimal T=15"
    )
    ax1.annotate(
        "T=15\noptimal",
        xy=(15, 0.735),
        xytext=(16.2, 0.727),
        fontsize=8, color="#1B5E20",
        fontweight="bold",
        arrowprops=dict(
            arrowstyle="->",
            color="#1B5E20", lw=1.0
        )
    )

    # 200ms reference line on right axis
    ax2.axhline(
        y=200, color="#B71C1C",
        lw=1.2, linestyle=":",
        alpha=0.6, label="200ms limit"
    )

    ax1.set_xlabel("Sequence Length (T frames)", fontsize=9)
    ax1.set_ylabel("F1 Score", fontsize=9,
                   color="#1B5E20")
    ax2.set_ylabel("Latency (ms)", fontsize=9,
                   color="#1565C0")
    ax1.set_xticks(ts)
    ax1.set_xlim(3.5, 22)
    ax1.set_ylim(0.67, 0.75)
    ax2.set_ylim(0, 220)
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.set_axisbelow(True)
    for sp in ax1.spines.values():
        sp.set_linewidth(0.7)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(
        h1 + h2, l1 + l2,
        fontsize=8, loc="lower right",
        framealpha=0.9
    )
    plt.tight_layout(pad=0.4)
    return fig


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():

    # Load model — stops the app with st.error if it fails
    model, device, size_mb = load_model()

    # Build sidebar
    T, compression, source = build_sidebar(size_mb)

    # ── Page header ──────────────────────────────────────────
    st.title("Real-Time Deepfake Detection")
    st.markdown(
        "Upload a video clip and click **Run Detection** "
        "to analyse it using a hybrid **CNN + LSTM** "
        "deep learning model. "
        "Adjust the frame rate **T** in the sidebar to "
        "control the accuracy-latency tradeoff."
    )
    st.divider()

    # ── Upload ───────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help=(
            "Upload a video clip to detect whether it "
            "has been manipulated using deepfake techniques."
        )
    )

    if uploaded:
        c_name, c_size, c_type = st.columns(3)
        c_name.info(f"**File:** {uploaded.name}")
        c_size.info(
            f"**Size:** "
            f"{uploaded.size / 1048576:.1f} MB"
        )
        c_type.info(f"**T setting:** {T} frames")

    run_btn = st.button(
        "Run Detection",
        disabled=(uploaded is None),
        use_container_width=True,
    )
    st.divider()

    # ── Results layout ───────────────────────────────────────
    col_left, col_right = st.columns(
        [1, 1.6], gap="large"
    )

    # ──────────────────────────────────────────────────────────
    # LEFT — Result card
    # ──────────────────────────────────────────────────────────
    with col_left:
        st.subheader("Detection Result")

        # Waiting state
        if not run_btn or uploaded is None:
            st.markdown(
                '<div class="box-wait">'
                '<p style="font-size:32px;margin:0;">🎭</p>'
                '<h3 style="color:#9e9e9e;margin:8px 0 4px;">'
                'Awaiting Video Input</h3>'
                '<p style="color:#bdbdbd;margin:0;'
                'font-size:14px;">'
                'Upload a video and click Run Detection'
                '</p></div>',
                unsafe_allow_html=True,
            )
            return

        # ── Extract frames ───────────────────────────────────
        with st.spinner(f"Extracting {T} frames…"):
            uploaded.seek(0)
            frames, meta = extract_frames(uploaded, T)

        if frames is None:
            return   # error already shown by extract_frames

        # ── Run inference ────────────────────────────────────
        with st.spinner(
            "Running CNN+LSTM detection model…"
        ):
            label, conf, latency = run_inference(
                model, device, frames, T
            )

        pct   = int(conf * 100)
        color = (
            "#c62828" if label == "FAKE"
            else "#1b5e20"
        )
        css   = (
            "box-fake" if label == "FAKE"
            else "box-real"
        )
        icon  = "⚠️" if label == "FAKE" else "✅"

        # Result box
        st.markdown(
            f'<div class="{css}">'
            f'<p style="font-size:36px;margin:0;">'
            f'{icon}</p>'
            f'<h2 style="color:{color};margin:6px 0 2px;">'
            f'{label}</h2>'
            f'<p style="color:{color};font-weight:700;'
            f'font-size:17px;margin:0 0 6px 0;">'
            f'Confidence: {pct}%</p>'
            f'<p style="color:#616161;'
            f'font-size:13px;margin:0;">'
            f'T = {T} frames  |  '
            f'{latency:.1f}ms inference  |  '
            f'{compression.upper()}  |  '
            f'{source.capitalize()}'
            f'</p></div>',
            unsafe_allow_html=True,
        )

        # Confidence bar
        st.markdown(
            confidence_bar(pct, label),
            unsafe_allow_html=True
        )

        # Metric row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Result",     label)
        m2.metric("Confidence", f"{pct}%")
        m3.metric("Latency",    f"{latency:.1f}ms")
        m4.metric("Frames",     f"T = {T}")

        # Risk interpretation
        risk_badge(pct, label)

        # ── Video metadata ───────────────────────────────────
        st.subheader("Video Information")
        if meta:
            col_a, col_b = st.columns(2)
            items = list(meta.items())
            half  = len(items) // 2
            with col_a:
                for k, v in items[:half + 1]:
                    st.markdown(f"**{k}:** {v}")
            with col_b:
                for k, v in items[half + 1:]:
                    st.markdown(f"**{k}:** {v}")

    # ──────────────────────────────────────────────────────────
    # RIGHT — Frames
    # ──────────────────────────────────────────────────────────
    with col_right:
        st.subheader("Extracted Frames")

        if frames and PIL_AVAILABLE:
            cols_per_row = 5
            for start in range(
                0, len(frames), cols_per_row
            ):
                chunk = frames[
                    start: start + cols_per_row
                ]
                cols = st.columns(len(chunk))
                for col, frm, idx in zip(
                    cols,
                    chunk,
                    range(start, start + len(chunk))
                ):
                    col.image(
                        frm,
                        caption=f"Frame {idx + 1}",
                        use_column_width=True,
                    )

            st.caption(
                f"{len(frames)} frames extracted at "
                f"evenly spaced intervals across the clip. "
                f"Each frame is resized to "
                f"224 × 224 for the CNN backbone."
            )

            # ── Frame prediction confidence per frame ─────────
            st.subheader("Per-Frame CNN Features")
            st.markdown(
                "The model processes all frames as a "
                "sequence. The bidirectional LSTM "
                "integrates temporal context across "
                f"all {T} frames before making the "
                "final prediction."
            )

            # Show individual frame thumbnails with index
            if len(frames) <= 10:
                thumb_cols = st.columns(len(frames))
                for i, (col, frm) in enumerate(
                    zip(thumb_cols, frames)
                ):
                    col.image(
                        frm.resize((80, 80)),
                        use_column_width=True
                    )
                    col.markdown(
                        f"<p style='text-align:center;"
                        f"font-size:10px;color:#9e9e9e;'>"
                        f"t={i+1}</p>",
                        unsafe_allow_html=True
                    )
        else:
            st.warning("Could not display frames.")

    # ──────────────────────────────────────────────────────────
    # BOTTOM — Analysis tabs
    # ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Model Performance Reference")

    tab1, tab2, tab3 = st.tabs([
        "Frame Rate Analysis",
        "Hypothesis Validation",
        "Architecture Details"
    ])

    # Tab 1 — Frame rate analysis
    with tab1:
        col_tbl, col_chart = st.columns([1, 1.2])

        with col_tbl:
            if PANDAS_AVAILABLE:
                df = pd.DataFrame(
                    {
                        "F1 Score":  [
                            0.694, 0.727, 0.735, 0.731
                        ],
                        "Accuracy":  [
                            "69.3%","72.7%",
                            "73.3%","73.3%"
                        ],
                        "FNR":       [
                            "28.7%","23.3%",
                            "22.7%","22.7%"
                        ],
                        "Latency":   [
                            "4.8ms","9.6ms",
                            "14.2ms","18.8ms"
                        ],
                        "Best":      ["","","★",""],
                    },
                    index=pd.Index(
                        [5, 10, 15, 20], name="T"
                    ),
                )
                st.dataframe(
                    df, use_container_width=True
                )
                st.caption(
                    "FNR = False Negative Rate  |  "
                    "★ = Recommended  |  "
                    "All latencies < 200ms"
                )

        with col_chart:
            fig = draw_performance_chart()
            if fig:
                st.pyplot(fig)
                plt.close()

    # Tab 2 — Hypothesis validation
    with tab2:
        st.markdown(
            "> *The CNN+LSTM architecture will outperform "
            "the CNN baseline by at least 5 percentage "
            "points in F1 score while maintaining "
            "sub-200ms per-frame inference latency on "
            "consumer GPU hardware.*"
        )
        st.markdown("")

        crit_data = [
            (
                "F1 Improvement ≥ 5 pts",
                "≥ +5.0 pts",
                "+5.3 pts  (0.674 → 0.727)",
                True
            ),
            (
                "CNN+LSTM Latency < 200ms",
                "< 200ms",
                "9.64ms  (20× under target)",
                True
            ),
            (
                "CNN Baseline Latency < 200ms",
                "< 200ms",
                "3.56ms  (56× under target)",
                True
            ),
        ]

        for criterion, target, result, passed in crit_data:
            c1, c2, c3, c4 = st.columns([2.5, 1, 2, 1])
            c1.markdown(f"**{criterion}**")
            c2.markdown(f"`{target}`")
            c3.markdown(f"**{result}**")
            if passed:
                c4.success("VALIDATED")
            else:
                c4.error("FAILED")

        st.markdown("")
        st.success(
            "**Overall Hypothesis: VALIDATED** — "
            "All three criteria satisfied."
        )

    # Tab 3 — Architecture details
    with tab3:
        col_arch1, col_arch2 = st.columns(2)

        with col_arch1:
            st.markdown("""
**CNN Backbone — MobileNetV2**
- Pretrained on ImageNet
- Produces 1280-dim feature per frame
- Applied independently to each of T frames
- ~3.2M parameters shared with both paths

**Bidirectional LSTM**
- Hidden size: 256 per direction
- Output: 512-dim at final timestep
- Processes forward + backward temporal context
- Gradient clipping: max norm = 1.0
            """)

        with col_arch2:
            st.markdown("""
**Fully Connected Classifier**
- Dropout(0.5) → Linear(512→128)
- ReLU → Dropout(0.3)
- Linear(128→2) → Softmax

**Training Details**
- Dataset: FaceForensics++ c23
- Split: 70% train / 15% val / 15% test
- Optimizer: Adam (lr=1e-4)
- Scheduler: StepLR (step=5, γ=0.5)
- Epochs: 30
- Loss: CrossEntropyLoss
            """)

        st.markdown(
            "**Input pipeline:**  "
            "Video → Extract T frames → "
            "Resize 224×224 → "
            "Normalize (ImageNet stats) → "
            "CNN features → LSTM → "
            "Classifier → REAL / FAKE"
        )

    # ── Footer ───────────────────────────────────────────────
    st.divider()
    st.markdown(
        '<p class="footer-text">'
        "HCI Deepfake Detector  |  "
        "COS640 Human-Computer Interaction  |  "
        "Full Sail University  |  "
        "Dylan Brewington  |  "
        "Streamlit + PyTorch + MobileNetV2 + BiLSTM"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()