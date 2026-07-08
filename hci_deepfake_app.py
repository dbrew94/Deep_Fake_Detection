# hci_deepfake_app.py
"""
HCI Deepfake Detection — Streamlit Web Application
COS590 Final Project — Full Sail University

Deploy locally:   streamlit run hci_deepfake_app.py
Deploy to cloud:  push to GitHub then connect to share.streamlit.io

All imports are handled safely so the app runs in demo mode
even if PyTorch or OpenCV are not yet installed on the server.
"""

# hci_deepfake_app.py
"""
HCI Deepfake Detection — Streamlit Web Application
COS590 Final Project — Full Sail University

Run locally:   streamlit run hci_deepfake_app.py
"""

import streamlit as st
import numpy as np
import time
import os
import sys
import tempfile

# ── Safe imports ────────────────────────────────────────────
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

try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

# ── Page config — must be first ──────────────────────────────
st.set_page_config(
    page_title="HCI Deepfake Detector",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — minimal, no color leaking ─────────────────────────
st.markdown(
    """
    <style>
        /* Remove Streamlit default top padding */
        .block-container { padding-top: 1.5rem; }

        /* Detect result boxes */
        .box-fake {
            background: #fff0f0;
            border: 2px solid #c62828;
            border-radius: 12px;
            padding: 22px 18px;
            text-align: center;
            margin-bottom: 12px;
        }
        .box-real {
            background: #f0fff4;
            border: 2px solid #1b5e20;
            border-radius: 12px;
            padding: 22px 18px;
            text-align: center;
            margin-bottom: 12px;
        }
        .box-wait {
            background: #f9f9f9;
            border: 2px dashed #bdbdbd;
            border-radius: 12px;
            padding: 22px 18px;
            text-align: center;
            margin-bottom: 12px;
        }

        /* Primary button */
        div[data-testid="stButton"] > button {
            background: #1565c0;
            color: #ffffff;
            font-weight: 700;
            font-size: 15px;
            border-radius: 8px;
            border: none;
            padding: 10px 0;
            width: 100%;
        }
        div[data-testid="stButton"] > button:hover {
            background: #0d47a1;
            color: #ffffff;
        }
        div[data-testid="stButton"] > button:disabled {
            background: #b0bec5;
            color: #eceff1;
        }

        /* Sidebar — scoped tightly so it doesn't bleed */
        [data-testid="stSidebar"] {
            background-color: #1a1a2e;
        }
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] small,
        [data-testid="stSidebar"] caption {
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
        [data-testid="stSidebar"] .stSelectbox > div,
        [data-testid="stSidebar"] .stSlider > div {
            color: #cfd8dc;
        }

        /* Footer text */
        .footer-text {
            text-align: center;
            color: #9e9e9e;
            font-size: 12px;
            padding: 12px 0 4px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ────────────────────────────────────────────────
FRAME_RATES = [5, 10, 15, 20]
IMG_SIZE    = 224
MODEL_PATH  = "models/cnn_lstm_best.pth"

GUIDE = {
    5:  dict(latency="4.8 ms",  f1="0.694",
             fnr="28.7 %", tag="Fast"),
    10: dict(latency="9.6 ms",  f1="0.727",
             fnr="23.3 %", tag="Balanced"),
    15: dict(latency="14.2 ms", f1="0.735",
             fnr="22.7 %", tag="Optimal ★"),
    20: dict(latency="18.8 ms", f1="0.731",
             fnr="22.7 %", tag="Thorough"),
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


# ── Model loading ────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not TORCH_AVAILABLE:
        return None, None, False, "PyTorch not installed"

    device = torch.device("cpu")

    try:
        sys.path.insert(
            0, os.path.dirname(os.path.abspath(__file__))
        )
        from src.model import CNNLSTMModel
    except ImportError as e:
        return None, device, False, f"Import error: {e}"

    if not os.path.exists(MODEL_PATH):
        try:
            from huggingface_hub import hf_hub_download
            os.makedirs("models", exist_ok=True)
            hf_hub_download(
                repo_id="YOUR_HF_USERNAME/deepfake-detection",
                filename="cnn_lstm_best.pth",
                local_dir="models",
            )
        except Exception as e:
            return None, device, False, (
                f"Model not found — download failed: {e}"
            )

    try:
        model = CNNLSTMModel(
            sequence_length=10,
            hidden_size=256,
            num_layers=1,
            bidirectional=True,
        ).to(device)
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model, device, True, "Model ready"
    except Exception as e:
        return None, device, False, f"Load error: {e}"


# ── Frame extraction ─────────────────────────────────────────
def extract_frames(uploaded_file, num_frames):
    if not CV2_AVAILABLE or not PIL_AVAILABLE:
        return None, None

    ext = os.path.splitext(
        getattr(uploaded_file, "name", "video.mp4")
    )[-1] or ".mp4"

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=ext
        ) as tmp:
            tmp.write(uploaded_file.read())
            path = tmp.name
    except Exception:
        return None, None

    try:
        cap   = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS)
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not cap.isOpened() or total < 1:
            cap.release()
            return None, None

        n       = min(num_frames, total)
        indices = [int(k * total / n) for k in range(n)]
        frames  = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(
                    Image.fromarray(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )
                )
        cap.release()

        meta = {
            "Resolution": f"{w} x {h}",
            "FPS":        str(round(fps, 1)),
            "Duration":   f"{round(total/fps, 1)}s"
                          if fps > 0 else "N/A",
            "Frames":     str(total),
            "File size":  f"{getattr(uploaded_file,'size',0)
                             / 1048576:.1f} MB",
        }
        return frames or None, meta

    except Exception:
        return None, None
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


# ── Inference ────────────────────────────────────────────────
def run_inference(model, device, frames, T):
    if model is None or not TORCH_AVAILABLE or not frames:
        time.sleep(0.5)
        c = float(np.random.uniform(0.18, 0.96))
        return ("FAKE" if c > 0.5 else "REAL"), c, 0.0, True

    try:
        seq = list(frames)
        while len(seq) < T:
            seq.append(seq[-1])
        seq = seq[:T]

        tensor = torch.stack(
            [TRANSFORM(f) for f in seq]
        ).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor)
        lat = (time.perf_counter() - t0) * 1000

        probs = torch.softmax(logits, dim=1)
        conf  = float(probs[0, 1].cpu())
        return ("FAKE" if conf > 0.5 else "REAL"), conf, lat, False

    except Exception as e:
        st.error(f"Inference error: {e}")
        c = float(np.random.uniform(0.2, 0.9))
        return ("FAKE" if c > 0.5 else "REAL"), c, 0.0, True


# ── Confidence bar ───────────────────────────────────────────
def conf_bar(pct, label):
    color = "#c62828" if label == "FAKE" else "#1b5e20"
    return f"""
    <div style="background:#e0e0e0;border-radius:8px;
                height:30px;position:relative;
                overflow:hidden;margin:8px 0 2px;">
      <div style="background:{color};width:{pct}%;
                  height:100%;opacity:.80;
                  border-radius:8px;"></div>
      <span style="position:absolute;top:50%;left:50%;
                   transform:translate(-50%,-50%);
                   font-weight:700;font-size:14px;
                   color:#fff;
                   text-shadow:0 1px 3px rgba(0,0,0,.6);">
        {pct}%
      </span>
      <div style="position:absolute;top:0;left:50%;
                  width:2px;height:100%;
                  background:rgba(255,255,255,.5);"></div>
    </div>
    <div style="display:flex;justify-content:space-between;
                font-size:11px;color:#757575;
                margin-bottom:10px;">
      <span>REAL — 0 %</span>
      <span>50 % threshold</span>
      <span>100 % — FAKE</span>
    </div>"""


# ── Sidebar ──────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("## HCI Deepfake Detector")
        st.markdown(
            "*COS590 — Full Sail University*"
        )
        st.divider()

        st.markdown("### Frame Rate (T)")
        T = st.select_slider(
            "T",
            options=FRAME_RATES,
            value=10,
            label_visibility="collapsed",
        )
        g = GUIDE[T]
        st.markdown(
            f"**T = {T}** — {g['tag']}  \n"
            f"F1: `{g['f1']}`  |  "
            f"Latency: `{g['latency']}`  \n"
            f"False Negative Rate: `{g['fnr']}`"
        )

        st.divider()
        st.markdown("### Metadata (optional)")
        compression = st.selectbox(
            "Compression level",
            ["raw", "c23", "c40"],
            index=1,
        )
        source = st.selectbox(
            "Video source",
            ["youtube", "actors"],
            index=0,
        )

        st.divider()
        st.markdown("### Performance Guide")
        if PANDAS_AVAILABLE:
            df = pd.DataFrame(
                {
                    "F1":     ["0.694","0.727","0.735","0.731"],
                    "ms":     ["4.8","9.6","14.2","18.8"],
                    "FNR":    ["28.7%","23.3%","22.7%","22.7%"],
                    "Note":   ["","","★",""],
                },
                index=pd.Index([5, 10, 15, 20], name="T"),
            )
            st.dataframe(df, use_container_width=True)
        st.caption("★ = recommended  |  all < 200 ms")

    return T, compression, source


# ── Main ─────────────────────────────────────────────────────
def main():

    model, device, loaded, msg = load_model()
    T, compression, source     = sidebar()

    # ── Header ──────────────────────────────────────────────
    st.title("Real-Time Deepfake Detection")
    st.markdown(
        "Upload a video clip and press **Run Detection** "
        "to analyse it with a hybrid CNN + LSTM model. "
        "Use the sidebar to adjust the frame rate."
    )

    # Status pill
    if loaded:
        st.success(f"Model ready  |  CPU inference")
    else:
        st.info(
            "**Demo Mode** — model or dependencies not yet "
            "available on this server. Results are simulated."
        )

    st.divider()

    # ── Upload + button ──────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        label_visibility="visible",
    )

    detect = st.button(
        "Run Detection",
        disabled=(uploaded is None),
        use_container_width=True,
    )

    st.divider()

    # ── Results area ─────────────────────────────────────────
    col_left, col_right = st.columns([1, 1.6], gap="large")

    # LEFT — result card
    with col_left:
        st.subheader("Result")

        if not detect or uploaded is None:
            st.markdown(
                '<div class="box-wait">'
                '<h2 style="color:#9e9e9e;margin:0;">—</h2>'
                '<p style="color:#9e9e9e;margin:6px 0 0;">  '
                'Upload a video and click Run Detection'
                '</p></div>',
                unsafe_allow_html=True,
            )
        else:
            # Extract
            with st.spinner(f"Extracting {T} frames…"):
                uploaded.seek(0)
                frames, meta = extract_frames(uploaded, T)

            # Infer
            with st.spinner("Running model…"):
                label, conf, lat, demo = run_inference(
                    model, device, frames or [], T
                )

            pct   = int(conf * 100)
            color = "#c62828" if label == "FAKE" else "#1b5e20"
            icon  = "FAKE" if label == "FAKE" else "REAL"
            css   = "box-fake" if label == "FAKE" else "box-real"
            note  = " (demo)" if demo else ""

            st.markdown(
                f'<div class="{css}">'
                f'<h2 style="color:{color};margin:0;">'
                f"{icon}</h2>"
                f'<p style="color:{color};font-size:18px;'
                f'font-weight:700;margin:6px 0 2px;">'
                f"Confidence: {pct}%{note}</p>"
                f'<p style="color:#616161;font-size:13px;'
                f'margin:0;">'
                f"T={T}  |  "
                f"{'%.1f' % lat + ' ms' if lat else 'demo'}  |  "
                f"{compression}  |  {source}"
                f"</p></div>",
                unsafe_allow_html=True,
            )

            st.markdown(conf_bar(pct, label),
                        unsafe_allow_html=True)

            # Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Result", label)
            c2.metric("Confidence", f"{pct}%")
            c3.metric(
                "Latency",
                f"{lat:.1f} ms" if lat else "demo",
            )

            # Risk note
            if pct >= 75:
                st.error(
                    "High confidence — likely manipulated."
                )
            elif pct >= 50:
                st.warning(
                    "Moderate confidence — treat with caution."
                )
            elif pct >= 30:
                st.info(
                    "Low confidence — leans real."
                )
            else:
                st.success(
                    "Very low risk — likely authentic."
                )

            # Video metadata
            if meta:
                st.subheader("Video Info")
                for k, v in meta.items():
                    st.markdown(f"**{k}:** {v}")

    # RIGHT — frames
    with col_right:
        st.subheader("Extracted Frames")

        if not detect or uploaded is None:
            st.info(
                "Frames will appear here after detection."
            )

        elif frames and PIL_AVAILABLE:
            cols_per_row = 5
            for start in range(0, len(frames), cols_per_row):
                chunk = frames[start: start + cols_per_row]
                cols  = st.columns(len(chunk))
                for col, frm, idx in zip(
                    cols, chunk,
                    range(start, start + len(chunk))
                ):
                    col.image(
                        frm,
                        caption=f"F{idx+1}",
                        use_column_width=True,
                    )
            st.caption(
                f"{len(frames)} frames extracted "
                f"at even intervals."
            )

        else:
            st.warning(
                "Frame display unavailable in demo mode."
            )

    # ── Performance table + chart ────────────────────────────
    st.divider()
    st.subheader("Frame Rate Tradeoff")

    tab_table, tab_chart = st.tabs(["Table", "Chart"])

    with tab_table:
        if PANDAS_AVAILABLE:
            df = pd.DataFrame(
                {
                    "F1 Score":  [0.694, 0.727, 0.735, 0.731],
                    "Accuracy":  ["69.3%","72.7%","73.3%","73.3%"],
                    "FNR":       ["28.7%","23.3%","22.7%","22.7%"],
                    "Latency":   ["4.8 ms","9.6 ms",
                                  "14.2 ms","18.8 ms"],
                    "Recommended": ["","","Yes ★",""],
                },
                index=pd.Index([5, 10, 15, 20], name="T"),
            )
            st.dataframe(df, use_container_width=True)
            st.caption(
                "FNR = False Negative Rate.  "
                "All latencies satisfy the < 200 ms target."
            )

    with tab_chart:
        if MPL_AVAILABLE:
            fig, ax1 = plt.subplots(figsize=(6, 3.2))
            ax2 = ax1.twinx()

            ts   = [5, 10, 15, 20]
            f1s  = [0.694, 0.727, 0.735, 0.731]
            lats = [4.8, 9.6, 14.2, 18.8]

            ax1.plot(
                ts, f1s, "o-",
                color="#1b5e20", lw=2,
                markersize=7, label="F1 Score",
            )
            ax2.plot(
                ts, lats, "s--",
                color="#1565c0", lw=2,
                markersize=6, label="Latency (ms)",
            )
            ax1.scatter(
                [15], [0.735],
                color="#f59e0b", s=160, zorder=5,
                edgecolors="black", linewidth=1.2,
            )
            ax1.annotate(
                "Optimal (T=15)",
                xy=(15, 0.735),
                xytext=(16, 0.728),
                fontsize=8, color="#1b5e20",
                fontweight="bold",
                arrowprops=dict(
                    arrowstyle="->",
                    color="#1b5e20", lw=1,
                ),
            )
            ax1.set_xlabel("Sequence Length (T)", fontsize=9)
            ax1.set_ylabel("F1 Score", fontsize=9,
                           color="#1b5e20")
            ax2.set_ylabel("Latency (ms)", fontsize=9,
                           color="#1565c0")
            ax1.set_xticks(ts)
            ax1.set_ylim(0.66, 0.76)
            ax2.set_ylim(0, 25)
            ax1.grid(alpha=0.25, linestyle="--")
            ax1.set_axisbelow(True)

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(
                h1 + h2, l1 + l2,
                fontsize=8, loc="lower right",
            )
            plt.tight_layout(pad=0.4)
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Install matplotlib to see the chart.")

    # ── Footer ───────────────────────────────────────────────
    st.divider()
    st.markdown(
        '<p class="footer-text">'
        "HCI Deepfake Detector &nbsp;|&nbsp; "
        "COS590 Human-Computer Interaction &nbsp;|&nbsp; "
        "Full Sail University &nbsp;|&nbsp; "
        "Streamlit + PyTorch + MobileNetV2"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
