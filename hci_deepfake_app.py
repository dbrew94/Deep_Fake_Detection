# hci_deepfake_app.py
"""
COS640 - HCI Deepfake Detection Application
Dylan Brewington
Prototype graphical user interface implementing:
  - Configurable frame rate selection (T = 5, 10, 15, 20)
  - Optional contextual metadata input
  - Real-time detection with confidence visualization

Requirements:
  - models/cnn_lstm_best.pth must exist (trained in main project)
  - frames/ directory must exist with real/ and fake/ subfolders
  - pip install opencv-python pillow torch torchvision

Run:
  python hci_deepfake_app.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
import sys
import numpy as np

# Guard imports that may fail
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import torch
    import torchvision.transforms as transforms
    from PIL import Image, ImageTk
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ==============================================================
# CONSTANTS
# ==============================================================

FRAME_RATES     = [5, 10, 15, 20]
DEFAULT_T       = 10
IMG_SIZE        = 224
MODEL_PATH      = 'cnn_lstm_best.pth'
WINDOW_TITLE    = 'HCI Deepfake Detector — COS640 Prototype'
WINDOW_W        = 820
WINDOW_H        = 640

COLOR_BG        = '#1A1A2E'
COLOR_PANEL     = '#0D1117'
COLOR_ACCENT    = '#2E8B57'
COLOR_CNN       = '#2979B8'
COLOR_REAL      = '#27AE60'
COLOR_FAKE      = '#E74C3C'
COLOR_NEUTRAL   = '#5D6D7E'
COLOR_TEXT      = '#ECF0F1'
COLOR_SUBTEXT   = '#95A5A6'
COLOR_GOLD      = '#F1C40F'
COLOR_BORDER    = '#2C3E50'


# ==============================================================
# MODEL IMPORT (lazy — graceful if project files missing)
# ==============================================================

def load_detection_model(device, metadata_dim=0):
    """
    Load the CNN+LSTM model from the existing project.
    Falls back to a stub if models are unavailable.
    Returns (model, transform) tuple.
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from src.metadata_model import CNNLSTMWithMetadata
        model = CNNLSTMWithMetadata(
            sequence_length=10,
            hidden_size=256,
            num_layers=1,
            bidirectional=True,
            metadata_dim=metadata_dim
        ).to(device)
        state = torch.load(MODEL_PATH, map_location=device)
        # Load only matching keys (metadata head may differ)
        model_dict = model.state_dict()
        filtered   = {
            k: v for k, v in state.items()
            if k in model_dict and
            model_dict[k].shape == v.shape
        }
        model_dict.update(filtered)
        model.load_state_dict(model_dict, strict=False)
        model.eval()
        print(f"Model loaded: {MODEL_PATH}")
        print(
            f"  Loaded {len(filtered)}/{len(state)} "
            f"weight tensors"
        )
    except Exception as e:
        print(f"WARNING: Could not load model — {e}")
        print("Running in DEMO MODE with random predictions.")
        model = None

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return model, transform


# ==============================================================
# FRAME EXTRACTION
# ==============================================================

def extract_frames_from_video(video_path, num_frames):
    """
    Extract num_frames evenly spaced frames from a video file.
    Returns list of PIL Images, or None on failure.
    """
    if not CV2_AVAILABLE:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < num_frames:
        num_frames = total

    indices = [
        int(k * total / num_frames)
        for k in range(num_frames)
    ]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img   = Image.fromarray(frame_rgb)
            frames.append(pil_img)

    cap.release()
    return frames if len(frames) == num_frames else None


def get_video_metadata(video_path):
    """
    Extract contextual metadata from a video file.
    Returns dict with codec, fps, resolution, duration.
    """
    meta = {
        'filename':    os.path.basename(video_path),
        'filesize_mb': round(
            os.path.getsize(video_path) / (1024 * 1024), 2
        ),
        'fps':         'N/A',
        'resolution':  'N/A',
        'duration_s':  'N/A',
        'frame_count': 'N/A',
        'codec':       'N/A',
    }

    if not CV2_AVAILABLE:
        return meta

    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        meta['fps'] = round(
            cap.get(cv2.CAP_PROP_FPS), 2
        )
        meta['resolution'] = (
            f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} × "
            f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        )
        meta['frame_count'] = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        fps = cap.get(cv2.CAP_PROP_FPS)
        fc  = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps > 0:
            meta['duration_s'] = round(fc / fps, 2)
        # Codec (fourcc)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        meta['codec'] = (
            chr(fourcc & 0xFF) +
            chr((fourcc >> 8)  & 0xFF) +
            chr((fourcc >> 16) & 0xFF) +
            chr((fourcc >> 24) & 0xFF)
        ).strip()
        cap.release()
    return meta


def build_metadata_vector(compression, source_type,
                           resolution_str):
    """
    Build a 7-dimensional metadata vector.
    compression:   'raw' | 'c23' | 'c40'
    source_type:   'youtube' | 'actors'
    resolution_str: 'WxH' string from video metadata
    """
    # Compression one-hot (3-dim): raw, c23, c40
    comp_map = {'raw': 0, 'c23': 1, 'c40': 2}
    comp_idx  = comp_map.get(compression.lower(), 1)
    comp_vec  = [0.0, 0.0, 0.0]
    comp_vec[comp_idx] = 1.0

    # Source one-hot (2-dim): youtube, actors
    src_map  = {'youtube': 0, 'actors': 1}
    src_idx  = src_map.get(source_type.lower(), 0)
    src_vec  = [0.0, 0.0]
    src_vec[src_idx] = 1.0

    # Frame dimensions (2-dim normalized)
    try:
        parts = resolution_str.replace(' ', '').split('×')
        w = float(parts[0]) / 1920.0
        h = float(parts[1]) / 1080.0
    except Exception:
        w, h = 0.5, 0.5

    return comp_vec + src_vec + [w, h]


def run_inference(model, transform, frames, metadata_vec,
                  device, sequence_length):
    """
    Run deepfake detection on extracted frames.
    Returns (label, confidence, latency_ms).
    """
    if model is None or not TORCH_AVAILABLE:
        # Demo mode: random prediction
        time.sleep(0.5)
        conf  = float(np.random.uniform(0.3, 0.95))
        label = 'FAKE' if conf > 0.5 else 'REAL'
        return label, conf, 42.0

    # Pad or truncate to sequence_length
    while len(frames) < sequence_length:
        frames.append(frames[-1])
    frames = frames[:sequence_length]

    # Build tensor (1, T, 3, H, W)
    tensors = [transform(f) for f in frames]
    seq     = torch.stack(tensors).unsqueeze(0).to(device)

    # Metadata tensor
    m_tensor = torch.tensor(
        [metadata_vec], dtype=torch.float32
    ).to(device)

    # Timed inference
    if device.type == 'cuda':
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    with torch.no_grad():
        try:
            logits = model(seq, m_tensor)
        except TypeError:
            # Fallback if model doesn't accept metadata
            logits = model(seq)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    t_end      = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000.0

    probs = torch.softmax(logits, dim=1)
    fake_prob = float(probs[0, 1].cpu())
    label     = 'FAKE' if fake_prob > 0.5 else 'REAL'
    return label, fake_prob, latency_ms


# ==============================================================
# GUI APPLICATION
# ==============================================================

class DeepfakeApp:
    """
    Main HCI deepfake detection application.
    Implements the GUI and detection pipeline.
    """

    def __init__(self, root):
        self.root   = root
        self.device = (
            torch.device('cuda')
            if TORCH_AVAILABLE and torch.cuda.is_available()
            else torch.device('cpu')
        ) if TORCH_AVAILABLE else None

        self.video_path  = None
        self.model       = None
        self.transform   = None
        self.is_running  = False

        self._setup_window()
        self._build_ui()
        self._load_model_async()

    # ----------------------------------------------------------
    # WINDOW SETUP
    # ----------------------------------------------------------

    def _setup_window(self):
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f'{WINDOW_W}x{WINDOW_H}')
        self.root.resizable(False, False)
        self.root.configure(bg=COLOR_BG)
        self.root.columnconfigure(0, weight=1)

    # ----------------------------------------------------------
    # UI CONSTRUCTION
    # ----------------------------------------------------------

    def _build_ui(self):
        # ---- Header ----
        hdr = tk.Frame(self.root, bg='#0D1117',
                        height=55)
        hdr.pack(fill='x', padx=0, pady=0)
        hdr.pack_propagate(False)

        tk.Label(
            hdr,
            text='🎭  HCI Deepfake Detector',
            font=('Arial', 15, 'bold'),
            bg='#0D1117', fg=COLOR_TEXT
        ).pack(side='left', padx=18, pady=12)

        self.status_lbl = tk.Label(
            hdr,
            text='Loading model...',
            font=('Arial', 9),
            bg='#0D1117', fg=COLOR_SUBTEXT
        )
        self.status_lbl.pack(side='right', padx=18)

        # ---- Main body: left + right panels ----
        body = tk.Frame(self.root, bg=COLOR_BG)
        body.pack(fill='both', expand=True,
                   padx=12, pady=10)

        left  = tk.Frame(body, bg=COLOR_BG, width=370)
        right = tk.Frame(body, bg=COLOR_BG, width=420)
        left.pack(side='left', fill='y', padx=(0, 6))
        right.pack(side='right', fill='both', expand=True)
        left.pack_propagate(False)

        self._build_left_panel(left)
        self._build_right_panel(right)

    def _section(self, parent, title):
        """Create a titled section frame."""
        outer = tk.Frame(parent, bg=COLOR_BORDER,
                          bd=0)
        outer.pack(fill='x', pady=5)
        tk.Label(
            outer,
            text=f'  {title}',
            font=('Arial', 9, 'bold'),
            bg=COLOR_BORDER, fg=COLOR_GOLD
        ).pack(fill='x', pady=2)
        inner = tk.Frame(outer, bg=COLOR_PANEL)
        inner.pack(fill='x', padx=1, pady=(0, 1))
        return inner

    # ----------------------------------------------------------
    # LEFT PANEL
    # ----------------------------------------------------------

    def _build_left_panel(self, parent):

        # ---- Video Input ----
        sec1 = self._section(parent, 'VIDEO INPUT')

        tk.Button(
            sec1,
            text='📂  Browse Video File',
            font=('Arial', 10, 'bold'),
            bg=COLOR_CNN, fg=COLOR_TEXT,
            activebackground='#1565C0',
            activeforeground=COLOR_TEXT,
            relief='flat', cursor='hand2',
            command=self._browse_video
        ).pack(fill='x', padx=10, pady=8)

        self.file_lbl = tk.Label(
            sec1,
            text='No file selected',
            font=('Arial', 8),
            bg=COLOR_PANEL, fg=COLOR_SUBTEXT,
            wraplength=330, anchor='w'
        )
        self.file_lbl.pack(fill='x', padx=10, pady=(0, 6))

        # ---- Frame Rate ----
        sec2 = self._section(parent, 'FRAME RATE  (T frames per clip)')

        self.t_var = tk.IntVar(value=DEFAULT_T)

        fr_frame = tk.Frame(sec2, bg=COLOR_PANEL)
        fr_frame.pack(fill='x', padx=10, pady=6)

        for t in FRAME_RATES:
            rb = tk.Radiobutton(
                fr_frame,
                text=f'T = {t}',
                variable=self.t_var, value=t,
                font=('Arial', 9),
                bg=COLOR_PANEL, fg=COLOR_TEXT,
                selectcolor='#1A3A5C',
                activebackground=COLOR_PANEL,
                activeforeground=COLOR_TEXT
            )
            rb.pack(side='left', padx=10, pady=4)

        self.t_info = tk.Label(
            sec2,
            text=self._t_description(DEFAULT_T),
            font=('Arial', 8),
            bg=COLOR_PANEL, fg=COLOR_SUBTEXT
        )
        self.t_info.pack(fill='x', padx=10, pady=(0, 6))
        self.t_var.trace_add(
            'write',
            lambda *_: self.t_info.configure(
                text=self._t_description(self.t_var.get())
            )
        )

        # ---- Contextual Metadata ----
        sec3 = self._section(parent, 'CONTEXTUAL METADATA  (optional)')

        tk.Label(
            sec3,
            text='Compression level:',
            font=('Arial', 8),
            bg=COLOR_PANEL, fg=COLOR_SUBTEXT
        ).pack(anchor='w', padx=10, pady=(6, 0))

        self.comp_var = tk.StringVar(value='c23')
        comp_cb = ttk.Combobox(
            sec3, textvariable=self.comp_var,
            values=['raw', 'c23', 'c40'],
            state='readonly', width=10,
            font=('Arial', 9)
        )
        comp_cb.pack(anchor='w', padx=10, pady=3)

        tk.Label(
            sec3,
            text='Video source type:',
            font=('Arial', 8),
            bg=COLOR_PANEL, fg=COLOR_SUBTEXT
        ).pack(anchor='w', padx=10, pady=(4, 0))

        self.src_var = tk.StringVar(value='youtube')
        src_cb = ttk.Combobox(
            sec3, textvariable=self.src_var,
            values=['youtube', 'actors'],
            state='readonly', width=10,
            font=('Arial', 9)
        )
        src_cb.pack(anchor='w', padx=10, pady=(3, 8))

        # ---- Detect Button ----
        tk.Button(
            parent,
            text='▶  RUN DETECTION',
            font=('Arial', 12, 'bold'),
            bg=COLOR_ACCENT, fg=COLOR_TEXT,
            activebackground='#1B5E20',
            activeforeground=COLOR_TEXT,
            relief='flat', cursor='hand2',
            height=2,
            command=self._run_detection
        ).pack(fill='x', padx=0, pady=10)

        # ---- Video Metadata Display ----
        sec4 = self._section(parent, 'VIDEO METADATA')
        self.meta_text = tk.Text(
            sec4,
            height=7, width=42,
            font=('Courier', 8),
            bg='#0A0E1A', fg='#90CAF9',
            relief='flat', state='disabled',
            wrap='word'
        )
        self.meta_text.pack(
            fill='x', padx=8, pady=6
        )

    # ----------------------------------------------------------
    # RIGHT PANEL
    # ----------------------------------------------------------

    def _build_right_panel(self, parent):

        # ---- Result Header ----
        sec1 = self._section(parent, 'DETECTION RESULT')

        self.result_lbl = tk.Label(
            sec1,
            text='—',
            font=('Arial', 38, 'bold'),
            bg=COLOR_PANEL, fg=COLOR_NEUTRAL
        )
        self.result_lbl.pack(pady=(12, 4))

        self.conf_lbl = tk.Label(
            sec1,
            text='Confidence: —',
            font=('Arial', 10),
            bg=COLOR_PANEL, fg=COLOR_SUBTEXT
        )
        self.conf_lbl.pack(pady=(0, 4))

        self.latency_lbl = tk.Label(
            sec1,
            text='Latency: —',
            font=('Arial', 9),
            bg=COLOR_PANEL, fg=COLOR_SUBTEXT
        )
        self.latency_lbl.pack(pady=(0, 8))

        # ---- Confidence Bar ----
        sec2 = self._section(parent, 'CONFIDENCE VISUALIZATION')

        bar_frame = tk.Frame(sec2, bg=COLOR_PANEL)
        bar_frame.pack(fill='x', padx=12, pady=8)

        tk.Label(
            bar_frame,
            text='REAL',
            font=('Arial', 8, 'bold'),
            bg=COLOR_PANEL, fg=COLOR_REAL
        ).pack(side='left')
        tk.Label(
            bar_frame,
            text='FAKE',
            font=('Arial', 8, 'bold'),
            bg=COLOR_PANEL, fg=COLOR_FAKE
        ).pack(side='right')

        self.conf_canvas = tk.Canvas(
            sec2, height=28, bg='#0A0E1A',
            highlightthickness=0
        )
        self.conf_canvas.pack(
            fill='x', padx=12, pady=(0, 8)
        )
        self.conf_canvas.bind(
            '<Configure>', self._draw_confidence_bar
        )
        self._conf_value  = 0.5
        self._result_label = '—'

        # ---- Per-Frame Grid ----
        sec3 = self._section(parent, 'EXTRACTED FRAMES')

        self.frames_canvas = tk.Canvas(
            sec3, height=200,
            bg='#0A0E1A',
            highlightthickness=0
        )
        self.frames_canvas.pack(
            fill='x', padx=8, pady=6
        )

        # ---- Frame Rate Guide ----
        sec4 = self._section(parent, 'FRAME RATE GUIDE')

        guide_text = (
            'T = 5   |  F1: 0.694  |  Latency:  4.8ms\n'
            'T = 10  |  F1: 0.727  |  Latency:  9.6ms\n'
            'T = 15  |  F1: 0.735  |  Latency: 14.2ms  ← optimal\n'
            'T = 20  |  F1: 0.731  |  Latency: 18.8ms'
        )
        tk.Label(
            sec4,
            text=guide_text,
            font=('Courier', 8),
            bg=COLOR_PANEL, fg='#90CAF9',
            justify='left', anchor='w'
        ).pack(fill='x', padx=10, pady=6)

        # ---- Progress Bar ----
        self.progress = ttk.Progressbar(
            parent, mode='indeterminate', length=400
        )
        self.progress.pack(fill='x', padx=0, pady=4)

    # ----------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------

    def _t_description(self, t):
        descs = {
            5:  'Fast (lower accuracy, ~5ms latency)',
            10: 'Balanced (good accuracy, ~10ms latency)',
            15: 'Optimal (best F1=0.735, ~14ms latency)',
            20: 'High accuracy, ~19ms latency'
        }
        return descs.get(t, '')

    def _set_status(self, text, color=COLOR_SUBTEXT):
        self.status_lbl.configure(text=text, fg=color)

    def _update_meta_panel(self, meta_dict):
        self.meta_text.configure(state='normal')
        self.meta_text.delete('1.0', 'end')
        for k, v in meta_dict.items():
            self.meta_text.insert(
                'end', f'{k:<14}: {v}\n'
            )
        self.meta_text.configure(state='disabled')

    def _draw_confidence_bar(self, event=None,
                               conf=None, label=None):
        if conf is not None:
            self._conf_value   = conf
            self._result_label = label
        c    = self.conf_canvas
        w    = c.winfo_width()
        h    = c.winfo_height()
        if w < 10:
            return
        c.delete('all')

        # Background
        c.create_rectangle(
            0, 0, w, h,
            fill='#0A0E1A', outline=''
        )

        fill_w = int(self._conf_value * w)
        lbl    = self._result_label

        if lbl == 'FAKE':
            bar_color = COLOR_FAKE
        elif lbl == 'REAL':
            bar_color = COLOR_REAL
        else:
            bar_color = COLOR_NEUTRAL

        # Filled bar
        c.create_rectangle(
            0, 2, fill_w, h - 2,
            fill=bar_color, outline=''
        )
        # Threshold line at 50%
        mid = w // 2
        c.create_line(
            mid, 0, mid, h,
            fill='white', width=2, dash=(4, 3)
        )
        # Label
        pct = int(self._conf_value * 100)
        c.create_text(
            w // 2, h // 2,
            text=f'{pct}%',
            fill='white', font=('Arial', 9, 'bold')
        )

    def _display_frames(self, pil_frames):
        """Show extracted frames as a thumbnail strip."""
        c = self.frames_canvas
        c.delete('all')
        if not pil_frames:
            return

        c_w = c.winfo_width() or 400
        c_h = c.winfo_height() or 200
        n   = len(pil_frames)
        tw  = max(1, (c_w - 4) // n)
        th  = c_h - 4

        self._thumb_refs = []
        for i, frame in enumerate(pil_frames):
            thumb = frame.resize(
                (tw, th), Image.LANCZOS
            )
            photo = ImageTk.PhotoImage(thumb)
            self._thumb_refs.append(photo)
            c.create_image(
                2 + i * tw, 2,
                anchor='nw', image=photo
            )
            c.create_rectangle(
                2 + i * tw, 2,
                2 + (i+1) * tw, 2 + th,
                outline='#2C3E50', width=1
            )

    # ----------------------------------------------------------
    # FILE BROWSE
    # ----------------------------------------------------------

    def _browse_video(self):
        path = filedialog.askopenfilename(
            title='Select a video file',
            filetypes=[
                ('Video files',
                 '*.mp4 *.avi *.mov *.mkv *.wmv'),
                ('All files', '*.*')
            ]
        )
        if not path:
            return

        self.video_path = path
        short = os.path.basename(path)
        self.file_lbl.configure(
            text=short, fg=COLOR_TEXT
        )

        # Show metadata immediately
        meta = get_video_metadata(path)
        self._update_meta_panel(meta)

    # ----------------------------------------------------------
    # MODEL LOADING
    # ----------------------------------------------------------

    def _load_model_async(self):
        def _load():
            if not TORCH_AVAILABLE:
                self._set_status(
                    'PyTorch not found — DEMO MODE',
                    color=COLOR_FAKE
                )
                return
            self.model, self.transform = \
                load_detection_model(self.device,
                                     metadata_dim=7)
            if self.model:
                gpu = (
                    self.device.type.upper()
                    if self.device else 'CPU'
                )
                self._set_status(
                    f'Model ready  [{gpu}]',
                    color=COLOR_REAL
                )
            else:
                self._set_status(
                    'DEMO MODE (model not found)',
                    color=COLOR_GOLD
                )

        t = threading.Thread(target=_load, daemon=True)
        t.start()

    # ----------------------------------------------------------
    # DETECTION
    # ----------------------------------------------------------

    def _run_detection(self):
        if self.is_running:
            return

        if not self.video_path:
            messagebox.showwarning(
                'No File',
                'Please browse and select a video file first.'
            )
            return

        self.is_running = True
        self.progress.start(12)
        self._set_status('Processing...', COLOR_GOLD)

        t = threading.Thread(
            target=self._detection_worker,
            daemon=True
        )
        t.start()

    def _detection_worker(self):
        try:
            T    = self.t_var.get()
            comp = self.comp_var.get()
            src  = self.src_var.get()

            # --- Extract frames ---
            self._set_status(
                f'Extracting {T} frames...', COLOR_GOLD
            )
            frames = extract_frames_from_video(
                self.video_path, T
            )

            if frames is None:
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        'Error',
                        'Could not extract frames.\n'
                        'Make sure OpenCV is installed and\n'
                        'the file is a valid video.'
                    )
                )
                return

            # Display frames in GUI
            self.root.after(
                0,
                lambda f=frames: self._display_frames(f)
            )

            # --- Build metadata vector ---
            meta_dict = get_video_metadata(self.video_path)
            self.root.after(
                0,
                lambda m=meta_dict:
                    self._update_meta_panel(m)
            )

            res_str = meta_dict.get('resolution', '1280×720')
            meta_vec = build_metadata_vector(
                comp, src, res_str
            )

            # --- Run inference ---
            self._set_status('Running inference...', COLOR_GOLD)

            if (TORCH_AVAILABLE and
                    self.device is not None):
                label, conf, latency = run_inference(
                    self.model, self.transform,
                    frames, meta_vec,
                    self.device, T
                )
            else:
                import random
                conf    = random.uniform(0.3, 0.95)
                label   = 'FAKE' if conf > 0.5 else 'REAL'
                latency = 42.0

            # --- Update GUI on main thread ---
            self.root.after(
                0,
                lambda lb=label, c=conf, lat=latency:
                    self._show_results(lb, c, lat)
            )

        except Exception as e:
            print(f"Detection error: {e}")
            self.root.after(
                0,
                lambda err=str(e):
                    messagebox.showerror(
                        'Detection Error', err
                    )
        )
        finally:
            self.root.after(0, self._detection_done)

    def _show_results(self, label, conf, latency):
        """Update result widgets with detection output."""
        color = COLOR_FAKE if label == 'FAKE' else COLOR_REAL

        self.result_lbl.configure(
            text=label, fg=color
        )
        pct = int(conf * 100)
        self.conf_lbl.configure(
            text=f'Confidence: {pct}%',
            fg=color
        )
        self.latency_lbl.configure(
            text=f'Latency: {latency:.1f} ms  '
                 f'(T = {self.t_var.get()} frames)',
            fg=COLOR_SUBTEXT
        )
        self._draw_confidence_bar(conf=conf, label=label)
        self._set_status(
            f'Done — {label} ({pct}% confidence)',
            color=color
        )

    def _detection_done(self):
        self.progress.stop()
        self.is_running = False


# ==============================================================
# ENTRY POINT
# ==============================================================

def main():
    if not TORCH_AVAILABLE:
        print("WARNING: PyTorch not installed.")
        print("Running in DEMO MODE.")

    if not CV2_AVAILABLE:
        print("WARNING: OpenCV (cv2) not installed.")
        print("Frame extraction will not work.")
        print("Install: pip install opencv-python")

    root = tk.Tk()

    # Apply dark theme to ttk widgets
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure(
        'TCombobox',
        fieldbackground='#0D1117',
        background='#0D1117',
        foreground='#ECF0F1'
    )
    style.configure(
        'TProgressbar',
        troughcolor='#0D1117',
        background='#2E8B57'
    )

    app = DeepfakeApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
