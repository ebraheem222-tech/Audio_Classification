# gui.py
import os
import sys
import json
import wave
import threading
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import librosa
import librosa.display
from torchvision import transforms, models

from pydub import AudioSegment
from pydub.playback import play
import pyaudio

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

BINARY_CLASSES = ['Indoor', 'Outdoor']
CLASS_COLORS = {'Indoor': '#2E86AB', 'Outdoor': '#A23B72'}
CLASS_DESCRIPTIONS = {
    'Indoor': 'Sounds typically heard inside buildings (clock, typing, etc.)',
    'Outdoor': 'Sounds from outdoor environments (cars, wind, waves, etc.)'
}

DATA_ROOT = 'Data'

CHECKPOINTS = [
    "checkpoints_cv/fold4_best.pth",
    # "checkpoints_cv/fold2_best.pth",
    # "checkpoints_cv/fold3_best.pth",
    # "checkpoints_cv/fold1_best.pth",
    # "checkpoints_cv/fold5_best.pth",
    # Or: "checkpoints_ft/best_finetune.ckpt",
]

T_MAX = 1000
SAMPLE_RATE = 44100
N_FFT = 1024
N_MELS = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CVBinaryNet(nn.Module):

    def __init__(self, head_dims, num_len_bins=None, len_emb_dim=None):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.fc = nn.Identity() 
        self.backbone = base

        self.num_len_bins = int(num_len_bins) if num_len_bins else 0
        self.len_emb_dim = int(len_emb_dim) if len_emb_dim else 0
        if self.num_len_bins > 0 and self.len_emb_dim > 0:
            self.len_emb = nn.Embedding(self.num_len_bins, self.len_emb_dim)
        else:
            self.len_emb = None

        layers = []
        for i in range(len(head_dims) - 1):
            in_d, out_d = head_dims[i], head_dims[i+1]
            layers.append(nn.Linear(in_d, out_d))
            if i < len(head_dims) - 2:
                layers.append(nn.BatchNorm1d(out_d))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(0.4))
        self.head = nn.Sequential(*layers)

    def forward(self, x, len_bin=None):
        f = self.backbone(x)  
        if self.len_emb is not None:
            if len_bin is None:
                raise RuntimeError("Model expects len_bin but got None")
            emb = self.len_emb(len_bin)  
            f = torch.cat([f, emb], dim=1)
        return self.head(f)


def _build_from_cv_state(state):
  
    clean = {}
    for k, v in state.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        clean[k] = v
    state = clean

    has_backbone = any(k.startswith('backbone.') for k in state.keys())
    has_head = any(k.startswith('head.') for k in state.keys())
    if not (has_backbone and has_head):
        return None, None  

    w0 = state.get('head.0.weight')
    w1 = state.get('head.4.weight')
    w2 = state.get('head.8.weight')
    if w0 is None or w1 is None or w2 is None:
        raise RuntimeError("Unexpected head layout (missing head.[0|4|8].weight)")

    head_in = w0.shape[1]
    h1 = w0.shape[0]
    h2 = w1.shape[0]
    h3 = w2.shape[0]  

    num_len_bins = len_emb_dim = 0
    if 'len_emb.weight' in state:
        len_w = state['len_emb.weight'] 
        num_len_bins, len_emb_dim = len_w.shape

    model = CVBinaryNet(
        head_dims=[head_in, h1, h2, h3],
        num_len_bins=num_len_bins or None,
        len_emb_dim=len_emb_dim or None
    ).to(DEVICE)

    model.load_state_dict(state, strict=True)
    model.eval()
    torch.set_grad_enabled(False)
    return model, {'num_len_bins': num_len_bins, 'len_emb_dim': len_emb_dim}


def _build_from_vanilla_state(state):
    """
    Fallback: plain torchvision resnet18 with 256→128→2 head (your earlier GUI layout).
    """
    clean = {}
    for k, v in state.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        clean[k] = v
    state = clean

    base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = base.fc.in_features
    base.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(128, 2)
    )
    base = base.to(DEVICE)

    try:
        base.load_state_dict(state, strict=True)
    except RuntimeError:
        alt = {}
        for k, v in state.items():
            alt[k.replace('model.', '', 1)] = v
        base.load_state_dict(alt, strict=True)

    base.eval()
    torch.set_grad_enabled(False)
    return base, None


def load_single_checkpoint(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location=DEVICE, weights_only=False)
    state = payload.get('model_state_dict') or payload.get('state_dict') or payload

    model, meta = _build_from_cv_state(state)
    if model is not None:
        print(f"[OK] CVBinaryNet loaded: {os.path.basename(path)}")
        return model, meta

    model, meta = _build_from_vanilla_state(state)
    print(f"[OK] Vanilla ResNet head loaded: {os.path.basename(path)}")
    return model, meta


class EnsembleWrapper(nn.Module):

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList([m for m, _ in members])
        self.metas = [meta for _, meta in members]  

    def forward(self, x, len_bin=None):
        logits = None
        for model, meta in zip(self.members, self.metas):
            if meta is not None and meta.get('num_len_bins', 0) > 0:
                out = model(x, len_bin=len_bin)
            else:
                out = model(x)
            logits = out if logits is None else (logits + out)
        return logits / len(self.members)


def load_model_ensemble(paths):
    members = []
    for p in paths:
        try:
            m, meta = load_single_checkpoint(p)
            members.append((m, meta))
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}", file=sys.stderr)
    if len(members) == 0:
        raise RuntimeError("No valid checkpoints loaded.")
    if len(members) == 1:
        return members[0][0], members[0][1]
    ens = EnsembleWrapper(members).to(DEVICE)
    ens.eval()
    torch.set_grad_enabled(False)
    print(f"[OK] Ensemble ready ({len(members)} models).")
    needs_len = any((meta is not None and meta.get('num_len_bins', 0) > 0) for _, meta in members)
    return ens, {'num_len_bins': 1 if needs_len else 0, 'len_emb_dim': 0}


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def load_audio_file(file_path):
    try:
        if file_path.lower().endswith(('.mp3', '.m4a', '.flac', '.ogg')):
            audio = AudioSegment.from_file(file_path)
            temp_wav = "temp_audio_gui.wav"
            audio.export(temp_wav, format="wav")
            y, sr = librosa.load(temp_wav, sr=SAMPLE_RATE, mono=True)
            os.remove(temp_wav)
        else:
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        return y, sr
    except Exception as e:
        raise Exception(f"Failed to load audio: {str(e)}")

def compute_spectrogram(y, sr):
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)), mode='constant')
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, n_mels=N_MELS, fmax=sr//2)
    spec_db = librosa.power_to_db(mel + 1e-10, ref=np.max)
    return spec_db

def prepare_input(spec_db):
    mean = spec_db.mean()
    std = spec_db.std() + 1e-8
    spec_db = (spec_db - mean) / std

    spec = torch.from_numpy(spec_db).float()  
    T = spec.shape[1]

    if T > T_MAX:
        start = (T - T_MAX) // 2
        spec = spec[:, start:start + T_MAX]
    elif T < T_MAX:
        pad_total = T_MAX - T
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        spec = F.pad(spec, (pad_left, pad_right, 0, 0))

    img = spec.unsqueeze(0).repeat(3, 1, 1) 
    img = transform(img).unsqueeze(0).to(DEVICE)  
    return img

if len(CHECKPOINTS) == 0:
    print("Error: please set CHECKPOINTS to at least one .pth/.ckpt file.", file=sys.stderr)
    sys.exit(1)

try:
    model, model_meta = load_model_ensemble(CHECKPOINTS)
    print("Model(s) loaded successfully.")
except Exception as e:
    print(f"Failed to load model(s): {e}", file=sys.stderr)
    try:
        messagebox.showerror("Error", f"Failed to load model(s): {str(e)}")
    except Exception:
        pass
    sys.exit(1)

def predict_audio(file_path):
    y, sr = load_audio_file(file_path)
    spec_db = compute_spectrogram(y, sr)
    T_raw = spec_db.shape[1]
    img = prepare_input(spec_db)

    len_bin = None
    nbins = int(model_meta.get('num_len_bins', 0)) if model_meta else 0
    if nbins > 0:
        frac = min(max(T_raw / float(T_MAX), 0.0), 1.0)
        len_bin_idx = min(int(frac * nbins), nbins - 1)
        len_bin = torch.tensor([len_bin_idx], device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        out = model(img, len_bin=len_bin) if len_bin is not None else model(img)
        probs = F.softmax(out, dim=1).squeeze().cpu().numpy()
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

    return {
        'class': BINARY_CLASSES[pred_class],
        'confidence': confidence,
        'probabilities': {BINARY_CLASSES[i]: float(probs[i]) for i in range(len(BINARY_CLASSES))},
        'audio_data': y,
        'sample_rate': sr,
        'spectrogram': spec_db
    }

class AudioPlayer:
    def __init__(self):
        self.is_playing = False
        self.audio_thread = None

    def play_audio(self, file_path):
        if self.is_playing:
            return

        def play_thread():
            try:
                self.is_playing = True
                audio = AudioSegment.from_file(file_path)
                play(audio)
            except Exception as e:
                print(f"Playback error: {e}")
            finally:
                self.is_playing = False

        self.audio_thread = threading.Thread(target=play_thread, daemon=True)
        self.audio_thread.start()

    def stop(self):
        self.is_playing = False


class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.stream = None

    def start_recording(self):
        self.recording = True
        self.frames = []

        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=1024,
            stream_callback=self._callback
        )
        self.stream.start_stream()

    def _callback(self, in_data, frame_count, time_info, status):
        if self.recording:
            self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)

    def stop_recording(self):
        self.recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        output_file = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        wf = wave.open(output_file, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        return output_file


class AudioClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("🎵 Audio Environment Classifier")
        self.geometry("1200x800")
        self.configure(bg='#f0f0f0')

        self.player = AudioPlayer()
        self.recorder = AudioRecorder()
        self.current_file = None
        self.prediction_history = []

        self.setup_styles()
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.style.configure('Title.TLabel', font=('Arial', 24, 'bold'), background='#f0f0f0')
        self.style.configure('Heading.TLabel', font=('Arial', 14, 'bold'), background='#f0f0f0')
        self.style.configure('Info.TLabel', font=('Arial', 11), background='#f0f0f0')
        self.style.configure('Result.TLabel', font=('Arial', 16, 'bold'), background='white')

        self.style.configure('Action.TButton', font=('Arial', 11, 'bold'), padding=10)
        self.style.configure('Play.TButton', font=('Arial', 10), padding=5)

    def create_widgets(self):
        main_frame = tk.Frame(self, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title_label = ttk.Label(main_frame, text="🎵 Audio Environment Classifier", style='Title.TLabel')
        title_label.pack(pady=(0, 20))

        self.create_control_section(main_frame)
        self.create_results_section(main_frame)
        self.create_history_section(main_frame)
        self.create_status_bar()

    def create_control_section(self, parent):
        control_frame = tk.Frame(parent, bg='white', relief=tk.RAISED, bd=1)
        control_frame.pack(fill=tk.X, pady=(0, 20))

        inner_frame = tk.Frame(control_frame, bg='white')
        inner_frame.pack(padx=20, pady=20)

        ttk.Label(inner_frame, text="Select Audio Source:", style='Heading.TLabel').grid(
            row=0, column=0, columnspan=4, pady=(0, 10)
        )

        ttk.Button(inner_frame, text="📁 Browse File", command=self.browse_file,
                   style='Action.TButton').grid(row=1, column=0, padx=5)
        ttk.Button(inner_frame, text="📂 Batch Process", command=self.batch_process,
                   style='Action.TButton').grid(row=1, column=1, padx=5)

        self.record_btn = ttk.Button(inner_frame, text="🎤 Start Recording",
                                     command=self.toggle_recording, style='Action.TButton')
        self.record_btn.grid(row=1, column=2, padx=5)

        ttk.Button(inner_frame, text="📊 Export Results", command=self.export_results,
                   style='Action.TButton').grid(row=1, column=3, padx=5)

        self.file_label = ttk.Label(inner_frame, text="No file selected", style='Info.TLabel')
        self.file_label.grid(row=2, column=0, columnspan=4, pady=(10, 0))

    def create_results_section(self, parent):
        results_frame = tk.Frame(parent, bg='white', relief=tk.RAISED, bd=1)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

        notebook = ttk.Notebook(results_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.create_prediction_tab(notebook)
        self.create_visualization_tab(notebook)
        self.create_info_tab(notebook)

    def create_prediction_tab(self, notebook):
        pred_frame = tk.Frame(notebook, bg='white')
        notebook.add(pred_frame, text="Prediction Results")

        self.results_container = tk.Frame(pred_frame, bg='white')
        self.results_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        ttk.Label(self.results_container, text="Load an audio file to see predictions",
                  style='Info.TLabel').pack(pady=50)

    def create_visualization_tab(self, notebook):
        viz_frame = tk.Frame(notebook, bg='white')
        notebook.add(viz_frame, text="Spectrogram")

        self.fig = Figure(figsize=(10, 4), dpi=80, facecolor='white')
        self.ax_spec = self.fig.add_subplot(121)
        self.ax_wave = self.fig.add_subplot(122)

        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def create_info_tab(self, notebook):
        info_frame = tk.Frame(notebook, bg='white')
        notebook.add(info_frame, text="Audio Information")

        self.info_text = tk.Text(info_frame, font=('Courier', 10), bg='white', wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        scrollbar = ttk.Scrollbar(self.info_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.info_text.yview)

    def create_history_section(self, parent):
        history_frame = tk.Frame(parent, bg='white', relief=tk.RAISED, bd=1)
        history_frame.pack(fill=tk.X)

        ttk.Label(history_frame, text="Prediction History", style='Heading.TLabel').pack(pady=(10, 5))

        columns = ('File', 'Prediction', 'Confidence', 'Time')
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show='headings', height=5)
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=200)

        self.history_tree.pack(fill=tk.X, padx=10, pady=(0, 10))

        history_scroll = ttk.Scrollbar(self.history_tree, orient=tk.VERTICAL)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_tree.config(yscrollcommand=history_scroll.set)
        history_scroll.config(command=self.history_tree.yview)

    def create_status_bar(self):
        ckpt_str = "; ".join([os.path.basename(p) for p in CHECKPOINTS])
        self.status_bar = tk.Label(self, text=f"Ready | Device: {DEVICE} | Ckpts: {ckpt_str}",
                                   relief=tk.SUNKEN, anchor=tk.W, bg='#e0e0e0')
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.m4a *.ogg"),
                       ("All Files", "*.*")]
        )
        if file_path:
            self.process_file(file_path)

    def batch_process(self):
        files = filedialog.askopenfilenames(
            title="Select Multiple Audio Files",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.m4a *.ogg"),
                       ("All Files", "*.*")]
        )
        if not files:
            return

        self.status_bar.config(text=f"Processing {len(files)} files...")
        results = []

        progress_window = tk.Toplevel(self)
        progress_window.title("Batch Processing")
        progress_window.geometry("400x150")

        progress_label = tk.Label(progress_window, text="Processing files...")
        progress_label.pack(pady=20)

        progress_bar = ttk.Progressbar(progress_window, length=300, mode='determinate', maximum=len(files))
        progress_bar.pack(pady=10)

        def process_batch():
            for i, file_path in enumerate(files):
                try:
                    result = predict_audio(file_path)
                    results.append({
                        'file': os.path.basename(file_path),
                        'prediction': result['class'],
                        'confidence': result['confidence']
                    })
                    self.add_to_history(file_path, result)
                except Exception as e:
                    print(f"Batch file error: {e}")
                    results.append({
                        'file': os.path.basename(file_path),
                        'prediction': 'Error',
                        'confidence': 0.0
                    })
                progress_bar['value'] = i + 1
                progress_label.config(text=f"Processing {i + 1}/{len(files)}: {os.path.basename(file_path)}")
                progress_window.update()

            progress_window.destroy()
            self.show_batch_results(results)
            self.status_bar.config(text=f"Batch processing complete: {len(files)} files")

        threading.Thread(target=process_batch, daemon=True).start()

    def toggle_recording(self):
        if not self.recorder.recording:
            self.recorder.start_recording()
            self.record_btn.config(text="⏹ Stop Recording")
            self.status_bar.config(text="Recording audio...")
        else:
            file_path = self.recorder.stop_recording()
            self.record_btn.config(text="🎤 Start Recording")
            self.status_bar.config(text=f"Recording saved: {file_path}")
            self.process_file(file_path)

    def process_file(self, file_path):
        self.current_file = file_path
        self.file_label.config(text=f"Current file: {os.path.basename(file_path)}")
        self.status_bar.config(text="Processing audio...")

        try:
            result = predict_audio(file_path)
            self.display_results(result)
            self.display_visualization(result)
            self.display_info(file_path, result)
            self.add_to_history(file_path, result)
            self.status_bar.config(text=f"Analysis: {result['class']} ({result['confidence']:.1%})")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process audio:\n{str(e)}")
            self.status_bar.config(text="Error processing file")

    def display_results(self, result):
        for widget in self.results_container.winfo_children():
            widget.destroy()

        pred_class = result['class']
        confidence = result['confidence']

        main_frame = tk.Frame(self.results_container, bg='white')
        main_frame.pack(fill=tk.BOTH, expand=True)

        color = CLASS_COLORS.get(pred_class, '#666666')
        pred_label = tk.Label(main_frame, text=pred_class.upper(),
                              font=('Arial', 32, 'bold'), fg=color, bg='white')
        pred_label.pack(pady=(20, 10))

        self.create_confidence_meter(main_frame, confidence)

        probs_frame = tk.Frame(main_frame, bg='white')
        probs_frame.pack(pady=20)

        ttk.Label(probs_frame, text="Class Probabilities:", style='Heading.TLabel').pack(pady=(0, 10))
        for class_name, prob in result['probabilities'].items():
            self.create_probability_bar(probs_frame, class_name, prob)

        desc_text = CLASS_DESCRIPTIONS.get(pred_class, "")
        desc_label = tk.Label(main_frame, text=desc_text, font=('Arial', 10, 'italic'),
                              fg='#666666', bg='white', wraplength=400)
        desc_label.pack(pady=10)

        if self.current_file:
            play_btn = ttk.Button(main_frame, text="▶ Play Audio",
                                  command=lambda: self.player.play_audio(self.current_file),
                                  style='Play.TButton')
            play_btn.pack(pady=10)

    def create_confidence_meter(self, parent, confidence):
        meter_frame = tk.Frame(parent, bg='white')
        meter_frame.pack(pady=10)

        tk.Label(meter_frame, text=f"Confidence: {confidence:.1%}",
                 font=('Arial', 14), bg='white').pack()

        meter_canvas = tk.Canvas(meter_frame, width=300, height=30, bg='white', highlightthickness=0)
        meter_canvas.pack(pady=5)

        meter_canvas.create_rectangle(0, 10, 300, 20, fill='#e0e0e0', outline='')

        bar_width = int(300 * confidence)
        if confidence > 0.8:
            color = '#4CAF50'
        elif confidence > 0.6:
            color = '#FFC107'
        else:
            color = '#F44336'
        meter_canvas.create_rectangle(0, 10, bar_width, 20, fill=color, outline='')

        for i in range(0, 101, 25):
            x = int(300 * i / 100)
            meter_canvas.create_line(x, 20, x, 25, fill='#666666')
            meter_canvas.create_text(x, 28, text=f"{i}%", font=('Arial', 8), fill='#666666')

    def create_probability_bar(self, parent, class_name, probability):
        bar_frame = tk.Frame(parent, bg='white')
        bar_frame.pack(fill=tk.X, pady=2)

        label = tk.Label(bar_frame, text=f"{class_name}:", width=10, anchor='w',
                         font=('Arial', 10), bg='white')
        label.pack(side=tk.LEFT, padx=(0, 10))

        bar_canvas = tk.Canvas(bar_frame, width=200, height=20, bg='white', highlightthickness=0)
        bar_canvas.pack(side=tk.LEFT)

        bar_canvas.create_rectangle(0, 5, 200, 15, fill='#e0e0e0', outline='')
        bar_width = int(200 * probability)
        color = CLASS_COLORS.get(class_name, '#666666')
        bar_canvas.create_rectangle(0, 5, bar_width, 15, fill=color, outline='')

        pct_label = tk.Label(bar_frame, text=f"{probability:.1%}", width=6,
                             font=('Arial', 10), bg='white')
        pct_label.pack(side=tk.LEFT, padx=(5, 0))

    def display_visualization(self, result):
        self.ax_spec.clear()
        self.ax_wave.clear()

        self.ax_spec.set_title('Mel Spectrogram')
        librosa.display.specshow(result['spectrogram'], sr=result['sample_rate'],
                                 x_axis='time', y_axis='mel', ax=self.ax_spec)
        self.ax_spec.set_xlabel('Time (s)')
        self.ax_spec.set_ylabel('Mel bins')

        self.ax_wave.set_title('Waveform')
        time = np.arange(len(result['audio_data'])) / result['sample_rate']
        self.ax_wave.plot(time, result['audio_data'], color='#2E86AB', linewidth=0.5)
        self.ax_wave.set_xlabel('Time (s)')
        self.ax_wave.set_ylabel('Amplitude')
        self.ax_wave.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()

    def display_info(self, file_path, result):
        self.info_text.delete('1.0', tk.END)

        file_stats = os.stat(file_path)
        duration = len(result['audio_data']) / result['sample_rate']

        ckpt_str = ", ".join([os.path.basename(p) for p in CHECKPOINTS])
        info = f"""
FILE INFORMATION
================
Path: {file_path}
Name: {os.path.basename(file_path)}
Size: {file_stats.st_size / 1024:.2f} KB
Duration: {duration:.2f} seconds
Sample Rate: {result['sample_rate']} Hz

AUDIO PROPERTIES
================
Total Samples: {len(result['audio_data'])}
Min Amplitude: {np.min(result['audio_data']):.4f}
Max Amplitude: {np.max(result['audio_data']):.4f}
Mean Amplitude: {np.mean(result['audio_data']):.4f}
Std Deviation: {np.std(result['audio_data']):.4f}

SPECTROGRAM INFO
================
Shape: {result['spectrogram'].shape}
Min Value: {np.min(result['spectrogram']):.2f} dB
Max Value: {np.max(result['spectrogram']):.2f} dB
Mean Value: {np.mean(result['spectrogram']):.2f} dB

PREDICTION DETAILS
==================
Predicted Class: {result['class']}
Confidence: {result['confidence']:.4f}

Class Probabilities:
"""
        for class_name, prob in result['probabilities'].items():
            info += f"  {class_name}: {prob:.4f} ({prob * 100:.2f}%)\n"

        info += f"""
MODEL INFORMATION
=================
Device: {DEVICE}
Checkpoints: {ckpt_str}
Input Size: 224x224
Max Time Steps: {T_MAX}
"""
        self.info_text.insert('1.0', info)

    def add_to_history(self, file_path, result):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.history_tree.insert('', 0, values=(
            os.path.basename(file_path),
            result['class'],
            f"{result['confidence']:.2%}",
            timestamp
        ))
        self.prediction_history.append({
            'file': file_path,
            'result': result,
            'timestamp': timestamp
        })

    def show_batch_results(self, results):
        window = tk.Toplevel(self)
        window.title("Batch Processing Results")
        window.geometry("600x400")

        columns = ('File', 'Prediction', 'Confidence')
        tree = ttk.Treeview(window, columns=columns, show='headings')
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=200)
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        indoor_count = 0
        outdoor_count = 0
        for r in results:
            tree.insert('', tk.END, values=(
                r['file'],
                r['prediction'],
                f"{r['confidence']:.2%}" if r['confidence'] > 0 else "Error"
            ))
            if r['prediction'] == 'Indoor':
                indoor_count += 1
            elif r['prediction'] == 'Outdoor':
                outdoor_count += 1

        summary_frame = tk.Frame(window)
        summary_frame.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(summary_frame,
                 text=f"Summary: Indoor: {indoor_count}, Outdoor: {outdoor_count}, Errors: {len(results) - indoor_count - outdoor_count}",
                 font=('Arial', 12, 'bold')).pack()

        ttk.Button(summary_frame, text="Close", command=window.destroy).pack(pady=10)

    def export_results(self):
        if not self.prediction_history:
            messagebox.showwarning("No Data", "No predictions to export")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("CSV Files", "*.csv")]
        )
        if not file_path:
            return

        try:
            if file_path.endswith('.json'):
                export_data = []
                for item in self.prediction_history:
                    export_data.append({
                        'file': item['file'],
                        'timestamp': item['timestamp'],
                        'prediction': item['result']['class'],
                        'confidence': item['result']['confidence'],
                        'probabilities': item['result']['probabilities']
                    })
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)

            elif file_path.endswith('.csv'):
                import csv
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['File', 'Timestamp', 'Prediction', 'Confidence', 'Indoor_Prob', 'Outdoor_Prob'])
                    for item in self.prediction_history:
                        writer.writerow([
                            item['file'],
                            item['timestamp'],
                            item['result']['class'],
                            f"{item['result']['confidence']:.4f}",
                            f"{item['result']['probabilities'].get('Indoor', 0):.4f}",
                            f"{item['result']['probabilities'].get('Outdoor', 0):.4f}"
                        ])
            messagebox.showinfo("Success", f"Results exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {str(e)}")

    def on_closing(self):
        if self.recorder.recording:
            self.recorder.stop_recording()
        self.destroy()


if __name__ == "__main__":
    try:
        app = AudioClassifierApp()
        app.mainloop()
    except Exception as e:
        print(f"Application error: {e}", file=sys.stderr)
        try:
            messagebox.showerror("Fatal Error", f"Application failed to start:\n{str(e)}")
        except Exception:
            pass
