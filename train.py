import os
import math
import random
import argparse
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import librosa
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report


INDOOR = {
    'clock_tick', 'mouse_click', 'keyboard_typing',
    'toilet_flush', 'vacuum_cleaner', 'washing_machine',
    'door_wood_creaks', 'door_wood_knock'
}
OUTDOOR = {
    'car_horn', 'airplane', 'thunderstorm',
    'sea_waves', 'helicopter', 'wind',
    'chainsaw', 'crickets'
}
ALL_SOUNDS = INDOOR | OUTDOOR

BINARY_CLASSES = ['Indoor', 'Outdoor']
LABEL_OF = lambda cls: 0 if cls in INDOOR else 1

SAMPLE_RATE = 44100
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128

BINS = [(1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0)]
BIN_MID = [math.sqrt(a * b) for (a, b) in BINS]
NUM_BINS = len(BINS)

FREEZE_UNTIL = 'layer2'
LEN_EMB_DIM = 32
BASE_LR = 5e-5
WEIGHT_DECAY_BACKBONE = 1e-4
WEIGHT_DECAY_HEAD = 5e-5
EPOCHS = 100
BATCH_SIZE = 16
PATIENCE = 20
CKPT = 'length_bins_best_overfit_defense.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

MIXUP_ALPHA = 0.3
FREQ_MASK_PARAM = 16
TIME_MASK_PARAM = 40
NUM_FREQ_MASKS = 2
NUM_TIME_MASKS = 2

FOCAL_GAMMA = 1.5
FOCAL_ALPHA = None

EMA_DECAY = 0.999

VAL_TTA_OFFSETS = [0.25, 0.75]
TEST_TTA_OFFSETS = [0.25, 0.75]

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def seconds_to_frames(sec: float) -> int:
    return max(1, int(round(sec * SAMPLE_RATE / HOP_LENGTH)))

def assign_bin(duration_sec: float) -> int:
    if duration_sec <= BINS[0][0]:
        return 0
    if duration_sec >= BINS[-1][1]:
        return NUM_BINS - 1
    diffs = [abs(duration_sec - m) for m in BIN_MID]
    return int(np.argmin(diffs))

def librosa_duration_fast(path: str) -> float:
    try:
        return librosa.get_duration(filename=path)
    except Exception:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        return len(y) / float(sr)



def add_pink_noise(y, snr_db_range=(10, 30)):#add smooth noise
    if len(y) == 0:
        return y
    snr = np.random.uniform(*snr_db_range)
    white = np.random.randn(len(y))
    Y = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(len(white), 1.0 / SAMPLE_RATE)
    Y /= np.maximum(freqs, 1.0)  # avoid zero
    pink = np.fft.irfft(Y, n=len(white)).real
    pink = pink / (np.std(pink) + 1e-8)

    sig_pwr = np.mean(y ** 2) + 1e-12
    noise_pwr = sig_pwr / (10 ** (snr / 10.0))
    pink = pink * np.sqrt(noise_pwr)
    return np.clip(y + pink, -1.0, 1.0)

def random_gain(y, low=0.7, high=1.3):#increase Sound volume
    g = np.random.uniform(low, high)
    return np.clip(y * g, -1.0, 1.0)

def random_shift(y, max_frac=0.1):#random Shift phase
    if len(y) == 0:
        return y
    shift = int(np.random.uniform(-max_frac, max_frac) * len(y))
    return np.roll(y, shift)

def spec_augment(spec: torch.Tensor,
                 num_time_masks=NUM_TIME_MASKS,
                 num_freq_masks=NUM_FREQ_MASKS,
                 time_mask_param=TIME_MASK_PARAM,
                 freq_mask_param=FREQ_MASK_PARAM):

    S = spec.clone()
    num_mels, T = S.shape
    # frequency masks
    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_param)
        if f == 0: continue
        f0 = random.randint(0, max(0, num_mels - f))
        S[f0:f0 + f, :] = 0
    # time masks
    for _ in range(num_time_masks):
        t = random.randint(0, time_mask_param)
        if t == 0: continue
        t0 = random.randint(0, max(0, T - t))
        S[:, t0:t0 + t] = 0
    return S



class LengthBinAudioDataset(Dataset):
    def __init__(self, root_dir, mode='train', indices=None, transform=None, tta_offsets=None):
        assert mode in ('train', 'val', 'test')
        self.root = root_dir
        self.mode = mode
        self.transform = transform
        self.tta_offsets = tta_offsets

        self.file_paths, self.labels, self.sound_types = [], [], []
        self.file_durations, self.file_bin = [], []

        for cls in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, cls)
            if cls not in ALL_SOUNDS or not os.path.isdir(class_dir):
                print(f"  Skipping unknown class: {cls}")
                continue
            label = LABEL_OF(cls)
            for root_subdir, _, files in os.walk(class_dir):
                for fname in files:
                    if fname.lower().endswith(('.wav', '.flac', '.mp3', '.m4a', '.ogg')):
                        p = os.path.join(root_subdir, fname)
                        dur = librosa_duration_fast(p)
                        b = assign_bin(dur)
                        self.file_paths.append(p)
                        self.labels.append(label)
                        self.sound_types.append(cls)
                        self.file_durations.append(dur)
                        self.file_bin.append(b)

        self.file_paths = np.array(self.file_paths)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.sound_types = np.array(self.sound_types)
        self.file_durations = np.array(self.file_durations, dtype=np.float32)
        self.file_bin = np.array(self.file_bin, dtype=np.int64)

        if indices is not None:
            self.file_paths = self.file_paths[indices]
            self.labels = self.labels[indices]
            self.sound_types = self.sound_types[indices]
            self.file_durations = self.file_durations[indices]
            self.file_bin = self.file_bin[indices]

        if self.mode == 'train':
            self.index = np.arange(len(self.file_paths))
        else:
            self.index = []
            self.window_meta = []  # (fid, b, start)
            for fid in range(len(self.file_paths)):
                dur = float(self.file_durations[fid])
                b = int(self.file_bin[fid])
                win_frames = seconds_to_frames(BIN_MID[b])
                total_frames_est = max(1, seconds_to_frames(dur) - int(N_FFT / HOP_LENGTH) + 1)
                stride = max(1, win_frames // 2)

                n_windows = max(1, 1 + (max(0, total_frames_est - win_frames)) // stride)
                base_starts = [w * stride for w in range(n_windows)]

                starts = list(base_starts)
                if self.tta_offsets:
                    for off in self.tta_offsets:
                        shift = int(off * stride)
                        starts += [min(max(0, s + shift), max(0, total_frames_est - win_frames)) for s in base_starts]

                starts = sorted(set(starts))
                for s in starts:
                    self.window_meta.append((fid, b, s))
            self.index = np.arange(len(self.window_meta))

        print("\n" + "=" * 50)
        print(f"Dataset Statistics ({self.mode})")
        print("=" * 50)
        print(f"Total files: {len(self.file_paths)}")
        in_tot = int((self.labels == 0).sum())
        out_tot = int((self.labels == 1).sum())
        print(f"Indoor files: {in_tot}")
        print(f"Outdoor files: {out_tot}")
        print("\nPer-class distribution:")
        print("\nIndoor sounds:")
        for s in sorted(INDOOR):
            c = int(np.sum(self.sound_types == s))
            if c > 0: print(f"  {s:20s}: {c:4d} files")
        print("\nOutdoor sounds:")
        for s in sorted(OUTDOOR):
            c = int(np.sum(self.sound_types == s))
            if c > 0: print(f"  {s:20s}: {c:4d} files")
        print("=" * 50 + "\n")

    def __len__(self):
        return len(self.index)

    def _load_wave(self, path):
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        if self.mode == 'train':
            if random.random() < 0.3:
                y = random_shift(y, 0.1)
            if random.random() < 0.5:
                y = random_gain(y, 0.8, 1.25)
            if random.random() < 0.4:
                y = add_pink_noise(y, (15, 30))
        return y

    def _to_spec(self, y):
        if len(y) < N_FFT:
            y = np.pad(y, (0, N_FFT - len(y)))
        mel = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, fmax=SAMPLE_RATE // 2
        )
        spec_db = librosa.power_to_db(mel + 1e-10, ref=np.max)
        # z-score per sample
        mean, std = spec_db.mean(), spec_db.std() + 1e-8
        spec_db = (spec_db - mean) / std
        S = torch.from_numpy(spec_db).float()
        if self.mode == 'train':
            S = spec_augment(S)
        return S  # (MELS, T)

    def _crop_window(self, spec, win_frames, start=None, center=False):
        T = spec.shape[1]
        if T <= win_frames:
            pad_total = win_frames - T
            left, right = pad_total // 2, pad_total - (pad_total // 2)
            return F.pad(spec, (left, right, 0, 0))
        if start is None:
            start = random.randint(0, T - win_frames)
        elif center:
            start = (T - win_frames) // 2
        else:
            start = min(max(0, start), T - win_frames)
        return spec[:, start:start + win_frames]

    def __getitem__(self, idx):
        if self.mode == 'train':
            fid = int(self.index[idx])
            path = self.file_paths[fid]
            label = int(self.labels[fid])
            b = int(self.file_bin[fid])

            y = self._load_wave(path)
            S = self._to_spec(y)
            win_frames = seconds_to_frames(BIN_MID[b])
            W = self._crop_window(S, win_frames, start=None)

            img = W.unsqueeze(0).repeat(3, 1, 1)
            if self.transform: img = self.transform(img)
            return img, label, b, fid

        else:
            fid, b, start = self.window_meta[int(idx)]
            path = self.file_paths[fid]
            label = int(self.labels[fid])
            y = self._load_wave(path)
            S = self._to_spec(y)
            win_frames = seconds_to_frames(BIN_MID[b])
            W = self._crop_window(S, win_frames, start=start)

            img = W.unsqueeze(0).repeat(3, 1, 1)
            if self.transform: img = self.transform(img)
            return img, label, b, fid



def make_backbone(name: str):
    name = name.lower()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat = m.fc.in_features; m.fc = nn.Identity()
    elif name == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        feat = m.fc.in_features; m.fc = nn.Identity()
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feat = m.fc.in_features; m.fc = nn.Identity()
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        feat = m.classifier[1].in_features; m.classifier = nn.Identity()
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        feat = m.classifier[-1].in_features; m.classifier = nn.Identity()
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        feat = m.classifier[2].in_features; m.classifier = nn.Identity()
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return m, feat

class LenAwareNet(nn.Module):
    def __init__(self, backbone_name="resnet18", len_emb_dim=LEN_EMB_DIM, num_bins=NUM_BINS, freeze_until=FREEZE_UNTIL):
        super().__init__()
        self.backbone, feat_dim = make_backbone(backbone_name)

        if backbone_name.startswith("resnet"):
            freeze_names = {'conv1', 'bn1', 'layer1', 'layer2'} if freeze_until == 'layer2' else set()
            for n, p in self.backbone.named_parameters():
                block = n.split('.')[0]
                if block in freeze_names:
                    p.requires_grad = False

        self.len_emb = nn.Embedding(num_bins, len_emb_dim)
        self.head = nn.Sequential(
            nn.Linear(feat_dim + len_emb_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x, bin_ids):
        feats = self.backbone(x)
        if feats.dim() > 2:
            feats = feats.mean([-2, -1])
        len_vec = self.len_emb(bin_ids)
        fused = torch.cat([feats, len_vec], dim=1)
        return self.head(fused)



class FocalLossCB(nn.Module):

    def __init__(self, gamma=FOCAL_GAMMA, alpha=None, epsilon=1e-8):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = epsilon

    def forward(self, logits, targets):

        probs = F.softmax(logits, dim=1).clamp(min=self.eps, max=1.0 - self.eps)
        pt = probs[torch.arange(probs.size(0)), targets]
        logpt = torch.log(pt)
        if self.alpha is not None:
            a = torch.tensor(self.alpha, dtype=probs.dtype, device=probs.device)
            at = a[targets]
        else:
            counts = torch.bincount(targets, minlength=probs.size(1)).float()
            p = counts / counts.sum().clamp(min=1.0)
            a = (1.0 - p).to(probs.device)
            at = a[targets]
        loss = - at * ((1 - pt) ** self.gamma) * logpt
        return loss.mean()



class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                new_avg = self.decay * self.shadow[name] + (1.0 - self.decay) * p.data
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self, model):
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.data.clone()
                p.data = self.shadow[name].clone()

    def restore(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.data = self.backup[name].clone()
        self.backup = {}



to_224_and_normalize = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])



def build_sampler_label_bin(train_ds: LengthBinAudioDataset):
    keys = []
    for i in range(len(train_ds)):
        fid = int(train_ds.index[i])
        lb = int(train_ds.labels[fid])
        b = int(train_ds.file_bin[fid])
        keys.append((lb, b))
    cnt = Counter(keys)
    weights = []
    for i in range(len(train_ds)):
        fid = int(train_ds.index[i])
        lb = int(train_ds.labels[fid]); b = int(train_ds.file_bin[fid])
        weights.append(1.0 / cnt[(lb, b)])
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.float32),
                                 num_samples=len(weights), replacement=True)

def aggregate_file_logits(file_ids, logits, labels):
    file_ids = np.array(file_ids)
    labels = np.array(labels)
    logits = np.stack(logits, axis=0)
    by_file = defaultdict(list); by_label = {}
    for fid, logit, y in zip(file_ids, logits, labels):
        by_file[fid].append(logit); by_label[fid] = y
    agg_logits = {fid: np.mean(np.stack(v, 0), 0) for fid, v in by_file.items()}
    fids = sorted(agg_logits.keys())
    y_true = np.array([by_label[fid] for fid in fids], dtype=np.int64)
    y_pred = np.array([np.argmax(agg_logits[fid]) for fid in fids], dtype=np.int64)
    return fids, y_true, y_pred



def mixup_batch(imgs, labels, bins, alpha=MIXUP_ALPHA, p=0.5):
    if alpha <= 0 or random.random() > p:
        return imgs, None, None, None

    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0), device=imgs.device)

    mixed_imgs = lam * imgs + (1 - lam) * imgs[idx]
    labels_a, labels_b = labels, labels[idx]
    return mixed_imgs, (labels_a, labels_b, lam), None, idx




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='Data')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    ap.add_argument('--backbone', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'mobilenet_v3_large', 'convnext_tiny'])
    ap.add_argument('--lr', type=float, default=BASE_LR)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--ckpt', type=str, default=CKPT)
    args = ap.parse_args()

    tmp = LengthBinAudioDataset(args.data_dir, mode='train', indices=None, transform=None)
    n = len(tmp.file_paths)
    labels = tmp.labels.copy()
    idx = np.arange(n)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED) 
    train_idx, hold_idx = next(sss1.split(idx, labels))
    hold_labels = labels[hold_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)  
    val_rel, test_rel = next(sss2.split(hold_idx, hold_labels))
    val_idx = hold_idx[val_rel]
    test_idx = hold_idx[test_rel]

    train_ds = LengthBinAudioDataset(args.data_dir, mode='train', indices=train_idx, transform=to_224_and_normalize)
    val_ds   = LengthBinAudioDataset(args.data_dir, mode='val',   indices=val_idx,   transform=to_224_and_normalize, tta_offsets=VAL_TTA_OFFSETS)
    test_ds  = LengthBinAudioDataset(args.data_dir, mode='test',  indices=test_idx,  transform=to_224_and_normalize, tta_offsets=TEST_TTA_OFFSETS)

    sampler = build_sampler_label_bin(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = LenAwareNet(backbone_name=args.backbone).to(DEVICE)

    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if n.startswith('backbone'):
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1, 'weight_decay': WEIGHT_DECAY_BACKBONE},
        {'params': head_params,     'lr': args.lr,       'weight_decay': WEIGHT_DECAY_HEAD}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    train_counts = np.bincount(train_ds.labels, minlength=2).astype(np.float32)
    p = train_counts / train_counts.sum()
    alpha = (1.0 - p).tolist() if FOCAL_ALPHA is None else FOCAL_ALPHA
    criterion = FocalLossCB(gamma=FOCAL_GAMMA, alpha=alpha)

    ema = EMA(model, decay=EMA_DECAY)

    start_epoch = 1
    best_val_acc = 0.0
    best_val_loss = float('inf')
    if args.resume and os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        ema = EMA(model, decay=EMA_DECAY)
        print(f"Resumed from epoch {start_epoch}")

    patience = 0
    print("\n" + "=" * 50)
    print("Training Configuration")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print(f"Backbone: {args.backbone}")
    print(f"Train files: {len(train_ds.file_paths)} | Val files: {len(val_ds.file_paths)} | Test files: {len(test_ds.file_paths)}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print("=" * 50 + "\n")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for bi, (imgs, labels, bin_ids, fids) in enumerate(train_loader):
            imgs = imgs.to(DEVICE); labels = labels.to(DEVICE); bin_ids = bin_ids.to(DEVICE)

            mixed, mix_labels, _, _ = mixup_batch(imgs, labels, bin_ids, alpha=MIXUP_ALPHA, p=0.7)

            optimizer.zero_grad()
            logits = model(mixed, bin_ids)

            if mix_labels is None:
                loss = criterion(logits, labels)
                preds = logits.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            else:
                la, lb, lam = mix_labels
                loss = lam * criterion(logits, la) + (1 - lam) * criterion(logits, lb)
                with torch.no_grad():
                    preds = logits.argmax(1)
                    correct += (preds == la).float().mul_(lam).sum().item()
                    correct += (preds == lb).float().mul_(1 - lam).sum().item()
                    total += labels.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)

            running_loss += loss.item() * imgs.size(0)

            if bi % 10 == 0:
                print(f"\rEpoch {epoch}: [{bi}/{len(train_loader)}] Loss: {loss.item():.4f}", end='')

        train_loss = running_loss / max(1, len(train_loader.dataset))
        train_acc = correct / max(1, total)

        model.eval()
        ema.apply_shadow(model)
        val_running_loss, val_seen = 0.0, 0
        window_logits, window_labels, window_fids = [], [], []

        with torch.no_grad():
            for imgs, labels, bin_ids, fids in val_loader:
                imgs = imgs.to(DEVICE); labels = labels.to(DEVICE); bin_ids = bin_ids.to(DEVICE)
                logits = model(imgs, bin_ids)
                loss = criterion(logits, labels)
                val_running_loss += loss.item() * imgs.size(0); val_seen += imgs.size(0)
                window_logits.extend(logits.cpu().numpy())
                window_labels.extend(labels.cpu().numpy())
                window_fids.extend(fids.numpy())

        val_loss = val_running_loss / max(1, val_seen)
        _, y_true, y_pred = aggregate_file_logits(window_fids, window_logits, window_labels)
        val_acc = (y_true == y_pred).mean()
        ema.restore(model)

        scheduler.step()

        print(f"\n{'=' * 50}")
        print(f"Epoch [{epoch}/{args.epochs}]  LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'=' * 50}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        improved = (val_acc > best_val_acc + 1e-4) or (val_loss < best_val_loss - 1e-4)
        if improved:
            best_val_acc = max(best_val_acc, val_acc)
            best_val_loss = min(best_val_loss, val_loss)
            patience = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc
            }, args.ckpt)
            print(f"✓ New best model saved! (Val Acc: {val_acc:.4f})")
        else:
            patience += 1
            print(f"Patience: {patience}/{PATIENCE}")
            if patience >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        ema = EMA(model, decay=EMA_DECAY)
        ema.apply_shadow(model)

        window_logits, window_labels, window_fids = [], [], []
        with torch.no_grad():
            for imgs, labels, bin_ids, fids in test_loader:
                imgs = imgs.to(DEVICE); labels = labels.to(DEVICE); bin_ids = bin_ids.to(DEVICE)
                logits = model(imgs, bin_ids)
                window_logits.extend(logits.cpu().numpy())
                window_labels.extend(labels.cpu().numpy())
                window_fids.extend(fids.numpy())

        fids, y_true, y_pred = aggregate_file_logits(window_fids, window_logits, window_labels)
        acc = (y_true == y_pred).mean()
        cm = confusion_matrix(y_true, y_pred)

        indoor_acc = (y_true[y_true == 0] == y_pred[y_true == 0]).mean() if np.any(y_true == 0) else 0.0
        outdoor_acc = (y_true[y_true == 1] == y_pred[y_true == 1]).mean() if np.any(y_true == 1) else 0.0

        print("\n" + "=" * 50)
        print("FINAL TEST EVALUATION (per-file, EMA)")
        print("=" * 50)
        print(f"Overall Test Accuracy: {acc:.4f}")
        print(f"Indoor Accuracy:  {indoor_acc:.4f}")
        print(f"Outdoor Accuracy: {outdoor_acc:.4f}")

        print("\nConfusion Matrix:")
        print("         Pred Indoor  Pred Outdoor")
        print(f"Indoor:      {cm[0, 0]:4d}         {cm[0, 1]:4d}")
        print(f"Outdoor:     {cm[1, 0]:4d}         {cm[1, 1]:4d}")

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=BINARY_CLASSES, digits=4))

        print(f"\nBest Val Acc:  {ckpt.get('best_val_acc', 0.0):.4f}")
        print(f"Best Val Loss: {ckpt.get('best_val_loss', float('inf')):.4f}")
        print(f"Best Epoch:    {ckpt.get('epoch', -1)}")


if __name__ == '__main__':
    main()
