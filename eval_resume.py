import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

from script1 import (
    LengthBinAudioDataset,
    to_224_and_normalize,
    LenAwareNet,
    EMA,
    DEVICE,
    CKPT,
    SEED,
    VAL_TTA_OFFSETS,
    TEST_TTA_OFFSETS,
    BINARY_CLASSES,
    seconds_to_frames,
)

def build_splits(data_dir: str):
    tmp = LengthBinAudioDataset(data_dir, mode='train', indices=None, transform=None)
    n = len(tmp.file_paths)
    labels = tmp.labels.copy()
    idx = np.arange(n)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)  # 70/30
    train_idx, hold_idx = next(sss1.split(idx, labels))
    hold_labels = labels[hold_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)  # 15/15
    val_rel, test_rel = next(sss2.split(hold_idx, hold_labels))
    val_idx = hold_idx[val_rel]
    test_idx = hold_idx[test_rel]
    return train_idx, val_idx, test_idx

# ---------- helpers ----------
def load_model(ckpt_path, backbone):
    m = LenAwareNet(backbone_name=backbone).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt['model_state_dict'])
    return m, ckpt

@torch.no_grad()
def collect_window_probs(model, loader, use_ema=True):
    """Return per-window Outdoor probability, labels and file ids."""
    model.eval()
    ema_ctx = None
    if use_ema:
        ema_ctx = EMA(model); ema_ctx.apply_shadow(model)

    probs_out, labels, file_ids = [], [], []
    for imgs, y, bin_ids, fids in loader:
        imgs = imgs.to(DEVICE); y = y.to(DEVICE); bin_ids = bin_ids.to(DEVICE)
        logits = model(imgs, bin_ids)
        p = F.softmax(logits, dim=1)[:, 1] 
        probs_out.append(p.cpu().numpy())
        labels.append(y.cpu().numpy())
        file_ids.append(fids.numpy())

    if ema_ctx is not None:
        ema_ctx.restore(model)

    return (np.concatenate(probs_out),
            np.concatenate(labels),
            np.concatenate(file_ids))

def aggregate_per_file_probs(file_ids, probs, labels):
    """Average window probs per file -> per-file (p_outdoor, y_true)."""
    by_file_probs = {}
    by_file_label = {}
    for fid, p, y in zip(file_ids, probs, labels):
        by_file_probs.setdefault(fid, []).append(p)
        by_file_label[fid] = y
    fids = sorted(by_file_probs.keys())
    p_out = np.array([np.mean(by_file_probs[fid]) for fid in fids], dtype=np.float32)
    y_true = np.array([by_file_label[fid] for fid in fids], dtype=np.int64)
    return fids, p_out, y_true

def search_best_threshold(p_out_val, y_val, t_min=0.3, t_max=0.7, steps=41):
    best_t, best_acc, best_f1 = 0.5, 0.0, 0.0
    for t in np.linspace(t_min, t_max, steps):
        preds = (p_out_val >= t).astype(int)
        acc = accuracy_score(y_val, preds)
        f1  = f1_score(y_val, preds)
        if acc > best_acc or (acc == best_acc and f1 > best_f1):
            best_t, best_acc, best_f1 = t, acc, f1
    return best_t, best_acc, best_f1

def report_from_probs(p_out, y_true, threshold):
    y_pred = (p_out >= threshold).astype(int)
    acc = (y_pred == y_true).mean()
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    rep = classification_report(y_true, y_pred, target_names=BINARY_CLASSES, digits=4)
    return acc, cm, rep, y_pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='Data')
    ap.add_argument('--ckpt', type=str, nargs='+', default=[CKPT], help='one or more checkpoints for ensemble')
    ap.add_argument('--backbone', type=str, default='resnet18',
                    choices=['resnet18','resnet34','resnet50','efficientnet_b0','mobilenet_v3_large','convnext_tiny'])
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--tmin', type=float, default=0.3)
    ap.add_argument('--tmax', type=float, default=0.7)
    ap.add_argument('--tsteps', type=int, default=41)
    args = ap.parse_args()

    _, val_idx, test_idx = build_splits(args.data_dir)

    val_ds  = LengthBinAudioDataset(args.data_dir, mode='val',
                                    indices=val_idx, transform=to_224_and_normalize,
                                    tta_offsets=VAL_TTA_OFFSETS)
    test_ds = LengthBinAudioDataset(args.data_dir, mode='test',
                                    indices=test_idx, transform=to_224_and_normalize,
                                    tta_offsets=TEST_TTA_OFFSETS)
    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    models = []
    ckpt_meta = []
    for path in args.ckpt:
        assert os.path.exists(path), f"Missing ckpt: {path}"
        m, meta = load_model(path, args.backbone)
        models.append(m); ckpt_meta.append(meta)

    def ensemble_probs(loader):
        all_file_ids, all_probs, all_labels = None, [], None
        for m in models:
            p, y, f = collect_window_probs(m, loader, use_ema=True)

            if all_file_ids is None: all_file_ids = f
            all_probs.append(p)
            if all_labels is None: all_labels = y
        probs_mean = np.mean(np.stack(all_probs, axis=0), axis=0)
        return probs_mean, all_labels, all_file_ids

    p_val_w, y_val_w, f_val = ensemble_probs(val_loader)
    _, p_val, y_val = aggregate_per_file_probs(f_val, p_val_w, y_val_w)
    t_best, val_acc_best, val_f1_best = search_best_threshold(p_val, y_val, args.tmin, args.tmax, args.tsteps)

    p_test_w, y_test_w, f_test = ensemble_probs(test_loader)
    _, p_test, y_test = aggregate_per_file_probs(f_test, p_test_w, y_test_w)

    acc_val_argmax, cm_val_argmax, rep_val_argmax, _ = report_from_probs(p_val, y_val, 0.5)
    acc_test_argmax, cm_test_argmax, rep_test_argmax, _ = report_from_probs(p_test, y_test, 0.5)

    acc_val_t, cm_val_t, rep_val_t, _ = report_from_probs(p_val, y_val, t_best)
    acc_test_t, cm_test_t, rep_test_t, _ = report_from_probs(p_test, y_test, t_best)

    print("\nLoaded checkpoints:")
    for i, path in enumerate(args.ckpt):
        be = ckpt_meta[i].get('epoch', None)
        ba = ckpt_meta[i].get('best_val_acc', None)
        bl = ckpt_meta[i].get('best_val_loss', None)
        print(f"  - {path} | epoch={be} | best_val_acc={ba} | best_val_loss={bl}")

    print("\n==================== THRESHOLD SEARCH (VAL) ====================")
    print(f"Best threshold (Outdoor prob) = {t_best:.3f} | "
          f"Val Acc @t* = {acc_val_t:.4f} (argmax= {acc_val_argmax:.4f}) | "
          f"Val F1* ≈ {val_f1_best:.4f}")

    print("\n==================== VALIDATION (per-file) ====================")
    print("Argmax (t=0.5):")
    print(f"Acc: {acc_val_argmax:.4f}\nConfusion:\n{cm_val_argmax}\n")
    print("Threshold-tuned:")
    print(f"Acc: {acc_val_t:.4f}\nConfusion:\n{cm_val_t}\n")
    print("Classification Report (tuned):")
    print(rep_val_t)

    print("\n==================== TEST (per-file) ====================")
    print("Argmax (t=0.5):")
    print(f"Acc: {acc_test_argmax:.4f}\nConfusion:\n{cm_test_argmax}\n")
    print(f"Threshold-tuned (t={t_best:.3f}):")
    print(f"Acc: {acc_test_t:.4f}\nConfusion:\n{cm_test_t}\n")
    print("Classification Report (tuned):")
    print(rep_test_t)

if __name__ == "__main__":
    main()
