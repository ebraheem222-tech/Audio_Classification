import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report

from script1 import (
    LengthBinAudioDataset,
    to_224_and_normalize,
    LenAwareNet,
    EMA,
    aggregate_file_logits,
    build_sampler_label_bin,
    mixup_batch,
    MIXUP_ALPHA,
    DEVICE,
    CKPT,
    SEED,
    VAL_TTA_OFFSETS,
    BINARY_CLASSES,
    FocalLossCB,
    FOCAL_GAMMA,
    WEIGHT_DECAY_BACKBONE,
    WEIGHT_DECAY_HEAD,
    LEN_EMB_DIM,
    NUM_BINS,
    FREEZE_UNTIL,
)

def build_splits(data_dir: str):
   
    tmp = LengthBinAudioDataset(data_dir, mode='train', indices=None, transform=None)
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
    return train_idx, val_idx, test_idx


@torch.no_grad()
def validate_epoch(model, loader, criterion, use_ema=True):
    """Window-level inference -> per-file aggregation (EMA applied)."""
    model.eval()
    ema_ctx = None
    if use_ema:
        ema_ctx = EMA(model)
        ema_ctx.apply_shadow(model)

    total_loss, total_seen = 0.0, 0
    window_logits, window_labels, window_fids = [], [], []

    for imgs, labels, bin_ids, fids in loader:
        imgs = imgs.to(DEVICE); labels = labels.to(DEVICE); bin_ids = bin_ids.to(DEVICE)
        logits = model(imgs, bin_ids)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * imgs.size(0)
        total_seen += imgs.size(0)
        window_logits.extend(logits.detach().cpu().numpy())
        window_labels.extend(labels.detach().cpu().numpy())
        window_fids.extend(fids.numpy())

    if ema_ctx is not None:
        ema_ctx.restore(model)

    val_loss = total_loss / max(1, total_seen)
    _, y_true, y_pred = aggregate_file_logits(window_fids, window_logits, window_labels)
    val_acc = (y_true == y_pred).mean()
    return val_loss, val_acc, y_true, y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='Data')
    ap.add_argument('--ckpt', type=str, default=CKPT, help='Path to the checkpoint to resume from')
    ap.add_argument('--out_dir', type=str, default='checkpoints_ft')
    ap.add_argument('--backbone', type=str, default='resnet18',
                    choices=['resnet18','resnet34','resnet50','efficientnet_b0','mobilenet_v3_large','convnext_tiny'])
    ap.add_argument('--epochs', type=int, default=80, help='fine-tune budget (extra epochs)')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--base_lr', type=float, default=3e-4, help='base (head) LR before shrink on resume')
    ap.add_argument('--lr_shrink', type=float, default=0.3, help='multiply LRs by this factor on resume')
    ap.add_argument('--min_lr', type=float, default=1e-7)
    ap.add_argument('--no_mixup_last', type=int, default=5, help='disable MixUp in last N epochs')
    ap.add_argument('--loss', type=str, default='ce_ls', choices=['ce_ls','focal'],
                    help='ce_ls = CrossEntropy(label_smoothing=0.1), focal = FocalLossCB')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_idx, val_idx, _ = build_splits(args.data_dir)

    train_ds = LengthBinAudioDataset(args.data_dir, mode='train', indices=train_idx, transform=to_224_and_normalize)
    val_ds   = LengthBinAudioDataset(args.data_dir, mode='val',   indices=val_idx,
                                     transform=to_224_and_normalize, tta_offsets=VAL_TTA_OFFSETS)

    sampler = build_sampler_label_bin(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = LenAwareNet(backbone_name=args.backbone,
                        len_emb_dim=LEN_EMB_DIM,
                        num_bins=NUM_BINS,
                        freeze_until=FREEZE_UNTIL).to(DEVICE)

    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if n.startswith('backbone'):
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.base_lr * 0.1, 'weight_decay': WEIGHT_DECAY_BACKBONE},
        {'params': head_params,     'lr': args.base_lr,       'weight_decay': WEIGHT_DECAY_HEAD},
    ])

    assert os.path.exists(args.ckpt), f"Checkpoint not found: {args.ckpt}"
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    if 'optimizer_state_dict' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception:
            pass
    for g in optimizer.param_groups:
        g['lr'] = max(g['lr'] * args.lr_shrink, args.min_lr)

    best_val_acc_ckpt  = ckpt.get('best_val_acc', 0.0)
    best_val_loss_ckpt = ckpt.get('best_val_loss', float('inf'))
    start_epoch_ckpt   = ckpt.get('epoch', 0)
    print(f"Resumed from: {args.ckpt}")
    print(f"  last_epoch       : {start_epoch_ckpt}")
    print(f"  best_val_acc_ckpt: {best_val_acc_ckpt:.4f}")
    print(f"  best_val_loss_ckpt: {best_val_loss_ckpt:.4f}")
    print(f"  LR after shrink  : {[pg['lr'] for pg in optimizer.param_groups]}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=args.min_lr
    )

    if args.loss == 'ce_ls':
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        tr_counts = np.bincount(train_ds.labels, minlength=2).astype(np.float32)
        p = tr_counts / tr_counts.sum()
        alpha = (1.0 - p).tolist()
        criterion = FocalLossCB(gamma=FOCAL_GAMMA, alpha=alpha)

    ema = EMA(model)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience, bad = 20, 0

    for e in range(1, args.epochs + 1):
        model.train()
        run_loss, correct, seen = 0.0, 0, 0

        mixup_p = 0.0 if (args.no_mixup_last > 0 and e > args.epochs - args.no_mixup_last) else 0.7

        for bi, (imgs, labels, bin_ids, fids) in enumerate(train_loader):
            imgs = imgs.to(DEVICE); labels = labels.to(DEVICE); bin_ids = bin_ids.to(DEVICE)

            mixed, mix_labels, _, _ = mixup_batch(imgs, labels, bin_ids, alpha=MIXUP_ALPHA, p=mixup_p)

            optimizer.zero_grad(set_to_none=True)
            logits = model(mixed, bin_ids)

            if mix_labels is None:
                loss = criterion(logits, labels)
                preds = logits.argmax(1)
                correct += (preds == labels).sum().item()
                seen += labels.size(0)
            else:
                la, lb, lam = mix_labels
                loss = lam * criterion(logits, la) + (1 - lam) * criterion(logits, lb)
                with torch.no_grad():
                    preds = logits.argmax(1)
                    correct += (preds == la).float().mul_(lam).sum().item()
                    correct += (preds == lb).float().mul_(1 - lam).sum().item()
                    seen += labels.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)

            run_loss += float(loss.item()) * imgs.size(0)

            if bi % 10 == 0:
                print(f"\rEpoch {e:03d} [{bi:03d}/{len(train_loader)}] "
                      f"Loss {loss.item():.4f}", end="")

        train_loss = run_loss / max(1, len(train_loader.dataset))
        train_acc = correct / max(1, seen)

        val_loss, val_acc, y_true, y_pred = validate_epoch(model, val_loader, criterion, use_ema=True)

        scheduler.step(e)  

        print(f"\n{'='*60}")
        print(f"Epoch {e:03d} | "
              f"LRs {[f'{pg['lr']:.2e}' for pg in optimizer.param_groups]} | "
              f"Train L {train_loss:.4f} A {train_acc:.4f} | "
              f"Val L {val_loss:.4f} A {val_acc:.4f}")

        improved = (val_acc > best_val_acc + 1e-4) or (val_loss < best_val_loss - 1e-4)
        if improved:
            best_val_acc = max(best_val_acc, val_acc)
            best_val_loss = min(best_val_loss, val_loss)
            bad = 0
            torch.save({
                'epoch': start_epoch_ckpt + e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss,
            }, os.path.join(args.out_dir, "best_finetune.ckpt"))
            print(f"✓ New BEST saved to {os.path.join(args.out_dir, 'best_finetune.ckpt')}")
        else:
            bad += 1
            print(f"Patience: {bad}/{patience}")
            if bad >= patience:
                print("Early stopping.")
                break

    print("\nFine-tune finished.")
    print(f"Best Val Acc : {best_val_acc:.4f}")
    print(f"Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
