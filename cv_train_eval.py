import os, json, numpy as np, torch, argparse
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

from script1 import (
    LengthBinAudioDataset, to_224_and_normalize, LenAwareNet, EMA,
    build_sampler_label_bin, aggregate_file_logits,
    DEVICE, SEED, VAL_TTA_OFFSETS, TEST_TTA_OFFSETS,
    BINARY_CLASSES, FocalLossCB, FOCAL_GAMMA,
    WEIGHT_DECAY_BACKBONE, WEIGHT_DECAY_HEAD, LEN_EMB_DIM, NUM_BINS, FREEZE_UNTIL,
    MIXUP_ALPHA, mixup_batch
)

def make_loaders(data_dir, train_idx, val_idx, test_idx, batch_size):
    train_ds = LengthBinAudioDataset(data_dir, mode='train', indices=train_idx, transform=to_224_and_normalize)
    val_ds   = LengthBinAudioDataset(data_dir, mode='val',   indices=val_idx,   transform=to_224_and_normalize, tta_offsets=VAL_TTA_OFFSETS)
    test_ds  = LengthBinAudioDataset(data_dir, mode='test',  indices=test_idx,  transform=to_224_and_normalize, tta_offsets=TEST_TTA_OFFSETS)

    sampler = build_sampler_label_bin(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader

@torch.no_grad()
def eval_loader_per_file(model, loader, criterion=None, use_ema=True):
    model.eval()
    ema = None
    if use_ema:
        ema = EMA(model); ema.apply_shadow(model)

    total_loss, total_seen = 0.0, 0
    window_logits, window_labels, window_fids = [], [], []
    for imgs, labels, bin_ids, fids in loader:
        imgs = imgs.to(DEVICE); labels = labels.to(DEVICE); bin_ids = bin_ids.to(DEVICE)
        logits = model(imgs, bin_ids)
        if criterion is not None:
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * imgs.size(0)
            total_seen += imgs.size(0)
        window_logits.extend(logits.detach().cpu().numpy())
        window_labels.extend(labels.detach().cpu().numpy())
        window_fids.extend(fids.numpy())

    if ema is not None: ema.restore(model)

    val_loss = (total_loss / total_seen) if total_seen else None
    _, y_true, y_pred = aggregate_file_logits(window_fids, window_logits, window_labels)
    acc = (y_true == y_pred).mean()
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    rep = classification_report(y_true, y_pred, target_names=BINARY_CLASSES, digits=4)
    f1 = f1_score(y_true, y_pred, average="macro")
    return val_loss, acc, f1, cm, rep, y_true, y_pred

def train_one_fold(args, fold_id, train_idx, val_idx, test_idx, out_dir):
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = make_loaders(
        args.data_dir, train_idx, val_idx, test_idx, args.batch_size
    )

    model = LenAwareNet(backbone_name=args.backbone,
                        len_emb_dim=LEN_EMB_DIM, num_bins=NUM_BINS,
                        freeze_until=FREEZE_UNTIL).to(DEVICE)

    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (backbone_params if n.startswith("backbone") else head_params).append(p)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1, 'weight_decay': WEIGHT_DECAY_BACKBONE},
        {'params': head_params,     'lr': args.lr,       'weight_decay': WEIGHT_DECAY_HEAD}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    if args.loss == "ce_ls":
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    elif args.loss == "ce_cls_weight":
        w = torch.tensor([1.0, 1.15], device=DEVICE)
        criterion = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    else:
        tr_counts = np.bincount(train_ds.labels, minlength=2).astype(np.float32)
        p = tr_counts / tr_counts.sum()
        alpha = (1.0 - p).tolist()
        criterion = FocalLossCB(gamma=FOCAL_GAMMA, alpha=alpha)

    ema = EMA(model)
    best = {"val_acc": 0.0, "val_loss": float("inf")}
    patience, bad = args.patience, 0

    for e in range(1, args.epochs + 1):
        model.train()
        run_loss, correct, seen = 0.0, 0, 0

        mixup_p = 0.0 if (args.no_mixup_last > 0 and e > args.epochs - args.no_mixup_last) else 0.7

        for bi, (imgs, labels, bin_ids, _) in enumerate(train_loader):
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

        train_loss = run_loss / max(1, len(train_loader.dataset))
        train_acc = correct / max(1, seen)

        val_loss, val_acc, val_f1, _, _, _, _ = eval_loader_per_file(model, val_loader, criterion, use_ema=True)
        scheduler.step()

        print(f"[Fold {fold_id}] Epoch {e:03d} | Train L {train_loss:.4f} A {train_acc:.4f} "
              f"| Val L {val_loss:.4f} A {val_acc:.4f} F1 {val_f1:.4f}")

        improved = (val_acc > best["val_acc"] + 1e-4) or (val_loss < best["val_loss"] - 1e-4)
        if improved:
            best["val_acc"] = max(best["val_acc"], val_acc)
            best["val_loss"] = min(best["val_loss"], val_loss)
            bad = 0
            path = os.path.join(out_dir, f"fold{fold_id}_best.pth")
            torch.save({
                "epoch": e,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best["val_acc"],
                "best_val_loss": best["val_loss"],
            }, path)
            print(f"  ✓ saved {path}")
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop (patience={patience})")
                break

    ckpt = torch.load(os.path.join(out_dir, f"fold{fold_id}_best.pth"), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    tloss, tacc, tf1, tcm, trep, y_true, y_pred = eval_loader_per_file(model, test_loader, None, use_ema=True)

    results = {
        "fold": fold_id,
        "best_val_acc": best["val_acc"],
        "best_val_loss": best["val_loss"],
        "test_acc": float(tacc),
        "test_f1_macro": float(tf1),
        "test_confusion": tcm.tolist(),
        "test_report": trep,
    }
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="Data")
    ap.add_argument("--backbone", type=str, default="resnet18",
                    choices=["resnet18","resnet34","resnet50","efficientnet_b0","mobilenet_v3_large","convnext_tiny"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--val_frac", type=float, default=0.2, help="fraction of (trainval) used for val inside each fold")
    ap.add_argument("--loss", type=str, default="ce_ls", choices=["ce_ls","ce_cls_weight","focal"])
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--no_mixup_last", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default="checkpoints_cv")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tmp = LengthBinAudioDataset(args.data_dir, mode='train', indices=None, transform=None)
    n = len(tmp.file_paths); idx = np.arange(n); labels = tmp.labels.copy()

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=SEED)
    all_results = []
    for fold_id, (trainval_idx, test_idx) in enumerate(skf.split(idx, labels), start=1):
        trlabels = labels[trainval_idx]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=SEED + fold_id)
        tr_rel, val_rel = next(sss.split(trainval_idx, trlabels))
        train_idx = trainval_idx[tr_rel]
        val_idx   = trainval_idx[val_rel]

        print(f"\n===== FOLD {fold_id}/{args.k} =====")
        print(f"train: {len(train_idx)} | val: {len(val_idx)} | test: {len(test_idx)}")
        res = train_one_fold(args, fold_id, train_idx, val_idx, test_idx, args.out_dir)
        all_results.append(res)
        print(f"[Fold {fold_id}] Test Acc: {res['test_acc']:.4f} | Test F1_macro: {res['test_f1_macro']:.4f}\n")

    test_accs = [r["test_acc"] for r in all_results]
    test_f1s  = [r["test_f1_macro"] for r in all_results]
    summary = {
        "k": args.k,
        "backbone": args.backbone,
        "loss": args.loss,
        "epochs": args.epochs,
        "mean_test_acc": float(np.mean(test_accs)),
        "std_test_acc": float(np.std(test_accs)),
        "mean_test_f1_macro": float(np.mean(test_f1s)),
        "std_test_f1_macro": float(np.std(test_f1s)),
        "folds": all_results
    }
    with open(os.path.join(args.out_dir, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n====== CV SUMMARY ======")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
