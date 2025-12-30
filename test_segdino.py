# -*- coding: utf-8 -*-
# test_segdinov3.py
import os
import csv
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# -------------------- Utils (consistent with train visualization) --------------------
def tensor_to_rgb(img_t: torch.Tensor, mean=None, std=None) -> np.ndarray:
    img = img_t.detach().cpu().float()
    img = img.clamp(0, 1).numpy()
    img = (img * 255.0).round().astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def mask_to_gray(mask_t: torch.Tensor, thr: float = 0.5) -> np.ndarray:
    m = mask_t.detach().cpu().float()
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    elif m.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected mask tensor shape: {m.shape}")
    if m.max() > 1.0 or m.min() < 0.0:
        m = torch.sigmoid(m)
    m_bin = (m > thr).float()
    m_img = (m_bin * 255.0).round().clamp(0, 255).byte().numpy()
    return m_img

@torch.no_grad()
def save_eval_visuals(idx, inputs, logits, targets, out_dir, thr=0.5, fname_prefix="test"):
    os.makedirs(out_dir, exist_ok=True)
    img_bgr = tensor_to_rgb(inputs)
    pred_gray = mask_to_gray(logits, thr)
    gt_gray   = mask_to_gray(targets, thr)
    base = os.path.join(out_dir, f"{fname_prefix}_{idx:05d}")
    cv2.imwrite(base + "_img.png",  img_bgr)
    cv2.imwrite(base + "_pred.png", pred_gray)
    cv2.imwrite(base + "_gt.png",   gt_gray)

# -------------------- Metrics: Dice / IoU / HD95 --------------------
def iou_binary_numpy(pred_bin: np.ndarray, tgt_bin: np.ndarray, eps=1e-6) -> float:
    inter = np.logical_and(pred_bin, tgt_bin).sum(dtype=np.float64)
    union = np.logical_or(pred_bin, tgt_bin).sum(dtype=np.float64)
    return float((inter + eps) / (union + eps))

def dice_binary_numpy(pred_bin: np.ndarray, tgt_bin: np.ndarray, eps=1e-6) -> float:
    inter = np.logical_and(pred_bin, tgt_bin).sum(dtype=np.float64)
    s = pred_bin.sum(dtype=np.float64) + tgt_bin.sum(dtype=np.float64)
    return float((2.0 * inter + eps) / (s + eps))

def precision_recall_f1_numpy(pred_bin: np.ndarray, tgt_bin: np.ndarray, eps=1e-6) -> tuple:
    """Calculate Precision, Recall, and F1 score."""
    tp = np.logical_and(pred_bin, tgt_bin).sum(dtype=np.float64)  # True Positive
    fp = np.logical_and(pred_bin, np.logical_not(tgt_bin)).sum(dtype=np.float64)  # False Positive
    fn = np.logical_and(np.logical_not(pred_bin), tgt_bin).sum(dtype=np.float64)  # False Negative
    tn = np.logical_and(np.logical_not(pred_bin), np.logical_not(tgt_bin)).sum(dtype=np.float64)  # True Negative
    
    precision = float((tp + eps) / (tp + fp + eps))
    recall = float((tp + eps) / (tp + fn + eps))
    f1 = float((2 * tp + eps) / (2 * tp + fp + fn + eps))
    accuracy = float((tp + tn + eps) / (tp + tn + fp + fn + eps))
    specificity = float((tn + eps) / (tn + fp + eps))
    
    return precision, recall, f1, accuracy, specificity

def _binary_boundary(mask_bin: np.ndarray) -> np.ndarray:
    """Extract binary mask boundary using OpenCV (returns same-size 0/1 array with boundary=1)."""
    mask_u8 = (mask_bin.astype(np.uint8) * 255)
    if mask_u8.max() == 0:
        return np.zeros_like(mask_bin, dtype=np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = np.zeros_like(mask_u8)
    for cnt in contours:
        cv2.drawContours(boundary, [cnt], -1, color=1, thickness=1)
    return boundary.astype(np.uint8)

def hd95_binary_numpy(pred_bin: np.ndarray, tgt_bin: np.ndarray) -> float:
    """
    HD95 (pixel units). 
    If both masks have no foreground => 0.
    If only one has foreground => +inf.
    Implemented with OpenCV distanceTransform (no SciPy dependency).
    """
    pred_has = pred_bin.any()
    tgt_has  = tgt_bin.any()
    if (not pred_has) and (not tgt_has):
        return 0.0
    if (pred_has and (not tgt_has)) or ((not pred_has) and tgt_has):
        return float("inf")

    Pb = _binary_boundary(pred_bin)
    Tb = _binary_boundary(tgt_bin)

    if Pb.max() == 0:
        Pb = pred_bin.astype(np.uint8)
    if Tb.max() == 0:
        Tb = tgt_bin.astype(np.uint8)

    def dist_to_border(border01: np.ndarray) -> np.ndarray:
        inv = np.where(border01 > 0, 0, 1).astype(np.uint8)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
        return dist

    dist_to_T = dist_to_border(Tb)
    dist_to_P = dist_to_border(Pb)

    Py, Px = np.where(Pb > 0)
    Ty, Tx = np.where(Tb > 0)
    if len(Py) == 0 or len(Ty) == 0:
        return float("inf")

    d_PT = dist_to_T[Py, Px]
    d_TP = dist_to_P[Ty, Tx]

    d_all = np.concatenate([d_PT, d_TP], axis=0)
    if d_all.size == 0:
        return float("inf")
    return float(np.percentile(d_all, 95))

# -------------------- Main Test --------------------
@torch.no_grad()
def run_test(model, loader, device, dice_thr=0.5, vis_dir=None, csv_path=None):
    model.eval()
    os.makedirs(vis_dir, exist_ok=True) if vis_dir else None

    rows = []
    dices, ious, hd95s = [], [], []
    precisions, recalls, f1s, accuracies, specificities = [], [], [], [], []
    idx_global = 0

    pbar = tqdm(loader, desc="[Test]")
    for batch in pbar:
        if len(batch) == 3:
            inputs, targets, ids = batch
            case_ids = list(ids)
        else:
            inputs, targets = batch
            case_ids = [f"case_{idx_global + i:05d}" for i in range(inputs.size(0))]

        inputs  = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)  
        probs  = torch.sigmoid(logits)
        preds  = (probs > dice_thr).float()

        B = inputs.size(0)
        for b in range(B):
            gt  = targets[b, 0].detach().cpu().numpy() > 0.5
            pr  = preds[b, 0].detach().cpu().numpy() > 0.5

            dsc  = dice_binary_numpy(pr, gt)
            iou  = iou_binary_numpy(pr, gt)
            hd95 = hd95_binary_numpy(pr, gt)
            prec, rec, f1, acc, spec = precision_recall_f1_numpy(pr, gt)

            dices.append(dsc)
            ious.append(iou)
            hd95s.append(hd95)
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
            accuracies.append(acc)
            specificities.append(spec)
            
            rows.append({
                "id": case_ids[b], 
                "dice": dsc, 
                "iou": iou, 
                "hd95": hd95,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "accuracy": acc,
                "specificity": spec
            })

            if vis_dir is not None:
                save_eval_visuals(idx_global, inputs[b], logits[b], targets[b], vis_dir, thr=dice_thr, fname_prefix="test")
            idx_global += 1

        pbar.set_postfix(
            dice=np.mean(dices) if dices else 0.0,
            iou=np.mean(ious) if ious else 0.0,
            hd95=np.nanmean([x for x in hd95s if np.isfinite(x)]) if hd95s else 0.0
        )

    mean_dice = float(np.mean(dices)) if dices else 0.0
    mean_iou  = float(np.mean(ious))  if ious  else 0.0
    finite_hd = [x for x in hd95s if np.isfinite(x)]
    mean_hd95 = float(np.mean(finite_hd)) if len(finite_hd) > 0 else float("inf")
    mean_precision = float(np.mean(precisions)) if precisions else 0.0
    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    mean_f1 = float(np.mean(f1s)) if f1s else 0.0
    mean_accuracy = float(np.mean(accuracies)) if accuracies else 0.0
    mean_specificity = float(np.mean(specificities)) if specificities else 0.0
    
    # Calculate std for uncertainty estimation
    std_dice = float(np.std(dices)) if dices else 0.0
    std_iou = float(np.std(ious)) if ious else 0.0
    std_precision = float(np.std(precisions)) if precisions else 0.0
    std_recall = float(np.std(recalls)) if recalls else 0.0

    if csv_path is not None:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "dice", "iou", "hd95", "precision", "recall", "f1", "accuracy", "specificity"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
            writer.writerow({
                "id": "MEAN", 
                "dice": mean_dice, 
                "iou": mean_iou, 
                "hd95": mean_hd95,
                "precision": mean_precision,
                "recall": mean_recall,
                "f1": mean_f1,
                "accuracy": mean_accuracy,
                "specificity": mean_specificity
            })
            writer.writerow({
                "id": "STD", 
                "dice": std_dice, 
                "iou": std_iou, 
                "hd95": 0.0,
                "precision": std_precision,
                "recall": std_recall,
                "f1": 0.0,
                "accuracy": 0.0,
                "specificity": 0.0
            })

    print("=" * 80)
    print("[Test Summary - Comprehensive Metrics]")
    print("=" * 80)
    print(f"Segmentation Overlap Metrics:")
    print(f"  Dice Coefficient:    {mean_dice:.4f} +/- {std_dice:.4f}")
    print(f"  IoU (Jaccard):       {mean_iou:.4f} +/- {std_iou:.4f}")
    print(f"  F1 Score:            {mean_f1:.4f}")
    print(f"-" * 80)
    print(f"Classification Metrics:")
    print(f"  Precision (PPV):     {mean_precision:.4f} +/- {std_precision:.4f}")
    print(f"  Recall (Sensitivity):{mean_recall:.4f} +/- {std_recall:.4f}")
    print(f"  Specificity (TNR):   {mean_specificity:.4f}")
    print(f"  Accuracy:            {mean_accuracy:.4f}")
    print(f"-" * 80)
    print(f"Boundary Metrics:")
    print(f"  HD95 (pixels):       {mean_hd95 if math.isfinite(mean_hd95) else 'inf':.2f}")
    print("=" * 80)

    return mean_dice, mean_iou, mean_hd95

def load_ckpt_flex(model, ckpt_path, map_location="cpu"):
    obj = torch.load(ckpt_path, map_location=map_location)
    if isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    else:
        state = obj
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[Warn] Missing keys:", missing)
    if unexpected:
        print("[Warn] Unexpected keys:", unexpected)

def main():
    import argparse
    import os
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/serverI/cornhub_data/CornHub_LITE_1024/CornHub_LITE_1024")
    parser.add_argument("--dataset", type=str, default="", help="Dataset subdirectory name. Leave empty if data_dir points directly to dataset root.")
    parser.add_argument("--mask_ext", type=str, default=".png")
    parser.add_argument("--input_h", type=int, default=1024)
    parser.add_argument("--input_w", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--dice_thr", type=float, default=0.5)

    # Segmentation model checkpoint (DPT + decoder)
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to the trained segmentation model checkpoint (.pth).")

    parser.add_argument("--save_root", type=str, default="./runs")
    parser.add_argument("--img_dir_name", type=str, default="Images")
    parser.add_argument("--label_dir_name", type=str, default="Masks")
    parser.add_argument("--auto_split", action="store_true", help="Enable auto train/test split for flat directory structure")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train ratio when using auto_split")
    parser.add_argument("--seed", type=int, default=42)

    # DINO backbone configuration
    parser.add_argument("--dino_size", type=str, default="s", choices=["b", "s"],
                        help="DINO backbone size: b=ViT-B/16, s=ViT-S/16")
    parser.add_argument("--dino_ckpt", type=str, required=True,
                        help="Path to the pretrained DINO checkpoint (.pth). "
                             "Use ViT-B/16 checkpoint for --dino_size b, or ViT-S/16 for --dino_size s.")
    parser.add_argument("--repo_dir", type=str, default="./dinov3",
                        help="Local path to the DINOv3 torch.hub repo (contains hubconf.py).")

    args = parser.parse_args()

    # Output directories
    dataset_name = args.dataset if args.dataset else os.path.basename(args.data_dir.rstrip('/'))
    save_root = os.path.join(args.save_root, f"segdino_{args.dino_size}_{args.input_h}_{dataset_name}")
    vis_dir   = os.path.join(save_root, "test_vis")
    csv_path  = os.path.join(save_root, "test_metrics.csv")
    os.makedirs(save_root, exist_ok=True)

    # Load DINO backbone with support for both .pth and .safetensors formats
    if args.dino_ckpt.endswith('.safetensors'):
        # Load safetensors format
        from safetensors.torch import load_file
        if args.dino_size == "b":
            backbone = torch.hub.load(args.repo_dir, 'dinov3_vitb16', source='local', pretrained=False)
        else:
            backbone = torch.hub.load(args.repo_dir, 'dinov3_vits16', source='local', pretrained=False)
        # Load weights from safetensors
        state_dict = load_file(args.dino_ckpt)
        backbone.load_state_dict(state_dict, strict=False)
        print(f"[Load DINO checkpoint] Loaded from safetensors: {args.dino_ckpt}")
    else:
        # Load standard .pth format
        if args.dino_size == "b":
            backbone = torch.hub.load(args.repo_dir, 'dinov3_vitb16', source='local', weights=args.dino_ckpt)
        else:
            backbone = torch.hub.load(args.repo_dir, 'dinov3_vits16', source='local', weights=args.dino_ckpt)
        print(f"[Load DINO checkpoint] Loaded from .pth: {args.dino_ckpt}")

    from dpt import DPT
    model = DPT(nclass=1, backbone=backbone)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Load segmentation checkpoint
    print(f"[Load segmentation ckpt] {args.ckpt}")
    load_ckpt_flex(model, args.ckpt, map_location=device)

    # Dataset
    from dataset import FolderDataset, ResizeAndNormalize
    if args.dataset:
        root = os.path.join(args.data_dir, args.dataset)
    else:
        root = args.data_dir
    test_transform = ResizeAndNormalize(size=(args.input_h, args.input_w))
    test_dataset = FolderDataset(
        root=root,
        split="test",
        img_dir_name=args.img_dir_name,
        label_dir_name=args.label_dir_name,
        mask_ext=args.mask_ext if hasattr(args, "mask_ext") else None,
        transform=test_transform,
        auto_split=args.auto_split,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    # Run evaluation
    run_test(model, test_loader, device, dice_thr=args.dice_thr, vis_dir=vis_dir, csv_path=csv_path)


if __name__ == "__main__":
    main()
