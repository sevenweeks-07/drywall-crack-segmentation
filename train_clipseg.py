"""
train_clipseg.py
────────────────────────────────────────────────────────────────────────────
Fine-tune CLIPSeg (CIDAS/clipseg-rd64-refined) for text-conditioned binary
segmentation on two prompts:
  • "segment taping area"  → drywall / joint-tape dataset
  • "segment crack"        → wall-crack dataset
 
Seeds   : SEED = 42  (Python / NumPy / PyTorch / CUDA all fixed)
Hardware: tested on 12 GB GPU (≈4 GB peak VRAM usage)
 
Install:
  pip install transformers torch torchvision Pillow opencv-python-headless
  pip install accelerate scikit-learn matplotlib tqdm
"""
 
# ─── stdlib ──────────────────────────────────────────────────────────────────
import os, json, random, time
from pathlib import Path
 
# ─── third-party ─────────────────────────────────────────────────────────────
import cv2
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
 
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from tqdm import tqdm
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 1.  SEEDS 
# ══════════════════════════════════════════════════════════════════════════════
SEED = 42
random.seed(SEED)                           # Python built-in RNG
np.random.seed(SEED)                        # NumPy RNG
torch.manual_seed(SEED)                     # PyTorch CPU RNG
torch.cuda.manual_seed_all(SEED)            # All CUDA device RNGs
torch.backends.cudnn.deterministic = True   # Reproducible CUDNN kernels
torch.backends.cudnn.benchmark     = False  # No auto-tuner (would break repro)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 2.  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID     = "CIDAS/clipseg-rd64-refined"   # Pre-trained CLIPSeg checkpoint
MASK_SIZE    = 352      # GT masks resized to this; CLIPSeg outputs 352×352
BATCH_SIZE   = 8        # Safe for 12 GB GPU at 352 px
EPOCHS       = 15       # Max epochs (early stopping will usually fire earlier)
LR           = 5e-5     # AdamW learning rate
WEIGHT_DECAY = 1e-4     # L2 regularisation
PATIENCE     = 5        # Early-stop: epochs without val-loss improvement
NUM_WORKERS  = 4        # DataLoader worker processes
 
BASE    = Path("/root/origin/automated-drywall-inspection")
OUT_DIR = Path("outputs")
PRED_DIR   = OUT_DIR / "predictions"   # Binary mask PNGs written here
VISUAL_DIR = OUT_DIR / "visuals"       # orig|GT|pred panels
for d in [OUT_DIR, PRED_DIR, VISUAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)
 
CKPT = OUT_DIR / "clipseg_finetuned.pth"   # Best-model checkpoint path
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 3.  PROMPT POOLS 
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_POOLS = {
    "taping": [
        "segment taping area",
        "segment joint tape",
        "segment drywall seam",
        "highlight drywall joint",
        "segment wall tape",
    ],
    "crack": [
        "segment crack",
        "segment wall crack",
        "highlight crack",
        "find crack in wall",
        "segment surface crack",
    ],
}
 
# Canonical prompts used for evaluation filenames and the final report
CANONICAL = {
    "taping": "segment taping area",
    "crack":  "segment crack",
}
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 4.  DATASET
# ══════════════════════════════════════════════════════════════════════════════
class SegDataset(Dataset):
    """
    Pairs source images with their ground-truth binary masks.
 
    Mask naming conventions handled automatically:
      • {stem}__segment_taping_area.png   
      • {stem}_mask.png                  
 
    At __getitem__ a prompt is sampled randomly from prompt_pool so the model
    sees all phrasings during training.
    """
 
    def __init__(self, image_dir, mask_dir, prompt_pool, processor):
        self.image_dir   = Path(image_dir)   # Folder containing source images
        self.mask_dir    = Path(mask_dir)    # Folder containing GT mask PNGs
        self.prompt_pool = prompt_pool       # List[str] – prompts for this class
        self.processor   = processor        # CLIPSegProcessor instance
 
        # ── Collect images ────────────────────────────────────────────────────
        valid = {".jpg", ".jpeg", ".png"}
        all_imgs = sorted(
            p for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid
        )
 
        # ── Match each image to exactly one mask ──────────────────────────────
        self.pairs = []
        for img in all_imgs:
            stem = img.stem
 
            # Priority 1: drywall SAM masks  e.g.  abc__segment_taping_area.png
            hits = list(self.mask_dir.glob(f"{stem}__*.png"))
 
            # Priority 2: crack polygon masks  e.g.  abc_mask.png
            if not hits:
                candidate = self.mask_dir / f"{stem}_mask.png"
                if candidate.exists():
                    hits = [candidate]
 
            if hits:
                self.pairs.append((img, hits[0]))  # Keep first match
 
        print(f"  {image_dir} → {len(self.pairs)} pairs")
 
    # ── Length ────────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.pairs)
 
    # ── Item ──────────────────────────────────────────────────────────────────
    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
 
        # Load image (BGR → RGB → PIL for HuggingFace processor)
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
 
        # Load mask, resize to MASK_SIZE, binarise
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask_gray = cv2.resize(
            mask_gray,
            (MASK_SIZE, MASK_SIZE),
            interpolation=cv2.INTER_NEAREST   # nearest-neighbour preserves {0,255}
        )
        mask_bin = (mask_gray > 127).astype(np.float32)   # → {0.0, 1.0}
 
        # Sample random prompt for text augmentation
        prompt = random.choice(self.prompt_pool)
 
        # CLIPSegProcessor: tokenises text + normalises image in one call
        enc = self.processor(
            text=[prompt],          # must be a list
            images=img_pil,
            return_tensors="pt",    # PyTorch tensors
            padding=True,
        )
 
        # Squeeze the batch-of-1 dimension the processor adds
        pixel_values   = enc["pixel_values"].squeeze(0)   # (3, H, W)
        input_ids      = enc["input_ids"].squeeze(0)      # (seq_len,)
        attention_mask = enc.get(
            "attention_mask",
            torch.ones_like(enc["input_ids"])
        ).squeeze(0)                                       # (seq_len,)
 
        return {
            "pixel_values":   pixel_values,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "mask":           torch.tensor(mask_bin),     # (MASK_SIZE, MASK_SIZE)
            "img_path":       str(img_path),              # kept for prediction saving
            "prompt":         prompt,
        }
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 5.  LOSS  –  BCE + Dice
# ══════════════════════════════════════════════════════════════════════════════
def bce_dice_loss(logits, target, w_bce=0.5, w_dice=0.5):
    """
    logits : (B, H', W') raw model output  (H' may differ from MASK_SIZE)
    target : (B, MASK_SIZE, MASK_SIZE) float32 binary ground truth
 
    BCE handles per-pixel classification; Dice handles class imbalance.
    Both are numerically stable implementations.
    """
    # Upsample logits to match target spatial size
    logits = F.interpolate(
        logits.unsqueeze(1),          # (B,1,H',W')
        size=target.shape[-2:],       # → (B,1,MASK_SIZE,MASK_SIZE)
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)                      # → (B,MASK_SIZE,MASK_SIZE)
 
    # Binary Cross-Entropy (uses log-sum-exp trick internally → stable)
    bce = F.binary_cross_entropy_with_logits(logits, target)
 
    # Dice loss
    prob   = torch.sigmoid(logits)                          # probabilities
    smooth = 1e-6                                           # avoid div/0
    inter  = (prob * target).sum(dim=(1, 2))               # true positives
    dice   = 1.0 - (2.0 * inter + smooth) / (
        prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + smooth
    )
 
    return w_bce * bce + w_dice * dice.mean()

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Custom collate that pads input_ids / attention_mask to the longest
    sequence in the batch.  pixel_values and mask are already fixed-size.
    
    Example: if batch has prompts tokenized to lengths [5, 7, 6],
    all are padded to length 7 with pad_token_id (0) and mask 0.
    """
    pixel_values   = torch.stack([b["pixel_values"]   for b in batch])
    masks          = torch.stack([b["mask"]           for b in batch])
    img_paths      = [b["img_path"] for b in batch]
    prompts        = [b["prompt"]   for b in batch]

    # pad_sequence stacks a list of (seq_len,) tensors → (max_len, B),
    # then we transpose to (B, max_len).  batch_first=True does that directly.
    input_ids = pad_sequence(
        [b["input_ids"] for b in batch],
        batch_first=True,
        padding_value=0,          # 0 is the standard pad token id
    )
    attention_mask = pad_sequence(
        [b["attention_mask"] for b in batch],
        batch_first=True,
        padding_value=0,          # 0 = ignore this position
    )

    return {
        "pixel_values":   pixel_values,
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "mask":           masks,
        "img_path":       img_paths,
        "prompt":         prompts,
    }

# ══════════════════════════════════════════════════════════════════════════════
# 6.  METRICS  –  mIoU & Dice (per-class + overall)
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(preds_bin, targets_bin):
    """
    preds_bin   : np.ndarray of shape (N, H, W) bool/uint8 0-or-1
    targets_bin : np.ndarray of shape (N, H, W) bool/uint8 0-or-1
 
    Returns dict with miou and dice (macro-averaged over samples).
    Handles edge case where both pred and GT are all-zero (empty mask).
    """
    ious, dices = [], []
    for p, t in zip(preds_bin, targets_bin):
        inter     = np.logical_and(p, t).sum()       # intersection
        union     = np.logical_or(p, t).sum()        # union
        iou       = inter / (union + 1e-6)           # IoU
        dice      = 2 * inter / (p.sum() + t.sum() + 1e-6)  # Dice
        ious.append(iou)
        dices.append(dice)
    return {
        "miou": float(np.mean(ious)),
        "dice": float(np.mean(dices)),
        "n":    len(ious),
    }
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 7.  TRAIN ONE EPOCH
# ══════════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer):
    """Run one full training epoch. Returns mean loss."""
    model.train()                    # Enable dropout / batchnorm train mode
    losses = []
 
    for batch in tqdm(loader, desc="  train", leave=False):
        pv  = batch["pixel_values"].to(DEVICE)     # preprocessed images
        ids = batch["input_ids"].to(DEVICE)         # tokenised prompts
        am  = batch["attention_mask"].to(DEVICE)    # attention masks
        gt  = batch["mask"].to(DEVICE)              # ground-truth binary masks
 
        out    = model(pixel_values=pv, input_ids=ids, attention_mask=am)
        loss   = bce_dice_loss(out.logits, gt)
        loss   = bce_dice_loss(out.logits, gt)
 
        optimizer.zero_grad()                           # clear stale gradients
        loss.backward()                                 # compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip for stability
        optimizer.step()                                # update weights
 
        losses.append(loss.item())
 
    return float(np.mean(losses))
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 8.  VALIDATE ONE EPOCH
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()                     # no gradient tracking → faster, less VRAM
def val_epoch(model, loader):
    """Run one validation pass. Returns mean loss."""
    model.eval()
    losses = []
    for batch in tqdm(loader, desc="  val  ", leave=False):
        pv  = batch["pixel_values"].to(DEVICE)
        ids = batch["input_ids"].to(DEVICE)
        am  = batch["attention_mask"].to(DEVICE)
        gt  = batch["mask"].to(DEVICE)
        out = model(pixel_values=pv, input_ids=ids, attention_mask=am)
        losses.append(bce_dice_loss(out.logits, gt).item())
    return float(np.mean(losses))
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 9.  FULL EVALUATION  –  collect predictions + compute metrics
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def full_eval(model, processor, configs):
    """
    configs: list of dicts  { image_dir, mask_dir, prompt_key, split }
    Runs inference with the canonical prompt for each class.
    Returns results dict and a list of (img_np, gt_np, pred_np, label) tuples
    for visual generation.
    """
    model.eval()
    results  = {}       # filled per-class
    visuals  = []       # (orig_rgb, gt, pred, label_str) for plotting
 
    for cfg in configs:
        key        = cfg["prompt_key"]          # "taping" or "crack"
        prompt     = CANONICAL[key]             # canonical string
        image_dir  = Path(cfg["image_dir"])
        mask_dir   = Path(cfg["mask_dir"])
 
        valid = {".jpg", ".jpeg", ".png"}
        img_paths = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid
        )
 
        all_preds, all_gts = [], []
 
        for img_path in tqdm(img_paths, desc=f"  eval {key}/{cfg['split']}", leave=False):
            stem = img_path.stem
 
            # ── Find GT mask ─────────────────────────────────────────────────
            hits = list(mask_dir.glob(f"{stem}__*.png"))
            if not hits:
                candidate = mask_dir / f"{stem}_mask.png"
                hits = [candidate] if candidate.exists() else []
            if not hits:
                continue             # No mask found – skip
 
            # ── Load original image ───────────────────────────────────────────
            img_bgr = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            orig_h, orig_w = img_bgr.shape[:2]   # save for output resize
 
            # ── Load GT mask ──────────────────────────────────────────────────
            gt_gray = cv2.imread(str(hits[0]), cv2.IMREAD_GRAYSCALE)
            gt_bin  = (gt_gray > 127).astype(np.uint8)   # {0,1}
 
            # ── Model inference ───────────────────────────────────────────────
            enc = processor(
                text=[prompt],
                images=img_pil,
                return_tensors="pt",
                padding=True,
            )
            pv  = enc["pixel_values"].to(DEVICE)
            ids = enc["input_ids"].to(DEVICE)
            am  = enc.get("attention_mask", torch.ones_like(enc["input_ids"])).to(DEVICE)
 
            t0  = time.perf_counter()
            out = model(pixel_values=pv, input_ids=ids, attention_mask=am)
            inf_ms = (time.perf_counter() - t0) * 1000   # inference time in ms
 
            logits = out.logits.squeeze().cpu().numpy()   # (H', W')
 
            # ── Resize prediction to original image size ──────────────────────
            prob    = 1.0 / (1.0 + np.exp(-logits))       # sigmoid
            pred_rs = cv2.resize(
                prob, (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR
            )
            pred_bin = (pred_rs > 0.5).astype(np.uint8)   # threshold → {0,1}
 
            all_preds.append(pred_bin)
            all_gts.append(gt_bin)
 
            # ── Save prediction mask to disk ──────────────────────────────────
            # Filename: {stem}__{canonical_prompt_underscored}.png
            prompt_slug = prompt.replace(" ", "_")
            out_name = PRED_DIR / f"{stem}__{prompt_slug}.png"
            cv2.imwrite(str(out_name), (pred_bin * 255).astype(np.uint8))
 
            # ── Collect up to 2 visuals per class ────────────────────────────
            if len([v for v in visuals if v[3].startswith(key)]) < 2:
                visuals.append((img_rgb, gt_bin, pred_bin, f"{key}/{stem}"))
 
        # ── Per-class metrics ─────────────────────────────────────────────────
        metrics = compute_metrics(all_preds, all_gts)
        results[f"{key}/{cfg['split']}"] = metrics
        print(f"  [{key} | {cfg['split']}]  "
              f"mIoU={metrics['miou']:.4f}  Dice={metrics['dice']:.4f}  "
              f"n={metrics['n']}")
 
    return results, visuals
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 10.  VISUAL PANELS  –  orig | GT | pred
# ══════════════════════════════════════════════════════════════════════════════
def save_visuals(visuals):
    """Save 4 side-by-side panels: original image | ground truth | prediction."""
    n = min(4, len(visuals))   # at most 4 panels
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]          # normalise to 2-D list
 
    for i, (orig, gt, pred, label) in enumerate(visuals[:n]):
        axes[i][0].imshow(orig);                    axes[i][0].set_title("Original")
        axes[i][1].imshow(gt,   cmap="gray", vmin=0, vmax=1); axes[i][1].set_title("Ground Truth")
        axes[i][2].imshow(pred, cmap="gray", vmin=0, vmax=1); axes[i][2].set_title("Prediction")
        for ax in axes[i]:
            ax.set_xlabel(label, fontsize=7)
            ax.axis("off")
 
    plt.tight_layout()
    plt.savefig(VISUAL_DIR / "examples.png", dpi=150)
    plt.close()
    print(f"  Visual panels saved → {VISUAL_DIR / 'examples.png'}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 11.  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"Device : {DEVICE}")
    print(f"Seed   : {SEED}\n")
 
    # ── Load processor + model ────────────────────────────────────────────────
    print("Loading CLIPSeg …")
    processor = CLIPSegProcessor.from_pretrained(MODEL_ID)
    model     = CLIPSegForImageSegmentation.from_pretrained(MODEL_ID)
    model.to(DEVICE)
 
    # ── Build datasets ────────────────────────────────────────────────────────
    print("\nBuilding datasets …")
    drywall_train = SegDataset(
        BASE/"drywall"/"train"/"images",
        BASE/"drywall"/"train"/"masks",
        PROMPT_POOLS["taping"], processor,
    )
    drywall_val = SegDataset(
        BASE/"drywall"/"valid"/"images",
        BASE/"drywall"/"valid"/"masks",
        PROMPT_POOLS["taping"], processor,
    )
    crack_train = SegDataset(
        BASE/"cracks"/"train"/"images",
        BASE/"cracks"/"train"/"masks",
        PROMPT_POOLS["crack"], processor,
    )
    crack_val = SegDataset(
        BASE/"cracks"/"valid"/"images",
        BASE/"cracks"/"valid"/"masks",
        PROMPT_POOLS["crack"], processor,
    )
 
    # Combine both classes into joint train / val loaders
    train_ds = ConcatDataset([drywall_train, crack_train])
    val_ds   = ConcatDataset([drywall_val,   crack_val])
 
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,collate_fn=collate_fn,
    )
 
    print(f"\nTrain: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")
 
    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )   # halve LR if val loss stalls for 3 epochs
 
    # ── Training loop ─────────────────────────────────────────────────────────
    print("\n──── Training ────")
    best_val, patience_ctr = float("inf"), 0
    train_hist, val_hist   = [], []
    train_start = time.perf_counter()
 
    for epoch in range(1, EPOCHS + 1):
        tl = train_epoch(model, train_loader, optimizer)   # one train pass
        vl = val_epoch(model, val_loader)                  # one val pass
        scheduler.step(vl)                                 # LR schedule
 
        train_hist.append(tl)
        val_hist.append(vl)
        print(f"Epoch {epoch:02d}/{EPOCHS}  train={tl:.4f}  val={vl:.4f}")
 
        # Save best checkpoint / early stopping
        if vl < best_val:
            best_val = vl
            patience_ctr = 0
            torch.save(model.state_dict(), CKPT)           # save weights
            print(f"  ✓ checkpoint saved (val={best_val:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stop at epoch {epoch}.")
                break
 
    train_time_min = (time.perf_counter() - train_start) / 60.0
    print(f"\nTraining time: {train_time_min:.1f} min")
 
    # ── Plot loss curve ───────────────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(train_hist, label="Train")
    plt.plot(val_hist,   label="Val")
    plt.xlabel("Epoch"); plt.ylabel("BCE+Dice Loss")
    plt.title("CLIPSeg Fine-tuning"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / "loss_curve.png", dpi=150)
    plt.close()
 
    # ── Load best checkpoint for evaluation ───────────────────────────────────
    print("\nLoading best checkpoint for evaluation …")
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
 
    # ── Full evaluation ───────────────────────────────────────────────────────
    print("\n──── Evaluation ────")
    eval_configs = [
        {"image_dir": BASE/"drywall"/"valid"/"images",
         "mask_dir":  BASE/"drywall"/"valid"/"masks",
         "prompt_key": "taping", "split": "val"},
        {"image_dir": BASE/"cracks"/"valid"/"images",
         "mask_dir":  BASE/"cracks"/"valid"/"masks",
         "prompt_key": "crack",  "split": "val"},
    ]
    results, visuals = full_eval(model, processor, eval_configs)
 
    # ── Overall metrics ───────────────────────────────────────────────────────
    all_mious = [v["miou"] for v in results.values()]
    all_dices = [v["dice"] for v in results.values()]
    results["overall"] = {
        "miou": float(np.mean(all_mious)),
        "dice": float(np.mean(all_dices)),
    }
 
    # ── Save metrics JSON ─────────────────────────────────────────────────────
    results["meta"] = {
        "seed":            SEED,
        "model":           MODEL_ID,
        "epochs_run":      len(train_hist),
        "best_val_loss":   round(best_val, 6),
        "train_time_min":  round(train_time_min, 2),
        "batch_size":      BATCH_SIZE,
        "lr":              LR,
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved → {OUT_DIR / 'metrics.json'}")
    print(f"Overall  mIoU={results['overall']['miou']:.4f}  "
          f"Dice={results['overall']['dice']:.4f}")
 
    # ── Save visual panels ────────────────────────────────────────────────────
    save_visuals(visuals)
    print(f"\n✓ Done.  Predictions → {PRED_DIR}")
 
 
if __name__ == "__main__":
    main()