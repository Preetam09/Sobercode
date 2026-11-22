"""
HW4 - Problem 5: Optimizing AUROC / AUPRC on PneumoniaMNIST with LibAUC.

Interfaces used: PyTorch, LibAUC 1.4.0 (APLoss+SOAP, AUCMLoss+PESG, ResNet18,
DualSampler, auc_roc_score / auc_prc_score), MedMNIST 3.x, matplotlib.

Usage
    python hw5_libauc.py tune          # runs the full hyper-parameter grid (15 runs)
    python hw5_libauc.py plot          # selects best configs on VAL, makes plots + table

Model selection: every configuration is trained on the train split; the
hyper-parameter is chosen by the final-epoch score on the official
PneumoniaMNIST *validation* split (AUPRC for APLoss, AUROC for AUCMLoss and CE).
The test split is never used for selection.
"""
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from medmnist import PneumoniaMNIST

from libauc.losses import APLoss, AUCMLoss, CrossEntropyLoss
from libauc.metrics import auc_prc_score, auc_roc_score
from libauc.models import resnet18 as ResNet18
from libauc.optimizers import PESG, SOAP
from libauc.sampler import DualSampler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RUN_DIR = os.path.join(HERE, "runs")
FIG_DIR = os.path.join(HERE, "figures")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# ---------------- hyper-parameters (from the HW statement / codebase) ----------------
SEED = 123
TOTAL_EPOCHS = 15
DECAY_EPOCHS = [9]          # lr /= 10 at epoch 9 (0-indexed loop, as in the codebase)
BATCH_SIZE = 64
WEIGHT_DECAY = 0.0
IMAGE_SIZE = 32             # codebase: resize to 32x32
CROP_SIZE = 26              # codebase crops 30 out of 32 (size-2); PneumoniaMNIST is 28x28 -> 26
SAMPLING_RATE = 0.5         # codebase: DualSampler sampling rate
AP_MARGIN = 0.6             # codebase (tutorial 3) margin for APLoss
AUCM_MARGIN = 1.0           # codebase (tutorial 2) margin for AUCMLoss
EPOCH_DECAY = 0.003         # codebase (tutorial 2) epoch_decay for PESG

GRID = {
    "ap_soap": [dict(lr=lr, gamma=g) for lr in (5e-4, 1e-3, 2e-3) for g in (0.2, 0.5, 0.8)],
    "aucm_pesg": [dict(lr=lr) for lr in (0.02, 0.05, 0.1)],
    "ce_adam": [dict(lr=lr) for lr in (5e-4, 1e-3, 2e-3)],
}
SELECT_METRIC = {"ap_soap": "auprc", "aucm_pesg": "auroc", "ce_adam": "auroc"}
NAMES = {"ap_soap": "APLoss + SOAP (Adam)", "aucm_pesg": "AUCMLoss + PESG (Adam)",
         "ce_adam": "CE + Adam"}


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------- (a) data loading + augmentation ----------------
class IndexedPneumoniaMNIST(PneumoniaMNIST):
    """PneumoniaMNIST whose __getitem__ returns (index, img, target).

    This is the "API mismatch" fix from Note 6 of the HW, done via a subclass
    instead of editing medmnist/dataset.py in site-packages. LibAUC's APLoss
    needs the sample index to update its per-sample moving averages.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.targets = self.labels.squeeze().astype(np.int64)  # used by DualSampler

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return index, img, torch.tensor(target, dtype=torch.float32)


train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop((CROP_SIZE, CROP_SIZE), padding=None),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
])
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
])


def get_datasets():
    os.makedirs(DATA_DIR, exist_ok=True)
    kw = dict(download=True, root=DATA_DIR, size=28)
    train_set = IndexedPneumoniaMNIST(split="train", transform=train_transform, **kw)
    train_eval_set = IndexedPneumoniaMNIST(split="train", transform=eval_transform, **kw)
    val_set = IndexedPneumoniaMNIST(split="val", transform=eval_transform, **kw)
    test_set = IndexedPneumoniaMNIST(split="test", transform=eval_transform, **kw)
    return train_set, train_eval_set, val_set, test_set


def build_model():
    model = ResNet18(pretrained=False, last_activation=None, num_classes=1)
    # Note 4: grayscale input -> first conv takes 1 channel instead of 3
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model.to(DEVICE)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    for _, data, targets in loader:
        y_prob = torch.sigmoid(model(data.to(DEVICE)))
        preds.append(y_prob.cpu().numpy())
        trues.append(targets.numpy())
    model.train()
    y_true = np.concatenate(trues).reshape(-1)
    y_pred = np.concatenate(preds).reshape(-1)
    return float(auc_roc_score(y_true, y_pred)), float(auc_prc_score(y_true, y_pred))


# ---------------- (b) training ----------------
def train_one(method, lr, gamma=None, datasets=None, verbose=True):
    set_all_seeds(SEED)
    train_set, train_eval_set, val_set, test_set = datasets or get_datasets()

    if method in ("ap_soap", "aucm_pesg"):
        # codebase: LibAUC DualSampler controls #pos / #neg in every mini-batch
        sampler = DualSampler(train_set, BATCH_SIZE, labels=train_set.targets,
                              sampling_rate=SAMPLING_RATE, random_seed=SEED)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    eval_loaders = {
        "train": torch.utils.data.DataLoader(train_eval_set, batch_size=256, shuffle=False),
        "val": torch.utils.data.DataLoader(val_set, batch_size=256, shuffle=False),
        "test": torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False),
    }

    model = build_model()
    if method == "ap_soap":
        loss_fn = APLoss(data_len=len(train_set), margin=AP_MARGIN, gamma=gamma, device=DEVICE)
        optimizer = SOAP(model.parameters(), lr=lr, mode="adam", weight_decay=WEIGHT_DECAY,
                         device=DEVICE, verbose=False)
    elif method == "aucm_pesg":
        loss_fn = AUCMLoss(margin=AUCM_MARGIN, device=DEVICE)
        optimizer = PESG(model.parameters(), loss_fn=loss_fn, lr=lr, mode="adam",
                         margin=AUCM_MARGIN, epoch_decay=EPOCH_DECAY,
                         weight_decay=WEIGHT_DECAY, device=DEVICE, verbose=False)
    elif method == "ce_adam":
        loss_fn = CrossEntropyLoss()  # sigmoid + binary cross-entropy (LibAUC)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    else:
        raise ValueError(method)

    hist = {k: [] for k in ("loss", "train_auroc", "train_auprc", "val_auroc",
                            "val_auprc", "test_auroc", "test_auprc")}
    for epoch in range(TOTAL_EPOCHS):
        if epoch in DECAY_EPOCHS:
            if method == "aucm_pesg":
                optimizer.update_regularizer(decay_factor=10)  # lr/10 + reset reference point
            elif method == "ap_soap":
                optimizer.update_lr(decay_factor=10)
            else:
                for g in optimizer.param_groups:
                    g["lr"] /= 10

        t0 = time.time()
        losses = []
        model.train()
        for index, data, targets in train_loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            logits = model(data)
            if method == "ap_soap":
                loss = loss_fn(torch.sigmoid(logits), targets, index.to(DEVICE))
            elif method == "aucm_pesg":
                loss = loss_fn(torch.sigmoid(logits), targets)
            else:
                loss = loss_fn(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        hist["loss"].append(float(np.mean(losses)))
        for split, loader in eval_loaders.items():
            roc, prc = evaluate(model, loader)
            hist[f"{split}_auroc"].append(roc)
            hist[f"{split}_auprc"].append(prc)
        if verbose:
            print(f"[{method} lr={lr} gamma={gamma}] ep {epoch:2d} loss {hist['loss'][-1]:.4f} | "
                  f"AUROC tr {hist['train_auroc'][-1]:.4f} va {hist['val_auroc'][-1]:.4f} "
                  f"te {hist['test_auroc'][-1]:.4f} | AUPRC tr {hist['train_auprc'][-1]:.4f} "
                  f"va {hist['val_auprc'][-1]:.4f} te {hist['test_auprc'][-1]:.4f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
    return hist


def run_tag(method, cfg):
    return method + "_" + "_".join(f"{k}{v}" for k, v in cfg.items())


def tune():
    os.makedirs(RUN_DIR, exist_ok=True)
    datasets = get_datasets()
    tr = datasets[0].targets
    print(f"device={DEVICE}  train={len(tr)} (pos={tr.sum()}, neg={len(tr) - tr.sum()})  "
          f"val={len(datasets[2])}  test={len(datasets[3])}", flush=True)
    for method, cfgs in GRID.items():
        for cfg in cfgs:
            path = os.path.join(RUN_DIR, run_tag(method, cfg) + ".json")
            if os.path.exists(path):
                continue
            hist = train_one(method, datasets=datasets, **cfg)
            with open(path, "w") as f:
                json.dump({"method": method, "cfg": cfg, "hist": hist}, f)


# ---------------- (c) model selection, curves, table ----------------
def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    runs = []
    for fn in sorted(os.listdir(RUN_DIR)):
        with open(os.path.join(RUN_DIR, fn)) as f:
            runs.append(json.load(f))

    lines = ["# Problem 5 results (seed %d, %d epochs)\n" % (SEED, TOTAL_EPOCHS),
             "## Hyper-parameter search (final-epoch scores)\n",
             "| method | config | val AUROC | val AUPRC | test AUROC | test AUPRC |",
             "|---|---|---|---|---|---|"]
    best = {}
    for method in GRID:
        mr = [r for r in runs if r["method"] == method]
        m = SELECT_METRIC[method]
        for r in mr:
            h = r["hist"]
            lines.append(f"| {NAMES[method]} | {r['cfg']} | {h['val_auroc'][-1]:.4f} | "
                         f"{h['val_auprc'][-1]:.4f} | {h['test_auroc'][-1]:.4f} | {h['test_auprc'][-1]:.4f} |")
        best[method] = max(mr, key=lambda r: r["hist"][f"val_{m}"][-1])

    lines += ["", "## Selected configurations (chosen on validation set)\n",
              "| method | selected by | config | train AUROC | train AUPRC | test AUROC | test AUPRC | best test AUROC (ep) | best test AUPRC (ep) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for method, r in best.items():
        h = r["hist"]
        bro, bpr = int(np.argmax(h["test_auroc"])), int(np.argmax(h["test_auprc"]))
        lines.append(f"| {NAMES[method]} | val {SELECT_METRIC[method].upper()} | {r['cfg']} | "
                     f"{h['train_auroc'][-1]:.4f} | {h['train_auprc'][-1]:.4f} | "
                     f"{h['test_auroc'][-1]:.4f} | {h['test_auprc'][-1]:.4f} | "
                     f"{h['test_auroc'][bro]:.4f} ({bro}) | {h['test_auprc'][bpr]:.4f} ({bpr}) |")
    with open(os.path.join(HERE, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    colors = {"ap_soap": "#1f77b4", "aucm_pesg": "#d62728", "ce_adam": "#2ca02c"}
    epochs = np.arange(1, TOTAL_EPOCHS + 1)

    # One figure with the 12 curves: 2 panels (AUROC, AUPRC) x 3 models x {train, test}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, metric in zip(axes, ("auroc", "auprc")):
        for method, r in best.items():
            h, c = r["hist"], colors[method]
            lab = f"{NAMES[method]} {r['cfg']}"
            ax.plot(epochs, h[f"train_{metric}"], "-", color=c, lw=2.2, label=f"{lab} - train")
            ax.plot(epochs, h[f"test_{metric}"], "--", color=c, lw=2.2, label=f"{lab} - test")
        ax.axvline(DECAY_EPOCHS[0] + 0.5, color="gray", ls=":", lw=1)
        ax.set_title(metric.upper().replace("AU", "AU-"), fontsize=15)
        ax.set_xlabel("epoch", fontsize=13)
        ax.set_ylabel(metric.upper(), fontsize=13)
        ax.set_xticks(epochs)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8.5, loc="lower right")
    fig.suptitle("PneumoniaMNIST - ResNet18, train (solid) vs test (dashed); lr decay after epoch 9",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "curves_all.png"), dpi=160)

    # Per-model figures (easier to read in the report)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for j, (method, r) in enumerate(best.items()):
        h = r["hist"]
        for i, metric in enumerate(("auroc", "auprc")):
            ax = axes[i, j]
            ax.plot(epochs, h[f"train_{metric}"], "-o", ms=3, color=colors[method], label="train")
            ax.plot(epochs, h[f"test_{metric}"], "--s", ms=3, color="black", label="test")
            ax.axvline(DECAY_EPOCHS[0] + 0.5, color="gray", ls=":", lw=1)
            ax.set_title(f"{NAMES[method]} {r['cfg']}\n{metric.upper()}", fontsize=10.5)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)
            if i == 1:
                ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "curves_per_model.png"), dpi=160)
    print("figures saved to", FIG_DIR)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tune"
    if cmd == "tune":
        tune()
    elif cmd == "plot":
        plot()
    elif cmd == "smoke":  # quick sanity check: 1 epoch of each method
        TOTAL_EPOCHS = 1
        ds = get_datasets()
        for m in GRID:
            train_one(m, datasets=ds, **GRID[m][0])
    else:
        raise SystemExit(__doc__)
