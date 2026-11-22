"""
HW4 - Problem 4 (d)/(e): pick the best lr per model on the validation set,
plot training / test loss curves and write results.md.

Usage:  python report.py      (after run_all.sh has finished)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(HERE, "runs")
FIG_DIR = os.path.join(HERE, "figures")
RESULTS = os.path.join(HERE, "results.md")
VARIANTS = ["resnet18", "no_residual", "no_batchnorm"]
NAMES = {"resnet18": "(i) ResNet18", "no_residual": "(ii) ResNet18 w/o residual",
         "no_batchnorm": "(iii) ResNet18 w/o BatchNorm"}
LRS = [0.003, 0.01, 0.03]


def load(variant, lr):
    path = os.path.join(RUN_DIR, "{}_lr{}".format(variant, lr), "history.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def final(h, key):
    """Last logged value, or chance accuracy / NaN if the run diverged."""
    if h["diverged"] or not h[key]:
        return 0.1 if key.endswith("acc") else float("nan")
    return h[key][-1]


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    runs = {(v, lr): load(v, lr) for v in VARIANTS for lr in LRS}

    lines = ["# Problem 4 results (CIFAR-10, 100 epochs, Adam, batch 128, StepLR /1.5 every 10 epochs)\n",
             "## (d) Learning-rate tuning (trained on 45k, selected on the 5k validation split)\n",
             "| model | lr | epochs run | final train loss | val loss | val acc | test acc |",
             "|---|---|---|---|---|---|---|"]
    best = {}
    for v in VARIANTS:
        for lr in LRS:
            h = runs[(v, lr)]
            if h is None:
                lines.append("| {} | {} | not run | | | | |".format(NAMES[v], lr))
                continue
            note = " (diverged)" if h["diverged"] else ""
            lines.append("| {} | {} | {}{} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                NAMES[v], lr, len(h["train_loss"]), note, h["train_loss"][-1],
                final(h, "valid_loss"), final(h, "valid_acc"), final(h, "test_acc")))
        done = [(lr, runs[(v, lr)]) for lr in LRS if runs[(v, lr)] is not None]
        if done:
            best[v] = max(done, key=lambda t: final(t[1], "valid_acc"))

    lines += ["", "## (e) Best learning rate per model\n",
              "| model | best lr | val acc | final test loss | **final test accuracy** |",
              "|---|---|---|---|---|"]
    for v, (lr, h) in best.items():
        lines.append("| {} | {} | {:.4f} | {:.4f} | **{:.4f}** |".format(
            NAMES[v], lr, final(h, "valid_acc"), final(h, "test_loss"), final(h, "test_acc")))
    with open(RESULTS, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    colors = {"resnet18": "#1f77b4", "no_residual": "#d62728", "no_batchnorm": "#2ca02c"}

    # (e) one panel per model: train loss (every epoch) + test loss (every 10 epochs)
    fig, axes = plt.subplots(1, len(best), figsize=(5.5 * len(best), 4.5), squeeze=False)
    for ax, (v, (lr, h)) in zip(axes[0], best.items()):
        ax.plot(np.arange(1, len(h["train_loss"]) + 1), h["train_loss"], color=colors[v],
                label="train loss")
        ax.plot(h["eval_epochs"], h["test_loss"], "o--", color="black", label="test loss")
        ax.set_title("{}  (lr={})\nfinal test acc = {:.2%}".format(NAMES[v], lr, final(h, "test_acc")))
        ax.set_xlabel("epoch")
        ax.set_ylabel("cross-entropy loss")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "loss_curves_best.png"), dpi=160)

    # all three models on shared axes
    fig, ax = plt.subplots(figsize=(8, 5))
    for v, (lr, h) in best.items():
        ax.plot(np.arange(1, len(h["train_loss"]) + 1), h["train_loss"], color=colors[v],
                label="{} lr={} - train".format(NAMES[v], lr))
        ax.plot(h["eval_epochs"], h["test_loss"], "o--", color=colors[v],
                label="{} lr={} - test".format(NAMES[v], lr))
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("CIFAR-10: train loss (solid, every epoch) vs test loss (dashed, every 10 epochs)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "loss_curves_all.png"), dpi=160)
    print("figures saved to", FIG_DIR)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="runs",
                        help="folder (relative to this script) holding <variant>_lr<lr>/history.json, "
                             "e.g. runs_hprc for the HPRC runs")
    args = parser.parse_args()
    RUN_DIR = os.path.join(HERE, args.run_dir)
    if args.run_dir != "runs":
        # keep the HPRC report separate from the local one
        FIG_DIR = os.path.join(RUN_DIR, "figures")
        RESULTS = os.path.join(RUN_DIR, "results.md")
    main()
