# ML Homework IV (Fall 2026)

| Folder | Problem | Main script | Write-up |
|---|---|---|---|
| `Fall26ML-HW4/sonar/` | 2 & 3: logistic regression and SVMs (liblinear / libsvm) | `hw4_sonar.py` | `Q2_Q3_writeup.md` |
| `Fall26ML-HW4/ResNet/` | 4: ResNet18 on CIFAR-10 | `main.py`, `report.py` | `results.md` |
| `Fall26ML-HW4/LibAUC/` | 5: AUROC/AUPRC on PneumoniaMNIST with LibAUC | `hw5_libauc.py` | `Q5_writeup.md` |

The local results for Problem 4 were produced on an Apple M2 GPU. The next section explains how to rerun Problem 4 on a CUDA GPU on **TAMU HPRC** so the two can be compared.

---

## Running Problem 4 on TAMU HPRC (Grace)

The job trains 9 models: 3 architectures × 3 learning rates, 100 epochs each.

| Model | Learning rates |
|---|---|
| ResNet18 | 0.003, 0.01, 0.03 |
| ResNet18 without residual connections | 0.003, 0.01, 0.03 |
| ResNet18 without BatchNorm | 0.003, 0.01, 0.03 |

The job script runs them as a **Slurm job array**: 9 jobs with one GPU each, running in parallel. Results are written to `Fall26ML-HW4/ResNet/runs_hprc/`, separate from the local `runs/`.

All commands below run on the **login node** unless a step says otherwise. Login nodes have internet access; compute nodes do not, so do every download (git, pip, dataset) before submitting.

### 1. Log in

```bash
ssh <NetID>@grace.hprc.tamu.edu        # on campus or via the TAMU VPN; Duo 2FA required
```

The web portal also works (https://portal.hprc.tamu.edu → *Clusters* → *Grace Shell Access*).

### 2. Get the code and create a Python environment in `$SCRATCH`

Use `$SCRATCH`, not `$HOME`: the home quota is small, and PyTorch with its CUDA libraries is several GB.

```bash
cd $SCRATCH
git clone <REPO_URL> HW-4_ML
cd HW-4_ML/Fall26ML-HW4/ResNet

# pick a Python 3.10+ module; `module spider Python` lists them,
# and `module spider Python/<version>` shows exactly what to load first
module purge
module load GCCcore/12.3.0 Python/3.11.3      # <- adjust to what module spider shows

export PIP_CACHE_DIR=$SCRATCH/.pip-cache        # keep pip's cache out of $HOME
python -m venv $SCRATCH/hw4-venv
source $SCRATCH/hw4-venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt                 # torch (CUDA build on Linux), numpy, matplotlib
```

> **Important:** if you load a different module line than the one above, put the same line in the `EDIT ME` block of `hprc_job.slurm`. The venv only works with the Python it was built from.

### 3. Download CIFAR-10 (python version, ~170 MB)

The code reads `ResNet/data/cifar-10-batches-py/`, which is not stored in git.

```bash
mkdir -p data && cd data
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xzf cifar-10-python.tar.gz && rm cifar-10-python.tar.gz
cd ..
ls data/cifar-10-batches-py        # data_batch_1 ... data_batch_5, test_batch, batches.meta
```

### 4. Smoke test (1 epoch, about 5 minutes)

```bash
mkdir -p logs
EPOCHS=1 sbatch --array=0 --time=00:30:00 hprc_job.slurm
squeue -u $USER                     # wait until it disappears
cat logs/hw4-resnet_*_0.out         # should show an A100, "Epoch 1 Loss ...", and a test accuracy
rm -rf runs_hprc logs/*             # clean up before the real run
```

If the job is rejected with an account error, uncomment the `--account` line in `hprc_job.slurm`. `myproject` shows your account number(s).

### 5. Submit the full run

```bash
sbatch hprc_job.slurm               # all 9 runs (array 0-8)
# or only some of them:
sbatch --array=0-2 hprc_job.slurm   # ResNet18 only (task ids are listed at the top of the script)
```

Each run requests 1 A100, 4 CPUs, 32 GB RAM and 8 hours. On the M2 Mac one epoch took about 45 s, so a run took about 1.2 h. An A100 should be faster. Data augmentation is done in NumPy on the CPU, so the GPU will not be fully busy; that is expected. If A100s are busy, change `--gres=gpu:a100:1` to `gpu:rtx:1` or `gpu:t4:1`.

### 6. Monitor

```bash
squeue -u $USER                          # state of the 9 array tasks
tail -f logs/hw4-resnet_<JOBID>_4.out    # follow one run (task 4 = no_residual, lr 0.01)
scancel <JOBID>                          # cancel all tasks; <JOBID>_<task> cancels one
seff <JOBID>_<task>                      # efficiency report after a task finishes
```

Every 10 epochs the log prints validation and test loss/accuracy, and `runs_hprc/<variant>_lr<lr>/history.json` is updated. A crashed run therefore still keeps its curves up to the last evaluation.

Some runs are **expected** to fail. ResNet18 without BatchNorm at lr 0.01 or 0.03 sits at 10% accuracy (chance level) and its loss spikes into the millions. That is a real result to report, not a bug in the setup.

### 7. Build the report and push the results

```bash
source $SCRATCH/hw4-venv/bin/activate
python report.py --run_dir runs_hprc     # -> runs_hprc/results.md and runs_hprc/figures/*.png

git add runs_hprc/                       # *.ckpt files are excluded by .gitignore
git commit -m "Problem 4: HPRC (A100) runs"
git push
```

### 8. Compare with the local runs

| | local (Apple M2, MPS) | HPRC (A100, CUDA) |
|---|---|---|
| tables | `ResNet/results.md` | `ResNet/runs_hprc/results.md` |
| curves | `ResNet/figures/` | `ResNet/runs_hprc/figures/` |
| per-run histories | `ResNet/runs/*/history.json` | `ResNet/runs_hprc/*/history.json` |

Both machines use the same code, seed and hyper-parameters. Accuracies will still differ slightly (typically within about 1 percentage point), because GPU kernels and floating-point order differ between MPS and CUDA. Compare which learning rate wins for each model and the size of the gaps between the three models, not exact decimals.

### Settings (from the HW statement)

| setting | value |
|---|---|
| epochs | 100 |
| learning rate | tuned over {0.003, 0.01, 0.03}, chosen on a 5k validation split |
| lr schedule | divided by 1.5 every 10 epochs |
| optimizer | Adam, weight decay 2e-4 |
| batch size | 128 |
| data | 45k train / 5k validation (from the 50k train set), 10k test |
| augmentation | pad 4 + random crop 32 + horizontal flip + per-channel normalization (NumPy, no torchvision) |
| test loss | computed every 10 epochs |
