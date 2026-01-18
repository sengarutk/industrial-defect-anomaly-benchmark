# Industrial Defect Anomaly Benchmark (Self-Supervised + kNN)

A compact, research-style benchmark for **industrial visual anomaly detection** using **feature extraction + kNN scoring**, evaluated on a subset of the **MVTec AD** dataset.  
Focus: **real factory-like defect detection** (rare anomalies, limited labels, low-latency inference).

✅ Includes:
- **ImageNet ResNet-18 features** vs **SimCLR self-supervised features**
- **Global** vs **Patch-level** embeddings (more realistic for surface defects)
- **Multi-seed benchmarking**
- **Speed–accuracy frontier plots** (deployment-aware analysis)

---

## Why this project matters (real-world relevance)

Industrial inspection systems must:
- detect **rare defects**
- work even when **defect types are unknown**
- run under **line-speed latency constraints**
- remain robust across **different materials/categories**

Pipeline used:
> learn or extract features → build a normal reference memory bank → score anomalies via nearest-neighbor distance.

---

## Methods Implemented

### 1) ImageNet-KNN (baseline)
- Backbone: **ResNet18 pretrained on ImageNet**
- Memory bank: embeddings from normal training images
- Scoring: **kNN distance** (k=5) after PCA reduction (dim=128)

### 2) SimCLR-KNN (self-supervised)
- Backbone: **ResNet18**
- Pretraining: SimCLR contrastive learning on normal images (per category)
- Memory bank + scoring: same as above

### Pooling modes
- **global**: one embedding per image (fast baseline)
- **patch**: multiple local embeddings per image (better for small/local defects)

---

## Dataset

This project uses the **MVTec Anomaly Detection dataset (MVTec AD)**.

Benchmarked categories:
- `bottle`
- `cable`
- `hazelnut`
- `metal_nut`

---

## Repository Structure
````
industrial-defect-anamoly-benchmark/
│
├── data/
│ ├── downloads/
│ └── mvtec_ad/
│
├── runs/
│ ├── artifacts/ # encoders + memorybanks (.pt, .pkl)
│ ├── plots/ # plots generated from runs.csv
│ └── runs.csv # all eval results logged here
│
├── scripts/
│ 00_env_check.py
│ 01_download_mvtec.py
│ 02_pretrain_simclr.py
│ 03_build_memorybank.py
│ 04_eval_anomaly.py
│ 05_make_plots.py
│ 06_multiseed_summary.py
│ 10_run_grid.py
│ 11_sanity_checks.py
│
└── src/
anomaly.py
config.py
eval_harness.py
metrics.py
models.py
mvtec.py
plotting.py
simclr.py
utils.py

````
---

## Setup

### 1) Create environment
```bash
python -m venv .venv
```
### Activate:

### Windows PowerShell

```
.venv\Scripts\Activate.ps1
```
### Linux/macOS
````
source .venv/bin/activate
````
### 2) Install requirements
````
pip install -r requirements.txt
````
If running multi-seed markdown tables:

````
pip install tabulate
````
----
## Run (end-to-end)
Run commands from repository root: industrial-defect-anamoly-benchmark/

### Step 0: Sanity checks
````
python -m scripts.00_env_check
python -m scripts.11_sanity_checks
````
### Step 1: Download dataset
````
python -m scripts.01_download_mvtec
````
### Step 2: SimCLR pretraining (per category)
Example (seed 42):

````
python -m scripts.02_pretrain_simclr --category bottle --epochs 10 --seed 42
python -m scripts.02_pretrain_simclr --category cable --epochs 10 --seed 42
python -m scripts.02_pretrain_simclr --category hazelnut --epochs 10 --seed 42
python -m scripts.02_pretrain_simclr --category metal_nut --epochs 10 --seed 42
````
Artifacts saved to:
````
runs/artifacts/simclr_encoder_<category>_seed<seed>.pt
````
### Step 3: Build memory banks
#### ImageNet (global / patch)
````
python -m scripts.03_build_memorybank --category bottle --seed 42 --mode global
python -m scripts.03_build_memorybank --category bottle --seed 42 --mode patch
````
#### SimCLR (global / patch)
````
python -m scripts.03_build_memorybank --category bottle --seed 42 --use_simclr --mode global
python -m scripts.03_build_memorybank --category bottle --seed 42 --use_simclr --mode patch
````
Artifacts saved to:
````
runs/artifacts/memorybank_<category>_<method>_<mode>_seed<seed>.pkl
````
### Step 4: Evaluate anomaly detection (writes to runs.csv)
#### ImageNet
````
python -m scripts.04_eval_anomaly --category bottle --seed 42 --mode global --run_name final
python -m scripts.04_eval_anomaly --category bottle --seed 42 --mode patch  --run_name final
````
#### SimCLR
````
python -m scripts.04_eval_anomaly --category bottle --seed 42 --use_simclr --mode global --run_name final
python -m scripts.04_eval_anomaly --category bottle --seed 42 --use_simclr --mode patch  --run_name final
````
Output written to:
````
runs/runs.csv
````
### Step 5: Generate plots
````
python -m scripts.05_make_plots
````
Plots saved to:
````
runs/plots/
````
### Step 6: Multi-seed summary table
````
python -m scripts.06_multiseed_summary --runs_csv runs/runs.csv
````
#### Outputs:
````
runs/summary_multiseed.csv
runs/summary_multiseed.md
````
---
## Multi-seed Benchmarking
- Seeds: 42, 43, 44

- Methods: ImageNet-KNN, SimCLR-KNN

- Modes: global, patch

- Metrics:
1) Image AUROC
2) Average latency per image (s/image)
---
## Key Findings
- Patch embeddings are more realistic for industrial defect detection because many anomalies are localized (scratches/cracks/dents).

- ImageNet features provide a strong and stable baseline across categories.

- SimCLR can contribute to category-specific representations, but with limited pretraining (10 epochs) can be unstable in patch mode.

- Reporting AUROC + latency enables deployment-aware evaluation (inspection line speed constraints).
--- 
## Outputs
- runs/runs.csv: evaluation logs (appended per run)

- runs/plots/*.png: plots (AUROC by category, latency vs AUROC)

- runs/artifacts/: encoders + memory banks

- runs/summary_multiseed.*: aggregated multi-seed metrics

## References
- MVTec AD dataset (industrial anomaly detection benchmark)

- SimCLR: A Simple Framework for Contrastive Learning of Visual Representations