# Fake News Detection Baseline Framework

Lightweight PyTorch experimentation harness for fake news detection competitions. The
framework auto-discovers datasets, normalises their schemas, and offers
plug-and-play text classification models with consistent training and evaluation
utilities.

## Key Features
- **Dataset autodiscovery** – every subdirectory under `dataset/` is parsed at runtime.
  Each dataset provides `train.csv`, `val.csv`, and `test.csv` files with `title`, `text`,
  and `label` columns. Labels are normalised to `{0: true/real, 1: fake}` automatically.
- **Unified vocabulary & tokenisation** – whitespace tokeniser with configurable vocab
  cap, padding, and truncation.
- **Model zoo** – Bag-of-Words MLP, CNN, BiLSTM, and Tiny Transformer baselines located
  in `model/`.
- **Training pipeline** – AdamW training with early stopping on validation F1, metrics
  logging, and artifact management under `runs/{dataset}/{model}/{timestamp}/`.
- **Notebooks** – interactive playground (`notebooks/test.ipynb`) and dataset explorer
  (`notebooks/analysis_all_datasets.ipynb`).

## Installation
1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Datasets
Place all competition datasets inside the top-level `dataset/` directory. The structure
per dataset should resemble:
```
dataset/
├─ dataset_a/
│  ├─ train.csv
│  ├─ val.csv
│  └─ test.csv
└─ dataset_b/
   ├─ train.csv
   ├─ val.csv
   └─ test.csv
```
Each CSV must include `title`, `text`, and `label` columns. The loader consumes only
`title` and `text` (concatenated) as model input, while `label` values (e.g. `fake`/`real`)
are mapped to numerical targets automatically. All splits are expected to be labeled and
precomputed; no additional splitting occurs.

## Training a Model
Use the CLI in `scripts/run_train.py` for individual experiments:
```bash
python scripts/run_train.py \
    --dataset_name fake-news-classification \
    --model bow_mlp \
    --epochs 5 \
    --batch_size 64 \
    --max_len 256
```
Outputs include:
- `metrics.csv` – per-epoch losses and validation scores.
- `best_metrics.json` – final metrics for train/val/test splits.
- `confusion_matrix.png` – validation confusion matrix.
- `best_model.pt` – trained weights.

All artifacts are stored under `runs/<dataset>/<model>/<timestamp>/`.

## Quick Benchmark Sweep
Evaluate every dataset/model combination for a short run:
```bash
python scripts/quick_eval_all.py --epochs 2 --output sweep_results.csv
```
The script records summary metrics for each configuration and saves an aggregate CSV.

## Notebooks
- **`notebooks/test.ipynb`** – interactive training notebook. Select a dataset and
  model, run training, inspect metrics, and compare multiple runs in a tabular summary.
- **`notebooks/analysis_all_datasets.ipynb`** – exploratory analysis across datasets.
  Generates label distributions, text length plots, top uni/bi-grams, missing value
  tables, and optional word clouds.

## Configuration
Default hyperparameters live in `configs/default.yaml`:
```yaml
epochs: 5
batch_size: 64
lr: 0.0003
max_len: 256
patience: 3
num_workers: 0
```
Override values via CLI arguments or the notebooks. Custom configs can be added to the
`configs/` directory and passed through the `--config` flag.

## Project Layout
```
core/                # Dataset loading, vocab, metrics, and training utilities
model/               # PyTorch model definitions and registry
scripts/             # CLI entrypoints for training and sweeps
configs/             # YAML hyperparameter configs
notebooks/           # Interactive experimentation notebooks
runs/                # Generated at runtime to store metrics and checkpoints
requirements.txt     # Python dependencies
README.md            # Project overview and usage
```

## Next Steps
- Extend `model/` with more advanced architectures.
- Plug in richer tokenisers (e.g., subword models) by updating `core/vocab.py`.
- Integrate experiment tracking services by augmenting `core/train_eval.py`.
