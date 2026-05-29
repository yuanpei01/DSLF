# DSLF: Dual-Stream Learning and Fusion for Rumor Detection

This repository contains the code for the Dual-Stream Learning and Fusion
(DSLF) framework used for rumor detection experiments on Twitter15, Twitter16,
and Weibo.

DSLF separates heterogeneous rumor-detection signals into two streams:

- **Stream I** learns semantic and user-interaction representations through a
  GLAN-based module with similarity-based comment enhancement.
- **Stream II** learns propagation-dispersion representations through a
  BiGCN-based module.

The stream outputs are fused by a fully connected classifier. The repository
also includes E2E comparison scripts and an `_St` variant for reusing previously
generated Stream II features when testing new Stream I settings.

## Repository Structure

```text
DSLF/
|-- run_dl.py                  # Run the modular DSLF pipeline
|-- run_dl_St.py               # Reuse Stream II features and retrain Stream I/fusion
|-- run_e2e.py                 # Run the end-to-end baseline pipeline
|-- dataset_config.py          # Dataset paths, checkpoints, and hyperparameters
|-- pipeline_config.py         # Output paths and timing/FLOPs logging helpers
|-- DL/                        # DSLF feature-generation and fusion scripts
|-- E2E/                       # End-to-end training and testing scripts
|-- models/                    # Fusion and E2E model wrappers
`-- process/                   # Data loading, preprocessing, alignment, FLOPs
```

Generated artifacts such as checkpoints, logs, `.npy` features, and pretrained
word vectors are intentionally excluded from version control.

## Requirements

The code was developed with Python 3.8. Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The pipeline also depends on the original GLAN and BiGCN code/data layouts.
Place or link these directories where `dataset_config.py` expects them, or edit
the paths in `dataset_config.py`:

```text
DSLF/
|-- GLAN/
|   |-- checkpoint/
|   `-- dataset/
|-- BiGCN/
|   |-- data/
|   `-- pretrained_BiGCNmodel_*.pth
`-- process/
    |-- twitter_w2v.bin
    |-- weibo_w2v.bin
    `-- dict.txt.big
```

### Required External Files

The following three external files are required but are not included in this
code release because of their file size:

```text
process/twitter_w2v.bin
process/weibo_w2v.bin
process/dict.txt.big
```

- `twitter_w2v.bin`: pretrained word2vec file used for Twitter15 and Twitter16.
- `weibo_w2v.bin`: pretrained word2vec file used for Weibo.
- `dict.txt.big`: Jieba dictionary file used for Chinese tokenization.

Download or copy these files separately and place them directly under the
`process/` folder before running the pipeline. The expected paths are already
configured in `dataset_config.py` and `process/preprocess_similarty_ST.py`.

Other external resources used by the scripts include:

- Original GLAN/RumorDetection code and checkpoints:
  [chunyuanY/RumorDetection](https://github.com/chunyuanY/RumorDetection).
- Original BiGCN code, graph data, and pretrained BiGCN checkpoints:
  [TianBian95/BiGCN](https://github.com/TianBian95/BiGCN).
- Twitter and Weibo word2vec files used by Stream I.
- `dict.txt.big` for Jieba Chinese tokenization.

## Usage

Run commands from the `DSLF/` directory after configuring the external data and
checkpoint paths.

Based on validation performance, the selected thresholds for similarity
filtering are `0.5` for Twitter15, `0.5` for Twitter16, and `0.8` for Weibo.
The examples below use these dataset-specific thresholds for `thred` runs.

### DSLF Pipeline

```bash
python run_dl.py thred 0.5 twitter15
python run_dl.py thred 0.5 twitter16
python run_dl.py thred 0.8 weibo
```

For DSLF-W, replace `thred` with `cos` in the commands above.

### DSLF with Stream II Feature Reuse

The `_St` scripts are used when Stream II propagation-dispersion features have
already been generated and should be reused. This is useful for testing different
Stream I semantic enhancement strategies or threshold settings without
retraining the BiGCN-based Stream II module.

First run the standard DSLF pipeline once for the same dataset, method, and seed
setting so that the Stream II feature files are created under `checkpoints/`:

```text
{data}_BiGCN{method}_train_outputs{rep}.npy
{data}_BiGCN{method}_val_outputs{rep}.npy
{data}_BiGCN{method}_test_outputs{rep}.npy
```

Then run the `_St` entry point to regenerate Stream I features and train the
fusion classifier while loading the existing Stream II features:

```bash
python run_dl_St.py thred 0.5 twitter15
python run_dl_St.py thred 0.5 twitter16
python run_dl_St.py thred 0.8 weibo
```

For DSLF-W with Stream II feature reuse, replace `thred` with `cos` in the
commands above.

In this mode, `DL/generate_train_val_features_St.py` and
`DL/generate_test_features_St.py` focus on the GLAN-based Stream I feature
generation. The BiGCN-based Stream II feature extraction code is skipped, and
the fusion stage reuses the saved Stream II `.npy` outputs from `checkpoints/`.

### E2E Baseline

```bash
python run_e2e.py thred 0.5 twitter15
python run_e2e.py thred 0.5 twitter16
python run_e2e.py thred 0.8 weibo
```

Arguments:

- `method`: `cos` for similarity weighting or `thred` for similarity filtering.
- `threshold`: similarity threshold used by Stream I. In our experiments, this
  value is `0.5`, `0.5`, and `0.8` for Twitter15, Twitter16, and Weibo,
  respectively. For `cos`, this argument is still accepted by the scripts but
  is not used by the cosine-weighting strategy.
- `data`: one of `twitter15`, `twitter16`, or `weibo`.

## Outputs

The scripts create output folders automatically:

- `checkpoints/`: trained models and extracted feature arrays.
- `logs/`: training logs, timing records, FLOPs records, and test reports.

These folders are ignored by Git because they contain generated artifacts.

## Notes on Reproducibility

The scripts use the public train/development/test files from the GLAN data
layout and align GLAN and BiGCN instances by event ID before feature fusion.
Model selection should be conducted on the development set, and the test set
should be used only for final evaluation.

If you use a different data layout or a different public split, update
`dataset_config.py` and document the split protocol before comparing results.
