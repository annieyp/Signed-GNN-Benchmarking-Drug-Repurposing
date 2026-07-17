# GNN-Benchmarking-On-Biomedical-KG

### Signed GNN Embedding Benchmarking for Drug Repurposing

Benchmarking pipeline comparing signed graph neural network (GNN) embeddings against traditional machine learning baselines for link-sign prediction on a drug-repurposing signed graph. Developed as part of research at the Xie Lab, Hunter College.

## Overview

Signed graph neural networks (SGCN, SDGNN, SNEA, SiGAT) learn node embeddings from graphs with positive and negative edges. This project evaluates how well those learned embeddings support downstream link-sign classification compared to classic ML models trained on the same embeddings, to understand whether GNN-based representations offer a meaningful edge over simpler baselines for this task.

## What this does

- Loads precomputed node embeddings and train/val/test edge splits for each signed GNN model
- Constructs edge-level feature vectors by concatenating the embeddings of an edge's source and target nodes
- Benchmarks four traditional classifiers on those edge features:
  - Random Forest
  - Logistic Regression
  - K-Nearest Neighbors
  - Linear SVC
- Tunes each classifier via exhaustive grid search over model-specific hyperparameters, selecting the configuration with the best macro F1 score on the validation set
- Re-evaluates the best configuration for each model across multiple random seeds and reports the averaged F1 score, to check that results aren't an artifact of a single lucky initialization

## Repo structure

```
.
├── Tuning.ipynb     # Data loading, model/grid definitions, tuning + multi-seed evaluation
└── README.md
```

## Methodology notes

- Edge features are built by concatenating source- and target-node embeddings (`X = [emb_source | emb_target]`)
- Hyperparameter search uses `sklearn`'s `ParameterGrid` for full grid coverage per model, rather than randomized search, to guarantee the reported best parameters are the true grid optimum
- Model selection criterion is macro-averaged F1, chosen to avoid bias toward the majority class in the (imbalanced) sign-prediction task
- Final reported metric is the mean F1 across three seeds (137698, 123, 42) at the best-found hyperparameters, to reduce variance from a single train/val split

## Tech stack

Python · scikit-learn · PyTorch · NumPy · Pandas
