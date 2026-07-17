# Correlation-Aware Affine Maximizer Auction (CA-AMA)

This repository contains the official implementation of the paper:

> **Enhancing Affine Maximizer Auctions with Correlation-Aware Payment**  
> Published at ICML 2026

## Overview

This codebase implements Contextual Affine Maximizer Auctions (CA-AMA), a neural network-based auction mechanism that leverages bidder context information to improve auction efficiency and revenue through correlation-aware payment rules.

## Architecture

The codebase is organized as follows:

### Core Auction Mechanism
- **`auction.py`** - Main auction implementation. Defines `ContextualAffineMaximizerAuction` class, which combines allocation and payment rules. It uses Transformer-based networks to compute contextual weights and correlation-aware adjustments.

### Neural Network Architectures
- **`net.py`** - Neural network modules including:
  - `TransformerMechanism`: Transformer-based architecture for computing affine weights
  - `Payment_Cor` / `Payment_Cor_max_min`: Correlation-aware payment adjustment modules
  - `SimpleLipschitzMLP`: Lipschitz-bounded MLP for stability
  - Ablation variants for controlled experiments

### Data Generation
- **`gen_values.py`** - Synthesizes valuation data for bidders across various experimental settings (different distributions, contexts, and scenarios).

### Training
- **`train_caama.py`** - Trains the CA-AMA mechanism with multi-objective optimization (revenue, efficiency, IR constraints).
- **`train_caama_post.py`** - Post-processing training for fine-tuning.
- **`train_baseline.py`** - Trains baseline auction mechanisms for comparison.

### Evaluation
- **`test_caama.py`** / **`test_baseline.py`** - Evaluates trained models on test data.

### Utilities
- **`logger.py`** - Logging and model checkpointing utilities.
- **`train_*.sh`** / **`test.sh`** - Shell scripts for batch training and testing.

## Quick Start

Train CA-AMA:
```bash
python train_caama.py --n_agents 2 --m_items 2 --train_steps 2000
```

Test the trained model:
```bash
python test_caama.py --load_path results/model.pt
```

Or use provided scripts:
```bash
bash train_caama.sh
```

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- tqdm
