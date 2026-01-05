# TicketSmith

**TicketSmith** is a production-style ML experimentation platform designed to demonstrate and operationalize the **Lottery Ticket Hypothesis (LTH)**. It proves that we can remove a large portion of a model’s parameters (making it smaller and cheaper) while maintaining similar quality, provided we prune and retrain in a disciplined way.

## 🚀 The Mission

Generative AI deployment is bottlenecked by **GPU costs**. As models grow larger, they become:

- **Expensive to infer**: Higher latency and dollar-cost per token/image.
- **Hard to deploy**: Massive memory footprints require expensive, high-end GPUs.
- **Risky to operate**: Scaling to millions of users scales costs linearly.

**TicketSmith** solves this by creating a reusable "Experiment Factory" that rigorously finds the **Cost-Quality Sweet Spot**. It answers the critical business question:

> _"At what % of sparsity does quality actually drop, and how much compute do we save?"_

---

## 🏗 System Architecture

TicketSmith is built to run on **Azure Kubernetes Service (AKS)**, leveraging cloud-native patterns for scalability and reproducibility.

### Core Components

1.  **Experiment Image (Docker)**:

    - Encapsulates PyTorch, CUDA runtime, and our custom training/pruning logic.
    - Ensures identical environments for dense baselines and sparse variants.

2.  **Job Runner (Kubernetes Jobs)**:

    - Each experiment (Dense, Lottery Ticket, Random Re-init) runs as an isolated K8s Job.
    - **Cost Guardrails**: GPU node pools scale to zero when idle. Jobs have strict time caps (`activeDeadlineSeconds`).

3.  **Artifact Store (Azure Blob)**:

    - Centralized storage for immutable results: `metrics.json`, loss curves, sample grids, and model checkpoints.

4.  **Report Generator**:
    - A CPU-only job that aggregates all run data into a single **Executive Summary**.
    - Outputs decision-ready plots (Quality vs. Sparsity) and "Serving Scorecards" (Throughput improvements).

---

## 🔬 Scientific Approach

We implement **Iterative Magnitude Pruning (IMP)** to find winning "lottery tickets":

1.  **Dense Baseline**: Train a full generic model (Theta_0) to convergence.
2.  **Prune**: Remove the bottom $p\%$ of weights by magnitude (creating a Mask $M$).
3.  **Rewind**: Reset the remaining weights back to their initial value (Theta_0).
4.  **Retrain**: Train the sparse network to convergence.
5.  **Quality Gate**: Automatically compare the sparse model against the dense baseline using strict signals (Loss Delta, Accuracy Drop).
6.  **Benchmark**: Measure actual wall-clock speedup (Latency/Throughput) on target hardware.

We validate this outcome by comparing against a **Random Re-initialization** baseline (keeping the mask structure but destroying the weight values), proving that _initialization matters_.

---

## 🛠 Features

- **Automated Quality Gates**: "Release safety" checks that fail runs if quality degrades beyond a threshold (e.g., >2% accuracy drop).
- **Serving Awareness**: Integrated benchmarking suite that measures real-world metrics (Tokens/sec, Latency ms) rather than just theoretical parameter counts.
- **Reproducibility First**: All runs use fixed seeds, highly-configurated YAML definitions, and immutable artifact logs.
- **Cost Control**: Built-in mechanisms to ensure GPU resources are only active during actual computation.

---

## 🚦 Getting Started

### Prerequisites

- Python 3.9+
- Docker
- Kubectl (configured for AKS context)
- Azure Storage Connection String

### Local Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run a full iterative pruning experiment (Teacher: MNIST CNN)
python -m ticketsmith.prune --config configs/imp_mnist.yaml

# 3. Generate the Executive Report
python -m ticketsmith.report --runs-prefix runs
```

### Key Scripts

- `ticketsmith.train`: Runs a standard dense training loop.
- `ticketsmith.prune`: Executes the Lottery Ticket IMP loop (Train -> Prune -> Rewind -> Retrain).
- `ticketsmith.benchmark_cli`: Runs standalone inference benchmarks on saved checkpoints.
- `scripts/submit_job.py`: Helper to submit jobs to your Kubernetes cluster.

---

## 📊 Sample Output (Executive Report)

The system automatically generates a report answering:

| Variant        | Sparsity | Accuracy | Gate Status | Speedup (BS=1) |
| :------------- | :------: | :------: | :---------: | :------------: |
| Dense Baseline |    0%    |  99.1%   |  **PASS**   |      1.0x      |
| Ticket (IMP)   |   80%    |  98.9%   |  **PASS**   |    **1.8x**    |
| Random Re-init |   80%    |  94.2%   |    FAIL     |      1.8x      |

_Actual speedups depend on hardware and kernel support for sparse operations._
