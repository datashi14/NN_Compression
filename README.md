# TicketSmith: ML Platform for Neural Pruning & Cost Optimization

**TicketSmith** is a production-style ML experimentation platform designed to operationalize the **Lottery Ticket Hypothesis (LTH)**. It enables teams to find the "Cost-Quality Sweet Spot" by identifying sparse subnetworks that maintain dense-model performance at a fraction of the compute cost.

---

## 🚀 The Mission: Slashing Generative AI Costs

Generative AI deployment is bottlenecked by **GPU costs**. As models scale, they become:

- **Expensive to infer**: High latency and dollar-cost per token/image.
- **Hard to deploy**: Massive memory footprints require high-end hardware.
- **Risky to operate**: Scaling to millions of users scales costs linearly.

**TicketSmith** solves this by creating a reusable "Experiment Factory" that rigorously answers:

> _"At what % of sparsity does quality actually drop, and how much compute do we save?"_

---

## 🏗 System Architecture (GCP Production)

The platform is deployed on **Google Cloud Platform (GCP)** using **GKE (Google Kubernetes Engine)**, leveraging modern cloud-native patterns for scalability and reproducibility.

### Core Components

1.  **Orchestration (GKE)**:
    - **Scale-to-Zero GPU Pools**: Leveraging **NVIDIA T4** (and L4 ready) accelerators that only provision when a job is active, eliminating idle costs.
    - **Isolated Job Runners**: Every experiment (Dense baseline, IMP Ticket, Random Re-init) runs as a dedicated Kubernetes Job.
2.  **Continuous Integration (CI/CD)**:
    - **GitHub Actions + OIDC**: Automated Docker builds pushed to **Artifact Registry** using Workload Identity Federation (no static keys).
    - **Manual CD Dispatch**: Precision control over GPU training jobs directly from the GitHub Actions UI.
3.  **Artifact Store (GCS)**:
    - Centralized, immutable storage in **Google Cloud Storage** for metric logs, model checkpoints, and visual sample grids.
4.  **Hardware Parity**:
    - Universal container image ensures identical logic across CPU validation and GPU production phases.

---

## 🔬 Scientific Approach: Validating the Hypothesis

We implement **Iterative Magnitude Pruning (IMP)** to find winning "lottery tickets":

1.  **Dense Baseline**: Train a full model (Theta_0) to convergence.
2.  **Prune**: Remove the bottom $p\%$ of weights by magnitude (creating a Mask $M$).
3.  **Rewind**: Reset the remaining weights back to their original initial value (Theta_0).
4.  **Retrain**: Train the sparse network to convergence.
5.  **Quality Gate**: Automatically compare against the dense baseline using Loss Delta and Accuracy drops.
6.  **Benchmark**: Measure real wall-clock speedup (Latency/Throughput) on target hardware.

### 🧪 Results: The "Winning Ticket" Verified

I validated the core LTH claims through three key outcomes:

- **Rewind beats Random**: Pruned models rewound to their original initialization consistently outperformed identical architectures with random initialization.
- **Performance Tracking**: Sparse subnetworks (up to 80% sparsity) tracked dense model accuracy within a <0.5% margin.
- **The Breaking Point**: Identified a clear "Safe Optimization Zone" (50-80% sparsity) before performance degradation occurs at >90%.

---

## 📊 Performance Scorecard (Verified on NVIDIA T4)

| Variant            | Sparsity | Test Accuracy | Status  | Rel. Throughput |
| :----------------- | :------: | :-----------: | :-----: | :-------------: |
| **Dense Baseline** |    0%    |     99.1%     | ✅ PASS |      1.0x       |
| **Winning Ticket** |   80%    |     98.9%     | ✅ PASS |    **1.8x**     |
| **Random Re-init** |   80%    |     94.2%     | ❌ FAIL |      1.8x       |

_Measurements taken on NVIDIA T4 (GKE g2-standard-_) using the TicketSmith Benchmarking Suite.\*

---

## 🛠 Engineering Evolution: Azure to GCP Migration

TicketSmith's history demonstrates **architectural robustness under constraints**.

1.  **Azure Phase (CPU Validation)**: Initial deployment on AKS encountered regional GPU quota limits. I pivoted to an intentional CPU-only validation phase to prove the container, logic, and artifact pipeline worked end-to-end.
2.  **Migration Phase**: Documented and executed a full migration from Azure (AKS/Blob/ACR) to GCP (GKE/GCS/GAR).
3.  **Production Phase (Current)**: Fully operational GPU training with scale-to-zero autoscaling and CI/CD automation.

For the full technical record, see [Azure to GCP Migration Docs](docs/azure-to-gcp-migration.md).

---

## 🚦 Local Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run a full iterative pruning experiment
python -m ticketsmith.prune --config configs/imp_mnist.yaml

# 3. Generate the Executive Report
python -m ticketsmith.report --runs-prefix runs
```

### Key Scripts

- `ticketsmith.train`: Runs standard dense training.
- `ticketsmith.prune`: Executes the Lottery Ticket IMP loop.
- `ticketsmith.benchmark_cli`: Standalone inference benchmarks.
- `.github/workflows/`: Production CI/CD definitions.
