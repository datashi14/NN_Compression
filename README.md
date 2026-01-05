# TicketSmith

**TicketSmith** is a production-style ML experimentation system for demonstrating the Lottery Ticket Hypothesis (LTH) and managing cost-quality tradeoffs in model optimization.

## Goals

- **Demonstrate cost-quality tradeoffs**: "At X% sparsity, quality drops by Y% and compute decreases by Z."
- **Azure Kubernetes Service (AKS)**: Runs as scalable Jobs with cost guardrails (scale-to-zero).
- **Reproducibility**: Experiments are isolated, config-driven, and produce immutable artifacts.

## Architecture

- **Experiment Image**: Docker container with PyTorch & experiment code.
- **Job Runner**: Kubernetes Jobs for isolation.
- **Artifact Storage**: Azure Blob Storage for metrics, plots, and models.
- **Report Generator**: CPU-only job for aggregating results.

## Usage

### Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run dense training
python -m ticketsmith.train --config configs/dense.yaml
```

### CLI Entrypoints

- `python -m ticketsmith.train --config <path>`
- `python -m ticketsmith.prune --config <path>`
- `python -m ticketsmith.report --runs-prefix <blob-path>`
