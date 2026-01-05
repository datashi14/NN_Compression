import argparse
import torch
import os
from ticketsmith.utils.config import load_config
from ticketsmith.utils.artifacts import ArtifactManager
from ticketsmith.models.mnist_cnn import MNISTCNN
from ticketsmith.utils.benchmark import run_benchmark

def main():
    parser = argparse.ArgumentParser(description="Run standalone benchmark.")
    parser.add_argument('--config', type=str, required=True, help='Experiment config')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pt)')
    parser.add_argument('--variant', type=str, default='standalone', help='Variant name for reporting')
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Init Artifact Manager (creates new run dir for benchmark results?)
    # Or should we output to specific dir?
    # Usually benchmark job is part of a workflow. Let's create a new run dir.
    am = ArtifactManager()
    am.save_config(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on {device}")
    
    # Load Model
    model = MNISTCNN().to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    
    # Run
    run_benchmark(model, config, device, am, variant_name=args.variant)
    
    am.save_metrics()
    print(f"Benchmark complete. Results in {am.run_dir}")

if __name__ == "__main__":
    main()
