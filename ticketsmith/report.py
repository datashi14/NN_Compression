import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import glob


def load_run_data(runs_dir):
    data = []
    
    # Find all metrics.json files
    metrics_files = glob.glob(os.path.join(runs_dir, '*', 'metrics.json'))
    
    for mf in metrics_files:
        run_dir = os.path.dirname(mf)
        run_id = os.path.basename(run_dir)
        
        try:
            with open(mf, 'r') as f:
                metrics = json.load(f)
            
            # Load config if available
            config_path = os.path.join(run_dir, 'config.yaml')
            config = {}
            if os.path.exists(config_path):
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
            experiment_name = config.get('experiment_name', 'unknown')
            
            # Extract data handling both IMP rounds and single dense runs
            
            # Helper to extract Round info
            # We look for keys `round_(\d+)_acc`
            found_rounds = set()
            for key in metrics.keys():
                if key.startswith('round_') and key.endswith('_acc'):
                    round_idx = int(key.split('_')[1])
                    found_rounds.add(round_idx)
            
            if not found_rounds:
                # Fallback for simple dense or broken runs (assume round 0 if val_acc exists)
                if 'val_acc' in metrics:
                    found_rounds.add(0)
            
            for round_idx in sorted(found_rounds):
                # 1. Basic Metrics
                if round_idx == 0 and 'round_0_acc' not in metrics and 'val_acc' in metrics:
                     # Fallback for dense
                     best_acc = max([x['value'] for x in metrics.get('val_acc', [{'value':0}])])
                     sparsity = 0.0
                else:
                    acc_entries = metrics.get(f'round_{round_idx}_acc', [{'value':0}])
                    best_acc = max([x['value'] for x in acc_entries])
                    
                    # Sparsity
                    if round_idx == 0:
                        sparsity = 0.0
                    else:
                        s_key = f'round_{round_idx}_sparsity'
                        sparsity = metrics.get(s_key, [{'value': 0.0}])[0]['value']
                
                # 2. Quality Gate
                gate_key = f'round_{round_idx}_gate'
                gate_passed = False
                gate_reasons = ""
                if gate_key in metrics:
                    gate_data = metrics[gate_key][0]['value'] # It's a list of events
                    # If logged multiple times, take last? Usually log once.
                    if isinstance(gate_data, list): gate_data = gate_data[-1] # Safety
                    gate_passed = gate_data.get('passed', False)
                    gate_reasons = "; ".join(gate_data.get('reasons', []))
                elif round_idx == 0:
                    # Baseline implies pass
                    gate_passed = True
                    gate_reasons = "Baseline"
                    
                # 3. Benchmark (Serving)
                # We logged: bench_{variant_name}_bs{bs}_throughput
                # Variant name constructed as: round_{round_idx}_sparsity_{int}
                bench_bs1_lat = 0
                bench_bs1_thru = 0
                bench_bs32_thru = 0
                
                # Construct variant name prefix used in metrics
                # Since we don't know the exact int(sparsity*100) due to float precision, 
                # we search keys? Or re-construct best guess.
                # Actually we can search keys starting with `bench_round_{round_idx}_`
                
                # Search for bench keys for this round
                for key in metrics.keys():
                    if key.startswith(f'bench_round_{round_idx}_'):
                        # key format: bench_round_0_sparsity_0_bs1_throughput
                        parts = key.split('_')
                        # Find batch size
                        if 'bs1' in parts and 'latency' in parts:
                             bench_bs1_lat = metrics[key][0]['value']
                        if 'bs1' in parts and 'throughput' in parts:
                             bench_bs1_thru = metrics[key][0]['value']
                        if 'bs32' in parts and 'throughput' in parts:
                             bench_bs32_thru = metrics[key][0]['value']

                data.append({
                    'RunID': run_id,
                    'Experiment': experiment_name,
                    'Round': round_idx,
                    'Sparsity': sparsity,
                    'Accuracy': best_acc,
                    'GatePassed': gate_passed,
                    'GateReasons': gate_reasons,
                    'Latency_BS1': bench_bs1_lat,
                    'Throughput_BS1': bench_bs1_thru,
                    'Throughput_BS32': bench_bs32_thru
                })
                    
        except Exception as e:
            print(f"Error reading {run_dir}: {e}")
            
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description="Generate experiment report.")
    parser.add_argument('--runs-prefix', type=str, required=True, help='Path to runs directory')
    args = parser.parse_args()
    
    print(f"Scanning runs in {args.runs_prefix}")
    df = load_run_data(args.runs_prefix)
    
    if df.empty:
        print("No run data found.")
        return

    print("Data Summary:")
    print(df)
    
    # Save CSV
    df.to_csv("run_ledger.csv", index=False)
    
    # Generate Plot
    plt.figure(figsize=(10, 6))
    for name, group in df.groupby('Experiment'):
        group = group.sort_values('Sparsity')
        # Filter purely failing points? No, show all, maybe different marker
        plt.plot(group['Sparsity'], group['Accuracy'], marker='o', label=name)
        
        # Mark Failed Gates with X
        failed = group[~group['GatePassed']]
        if not failed.empty:
             plt.scatter(failed['Sparsity'], failed['Accuracy'], marker='x', c='red', s=100, zorder=10)

    plt.xlabel('Sparsity')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Sparsity (Red X = Gate Failed)')
    plt.grid(True)
    plt.legend()
    plt.savefig('report_plot.png')
    
    # Generate Markdown Report
    with open('executive_summary.md', 'w') as f:
        f.write("# TicketSmith Executive Summary\n\n")
        
        # Recommendation
        # Find highest sparsity that Passed Gate
        passed_df = df[df['GatePassed']]
        if not passed_df.empty:
            # Sort by Sparsity Descending
            best_safe = passed_df.sort_values('Sparsity', ascending=False).iloc[0]
            f.write(f"## Recommendation\n\n")
            f.write(f"**Safe Optimisation Zone**: Up to **{best_safe['Sparsity']*100:.1f}% sparsity**.\n")
            f.write(f"Best variant: {best_safe['Experiment']} (Round {best_safe['Round']}).\n")
            f.write(f"Speedup: {best_safe['Throughput_BS1']:.1f} img/s (BS1) vs baseline.\n\n")
        else:
             f.write("## Recommendation\n\nNo variants passed the quality gate.\n\n")

        f.write("## Quality Gate Table\n\n")
        f.write(df[['Experiment', 'Round', 'Sparsity', 'Accuracy', 'GatePassed', 'GateReasons']].to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Serving Scorecard (Benchmark)\n\n")
        f.write(df[['Experiment', 'Sparsity', 'Latency_BS1', 'Throughput_BS1', 'Throughput_BS32']].to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Visualizations\n\n")
        f.write("![Accuracy vs Sparsity](report_plot.png)\n")

    print("Report generated: executive_summary.md, run_ledger.csv, report_plot.png")


if __name__ == "__main__":
    main()
