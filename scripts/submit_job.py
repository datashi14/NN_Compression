import argparse
import os
import subprocess
import time
import yaml

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def main():
    parser = argparse.ArgumentParser(description="Submit TicketSmith Job to AKS")
    parser.add_argument('--job-type', type=str, required=True, choices=['dense', 'imp', 'benchmark'], help='Job type')
    parser.add_argument('--config', type=str, default='configs/dense_mnist.yaml', help='Path to config')
    parser.add_argument('--image-tag', type=str, default='ticketsmith:latest', help='Image tag to use/push')
    parser.add_argument('--registry', type=str, help='Container registry (ACR) name')
    parser.add_argument('--push', action='store_true', help='Build and push image')
    args = parser.parse_args()

    # 1. Build and Push
    if args.push:
        print("Building Docker image...")
        run_command(f"docker build -t {args.image_tag} .")
        if args.registry:
            full_tag = f"{args.registry}.azurecr.io/{args.image_tag}"
            run_command(f"docker tag {args.image_tag} {full_tag}")
            print(f"Pushing to {args.registry}...")
            run_command(f"docker push {full_tag}")
            image_uri = full_tag
        else:
            image_uri = args.image_tag
    else:
        # Assume already pushed or local
        image_uri = args.image_tag if not args.registry else f"{args.registry}.azurecr.io/{args.image_tag}"

    # 2. Prepare K8s YAML
    # Select template
    if args.job_type == 'dense':
        template_path = 'k8s/train_job.yaml'
        cmd_arg = 'ticketsmith.train'
    elif args.job_type == 'imp':
        template_path = 'k8s/train_job.yaml' # Reuse train job but change command
        cmd_arg = 'ticketsmith.prune'
    elif args.job_type == 'benchmark':
        template_path = 'k8s/benchmark_job.yaml'
        cmd_arg = 'ticketsmith.benchmark_cli'

    with open(template_path, 'r') as f:
        job_yaml = f.read()

    run_id = f"{int(time.time())}"
    job_name = f"ticketsmith-{args.job_type}-{run_id}"
    
    # Replace placeholders
    job_yaml = job_yaml.replace('{{RUN_ID}}', run_id)
    job_yaml = job_yaml.replace('{{IMAGE_TAG}}', image_uri)
    # Check if template has command replacement (my train_job.yaml hardcoded command previously)
    # I should update train_job.yaml to be more flexible or edit it here
    
    # Actually my train_job.yaml has hardcoded command: ["python", "-m", "ticketsmith.train", ...]
    # I should probably update the YAMLs to use variables or generic entrypoint.
    # For MVP, checking if I need to patch the command
    if args.job_type == 'imp':
        job_yaml = job_yaml.replace('ticketsmith.train', 'ticketsmith.prune')
        job_yaml = job_yaml.replace('configs/dense_mnist.yaml', args.config)
    elif args.job_type == 'dense':
        job_yaml = job_yaml.replace('configs/dense_mnist.yaml', args.config)
        
    generated_yaml_path = f"generated_job_{run_id}.yaml"
    with open(generated_yaml_path, 'w') as f:
        f.write(job_yaml)
        
    print(f"Generated Job: {generated_yaml_path}")
    
    # 3. Submit
    # run_command(f"kubectl apply -f {generated_yaml_path}")
    print("Simulated submission. Run 'kubectl apply -f ...' to execute.")

if __name__ == "__main__":
    main()
