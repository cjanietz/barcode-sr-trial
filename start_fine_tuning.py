#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Start fine-tuning SPAN model.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the dataset directory containing train_config.yml",
    )
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--port", type=str, default="29500", help="Master port for distributed training"
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    config_path = dataset_dir / "train_config.yml"

    if not dataset_dir.exists():
        print(f"Error: Dataset directory does not exist: {dataset_dir}")
        sys.exit(1)

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        print(f"Did you run generate_barcode_sr_dataset.py with --generate_config?")
        sys.exit(1)

    print(f"Found configuration: {config_path}")

    # Locate SPAN directory relative to this script
    script_dir = Path(__file__).resolve().parent
    span_dir = script_dir / "SPAN"
    dist_train_script = span_dir / "scripts" / "dist_train.sh"
    train_script = span_dir / "basicsr" / "train.py"

    if not dist_train_script.exists():
        print(f"Error: Training script not found at {dist_train_script}")
        print("Ensure 'models/sr/SPAN' exists and is correctly structured.")
        sys.exit(1)

    # Change to SPAN directory as working directory
    os.chdir(span_dir)
    print(f"Working directory: {os.getcwd()}")

    # Check for CUDA
    import torch

    if not torch.cuda.is_available():
        print("CUDA not available. Running in non-distributed mode (CPU/MPS).")
        cmd = [sys.executable, str(train_script), "-opt", str(config_path)]
    else:
        cmd = [str(dist_train_script), str(args.gpus), str(config_path)]

    # Pass PORT env var
    env = os.environ.copy()
    env["PORT"] = args.port
    # Ensure local python path includes current dir for modules
    env["PYTHONPATH"] = f"{span_dir}:{env.get('PYTHONPATH', '')}"

    print(f"Executing: {' '.join(cmd)}")
    print("-" * 60)

    try:
        # Use subprocess.check_call to run
        subprocess.check_call(cmd, env=env)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
