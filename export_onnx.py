#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import torch

# Add SPAN directory to python path to allow importing basicsr
script_dir = Path(__file__).resolve().parent
span_dir = script_dir / "SPAN"
sys.path.append(str(span_dir))

try:
    from basicsr.archs.span_arch import SPAN
except ImportError as e:
    print(f"Error importing SPAN architecture: {e}")
    print(f"Ensure that {span_dir} exists and contains basicsr/archs/span_arch.py")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Convert SPAN model to ONNX.")
    parser.add_argument(
        "--input",
        type=str,
        default="models/spanx2_ch48.pth",
        help="Path to input .pth model file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model.onnx",
        help="Path to output .onnx file",
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=2,
        help="Upscale factor (default: 2)",
    )
    parser.add_argument(
        "--feature_channels",
        type=int,
        default=48,
        help="Number of feature channels (default: 48)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = script_dir / input_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = script_dir / output_path

    if not input_path.exists():
        print(f"Error: Input model not found at {input_path}")
        sys.exit(1)

    print(f"Loading model from {input_path}...")

    # Initialize model
    model = SPAN(
        num_in_ch=3,
        num_out_ch=3,
        feature_channels=args.feature_channels,
        upscale=args.upscale,
    )
    model.to("cpu")

    # Load weights
    try:
        checkpoint = torch.load(input_path, map_location=torch.device("cpu"))
        if "params_ema" in checkpoint:
            state_dict = checkpoint["params_ema"]
            print("Loaded 'params_ema' from checkpoint.")
        elif "params" in checkpoint:
            state_dict = checkpoint["params"]
            print("Loaded 'params' from checkpoint.")
        else:
            state_dict = checkpoint
            print("Loaded raw state_dict from checkpoint.")

        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)

    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, 256, 256)

    # Export to ONNX
    print(f"Exporting to {output_path}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "output": {0: "batch_size", 2: "height", 3: "width"},
            },
        )
        print("Export successful!")
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        sys.exit(1)

    # Verify ONNX model
    try:
        import onnx

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully.")
    except ImportError:
        print("Skipping verification (onnx package not installed).")
    except Exception as e:
        print(f"ONNX verification failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
