#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def preprocess(img_path):
    print(f"Reading image from {img_path}...")
    # Read image using cv2 (BGR)
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # HWC to CHW
    img = img.transpose((2, 0, 1))

    # Add batch dimension: (1, 3, H, W)
    img = np.expand_dims(img, axis=0)

    return img


def postprocess(output_tensor, output_path):
    # Remove batch dimension
    output = output_tensor[0]

    # CHW to HWC
    output = output.transpose((1, 2, 0))

    # Clip to [0, 1]
    output = np.clip(output, 0, 1)

    # Scale to [0, 255] and convert to uint8
    output = (output * 255.0).round().astype(np.uint8)

    # RGB to BGR for cv2 saving
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    print(f"Saving output to {output_path}...")
    cv2.imwrite(str(output_path), output)


def main():
    parser = argparse.ArgumentParser(description="Test SPAN ONNX model.")
    parser.add_argument(
        "--model", type=str, default="model.onnx", help="Path to ONNX model"
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--output", type=str, default="output.png", help="Path to output image"
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    # Start Inference Session
    print(f"Loading model {model_path}...")

    available_providers = ort.get_available_providers()
    print(f"Available ONNX providers: {available_providers}")

    # prioritize CoreML (MPS/Neural Engine) if on Mac
    providers = []
    if "CoreMLExecutionProvider" in available_providers:
        providers.append("CoreMLExecutionProvider")
    providers.append("CPUExecutionProvider")

    try:
        session = ort.InferenceSession(str(model_path), providers=providers)
    except Exception as e:
        print(f"Error loading model with providers {providers}: {e}")
        print("Falling back to CPUExecutionProvider...")
        try:
            session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
        except Exception as e2:
            print(f"Fatal error loading model: {e2}")
            sys.exit(1)

    print(f"Active providers: {session.get_providers()}")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Model Input name: {input_name}")
    print(f"Model Output name: {output_name}")

    # Preprocess
    try:
        input_data = preprocess(args.input)
    except Exception as e:
        print(f"Error processing input: {e}")
        sys.exit(1)

    print(f"Input shape: {input_data.shape}")

    # Inference
    print("Running inference...")
    try:
        result = session.run([output_name], {input_name: input_data})
        output_data = result[0]
    except Exception as e:
        print(f"Inference failed: {e}")
        sys.exit(1)

    print(f"Output shape: {output_data.shape}")

    # Postprocess
    try:
        postprocess(output_data, args.output)
    except Exception as e:
        print(f"Error processing output: {e}")
        sys.exit(1)

    print("Done!")


if __name__ == "__main__":
    main()
