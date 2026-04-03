#!/usr/bin/env python3
"""
Main CLI for portrait enhancement.

Examples:
1) Single image:
   python main.py --input examples/input.jpg --output examples/output.jpg

2) Multiple images:
   python main.py --input examples/a.jpg examples/b.jpg --output examples/outputs

3) Folder processing:
   python main.py --input examples/inputs --output examples/outputs
"""

import argparse
import os
from pathlib import Path
from PIL import Image
from pipeline.enhance import enhance_image


def parse_args():
    parser = argparse.ArgumentParser(description="Portrait -> Studio pipeline")

    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input image(s) or directory containing images"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output image path (single input) or directory (multiple inputs)"
    )

    parser.add_argument(
        "--face_restore_strength",
        type=float,
        default=0.8,
        help="0-1 float: strength of face restoration"
    )

    parser.add_argument(
        "--bokeh_strength",
        type=float,
        default=15.0,
        help="Gaussian blur radius for background"
    )

    return parser.parse_args()


def collect_images(inputs):
    image_paths = []

    for inp in inputs:
        p = Path(inp)

        if p.is_dir():
            image_paths.extend(
                list(p.glob("*.jpg")) +
                list(p.glob("*.jpeg")) +
                list(p.glob("*.png"))
            )
        elif p.is_file():
            image_paths.append(p)
        else:
            raise FileNotFoundError(f"Input not found: {inp}")

    if not image_paths:
        raise ValueError("No valid images found.")

    return image_paths


def main():
    args = parse_args()

    image_paths = collect_images(args.input)

    output_path = Path(args.output)

    # Case 1: single input → single output file
    if len(image_paths) == 1 and not output_path.is_dir():
        img = Image.open(image_paths[0]).convert("RGB")
        out = enhance_image(
            img,
            bokeh_strength=args.bokeh_strength,
            face_restore_strength=args.face_restore_strength
        )
        os.makedirs(output_path.parent or ".", exist_ok=True)
        out.save(output_path)
        print(f"Saved enhanced image to {output_path}")
        return

    # Case 2: multiple inputs → output must be a directory
    output_path.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        print(f"Processing: {img_path.name}")
        img = Image.open(img_path).convert("RGB")

        out = enhance_image(
            img,
            bokeh_strength=args.bokeh_strength,
            face_restore_strength=args.face_restore_strength
        )

        out_file = output_path / img_path.name
        out.save(out_file)
        print(f"Saved → {out_file}")

    print(f"\nProcessed {len(image_paths)} image(s). Output directory: {output_path}")


if __name__ == "__main__":
    main()
