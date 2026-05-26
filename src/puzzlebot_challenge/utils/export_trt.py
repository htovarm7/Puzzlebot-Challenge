#!/usr/bin/env python3
"""
export_trt.py — Converts best.pt → best.engine (TensorRT FP16) for Jetson Nano.

Run once on the Jetson before launching sign_detector:
  python3 export_trt.py
  python3 export_trt.py --pt /path/to/best.pt --imgsz 320

The resulting best.engine lives next to best.pt and is auto-loaded by
sign_detector.py when present.

Requirements: JetPack 4.6+, ultralytics, tensorrt (included in JetPack image).
"""

import argparse
import os
import sys
import time


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    default_pt = os.path.join(here, "best.pt")

    p = argparse.ArgumentParser(description="Export YOLOv8 → TensorRT engine")
    p.add_argument("--pt",     default=default_pt, help="Path to best.pt")
    p.add_argument("--imgsz",  type=int, default=320, help="Inference image size")
    p.add_argument("--batch",  type=int, default=1,   help="Batch size (keep 1 for real-time)")
    p.add_argument("--no-fp16", dest="fp16", action="store_false",
                   help="Disable FP16 (use FP32 instead)")
    p.set_defaults(fp16=True)
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.pt):
        print(f"ERROR: weights not found: {args.pt}")
        sys.exit(1)

    engine_path = os.path.splitext(args.pt)[0] + ".engine"
    if os.path.exists(engine_path):
        print(f"Engine already exists: {engine_path}")
        ans = input("Re-export? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            sys.exit(0)

    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available. Run this script on the Jetson.")
            sys.exit(1)
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("ERROR: torch not installed.")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print(f"\nExporting {args.pt}")
    print(f"  imgsz={args.imgsz}  batch={args.batch}  fp16={args.fp16}")
    print("This can take 5-10 minutes on Jetson Nano — please wait...\n")

    model = YOLO(args.pt)
    t0 = time.time()
    model.export(
        format="engine",
        imgsz=args.imgsz,
        half=args.fp16,
        batch=args.batch,
        device=0,
        workspace=2,   # GB — Jetson Nano has 4 GB shared; keep workspace small
        verbose=True,
    )
    elapsed = time.time() - t0

    if os.path.exists(engine_path):
        size_mb = os.path.getsize(engine_path) / 1e6
        print(f"\nDone in {elapsed:.0f}s — {engine_path} ({size_mb:.1f} MB)")
        print("sign_detector.py will auto-load this engine on next launch.")
    else:
        print("\nERROR: export finished but .engine file not found.")
        print("Check ultralytics / TensorRT version compatibility.")
        sys.exit(1)


if __name__ == "__main__":
    main()
