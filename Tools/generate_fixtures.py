#!/usr/bin/env python3
"""Generate per-stage numerical-parity fixtures for the Swift MegaLoc port.

Runs the reference PyTorch model (gmberton/MegaLoc) on a fixed seeded input and
saves the input plus each intermediate stage to a safetensors file that the
Swift parity test loads and compares against.

Usage:
    uv run --with torch --with torchvision --with safetensors --with numpy \
        Tools/generate_fixtures.py \
        --repo /path/to/python/MegaLoc \
        --weights /path/to/model.safetensors \
        --out Tests/MLXMegaLocTests/Fixtures/parity_518.safetensors \
        --size 518
"""
import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Path to the MegaLoc python repo (has megaloc_model.py)")
    ap.add_argument("--weights", required=True, help="Path to model.safetensors")
    ap.add_argument("--out", required=True, help="Output safetensors fixture path")
    ap.add_argument("--size", type=int, default=518, help="Square input size (multiple of 14)")
    ap.add_argument("--height", type=int, default=0, help="Non-square height (overrides --size if >0)")
    ap.add_argument("--width", type=int, default=0, help="Non-square width")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    from megaloc_model import MegaLoc  # noqa: E402

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    model = MegaLoc()
    state = load_file(args.weights)
    model.load_state_dict(state)
    model.eval()

    if args.height > 0 and args.width > 0:
        h, w = args.height, args.width
    else:
        h = w = args.size
    assert h % 14 == 0 and w % 14 == 0, "H and W must be multiples of 14 to skip resize"

    # Fixed "already-preprocessed" input in NCHW. The model does not normalise
    # internally, so we feed the same raw tensor to both sides.
    images = torch.randn(1, 3, h, w, dtype=torch.float32)

    # Stage-by-stage forward (mirrors MegaLoc.forward without the resize branch).
    patch_features, cls_token = model.backbone(images)     # [1,768,Hp,Wp], [1,768]
    salad = model.aggregator.agg((patch_features, cls_token))  # [1, 16640]
    linear_out = model.aggregator.linear(salad)            # [1, 8448]
    final = model.l2norm(linear_out)                       # [1, 8448]

    # Sanity: whole-model forward must equal the staged final.
    whole = model(images)
    max_diff = (whole - final).abs().max().item()
    print(f"whole-vs-staged max abs diff: {max_diff:.2e}")

    tensors = {
        "input_nchw": images.contiguous(),
        "patch_features_nchw": patch_features.contiguous(),
        "cls_token": cls_token.contiguous(),
        "salad": salad.contiguous(),
        "linear_out": linear_out.contiguous(),
        "final": final.contiguous(),
        "meta_hw": torch.tensor([h, w], dtype=torch.int32),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out))
    print(f"Saved {len(tensors)} tensors -> {out}")
    for k, v in tensors.items():
        print(f"  {k}: {list(v.shape)} {v.dtype}")


if __name__ == "__main__":
    main()
