#!/usr/bin/env python3
"""Dump per-block backbone token stages for bisecting Swift/PyTorch divergence."""
import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--fixture", required=True, help="parity_518.safetensors to reuse its input")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    from megaloc_model import MegaLoc  # noqa: E402

    torch.set_grad_enabled(False)
    model = MegaLoc()
    model.load_state_dict(load_file(args.weights))
    model.eval()

    fx = load_file(args.fixture)
    images = fx["input_nchw"]
    B, _, H, W = images.shape

    bb = model.backbone
    xp = bb.patch_embed(images)
    cls = bb.cls_token.expand(B, -1, -1)
    x = torch.cat((cls, xp), dim=1)
    emb = x + bb.interpolate_pos_encoding(x, H, W)

    out = {"emb": emb.contiguous()}
    h = emb
    for i, blk in enumerate(bb.blocks):
        h = blk(h)
        out[f"block_{i}"] = h.contiguous()
    out["post_norm"] = bb.norm(h).contiguous()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(out, args.out)
    print(f"Saved {len(out)} stage tensors -> {args.out}")


if __name__ == "__main__":
    main()
