#!/usr/bin/env python3
"""Benchmark the PyTorch reference and generate synthetic 'place' sample images.

The samples are three scenes, each rendered from two slightly different
viewpoints (shift + brightness), so retrieval can be demonstrated: a query that
is a third variant of a scene should rank that scene's images highest.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file


def draw_scene(seed, shift=(0, 0), bright=1.0, size=322):
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (size, size), tuple(rng.integers(30, 90, 3).tolist()))
    d = ImageDraw.Draw(img)
    for _ in range(18):
        x = int(rng.integers(0, size)) + shift[0]
        y = int(rng.integers(0, size)) + shift[1]
        w = int(rng.integers(20, 90))
        h = int(rng.integers(20, 90))
        color = tuple(rng.integers(60, 255, 3).tolist())
        shape = rng.integers(0, 3)
        if shape == 0:
            d.rectangle([x, y, x + w, y + h], fill=color)
        elif shape == 1:
            d.ellipse([x, y, x + w, y + h], fill=color)
        else:
            d.line([x, y, x + w, y + h], fill=color, width=int(rng.integers(3, 12)))
    arr = np.clip(np.asarray(img).astype(np.float32) * bright, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def gen_samples(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for scene in range(3):
        for view, (shift, bright) in enumerate([((0, 0), 1.0), ((12, -8), 0.85)]):
            img = draw_scene(100 + scene, shift=shift, bright=bright)
            name = f"scene{scene}_view{view}.png"
            img.save(out / name)
            names.append(name)
    # A query: a third variant of scene 0.
    draw_scene(100, shift=(-6, 10), bright=1.1).save(out / "query_scene0.png")
    print(f"Wrote {len(names) + 1} sample images to {out}")


def bench(repo, weights, size, iters):
    sys.path.insert(0, repo)
    from megaloc_model import MegaLoc  # noqa: E402

    torch.set_grad_enabled(False)
    model = MegaLoc()
    model.load_state_dict(load_file(weights))
    model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    x = torch.randn(1, 3, size, size, dtype=torch.float32, device=device)

    for _ in range(3):
        model(x)
    if device == "mps":
        torch.mps.synchronize()

    t = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        if device == "mps":
            torch.mps.synchronize()
        t.append((time.perf_counter() - t0) * 1000)
    t.sort()
    mean = sum(t) / len(t)
    print(f"PyTorch [{device}] MegaLoc forward @ {size}x{size}, {iters} iters:")
    print(f"  mean={mean:.1f} ms  p50={t[len(t)//2]:.1f} ms  ({1000/mean:.1f} img/s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--samples-out", default="")
    ap.add_argument("--size", type=int, default=322)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    if args.samples_out:
        gen_samples(args.samples_out)
    bench(args.repo, args.weights, args.size, args.iters)


if __name__ == "__main__":
    main()
