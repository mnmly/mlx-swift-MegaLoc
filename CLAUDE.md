# mlx-swift-MegaLoc

Swift/MLX port of MegaLoc (visual place recognition) — DINOv2 ViT-B/14 backbone
+ SALAD optimal-transport aggregator. Mirrors the reference `gmberton/MegaLoc`.

## Layout

- `Sources/MLXMegaLoc/` — the library. Module tree mirrors the checkpoint keys
  exactly (`MegaLoc.load` only casts dtype + transposes 4-D conv weights).
- `Tools/*.py` — parity fixture generators + benchmark (run with `uv`).
- `Tools/megaloc-cli/` — CLI (`download`, `embed`, `similarity`, `rank`, `bench`).
- `Examples/MegaLocExample/` — SwiftUI demo (xcodeproj), drives `MegaLocSession`.
- `Tests/MLXMegaLocTests/` — numerical parity (CPU tight, GPU cosine).

## Invariants

- **Build/test with `xcodebuild`, not `swift run`** — MLX needs the Metal
  toolchain. Scheme: `mlx-swift-MegaLoc-Package`.
- **Attention uses fused SDPA** — numerically equivalent to the reference's
  explicit softmax (verified); GPU differs from CPU only by fp32 accumulation.
- **All non-presentation logic lives in `MegaLocSession`** (shared driver) so
  the CLI and GUI can't drift. Descriptors are `Sendable [Float]` (no MLXArray).
- **Weights are never committed** — loaded from `~/.cache/huggingface`.

## Documentation

`MLXMegaLoc` ships DocC reference docs (see `Sources/MLXMegaLoc/Documentation.docc/`
and `Scripts/build_docs.sh`). **`///` comments on public symbols are published.**

When you add or modify a `public` symbol:

- Write a `///` comment (one-sentence summary; a paragraph if the *why* is
  non-obvious). Document each parameter with `- Parameter name:` (internal name).
- Cross-reference with double-backtick links, e.g. `` ``MegaLocSession/embed(image:)`` ``
  (signature-sensitive: `foo(_:)` ≠ `foo(_:_:)`).
- Add new top-level symbols under the right `## Topics` group in
  `Sources/MLXMegaLoc/Documentation.docc/MLXMegaLoc.md` (grouped by user task).

Verify: `BUILD_DOC=1 ./Scripts/build_docs.sh` (expect exit 0, no link warnings).
