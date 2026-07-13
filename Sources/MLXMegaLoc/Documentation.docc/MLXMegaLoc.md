# ``MLXMegaLoc``

A Swift/MLX port of **MegaLoc** — "One Retrieval to Place Them All" — for
visual place recognition on Apple Silicon.

## Overview

MegaLoc turns an image into a single L2-normalised **descriptor** (8448-D). Two
images of the same place produce descriptors with high cosine similarity, so
retrieval is a nearest-neighbour search over descriptors.

The model is a DINOv2 ViT-B/14 backbone feeding a SALAD optimal-transport
aggregator and a linear head. This port mirrors the reference
(`gmberton/MegaLoc`) numerically: on the CPU device it matches PyTorch to
~1e-7, and on the Metal GPU the descriptors stay at cosine ≈ 0.9999 (the only
difference is fp32 accumulation order — see the parity tests).

The high-level entry point is ``MegaLocSession``: load a checkpoint once, embed
images, then rank cheaply.

```swift
import MLXMegaLoc

// Load the checkpoint (downloaded into ~/.cache/huggingface).
let url = MegaLocHub.cachedModelURL() ?? (try await MegaLocHub.download())
let session = try MegaLocSession.load(weights: url)

// Embed a database and a query, then rank by place similarity.
let db = session.embed(urls: databaseURLs)               // [(url, descriptor)]
let query = session.embed(image: queryCGImage)
let matches = MegaLocSession.rank(query: query, database: db.map(\.descriptor))
// matches[0].index is the most similar place.
```

## Topics

### Retrieval pipeline (start here)

- ``MegaLocSession``
- ``MegaLocDescriptor``
- ``MegaLocMatch``

### Acquiring the checkpoint

- ``MegaLocHub``

### Preprocessing

- ``MegaLocPreprocess``
- ``ImageNetNorm``

### The model

- ``MegaLoc``
- ``DINOv2Backbone``
- ``FeatureAggregator``
- ``Aggregator``

### Configuration

- ``MegaLocConfiguration``
- ``DINOv2Configuration``
