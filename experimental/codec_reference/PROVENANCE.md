# Input and comparator provenance

The small synthetic fixtures are included in this branch. The large comparison
also used two external photographs whose bytes are not distributed here.
[The corpus receipt](evidence/jpeg-baseline/corpus-source.json) records their
revision-pinned download URLs, byte lengths, Git object IDs, and SHA-256 hashes:

| Input | Dimensions | JPEG sampling | SHA-256 |
| --- | --- | --- | --- |
| `006mp-cathedral.jpg` | 2000 × 3008 | 4:4:4 | `ee148ec6f6475538466004fe5fbb02d0b9146557715d85c6860ea8ff6aaeab74` |
| `012mp-bus.jpg` | 4032 × 3024 | 4:2:0 | `08eeaf6cf97e9d188efc2c2608d8b45f7fae2a2193f0c1c8913b84b4eb3e0e24` |

Both come from `nolmoonen/jpeg-test-images` at revision
`0b23d4b59353c31b2e2471a1c5b3f51e3b78e537`. Its source attribution identifies
the cathedral image with imagecompression.info's test images and the bus image
with Nol Moonen. Their original distribution carries their usage terms.

After retrieving and hash-checking those files, the preserved CPU program can
regenerate the synthetic 256 × 256 and 1024 × 768 controls and coefficient-exact
no-restart, restart-64, and progressive variants:

```sh
build_tools/bin/iree-bazel-run --config=opt \
  //experimental/codec_reference/cpu:jpeg_baseline -- \
  prepare /tmp/codec-reference-inputs \
  /tmp/006mp-cathedral.jpg /tmp/012mp-bus.jpg
```

Preparation verifies complete decoded pixels, guards, quantized coefficients,
and quantization tables against the source before reporting each variant. It
does not measure decoder throughput. The final persistent GPU experiment used
the `.r0.jpg` variants: each of the four distinct images appeared four times,
with eight queued replays. `research/jpeg-entropy/resize_fixture.py` in the
source bundle records the exact output geometries, filter plans, and poisoned
state. Historical staging scripts retain their original fleet and temporary
directory assumptions; the bounded small-fixture entry point in the README
is independent of that environment.

## CPU reference identity

The comparator uses the repository-pinned libjpeg-turbo 3.1.3 source. The
[BCR source receipt](evidence/jpeg-reference/bcr-source.json) and
[source identity receipt](evidence/jpeg-reference/source-identity.json)
identify the public reference without vendoring its implementation.
`evidence/jpeg-baseline/` retains the original full-frame CPU qualification,
input structure, raw sample rows, and summary. Those are separate measurements
from the final decode-plus-resize comparison in `RESULTS.md`.

The source bundle additionally preserves the original CPU comparator and API
probe bytes, alongside their analysis and staging scripts, under
`research/jpeg-baseline/` and `research/jpeg-reference/`. The readable `cpu/`
and `reference/` copies follow repository formatting. The manifest records
both source and archived hashes where historical path strings were relocated.
Binary-cost observations are historical linked-tool results, not a claim about
the size of a later main build or proof that this decoder removes a dependency.
