---
license: other
license_name: apache-2.0-with-llvm-exception
license_link: LICENSE
library_name: loom
pipeline_tag: text-generation
tags:
  - example
  - source-defined
---

# Tiny text, published with Loom

`example/tiny-text` is deterministic arithmetic, not a trained chatbot. It
accepts `a`, `b`, and `x`, uses `#` as EOS, and has one four-byte checkpoint
parameter. Prompt `abxa` with `max_tokens: 2` produces `bx`. The checkpoint is
small enough to embed in the model card; fetch materializes it as safetensors.
Execution uses real JIT-compiled kernels, pooled state, and the text service.

The [model card](model/tiny-text/model.json) selects text service ABI 2.
The [preparation program](model/tiny-text/program/prepare.loom) sizes one
`advance` stage for the token and span limits. The
[controller](model/tiny-text/program/control.loom) packs each submission
and runs that stage with its live buffers.

Token IDs are `a=0`, `b=1`, `x=2`, and `#=3`. Starting at zero, each token updates
`h = (5*h + token + 2) % 97` and `k = (7*k + h + position) % 101`, then appends
`k` to history and advances position. Prediction is `(h + sum(history)) % 3`,
or EOS every eleven consumed tokens. This is a small retained-state example,
not a performance proxy. Tools, reasoning, and images are not supported.

With the [Loom SDK](https://rocm.github.io/hrx-system/loom/getting-started/acquiring-loom/)
installed, `python smoke.py` checks the catalog and locks, fetches this model,
and checks its output on a private CPU server. `python refresh.py` explicitly
updates asset locks and the catalog after authoring changes. Both scripts have
`--help`; neither needs a source checkout or build system.

The [Hugging Face publishing guide](https://rocm.github.io/hrx-system/loom/serving/sharing/hugging-face/)
owns the sharing workflow, upload commands, and examples of external weights.
This repository owns its model programs, metadata, and output check.
