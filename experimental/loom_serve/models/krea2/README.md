# Krea 2 Turbo component port

This is a **component port, not an image-serving implementation**. The image
input projection and its official softwatercolor LoRA run through Loom's live
source JIT, queued safetensors loading, command programs, and shared device
ownership. The first transformer block's normalization and fused time
modulation, dense projections, fused Q/K normalization/rotation, masked
attention, gated residuals, and SwiGLU have independent component comparisons.
One complete base transformer block also runs as a single queued command and
matches its separately qualified native stages bit-for-bit. Full text
conditioning, all DiT layers, denoising, and VAE decoding are not yet
implemented here; block-level LoRA is not yet connected.

The independent [reference script](reference.py) runs the complete model using
PyTorch and Diffusers. Its images are reference outputs, not Loom outputs. It
also captures actual component tensors so the port can advance against real
values instead of judging correctness from a plausible final image.

## Pinned model inputs

The publisher calls this release **Krea 2**, with Raw and Turbo variants. This
port uses Turbo: eight steps, no classifier-free guidance, and the publisher's
Qwen-Image VAE. Source evidence:

- [Original model implementation](https://github.com/krea-ai/krea-2/tree/db3984fbc6e13b34c0064990fc2d95ac64d00058).
- [Turbo checkpoint](https://huggingface.co/krea/Krea-2-Turbo/tree/98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7).
- [Softwatercolor adapter](https://huggingface.co/krea/Krea-2-LoRA-softwatercolor/tree/12ff680b14a869b0e72f225bbdb237ad244b6a94).
- [Pinned Diffusers implementation](https://github.com/huggingface/diffusers/tree/8b33bfc04b6b5e8bb58a58e55f68746c1bbee4cd/src/diffusers/pipelines/krea2).

The Hugging Face account must have accepted the publisher's access terms and
authenticated with `hf auth login`. These commands download approximately
36.2 GB into one persistent checkpoint set; they exclude the alternate
Diffusers transformer shards, which would duplicate the main weights.

```sh
krea_weights=/path/to/models/krea-2-turbo
krea_adapter=/path/to/models/krea-2-softwatercolor
hf download krea/Krea-2-Turbo \
  --revision 98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7 \
  --local-dir "$krea_weights" \
  turbo.safetensors model_index.json transformer/config.json \
  text_encoder/config.json text_encoder/model.safetensors \
  tokenizer/tokenizer.json tokenizer/tokenizer_config.json \
  tokenizer/chat_template.jinja scheduler/scheduler_config.json \
  vae/config.json vae/diffusion_pytorch_model.safetensors
hf download krea/Krea-2-LoRA-softwatercolor \
  --revision 12ff680b14a869b0e72f225bbdb237ad244b6a94 \
  --local-dir "$krea_adapter" softwatercolor.safetensors
```

The weight-file SHA256 values are:

| File | SHA256 |
| --- | --- |
| `turbo.safetensors` | `78bbf8f4165eda19cea3cb06c78089221932a39e2eed8af9da741f942c47ffb3` |
| `text_encoder/model.safetensors` | `8434db05292f95e0041589a7c82abeb39385be59c85b54ae11caa7b45e9f4f13` |
| `vae/diffusion_pytorch_model.safetensors` | `ab1b61103959913d6c7e628cf793dbb2ca4726a40a3b3ae206c52b8e75bf6f08` |
| `softwatercolor.safetensors` | `3805e8655f19fbcac116542685e3f78f3a642e8fbfb857b5352bb32a4b3d445a` |

## Independent numerical reference

Install a PyTorch build supporting the reference GPU in a separate Python
environment. The qualified gfx1151 run used Python 3.14, the AMD
`torch[device-gfx1151]==2.13.0+rocm10.0.0` wheel from
`https://stable.repo.amd.com/rocm/whl-next/`, Transformers 5.10.1, PEFT 0.21.2,
Safetensors 0.8.0, Accelerate 1.15.0, NumPy 2.5.3, Pillow 12.3.0, and
Diffusers at the exact commit above. These are development dependencies;
the native Loom checker does not link or load them.

After installing the matching PyTorch wheel, the remaining environment can be
reproduced with:

```sh
python -m pip install \
  'diffusers @ git+https://github.com/huggingface/diffusers.git@8b33bfc04b6b5e8bb58a58e55f68746c1bbee4cd' \
  transformers==5.10.1 peft==0.21.2 safetensors==0.8.0 \
  accelerate==1.15.0 numpy==2.5.3 pillow==12.3.0 einops==0.8.2
krea_reference=/path/to/new-reference-output
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea2/reference.py \
  --checkpoint "$krea_weights" --adapter "$krea_adapter" \
  --output "$krea_reference"
```

The output directory must be new. At the default 384x384 size it retains about
110 MiB of images, component tensors, denoising steps, and raw BF16 projection
fixtures. The observed reference GPU allocation peak was approximately
33.4 GiB. That is a PyTorch reference measurement, not Loom's model residency.
Execution deliberately synchronizes for inspection and is not a benchmark.
Use the machine's normal exclusive-run mechanism when sharing a GPU.

The reference fixes initial packed noise rather than relying on matching
independent PRNG implementations. Its precision is explicit: GPU/BF16 text
encoder and DiT, then CPU/F32 Qwen-Image VAE. The selected Diffusers
`from_single_file` path casts all 430 DiT parameters, including norms, to BF16
despite the model class's keep-in-F32 declaration. The adapter loader similarly
casts all 528 F32 file tensors to BF16. The script records actual dtypes.
The CPU decoder provides a repeatable numerical oracle independently of GPU
convolution-library choices; it is not the intended Loom execution placement.

`base.png` and `zero.png` must match exactly; `style.png` must differ. Each
intermediate must be finite, and adapter use must preserve base parameter
storage and version counters. Hooks are attached after adapter injection so
the captured projection includes the adapter, not just its wrapped base layer.

## Run the Loom component

Build through the repository's configured build service, then run on a
qualified AMDGPU execution host with the checkpoint files and source tree:

```sh
build_tools/bin/iree-bazel-build --config=asan \
  //experimental/loom_serve:krea_projection_check
bazel-bin/experimental/loom_serve/krea_projection_check \
  --model=experimental/loom_serve/models/krea2 \
  --weights="$krea_weights/turbo.safetensors" \
  --adapter="$krea_adapter/softwatercolor.safetensors" \
  --input="$krea_reference/initial_latents.bf16" \
  --expected_base="$krea_reference/base-image_projection.bf16" \
  --expected_adapter="$krea_reference/style-image_projection.bf16"
```

The default matrix has 576 input rows, 64 input channels, and 6144 output
channels: 3,538,944 checked output elements per phase. The four phases are
base, zero-strength adapter, strength-one adapter, and base after adapter use.
The complete-array comparison allows one BF16 relative rounding interval plus
a small absolute cancellation floor; zero-strength and retained-base identity
are additionally bitwise checks. The first gfx1151 qualification matched the
reference bit-for-bit in all four phases, without using that tolerance. The
same executable also passed independent 16-row and 32-row specializations
bit-for-bit; a non-tiled 17-row request is rejected before device creation.

`--rows` JIT-specializes a positive multiple of 16. Input/reference files must
contain exactly that many rows. For an independent smaller specialization,
the first 16 rows of each fixture suffice because this projection has no
cross-row operation:

```sh
head -c 2048 "$krea_reference/initial_latents.bf16" > /tmp/krea-input-16.bf16
head -c 196608 "$krea_reference/base-image_projection.bf16" > /tmp/krea-base-16.bf16
head -c 196608 "$krea_reference/style-image_projection.bf16" > /tmp/krea-style-16.bf16
```

Use those three files with the same command and `--rows=16`. No native
executable rebuild is needed to change a source kernel or specialization.

## Source and ownership boundary

[`projection.loom`](projection.loom) owns model names, dimensions, and concrete
wrappers. [`kernels/linear.loom`](kernels/linear.loom) receives dimensions as
explicit SSA operands in templates. The wrappers fix those operands during
JIT specialization; shape scalars are not carried in the device launch ABI.
The WMMA layout contract is GFX11 wave32, including the fused bias epilogue.

The first projection is `X W^T + bias`. F32 checkpoint values round to BF16
before contraction; F32 accumulation and bias addition precede the single
BF16 output rounding. This explicitly models BF16 checkpoint loading. It is
not a claim that F32 and BF16 file bytes are interchangeable, or a decision to
convert every full-model weight on each invocation.

The adapter computes `base + strength * B(A X)` with the reference's BF16
rounding boundaries. Its rank is 32 and alpha/rank is one. Base and adapter
use separate immutable parameter roots and checkpoint loads. The loader reads
only the requested tensors, not the whole 36.2 GB set. Low-rank and delta
workspace comes from command reflection and is allocated once, not per run.
The native checker queues work through the existing execution domain and waits
only for its complete-array observations. Cleanup drains accepted work while
borrowed host payloads remain alive.

The remaining numerical gates include block-level LoRA, model conditioning,
and the entire source-JIT denoising and VAE path to a comparable image.
This component does not yet establish a generic model bootstrap, image request
scheduler, full-model weight-preparation strategy, or throughput result.

## Transformer-block component comparisons

Adding `--block-details` to the reference invocation captures raw tensors at
the first denoising step's block-zero boundaries, for both base and adapter
runs. `events.jsonl` records every file's shape and dtype. These intermediate
files are regenerable qualification inputs, not deployment assets; retain
them in temporary storage. The detailed 384x384 capture occupies about 660 MiB
in total.

At 384x384 the block processes 1088 rows: 512 text positions and 576 image
positions. The mask excludes padded text keys; it is not a causal mask.
`block0_norm1` exposes zero-centered RMSNorm. `block0_attention_input` fuses
that normalization with shared time modulation and the block's learned table.
The fused kernel retains the reference's BF16 rounding boundaries between
pointwise operations, while the normalization reduction accumulates in F32.

The model-independent, test-only `component_check` executes one command twice
and compares its entire BF16 output. Repeated `--input` arguments supply raw
buffers in command binding order; the output and optional reflected workspace
follow them. Repeated `--weights` paths supply checkpoints in reflected
parameter-root order. Each root is loaded from its own checkpoint domain;
equal tensor names in different domains remain independent. Parameter-free
commands need no checkpoint. This tool is not the image server.

```sh
build_tools/bin/iree-bazel-build --config=asan \
  //experimental/loom_serve:component_check
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea2/check_block.py \
  --checker bazel-bin/experimental/loom_serve/component_check \
  --model=experimental/loom_serve/models/krea2 \
  --checkpoint="$krea_weights/turbo.safetensors" \
  --reference="$krea_reference" --output=/path/to/new-component-results
```

This script uses the reference Python environment and requires exclusive use
of the GPU for its native subprocesses. It qualifies normalization against the
external oracle, then compares fusion **bitwise** against an independent CPU
pointwise calculation using that qualified normalization. It also proves the
pointwise calculation matches the external model using the external model's
normalization. Both ordinary and adapter-conditioned inputs are checked. The
same invocation qualifies the head normalization and rotary fusion below.

This separation matters: different valid F32 reductions can change one BF16
normalization step, and a subsequent subtractive shift can magnify relative
error near zero. The first 1088-row base run differed at 17 normalization
values and nine modulated values, with fused relative L2 error 5.54e-6.
Every modulation difference was explained by normalization, not the fusion.
The adapter-conditioned run differs at 27 normalization values and 15 modulated
values, with fused relative L2 error 8.18e-6; exact fusion equivalence passes
there too. A 16-row JIT specialization matches normalization bit-for-bit, and
the checker rejects a missing modulation input before weight loading.
A full-block/image comparison is still required; component equivalence is not
an end-to-end accuracy claim. Including head normalization and rotary checks,
the component result directory retains about 206 MiB of reproducible tensors.

For direct checker calls, `--root=block0_norm1` takes one input and
`base-block0-norm1.bf16` as its reference. `--config=key=value` supplies explicit
JIT specialization; no executable rebuild is involved. Tolerance defaults
match the projection check; `--atol=0 --rtol=0` requires exact numerical
equality. Every output must be finite at any tolerance. `--actual=path` writes
the first completed output before comparison, including on numerical failure,
so a mismatch can be investigated without rerunning the model oracle.

### Dense BF16 block projections

[`block_projections.loom`](block_projections.loom) maps the eight attention and
feed-forward matrices to four concrete contraction shapes. Weights remain in
their native BF16 checkpoint representation; accumulation is F32 and output is
BF16. The reusable contraction combines four independent accumulator chains
every 256 inputs to shorten cancellation-sensitive reductions. It does not
allocate global partial sums or copy/expand the weight matrices.

[`check_projections.py`](check_projections.py) compares every projection against
an independent CPU/F64 contraction, rounded once to BF16, at the full captured
token count and a separately JITed 16-row prefix. Each command runs twice.
The same BF16 error envelope applies to all shapes. The final feed-forward
projection uses the actual captured
`feed_forward_product` input, including the external model's SiLU/product
rounding. These are base-model comparisons: style projections include an
additional LoRA update and are not equivalent to the base matrix alone.

```sh
python -B experimental/loom_serve/models/krea2/check_projections.py \
  --checker bazel-bin/experimental/loom_serve/component_check \
  --model=experimental/loom_serve/models/krea2 \
  --checkpoint="$krea_weights/turbo.safetensors" \
  --reference="$krea_reference" --output=/path/to/new-projection-results
```

This driver uses the reference Python environment, with all Python tensor
operations on the CPU. The result directory contains about 260 MiB of
reproducible oracle/output tensors and small specialization inputs. F64 weights
exist only in this test process, not the native runner. Library-versus-F64
differences are reported separately. The shorter reductions were substantially
closer to rounded F64 even when they differed from the library at more elements;
matching the library's rounding errors is not the accuracy objective. Full-block
and image accuracy remain separate gates; this is not a throughput measurement
or a qualification of the attention operation described below.

The 1088-row and 16-row qualifications pass all eight projections twice:
133,398,528 output-element comparisons, with none outside the error envelope.
For the longest, 16384-term feed-forward contraction, full-size relative L2
error against rounded F64 is 2.19e-5. These checks cover every output value,
not a sampled subset or a small synthetic matrix.

### Fused Q/K normalization and rotary embedding

[`block_rotary.loom`](block_rotary.loom) exposes query/key normalization and
fused normalization-plus-rotation commands. The reusable helpers take explicit
shape operands: neither the normalization nor rotation helper reads model
configuration. The same row-normalization body handles 6144-wide token rows
and 128-wide attention heads.

One 64-thread workgroup owns each 128-wide head. Normalization rounds to BF16
before interleaved-pair rotation uses F32 cosine/sine tables and arithmetic;
the output then rounds to BF16. Query and key retain 48 and 12 heads
respectively. No KV-head replication or global normalized intermediate is
needed by the fused command.

`check_block.py` compares ordinary head normalization with independent F64
arithmetic at the unchanged normalization tolerance. It then requires the
fused result to match CPU rotation of that qualified normalization bit-for-bit,
including at an independently JITed 16-token image slice. The slice retains
its original position tables, so it exercises nonidentity rotations rather
than the all-zero text coordinates. Both full-size base and adapter-conditioned
inputs pass this exact fusion check.

The script also reports final error against rotation of F64-normalized values.
For full-size base queries, 53 values differ, six exceed the single-operation
BF16 envelope, and relative L2 is 7.51e-5. Exact fusion establishes that these
differences come from normalization rounding amplified by rotation, not from
changing the pointwise arithmetic. Complete-block and image qualification must
still establish the accumulated model-level error.

At 1088 tokens the fused Q/K path avoids writing and rereading 8,355,840 BF16
normalized elements: 31.875 MiB of intermediate traffic, plus two dispatches,
per block. This is traffic accounting, not a measured throughput improvement.

### Masked grouped-query attention

[`block_attention.loom`](block_attention.loom) exposes ungated attention and
attention with its sigmoid output gate. Both call the same config-free,
128-channel attention template. Four wave64 subgroups process 16 query rows
and successive 64-key tiles using LDS exchanges and F32 online-softmax state.
Query-head grouping directly selects the original KV head; there is no
KV-head replication or global attention-score matrix. A byte mask selects
valid keys, independently of query position. Every request must have at least
one valid key, which the model guarantees through its image tokens.

Some actual Krea heads have logits around 30,000 with much smaller differences
between keys. Computing those logits directly in F32 loses meaningful bits
before softmax subtraction. This implementation contracts `Q * (K - K_last)`;
the shared per-query offset cancels mathematically in softmax. Centered keys
and softmax probabilities each use a high BF16 value plus a BF16 residual.
Their native matrix products accumulate in F32, as does the running output.
No F64 device arithmetic or expanded global tensor is required.

```sh
python -B experimental/loom_serve/models/krea2/check_attention.py \
  --checker bazel-bin/experimental/loom_serve/component_check \
  --model=experimental/loom_serve/models/krea2 \
  --reference="$krea_reference" --output=/path/to/new-attention-results
```

The reference Python environment supplies independent CPU/F64 scores,
softmax, and value contractions from the captured normalized/rotated inputs.
No checkpoint argument is needed: these commands contain no fixed parameters.
The native tool runs on the leased GPU. The result directory retains about
180 MiB of regenerable tensors. The 16-row case uses image positions; the
80-row case starts with 64 masked keys and ends with a partial 16-key tile.
Both base and adapter-conditioned inputs also run at the full 1088 rows.

All 24 repeated comparisons pass, covering 58,195,968 output elements. Ungated
attention uses the unchanged BF16 envelope against F64. Gate fusion is bitwise
equivalent to CPU gating of that independently qualified attention output.
The script separately records composed F64 error: full-size gated relative L2
is 3.10e-5 for base inputs and 9.84e-5 for adapter-conditioned inputs. Eleven
base values and three style values exceed a single-operation envelope after
the additional BF16 rounding; exact gate equivalence accounts for the fusion.
The external model's corresponding relative L2 values are 1.13e-3 and 1.45e-3.
These numbers describe this component boundary, not full-model image quality
or execution performance.

### Residuals and feed-forward pointwise fusion

[`block_pointwise.loom`](block_pointwise.loom) supplies the two gated residual
updates and SiLU/up-product fusion. [`block.loom`](block.loom) shares one
normalization/modulation kernel between the pre-attention and pre-feed-forward
paths. Command-program subviews select the appropriate coefficient rows from
the shared modulation and learned parameter table. Selection requires neither
a copy nor a specialized kernel for each coefficient row.

```sh
python -B experimental/loom_serve/models/krea2/check_pointwise.py \
  --checker bazel-bin/experimental/loom_serve/component_check \
  --model=experimental/loom_serve/models/krea2 \
  --checkpoint="$krea_weights/turbo.safetensors" \
  --reference="$krea_reference" --output=/path/to/new-pointwise-results
```

The driver uses the reference Python environment. It independently qualifies
post-attention normalization and SiLU against F64, then requires the fused
affine and SwiGLU operations to match CPU composition of those qualified
outputs bit-for-bit. Both residual commands must match the CPU calculation
and the captured external model exactly. Base and adapter-conditioned inputs
run at 16 and 1088 tokens, twice each: 253,231,104 output-element comparisons
pass. This qualifies arithmetic on adapter-conditioned tensors, not the full
block's LoRA projections.

Full-size SiLU differs from rounded F64 at eight base values and three style
values out of 17,825,792 each, all inside the unchanged BF16 envelope. The
fused product preserves its intermediate BF16 rounding; removing that rounding
would implement different arithmetic. The test result directory contains about
500 MiB of reproducible tensors.

### Complete base transformer block

[`transformer.loom`](transformer.loom) composes the qualified operations into
`block0_forward`. Its caller supplies hidden states, shared time modulation,
F32 rotary tables, and the key mask. One immutable parameter root supplies the
block's weights. Command reflection plans one reusable transient slab; the
command contains 16 dispatches using ten distinct JITed kernels. Independent
Q/K/V/gate and feed-forward branches have explicit concurrent scopes. There
are no host waits, intermediate readbacks, or per-dispatch allocations inside
the block.

```sh
python -B experimental/loom_serve/models/krea2/check_transformer.py \
  --checker bazel-bin/experimental/loom_serve/component_check \
  --model=experimental/loom_serve/models/krea2 \
  --checkpoint="$krea_weights/turbo.safetensors" \
  --reference="$krea_reference" --output=/path/to/new-transformer-results
```

This driver requires the reference environment, detailed block capture, and
exclusive GPU execution. It runs an independent CPU/F64 block calculation
with the model's BF16 tensor boundaries and F32 rotary arithmetic. A separate
native chain checks each reduction against that arithmetic and each fusion
against composition of independently qualified operands. Finally the single
queued command must match the native chain bit-for-bit on both executions.
The result directory retains about 700 MiB of regenerable tensors.

That exact command-composition check passes at both 16 image rows and all
1088 captured rows. Against the independent complete-block calculation,
full-size relative L2 error is 0.00039353; the pinned external model's error
is 0.00107261. The complete block has 38,790 values outside the single-operation
BF16 envelope, versus 371,847 for the external block; those counts remain
visible because accumulated rounding is not a single-operation error bound.
The acceptance gate requires exact native composition and complete-block
relative L2 no worse than the external baseline. It does not loosen the
individual component tolerances. These are block-zero accuracy observations,
not a full-model image or performance result.
