# Krea 2 Turbo component port

This is a **native encoder-taps-to-RGB port, not an image-serving endpoint**.
The `sample_image` and `sample_image_adapted` roots run through Loom's live
source JIT, queued safetensors loading, command programs and shared device
ownership. Their caller supplies twelve Qwen3-VL hidden-state taps per text
token, initial packed noise, timesteps, rotary tables, a key mask, Euler deltas
and the VAE affine table. Qwen3-VL text encoding is still external; this is not
yet a native prompt-to-image path.

The command fuses the encoder taps into 2,560-wide text features and projects
them into the DiT's 6,144-wide prefix once. It batches time conditioning, then
runs eight denoising steps through image projection, all 28 transformer layers,
the velocity head and Euler updates. The complete native still-image VAE
decodes the packed final latent into clamped F32 RGB. Intermediate features,
conditioning and latents stay on device. Explicit phase lifetimes let one
planned workspace serve the whole command.

The official softwatercolor adapter covers fusion, text projection and DiT
projections. Turbo, adapter and VAE weights remain separate immutable domains;
fusion and denoising share the same Turbo and adapter images. Lower-level
command roots expose each stage independently for numerical qualification.
The checks below distinguish independent primitive arithmetic, exact native
composition, and accumulated image differences.

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
only the requested tensors, not the whole 36.2 GB set. The B contraction,
scaling, and addition are fused; no full-width delta is materialized. Rank-32
workspace comes from command reflection and is allocated once, not per run.
The native checker queues work through the existing execution domain and waits
only for its complete-array observations. Cleanup drains accepted work while
borrowed host payloads remain alive.

The remaining native producer is Qwen3-VL text encoding.
The denoising qualification below receives captured conditioning and uses an
external VAE for its preview images; the later image-command qualification
includes native text fusion, projection and VAE decoding.
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
`qualify.block_norm1` exposes zero-centered RMSNorm.
`qualify.block_attention_input` fuses that normalization with shared time
modulation and the block's learned table.
The fused kernel retains the reference's BF16 rounding boundaries between
pointwise operations, while the normalization reduction accumulates in F32.

The model-independent, test-only `component_check` executes one command twice
and compares its entire BF16 output. Repeated `--input` arguments supply raw
buffers in command binding order; the output and optional reflected workspace
follow them. Repeated `--weights` paths supply checkpoints in reflected
parameter-root order. Each root is loaded from its own checkpoint domain;
equal tensor names in different domains remain independent. Parameter-free
commands need no checkpoint. This tool is not the image server.

Component entry points live in the separate [`qualification/`](qualification/)
source catalog. The Python drivers select that catalog automatically; direct
`component_check` calls use `--model=experimental/loom_serve/models/krea2/qualification`.
Commands loading checkpoints also specify
`--weight_policy=experimental/loom_serve/models/krea2/weights.loom`, independently
of the source catalog, so qualification uses the deployment's actual preparation
policy rather than a copied or inferred policy.
Its `krea2.block_index` specialization defaults to zero. The model commands
receive an explicit compile-time layer index and format checkpoint keys from
it; native kernels are independent of layer identity. Qualification wrappers
are not part of the deployment catalog.

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

For direct checker calls, `--root=qualify.block_norm1` takes one input and
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
the shared per-query offset cancels mathematically in softmax. The online
maximum remains in unscaled dot-product units; subtraction precedes scaling
to preserve small logit differences. Centered keys use two BF16 terms, while
softmax probabilities use three. The three probability/value contractions
accumulate independently in F32, combining the two corrections before adding
the high term. This preserves small contributions through cancellation without
F64 device arithmetic or an expanded global tensor.

The probability tiles use a padded 18-element column stride in LDS. The
gfx1151 compile report records 25,600 bytes of workgroup storage, 128 vector
registers, 28 scalar registers, and no spills. This retains the original
two-term kernel's modeled occupancy tier; it does not establish equal runtime.

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
is 1.35e-5 for base inputs and 5.00e-5 for adapter-conditioned inputs. Two
base values and no style values exceed a single-operation envelope after
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
the layer-parameterized `krea2.block_forward` template, exposed for qualification
as `qualify.block_forward`. Its caller supplies hidden states, shared time
modulation, F32 rotary tables, and the key mask. One immutable parameter root supplies the
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

`--layer=0..27` selects a layer and its matching `base-blockN-*` or
`style-blockN-*` reference files. The default reference capture records layer
zero; another layer requires actual boundary tensors from that layer, not a
renamed layer-zero fixture.

[`reference_stack.py`](reference_stack.py) advances the captured first-step
inputs through the pinned external transformer's 28 layers and captures the
last layer. It loads neither the text encoder nor VAE and retains about 80 MiB
of temporary tensors at 384x384:

```sh
krea_stack_reference=/path/to/new-stack-reference
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea2/reference_stack.py \
  --checkpoint="$krea_weights" --adapter="$krea_adapter" \
  --reference="$krea_reference" --output="$krea_stack_reference"
```

Use `--layer=27 --reference="$krea_stack_reference/1088"` with the whole-block
driver for that actual final-layer input distribution. The separate `16/`
capture is a shape check that advances 16 image rows through all layers; it
does not represent an entire prompt/image request.

That exact command-composition check passes at both 16 image rows and all
1088 captured rows. Against the independent complete-block calculation,
full-size relative L2 error is 0.00038576; the pinned external model's error
is 0.00107261. The complete block has 37,803 values outside the single-operation
BF16 envelope, versus 371,847 for the external block; those counts remain
visible because accumulated rounding is not a single-operation error bound.
The acceptance gate requires independently qualified components and exact
native composition. Complete-block relative L2 remains a diagnostic; the
full-stack gate below measures accumulated error at its consumer boundary.
These are block-zero accuracy observations,
not a full-model image or performance result.

### Block LoRA projections

[`block_adapters.loom`](block_adapters.loom) applies the official adapter to
all eight projections of a selected block. Each adapted command binds separate
base and adapter parameter roots. The base contraction and rank-32 A contraction
have an explicit concurrent scope; the B contraction fuses BF16 rounding,
strength multiplication, and addition into the base output. Its same-tile
read/write permits in-place addition without a full-width delta buffer.
The zero-strength branch preserves the base instead of adding a rounded zero.

```sh
python -B experimental/loom_serve/models/krea2/check_adapters.py \
  --checker bazel-bin/experimental/loom_serve/component_check \
  --model=experimental/loom_serve/models/krea2 \
  --checkpoint="$krea_weights/turbo.safetensors" \
  --adapter="$krea_adapter/softwatercolor.safetensors" \
  --reference="$krea_reference" --output=/path/to/new-adapter-results
```

The independent CPU/F64 calculation checks the base, A, and B contractions.
Both fusion and the complete two-domain command then require bitwise agreement
with composition of those qualified native outputs. Actual base and
adapter-conditioned inputs run at 16 and 1088 rows, with strengths 0, 0.5,
and 1. All 576 repeated output comparisons pass: 2,135,506,944 element
comparisons, none outside their respective accuracy or exactness gates.
The output directory retains approximately 1.5 GiB of regenerable tensors;
one oracle file is reused to avoid retaining every strength's expected output.

At 1088 rows, the fused B epilogue removes 250.75 MiB of delta write/read
traffic and eight dispatches across the block's projections. Each adapted
projection needs only a 68 KiB rank-32 intermediate beyond its ordinary base
output. Those are storage/traffic counts, not measured performance gains.
The component path still reads the adapter's F32 file factors and rounds them
to BF16 in the contraction. Compact startup-prepared adapter storage requires
separate qualification before making full-model residency or performance
claims. These projection checks do not establish full-model LoRA image
generation.

### Complete transformer block with LoRA

`qualify.block_forward_adapted` adds a second immutable parameter root and an F32
strength input to the complete block. A shared command template receives
explicit shape and adapter-presence operands; concrete entry points own
configuration lookup. Source specialization removes the adapter path from
`qualify.block_forward`, retaining its original bindings and ten kernels. The
adapted variant uses 15 distinct kernels and 32 dispatches. Strength remains device
data, so changing it does not require recompilation or base-weight mutation.

The whole-block driver accepts the same adapter as the projection checker:

```sh
python -B experimental/loom_serve/models/krea2/check_transformer.py \
  --checker bazel-bin/experimental/loom_serve/component_check \
  --model=experimental/loom_serve/models/krea2 \
  --checkpoint="$krea_weights/turbo.safetensors" \
  --adapter="$krea_adapter/softwatercolor.safetensors" \
  --phase=style --strength=1 \
  --reference="$krea_reference" --output=/path/to/new-adapted-block-results
```

For the zero-strength identity case, use `--phase=base --strength=0` and a
different output directory. The driver checks each base/A/B contraction
against CPU/F64 arithmetic, requires every fused projection to reproduce
composition of qualified operands exactly, and compares the complete queued
block with that native chain bit-for-bit. Zero strength also must reproduce
the original base command exactly. Each phase runs at 16 and 1088 tokens,
twice per native command, retaining approximately 900 MiB of regenerable
tensors. Full-size accumulated error is measured against a separate CPU/F64
block alongside the corresponding external block.

Both phases pass all component and exact-composition gates. Zero strength
preserves the base block bit-for-bit at both shapes. For the 1088-token active
adapter case, relative L2 error against the independent block calculation is
0.00047224, versus 0.00246784 for the external model. Maximum absolute BF16
error is 4 versus 8; the single-operation envelope counts are 60,420 versus
661,300 and remain visible as accumulated block error. This qualifies one
LoRA-enabled block, not the full 28-block denoiser or an image.

The layer-27 fixture exercises actual final-layer inputs, including every
component, exact queued composition, and zero-strength identity at 16 and
1088 rows. Layer selection changes parameter keys, not numerical kernels:
the base and adapted commands still use ten and fifteen kernels.

### Complete transformer stack

[`stack.loom`](stack.loom) composes all 28 blocks into `transformer_stack` and
`transformer_stack_adapted`. Both receive the initial combined text/image
hidden states, shared time modulation, rotary tables, and key mask. The
adapted root additionally receives the adapter parameter root and strength.
These are transformer-stack boundaries, not prompt-to-image entry points.

The first block writes caller-owned output. Subsequent blocks advance it in
place: the block's final residual is its only output writer, and all original
input consumers finish before that write. Intermediate values remain separate
within each block. Command-program lifetime planning reuses that transient
storage across layers; the host neither loops over layers nor allocates their
intermediates. Layer identity changes parameter keys, not cached kernels.

[`check_stack.py`](check_stack.py) uses the captured first-step inputs from
`reference_stack.py`. It advances separate native and CPU/F64 chains. Each
native block is observed against independent arithmetic on its actual incoming
state, alongside the external block on that same state. These local rankings
are diagnostics: valid BF16 rounding can make one block locally farther from
F64 while the complete stack is closer. The native image rows consumed by the
final head must be no farther from
the independent F64 chain than the external image rows. All-row differences
remain diagnostics: the model discards text rows before its head, and padded
text rows are never valid attention keys. This accumulated gate supplements
the individual component and exact-fusion checks above.

```sh
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea2/check_stack.py \
  --checker bazel-bin/experimental/loom_serve/component_check \
  --model=experimental/loom_serve/models/krea2 \
  --checkpoint="$krea_weights" --adapter="$krea_adapter" \
  --reference="$krea_stack_reference" --image_rows=576 \
  --output=/path/to/new-stack-results
```

The single queued stack must equal the separate native chain byte-for-byte
on repeated executions. Zero-strength LoRA must equal the base stack exactly.
Reflection must show ten base or fifteen adapted kernels, exactly 28 unique
layers of parameters, and the same workspace size as a single block.
`--rows 16` selects the reduced-shape stress check; `--rows 1088` selects the
entire captured 384x384/text sequence. Neither run is a performance benchmark.
`--image_rows` supplies the consumer boundary explicitly; 384x384 images have
576 packed image rows. Reduced-shape checks use the smaller retained suffix.
`--phase=base` or `--phase=style` limits a diagnostic run to one phase.

The driver retains less than 200 MiB across both default shapes and phases,
overwriting its per-layer scratch tensors. Only this qualification process
loads an independent external model alongside the native model. The native
command uses one immutable copy of each required base and adapter tensor.

Both shapes pass repeated exact composition and zero-strength identity. At
1088 rows, the 576 image rows have these relative L2 errors against the
independent F64 chain:

| Phase | Loom | External implementation | Transient workspace |
| --- | ---: | ---: | ---: |
| Base | 0.87358% | 0.90045% | 149.8125 MiB |
| Softwatercolor, strength 1 | 1.49491% | 1.62224% | 150.078125 MiB |

The base stack references 364 unique parameter tensors and ten kernels;
the adapted stack references 812 tensors across two checkpoint roots and
fifteen kernels. Each tensor has one storage range, and workspace matches a
single block exactly. These are first-step transformer-state measurements,
not denoised-image quality or server throughput.

For explicit whole-block numerical checks, `component_check` accepts a
positive `--relative_l2_tolerance` to select an aggregate error gate. Zero
retains the default elementwise gate. Its JSON records both the selected
comparison and element-envelope differences, and nonfinite pairs fail in
either mode. Exact composition uses zero absolute and relative elementwise
tolerances, followed by a byte comparison in the Python driver. Local stack
observations use `--report_only`: both executions report their differences
without an error bound, while nonfinite pairs and execution failures still
fail. The driver applies the unchanged accumulated bound at the image-row
consumer boundary.

### Final velocity head

[`head.loom`](head.loom) converts the final image-only hidden states into
64-channel flow velocities. `velocity_head` accepts a base parameter root,
image states, one 6144-value time embedding, and caller-owned output.
`velocity_head_adapted` additionally accepts the adapter root and device
strength. These commands consume the image suffix, not the discarded text
prefix of the combined transformer sequence.

The first kernel fuses zero-centered RMS normalization with learned affine
modulation, retaining each BF16 rounding boundary. The same embedding feeds
both scale and shift; no expanded modulation tensor is needed. The second
kernel performs the biased 6144-to-64 projection. The adapted command overlaps
the base and rank-32 A projections, then fuses B with scaling and in-place
addition. All arithmetic comes from the same config-free normalization and
projection helpers used by the transformer blocks. Base weights stay immutable.

At 576 image rows, the base head needs 7,077,888 bytes of transient storage
and two kernels. LoRA adds only 36,864 bytes and two kernels, with no
full-width delta buffer. The source command planner owns those lifetimes.

[`check_head.py`](check_head.py) consumes the external stack captures and both
native stack-result directories. It reconstructs the first-step external time
embedding and requires its modulation to match the retained capture exactly.
That embedding is a supplied input, not native time conditioning. Separate
qualification kernels expose normalization and the B projection; production
keeps them fused with modulation and addition respectively.

```sh
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea2/check_head.py \
  --checker bazel-bin/experimental/loom_serve/component_check \
  --model=experimental/loom_serve/models/krea2 \
  --checkpoint="$krea_weights" --adapter="$krea_adapter" \
  --reference="$krea_stack_reference/1088" \
  --base_stack=/path/to/stack-results/base-1088 \
  --style_stack=/path/to/stack-results/style-1088 \
  --image_rows=576 --output=/path/to/new-head-results
```

Both external and native incoming states pass at 16 and 576 image rows,
with base and active adapter conditioning. The 144 repeated comparisons cover
60,166,144 values with zero primitive-envelope violations or nonfinite pairs.
Fused modulation, adapter addition, whole-head composition, and zero strength
are bitwise checks. Head-local relative L2 on full native inputs is 0.00384%
base and 0.00389% LoRA against independent F64 arithmetic with BF16 boundaries.

The complete native stack followed by this head has velocity errors of
0.57460% base and 0.98863% LoRA against the independent full-chain oracle;
the corresponding external results are 0.60111% and 0.88059%. Hidden-state
error rankings therefore do not imply the same velocity ranking. These are
first-step numerical observations, not image-quality or throughput claims.
The driver retains less than 128 MiB of regenerable head tensors.

### Queued stack-to-head composition

[`velocity.loom`](velocity.loom) exposes `transformer_velocity` and
`transformer_velocity_adapted`. Their inputs are the stack inputs followed by
the time embedding, optional adapter strength, and caller-owned velocity
output. The combined token count must be at least the image token count.
The combined hidden state is transient storage; a typed view at the image
suffix binds directly into the head's first kernel. No suffix copy, text-row
projection, or intermediate host synchronization is needed.

[`check_velocity.py`](check_velocity.py) compares this single command with the
already-qualified native outputs from `check_head.py`, not with a newly chosen
numerical tolerance:

```sh
python -B experimental/loom_serve/models/krea2/check_velocity.py \
  --checker bazel-bin/experimental/loom_serve/component_check \
  --model=experimental/loom_serve/models/krea2 \
  --checkpoint="$krea_weights" --adapter="$krea_adapter" \
  --reference="$krea_stack_reference/1088" \
  --head_results=/path/to/head-results --image_rows=576 \
  --output=/path/to/new-velocity-results
```

Base, zero strength, and active LoRA all match byte-for-byte on both native
executions at 1088 combined rows and 576 image rows. Standalone head
qualification retains all 144 passing comparisons after the composition
refactor. The base command shares 12 kernels and 368 unique parameters;
LoRA shares 19 kernels and 818 unique parameters. Peak transient storage is
162.5625 MiB base and 162.828125 MiB adapted: exactly single-block scratch
plus the combined hidden state. Head temporaries reuse expired block storage.
The driver retains only three final velocity tensors, less than 1 MiB.

### Batched time conditioning

[`time.loom`](time.loom) exposes `time_conditioning` and
`time_conditioning_adapted`. The base root accepts a checkpoint root, a BF16
time vector, and caller-owned output. The adapted root additionally accepts
the adapter checkpoint and an F32 strength buffer. `krea2.time_count` is the
logical vector length, from 1 through 256; the default is Turbo's eight steps.
The values are the scheduler timesteps divided by its training-step count,
rounded to BF16 as in the model pipeline.

All known times are processed together. The command computes 256 sinusoidal
features, the two-layer GELU time MLP, and the GELU/modulation projection.
The three base matrices remain in their F32 checkpoint storage and round to
BF16 in the shared contraction helper. LoRA covers all three projections;
base/A contractions may overlap, and B/scaling/addition is fused in place.
The model leaves bind exact shapes; mathematical helpers receive SSA operands
and contain no model configuration dependencies.

The physical row count is `ceil(time_count / 16) * 16`. Padding has a defined
zero time and never reads beyond the logical input. Output contains the whole
physical `rows × 6144` BF16 embedding table followed by the whole
`rows × 36864` BF16 modulation table. A step consumer binds its row from each
table; padding is storage, not an additional denoising step. No per-layer
embedding replication is needed.

```sh
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea2/check_time.py \
  --checker bazel-bin/experimental/loom_serve/component_check \
  --model=experimental/loom_serve/models/krea2 \
  --checkpoint="$krea_weights" --adapter="$krea_adapter" \
  --reference="$krea_stack_reference/1088" \
  --output=/path/to/new-time-results
```

This qualification uses the reference environment and exclusive GPU execution.
It derives the eight actual Turbo scheduler times and exercises logical counts
1, 8, 16, and 17, including the physical-tile boundary. Independent F64
arithmetic checks sinusoidal features, GELU, and each base/A/B contraction.
Fused adapter addition, complete native composition, and zero-strength identity
are bitwise gates. The independent external first-step modulation must still
match the retained stack capture exactly. Accumulated accuracy reports select
logical rows; primitive and exact-composition checks include all physical
padding. The driver retains about 75 MiB of regenerable tensors.

All 216 repeated comparisons pass over 63,994,880 values, with no primitive
envelope violations or nonfinite pairs. For eight times, the base command has
five kernels, six unique parameters, and 392 KiB of transient workspace.
LoRA uses nine kernels, twelve parameters across two roots, and 393 KiB of
workspace. The denoising command below consumes these native tables directly.

### Native denoising trajectory

[`denoise.loom`](denoise.loom) exposes `denoise` and `denoise_adapted`.
The deterministic Euler trajectory is one source-JIT command: time tables
are computed once, and all requested steps execute without a host step loop
or intermediate readback. The first step reads immutable initial noise;
later steps update caller-owned output in place. This is the Turbo path,
without classifier-free guidance, stochastic sampling, or per-token schedules.

The base command accepts a parameter root followed by these buffers:

| Buffer | Representation |
| --- | --- |
| Initial sample | `image_tokens × 64` BF16 packed latent values |
| Text prefix | `(block_tokens - image_tokens) × 6144` BF16 conditioned rows |
| Times | `time_count` BF16 normalized timesteps |
| Cosine and sine | Two `block_tokens × 128` F32 rotary tables |
| Key mask | `block_tokens` bytes, with nonzero values marking live keys |
| Deltas | `time_count` F32 Euler step deltas |
| Output | `image_tokens × 64` BF16 packed latent values |

The adapted command adds the immutable adapter root immediately after the
base root, and an F32 strength buffer immediately before output. Both token
counts are multiples of 16, with the combined count at least the image count.
`time_count` selects the command's step count; times and deltas are device
inputs, not values baked into its kernels.

Each step copies only the immutable text prefix and projects the current
latent directly into the combined state's image suffix. These independent
producers join before the stack advances the state in place. The head binds
the image suffix directly. Exact byte offsets select the step's embedding
and modulation rows from the shared table; neither is copied or read back.
The update keeps the F32 delta unrounded, rounds `delta × velocity` to BF16,
then adds to the F32-upcast sample and rounds the result to BF16. Removing
either BF16 boundary changes the model scheduler's semantics.

[`check_denoise.py`](check_denoise.py) uses the reference environment and
previously qualified time tables. It builds the independent staged native
trajectory, checks the first and last composed steps bit-for-bit, then
requires both executions of the complete command to match that trajectory
bit-for-bit. Zero-strength LoRA must preserve the complete base trajectory.
Base, A, and B image contractions each use the unchanged F64 primitive gate;
B receives the actual rounded native A output. The fused image projection
must then equal addition of the qualified native base and B values exactly.
A primitive's tolerance is not an accumulated-chain tolerance near cancellation.
External velocities on identical incoming states and retained external
trajectories are reported separately; numerical drift is not confused with
composition equality.

```sh
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea2/check_denoise.py \
  --checker bazel-bin/experimental/loom_serve/component_check \
  --model=experimental/loom_serve/models/krea2 \
  --checkpoint="$krea_weights" --adapter="$krea_adapter" \
  --reference="$krea_reference" \
  --stack_reference="$krea_stack_reference/1088" \
  --time_results=/path/to/time-results \
  --output=/path/to/new-denoise-results
```

The default runs all eight steps for base and active LoRA. `--steps=N`
selects a prefix of the same eight-step schedule, not a newly configured
N-step scheduler; `--phase=base` or `--phase=style` narrows the qualification.
All eight steps additionally produce `*-native-denoise-external-vae.png`
previews and pixel RMSE/PSNR against the retained reference. Those previews
use the CPU/F32 external VAE and captured external text conditioning. They
are not evidence of native text encoding, VAE execution, image serving, or
throughput. Regenerable trajectory artifacts occupy less than 128 MiB.

The full 384x384 qualification passes 126 repeated component/composition
comparisons over 228,483,072 values, with no primitive-envelope violations
or nonfinite pairs. The eight-step command matches the staged native chain
bit-for-bit for base, zero strength, and active LoRA on both executions.

| Eight-step command | Base | Softwatercolor, strength 1 |
| --- | ---: | ---: |
| Unique kernels | 20 | 33 |
| Unique parameter tensors | 376 | 834 |
| Parameter roots | 1 | 2 |
| Parameter bytes | 25,382,416,640 | 25,821,294,848 |
| Transient workspace bytes | 172,310,528 | 172,311,552 |
| Final latent relative L2 versus external trajectory | 6.92078% | 4.46478% |
| Preview pixel RMSE, 8-bit channels | 8.26228 | 9.32944 |
| Preview PSNR | 29.78880 dB | 28.73369 dB |

Transient storage is independent of step count; active LoRA adds only 1 KiB
because its low-rank scratch fits already available lifetime gaps. The preview
statistics use the same external CPU/F32 decoder on both trajectories. For the
retained seed-42 deer prompt, the native previews preserve the composition
and the adapter's visibly different rendering, but are not pixel-identical.
This single prompt is a numerical and ownership witness, not a distributional
image-quality evaluation or a performance measurement.

### F32 VAE input path

[`vae.loom`](vae.loom) exposes `vae_input`, the first three operations of the
still-image decoder: unpack/affine, post-quant 1x1 convolution, and the initial
3x3 convolution. It receives an immutable VAE parameter root, packed BF16
latent, a `2 × 16` F32 affine table, and caller-owned F32 output. The affine
table contains inverse standard deviations followed by means from the pinned
VAE configuration. `krea2.latent_height` and `krea2.latent_width` specialize
even spatial dimensions; their defaults are 48 for a 384x384 image.

The config-free [convolution helper](kernels/convolution.loom) consumes NCHW
activations and original F32 OI(T)HW weights. A fresh causal frame selects the
last temporal weight plane; this is not a video-history implementation.
Nearest-neighbor upsampling can be folded into its input indexing without an
expanded image buffer. Concrete kernel leaves resolve geometry before applying
the helper, so no shape scalars enter the device ABI.

```sh
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea2/check_vae_input.py \
  --checker bazel-bin/experimental/loom_serve/component_check \
  --model=experimental/loom_serve/models/krea2 \
  --checkpoint="$krea_weights" \
  --denoise_results=/path/to/denoise-results \
  --output=/path/to/new-vae-input-results
```

The driver checks 1x1/3x3 spatial kernels, one/three temporal planes, and
1x/2x spatial scales at odd and one-row shapes. Actual base and LoRA latents
exercise the original checkpoint at 2x6 and 48x48. Unpacking and native
composition are bitwise gates; contractions compare with independent F64
arithmetic rounded to F32. All 64 repeated comparisons pass over 7,476,200
values, with no envelope violations or nonfinite pairs.

The full input command has three kernels, four unique parameters occupying
666,368 bytes, and 294,912 bytes of transient workspace. Relative L2 against
the pinned CPU/F32 prefix is 2.16e-7 for base and 2.14e-7 for LoRA; maximum
absolute error is 8.35e-7 for both. This establishes the native decoder input
boundary, not residual blocks, spatial attention, upsampling stages, or RGB
output. Regenerable qualification tensors occupy less than 64 MiB.

### Fused VAE residual and in-place prefix

[`vae_residual.loom`](vae_residual.loom) adds the first middle residual block.
Channel normalization clamps the L2 norm at `1e-12`, multiplies by
`sqrt(channels)` and learned gamma, and fuses SiLU without a normalized
intermediate tensor. This is distinct from the DiT's zero-centered RMSNorm.
The last convolution retains the F32 bias rounding before adding the skip.

The residual uses two feature-sized temporaries. Its second normalization is
in-place, and its final write may alias the original residual because every
earlier original-state reader has completed. `vae_input_residual` exercises
that ownership by running the native input path and advancing the caller's
feature buffer in place, in one command.

```sh
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea2/check_vae_residual.py \
  --checker bazel-bin/experimental/loom_serve/component_check \
  --model=experimental/loom_serve/models/krea2 \
  --checkpoint="$krea_weights" --input_results=/path/to/vae-input-results \
  --output=/path/to/new-vae-residual-results
```

At 2x6 and 48x48, both base/LoRA feature inputs pass all 56 repeated
comparisons over 24,901,632 values. Primitive normalization/convolution checks
use independent F64 arithmetic; fused skip addition, the residual command, and
the entire in-place prefix match staged native output bit-for-bit. The full
prefix has six unique kernels, ten parameters occupying 32,523,008 bytes, and
7,372,800 bytes of transient workspace. The residual itself needs 6.75 MiB.

For the 48x48 first residual, relative L2 against the independent F64 chain
is 8.57e-7 base and 8.75e-7 LoRA, below the fixed 2e-5 accumulated bound.
The corresponding CPU/F32 reference is closer to F64; native accuracy here
is sufficient for the bounded block gate, not a claim of superior precision.
The input-path regression retains all 64 passing comparisons and unchanged
feature bits. Spatial attention, the remaining residual/upsampling stages,
and final RGB remain outside this prefix. Residual fixtures occupy less than
192 MiB and can be regenerated from the retained input results.

### Bounded-memory VAE spatial attention

[`vae_attention.loom`](vae_attention.loom) composes channel normalization,
QKV projection, single-head noncausal spatial attention, and a fused output
projection/skip. Its F32 [online attention helper](kernels/image_attention.loom)
handles 1–512 channels with 128-key tiles. Each workgroup retains one query's
output accumulators and a 512-byte probability tile, plus reduction storage.
It does not materialize the 20.25 MiB score matrix for a 48x48 latent grid.

The standalone command uses 13.5 MiB of global scratch at 48x48: QKV plus one
feature plane reused between normalization and attended output. Allocation
points express those lifetimes. The final projection may write the original
input in place after QKV has consumed it. `vae_input_attention` exercises this
ownership through the complete native input/residual/attention prefix.

```sh
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea2/check_vae_attention.py \
  --checker bazel-bin/experimental/loom_serve/component_check \
  --model=experimental/loom_serve/models/krea2 \
  --checkpoint="$krea_weights" --input_results=/path/to/vae-input-results \
  --residual_results=/path/to/vae-residual-results \
  --output=/path/to/new-vae-attention-results
```

The driver checks 45 shape combinations crossing key-tile and channel-tail
boundaries, then real base/LoRA features at 2x6 and 48x48. All 146 repeated
comparisons pass over 34,655,004 values with unchanged 2e-5 primitive absolute
and relative tolerances and no nonfinite pairs. Primitive attention uses an
independent F64 softmax/contraction; fused addition, command composition, and
the full in-place prefix match staged native results bit-for-bit.

At 48x48, attention-block relative L2 against the independent F64 chain is
3.58e-7 base and 3.72e-7 LoRA. The complete prefix uses ten unique kernels,
15 parameters occupying 34,889,984 bytes, and 17,989,632 bytes of scratch.
The preceding residual's 56-comparison regression retains identical output
bits for every stage. These are correctness and memory results, not throughput
measurements. Attention fixtures occupy less than 256 MiB; remaining residual
and upsampling stages must still produce the final native RGB consumer.

### Complete native still-image decoder

[`vae_decoder.loom`](vae_decoder.loom) exposes `vae_decode`, from packed BF16
latent and the affine table to NCHW F32 RGB in `[-1, 1]`. Its single command
composes the qualified prefix, second middle residual, all four upsampling
stages, output normalization/convolution, and clamping. An image is eight
times the latent height and width. The caller supplies immutable VAE weights,
latent and affine buffers, plus a distinct output buffer.

The [upsampling leaves](vae_upsample.loom) specialize the shared config-free
convolution and normalization motifs at each spatial/channel shape. Three
residuals share kernels within each stage. At the 192-to-384 transition, the
learned shortcut writes the destination concurrently with input normalization;
later convolutions add into that destination. Equal-width blocks advance
source-owned features in place. Spatial nearest-neighbor expansion is part of
convolution indexing and never materializes an enlarged input image.

The fresh-frame contract excludes video history. The original temporal weight
planes remain in immutable storage, but only the last causal plane is read.
The two temporal-upsampling convolutions are absent from the command and its
parameter residency; the pinned reference skips them for the first frame.

```sh
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea2/check_vae_decoder.py \
  --checker bazel-bin/experimental/loom_serve/component_check \
  --model=experimental/loom_serve/models/krea2 \
  --checkpoint="$krea_weights" --input_results=/path/to/vae-input-results \
  --attention_results=/path/to/vae-attention-results \
  --output=/path/to/new-vae-decoder-results
```

Qualification runs base and active-LoRA latents at 2x6 and 48x48. Primitive
resize, output normalization and convolution use F64 oracles with 2e-5
absolute/relative bounds. Entire residual chains use a fixed 2e-5 relative-L2
bound. Clamping and both composed decoder commands must equal separately
executed native stages bit-for-bit on both executions. The complete RGB result
also has a fixed 2e-5 relative-L2 bound against the pinned CPU/F32 VAE on the
same latent. All 168 repeated comparisons pass over 469,573,632 values, with
no element-envelope violations or nonfinite pairs.

| 384x384 decoder result | Base | Softwatercolor latent |
| --- | ---: | ---: |
| F32 relative L2 versus CPU VAE | 1.80376e-6 | 3.81027e-6 |
| Different 8-bit channel values, out of 442,368 | 19 | 47 |
| Maximum 8-bit difference | 1 | 1 |
| Pixel RMSE | 0.00655368 | 0.01030759 |

The complete command has 30 unique kernels and 104 parameters occupying
286,100,492 bytes. Transient workspace is 219,709,440 bytes (209.53 MiB) at
384x384. These are correctness and memory observations, not performance data.
The images use the earlier native eight-step latents and external text
conditioning. The decoder does not apply LoRA itself; it receives the latent
produced by the adapted DiT.

The driver overwrites one active primitive fixture and retains final native
and reference RGB/PNGs. Its temporary storage stays below 256 MiB at these
shapes, and a failed check retains its exact inputs, command and expected
output. Native text encoding and the image-serving adapter are separate
remaining integration boundaries.

### One native encoder-taps-to-RGB command

[`sample.loom`](sample.loom) exposes `sample_image` and `sample_image_adapted`.
Both consume initial packed noise, BF16 encoder taps of shape
`[text_tokens, 12, 2560]`, timesteps, rotary tables, mask, Euler deltas and the
VAE affine table. The adapted root also consumes a scalar LoRA strength.
Fusion and text projection run once; the 6,144-wide projected output stays on
device for all denoising steps. Fusion consumes the text prefix of the combined
key mask directly, without a copy. Both roots produce one distinct NCHW F32
RGB buffer. Fused features, projected text and the packed final latent are
source-owned intermediates that never return to the host. Checkpoint roots
are ordered Turbo/VAE or Turbo/LoRA/VAE; fusion shares the same Turbo and
adapter images as the DiT.

The model entry checks its geometry through template selection: image-token
count must equal `(latent_height / 2) * (latent_width / 2)`, and combined token
count must cover the image suffix, and `text_tokens` must equal that remaining
prefix length. Invalid relations reject compilation before weight loading.
Neither a host graph walker nor a command-program ABI extension is involved.

The exact composition check first projects the qualified native fusion results,
then executes denoising and decoding separately. The combined command starts
with raw encoder taps and must reproduce that staged RGB exactly. The driver
also verifies that the combined mask's prefix equals the encoder mask. It needs
NumPy, Pillow and the native checker, but loads no external model framework:

```sh
python -B experimental/loom_serve/models/krea2/check_sample.py \
  --checker bazel-bin/experimental/loom_serve/component_check \
  --model=experimental/loom_serve/models/krea2 \
  --checkpoint="$krea_weights" --adapter="$krea_adapter" \
  --denoise_results=/path/to/denoise-results \
  --stack_reference=/path/to/stack-reference/1088 \
  --input_results=/path/to/vae-input-results \
  --decoder_results=/path/to/vae-decoder-results \
  --text_results=/path/to/text-projection-results \
  --fusion_results=/path/to/text-fusion-results \
  --reference=/path/to/reference-results \
  --output=/path/to/new-sample-results
```

This is an exact native-composition oracle. The preceding stage checks supply
the independent numerical evidence; matching their output does not replace
those checks or establish performance. Differences from the earlier
external-text trajectory are reported separately: changed conditioning makes
those different computations, not an exact-composition oracle. Text encoding
is still external, so the command is not a complete native prompt-to-image
implementation.

Base, zero-strength LoRA and active LoRA pass twice, with zero differing bits
over 2,654,208 F32 values. Both roots reject all three invalid geometry cases,
six rejections in total. Twelve additional staged comparisons report numerical
differences from the earlier external-text inputs; those are not exact gates.

| Complete 384x384 command | Base | LoRA |
| --- | ---: | ---: |
| Unique kernels | 71 | 95 |
| Parameter roots | 2 | 3 |
| Parameters | 534 | 1,062 |
| Parameter bytes | 26,569,389,580 | 27,038,612,748 |
| Workspace bytes | 276,299,776 | 277,872,640 |

At 384x384 with 512 text rows, fused features occupy 2.5 MiB, the retained
projected prefix is 6 MiB and the packed latent is 72 KiB. These lifetimes do
not all overlap. The whole command's high-water mark is 263.5 MiB base or
265 MiB adapted, exactly 2.5 MiB above standalone fusion. Projection, denoising
and decoding reuse retired fusion scratch instead of adding their independent
workspaces. This measures planned storage reuse, not reduced computation.
The driver retains less than 96 MiB of taps, conditioning, latent and RGB
evidence, with no copied checkpoints.

Small conditioning-rounding differences can grow through a denoising trajectory.
On the captured prompt, the new image's pixel RMSE versus the earlier native
image with external text conditioning is 7.56 base and 11.16 LoRA; versus the
complete external reference it is 9.30 and 9.75. Visual inspection preserves
the deer/forest composition and watercolor style, with local detail changes.
One prompt does not establish distribution-level image quality. These
observations complement, rather than replace, the independent primitive checks.

### Native text projection

[`text_projection.loom`](text_projection.loom) contains `text_projection` and
`text_projection_adapted`. Their input is a text-fusion result,
not token IDs or Qwen hidden-state taps. The command applies zero-centered
RMSNorm at width 2,560, a biased 2,560-to-6,144 projection, tanh GELU, and a
biased 6,144-to-6,144 projection. Both dense layers support the official
rank-32 LoRA factors. The arithmetic uses the existing config-free motifs;
concrete leaves resolve `krea2.text_tokens` and fixed model widths.

The input and weight domains stay immutable. Normalized features and hidden
activations are source-owned scratch; GELU updates the latter in place.
The stage is independent of denoising time and only needs to execute once
for a new text-conditioning request.

```sh
python -B experimental/loom_serve/models/krea2/check_text_projection.py \
  --checker bazel-bin/experimental/loom_serve/component_check \
  --model=experimental/loom_serve/models/krea2 \
  --checkpoint="$krea_weights" --adapter="$krea_adapter" \
  --reference=/path/to/reference-results \
  --output=/path/to/new-text-projection-results
```

The driver uses CPU-only F64 reference arithmetic on the required checkpoint
tensors, not a full external DiT instance. Captured base/LoRA inputs run at
16, 80 and 512 rows. All 114 repeated comparisons pass over 118,370,304
values, with no primitive-envelope violations or nonfinite pairs. Adapter
addition, stage composition, whole-command composition and zero strength
are bitwise exact against independently staged native operations.

At 512 rows, the complete native chain's relative L2 against the independent
F64 chain is 9.92026e-5 base and 1.52610e-4 LoRA. Relative L2 against the
captured external projection is 3.47809e-4 and 3.81059e-4, respectively;
these aggregate observations are separate from primitive acceptance.

| Text projection at 512 rows | Base | LoRA |
| --- | ---: | ---: |
| Unique kernels | 4 | 7 |
| Parameters | 5 | 9 |
| Parameter bytes | 213,968,896 | 216,655,872 |
| Workspace bytes | 8,912,896 | 8,945,664 |

One overwritten working fixture bounds disk use. Final per-shape projections
from these captured inputs supply a numerical diagnostic for image qualification;
its exact oracle projects the fully native fusion results instead. Native text
encoding remains outside the image command.

### Native text-fusion blocks

[`text_fusion_blocks.loom`](text_fusion_blocks.loom) composes the two kinds of
2,560-wide transformer used between Qwen3-VL's twelve hidden-state taps and
the text projection. Layerwise blocks attend across twelve encoder layers
independently for every token; refiner blocks attend across tokens with the
captured key-padding mask. Neither applies rotary embedding or time modulation.
Both retain zero-centered RMSNorm, twenty 128-channel heads, a sigmoid attention
gate, ungated residuals and a 6,912-wide SwiGLU. All eight projections support
the official rank-32 LoRA with immutable base and adapter parameter roots.

Dense layerwise work flattens `[token,12,2560]` without padding the layer axis.
The [short-attention motif](kernels/segmented_attention.loom) gives one wave32
one query/head, retaining four channels per lane and accumulating twelve keys
with F32 online softmax. Every query stays inside its own twelve-row segment;
there is no global score matrix or LDS tile. The refiner reuses the masked
long-attention motif. Concrete [kernel leaves](text_fusion_kernels.loom) own
configuration lookup; shared arithmetic receives explicit dimensions.

Input features and checkpoint storage stay immutable. Command-owned scratch
allows Q/K normalization and the SwiGLU product to advance in place, and
independent projections have explicit concurrent scopes. Separate B outputs,
ungated attention and SiLU exist only in the qualification catalog, not in
the deployment catalog.

The canonical reference extraction loads only the fusion module, not the
entire DiT. It requires its base/zero/LoRA outputs to reproduce the original
full-model captures byte-for-byte before accepting its intermediate files:

```sh
krea_fusion_reference=/path/to/new-fusion-reference
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea2/reference_text_fusion.py \
  --checkpoint="$krea_weights" --adapter="$krea_adapter" \
  --reference="$krea_reference" --output="$krea_fusion_reference"
python -B experimental/loom_serve/models/krea2/check_text_fusion_blocks.py \
  --checker bazel-bin/experimental/loom_serve/component_check \
  --model=experimental/loom_serve/models/krea2 \
  --checkpoint="$krea_weights" --adapter="$krea_adapter" \
  --reference="$krea_reference" --fusion_reference="$krea_fusion_reference" \
  --output=/path/to/new-fusion-block-results
```

Both commands use the pinned reference environment and exclusive GPU execution.
Extraction retains 135 MiB of canonical stage tensors. Native qualification
uses CPU-only F64 primitives on real inputs for both layer indices and axes,
at 16 and 512 tokens. Every device command executes twice; composed blocks
must match the separately qualified native stages bit-for-bit, including
zero-strength identity. Whole-block F64 and external-model errors are reported
separately from primitive acceptance. One overwritten fixture plus final
block outputs bounds the native result directory below 1 GiB.

The 16 qualified block cases produced 1,008 complete-array comparisons over
5,306,695,680 elements, with no nonfinite values or elements outside the
unchanged primitive envelope. Complete native composition and zero-strength
identity were bitwise on both executions. At 512 tokens, reflected scratch is:

| Block axis | Base | Rank-32 adapter |
| --- | ---: | ---: |
| Twelve layers per token (6,144 rows) | 231 MiB | 232.5 MiB |
| Token sequence (512 rows) | 19.25 MiB | 19.375 MiB |

Each base block binds twelve tensors and uses eight unique kernels; adapted
blocks bind 28 tensors and use twelve kernels. These ASAN-checker results
establish numerical and composition correctness, not throughput.

### Complete native text fusion

[`text_fusion.loom`](text_fusion.loom) joins two layerwise blocks, the learned
twelve-to-one reduction, and two token refiners. The public `text_fusion` root
receives the base parameters, immutable `[token,12,2560]` BF16 taps, a byte key
mask and distinct `[token,2560]` BF16 output. `text_fusion_adapted` additionally
receives the adapter root and device F32 strength. The original taps remain
unchanged; one source-owned tap buffer advances through the layerwise blocks,
and the refiners advance the reduced output in place. Explicit command edges
order every transition without intermediate host readbacks.

The [layer-mixing motif](kernels/layer_mix.loom) retains twelve layer values
per lane and evaluates rank-32 A/B in registers. Adjacent
lanes own adjacent channels, preserving coalesced input access without a
transpose. A, B, scaling and base addition retain their BF16 rounding boundaries.
The production projector has no global rank tensor or transient workspace.
Its gfx1151 emitted resource report has 34 VGPRs, 24 SGPRs, zero spills,
zero LDS/private storage and 2,984 code bytes. The reported sixteen resident
waves per SIMD is a resource ceiling, not measured device utilization.

```sh
python -B experimental/loom_serve/models/krea2/check_text_fusion.py \
  --checker bazel-bin/experimental/loom_serve/component_check \
  --model=experimental/loom_serve/models/krea2 \
  --checkpoint="$krea_weights" --adapter="$krea_adapter" \
  --reference="$krea_reference" --fusion_reference="$krea_fusion_reference" \
  --output=/path/to/new-text-fusion-results
```

This uses the same pinned reference environment and exclusive execution route.
Forty primitive comparisons pass the unchanged F64/BF16 envelope over
278,446,080 elements. Forty exact comparisons cover 27,033,600 elements,
including independently staged A/B, strengths zero/half/one, full composition
and zero-strength identity. Each command executes twice at 16/512 tokens with
base/style inputs. Another 32 finite-only block observations report cumulative
rounding differences; they are not counted as primitive tolerance passes.
The sole overwritten fixture and final features retain less than 384 MiB;
only qualification materializes the full-size 80 MiB rank tensor.

At 512 tokens the complete source command reflects:

| Resource | Base | Rank-32 adapter |
| --- | ---: | ---: |
| Unique kernels | 17 | 25 |
| Parameter tensors | 49 | 115 |
| Parameter storage bytes, including alignment | 686,903,552 | 714,561,536 |
| Scratch | 261 MiB | 262.5 MiB |

Full-feature relative L2 versus the independent F64 chain is 0.001684 base
and 0.000589 adapted; versus the canonical BF16 implementation it is 0.003549
and 0.001545. These aggregate diagnostics are separate from primitive acceptance
and exact composition. The image command composes this stage with projection,
denoising and decoding; native encoding and the image request adapter remain
distinct subsequent boundaries.

### Text encoder reference boundary

[`reference_text_encoder.py`](reference_text_encoder.py) isolates the pinned
Qwen3-VL encoder and tokenizer without loading the DiT or VAE. It uses the
pipeline's actual prompt-layout method, checks cumulative rotary positions,
and extracts embedding, shared rotary and layer-0/layer-34 primitive tensors.
The full twelve-tap result must match the original reference capture exactly.
Decoder hooks also establish that tuple indices 2, 5, ... 35 are decoder
outputs 1, 4, ... 34, before final normalization.

```sh
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea2/reference_text_encoder.py \
  --checkpoint="$krea_weights" --reference="$krea_reference" \
  --output=/path/to/new-encoder-reference
```

Run this in the pinned reference environment with exclusive GPU access.
The fixed 512-feature request has 546 encoder positions: 34 prefix positions,
prompt tokens, middle padding and five live suffix tokens. Padding does not
advance RoPE position, but causal ordering still follows sequence position.
Qwen3-VL uses BF16-rounded normalized values before multiplying its learned
RMS scale, half-split rotary pairs and ungated causal attention. These are
different arithmetic contracts from Krea's zero-centered RMS and DiT rotation.

The original encoder and a second execution omitting decoder 35 and the final
normalization both reproduce all 15,728,640 captured BF16 tap values bitwise.
The selected text-only dependency set contains 386 tensors occupying
7,843,069,440 bytes, excluding the vision model. This is a verified reference
dependency boundary, not a native encoder or serving result. The capture
retains 139,501,362 bytes of primitive/request fixtures plus a small manifest;
the full tap result and checkpoints are reused, not duplicated on disk.

### Native encoder normalization and rotary

[`encoder_normalization.loom`](encoder_normalization.loom) provides the two
hidden-state norms and fused Q/K head normalization plus half-split rotation.
Its shared motifs receive shapes, epsilon and buffers as SSA operands; only
the concrete leaves know model configuration and checkpoint names. The
normalized value rounds to BF16 before scale multiplication, and each rotary
product and sum retains its BF16 boundary. A thread owns both channels of a
half-split pair. Each leaf uses one dispatch and no global workspace.

The physical encoder extent is derived as `text_tokens + 48`: 34 prefix rows
plus 14 trailing rows for the dense kernels' 16-row tile. Those extra keys are
masked and discarded; the model's middle padding and live suffix stay intact.
There is no independent encoder-length setting.

```sh
python -B experimental/loom_serve/models/krea2/check_encoder_normalization.py \
  --checker bazel-bin/experimental/loom_serve/component_check \
  --model=experimental/loom_serve/models/krea2 \
  --checkpoint="$krea_weights" --reference=/path/to/new-encoder-reference \
  --output=/path/to/new-encoder-normalization-results
```

Qualification uses real layer-0 and layer-34 tensors at 64 and 560 physical
rows. Independent F64 primitives passed 48 comparisons over 38,338,560 values;
24 differed in BF16 bits, with no value outside the unchanged elementwise
envelope and no nonfinite result. Fused execution matched separately staged
native normalization and rotation exactly in 16 comparisons over 12,779,520
values. Both executions of every case are checked. This establishes the
arithmetic leaves, not causal attention, a complete encoder or prompt serving.
The driver retains one overwritten fixture plus final Q/K results, below
96 MiB, without duplicating checkpoints.

### Native causal encoder attention

[`encoder_attention.loom`](encoder_attention.loom) specializes the shared
128-channel GQA motif for 32 query heads and eight KV heads. A compile-time
causal operand selects sequence-order masking and bounds key work by the
query tile; it adds no device ABI argument. Physical buffer bounds and causal
visibility are separate predicates. The encoder centers logits against its
valid prefix key, while noncausal callers retain their original final-key
centering. Online softmax needs neither a global score matrix nor replicated
KV heads. The command uses one kernel and zero global workspace.

```sh
python -B experimental/loom_serve/models/krea2/check_encoder_attention.py \
  --checker bazel-bin/experimental/loom_serve/component_check \
  --model=experimental/loom_serve/models/krea2 \
  --reference=/path/to/new-encoder-reference \
  --normalization_results=/path/to/new-encoder-normalization-results \
  --output=/path/to/new-encoder-attention-results
```

Real layer-0/layer-34 Q/K/V at 64 and 560 rows passed 16 independent F64
comparisons over 20,447,232 values, with no nonfinite or out-of-envelope
result. Replacing masked K/V with nonzero values leaves all outputs bitwise
exact: eight comparisons over 10,223,616 values. Perturbing future K/V inside
a key tile leaves the earlier prefix exact in four captured comparisons over
344,064 values, while changing later outputs. The full case retains the
canonical middle padding, live suffix and partial final key tile. These are
arithmetic and isolation checks, not an encoder throughput measurement.

The shared-motif extension also reproduces the retained noncausal DiT results
bit-for-bit at 16, 80 and 1088 rows, base and active adapter, both ungated and
gated. The complete 512-row adapted text-refiner block remains bitwise exact.
