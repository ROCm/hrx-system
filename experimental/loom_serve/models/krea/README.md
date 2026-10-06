# Krea 2 Turbo source-JIT image generator

This includes a native prompt-to-image CLI and a retained HTTP image server.
The `generate_image` and `generate_image_adapted` roots run through Loom's live
source JIT, queued safetensors loading, command programs and shared device
ownership. Source VM control supplies a 16-byte seed/count/strength header and
validated, unpadded token IDs. Source kernels generate initial packed noise,
timesteps, encoder/DiT rotary tables, padding/masks, Euler deltas and VAE affine
directly in the command workspace. The command computes its own Qwen3-VL taps
and derives the combined text/image key mask on-device. The generic
[`image generator`](../../image/generate.c) and
[`image server`](../../image/server.c) load this external source package through
the [diffusion-image ABI](../../image/model.h). Source owns model policy;
native capabilities provide IREE tokenization, IO and device ownership.
The server shares one residency across requests.
No captured tensors, Python inference library or compiled model artifact is
required by that path. The independent reference libraries below are used
only for numerical qualification.

The command fuses the encoder taps into 2,560-wide text features and projects
them into the DiT's 6,144-wide prefix once. It batches time conditioning, then
runs eight denoising steps through image projection, all 28 transformer layers,
the velocity head and Euler updates. The complete native still-image VAE
decodes the packed final latent into clamped F32 RGB. Intermediate features,
conditioning and latents stay on device. Explicit phase lifetimes let one
planned workspace serve the whole command.

The official softwatercolor adapter covers fusion, text projection and DiT
projections. Encoder, Turbo, adapter and VAE weights remain separate immutable
domains; fusion and denoising share the same Turbo and adapter images. Lower-level
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

## Generate an image natively

Build with the repository's configured build service. Run the resulting binary
on a qualified GFX11 AMDGPU host with the source tree and checkpoints above:

```sh
build_tools/bin/iree-bazel-build --config=asan \
  //experimental/loom_serve/image:generate
bazel-bin/experimental/loom_serve/image/generate \
  --model=experimental/loom_serve/models/krea \
  --checkpoint="$krea_weights" \
  --prompt="A small brass robot tending red flowers in a sunlit greenhouse, watercolor illustration" \
  --seed=42 --height=384 --width=384 --text_tokens=512 \
  --output=/path/to/robot.ppm
```

Add `--adapter="$krea_adapter/softwatercolor.safetensors" --strength=1`
for watercolor adaptation. Strength zero retains the base image. Output is
binary PPM; an image viewer or `magick robot.ppm robot.png` can convert it.
The executable prints reflected weight/workspace bytes and the JIT's kernel
count. The instrumented build is for correctness, not timing comparisons.
Use the execution host's normal exclusive-run mechanism on a shared device.

This caller creates one immutable set of encoder, Turbo, optional LoRA and VAE
parameter domains, shared by its cold-compiled retained commands. They also
share one input/output bank and the maximum reflected scratch allocation.
The model borrows a caller-owned serving device and its physical pool. Cold
creation indexes parameters without reading payloads; first generation or
explicit activation streams them. Deactivation releases all base/adapter
parameter backing without rebuilding commands. The next request reloads and
prepares into the same roots. `image:model_check --reload_weights` exercises
this lifecycle across base and adapted output. Workspace remains a retained
allocation separate from the elastic parameter budget.
Request inputs upload once. A single source command runs all model stages;
only completed F32 RGB comes back. Both normal completion and failure drain
accepted work before borrowed upload/readback storage is freed.

[`prepare.loom`](prepare.loom) is the source-JIT bootstrap. It receives image
dimensions, maximum text capacity, the checkpoint directory and optional adapter
path, then declares command roots, compiler configuration values and fixed
parameter bindings through the runner-private `prepare` module. That module
copies the declarations and releases the bootstrap program before device setup.
The source also returns its public model name, tokenizer asset, request byte
capacity and opaque state. Returned buffers remain alive independently of that
program; their VM environment outlives the residency.
The ordinary JIT and streaming loader consume the result; no native table of
Krea command names, specialization formulas or weight filenames is involved.
This is a cold-path boundary; warm request processing uses the source control
program below without preparing new kernels or weights.

[`control.loom`](control.loom) owns prompt framing, token validation, compact
payload construction, warm stage selection, command submission and final RGB
feedback. Its `prepare_request` receives opaque bootstrap state, source-declared
stage tags, prompt, seed and strength, then returns an ordinal and payload.
Both base and adapted commands use the same request bindings. One source-JIT process
serves the entire residency; requests are buffers and scalar arguments, not VM
instances. The existing runner-private `execute_N` and `feedback` imports enqueue
work without waiting. Native code retires both accepted queue frontiers before
encoding the image or reusing request storage, including after partial failure.
The VM's return is not a transfer of native storage ownership.

The generic [`input` module](../../runtime/input.h) exposes bounded encoding,
vocabulary lookup and explicit input rejection. It knows no prompt template,
token IDs or embedding size. [`request.loom`](request.loom) owns numerical request
preparation, composed by [`generate.loom`](generate.loom) with the independently
callable `sample_image` components. Its F32 device transcendental approximations
are checked against independent BF16/F64 references; bit identity with host
libm is not the numerical contract. The complete request needs no host step
loop or per-request VM instance. At 1024 square and text512, the compact inputs
total 2,200 bytes instead of the former 5.53 MB of host-generated tensors.

`--seed` is an unsigned decimal 64-bit key for Philox4x32-10 followed by an
F32 device Box-Muller transform, rounded to BF16 in packed order. It defines
this runner's counter/key convention, not PyTorch's CPU generator sequence.
Same-request repetition is checked; different target math approximations are
not promised to yield identical floating-point bits. Comparisons with other
implementations consume the same initial noise.

Pixel dimensions are multiples of 16, and their patch-grid area must be a
multiple of 16. `--text_tokens` is a positive multiple of 16 and bounds the
maximum retained text including the suffix. A maximum greater than 128 and
divisible by 64 retains both maximum and text128 commands; other maxima retain
only the configured command. The native tokenizer encodes the whole framing
prefix plus prompt once at the maximum capacity. Combined counts up to 98 use
text128 when available, preserving the prompt/suffix live-key partitions;
larger counts use the maximum. Materialization right-pads before the live
suffix and masks additional physical tile rows without truncating again.
These values specialize the live source at startup, not a precompiled shape
catalog or a warm shape cache.
The CLI runs one fresh image per invocation. The image server below accepts
concurrent clients and serializes complete images through one residency; it
does not batch images or use the Qwen token scheduler.

### Retained model ownership

[`image/model.h`](../../image/model.h)/[`image/model.c`](../../image/model.c)
is the reusable native residency
behind the CLI. Creation loads the tokenizer and specializes the bounded
retained command set declared by the bootstrap. All checkpoint declarations,
reflected parameter roots and tensor placements must match exactly before it
streams each parameter domain once. Every command
records the same immutable buffers. Allocation uses the source-declared input
capacity and maximum reflected workspace length and alignment.
Serialized `model_generate` calls take only prompt, seed and adapter strength.
They neither compile nor reload weights nor allocate device backing. Pixel
shape and maximum text capacity are fixed when the residency is created.

The `image_residency` event reports the one-bank parameter/input/output and
workspace bytes, workspace alignment, each source stage tag and its kernel
and entry counts. The aggregate counts include both cold compilations; shared
weights do not imply deduplicated executable objects. Each `image_prepared`
event reports the selected stage ordinal and opaque payload byte count.
Krea interprets its stage tags as text extents; the generic runner does not.

The returned F32 NCHW RGB is borrowed until the next generation or destruction.
A consumer encodes or copies it before submitting another request. Host request
storage survives every accepted upload, including execution failure; teardown
joins accepted work before releasing the device. Invalid request data rejected
before submission leaves the model usable. A device execution failure is
terminal for its owner. The application serializes calls; this leaf does not
create an HTTP server, request queue, or implicit model worker.

The full-checkpoint reuse witness runs short→long→short in one residency,
requiring source-selected extents 128→512→128 for combined counts 40→99→40. With an
adapter, it repeats that sequence at strengths zero and one; without an
adapter, it runs the base sequence. A nonfinite-strength request precedes every
valid call. Final pixels are compared exactly with isolated CLI outputs, and
each repeated short image must have identical F32 bytes. An older qualified
fixed-extent CLI can be supplied as `--generator` for the counterfactual.

```sh
build_tools/bin/iree-bazel-build --config=asan \
  //experimental/loom_serve/image:generate \
  //experimental/loom_serve/image:model_check
build_tools/bin/iree-bazel-test --config=asan \
  //experimental/loom_serve/models/krea:model_test \
  //experimental/loom_serve/models/krea:control_test
python -B experimental/loom_serve/models/krea/check_model.py \
  --generator bazel-bin/experimental/loom_serve/image/generate \
  --checker bazel-bin/experimental/loom_serve/image/model_check \
  --model experimental/loom_serve/models/krea \
  --checkpoint "$krea_weights" \
  --adapter "$krea_adapter/softwatercolor.safetensors" \
  --output /path/to/new-residency-results
```

This uses the reference environment's NumPy and Pillow only to compare output
pixels. It retains less than 16 MiB of final images and logs and runs under the
execution host's exclusive device lease. Omit `--adapter` to exercise the
three-domain base residency separately.

The earlier fixed-extent qualification passed all four exact pixel comparisons
and the repeated F32 identity check, with one JIT, one residency and one load per
parameter domain. The extracted CLI also reproduced both previously qualified
image/text shapes exactly; all 154 independent request-input comparisons still
had zero differing bits. These are correctness checks, not throughput results.

### Serve images over HTTP

The server reuses the model leaf above and the runner's model-independent
[`image_service`](../../image/service.h). It exposes one configured pixel shape,
maximum text capacity and one optional adapter. Image execution runs on one
worker; the application owner handles bounded admission, health, completed
responses and peer cancellation while the existing `iree/net` TCP carrier owns
network I/O.
There is no inference subprocess or per-request JIT/model construction.

```sh
build_tools/bin/iree-bazel-build --config=asan \
  //experimental/loom_serve/image:server
bazel-bin/experimental/loom_serve/image/server \
  --model=experimental/loom_serve/models/krea \
  --checkpoint="$krea_weights" \
  --adapter="$krea_adapter/softwatercolor.safetensors" \
  --height=384 --width=384 --text_tokens=512 \
  --port=8080 --connections=64 --pending_requests=32
```

It binds loopback only. `--port=0` requests an ephemeral port, reported by the
`image_ready` JSONL event. `GET /healthz` reports active, queued and completed
counts; `GET /v1/models` reports the model identifier, size and supported
conditioning. Logs identify request lifecycle transitions without recording
prompts. A complete request is:

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"krea2-turbo","prompt":"A deer grazing in the forest, Art Deco watercolor style","seed":"0","strength":1,"size":"384x384","n":1,"response_format":"b64_json"}' \
  -o response.json
jq -r '.data[0].b64_json' response.json | base64 --decode > image.png
```

`prompt` is required; an empty prompt is valid. `seed` defaults to zero and
accepts an unsigned 64-bit integer or decimal string; the string avoids client
JSON number precision loss. `strength` defaults to one; zero selects the base
path when an adapter is loaded. A residency without an adapter accepts only
strength one. Optional `model` and `size` must match the residency, `n` must be
one, and the response format is `b64_json`. Unknown fields, duplicate fields,
wrong types, invalid UTF-8, trailing JSON and nonfinite strengths reject before
model submission. JSONC syntax follows the IREE parser. The body limit defaults
to 64 KiB and is configurable with `--request_body_bytes`.

The response is `{"created":<unix-seconds>,"data":[{"b64_json":"<PNG>"}]}`.
PNG conversion is native C and consumes only completed F32 CHW RGB. An admitted
response copies its bytes into transport-owned storage, so a slow reader does
not hold the model's workspace or output. Excess pending images receive HTTP
503; malformed generation requests receive 400. A queued peer reset removes
that request. An active reset discards its eventual result without recycling
device state early. SIGINT/SIGTERM stops admission and joins active generation
before releasing inputs, weights or the device.

Timing events separate native request preparation, submission, the remaining
device-completion wait, PNG encoding, JSON/base64 encoding, and transport
admission. Durations are nanoseconds. Submission overlaps device execution;
`completion_wait_ns` includes pending uploads and final readback, so it is not
an isolated kernel timer. `send_call_ns` measures copying/queueing the response,
not its asynchronous network drain. A client-side complete-response timer is
the end-to-end latency boundary. Optimized builds and an isolated execution
host are required for performance comparisons.

`image_heartbeat` is emitted every second, including during a device wait and
shutdown drain. It reports active request/age, worker phase, queue depth and
completed images; `--heartbeat_ms=0` disables it. `generating` means that the
native call has not returned, not that any particular dispatch has completed.
Immediate `image_preparing`, `image_prepared` and `image_submitted` events
identify the last reached host boundary before that wait.

The existing system observer adds independently sampled temperatures, power,
clocks, memory, process CPU and device utilization to the same JSONL stream.
From the source-tree root, inside the execution host's benchmark lease:

```sh
python -B -m experimental.loom_serve.tools.observe --log=/path/to/run.jsonl -- \
  bazel-bin/experimental/loom_serve/image/server \
  --model=experimental/loom_serve/models/krea \
  --checkpoint="$krea_weights" --height=1024 --width=1024
```

The observer records sensor identity and units once. Unsupported, suspended or
failed sources remain explicitly unavailable, not zero. It is optional external
telemetry, not part of native model execution or a Python inference dependency.

Both native callers honor the existing HAL profiling flags. Add
`--device_profiling_mode=dispatch-events,executable-metadata`
and `--device_profiling_output=/path/to/image.ireeprof` to retain dispatch
timings and names. `--device_profiling_flush_interval_ms=1000` periodically
flushes available profiling records. The session ends after accepted model
work drains, including error cleanup. Inspect the resulting bundle with
`iree-profile dispatch --format=jsonl /path/to/image.ireeprof`.
Instrumentation is for attribution; unprofiled optimized runs establish
end-to-end performance.

Multiple weighted adapters and reference-image conditioning are not accepted
by this schema. Discovery explicitly reports `reference_images:false`; passing
such a field fails instead of silently ignoring it. Krea's community edit
LoRAs require vision/VAE reference encoding and their trained conditioning
contract, which this text-only encoder path does not implement. The single
adapter boundary here is a concrete retained service foundation, not a claim
that arbitrary LoRA combinations are already prepared or cached.

The real-checkpoint HTTP witness consumes the isolated PPMs from
`check_model.py` above. It decodes responses with Pillow and requires exact
pixels for the short fox prompt, the long repeated-word prompt, active LoRA,
and base repetition. Their measured combined token counts 40 and 99 must
select text extents 128 and 512 respectively, including after active peer
reset and during joined shutdown. It also exercises health during generation,
queue overflow, a deliberately slow reader, and queued cancellation. All seven
generated images share one residency: logs require two cold JIT stages, shared
maximum workspace backing, and exactly one load per parameter domain.

```sh
build_tools/bin/iree-bazel-test --config=asan \
  //experimental/loom_serve/image:request_test \
  //experimental/loom_serve/image:output_test \
  //experimental/loom_serve/http:request_test \
  //experimental/loom_serve/http:server_test
build_tools/bin/iree-bazel-build --config=asan \
  //experimental/loom_serve/image:server
python -B experimental/loom_serve/models/krea/check_service.py \
  --server bazel-bin/experimental/loom_serve/image/server \
  --model experimental/loom_serve/models/krea \
  --checkpoint "$krea_weights" \
  --adapter "$krea_adapter/softwatercolor.safetensors" \
  --isolated /path/to/new-residency-results \
  --output /path/to/new-service-results
```

The HTTP witness runs on the same qualified device under its exclusive lease
and retains less than 16 MiB. Its lifecycle events provide readiness/completion
synchronization; it does not approximate readiness with sleeps. Host-ASAN
results establish ownership/correctness, not image-generation performance.

The earlier fixed-extent HTTP qualification generated seven images in one
model residency. All five returned PNGs matched the isolated CLI pixels exactly;
the two other completed images retired safely after active peer reset and
shutdown. Health, bounded overflow, queued cancellation and slow-reader
independence passed. The run observed one JIT and four parameter-domain loads
total. These are correctness results, not a throughput comparison.

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
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea/reference.py \
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
  //experimental/loom_serve/models/krea:projection_check
bazel-bin/experimental/loom_serve/models/krea/projection_check \
  --model=experimental/loom_serve/models/krea \
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
wrappers. The shared [`linear.loom`](../../motifs/tensor/linear.loom) motif receives dimensions and
output encodings as explicit SSA operands in templates. The wrappers fix those
operands during JIT specialization; shapes and layouts are not carried in the
device launch ABI.
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

The early denoising qualification below receives captured conditioning and uses an
external VAE for its preview images; the later image-command qualification
includes native text encoding, fusion, projection and VAE decoding.
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

Component entry points live in the separate [`qualification/`](../krea2/qualification)
source catalog. The Python drivers select that catalog automatically; direct
`component_check` calls use `--model=experimental/loom_serve/models/krea/qualification`.
Commands loading checkpoints also specify
`--weight_policy=experimental/loom_serve/models/krea/weights.loom`, independently
of the source catalog, so qualification uses the deployment's actual preparation
policy rather than a copied or inferred policy.
Its `krea2.block_index` specialization defaults to zero. The model commands
receive an explicit compile-time layer index and format checkpoint keys from
it; native kernels are independent of layer identity. Qualification wrappers
are not part of the deployment catalog.

```sh
build_tools/bin/iree-bazel-build --config=asan \
  //experimental/loom_serve/tools:component_check
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea/check_block.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
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
BF16. The default contraction combines four independent accumulator chains
every 256 inputs to shorten cancellation-sensitive reductions; selected large
shapes instead retain a full-K recurrence. Neither strategy allocates global
partial sums or copies/expands the weight matrices.

The baseline for the four DiT shapes is the cooperative
[`linear_tiled_bf16`](../../motifs/tensor/linear_tiled.loom) motif. Eight wave32s share a
64x64 output tile, acquiring K64 operand tiles into 18 KiB of padded LDS.
The loop pipeline overlaps next-tile acquisition with current WMMA work while
retaining the same four-chain/K256 arithmetic and final BF16 rounding. A last
partial workgroup zero-fills absent input rows; every thread participates in
the shared-memory barriers, and only complete valid 16-row fragments store.
The supported row count therefore remains any positive multiple of 16.
Model configuration and launch geometry stay in the concrete wrappers;
dimensions and output encoding enter the motif as SSA operands, not dispatch
parameters. The V projection writes its logical `[tokens, 1536]` matrix with
strides `[1, tokens]`, physically contiguous as `[1536, tokens]`. Attention reads
that same storage directly. All other block projections remain row-major.
This changes the final writer's coordinates, not the allocation size, weights,
arithmetic, or number of dispatches; there is no intervening transpose.

The motif takes a positive row-tile group size as an explicit SSA operand.
The workgroup-coordinate bijection groups that many 64-row tiles before
advancing the input panel. Adjacent physical workgroups can reuse that bounded
panel across output columns, without changing storage, the physical grid, or issued
operand bytes. The final group uses its actual row-tile count; it introduces
no padded workgroups or duplicate stores. This is a cache-locality policy, not
a guarantee of physical workgroup execution order. Tile traversal and tile
geometry are independent tuning dimensions.

[`linear_tiled_wide_bf16`](../../motifs/tensor/linear_wide.loom) is an alternative full-K
motif. The same eight wave32s own a 64x128 output tile: each wave carries
four ascending-K16 accumulator
chains, reusing each input fragment across twice as many output columns.
Cooperative K64 loads use one padded 64x72 input stage and 128x72 weight stage
(27 KiB total LDS). The publication/retirement barriers and two-deep pipeline
overlap next-tile acquisition without double-buffering LDS. Tile selection
belongs to model wrappers, not the configuration-free motifs.

At 4,224 rows, the 6,144-wide query/gate/output projections select
[`linear_temporal_bf16`](../../motifs/tensor/linear_temporal.loom), as do FFN up/gate
and down at 4,224 and 4,608 rows. Four wave32s own a 128x96 output tile with
twelve full-K result fragments per wave. Two temporal
LDS banks occupy 65,024 bytes; an explicit packet queue overlaps global reads
with current-tile WMMA. The steady loop only acquires valid future K64 tiles.
A peeled penultimate tile publishes the final queued tile without acquiring
an unused successor, followed by a final drain. This keeps a bounds-check
zero merge from forcing future loads to finish before current-tile arithmetic.
At 4,224 rows, up/gate and square projections group eight 128-row tiles per
traversal panel; down and the 4,608-row up/gate projections retain four.
For the 6,144-wide inputs, those policies revisit logical input panels of
12 MiB and 6 MiB respectively. This changes cache locality, not the issued
operand bytes or arithmetic. The larger panel reduced isolated up/gate time
by about 5% and square time by 6–7%; sixteen tiles slowed up/gate and gave a
smaller square gain. In matched full-model profiles the selected square gain
was about 3%, while up/gate improved only 0.4%; unchanged kernel families also
varied between controls. Isolated cache reuse does not establish a full-model
speedup. These are measured shape selections, not a cache-capacity guarantee
or a general rule that larger panels are better. Other projection shapes
retain the baseline motif.

The same motif specializes its RHS acquisition order through an ordinary SSA
operand supplied by the model leaf. Up/gate and square contractions with
6,144-wide inputs allow the next RHS pair's LDS loads to overlap the current
pair's four WMMAs. Compiler scheduling fences retire the old pair before a
third pair can become live. The 16,384-wide down contraction retains
pair-at-a-time acquisition. This changes neither temporal-bank publication
nor accumulation order, issued operand bytes, or runtime dispatches; the
schedule branch folds during JIT specialization. On the measured shapes,
read-ahead uses 216 rather than 208 VGPRs, retains the same LDS-limited
residency without spills, and improves isolated up/square time by 1–3%.
The shorter-text down contraction regressed slightly with that order, so the
selection belongs to concrete callers rather than becoming a universal
policy in the reusable helper.

A workgroup-uniform branch separates full output tiles from the final partial
tile around the entire contraction. The interior path carries an explicit
origin bound into the packet loader, letting path-dependent facts eliminate
its repeated bounds checks. Otherwise a zero merge can drain each future
global load before current-tile arithmetic even in fully valid workgroups.
The tail keeps guarded loads and stores, sharing the same physical LDS banks;
there is no padded global allocation or additional dispatch. When the output
width is divisible by 96, the branch specializes away. Shape selection alone
is not enough: preserving bounds across the complete loop is what allows the
prefetch queue to remain asynchronous.

The remaining tiled model wrappers select eight-row-tile groups at 2,048 rows for FFN up,
1,088 for FFN down, and 2,560 for square/narrow projections. Below those
crossovers they pass the whole row-tile count, which specializes to the original
physical coordinates. These are the earliest clearly winning tested row counts
in the [traversal experiment](../../docs/PERFORMANCE.md#workgroup-traversal-is-an-independent-tuning-dimension),
not a universal cache-size rule or an optimality claim for every intervening
shape. Selection happens during JIT specialization; it adds no device decision,
weight residency, workspace, or command dispatch.

[`tests/linear_tiled.loom`](../../motifs/tensor/tests/linear_tiled.loom) compares the baseline motif bitwise
against the original single-wave motif at all four widths, covering
16/32/48/64/80/512/528/544/560/640/704/768/832/896/960/1088/2048/2560/3072/4608
rows. Group-eight cases cover each traversal-group remainder and each 16-row
matrix tail; both policies run at the five largest row counts, for 100 row-major
cases. Eight additional column-major cases cover all four widths at 16 and 80
rows, comparing the complete physical byte streams. They link the real helper
sources instead of copying their implementations.
The checkpoint comparisons below independently exercise the actual command
wrappers and numerical error envelope.

[`tests/linear_wide.loom`](../../motifs/tensor/tests/linear_wide.loom) checks the real wide helper
against the full-K64 helper with nonuniform BF16 inputs. Its `exact_short`
case binds `wide_test.rows=16`, `wide_test.inputs=6144` and
`wide_test.outputs=16384`; `exact_tail` binds 528, 6144 and 256 respectively.
[`tests/linear_temporal.loom`](../../motifs/tensor/tests/linear_temporal.loom) compares the real
temporal helper's two RHS schedules with full-K64 at 528x256x320. This exercises
the shortest supported K, both matrix tails, the interior/tail split and a
partial traversal group without external weights or stored expected tensors.
The two full-K algorithms retain the same K16 accumulation order; this exact
helper comparison does not require equality with the baseline's four-chain/K256
reduction.

```sh
build_tools/bin/iree-bazel-build --config=asan \
  //loom/src/loom/tools/iree-test-loom:iree-test-loom
python -B experimental/loom_serve/motifs/tensor/tests/check_linear_tiled.py \
  --checker=bazel-bin/loom/src/loom/tools/iree-test-loom/iree-test-loom
```

Run the check on the qualified execution host under its exclusive device
lease. Compiler configuration and tensor sample selection are bound together
by the driver; every invocation must report one passing sample and no skips.

[`check_projections.py`](check_projections.py) compares every projection against
an independent CPU/F64 contraction, rounded once to BF16, at the full captured
token count and a separately JITed 16-row prefix. Each command runs twice.
The same BF16 error envelope applies to all shapes. The final feed-forward
projection uses the actual captured
`feed_forward_product` input, including the external model's SiLU/product
rounding. These are base-model comparisons: style projections include an
additional LoRA update and are not equivalent to the base matrix alone.

```sh
python -B experimental/loom_serve/models/krea/check_projections.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
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
attention with its sigmoid output gate. Both select the config-free,
128-channel [K32 template](../../motifs/tensor/attention_shared_k32.loom) when the token
count is divisible by 32. Other positive multiples of 16 use the
[K16 template](../../motifs/tensor/attention_shared.loom). Each
128-thread workgroup owns four query heads sharing one KV head. Its four
wave32 subgroups process the same 16 query positions while cooperatively
loading successive key tiles into LDS. This shares K/V reads across the
four heads instead of launching a separate workgroup for each head. There is
no KV-head replication or global attention-score matrix. A byte mask selects
valid keys, independently of query position. Every request must have at least
one valid key, which the model guarantees through its image tokens.

Q and K retain row-major storage. DiT V is logically `[tokens, 1536]` with
strides `[1, tokens]`; its projection and any in-place LoRA update write this
layout directly into the existing V buffer. The generic attention motif takes
the V encoding explicitly; the shared-head motif fixes this transposed layout
so that threads load contiguous keys. The text encoder and fusion callers
continue to select row-major V. Component checks serialize V in its physical
order but retain the logical matrix for the independent F64 oracle.

DiT uses ordinary BF16 Q/K operands and one BF16 probability/value
contraction. Dot products, the online maximum, the unrounded probability sum
and the running output accumulate in F32. Each key tile rounds its unnormalized
probabilities to BF16 before P*V; rounding a completed global softmax would be
a different algorithm. The online maximum remains in raw dot-product units,
so subtraction precedes scaling. The denominator accumulates lane-local
partials, with one reduction after the key loop. K32 combines two score
fragments before the maximum and rescales the running output only once for
both. It halves online-softmax updates and workgroup barriers without
duplicating QK or P*V contractions. The grouping changes probability rounding;
it is not bit-identical to K16 on arbitrary inputs.

K32 retains 32 query channels in registers and stages the remaining 96 in
wave-owned LDS planes. K16 splits those channels evenly. Loop-carried K/V
packets read ahead while the current tile is consumed; workgroup barriers
publish and retire shared storage. Current mask reads precede future K/V
reads, allowing the mask's completion wait to leave the next tile's data in
flight. Clamping the final prefetch to the last complete tile avoids padded
reads and extra global storage. At 4608 rows the gfx1151 K32 code uses
32,256 bytes of workgroup storage and 240 vector registers, with no spills;
K16 uses 19,712 bytes and 232 registers.

The explicit `attention.expanded_masked_gqa_128_bf16` variant retains
centered two-term QK and three-term probabilities for the text encoder and
refiner. Those callers' arithmetic is unchanged. This expansion can improve
agreement with F64 when large common logits hide small key differences, but
its two QK and three P*V contractions are not a DiT quality requirement.
The ordinary DiT path executes one of each; its accepted full-image base and
nonzero-LoRA outputs preserve coherent detail without requiring pixel identity
with the expanded path. A single prompt/seed comparison is bounded evidence,
not a distributional image-quality certification.

```sh
python -B experimental/loom_serve/models/krea/check_attention.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
  --reference="$krea_reference" --output=/path/to/new-attention-results
```

The reference Python environment supplies independent CPU/F64 scores,
softmax, and value contractions from the captured normalized/rotated inputs.
No checkpoint argument is needed: these commands contain no fixed parameters.
The native tool runs on the leased GPU. The result directory retains less than
256 MiB of regenerable tensors for the standard capture. The 16-row case uses
image positions; the 80-row case starts with four entirely masked key tiles
and ends with one valid tile.
Both base and adapter-conditioned inputs also run at the full 1088 rows.

Three additional 80-row analytic cases require exact BF16 answers: a single
valid first key, a single valid last key, and 64 equal-score keys spanning four
key tiles. Positive integer values vary by key, KV head and channel, exposing
masking, grouping, storage and tail errors. Zero gate inputs require an exact
one-half scale. Real captures require finite output, exact fresh replay and
invariance under changes to masked K/V. Gated attention must equal composition
of ungated attention with the unchanged native gate helper bit-for-bit.

The script reports both independent F64 and ordinary-K64 CPU distances without
making either an elementwise admission threshold for real attention captures.
The CPU K64 calculation retains the preceding per-head algorithm as a
diagnostic baseline; it does not model the selected shared-head rounding
boundaries, native WMMA or approximate-exponential rounding. CPU sigmoid
differences are likewise diagnostic; the native gate-composition check isolates fusion
correctness. Kernel experiments are judged by measured speed, real-tensor error
and image quality. Exact replay, masking and gate-composition checks isolate
logic errors without requiring pixel identity between different floating-point
algorithms.

The focused [K32 scenarios](tests/attention_shared_k32.loom) call both
production leaves at 64 tokens. An independent VM function computes the exact
mean over masked, signed, nonuniform values and broadcasts each KV head to its
four query heads; a zero sigmoid input gives one-half gating. An absolute
1e-6 bound accommodates approximate-exponential cancellation residues at
mathematical zero; inputs remain bitwise unchanged.
This checks the shared layout, mask, output-rounding and gate boundaries
without a copied device implementation or baked output tensor:

```sh
iree-test-loom experimental/loom_serve/models/krea/tests/attention_shared_k32.loom \
  --library=experimental/loom_serve/models/krea/block_attention.loom \
  --library=experimental/loom_serve/motifs/tensor/attention_shared.loom \
  --library=experimental/loom_serve/motifs/tensor/attention_shared_k32.loom \
  --config=krea2.block_tokens=64 \
  --device=amdgpu --target=amdgpu:gfx1151
```

### Residuals and feed-forward pointwise fusion

[`block_pointwise.loom`](block_pointwise.loom) supplies the two gated residual
updates and SiLU/up-product fusion. [`block.loom`](block.loom) shares one
normalization/modulation kernel between the pre-attention and pre-feed-forward
paths. Command-program subviews select the appropriate coefficient rows from
the shared modulation and learned parameter table. Selection requires neither
a copy nor a specialized kernel for each coefficient row.

```sh
python -B experimental/loom_serve/models/krea/check_pointwise.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
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
command contains 16 dispatches using eleven distinct JITed kernels. Independent
Q/K/V/gate and feed-forward branches have explicit concurrent scopes. There
are no host waits, intermediate readbacks, or per-dispatch allocations inside
the block.

```sh
python -B experimental/loom_serve/models/krea/check_transformer.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
  --checkpoint="$krea_weights/turbo.safetensors" \
  --reference="$krea_reference" --output=/path/to/new-transformer-results
```

This driver requires the reference environment, detailed block capture, and
exclusive GPU execution. It runs an independent CPU/F64 block calculation
with the model's BF16 tensor boundaries and F32 rotary arithmetic. A separate
native chain checks non-attention reductions against that arithmetic and each
fusion against native composition. Ordinary attention reports its independent
arithmetic distance without imposing expanded-precision accuracy. Finally the
single queued command must match the native chain bit-for-bit on both executions.
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
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea/reference_stack.py \
  --checkpoint="$krea_weights" --adapter="$krea_adapter" \
  --reference="$krea_reference" --output="$krea_stack_reference"
```

Use `--layer=27 --reference="$krea_stack_reference/1088"` with the whole-block
driver for that actual final-layer input distribution. The separate `16/`
capture is a shape check that advances 16 image rows through all layers; it
does not represent an entire prompt/image request.

The historical expanded-attention command-composition check passed at both
16 image rows and all 1088 captured rows. Against the independent complete-block
calculation, full-size relative L2 error is 0.00038576; the pinned external
model's error is 0.00107261. The complete block has 37,803 values outside the
single-operation BF16 envelope, versus 371,847 for the external block; those
counts remain visible because accumulated rounding is not a single-operation
error bound.
The acceptance gate requires independently qualified components and exact
native composition. Complete-block relative L2 remains a diagnostic; the
full-stack driver below reports accumulated error at its consumer boundary.
These are historical expanded-attention block-zero accuracy observations,
not a full-model image or performance result.

### Block LoRA projections

[`block_adapters.loom`](block_adapters.loom) applies the official adapter to
all eight projections of a selected block. Each adapted command binds separate
base and adapter parameter roots. The base contraction and rank-32 A contraction
have an explicit concurrent scope; the B contraction fuses BF16 rounding,
strength multiplication, and addition into the base output. Its same-tile
read/write permits in-place addition without a full-width delta buffer.
The zero-strength branch preserves the base instead of adding a rounded zero.
The 16384-input rank-32 A contraction visits both output halves of each input
panel adjacently, reusing those rows before advancing through the activation.
The 6144-input A contraction traverses rows first. These orders are selected
from the complete adapted graph: overlapping base contractions change the
resource/cache context, so an isolated A-kernel gain need not survive there.
The 6144- and 16384-wide row-major additions traverse output tiles first, so
successive workgroups visit adjacent columns rather than striding over the
entire activation matrix. Contraction, base-read and store motifs take the
same explicit tile origins; their arithmetic is independent of traversal.
For V, the base fragment read and final store both use the column-major
encoding selected by the model wrapper. The rank-32 intermediate and the
diagnostic, unfused B output remain row-major. The checker encodes only the
base/final V buffers physically; its independent arithmetic stays in logical
row/column order.

[`tests/value_adapter.loom`](tests/value_adapter.loom) links the actual V
adapter leaf. Nonuniform base values check coordinate ownership at 16 and 80
rows, with both distinct outputs and true same-buffer read/write. Exactly
representable factors isolate the epilogue from contraction error, while a
separate zero-strength probe checks negative-zero preservation. With the ASAN
`iree-test-loom` tool built above, run the two shape specializations under the
same exclusive device lease:

```sh
for rows in 16 80; do
  bazel-bin/loom/src/loom/tools/iree-test-loom/iree-test-loom \
    experimental/loom_serve/models/krea/tests/value_adapter.loom \
    --library=experimental/loom_serve/models/krea/block_adapters.loom \
    --library=experimental/loom_serve/motifs/tensor/linear.loom \
    --target=amdgpu:gfx1151 --device=amdgpu \
    --config=krea2.block_tokens="$rows" --case="@value_adapter_$rows"
done
```

Each invocation must report one passing sample, five passing bitwise
expectations, and no planning issues or skips. This is a storage/alias check;
the checkpoint-backed numerical qualification remains independent.

[`tests/adapter_traversal.loom`](tests/adapter_traversal.loom) exercises both
wide row-major writers at 80 rows, including distinct output, true alias and
signed-zero bypass. With the same tool and two libraries, set
`--config=krea2.block_tokens=80`; its two samples each have three exact
expectations.

[`tests/adapter_down_traversal.loom`](tests/adapter_down_traversal.loom) calls
both rank-32 A leaves at 80 rows with the same libraries and configuration.
Five row tiles and two output tiles check launch/origin coverage for both
the rows-first 6144-input leaf and output-first 16384-input leaf. Exactly
representable factors yield two bitwise expectations without a numerical
tolerance or a separate test kernel.

```sh
python -B experimental/loom_serve/models/krea/check_adapters.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
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
`qualify.block_forward`, retaining its original bindings and eleven kernels. The
adapted variant uses 17 distinct kernels and 32 dispatches. Strength remains device
data, so changing it does not require recompilation or base-weight mutation.

The whole-block driver accepts the same adapter as the projection checker:

```sh
python -B experimental/loom_serve/models/krea/check_transformer.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
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

The expanded-attention baseline passed all component and exact-composition
gates. Zero strength preserves the base block bit-for-bit at both shapes. For
the 1088-token active adapter case, relative L2 error against the independent
block calculation is 0.00047224, versus 0.00246784 for the external model.
Maximum absolute BF16 error is 4 versus 8; the single-operation envelope
counts are 60,420 versus 661,300 and remain visible as accumulated block
error. This qualifies one LoRA-enabled block, not the full 28-block denoiser
or an image.

The layer-27 fixture exercises actual final-layer inputs, including every
component, exact queued composition, and zero-strength identity at 16 and
1088 rows. Layer selection changes parameter keys, not numerical kernels:
the base and adapted commands still use eleven and seventeen kernels.

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
F64 while the complete stack is closer. Accumulated image-row and all-row
distances remain diagnostics as well; ranking two implementations by distance
to F64 does not rank their generated images. Image rows are reported separately
because the model discards text rows before its head, and padded text rows are
never valid attention keys. Exact native composition, zero-strength identity,
finite results and one-residency ownership remain hard gates.

```sh
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea/check_stack.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
  --checkpoint="$krea_weights" --adapter="$krea_adapter" \
  --reference="$krea_stack_reference" --image_rows=576 \
  --output=/path/to/new-stack-results
```

The single queued stack must equal the separate native chain byte-for-byte
on repeated executions. Zero-strength LoRA must equal the base stack exactly.
Reflection must show eleven base or seventeen adapted kernels, exactly 28 unique
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

The expanded-attention baseline passed repeated exact composition and
zero-strength identity. At 1088 rows, the 576 image rows have these relative L2
errors against the independent F64 chain:

| Phase | Loom | External implementation | Transient workspace |
| --- | ---: | ---: | ---: |
| Base | 0.87358% | 0.90045% | 149.8125 MiB |
| Softwatercolor, strength 1 | 1.49491% | 1.62224% | 150.078125 MiB |

The base stack references 364 unique parameter tensors and eleven kernels;
the adapted stack references 812 tensors across two checkpoint roots and
seventeen kernels. Each tensor has one storage range, and workspace matches a
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
fail. The image-row consumer boundary reports accumulated numerical distances;
it does not require one implementation to reproduce another's error ranking.

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
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea/check_head.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
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
python -B experimental/loom_serve/models/krea/check_velocity.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
  --checkpoint="$krea_weights" --adapter="$krea_adapter" \
  --reference="$krea_stack_reference/1088" \
  --head_results=/path/to/head-results --image_rows=576 \
  --output=/path/to/new-velocity-results
```

Base, zero strength, and active LoRA all match byte-for-byte on both native
executions at 1088 combined rows and 576 image rows. Standalone head
qualification retains all 144 passing comparisons after the composition
refactor. The base command shares 13 kernels and 368 unique parameters;
LoRA shares 21 kernels and 818 unique parameters. Peak transient storage is
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
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea/check_time.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
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
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea/check_denoise.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
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
| Unique kernels | 21 | 35 |
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

The config-free [convolution helper](../../motifs/tensor/convolution.loom) consumes NCHW
activations and original F32 OI(T)HW weights. A fresh causal frame selects the
last temporal weight plane; this is not a video-history implementation.
Nearest-neighbor upsampling can be folded into its input indexing without an
expanded image buffer. Concrete kernel leaves resolve geometry before applying
the helper, so no shape scalars enter the device ABI.

The element motif computes one valid output in ascending input-channel/tap
order with F32 FMA, followed by a separately rounded bias addition. Work
distribution and the store epilogue belong to its caller. The ordinary wrapper
stores NCHW and optionally adds a residual; the attention QKV leaf uses the
same computation with a consumer-specific layout. Neither needs a second
convolution implementation or a layout flag in the arithmetic motif.

```sh
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea/check_vae_input.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
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
The shared 3x3 matrix helper rounds operands to BF16 and accumulates in F32.
The last convolution retains the F32 bias rounding before adding the skip.

The residual uses two feature-sized temporaries. Its second normalization is
in-place, and its final write may alias the original residual because every
earlier original-state reader has completed. `vae_input_residual` exercises
that ownership by running the native input path and advancing the caller's
feature buffer in place, in one command.

```sh
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea/check_vae_residual.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
  --checkpoint="$krea_weights" --input_results=/path/to/vae-input-results \
  --output=/path/to/new-vae-residual-results
```

The driver uses independent F64 arithmetic, with BF16 operand rounding for
convolutions. Fused skip addition, the residual command, and the entire
in-place prefix must match staged native output bit-for-bit. The full
prefix has six unique kernels, ten parameters occupying 32,523,008 bytes, and
7,372,800 bytes of transient workspace. The residual itself needs 6.75 MiB.

The earlier scalar-convolution path passed 56 repeated comparisons over
24,901,632 values at 2x6 and 48x48. Its first-residual relative L2 was
8.57e-7 base and 8.75e-7 LoRA against the F64 chain. Those precision figures
do not describe the selected BF16 matrix arithmetic; its real-latent decoded
images are reported below. Spatial attention and RGB remain outside this
prefix. Residual fixtures occupy less than 192 MiB and can be regenerated
from the retained input results.

### Bounded-memory VAE spatial attention

[`vae_attention.loom`](vae_attention.loom) composes channel normalization,
QKV projection, single-head noncausal spatial attention, and a fused output
projection/skip. For 384 channels and token counts divisible by 16, the
[matrix attention helper](../../motifs/tensor/image_attention_matrix.loom) processes 16
queries per 128-thread workgroup. Four waves partition the full head into
96-channel QK partial sums, combine those sums before softmax, then each own
96 output channels. They are not separate attention heads. Q/K/V and each
unnormalized probability tile round to BF16; accumulators, online rescaling,
the unrounded probability denominator and output remain F32. Padded operand
and partial-score planes use 55,552 bytes of LDS, with no global score tensor.

Irregular token counts use the unchanged F32
[online attention helper](../../motifs/tensor/image_attention.loom), which handles 1–512
channels with 128-key tiles. Each workgroup retains one query's output
accumulators and a 512-byte probability tile, plus reduction storage. Neither
path materializes the 20.25 MiB score matrix for a 48x48 latent grid.

The QKV allocation contains three equal-size F32 planes: Q and K are
`[channels, tokens]`, while V is `[tokens, channels]`. Q/K reads are contiguous
when neighboring lanes score different keys; V reads are contiguous when
neighboring lanes accumulate different output channels. The projection writes
this layout directly into the existing allocation. There is no transpose
dispatch, additional tensor, or host synchronization. Attention output stays
NCHW, so its projection and residual consumers retain their original layout.

The standalone command uses 13.5 MiB of global scratch at 48x48: QKV plus one
feature plane reused between normalization and attended output. Allocation
points express those lifetimes. The final projection may write the original
input in place after QKV has consumed it. `vae_input_attention` exercises this
ownership through the complete native input/residual/attention prefix.

```sh
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea/check_vae_attention.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
  --checkpoint="$krea_weights" --input_results=/path/to/vae-input-results \
  --residual_results=/path/to/vae-residual-results \
  --output=/path/to/new-vae-attention-results
```

The driver checks 45 scalar-helper shape combinations crossing key-tile and
channel-tail boundaries, then real base/LoRA features at 2x6 and 48x48. The
matrix oracle explicitly rounds Q/K/V and each online probability tile to
BF16, with F64 contraction and softmax arithmetic. Its distance from the
full-F32 computation is reported separately, not treated as a mapping failure.
Fused addition, command composition, and the full in-place prefix still compare
with staged native results bit-for-bit.
The oracle operates on logical Q/K/V tensors independently of the physical
packing; the QKV producer is checked before its packed output feeds attention.

The focused [matrix attention scenario](../../motifs/tensor/tests/image_attention_matrix.loom)
compares the real helper against an independent scalar VM oracle on signed,
nonuniform data, with F64 contractions and an explicit F32 exponential. Two
query/key tiles exercise channel ownership and online rescaling without model
weights or baked output. The complete 48x48 prefix
retains ten unique kernels, 15 parameters occupying 34,889,984 bytes, and
17,989,632 bytes of global scratch. Attention fixtures occupy less than 256 MiB.

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

The middle residuals and all four upsampling stages use the config-free
[`convolution_3x3_matrix_f32`](../../motifs/tensor/convolution_matrix.loom) helper for
plain and residual 3x3 convolutions, including the 192-to-384 channel expansion.
All three resize leaves use the same helper with nearest-neighbor expansion
and 384-to-192 / 192-to-96 channel reduction. A workgroup owns 32 output
spatial positions by 32 output channels.
Cooperative acquisition shares inputs across output channels and coefficients
across pixels, while retaining original global F32 tensors. Convolutions read
the last of three temporal weight planes; resize coefficients have one plane.
The 1x1 learned shortcut, input projection and RGB convolutions retain their
original F32 leaves.

Height and width describe physical input storage. A separate SSA upsample
factor selects output geometry: 1 for residual convolutions, 2 for resize
stages. Tap bounds are checked in the expanded output domain before division
maps a valid tap back to its nearest physical input pixel.
This preserves padding at the expanded border without allocating an expanded
tensor or folding coefficients. The scale-1 specialization emits the same
native code as the non-expanding helper.

The helper rounds each operand to BF16 before LDS staging and uses ordinary
BF16 matrix products with F32 accumulation. This changes the arithmetic, not
the stored model precision; global tensors and bias/residual arithmetic remain
F32. Operand conversion must remain finite. Four wave32s share 4,736 bytes of
padded LDS and store directly into NCHW output. The arithmetic does not emulate
an F32 product using extra low-part or cross-product matrix operations.
No prepared weight bank, global im2col, extra dispatch or workspace is needed.
Bias rounds separately, followed by the optional residual addition. The
convolution input remains distinct from output, while the residual may be that
same output buffer. Invalid border taps supply zero; inactive spatial lanes
still participate in tile publication and retirement. Geometry enters as SSA
specialization operands; the helper contains no model configuration or runtime
scalar ABI.

[`tests/convolution_matrix.loom`](../../motifs/tensor/tests/convolution_matrix.loom) compares the
actual helper with an independent scalar VM function that rounds operands to
BF16, accumulates in F64, and rounds the final result to F32.
The two scenarios use physical 1x1 and 3x11 inputs at full 96/192/384-channel
reduction depths. Six configurations cover the three equal-width, three-plane
convolutions, channel expansion, and both mixed-width, one-plane resize shapes.
Each runs plain output, distinct residual output and genuinely aliased
residual/output storage.
Independent temporal entries, nonzero bias/skip, row-crossing tiles and output
sentinels expose indexing and publication errors. Scale 2 expands those inputs
to 2x2 and 6x22 outputs, exercising padding before nearest division and a
four-pixel final tile. Fixtures reserve flat capacity for 384 channels and
2x spatial expansion; both subjects view only the compact configured prefixes,
including compact channel/plane strides in the weights. Comparisons include
the unused output/state suffix:

```bash
while read -r inputs outputs planes upsample; do
  iree-test-loom experimental/loom_serve/motifs/tensor/tests/convolution_matrix.loom \
    --library=experimental/loom_serve/motifs/tensor/convolution_matrix.loom \
    --device=amdgpu --target=amdgpu:gfx1151 \
    --case=@convolution_single_pixel \
    --config=matrix_test.height=1 --config=matrix_test.width=1 \
    --config=matrix_test.inputs="$inputs" --config=matrix_test.outputs="$outputs" \
    --config=matrix_test.planes="$planes" --config=matrix_test.upsample="$upsample"
  iree-test-loom experimental/loom_serve/motifs/tensor/tests/convolution_matrix.loom \
    --library=experimental/loom_serve/motifs/tensor/convolution_matrix.loom \
    --device=amdgpu --target=amdgpu:gfx1151 \
    --case=@convolution_spatial_tail \
    --config=matrix_test.height=3 --config=matrix_test.width=11 \
    --config=matrix_test.inputs="$inputs" --config=matrix_test.outputs="$outputs" \
    --config=matrix_test.planes="$planes" --config=matrix_test.upsample="$upsample"
done <<'SHAPES'
96 96 3 1
192 192 3 1
384 384 3 1
192 384 3 1
384 192 1 2
192 96 1 2
SHAPES
```

These portable checks exercise the actual operand arithmetic, mapping, borders
and aliasing. Complete real-latent decoder comparisons separately report
numerical distance and final RGB differences against the independent CPU/F32
model. Preserving that model's exact arithmetic is not the image-quality goal.

[`tests/convolution_tiled.loom`](../../motifs/tensor/tests/convolution_tiled.loom) links the actual
scalar and cooperative helpers at full 96/192/384-channel reduction depths.
Four small spatial shapes (1x1, 1x2, 1x33 and 17x19) cover borders, row crossings
and partial spatial tiles; 17x19 also exercises both wider channel counts.
Each runs with and without the residual epilogue, comparing distinct-output
and genuinely aliased skip/output launches bitwise.
The driver binds tensor samples and JIT geometry together:

```sh
build_tools/bin/iree-bazel-build --config=asan \
  //loom/src/loom/tools/iree-test-loom:iree-test-loom
python -B experimental/loom_serve/motifs/tensor/tests/check_convolution_tiled.py \
  --checker=bazel-bin/loom/src/loom/tools/iree-test-loom/iree-test-loom
```

Run on the qualified execution host under its exclusive device lease. These
small exact helper checks complement the real-checkpoint decoder comparisons
below; they are not a full-image accuracy or performance measurement.

The fresh-frame contract excludes video history. The original temporal weight
planes remain in immutable storage, but only the last causal plane is read.
The two temporal-upsampling convolutions are absent from the command and its
parameter residency; the pinned reference skips them for the first frame.

```sh
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea/check_vae_decoder.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
  --checkpoint="$krea_weights" --input_results=/path/to/vae-input-results \
  --attention_results=/path/to/vae-attention-results \
  --output=/path/to/new-vae-decoder-results
```

The staged driver runs base and active-LoRA latents at 2x6 and 48x48. Its
independent F64 primitive calculations round operands to BF16 exactly where
the native residual/resize matrix leaves do; the learned shortcut and RGB
retain F32 operands. Primitive comparisons retain 2e-5 absolute/relative bounds,
and residual chains retain a 2e-5 relative-L2 bound against that matched
arithmetic. Clamping and both composed decoder commands must equal separately
executed native stages bit-for-bit. Nonfinite values always fail. The final
CPU/F32 model comparison reports numerical and pixel differences instead of
requiring the native model to reproduce F32 arithmetic.

Three actual retained base/active-LoRA latents were decoded with all wide
3x3 convolutions using ordinary BF16 matrix operands and the original
checkpoint, twice each. Attention was held at the scalar implementation to
isolate the convolution arithmetic. Relative-L2 and RGB8 errors below compare
against independent CPU/F32 decodes of those same latents;
the 1024-square case is a full-size model output, not enlarged small data.
All results were finite and the actual reference/native PNGs were inspected
without visible degradation.

| BF16-operand decoder result | Base 384x384 | Softwatercolor 384x384 | Base 1024x1024 |
| --- | ---: | ---: | ---: |
| F32 relative L2 versus CPU VAE | 0.00249737 | 0.00552388 | 0.00358484 |
| Maximum 8-bit difference | 6 | 15 | 16 |
| Pixel RMSE, 8-bit codes | 0.28642 | 0.43239 | 0.36103 |
| RGB PSNR | 58.99 dB | 55.41 dB | 56.98 dB |

These distances are observations, not universal image-quality thresholds or an
exact-pixel contract. They make the selected arithmetic's cost visible while
the portable oracle checks its indexing and operations independently.

On gfx1151, same-input ABABA dispatch profiles compare the replaced scalar
leaves with the shared matrix helper at their actual 1024-image shapes:

| Convolution shape | Scalar | BF16 matrix |
| --- | ---: | ---: |
| Middle/up0, 128 square, 384 to 384 | 98.65 ms | 10.95 ms |
| Expansion, 256 square, 192 to 384 | 313.52 ms | 24.79 ms |
| Resize0, physical 128 square, 384 to 192, scale 2 | 163.48 ms | 19.24 ms |

These are instrumented component measurements with synthetic bounded inputs,
not complete-request latency or accuracy evidence. The actual images above
provide the independent model-level witness. The helper is unchanged across
these leaves, uses 48 VGPR and 4,736 LDS bytes, and reports no spills.

The complete command has 30 unique kernels and 104 parameters occupying
286,100,492 bytes. Transient workspace is 219,709,440 bytes (209.53 MiB) at
384x384. These are correctness and memory observations, not performance data.
The 384x384 images use native eight-step latents with external text conditioning;
the 1024x1024 latent also uses source-native text encoding. The decoder does
not apply LoRA itself; it receives the latent produced by the adapted DiT.

The driver overwrites one active primitive fixture and retains final native
and reference RGB/PNGs. Its temporary storage stays below 256 MiB at these
shapes, and a failed check retains its exact inputs, command and expected
output. These decoder-only checks do not exercise native text encoding or the
image-serving adapter; those paths have separate full-request qualification.

### Encoder-taps-to-RGB composition boundary

[`sample.loom`](sample.loom) owns the reusable `krea2.sample_image` template.
Its qualification wrappers, `qualify.sample_from_taps` and
`qualify.sample_from_taps_adapted`, consume initial packed noise and BF16 taps of shape
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
python -B experimental/loom_serve/models/krea/check_sample.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
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
those different computations, not an exact-composition oracle. This observation
boundary supplies taps externally; the production image entry owns native
encoding as described below. Neither boundary constructs a prompt request yet.

Base, zero-strength LoRA and active LoRA pass twice, with zero differing bits
over 2,654,208 F32 values. Both wrappers reject all three invalid geometry cases,
six rejections in total. Twelve additional staged comparisons report numerical
differences from the earlier external-text inputs; those are not exact gates.

| Complete 384x384 command | Base | LoRA |
| --- | ---: | ---: |
| Unique kernels | 72 | 97 |
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
python -B experimental/loom_serve/models/krea/check_text_projection.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
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
The [short-attention motif](../../motifs/tensor/segmented_attention.loom) gives one wave32
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
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea/reference_text_fusion.py \
  --checkpoint="$krea_weights" --adapter="$krea_adapter" \
  --reference="$krea_reference" --output="$krea_fusion_reference"
python -B experimental/loom_serve/models/krea/check_text_fusion_blocks.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
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

The [layer-mixing motif](../../motifs/tensor/layer_mix.loom) retains twelve layer values
per lane and evaluates rank-32 A/B in registers. Adjacent
lanes own adjacent channels, preserving coalesced input access without a
transpose. A, B, scaling and base addition retain their BF16 rounding boundaries.
The production projector has no global rank tensor or transient workspace.
Its gfx1151 emitted resource report has 34 VGPRs, 24 SGPRs, zero spills,
zero LDS/private storage and 2,984 code bytes. The reported sixteen resident
waves per SIMD is a resource ceiling, not measured device utilization.

```sh
python -B experimental/loom_serve/models/krea/check_text_fusion.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
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
HF_HUB_OFFLINE=1 python -B experimental/loom_serve/models/krea/reference_text_encoder.py \
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
python -B experimental/loom_serve/models/krea/check_encoder_normalization.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
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
expanded 128-channel GQA motif for 32 query heads and eight KV heads. A
compile-time causal operand selects sequence-order masking and bounds key work
by the query tile; it adds no device ABI argument. Physical buffer bounds and
causal visibility are separate predicates. The encoder centers logits against
its valid prefix key, while the expanded noncausal refiner retains its final-key
centering. Online softmax needs neither a global score matrix nor replicated
KV heads. The command uses one kernel and zero global workspace.

```sh
python -B experimental/loom_serve/models/krea/check_encoder_attention.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
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

The original shared-motif extension reproduced the expanded noncausal DiT
results bit-for-bit at 16, 80 and 1088 rows. DiT now selects ordinary arithmetic
separately; encoder and text-refiner callers still select this unchanged
expanded variant.

### Native encoder decoder blocks

[`encoder_block.loom`](encoder_block.loom) composes scaled RMS, Q/K/V dense
projections, fused Q/K rotation, causal attention, output projection, residual
addition and SwiGLU. [`encoder_projections.loom`](encoder_projections.loom)
specializes the shared BF16 WMMA motif at five input/output shapes; layer
indices select checkpoint names without recompiling a kernel for every layer.
Independent Q/K/V paths overlap. Q/K transforms and the feed-forward product
advance source-owned buffers in place, and the residual destination may
coincide with the input after its original consumers finish. Normalization
always writes separate scratch. All ordering is in command programs.

```sh
python -B experimental/loom_serve/models/krea/check_encoder_block.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
  --checkpoint="$krea_weights" --reference=/path/to/new-encoder-reference \
  --output=/path/to/new-encoder-block-results
```

The driver qualifies every primitive on real first/late decoder inputs at
64 and 560 physical rows, then requires exact whole-command composition. It
also feeds native layer 0 into layer 1 and compares their combined, in-place
execution against separate native stages. All commands execute twice.
The matrix passed 180 independent primitive comparisons over 224,280,576 values
with no nonfinite or out-of-envelope result, and 52 exact comparisons over
68,370,432 values with no differing bits. Complete-block accumulated F64 and
canonical-model differences are reported separately, not used as primitive
tolerance passes.

A block uses 11 kernels and 11 BF16 parameter tensors occupying 201,861,632
bytes. The adjacent pair reuses those 11 kernels and doubles only the weights.
Both use 2,883,584 bytes of workspace at 64 rows and 25,231,360 bytes at 560
rows. Retained qualification output is below 128 MiB; checkpoints are reused.
This establishes decoder arithmetic and its in-place lifecycle, not the
complete embedding-to-taps encoder or native request construction.

### Native embedding-to-taps encoder

[`text_encoder.loom`](text_encoder.loom) owns embedding lookup, the first
35 decoder layers and collection of twelve taps. Its input IDs are validated
I32 values in `[0,151936)`, covering `text_tokens + 34` logical positions.
BF16 cosine/sine tables have shape `[text_tokens + 48,128]`; the byte key mask
has that physical row count and fourteen trailing zeros. The embedding kernel
initializes those trailing hidden rows to zero. Tokenization, suffix placement
and cumulative rotary positions belong to request construction, not this
low-level device command.

One source-owned hidden buffer advances in place. After layers 1, 4, ... 34,
one shared gather kernel strips the 34 prefix rows and writes into a strided
`[text_tokens,12,2560]` BF16 tap tensor. Ordinary command bindings select each
tap's byte offset; there is no per-layer kernel variant or scalar device ABI
extension. Explicit command edges finish each gather before the next layer
overwrites hidden state. The unused final decoder, final norm and vision
parameters never enter the native residency.

```sh
python -B experimental/loom_serve/models/krea/check_text_encoder.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
  --checkpoint="$krea_weights" --reference="$krea_reference" \
  --encoder_reference=/path/to/new-encoder-reference \
  --output=/path/to/new-text-encoder-results
```

The driver requires exact embedding lookup and bitwise equality between the
whole source command and separately executed native decoder blocks, twice at
16/512 retained features. The small case is a causal prefix of the canonical
request, not a newly tokenized short prompt; the full case includes the real
suffix after middle padding. Independent F64 local-block and full-chain
differences, and differences from canonical taps, are reported separately.
These accumulated observations do not replace the primitive arithmetic gates
above. One overwritten fixture plus final taps and request data retains less
than 160 MiB; checkpoints and intermediate layer histories are not copied.

Both shapes passed: eight exact comparisons cover 35,635,200 BF16 values with
zero differing bits, including embedding and whole-encoder composition.
The full command uses 13 unique kernels and one immutable text-only domain of
386 tensors occupying 7,843,069,440 bytes. Scratch is 3,211,264 bytes at 64
physical rows and 28,098,560 bytes at 560 rows: one hidden buffer plus the
single-block workspace, independent of the number of layers. Retained output
occupies 94 MiB.

At the full extent, tap relative L2 is 0.014249 against the independent F64
chain and 0.014858 against the canonical encoder. Restricting to live text
positions gives 0.012590 and 0.011924 respectively. These are accumulated
rounding observations, not primitive tolerance passes or image-quality claims.
The separately executed block observations are finite-only diagnostics; their
whole-block envelope misses are intentionally visible rather than counted as
successful independent primitive checks.

### Source-owned encoding through final RGB

The production `sample_image` and `sample_image_adapted` roots in
[`sample.loom`](sample.loom) invoke `text_encoder` and own the tap buffer until
fusion finishes. A small input-preparation kernel derives the combined key
mask: it strips the encoder's 34 prefix positions, retains the text visibility
bits and marks all image positions live. Callers cannot provide inconsistent
text masks to encoding and denoising. Mask preparation overlaps encoding;
fusion waits for both. Subsequent stages reuse the retired encoder workspace.

Checkpoint roots are ordered encoder/Turbo/VAE for base and
encoder/Turbo/LoRA/VAE for the adapted root. The encoder is not adapted.
The rebindable input order, before the distinct output and reflected workspace,
is:

| Input | Element type and shape |
| --- | --- |
| Initial packed noise | BF16 `[image_tokens,64]` |
| Validated token IDs | I32 `[text_tokens+34]` |
| Encoder cosine, then sine | Each BF16 `[text_tokens+48,128]` |
| Encoder key mask | Byte `[text_tokens+48]`, fourteen trailing zeros |
| Timesteps | BF16 `[time_count]` |
| DiT cosine, then sine | Each F32 `[block_tokens,128]` |
| Euler deltas | F32 `[time_count]` |
| LoRA strength, adapted root only | One F32 |
| VAE affine | F32 `[2,16]`: inverse standard deviation, then mean |

Output remains F32 `[3,latent_height*8,latent_width*8]` RGB in `[-1,1]`.
Only that completed output returns to the host. Request tokenization, initial
noise and the small mathematical tables come from the native caller described
above. The component check here deliberately isolates the device command from
that request producer.

```sh
python -B experimental/loom_serve/models/krea/check_encoded_sample.py \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
  --checkpoint="$krea_weights" --adapter="$krea_adapter" \
  --encoder_results=/path/to/new-text-encoder-results \
  --denoise_results=/path/to/denoise-results \
  --stack_reference=/path/to/stack-reference/1088 \
  --input_results=/path/to/vae-input-results \
  --sample_results=/path/to/new-sample-results \
  --reference="$krea_reference" \
  --output=/path/to/new-encoded-sample-results
```

This check first replays the prior raw-tap base/zero/style images exactly.
It then feeds qualified native taps into the observation wrapper and requires
the production token-input command to reproduce those final RGB values bitwise
twice, including zero-strength identity. It also checks early geometry
rejection, shared checkpoint residency and phase workspace reuse. The wrapper's
comparison against earlier images is finite-only: native versus external
encoder conditioning represents different arithmetic. Pixel differences and
final PNGs support visual inspection separately from exact composition.
The driver uses NumPy and Pillow, not an external inference framework, and
retains less than 96 MiB without copying checkpoints.

The integrated base, zero-strength and active-adapter commands each passed
twice: six exact comparisons over 2,654,208 F32 values, with zero differing
bits. The preceding raw-tap regressions cover another six exact comparisons
over the same number of values. All six invalid geometry cases fail before
weight residency. The two staged image paths also execute twice; their
comparisons against external-encoder images remain finite-only observations.

| Complete image root | Unique kernels | Parameter tensors | Weight bytes | Workspace bytes |
| --- | ---: | ---: | ---: | ---: |
| Base | 86 | 920 | 34,412,459,020 | 333,235,456 |
| Adapted | 111 | 1,448 | 34,881,682,188 | 333,235,456 |

Both inspected 384×384 images preserve the canonical deer/forest composition.
On 8-bit RGB, base/active-adapter pixel RMSE is 12.9284/8.6123 against the
canonical images and 9.6886/9.5391 against the previous native images using
external encoder taps. These are observations for one prompt, not a general
image-quality or performance claim. Retained output occupies 45 MiB.

### Prompt/seed and source preparation checks

[`check_request.py`](check_request.py) invokes the actual source VM request
producer and JITs the source preparation kernels without loading model weights.
It compares eighteen prompt/shape cases against the real Hugging Face tokenizer,
Diffusers rotary and scheduler, VAE configuration and an independently
implemented Philox/Box-Muller sequence.
The integer noise oracle first passes the published
[Random123 known answers](https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors).
Every source request is prepared twice and every GPU component executes twice.
Cases include empty input, Unicode, embedded
and terminal special tokens, long/truncated text, three seeds including all-one
bits, the exact 98/99 adaptive-stage boundary, and 32/128/512 selected text rows
at 256×384/384×384 pixels. Requested maxima are 32 and 512.

```sh
build_tools/bin/iree-bazel-build --config=asan \
  //experimental/loom_serve/image:request_check \
  //experimental/loom_serve/image:generate \
  //experimental/loom_serve/tools:component_check \
  //loom/src/loom/tools/iree-test-loom:iree-test-loom
build_tools/bin/iree-bazel-test --config=asan \
  //experimental/loom_serve/models/krea:control_test
python -B experimental/loom_serve/models/krea/check_request.py \
  --native=bazel-bin/experimental/loom_serve/image/request_check \
  --checker=bazel-bin/experimental/loom_serve/tools/component_check \
  --loom_checker=bazel-bin/loom/src/loom/tools/iree-test-loom/iree-test-loom \
  --model=experimental/loom_serve/models/krea \
  --checkpoint="$krea_weights" \
  --output=/path/to/new-request-results
python -B experimental/loom_serve/models/krea/check_generate.py \
  --native bazel-bin/experimental/loom_serve/image/generate \
  --checker bazel-bin/experimental/loom_serve/tools/component_check \
  --model=experimental/loom_serve/models/krea \
  --checkpoint="$krea_weights" --adapter="$krea_adapter" \
  --requests=/path/to/new-request-results \
  --output=/path/to/new-generation-results
```

Token IDs, masks and compact headers have exact comparisons. BF16 noise and
encoder rotary permit neighboring rounding; F32 DiT rotary has an absolute
error bound against F64 trigonometry. The scheduler and VAE constants have a
separate F32 envelope. Empty, negative, trailing-junk, fractional and overflowing
seeds are rejected before checkpoint access. Native tests additionally exercise
admission and failed-allocation ownership without a tokenizer or device. The
driver retains the actual GPU noise for the independent final-image reference.

The generation driver tests the actual prompt/seed CLI at two image/text shapes,
repeated base output, zero-strength identity and active adaptation. A separate
canonical reference process consumes the same initial noise but owns all other
conditioning and model arithmetic. Finally, the native command observer runs
twice with independently checked input files; its quantized RGB must equal the
production CLI's PPM pixels exactly. Reference pixel errors are reported and
images inspected separately, not counted as independent primitive passes.
Retained image/reference output stays below 64 MiB; checkpoints are reused.

The first fresh-prompt qualification passed all four final-pixel composition
checks. Repeated base and zero-strength PPM files were byte-identical; active
LoRA changed both images. The independently generated reference images used
the same initial noise, not a supposedly equivalent seed from another PRNG.

| Prompt / image shape / text rows | Base pixel RMSE | Active LoRA pixel RMSE | Workspace |
| --- | ---: | ---: | ---: |
| Deer in forest / 384×384 / 512 | 5.56493 | 14.21315 | 317.8 MiB |
| Brass robot in greenhouse / 256×384 / 32 | 20.82349 | 10.35997 | 137.2 MiB |

RMSE is over 8-bit RGB channels against the independent reference. Visual
inspection preserves both prompts' subjects, composition and distinct adapter
rendering, but the outputs are not reference-identical. These are bounded
numerical/ownership witnesses, not distributional quality or performance
measurements. Final image/reference artifacts occupy 15.3 MiB.
