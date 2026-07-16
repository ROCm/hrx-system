# Ideogram 4 Quickstart

This path downloads the model files consumed by the current implementation,
builds the optimized CLI, and runs one complete 1024x1024 generation directly
from those files. No parameter conversion or second model-sized copy is
required.

The measured configuration keeps Qwen and DiT weights in compact FP8 execution
layouts, uses BF16 activations and F32 accumulation, and retains all stage
parameter bundles on one AMD GPU. It was verified on a Radeon Pro W7900 with a
sampled 1024x1024 physical VRAM peak of about 25.8 GiB. The selected checkpoint
files occupy 29.52 GB on disk.

Run every command in this guide from the repository root. The CLI passes file
paths directly to the ordinary IREE file loaders: relative `--flagfile`,
`--tokenizer`, and `--parameters` paths are resolved from the process working
directory, not from the executable or the flagfile containing them.

## Configure And Build

Install the repository's Bazel tools and configure the AMDGPU HAL and Loom
AMDGPU target. Replace `/opt/rocm` when the SDK is elsewhere:

```bash
python dev.py bazel setup
python dev.py bazel configure \
  -DIREE_HAL_DRIVER_AMDGPU=ON \
  -DLOOM_TARGET_AMDGPU=ON \
  -DIREE_ROCM_PATH=/opt/rocm \
  -DIREE_ROCM_DEPENDENCY_MODE=pinned
python dev.py bazel build \
  -c opt \
  --features=thin_lto \
  --copt=-O3 \
  --cxxopt=-O3 \
  --host_copt=-O3 \
  --host_cxxopt=-O3 \
  --copt=-march=native \
  --cxxopt=-march=native \
  --host_copt=-march=native \
  --host_cxxopt=-march=native \
  //experimental/id4/binding/cli:id4
```

These are the host optimization flags used for the reported performance lane;
`-march=native` discovers the build host's CPU and does not select a GPU. The
optimized executable is `bazel-bin/experimental/id4/binding/cli/id4`.

## Download The Models

The Ideogram FP8 repository is gated under its non-commercial license. Accept
the license on the
[model page](https://huggingface.co/ideogram-ai/ideogram-4-fp8), authenticate
the Hugging Face CLI, and download the conditioned and unconditioned DiTs from
the pinned snapshot:

```bash
hf auth login
hf download ideogram-ai/ideogram-4-fp8 \
  transformer/diffusion_pytorch_model.safetensors \
  unconditional_transformer/diffusion_pytorch_model.safetensors \
  --revision ee79a7237b519f1402ceacf952f30c8a31ec5073 \
  --local-dir models/ideogram-4-fp8
```

The current Qwen path consumes the stock block-scaled FP8 checkpoint. Download
its tokenizer, index, and two weight shards at the pinned revision:

```bash
hf download Qwen/Qwen3-VL-8B-Instruct-FP8 \
  tokenizer.json \
  model.safetensors.index.json \
  model-00001-of-00002.safetensors \
  model-00002-of-00002.safetensors \
  --revision 9cdc6310a8cb770ce18efaf4e9935334512aee45 \
  --local-dir models/qwen3-vl-8b-instruct-fp8
```

Ideogram's bundled `text_encoder/model.safetensors` is not downloaded. It uses
the base-model `language_model.*` key schema and per-output-row FP8 scales; the
current Qwen parameter path consumes the stock checkpoint's
`model.language_model.*` keys and 128x128 block inverse scales. Both checkpoints
have the same Qwen3-VL text topology, but their source quantization contracts
are different.

Finally, accept the FLUX.2 non-commercial license on the
[FLUX.2-dev model page](https://huggingface.co/black-forest-labs/FLUX.2-dev)
and download its single-file autoencoder:

```bash
hf download black-forest-labs/FLUX.2-dev \
  ae.safetensors \
  --revision 26afe3a78bb242c0a8bb181dcc8937bb16e5c66c \
  --local-dir models/flux2-dev
```

Ideogram's bundled `vae/diffusion_pytorch_model.safetensors` is not yet the
parameter contract consumed by this decoder. It stores the same Flux2 decoder
weights in Diffusers names and BF16 source tensors, while the current parameter
preparation path accepts the original Flux names and F32 source tensors. The
pinned FLUX.2 `ae.safetensors` file has SHA-256
`868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3`.

## Model Locations And Flagfiles

The download commands above create the layout expected by the checked
flagfile:

```text
models/
  ideogram-4-fp8/
    transformer/diffusion_pytorch_model.safetensors
    unconditional_transformer/diffusion_pytorch_model.safetensors
  qwen3-vl-8b-instruct-fp8/
    tokenizer.json
    model.safetensors.index.json
    model-00001-of-00002.safetensors
    model-00002-of-00002.safetensors
  flux2-dev/
    ae.safetensors
```

There is no directory search or implicit Hugging Face cache lookup. The checked
[`ideogram4-fp8-policy.flags`](ideogram4-fp8-policy.flags) file contains only
the measured compact-FP8 execution and residency policy. The ready-to-run
[`ideogram4-fp8.flags`](ideogram4-fp8.flags) file includes that policy and maps
the default paths above onto the runtime's tokenizer, Qwen, conditioned DiT,
unconditioned DiT, and VAE parameter scopes. Both are ordinary IREE flagfiles,
not separate model formats or CLI code paths.

When an existing model directory already has the layout above, a repository-
root symlink preserves the checked command unchanged:

```bash
ln -s /absolute/path/to/model-directory models
```

For any other layout, copy the ready-to-run flagfile to a local file and replace
its tokenizer and four parameter paths with absolute paths. Keep the policy
include unchanged while running from the repository root:

```bash
cp experimental/id4/docs/ideogram4-fp8.flags /path/to/id4-local.flags
```

The resulting local file has this shape:

```text
--flagfile=experimental/id4/docs/ideogram4-fp8-policy.flags
--tokenizer=/absolute/path/to/qwen/tokenizer.json
--generation_parameter_source=checkpoint
--parameters=qwen=/absolute/path/to/qwen/model.safetensors.index.json
--parameters=dit_cond_fp8=/absolute/path/to/id4/transformer/diffusion_pytorch_model.safetensors
--parameters=dit_uncond_fp8=/absolute/path/to/id4/unconditional_transformer/diffusion_pytorch_model.safetensors
--parameters=vae=/absolute/path/to/flux2/ae.safetensors
```

Select it by replacing the checked `--flagfile` argument with
`--flagfile=/path/to/id4-local.flags`. Qwen shard names inside
`model.safetensors.index.json` are resolved relative to that manifest, so the
two shard files should remain beside it.

Do not load `ideogram4-fp8.flags` and then append replacement `--parameters`
flags. `--parameters` is a repeatable list, so additional values accumulate
instead of replacing the checked paths. Use either the checked complete
flagfile or one complete local copy. Running the executable outside the
repository root also requires making the nested policy flagfile path absolute.

## Select A GPU

The generic URI selects the AMDGPU driver's default device:

```text
--device=amdgpu://
```

When several GPUs are visible, list their discovered canonical URIs and select
one explicitly:

```bash
bazel-bin/experimental/id4/binding/cli/id4 --list_devices=amdgpu
```

Pass the resulting `amdgpu://GPU-...` URI. `ROCR_VISIBLE_DEVICES=0` can also
constrain the process before using the generic URI. The runtime queries the
selected target and asks Loom for a compatible kernel provider; the model
flagfile does not encode a GPU architecture.

An unpacked SDK whose HSA runtime is outside the dynamic-loader search path can
provide it explicitly:

```text
--amdgpu_libhsa_search_path=/path/to/libhsa-runtime64.so.1
```

## Generate From JSON

Run the checked 1024x1024 structured request:

```bash
bazel-bin/experimental/id4/binding/cli/id4 \
  --flagfile=experimental/id4/docs/ideogram4-fp8.flags \
  --device=amdgpu:// \
  --prompt_json_file=experimental/id4/docs/requests/long_1024.json \
  --output=ideogram4.ppm
```

The result is a binary RGB PPM. The CLI validates every decoded pixel as finite
before writing it. The current process owns one generation and exits, so its
wall time includes model indexing, parameter loading and layout encoding, Loom
compilation, generation, VAE decode, readback, and output.

The checked request uses Ideogram's structured caption form, including
normalized bounding boxes, the 20-step sampler, and a deterministic seed. Other
requests under `experimental/id4/docs/requests/` provide 128, 256, and 512 pixel
plumbing cases; the 1024 request is the representative image path.

## Generate From A Prompt

Plain text can be supplied directly. Width and height are diffusion latent
dimensions; each position decodes to 16 pixels along each axis, so `64x64`
produces `1024x1024`:

```bash
bazel-bin/experimental/id4/binding/cli/id4 \
  --flagfile=experimental/id4/docs/ideogram4-fp8.flags \
  --device=amdgpu:// \
  --prompt="A letterpress poster reading LOOM, with crisp black type and a red geometric border." \
  --generation_latent_width=64 \
  --generation_latent_height=64 \
  --generation_sampler=V4_DEFAULT_20 \
  --generation_seed=20260715 \
  --output=loom-poster.ppm
```

This sends the text verbatim through the local Qwen encoder. It does not call
Ideogram's hosted magic-prompt service or expand the text with another LLM.
Structured captions generally provide stronger composition and typography
control.

## Use A LoRA

The dynamic path consumes Ideogram 4 LoRA safetensors while loading the same
base checkpoint. The DeverStyle Archer adapter is the current real-file smoke
case and recommends strength `0.6`:

```bash
hf download DeverStyle/Ideogram-4.0-Loras \
  "dever_archer_ideogram4 (dvr_arch_tv).safetensors" \
  --revision 54c5d62741ff58faa0a2e6b107c6a71ff272d617 \
  --local-dir models/ideogram4-loras

bazel-bin/experimental/id4/binding/cli/id4 \
  --flagfile=experimental/id4/docs/ideogram4-fp8.flags \
  --device=amdgpu:// \
  --lora="models/ideogram4-loras/dever_archer_ideogram4 (dvr_arch_tv).safetensors" \
  --lora_strength=0.6 \
  --prompt_json_file=experimental/id4/docs/requests/long_1024.json \
  --output=ideogram4-archer.ppm
```

This adapter patches 204 conditioned-DiT linear targets. On the verified
machine, the exact 1024x1024 command above completed in about 270 seconds, and
its dry-run logical peak was 26,681 MiB versus 25,874 MiB without an adapter.
These are current dynamic-path numbers; the base-path timing and memory claims
do not silently include LoRA execution.

Repeat `--lora` and `--lora_strength` in matching order to compose adapters:

```text
--lora=first.safetensors  --lora_strength=0.6 \
--lora=second.safetensors --lora_strength=0.25
```

Omitting all strengths selects `1.0` for every adapter. Each file is validated
against the Ideogram 4 conditioned-DiT dimensions before execution.

## Optional Execution-Layout Cache

Parameter baking is optional. It writes target- and policy-specific IRPA files
that already contain the compact execution layouts, which can be useful for a
long-lived deployment that values repeatable startup over disk capacity.

It is deliberately not the quickstart path. On the verified machine, five
hot-cache 128px runs measured a `1.27 s` median checkpoint penalty; two
controlled cold-cache pairs averaged a `0.37 s` penalty; and the representative
1024px run measured `93.25 s` from checkpoints versus `92.90 s` from baked
archives. Every compared output was byte-identical. Baking took `23.12 s` and
created another 26.635 GB of archives.

Create that cache only when its deployment tradeoff is useful:

```bash
bazel-bin/experimental/id4/binding/cli/id4 \
  --flagfile=experimental/id4/docs/ideogram4-fp8.flags \
  --device=amdgpu:// \
  --prompt_json_file=experimental/id4/docs/requests/long_1024.json \
  --bake_parameter_layout_directory=models/ideogram-4-fp8-layouts
```

Then select the checked archive flagfile for generation:

```bash
bazel-bin/experimental/id4/binding/cli/id4 \
  --flagfile=experimental/id4/docs/ideogram4-fp8-execution-layout.flags \
  --device=amdgpu:// \
  --prompt_json_file=experimental/id4/docs/requests/long_1024.json \
  --output=ideogram4.ppm
```

Regenerate the cache after changing the selected target, parameter execution
format, or another layout policy. The archives are reusable across prompts and
the checked image sizes; they are not generated per request.

## Request JSON

A request has a prompt and generation settings:

```json
{
  "prompt": "A small red boat on a quiet lake at sunrise.",
  "generation": {
    "latent_width": 64,
    "latent_height": 64,
    "sampler": "V4_DEFAULT_20",
    "seed": 20260715
  }
}
```

`prompt` may be a string or a structured JSON object. Structured bounding boxes
use normalized top-left-origin coordinates `[y1, x1, y2, x2]`, with each value
in `[0, 1000]`. The checked [1024 request](requests/long_1024.json) demonstrates
descriptions, per-element boxes, and palette control. Ideogram's upstream
[prompting guide](https://github.com/ideogram-oss/ideogram4/blob/main/docs/prompting.md)
documents the complete caption form.

## Planning And Diagnostics

`--dry_run` parses and tokenizes the request and builds the complete program and
resource plan without opening model parameter files or issuing GPU work:

```bash
bazel-bin/experimental/id4/binding/cli/id4 \
  --flagfile=experimental/id4/docs/ideogram4-fp8.flags \
  --device=amdgpu:// \
  --prompt_json_file=experimental/id4/docs/requests/long_1024.json \
  --dry_run \
  --dump_plan=ideogram4-plan.json
```

Failure evidence can be retained with `--dump_diagnostics=<directory>`,
`--dump_result_summary=<file>`, `--dump_result_tensors=<directory>`, and
`--profile_output=<file>`. These modes add diagnostic work and are not the
performance path.

This snapshot has full-model evidence on `gfx1100`; other AMDGPU targets work
when their required Loom providers are available. Missing target providers fail
explicitly rather than selecting a hidden architecture or precision fallback.
The [case study](case_study.md) records the current correctness, memory, and
performance evidence.
