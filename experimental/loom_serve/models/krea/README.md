# Krea 2 Turbo

This package supplies a text-to-image model to the supported `loom-serve` image
service. Its Loom programs run the text encoder and feature fusion, eight-step
DiT denoising, and VAE decoding. Intermediate features and latents remain on the
device; the host encodes the final image for HTTP. No Python inference runtime or
model-specific native executable is needed.

## Assets and deployment

The BF16 package consumes these files from a Krea 2 Turbo checkpoint directory:

```text
turbo.safetensors
text_encoder/model.safetensors
tokenizer/tokenizer.json
vae/diffusion_pytorch_model.safetensors
```

The publisher's `krea/Krea-2-Turbo` repository provides them. Access may require
accepting its terms and authenticating with Hugging Face. Downloading the original
files avoids also storing the alternate Diffusers transformer shards:

```sh
krea_weights=/path/to/models/krea-2-turbo
hf download krea/Krea-2-Turbo --local-dir "$krea_weights" \
  turbo.safetensors text_encoder/model.safetensors \
  tokenizer/tokenizer.json vae/diffusion_pytorch_model.safetensors
```

For the supported rank-32 softwatercolor adapter:

```sh
krea_adapter=/path/to/models/krea-2-softwatercolor
hf download krea/Krea-2-LoRA-softwatercolor --local-dir "$krea_adapter" \
  softwatercolor.safetensors
```

From the repository root, produce a deployment document with explicit asset
paths and geometry:

```sh
jq --arg directory "$PWD/experimental/loom_serve/models/krea" \
   --arg checkpoint "$krea_weights" \
   --arg adapter "$krea_adapter/softwatercolor.safetensors" \
   '.base_directory = $directory | .definition += {
      checkpoint: $checkpoint, adapter: $adapter,
      width: 1024, height: 1024, text_tokens: 512
    }' experimental/loom_serve/models/krea/model.json \
   > /tmp/krea-deployment.json
```

Omit the `adapter` field for an unadapted deployment. Build and start the
[supported server](../../README.md#build-and-load) with this document. An HTTP
request selects the registered alias and adapter strength:

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"krea","prompt":"A small brass robot tending red flowers in a sunlit greenhouse","seed":42,"size":"1024x1024","strength":1,"response_format":"b64_json"}' \
  | jq -r '.data[0].b64_json' | base64 --decode > /tmp/robot.png
```

Request size matches the registered geometry. A small 384×384 deployment is
useful for lifecycle checks. The source selects a short or long text stage from
the prepared token count; both share parameter domains. An adapter is prepared
once with the model, not reloaded for every image. Strength zero selects the base
path; nonzero strength applies the loaded adapter. Arbitrary adapter ranks,
multi-adapter request stacks and reference-image input need different source or
request contracts and are not accepted by this package.

## Low-memory VAE decoding

The default deployment uses the untiled decoder. To trade additional decoder
work for lower transient memory, give the model a positive workspace limit in
bytes:

```json
"options": {
  "workspace_limit_bytes": 805306368
}
```

When the full decoder exceeds the limit, preparation selects an exact
overlapping tile specialization. The tiled path preserves the untiled pixels;
it does not blend or approximate tile boundaries. The compiler still enforces
the limit for the complete stage and rejects a deployment that cannot satisfy
it. This option can share the same `options` object with the INT8 checkpoint
selection below.

## INT8 ConvRot DiT

The encoder, tokenizer and VAE are unchanged. Instead of `turbo.safetensors`,
the DiT can use `diffusion_models/krea2_turbo_int8_convrot.safetensors` from
`Comfy-Org/Krea-2`. Set this source-owned option in the deployment:

```json
"options": {
  "int8_convrot_checkpoint": "/path/to/krea2_turbo_int8_convrot.safetensors"
}
```

This is not a request to quantize an arbitrary checkpoint. It selects the
checkpoint's signed I8 projection matrices, F32 channel scales and Hadamard-256
rotation. Row quantization and I32 matrix accumulation remain on the device;
adapters use the original unrotated activations. The loader does not expand a
second BF16 copy of the DiT. `prepare.loom` validates this option; native serving
code does not interpret Krea-specific quantization names.

## Source map

`prepare.loom` defines stages, parameter domains and specialization;
`weights.loom` defines parameter placement and preparation. `control.loom`
validates and frames requests and drives prepared execution. `generate.loom`
composes text encoding, denoising and final image production. Encoder, DiT,
adapter and VAE numerical leaves are grouped by their stage names. Shared
attention, projection, convolution, quantization and image helpers live in
`../../motifs/tensor/` and receive dimensions as SSA/template arguments.

The named roots are independently compilable for reports and numerical checks.
Traversal selection and tile shapes belong in those programs, where JIT
specialization can choose them without a new native server branch.

## Mixed-model lifecycle check

This test runs an isolated image baseline, an isolated text baseline, then
Qwen → Krea → Qwen under a shared retained-memory cap. It checks exact image
pixels and text responses, checkpoint reuse after weight eviction, prepared and
queued cancellation, health during device execution, and joined shutdown.
It uses Python's standard library plus Pillow for independent PNG decoding.

Build the exact ASan server and run from the repository root on an exclusive
GFX11 execution host. Existing checkpoint files are read in place:

```sh
iree-bazel-build --config=asan //loom/src/loom/tools/loom-serve:loom-serve
PYTHONPATH=loom/src/loom/serve/testing python3 -B \
  -m experimental.loom_serve.models.krea.check_router \
  --server=bazel-bin/loom/src/loom/tools/loom-serve/loom-serve \
  --text-source=experimental/loom_serve/models/qwen \
  --weights=/path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --tokenizer=/path/to/qwen/tokenizer.json \
  --image-source=experimental/loom_serve/models/krea \
  --checkpoint="$krea_weights" \
  --adapter="$krea_adapter/softwatercolor.safetensors" \
  --workers=1 --memory-bytes=38654705664 --output=/tmp/mixed-model-check
```

The output directory must not exist. It retains deployment documents, response
records, server JSONL and images, not model copies. The image case uses the prompt
`a red fox in the snow`, seed zero and strength zero at 384×384, with adapter
weights loaded but the base path selected. `--reference-image` can select an
independently generated matching RGB reference instead of starting the isolated
image server. This differential check establishes serving ownership and
composition; the source numerical cases separately establish arithmetic and
nonzero adapter contributions. It is not a performance benchmark.
