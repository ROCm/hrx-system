# Exercise incremental speech output

This package produces a deterministic waveform through the ordinary speech
endpoint, without downloading weights. It exercises source request preparation,
reusable conditioning, independent stream state, device feedback, PCM encoding
and HTTP delivery. It does not synthesize intelligible speech or establish
voice-cloning quality.

From the repository root with the CPU task HAL and VM/x86 targets enabled:

```sh
iree-bazel-run --config=asan //loom/src/loom/tools/loom-serve -- \
  --device=task --pool_backing=fixed \
  --model=loom/src/loom/serve/fixture/speech/model.json \
  --listen=127.0.0.1:8080 --workers=1
```

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"speech","input":"hi","voice":"positive","seed":3,"response_format":"pcm"}' \
  --output /tmp/speech-fixture.pcm
```

The output has 17 mono frames at 24 kHz: 34 little-endian PCM16 bytes. For frame
`i`, its signed sample is `((i + 3) % 8) * 4096`. Selecting `negative` negates
those samples. In general, an input of `N` UTF-8 bytes produces `8*N + 1` frames;
the seed selects the phase modulo eight. Input length is 1–1024 bytes. The
fixture accepts speed 1 and empty instructions, rejecting other source controls.

The two voice gains are immutable conditioning entries shared across requests.
The first device stage initializes fresh stream state and emits no feedback.
Later units alternate capacities of three and five frames, with an empty
nonterminal second block and a shortened final block. Generation and publication
use two kernels sharing compiler-private command workspace. Native serving
code has no fixture selector, sample formula or special-case response.

The [speech service contract](../../service/speech/README.md) describes ownership
and the explicit PCM format. The [model header](../../service/speech/model.h)
defines cold and warm source signatures; the package's three `.loom` files
implement them directly.

## Verification

The `source_speech_service_test` suites under
`//loom/src/loom/tooling/target/{cpu,amdgpu,spirv}/serve` run this package on the
selected native provider. Their shared tests cover exact variable/empty-block
output, source rejection, first bytes before EOF, reset/reuse, dynamic model
registration/removal and shutdown. Each process owns one shared application;
the base and `source_speech_service_parallel_test` registrations use one and
two workers respectively.

The same tests also run the weighted source variant. Initialization has four
fixed roots, while decoding has two: a shared base factor, disjoint encoder and
decoder tensors composed from a second file, a separate voice archive with the
same factor key, and an unprepared view of the base factor. Source declarations
intentionally use a different order from fixed bindings. In-place preparation
doubles only the factor tensors; the exact waveform exposes repeated preparation,
incorrect binding order or sharing across different checkpoint file lists or
policies.
The tiny checkpoints are authored by the test, without external downloads.

Additional dependency-controlled cases use real held HAL transfers with the
production service and TCP carrier. They verify health progress during device
waits, bounded send-credit delivery, cancellation cleanup and failed-readiness
ownership. Numerical model output is the substituted dependency in those cases;
source compilation and exact waveform results are covered by this package.

## Uploaded voices

`voice_card.json` selects speech ABI2 and two dynamically registered voices. It
accepts the same raw PCM and enabled audio containers as dictation. CPU Loom
computes the recording's mean sample; a numerical command stores that gain in
the reserved conditioning slot. Subsequent waveform requests read the stored
gain through the ordinary fixture kernel. A mean of `0.125` produces the same
positive oracle above. This isolates one-time processing and device ownership
without a learned checkpoint.

The native and HTTP tests verify publication after retirement, stereo and
multichannel input, S16/F32 and WAV, repeated synthesis, duplicate IDs, capacity
pressure, cancellation during accepted processing, joined shutdown, and removal
with queued captures. Text, bytecode, and mixed-provider card tests repeat the
HTTP upload-to-speech flow through the actual server binary, including removal
while inference is drained. The fixture qualifies the serving contract; Pocket's
learned encoder separately qualifies model-specific voice preparation.
