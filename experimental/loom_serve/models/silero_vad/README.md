# Silero voice activity

This package runs Silero VAD's learned waveform frontend, convolutional encoder,
and recurrent classifier in Loom on the CPU. Each continuing audio stream emits
a speech probability with its covered input-frame interval. A conversation owner
can use these observations to retain pre-roll, open a transcription request, end
an utterance, or interrupt an answer.

Eight requests can be resident together. Their compatible next hops share a
numerical batch even when their capture positions and retained histories differ.
Each resident row owns its waveform context and recurrent state. Changing packed
positions does not move an entire history, and a stalled observation consumer
blocks only its own next hop. Cancellation keeps accepted work alive through
retirement before the row becomes reusable.

| Input | Hop | Retained waveform context | State per resident row |
| --- | --- | --- | --- |
| 8 kHz mono | 256 frames / 32 ms | 32 frames | 1,152 bytes |
| 16 kHz mono | 512 frames / 32 ms | 64 frames | 1,280 bytes |

The two rate families use their corresponding original F32 weights from
[Silero VAD](https://github.com/snakers4/silero-vad/tree/5cd7945676eb32225748052e2e6a0580e4686a08).
The checkpoint is pinned by revision and SHA-256. Its [MIT license](LICENSE.silero)
is included with the package and prepared assets. Learned execution and changing
stream membership are qualified on the x86 CPU task device.

## Prepare and load

From the repository root, prepare both rate families in one new directory:

```sh
activity_assets="$HOME/models/loom-prepared/silero-vad/default"
uv run --python 3.13 experimental/loom_serve/models/silero_vad/prepare_assets.py \
  --output "$activity_assets"

iree-bazel-run --config=opt //loom/src/loom/tools/loom-serve -- \
  --device=task --pool_backing=fixed --workers=8 --listen=127.0.0.1:8080 \
  --model="$activity_assets/model_16000.json"
```

Preparation downloads the 2.3 MB pinned ONNX checkpoint, extracts its named
tensors, pads matrix rows for the CPU contraction, and writes Safetensors plus
two deployment documents. `--onnx /path/to/silero_vad.onnx` uses an existing
checkpoint offline, with the same integrity check. `--name` changes the model
name prefix; the rate remains its suffix. Preparation preserves existing
deployment directories and publishes documents after their assets are complete.

Python, NumPy, and ONNX are preparation tools. The running server loads the
prepared parameters and executes the source programs without a Python model or
ONNX Runtime. The original checkpoint and generated parameters stay beneath the
chosen asset directory.

In a multi-device server, `--device 0` on the preparation command assigns the
activity documents to the first device. For example, a server configured with
`--device=task --device=amdgpu --pool_backing=fixed` can place activity on CPU
and a recognition model on device 1. Device placement, residency, and model
admission use the same
[deployment contract](../../../../loom/docs/src/serving/model-documents.md) as
the other model families.

## Consume activity

The deployment kind is `activity`. Model management and readiness use the
ordinary registry. Inference is native: open a continuing recognition request
with `LOOM_SERVE_RECOGNITION_OUTPUT_ACTIVITY`, honor input credit, and consume
`LOOM_SERVE_RECOGNITION_EVENT_ACTIVITY` records. The
[recognition contract](../../../../loom/src/loom/serve/service/recognition/README.md)
describes stream admission, event ownership, and retirement.

An observation contains a positive `[begin, end)` interval at the input sample
rate and a probability in `[0, 1]`. That interval describes evaluated audio;
its endpoints are not precise speech onset or word boundaries. Thresholds,
pre-roll, hangover, and turn policy belong to the conversation owner. There is
no transcription HTTP endpoint for this deployment and no fabricated transcript
attached to a score.

The native input is mono F32 PCM at the selected rate. A microphone owner can
convert capture audio once and share the normalized samples with a recognizer
at the same rate. Conversion history belongs to the microphone stream across
utterance boundaries. Different model input rates require separate conversion
clocks.

Explicit EOF flushes a partial hop by padding its numerical input with zeros.
The emitted interval covers only real accepted frames. Empty EOF emits no
observation and retires without a numerical invocation. Final caller-owned
events remain readable after model removal.

## Numerical and state layout

The complete hop performs a reflected STFT, four convolution/ReLU stages, an
IFGO LSTM step, and a sigmoid classification head. Twelve output accumulators
reuse each sixteen-column activation load. Immutable padded weights are bound
once; hops never copy the checkpoint. Input and recurrent gate projections use
the same contraction with separate operands.

Each invocation gathers the selected rows' small waveform/recurrent state into
packed workspace, executes the hop, and writes each result back to its original
resident row. Inactive columns are initialized workspace. A request's first
positive hop starts from zero history; reused rows cannot inherit a predecessor.
Per-invocation workspace is separate from persistent history and immutable
weights. The source selects the fixed eight-row geometry and 32 ms quantum;
neither is encoded in the native service's event contract.

The model accepts no language, prompt, sampling temperature, or model options.
Its probability is an observation, so a silence stream continues producing
scores until the input owner finishes it.

## Check the source

```sh
iree-bazel-test --config=asan //experimental/loom_serve/models/silero_vad/...
```

The package checks exercise the actual numerical programs, resident state,
input readiness, PCM padding, and complete activity records without downloading
a checkpoint. Both rates cover empty, partial, full, and excess input with and
without EOF, including frame positions beyond 32 bits. Learned qualification
additionally compares per-hop probabilities against the pinned ONNX oracle,
checks bitwise isolated versus changing-cohort histories, and uses actual
activity intervals to select
Moonshine input. The selected recording produces the same final text as its
complete-input control; silence opens no transcription request. These establish
numerical and ownership behavior, not a conversation latency guarantee.
