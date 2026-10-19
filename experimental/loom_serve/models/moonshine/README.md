# Moonshine streaming recognition

This package runs Moonshine Streaming Tiny's learned waveform frontend, encoder,
and autoregressive decoder in Loom. It accepts 16 kHz mono audio incrementally,
retains each conversation's history, and publishes revisable transcripts followed
by a final utterance. The ordinary `/v1/audio/transcriptions` endpoint collects
the final text for dictation clients such as Open WebUI.

Four requests can be resident at once. Ready encoder quanta batch together, as do
ready decoder tokens, even when requests have different input lengths, encoder
positions, or decoder positions. A decoder batch can verify sixteen token
positions across those requests, using each request's previous hypothesis as a
proposal for the next revision. Each resident owns its convolution history,
encoder rings, attention memory, and decoder cache. A short recording can finish
while a longer recording continues. Cancellation and output credit remain
independent.

Native consumers select transcript output and receive ordered revision events.
Each event describes a replacement hypothesis at the model's input clock; the
final document remains available after model retirement. Turn detection can run
as an independent activity model on another device, sharing the server's worker
pool. The microphone owner supplies pre-roll and decides when to finish an
utterance, while Moonshine retains its own encoder and decoder history.

The package consumes the original F32 tensors from
[Moonshine Streaming Tiny](https://huggingface.co/moonshine-ai/moonshine-streaming-tiny/tree/f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a)
at revision `f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a`. The numerical reference is
the eager F32 implementation in Transformers 5.19.0. No Python inference model
runs inside the server. Learned execution is qualified on AMDGPU; CPU and NPU
qualification requires running these numerical programs on those backends, in
addition to the service's device-independent lifecycle tests.

## Prepare and serve

From the repository root:

```sh
moonshine_assets="$HOME/models/loom-prepared/moonshine-streaming-tiny/default"
uv run --python 3.12 experimental/loom_serve/models/moonshine/prepare_assets.py \
  --output "$moonshine_assets"

iree-bazel-run --config=opt //loom/src/loom/tools/loom-serve -- \
  --device=amdgpu --workers=8 --listen=127.0.0.1:8080 \
  --model="$moonshine_assets/model.json"
```

Preparation links cached checkpoint, tokenizer, and configuration files into a
new deployment directory, writes a 64-byte rotary frequency tensor, and records
input hashes. It requires Hugging Face Hub and Safetensors, without PyTorch,
Transformers, NumPy, or an audio library. `--offline` uses already-cached assets;
`--name` sets the served model name. An existing deployment is never overwritten.
The running server needs only the deployment assets and this source package.

With optional audio decoding enabled, upload WAV, MP3, or FLAC:

```sh
curl --fail-with-body -H 'Expect:' \
  http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=moonshine-streaming-tiny \
  -F file=@recording.wav
```

The result is a JSON object such as `{"text":"Good day."}`. The decoder mixes
channels and resamples encoded recordings to the model's 16 kHz mono format.
Without optional codecs, the same endpoint accepts signed little-endian PCM16.
Its declared sample rate and channel count drive native conversion to the
model's 16 kHz mono geometry:

```sh
curl --fail-with-body -H 'Expect:' \
  http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=moonshine-streaming-tiny \
  -F 'file=@recording.pcm;headers="Content-Type: audio/pcm;rate=16000;channels=1"'
```

The MIME parameters are required for raw PCM. Curl's `headers=` form preserves
them in the file part. The shared [HTTP profile](../../../../loom/docs/src/serving/http-profiles.md)
documents response formats, admission errors, and optional codec configuration.

## Open WebUI dictation

Select the OpenAI speech-to-text engine in Open WebUI's audio settings:

| Setting | Value |
| --- | --- |
| API base URL | `http://HOST:8080/v1`, reachable from the backend |
| API key | A nonempty local value, or the key required by the reverse proxy |
| Model | `moonshine-streaming-tiny` |
| Request format | Multipart |
| Language | Empty or `en` |

Use the codec-enabled server for the ordinary microphone workflow: Open WebUI
preprocesses recordings into MP3 before uploading them. Browser dictation through
the authenticated backend has been exercised with this model, including the
captured audio's independent F32 reference. The finite transcription endpoint
receives the complete upload before feeding its decoded quanta into the
continuing recognition service; it does not expose a live microphone transport
by itself.

## State, batching, and bounds

This package selects a fixed bounded profile. Unknown model options fail during
preparation rather than silently leaving the source geometry unchanged.

| Property | Value |
| --- | --- |
| Input | 16 kHz, mono F32 inside the service |
| Numerical input quantum | 320 samples, or 20 ms |
| Resident requests / maximum compatible cohort | 4 / 4 |
| Decoder activation capacity | 16 token positions, up to 4 per request |
| Input bound | 655,360 samples, or 40.96 seconds |
| Encoder positions | 2,048 |
| Decoder hypothesis capacity | 512 tokens, including EOS |
| Provisional hypothesis cadence | An attempt every 6,400 consumed samples, or 400 ms |
| Provisional prediction budget | 6.5 predictions per second of published encoder memory, rounded up |
| Per-resident device history | 40,320,208 bytes |
| Language and decoding | English, empty prompt, temperature zero |

These storage bounds belong to this source package. The shared recognition
service does not impose them on other models. A request exceeding the input
bound or final hypothesis capacity fails explicitly. The checkpoint provides
neither diarization nor word alignment. The source emits one stable utterance
ID, increasing revisions, and the real consumed sample range; it does not
invent speakers or word timestamps.

An incomplete audio prefix can make the decoder repeat tokens without reaching
EOS. Provisional attempts have a duration-based work budget: if an attempt
exhausts that budget, the source keeps its last published transcript and resumes
encoding new audio. The next attempt starts with an empty decoder draft against
the updated encoder memory. It neither publishes a truncated hypothesis nor
marks the utterance complete. Final input retains the full 512-token capacity
and requires EOS; a final hypothesis that exhausts that capacity returns an
error. Other conversations retain their independent progress throughout.

The frontend normalizes 80-sample frames, applies learned compression and a
projection, and runs two causal stride-two convolutions. Six encoder layers
retain 32-position rings. Four layers need three positions of right context
each, totaling 240 ms of model lookahead. EOF drains those layers while preserving
the real input clock, including a final partial quantum.

Published encoder memory and cross-attention projections are immutable. Each
new hypothesis recomputes its decoder prefix against the available memory,
including all decoder layers and their KV. The previous hypothesis supplies
token proposals; its higher-layer decoder KV cannot be reused after the encoder
memory changes.

Each request offers up to four consecutive token positions. The decoder computes
their causal predictions together, accepts matching proposals and the first
replacement token, and advances that request's accepted cache frontier. After a
correction, the remaining suffix gets another verification opportunity against
the corrected prefix. Two consecutive mismatches abandon the suffix; a verified
match restores the retry allowance. Every accepted token is therefore a fresh
greedy prediction. Rejected KV stays allocated beyond the accepted frontier and
is overwritten before continuation can read it. No history copy or allocation
is needed when a proposal fails.

Decoder activation rows carry separate positions and valid lengths. Several
rows may refer to one resident history without giving another request that
history's cursor. Work packets map packed activation rows to resident histories;
the CPU completion plan also retains each original request's first activation
and activation count. Both mappings stay alive through actual device retirement,
including after cancellation, so completion uses the original ordinal even when
another request has left. Reset and encoder phases retain their four-row shape.

Layer normalization assigns one workgroup to each packed row; decoder softmax
assigns one to each attention head. Lanes retain their input values through the
reductions, so intermediate moments and exponentials do not make extra trips
through device memory. Encoder readiness still follows resident identities,
while activations and decoder work follow the accepted packed mapping.

Decoder projections share one input panel across sixteen output channels and
reuse each weight across sixteen packed token positions. Eight lanes cooperate
on each output, retaining separate F32 accumulators and the same balanced
reduction order for every position. Inactive panel rows are zero-filled, so the
inner arithmetic needs no per-row branch; only live outputs are stored. This
covers attention projections, the feed-forward layers, and the vocabulary head
without copying resident histories. Input and encoder projections retain their
individual readiness gates.

`control.loom` runs request policy and tokenizer decoding on the shared CPU
executor. `encoder.loom`, `decoder.loom`, and `reset.loom` compose the numerical
phases from shared kernels with explicit layer offsets. The service owns async
device execution, input/output credit, cancellation, and retirement. Audio file
decoding and raw PCM conversion are bounded shared-executor work. Worker count
does not reserve compute for voice or establish a latency guarantee.

## Check the package

Checkpoint-free tests cover frame and layer normalization, masked decoder
softmax, every decoder projection shape, and independent lookahead/EOF with
reordered resident identities. Closed-form numerical references check one
through sixteen projection rows, bias, inactive outputs, and unready encoder
rows. Decoder-state tests cover repeated resident identities at different causal
positions, the last cache position, embedding selection, and greedy tie breaking.
Projection, reduction, and decoder-state tests also instrument memory access
and synchronization:

```sh
iree-bazel-test --config=asan //experimental/loom_serve/models/moonshine/...
```

For learned endpoint checks, prepare a JSON array of exact input/reference pairs:

```json
[
  {"audio":"short.wav","text":"Good day."},
  {"audio":"long.wav","text":"The complete reference transcript."}
]
```

Run against the loaded server, with up to four concurrent uploads:

```sh
python experimental/loom_serve/models/moonshine/check_service.py references.json \
  --base-url http://127.0.0.1:8080/v1 --workers=4
```

The check sends chunked multipart requests and compares complete JSON text
exactly. `--workers=1` exercises the same inputs in isolation. PCM entries specify
`"content_type":"audio/pcm;rate=16000;channels=1"`. References must describe the
actual samples after preprocessing: different resamplers can change punctuation,
so text normalization is not used to hide input differences.

The learned qualification includes differing short recordings, a 22.56-second
paragraph, changing cohorts, cancellation after submission, stalled-reader
resumption, and row reuse. Correctness checks do not establish throughput or
interactive latency; those require an optimized server and an isolated runner.

The upstream [Moonshine license](https://github.com/moonshine-ai/moonshine/blob/main/LICENSE)
places streaming speech-to-text models under MIT. The applicable notice is
preserved in [LICENSE.moonshine](LICENSE.moonshine).
