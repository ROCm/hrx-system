# Pocket TTS source package

This package implements Pocket's voice encoder, FlowLM, and streaming Mimi
waveform decoder in Loom. It consumes the original BF16 Safetensors checkpoint
by tensor key. Activations, recurrent state, and waveform samples are F32.
The source modules provide numerical phases, native request commands, and the
speech controller. Text preparation, registered-voice selection, chunking, and
termination belong to the package. The serving runtime owns HTTP streaming and
PCM encoding. Requests produce 80 ms blocks at 24 kHz, with independent history,
sampling, cancellation, and output credit.

The architecture follows
[Pocket TTS](https://github.com/kyutai-labs/pocket-tts/tree/ec569b2dd3fec4a601b391aace185791e92b3785)
and its English checkpoint at revision
`7f364c05d1be3faf2b234cdbe4cbb68a026d4218`, path
`languages/english/model.safetensors`. This checkpoint includes the learned
voice encoder and BOS-before-voice embedding. Its original `flow_lm.*` and
`mimi.*` names and BF16 storage remain intact. The upstream permission notice
is preserved in [LICENSE.pocket-tts](LICENSE.pocket-tts).

## Prepare and serve

The prepared deployment registers four reference voices: `alba`, `marius`,
`jean`, and `anna`. With [uv](https://docs.astral.sh/uv/) installed and access to
the [cloning checkpoint](https://huggingface.co/kyutai/pocket-tts), run from the
repository root:

```sh
pocket_assets="$HOME/.cache/loom/models/pocket-tts/default"
uv run --python 3.12 experimental/loom_serve/models/pocket_tts/prepare_assets.py \
  --output "$pocket_assets"

iree-bazel-run --config=opt //loom/src/loom/tools/loom-serve -- \
  --device=amdgpu --workers=8 --listen=127.0.0.1:8080 \
  --model="$pocket_assets/model.json"
```

The server build needs the AMDGPU target and HAL enabled; see
[building and loading models](../../README.md#build-and-load). An installed
`loom-serve` accepts the same arguments. This package's learned execution is
qualified on AMDGPU. Other devices need numerical qualification of these
programs in addition to the service's device-independent lifecycle.

Preparation pins the English checkpoint, tokenizer, and reference samples.
Original downloads remain in the Hugging Face cache. The new deployment
directory holds links to the checkpoint/tokenizer, two small tensor archives,
and its model document. `--offline` requires already-cached downloads. A new
directory is required for each prepared deployment, so preparing a voice set
cannot overwrite assets used by a running server.

The preparation environment is only needed to build assets. The running server
uses Loom programs for voice encoding, text prefill, sampling, and waveform
generation. It preserves the original BF16 checkpoint and performs no Python
model inference.

Request audio through the standard endpoint:

```sh
curl --fail --silent --show-error --no-buffer \
  http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"pocket-tts","input":"Good day. Each voice has its own stream.","voice":"alba","seed":42,"response_format":"pcm"}' |
  ffplay -nodisp -autoexit -f s16le -sample_rate 24000 -ch_layout mono -i pipe:0
```

The first request for a voice encodes its reference sample. Later requests reuse
the immutable voice prefix. Audio arrives in 80 ms model blocks; HTTP reads may
split or combine them. Disconnecting cancels the request, and slow readers do
not stop other voices. A fixed seed retains the request's sampling sequence
when batch membership changes.

## Register your own voices

Supply one or more named reference files instead of the default set:

```sh
uv run --python 3.12 experimental/loom_serve/models/pocket_tts/prepare_assets.py \
  --output "$HOME/.cache/loom/models/pocket-tts/my-voices" \
  --voice narrator=/path/to/reference.wav --voice alba
```

Each `--voice` adds one entry. Bare `alba`, `marius`, `jean`, and `anna` select
the pinned reference recordings; `ID=PATH` registers a local recording under
that ID. Audio is downmixed to mono, resampled to 24 kHz, and normalized to end
on a short pause using the pinned reference implementation. Samples are padded
to complete codec blocks without truncating speech. Preparation grows context
and rotary capacity to cover the complete reference and generation allowance;
an unsupported extent fails explicitly. WAV, FLAC, and other formats supported
by the preparation environment's audio reader can be imported.

Load the resulting `model.json` and use `"voice":"narrator"` in requests.
The asset archive retains the source identity and digest for every reference;
its ID/extent pairs stay in the model document. Reference files are deployment
inputs, rather than per-request HTTP uploads.

## Open WebUI

In a full Open WebUI installation with FFmpeg, select the OpenAI TTS engine:

| Setting | Value |
| --- | --- |
| API base URL | `http://HOST:8080/v1`, reachable from the Open WebUI backend |
| API key | A nonempty local value, or the key required by your reverse proxy |
| Model | `pocket-tts` |
| Voice | `alba`, or another registered ID typed into the voice field |
| Extra parameters | `{"response_format":"pcm"}` |

Keep audio preprocessing enabled. The ordinary Open WebUI TTS adapter receives
24 kHz mono PCM, converts it to MP3, and serves a cached playable file. It buffers
the complete provider response, so this integration does not provide playback
of partial responses inside Open WebUI. Its slim installation and preprocessing
bypass do not support this raw PCM path. The server's native streaming and
cancellation behavior remains available to clients that consume PCM directly.

## Speech control and admission

`prepare.loom` declares seven numerical phases and shared checkpoint domains.
`control.loom` implements the speech service's request, next-work, batch,
generation, and completion contracts. Compatible next phases batch even when
requests have different voices, prefix lengths, chunk indices, and playback
positions. A completed or cancelled request leaves the cohort independently.

`options.loom` admits the deployment geometry and its registered voices. The
`voices` array contains unique `{ "id": "alba", "frames": 126 }` records;
`frames` counts 1,920-sample waveform blocks. IDs contain 1–64 UTF-8 bytes, and
one model can register 1–256 voices. The asset directory contains the original
`model.safetensors`, `tokenizer.json`, `rotary.safetensors`, and a separate
`voices.safetensors` waveform bank. `weights.loom` preserves checkpoint bytes.

| Option | Default | Meaning |
| --- | ---: | --- |
| `context_capacity` | 512 | FlowLM positions per resident request |
| `rotary_capacity` | 4096 | Absolute positions in each rotary table |
| `chunk_tokens` | 50 | Maximum final tokens in one text chunk |
| `input_tokens` | 4096 | Maximum tokens in the complete normalized input |
| `request_capacity` | 4 | Independently resident request rows |
| `batch_capacity` | 4 | Compatible next operations in one batch |
| `threads` | 128 | Workitems per numerical workgroup |

Admission checks that a request context covers BOS, the longest voice prefix,
one maximum-sized text chunk, and its complete generation allowance. Rotary
tables cover both FlowLM and the codecs' sixteen positions per block. Batch
capacity cannot exceed request capacity. Unknown or duplicate options, invalid
voice metadata, and insufficient geometry fail during preparation.

Whole-paragraph preparation follows the pinned English model's token-boundary
sentence grouping, decimal handling, punctuation fallback, and final prompt
normalization. Every chunk is admitted before numerical work begins. Oversized
input or an indivisible sentence fails explicitly. Text packets retain token
IDs, the final post-EOS tail, and the generation bound; output is never silently
truncated to fit a request allocation.

Literal tokenizer controls such as `<s>`, `<pad>`, and `</s>` retain their
registered token IDs. Each ordinary text span around a raw control token receives
the tokenizer's declared normalization, including its prefix marker. These
markers participate in chunk sizing and final model input.

Each chunk restores its selected voice's retained prefix and starts fresh codec
history. Its chunk index participates in sampling, while the published audio
position remains continuous across the paragraph. The controller emits only
blocks before the EOS tail boundary and fails if generation exhausts its
admitted extent. Requests accept speed `1` and no instruction text.

## Phase and storage contract

The three public command roots share seven ordered buffer bindings:
phase weights, FlowLM backbone weights, rotary tables, request input,
conditioning, mutable stream state, and output. The conditioning binding is
reserved for the serving speech contract. Each phase declares only the weights
it consumes; the existing weight owner can share the backbone across all three.

| Root | Request input | Effect |
| --- | --- | --- |
| `prefill` | One I32 token, `-1` for BOS-before-voice or `0..4000` for text | Append one FlowLM position |
| `voice` | 1,920 F32 mono samples at 24 kHz | Encode an 80 ms voice block and append its speaker projection to FlowLM |
| `decode` | 32 F32 noise values, followed by F32 flow times `0` and `1` | Advance FlowLM, retain its normalized latent, and decode an 80 ms waveform block |

`decode` implements the checkpoint's one-step flow schedule and writes one F32
EOS logit followed by 1,920 F32 waveform samples.
The first waveform block includes a 120-sample linear fade. Its normalized
latent is retained before applying the checkpoint's mean and scale for Mimi.
Voice and text phases leave the waveform output untouched.

State is initially zero. Byte offsets 0, 4, and 8 hold the
I32 absolute FlowLM, decoder, and encoder positions. Offset 16 holds the
32-element normalized latent; offset 144 begins the six FlowLM key/value pairs.
Subsequent ranges hold the independent Mimi caches and convolution histories.
`layout.loom` computes the shared prefix and complete state sizes from context
capacity: `144 + 49152 * capacity` and `4450984 + 49152 * capacity` bytes. Word
counts remain logical indices; byte sizes use offsets, including above 2 GiB.
Command-local scratch is allocated through `buffer.alloca`; command planning
owns its reuse and lifetime.

`pocket.context_capacity` selects space for 1–65,536 total FlowLM positions,
including BOS, voice, text, and generated frames. The caller reserves their
combined extent and keeps the position below capacity before every phase.
A full context is an admission or termination boundary, never permission to
discard earlier FlowLM positions. `pocket.rotary_capacity` selects 16–1,048,576
absolute positions and must cover the selected FlowLM context and each codec's
16-position blocks. Mimi retains its actual 250-position causal window
independently of those absolute positions.

The separate rotary archive contains F32 `rotary.cosine` and `rotary.sine`,
each shaped `[pocket.rotary_capacity,32]`. Angles use a 64-channel head and
maximum period 10,000, with F32 rounding at the frequency and angle boundaries.
Cosine and sine are
evaluated at absolute positions; cache-slot wrapping never changes the angle.

`pocket.threads` selects 1–256 workitems per workgroup. The complete numerical
path is qualified on AMDGPU with 128. FlowLM and voice projections accumulate
in F64, Mimi decoder projections and convolutions in F32, and normalization
and attention reductions in F64. These choices establish recurrent numerical
accuracy; they are not a throughput qualification.

## Isolated request commands

`request.loom` provides isolated roots for direct numerical comparisons. The
speech controller uses the batch roots described below. The isolated roots
compose the numerical phases with an immutable request payload.
Its first 16 bytes contain a 64-bit seed, I32 voice frame count, and I32 chunk
count. Each chunk occupies `16 + 4 * pocket.chunk_tokens` bytes: four I32 header
words (token count, post-EOS tail, generation bound, reserved) followed by the
token IDs. The controller validates those extents before submission.

The private stream allocation adds two I32 cursors after numerical state:
current chunk and next token. `initialize_request` resets the entire allocation
and sets the chunk to -1. `begin_voice` appends BOS; `encode_voice` selects the
next 1,920-sample block from the independent registered-voice archive. The
`pocket.voice_index` and `pocket.voice_frames` configurations select the F32
tensor `voices.<index>.samples` with shape `[frames,1920]`. Learned voice
encoding uses the original checkpoint.

`save_conditioning` copies the FlowLM prefix into the model's conditioning
bank. Its controller invalidates the previous cache identity before accepting
that overwrite and publishes the replacement only after execution retires.
`restore_conditioning` restores that prefix, resets codec state, advances the
chunk cursor, and clears the token cursor. `prefill_text` consumes the next
token from the immutable packet. Each command uses the serving owner's request,
conditioning, stream, and output bindings; numerical commands additionally bind
their phase/backbone/rotary archives, and voice encoding binds the voice archive.

`generate_audio` derives 32 normal samples from the shared tensor random motifs,
then runs the one-step flow and waveform decoder. Philox4x32-10 uses the full
64-bit seed and counters `(normal_quadruple, generation_step, chunk, 0)`;
Box-Muller scales by `sqrt(0.3)`. Launch order cannot affect a request's random
stream. Fresh state with the same payload replays exactly on the same target.
Sampling math can round differently across targets, and autoregressive feedback
can amplify small differences; an equal seed does not promise equal waveforms
across implementations.

## Source map and checks

`text_normalization.loom` preserves the pinned English checkpoint's prompt
preparation: Unicode whitespace trimming, full first-character uppercase,
single-pass ASCII space-pair replacement, and terminal/weak punctuation with
closing quotes and brackets. It returns exact UTF-8 bytes and the post-EOS tail
of five frames for at most four words, or three frames otherwise. Empty output
is an admission failure for the controller. Its independent oracle records the
upstream `prepare_text_prompt` results with the English configuration, including
every expanding uppercase character and whitespace codepoint.

`flow_layers.loom` shares transformer and flow-residual layer bodies.
`mimi_layers.loom` shares encoder and decoder transformer structure while
preserving their precision choices. `prefill.loom`, `voice.loom`, and
`decode.loom` compose those layers and the convolution stages.
Their phase templates also serve the request commands without copying numerical
bodies or adding intermediate host uploads.

The arithmetic modules group projection, normalization, attention, convolution,
pointwise maps, and recurrent state. Concrete model shapes wrap shared template
bodies, keeping shape-dependent indexing constant in device code. `dispatch.loom`
owns element assignment independently of any arithmetic family.

`batching.loom` separates packed numerical work from resident request identity.
Its immutable membership packet contains an I32 active count followed by resident
row IDs. `pocket.batch_capacity` bounds the dispatch shape and packet;
`pocket.request_capacity` bounds persistent row storage. Inactive dispatch rows
do not read a resident ID or access request state. Changing membership changes
the packet without moving histories.

`batch_attention.loom` applies that mapping to the same arithmetic used by direct
commands. Cache and position origins follow resident rows; queries, scores,
probabilities, and outputs follow packed positions. Each argument carries its
source-owned byte offset and row stride; zero strides broadcast rotary tables.
FlowLM retains its full configured context, while Mimi wraps cache slots and
keeps absolute rotary positions. These templates provide numerical batching;
the controller owns phase selection and request eligibility.

The other `batch_*.loom` modules apply the same ownership to projections,
normalization, pointwise maps, convolution, and FlowLM state. Convolution
histories, overlap tails, previous latents, and position counters remain in
resident rows. Intermediate activations, solver inputs, and output blocks use
packed rows. Shared checkpoint tensors have zero row stride. Direct and batched
commands invoke the same arithmetic bodies, including their precision choices;
membership and byte origins determine where that arithmetic reads and writes.

`batch_prefill.loom`, `batch_voice.loom`, and `batch_decode.loom` compose complete
phases through `pocket.batch.prefill.step`, `pocket.batch.voice.step`, and
`pocket.batch.decode.step`. Their layer templates retain the original checkpoint
keys and layer indices. Each phase takes the membership packet, packed input
stride, resident state stride, and its weight/input/state bindings; decode also
takes a packed output stride and output binding. Voice and prefill have no output
binding. Scratch allocations scale with batch capacity, while cache origins come
from the same configured layout as isolated commands. Different prefix lengths
and codec positions do not require different compiled phases.

`sampling.loom` owns the shared counter-based sampling body.
`batch_sampling.loom` reads seeds from packed request rows and generation/chunk
counters from the selected resident rows. Its packed output can feed the decode
phase directly. Reordering, removing, or temporarily withholding another request
does not advance a surviving request's random stream.

`request_layout.loom` defines resident state and compact batch work geometry.
The work header retains the membership map and selected stage. Each packed
record carries phase, registered voice ordinal, phase cursor, chunk identity,
voice length, a unique-writer bit, and the selected prefill token. Progress
belongs to the record rather than the executable shape.

`batch_request.loom` connects those work records to complete numerical phases.
`batch_prefill_text` packs the selected tokens, including `-1` for BOS;
`batch_encode_voice` packs each selected 1,920-sample block from the registered
`waveforms` tensor; `batch_generate_audio` packs seeds, samples independent
noise, and decodes one block per ready request. The waveform tensor is shaped
`[pocket.voice_count,pocket.voice_frame_capacity,1920]`, with each voice's actual
length retained in its work record. Padding is never encoded.

Immutable device request rows contain only the full 64-bit seed, padded to 16
bytes. Complete text packets remain in request-owned CPU storage. Preparing a
work offer selects its next token without advancing the request; retirement
advances the cursor. Text length therefore does not enlarge the device request
bank, and a changed batch membership never changes token or seed identity.

`batch_conditioning.loom` retains a separate FlowLM prefix for each registered
voice. The bank stores twelve key/value planes with room for BOS and the longest
registered waveform, independently of the larger text-context capacity. Four
voices of at most 154 blocks occupy 30,474,240 bytes. `pocket.voice_count` and
`pocket.voice_frame_capacity` declare that geometry. Codec histories, generated
latents, and text remain private to requests.

| Command | Effect |
| --- | --- |
| `batch_initialize_request` | Clear every selected resident allocation |
| `batch_save_conditioning` | Copy each unpublished voice prefix through one selected writer |
| `batch_restore_conditioning` | Restore selected voice prefixes, reset all other request state, and install each chunk's sampling identity |

All three commands use the five speech bindings: work, immutable requests,
conditioning, resident state, and packed output. The controller selects one
writer per voice in a save cohort and publishes readiness after that command
retires. Published entries remain immutable for the model lifetime. Requests
may build a missing prefix independently, then reuse another request's completed
entry at a retired boundary. No shared pending owner can strand a voice when a
reader disconnects. Restore never reads unused cache padding, and overwrites the
entire request row before a new chunk or request uses it.

The checkpoint-free tests use closed-form projection, strided-convolution,
normalization, and attention oracles. Attention checks retain positions beyond
256, mask future slots, and reach the last configured absolute rotary row.
Streaming checks observe consecutive outputs and retained history, including
startup replication, overlap bias, latent recurrence, and
fade behavior. Batched checks compare complete banks and outputs against isolated
calls across staggered, reordered, removed, and resumed rows. Padded rows and
nonzero origins expose accidental writes outside each tensor, and different
packed and resident capacities expose identity/addressing confusion. An empty
cohort leaves every cache and output unchanged. Sampling compares reordered and
shrinking cohorts with isolated calls, preserves state and padding, and retains
the independent Philox/Box-Muller oracle. Conditioning checks compare entire
banks with an independent rectangle-copy oracle across unequal prefix lengths,
duplicate voices, reused request rows, and empty cohorts. Both hardware and
memory-access-checking profiles also verify packed tokens, full-width seeds,
and different voice blocks with inactive output guards. Run them with:

```sh
iree-bazel-test --config=asan \
  --//runtime/config/hal:drivers=task,amdgpu \
  --//loom/config/emit:enable=amdgpu \
  //experimental/loom_serve/models/pocket_tts/...
```
