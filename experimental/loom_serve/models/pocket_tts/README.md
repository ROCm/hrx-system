# Pocket TTS source commands

This package implements Pocket's voice encoder, FlowLM, and streaming Mimi
waveform decoder in Loom. It consumes the original BF16 Safetensors checkpoint
by tensor key. Activations, recurrent state, and waveform samples are F32.
The source modules provide numerical phases and native request commands. A
speech controller owns text preparation, voice admission, termination, and wire
encoding; request commands own immutable token selection and seeded sampling.

The architecture follows
[Pocket TTS](https://github.com/kyutai-labs/pocket-tts/tree/ec569b2dd3fec4a601b391aace185791e92b3785)
and its English checkpoint at revision
`7f364c05d1be3faf2b234cdbe4cbb68a026d4218`, path
`languages/english/model.safetensors`. This checkpoint includes the learned
voice encoder and BOS-before-voice embedding. Its original `flow_lm.*` and
`mimi.*` names and BF16 storage remain intact. The upstream permission notice
is preserved in [LICENSE.pocket-tts](LICENSE.pocket-tts).

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

## Request commands

`request.loom` composes the numerical phases with an immutable request payload.
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

The checkpoint-free tests use closed-form projection, strided-convolution,
normalization, and attention oracles. Attention checks retain positions beyond
256, mask future slots, and reach the last configured absolute rotary row.
Streaming checks observe consecutive outputs and retained history, including
startup replication, overlap bias, latent recurrence, and
fade behavior. Batched attention checks compare complete cache banks and outputs
across staggered, reordered, removed, and resumed rows. An empty cohort leaves
every cache and output unchanged. Both hardware and memory-access-checking
profiles run with:

```sh
iree-bazel-test --config=asan \
  --//runtime/config/hal:drivers=task,amdgpu \
  --//loom/config/emit:enable=amdgpu \
  //experimental/loom_serve/models/pocket_tts/...
```
