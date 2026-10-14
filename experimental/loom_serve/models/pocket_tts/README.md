# Pocket TTS numerical source

This package implements Pocket's voice encoder, FlowLM, and streaming Mimi
waveform decoder in Loom. It consumes the original BF16 Safetensors checkpoint
by tensor key. Activations, recurrent state, and waveform samples are F32.
The source modules form the numerical portion of a speech model; the caller
owns text preparation, sampling, voice selection, termination, and wire encoding.

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

State is 17,033,896 bytes, initially zero. Byte offsets 0, 4, and 8 hold the
I32 absolute FlowLM, decoder, and encoder positions. Offset 16 holds the
32-element normalized latent; offset 144 begins the six FlowLM key/value pairs.
Subsequent ranges hold the independent Mimi caches and convolution histories.
Command-local scratch is allocated through `buffer.alloca`; command planning
owns its reuse and lifetime.

This numerical configuration has space for 256 total FlowLM positions,
including BOS, voice, text, and generated frames. The caller must keep the
position below 256 before every phase; a full context is an admission or
termination boundary, never permission to discard earlier FlowLM positions.
Encoder and decoder calls require their own position at most 2,032, so the
complete 16-position block fits the rotary table. Mimi retains its actual
250-position causal window independently of those absolute positions.

The separate rotary archive contains F32 `rotary.cosine` and `rotary.sine`,
each shaped `[2048,32]`. Angles use a 64-channel head and maximum period 10,000,
with F32 rounding at the frequency and angle boundaries. Cosine and sine are
evaluated at absolute positions; cache-slot wrapping never changes the angle.

`pocket.threads` selects 1–256 workitems per workgroup. The complete numerical
path is qualified on AMDGPU with 128. FlowLM and voice projections accumulate
in F64, Mimi decoder projections and convolutions in F32, and normalization
and attention reductions in F64. These choices establish recurrent numerical
accuracy; they are not a throughput qualification.

## Source map and checks

`flow_layers.loom` shares transformer and flow-residual layer bodies.
`mimi_layers.loom` shares encoder and decoder transformer structure while
preserving their precision choices. `prefill.loom`, `voice.loom`, and
`decode.loom` compose those layers and the convolution stages.

The arithmetic modules group projection, normalization, attention, convolution,
pointwise maps, and recurrent state. Concrete model shapes wrap shared template
bodies, keeping shape-dependent indexing constant in device code. `dispatch.loom`
owns element assignment independently of any arithmetic family.

The checkpoint-free tests use closed-form projection, strided-convolution, and
normalization oracles. Streaming checks observe consecutive outputs and retained
history, including startup replication, overlap bias, latent recurrence, and
fade behavior. Both hardware and memory-access-checking profiles run with:

```sh
iree-bazel-test --config=asan \
  --//runtime/config/hal:drivers=task,amdgpu \
  --//loom/config/emit:enable=amdgpu \
  //experimental/loom_serve/models/pocket_tts/...
```
