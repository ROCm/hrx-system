# Stream recognition into owned transcripts and activity events

A `recognition` deployment accepts finite uploads at
`POST /v1/audio/transcriptions` and continuing audio through native stream
handles. Both callers share one prepared model, bounded admission, compatible
numerical cohorts, and the application's task executor and proactors.

A native-only `activity` deployment uses the same owner for speech activity
observations. Registration requires the source to support its declared role:
transcripts for `recognition`, activity for `activity`. A compound source can
provide both through either role's native interface. Discovery publishes actual
prepared output modalities; an activity deployment omits `loom.endpoint` and
rejects HTTP inference while retaining normal management and residency controls.

The [recognition fixture](../../fixture/recognition/model.json) can be served
directly. It integrates sample values into deterministic letter revisions,
providing an independently checkable example of the lifecycle. Learned packages
supply their own frontend, acoustic state, decoder, and transcript policy.

## Native input and output

The serving owner calls `loom_serve_registry_try_open_recognition` with the
deployment name, borrowed language/prompt options, and an explicit nonempty
selection of supported outputs. `LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT`
selects revisions; `LOOM_SERVE_RECOGNITION_OUTPUT_ACTIVITY` selects speech
probabilities. A compound source supports both in one request. A successful
admission copies those options and returns a caller-owned
[`loom_serve_recognition_stream_t`](stream.h). Success with a NULL handle means
preparation or bounded admission pressure; the caller retries when the owner
has new readiness. Existing HTTP admissions retain their queue position.

`loom_serve_recognition_stream_format` supplies the model's normalized sample
rate and channel count. Input consists of interleaved F32LE frames. Each
`stream_write` copies the accepted prefix into a fixed input ring and reports
the accepted frame count. `stream_credit` reports the available frame budget;
one input's pressure does not consume another's credit. The caller supplies
explicit EOF with `stream_finish` after its final accepted frame. Packet
boundaries never imply padding, endpoint detection, or EOF.
`stream_input_capacity` supplies the immutable source window size, including
lookahead, independently of current credit and model lifetime.

`stream_peek` borrows the next typed event from one source-ordered packet. A
completion can produce several observations, including multiple replacements of
one transcript segment. The event kind selects its transcript or activity
payload. A transcript's segment identity and increasing revision number identify
which hypothesis it replaces. A final flag commits that segment; it need not
finish the entire recognition request.
Sample intervals, optional speaker labels, and optional word alignment retain
the source's evidence. The service does not invent diarization or alignment.
An activity event carries a finite speech probability in `[0,1]` and the input
interval it covers. It does not select a turn threshold or claim exact onset.
When activity is requested, each completion's observations partition its
acknowledged input prefix without gaps or overlap. Future context remains
unconsumed in the source window until its observation is established. A caller
retaining audio for another consumer accounts for pre-roll, the accepted input
window, and at most one pending activity packet of one window's extent.
`stream_consume` advances to the next event; the request becomes eligible for
more model work after the entire packet is consumed. A stalled reader makes
only that request ineligible.

After execution ownership ends and the last event is consumed,
`stream_is_complete` becomes true. `stream_status` returns an owned terminal
status. On success with transcripts selected, `stream_document` borrows all
finalized segments without copying the transcript again. The caller closes its
handle after consuming that result, including on failure or cancellation.

## Ownership across model changes

Each stream has independent caller and execution ownership. Closing the caller
suppresses interest immediately; accepted CPU input offers, source cohorts,
feedback collection, and asynchronous backing release retain their storage
until retirement. Source row identity remains separate from packed batch
position throughout that interval.

Successful retirement returns the numerical slot and frees the input ring.
The final document, pending final event packet, input geometry, and terminal
status remain with the caller. They can outlive model removal, a new registration
using the same name, and registry destruction. Retaining a completed transcript
does not pin model weights or reserve a source row. A release failure remains
visible even if the source produced its final transcript before that failure.
On terminal readiness failure, host borrowers still join before stream storage
is detached; accepted device resources remain owned by model teardown until
actual retirement. An error notification alone does not release that storage.

The [common service](service.h) owns one bounded request ledger. Native requests
remain counted while they await input or output credit. Graceful registration
drain seals new admission and waits for these requests; abort cancels them and
joins their accepted work. Native completion retires its ledger entry through
an owner-side callback. HTTP retains its entry through finite response
publication. This keeps drain decisions and request-outcome telemetry accurate
without counting the core stream and its HTTP adapter twice.
Native request accounting begins when admission returns a handle. Immediate
validation or allocation failure and admission pressure accept no request.

## Dataflow and execution

HTTP upload decoding and PCM conversion run as bounded tasks on the shared
executor. Encoded WAV, MP3, and FLAC support follows the optional audio codec
configuration; declared S16LE/F32LE uploads and native F32 input use the core
path. Raw uploads use `audio/pcm;rate=24000;channels=1`, with optional
`format=f32le` for F32 samples (`s16le` is the default). The shared
[audio source](../../media/audio_source.h) validates geometry and decodes raw
samples without interpreting filenames. F32 input is finite and normalized to
`[-1, 1]`; invalid samples reject only their request.
Each upload retains its own decoder and resampling history through cancellation.
Raw PCM declares its input rate and matching channel count. The core
[resampler](../../media/audio_resampler.h) retains a rational sample clock and
bounded history across worker chunks, independently of source batch membership.
Input and output frames are bounded separately on each CPU turn. Immutable
coefficients can be shared by independent streams; mutable history belongs to
each input. Explicit EOF produces `ceil(N * output_rate / input_rate)` frames,
flushing lookahead without adding filter-tail duration. Matching-rate PCM uses
direct conversion. Neither raw path requires miniaudio.

The [stream service](stream.h) captures an immutable input-ring prefix for each
source CPU offer. Source preparation owns any PCM needed beyond that call.
Compatible next operations batch within source-defined work capacity; different
input positions or hypothesis lengths can remain independent. Source kernels
execute on the selected device. After both numerical and feedback frontiers
retire, shared workers collect every original member and update its bounded
document. Each request allocates one packet and decoded-event array at admission.
Collection copies the produced source extent into that storage before the source
can reuse its buffer for another ordinal. The copy preserves earlier revisions
independently of the latest document value, without allocating a new VM buffer
per completion. The serialized owner publishes results and returns input credit.
Source completion or document errors terminate only the affected request. Every
other original ordinal is still collected, including cancelled members. Failed
output is discarded, and the failed row releases at the same retired batch cut
as successful rows. The HTTP caller receives an error response while peers and
subsequent requests continue using the prepared model.

No service worker waits for a device or socket. The model package determines
its input quantum, lookahead, revision cadence, and numerical phases. Native
admission exposes those continuing-input semantics to an in-process caller;
the finite HTTP upload endpoint selects transcript output and returns a finalized
JSON or text document. Its response does not expose activity events.
