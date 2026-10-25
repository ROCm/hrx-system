# Native media queues

The optional [`amdf_media_api_t`](../include/amdf/media/api.h) extension exposes
native codecs through core devices, memory, and kernel queues. Enabling
`AMDF_MEDIA` in CMake or `//libamdf/config:media` in Bazel selects the platform
provider independently of GPU and NPU compute families. Table negotiation
performs no native initialization. Activating media opens its execution domain;
it neither activates compute nor creates a queue or pixel allocation.

The implemented codec set is queried per live queue family. Exact
profile/format/dimension/reference admission is a separate query:

| Provider | Decode schemas | Output storage |
| --- | --- | --- |
| Windows D3D12 | H264; HEVC Main/Main10; VP9 profiles 0/2; AV1 profile 0 | NV12 or P010 according to actual sample depth; application-placed native images |
| Linux DRM VCN 4 | H264; HEVC Main/Main10; VP9 profiles 0/2; AV1 profile 0 | Linear NV12/P010 output and separate application-owned linear references |
| Linux DRM VCN 5 | H264; HEVC Main/Main10; VP9 profiles 0/2; AV1 profile 0 | Linear NV12/P010 output and separate application-owned native references |
| Linux DRM JPEG | Eight-bit baseline sequential 4:2:0 JPEG on the layout-2 JPEG engine | Linear NV12 planes, directly or through a placed image identity |

A native driver's broader codec list does not advertise schemas implemented by
this extension. Configuration queries preserve requested dimensions and return
an explicit unsupported predicate. Image placement queries separately report
alignment and storage padding; allocation dimensions never silently replace
coded dimensions.

HEVC Main10 admits compatible eight-bit syntax. VP9 admits eight-bit profile 0
and ten-bit profile 2, both 4:2:0. VP9 reference maps may alias the caller's
unique physical bindings; active LAST/GOLDEN/ALTREF entries select that map.
The caller resolves loop-filter and segmentation state, while the session
retains native probability contexts. A show-existing-frame header reuses the
application's existing image without submitting a decode command.

AV1 profile 0 admits eight-bit and ten-bit 4:2:0 pictures. The session extent
is the upscaled extent; super-resolution changes the entropy-coded sample count
without changing the output allocation. The caller supplies resolved tile
boundaries, quantization, segmentation, loop filters, restoration, global motion,
and film grain. Seven active references select an eight-entry logical map over
stable physical reference slots. Grain-free reconstructed references remain
separate from the displayed output, including when film grain is enabled.

## Ownership through one picture

The application activates media with `device_create`, queries its configuration,
and creates a `kernel_queue` with explicit pending-submission and dependency
operand bounds. Configuration results identify core memory scopes/profiles for
images, compressed input, and any codec feedback. The access set includes every
required live consumer at core memory construction. Imports and consumer layout
establishment happen before use.

An `amdf_media_image_t` is a native identity borrowing a range of
`amdf_memory_t`. It allocates no pixel backing. The application chooses heap
placement, array layers, aliasing transitions, and final use. An
`amdf_media_frame_t` carries either that identity and its exact placement or
explicit planes with direct memory owners, access ordinals, offsets, extents,
and row pitches. A linear image query returns the same plane representation,
including the selected layer's offsets. The full returned extent belongs to
native execution; row and codec-block padding has unspecified contents.

The Windows shared-heap profile provisions backing in 2 MiB units so its
pre-allocation native charge remains exact across the driver's page-size
transition. Logical length and queried image placement alignment remain
independent. Applications can suballocate several images in that explicit heap;
the library neither pools heaps nor hides the padding charge.

A session owns native codec state and borrows one ordered queue. The application
supplies the reference images and stable reference-slot indices. A reconstructed
picture retains its slot until its last reference use; later pictures can reuse
the retired slot. The session contains no image pool or reference scheduler.
The exact configuration reports `requires_begin_end`. When set, the application
submits an explicit BEGIN before its first picture and END after its last one.
These operations share ordinary recording/retirement rules. Rejected publication
changes no session state; accepted END prevents later submissions to that
session. Construction and destruction publish no hidden firmware work.

Each command is one reusable native recording, provisioned for an explicit
codec and maximum slice/tile, reference, and activation counts. Matching sessions
can reuse that storage. `command_prepare` consumes the
caller's temporary parsed metadata and records into that owner. Compressed
payload and pixel backing remain borrowed through accepted execution. JPEG
consumes a complete marker/header stream; the caller's admitted dimensions,
sampling, and precision must describe that stream. Video schemas carry explicit
parsed syntax. Preparation performs no compressed-payload readback or pixel copy.

`kernel_queue_submit` publishes the prepared recording and already-created
native dependency bindings. It performs no libamdf allocation, parsing, import,
completion polling, or host completion wait. A nonzero accepted point owns the
recording and operands even when a later native operation returns failure.
The application refreshes checked queue status or waits at its chosen drain
boundary, reads the retired command result, and reuses its own slots. Preparing
a new operation clears the previous result association.

Shutdown follows those borrows: checked retirement, dependency bindings and
commands, sessions/queues, image identities, core memory attachments, then media
devices. A failed native detach preserves its unrelinquished backing; a failing
status is never permission to recycle storage that native execution may retain.

## Linux native ownership and retirement

One independently activated media device owns a DRM render file and private VM.
Allocation uses shareable GTT backing. The system-memory import profile checks
native placement before admitting a DMA-BUF; it does not label local VRAM as
system memory. Cold import preserves physical identity and logical subrange
placement, including through re-export.

Repeated PRIME imports in one render file return the same GEM handle without
another native handle reference. Resource attachments share a canonical native
buffer owner through a cold, per-device weak index. Its final attachment releases
the VM mapping and native handle. The index owns no references and is not an
allocation pool. Each hot operand resolves its already-established access
ordinal directly. A failed native release quarantines that render file's memory
ownership so another alias cannot accidentally close or reuse unresolved state.

A JPEG or VCN publication is one DRM CS ioctl carrying the IB, residency, dependency
waits, and signals. Cold queue construction provisions one original-fence
sync object per pending slot and one exportable timeline. Both signals attach
the same job fence; no extra hardware signal job is introduced.

Timeline signaling establishes execution order but cannot certify original job
success. Checked retirement first samples completed timeline progress, then
exports and inspects each newly completed original fence before advancing its
frontier. Sync-file export/query/close is an explicit retirement cost. Temporary
observation failure leaves the point unretired; an original execution error is
sticky and cannot be hidden by a later signaled point. Eventfd notification is a
wake hint prompting that same status check.

VCN sessions own firmware context, with charges reported by `session_query_info`;
all output and reconstructed-reference images remain application-owned. One
recording provisions the IB, parsed firmware message, and codec auxiliary data.
JPEG, H264, HEVC, and VP9 use one native page; AV1 uses 28 KiB with 4 KiB
pages, including its 20 KiB grain tables. Preparation writes only that retired
recording, and submission selects its provisioned residency list. Codec
feedback is unavailable on these Linux decode paths; successful queue
retirement is not reported as a codec success result.

VCN 5 H264, HEVC, VP9, and AV1 keep consumer output linear while storing
decoder-private references in the firmware's 256B-D layout with 64-row luma
alignment. The reference image still exposes its caller-owned memory identity,
placement, and complete allocation requirements. It does not expose a linear
plane layout or accept direct-plane reference bindings. The application binds
it as a native image and preserves each slot through its last dependent
picture; libamdf adds no reference pool, copy, or detile operation.

VP9's firmware context reserves about 10.4 MiB even for small images. This
generation-specific native cost is included in `session_query_info`; it is
separate from application-owned decoded and reference pixels. Reference storage
queries also expose reconstruction padding: sixteen rows on D3D12 and sixty-four
on VCN for VP9 and AV1, with exact coded dimensions retained in the session.

VCN 4 AV1 reserves 36,389,888 bytes of context and VCN 5 reserves 36,688,896,
plus 128 KiB of session storage in either case. The complete charge is reported
by `session_query_info`. Four standard CDF contexts expand from 26,546 immutable
library bytes during session construction. Grain-enabled preparation computes
the native Gaussian/autoregression tables into the recording's provisioned
auxiliary storage, using the padded VCN 4 or dense VCN 5 block arrangement. No
codec library, pixel readback, or per-picture allocation participates.

## Compute integration

Native queue bindings connect compatible KMQ transports. A user queue's memory
signal protocol remains an application integration boundary. A media API alone
does not turn such a protocol into a kernel-owned timeline or supply an implicit
host relay.
