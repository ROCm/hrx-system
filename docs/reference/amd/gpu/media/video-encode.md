# VCN video encoding

[Media engines](../media.md) describes the shared resource and synchronization
contracts. An encode job reads generated or processed frames and produces
compressed bytes, reconstructed pictures for later prediction, and feedback
describing the output. Those products have independent lifetimes.

## Admission and policy

Mesa exposes AVC/H.264 and HEVC/H.265 on encode-capable VCN devices, with AV1
from VCN 4. VCN 4.0.3, 5.0.1, and 5.0.2 are decode-only in this source.
Kernel video-capability queries can further change validity and dimensional
limits.
[Codec capabilities][encode-caps], [decode-only predicate][decode-only],
[kernel reconciliation][kernel-video-caps].

The initial table distinguishes these contracts:

| Property | AVC | HEVC | AV1 |
| --- | --- | --- | --- |
| Initial maximum dimensions | 4096 × 2304 | 4096 × 2304 | 8192 × 4352 |
| Reference storage | Up to 17 slots; up to 2 active references from VCN 3, otherwise 1. | Up to 17 slots; 1 active reference in this table. | Up to 9 slots; up to 2 active references from VCN 5, otherwise 1. |
| B-picture/reference-list capability | VCN 3 adds the listed L1/B-reference capability. | The listed B/L1 counts are zero. | VCN 5 adds bidirectional reference capabilities. |
| Pixel formats | NV12 in this table. | NV12, with Main10/P010 from VCN 2. | NV12/P010 in the AV1 capability table. |

Minimum dimensions, input/output alignment, temporal layers, slice/tile limits,
QP maps, and quality/rate-control modes also vary. For example, QVBR admission
checks encoder firmware minor versions 30 for VCN 3, 15 for VCN 4, and 3 for
VCN 5.
[Firmware feature predicates][video-firmware], [complete encode table][encode-caps].

The caller chooses GOP structure, picture type, references, timestamps, bitrate
or quantizer policy, and headers. Firmware consumes those choices as encode
parameters. Mesa's VA H.264 caller maintains DPB entries, picture order,
long-term-reference state, and IDR/P/B picture types.
A low-latency stream with no future-picture dependency has a different
buffering/quality tradeoff from a reordered GOP.
[Reference and picture-policy caller][encode-picture-owner].

## Firmware task representation

An encode parameter block starts with a **byte size** and command ID, followed
by command-specific DWORDs. `RADEON_ENC_END` fills the size and accumulates
the task's byte count. Address-emitting helpers register the resource and
write high then low halves of its GPU byte address. Opcode numbers change
across generations: for example, the bitstream block is `0xE` in the original
table and `0x12` from VCN 2. VCN 5 adds metadata/context-override blocks and
changes AV1 package assignments.
[Block builder][encode-blocks], [address/resource helper][encode-addresses],
[generation opcode selection][encode-opcodes].

Each task carries session info and task info. Session info supplies the
firmware-interface version and session-state address; task info supplies a
task ID, total task bytes, and permitted feedback count. VCN 4 wraps these
tasks in the [unified engine envelope](video-decode.md#command-transport) with
engine type `2` for encode. The firmware-interface version in session info
is another identity, distinct from both graphics ISA and VCN IP version.
[Session/task fields][encode-session-task], [VCN 4 wrapper][encode-unified].

Encoder tasks use these operation codes:

| Operation | Value | Role |
| --- | --- | --- |
| `RENCODE_IB_OP_INITIALIZE` | `0x01000001` | Initialize a session with the accompanying parameters. |
| `RENCODE_IB_OP_CLOSE_SESSION` | `0x01000002` | Close encoder session state. |
| `RENCODE_IB_OP_ENCODE` | `0x01000003` | Execute the described picture encode. |
| `RENCODE_IB_OP_INIT_RC` | `0x01000004` | Initialize rate control. |
| `RENCODE_IB_OP_INIT_RC_VBV_BUFFER_LEVEL` | `0x01000005` | Initialize the rate-control buffer level. |

[Operation definitions][encode-operations] and [task builders][encode-task-build].

The builders organize session and picture tasks as follows:

```text
session start:
  SESSION_INFO -> TASK_INFO -> INITIALIZE -> SESSION_INIT
  -> temporal-layer/rate-control/latency parameters -> INIT_RC -> INIT_RC_VBV

picture:
  SESSION_INFO -> TASK_INFO -> slice/tile and codec parameters
  -> header instructions -> input/reference/reconstruction context
  -> bounded bitstream output -> feedback and optional metadata/QP maps
  -> format and quality controls -> ENCODE

session close:
  SESSION_INFO -> TASK_INFO -> CLOSE_SESSION
```

The generation-selected builders determine the exact parameter ordering,
presence, opcodes, and layouts within these phases.

## Resource ownership and reconstruction

| Storage | Why it exists | Last user |
| --- | --- | --- |
| Input surface | Generated YUV, or an admitted RGB input/conversion path. | The encode/conversion reads for that picture. |
| Session state | Firmware and rate-control state. Radeonsi allocates 128 KiB. | Final ordered session operation and retirement. |
| Reconstructed pictures | Lossy reconstructions used to predict later pictures. | All later codec references to each picture. |
| Codec/pre-encode scratch | Collocated data, AV1 CDF/CDEF context, search/pre-analysis images, and generation-specific metadata. | Jobs using the selected session/quality configuration. |
| Compressed output | Bounded writable bitstream destination. | Encoder, then the downstream bitstream consumer. |
| Feedback/statistics | Produced extent, status, codec-unit locations, and requested statistics. | Encode/resolve, then the feedback consumer. |

Radeonsi derives reconstruction capacity from sequence references and the
caller's DPB. Before VCN 5 it packs reconstructed surfaces and auxiliary
state into a combined buffer. Its VCN 5 path can use separate reference
surfaces and metadata; `separate_refs` is an explicit capability. In both
layouts, each reconstruction remains live through its later reference uses.
[Resource construction][encode-resource-owner], [DPB sizing][encode-dpb-size],
[VCN 5 selection and auxiliary buffers][encode-separate-refs].

The sizing path aligns pitch/height, doubles sample storage for high bit depth,
and adds pre-encode pictures and search data when enabled. A quality-mode
comparison that holds only resolution fixed can therefore change the live
working set and traffic. [Sizing and pre-encode policy][encode-dpb-size].

For a generated-frame sequence, the caller first publishes input frame N,
then submits encode N with its reference set and output/feedback slots.
It can recycle input N after those reads complete while retaining reconstructed
N for encode N+1. Output N remains live until the network/file consumer releases
it.

## Bitstream assembly and feedback

Radeonsi can generate non-slice headers on the CPU, place them in the destination,
align the firmware slice region to 16 bytes, and record codec-unit offsets.
Slice-header or AV1 OBU instructions also form
part of the firmware task. Container muxing and transport framing are above
this elementary-bitstream contract.
[Header assembly][encode-headers], [AV1 bitstream instructions][encode-av1-headers].

Before encoding, `radeon_enc_encode_bitstream` takes the caller's destination
capacity, reserves any header prefix, sets the allowed maximum firmware
bitstream size, and allocates a 4096-byte feedback buffer. The bitstream
parameter block supplies an address, capacity, and data offset.
[Output construction][encode-output-owner], [bitstream/feedback blocks][encode-output-blocks].

After completion, the pinned feedback reader uses DWORD 1 as its result
indicator and derives the firmware extent from DWORD 6 minus DWORD 8. A zero
or oversized result becomes failed encode metadata. The reader also returns
separate header/slice locations when headers were assembled outside firmware.
The callback synchronizes and maps the feedback buffer to read these fields.
[Feedback consumption][video-encode-feedback].

For example, a 4 MiB output allocation with a small valid encoded picture is
still a 4 MiB live allocation until released; only the reported valid codec
units belong in the outgoing stream.

## Format conversion and color

The input may already be NV12/P010. When it is RGB, Mesa's encoder format
conversion can remove a separate conversion pass only for admitted cases.
Its capability predicate starts at VCN 2 excluding 2.2; the caller rejects
nondefault orientation, blending, nonzero crop origins, mismatched input/output
extents, changed primaries/transfer characteristics, unsupported matrices or
chroma siting, and DCC-compressed input. Accepted RGB formats map to supported
NV12/P010 output cases.
[Conversion capability][encode-caps], [actual conversion admission][encode-conversion].

A compute conversion can instead write the exact supported encoder surface.
Its color/range and chroma-sampling policy must agree with the stream headers.

## Ordering, changes, and teardown

Encode jobs use the native resource/dependency submission contract, including
the input producer's completion and reference readiness. Stateful tasks for
one session remain ordered; independent sessions permit independent work.
The Linux VCN 4.0 instance restriction for AV1 also applies to inspected
encode-session initialization. [Native engine routing][linux-video-routing],
[shared dependencies](../media.md#native-dependencies-and-completion).

Radeonsi detects bitrate/frame-rate and per-picture quantizer changes and
emits corresponding rate-control parameter updates. Changing resolution,
reference capacity, or bit depth has additional storage and sequence-header
consequences. Existing references and accepted jobs retain their old storage
until their final uses finish.
[Per-picture state and growth][encode-resource-owner].

The native encoder emits CLOSE_SESSION on destruction and submits that work
before releasing its session resource reference. Its winsys resource tracking
retains accepted GPU use. Higher-level drain additionally collects queued
output; AMF exposes that operation explicitly, while D3D12 requires the caller
to retain encoder/heap objects through all recorded uses.
[Native close and submission][encode-close],
[Windows lifetime and drain](windows.md#encoding).

Native acceptance, successful device completion, codec feedback, and usable
output remain separate observations. A reset or failed encode can invalidate
reference and rate-control state for later pictures. Reuse follows terminal
retirement, and stream continuation requires a valid codec restart/reference
state.

[encode-caps]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_video.c#L389-537
[decode-only]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_video.c#L18-23
[kernel-video-caps]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_video.c#L670-719
[video-firmware]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_video.c#L11-119
[encode-picture-owner]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/frontends/va/picture_h264_enc.c#L34-275
[encode-blocks]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/radeon_vcn_enc.h#L18-32
[encode-addresses]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/radeon_vcn_enc.c#L2323-2332
[encode-opcodes]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_vcn_enc.c#L14-135
[encode-session-task]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/radeon_vcn_enc_1_2.c#L17-39
[encode-unified]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/radeon_vcn_enc_4_0.c#L17-38
[encode-operations]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_vcn_enc.h#L34-42
[encode-task-build]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/radeon_vcn_enc_1_2.c#L862-1008
[encode-resource-owner]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/radeon_vcn_enc.c#L1566-1722
[encode-dpb-size]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/radeon_vcn_enc.c#L1327-1489
[encode-separate-refs]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/radeon_vcn_enc.c#L2106-2266
[encode-headers]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/radeon_vcn_enc.c#L1775-1849
[encode-av1-headers]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/radeon_vcn_enc_4_0.c#L357-398
[encode-output-owner]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/radeon_vcn_enc.c#L1851-1890
[encode-output-blocks]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/radeon_vcn_enc_1_2.c#L712-737
[video-encode-feedback]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/radeon_vcn_enc.c#L2040-2087
[encode-conversion]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/radeon_vcn_enc.c#L1892-1951
[linux-video-routing]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/vcn_v4_0.c#L1800-L1981
[encode-close]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/radeon_vcn_enc.c#L2005-2037
