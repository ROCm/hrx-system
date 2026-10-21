# VCN video decoding

[Media engines](../media.md) describes discovery, compute handoff, and memory
accounting. VCN video decoding consumes compressed picture data plus a
firmware message describing the codec state and its resources. The
[dedicated JPEG engine](jpeg.md) has its own chapter.

## Admission and codec state

The following describes Mesa's initial VCN capability table. Successful kernel
video-capability queries can further disable codecs and replace dimensional
limits. Profile, format, firmware, and frontend admission remain part of the
query.
[VCN decode capabilities][decode-caps], [kernel reconciliation][kernel-video-caps].

| Codec | Generation/format conditions in the table | Reference limits |
| --- | --- | --- |
| AVC/H.264 | NV12; initial maximum 4096 × 4096. | 17 DPB slots, 16 active references. |
| HEVC/H.265 | NV12 and P010/Main10; initial maximum 4096 × 4096 before VCN 2, 8192 × 4352 thereafter. | 17 slots, 15 active references. |
| VP9 | NV12 and P010; the same initial maximum dimensions as HEVC. Uncompressed-header-offset support has a separate firmware predicate. | 9 slots, 3 active references. |
| AV1 | VCN 3 or newer, except 3.0.33; NV12/P010. Profile 2/P012 is exposed for VCN 4.0.0 and VCN 5 or newer. Initial maximum 8192 × 4352. | 9 slots, 7 active references. |
| MPEG-2 / VC-1 | Exposed before VCN 3.0.33 in this table; NV12 and the single-buffer reference path. | 6/5 slots and 2 active references respectively. |

DPB means decoded picture buffer. Its capacity accounts for both the references
actively read by a picture and other retained pictures. The stream's reference
and output requirements determine how many of the available slots it needs.

The CPU-side caller supplies parsed sequence/picture metadata: H.264/HEVC
parameter sets and reference identifiers, VP9 segmentation/probability state,
or AV1 tile, reference, and film-grain information. The compressed picture is
still entropy-decoded and reconstructed by the media engine. RADV's
`radv_CmdDecodeVideoKHR` translates API metadata, resolves reference surfaces,
and binds the bitstream; rocDecode's parser separately determines sequence
changes, decode order, and display order.
[RADV picture submission][radv-decode], [parser callbacks][rocdecode-parser].

## Command transport

The older and unified paths bind largely the same resources through different
command representations:

| VCN path | User command representation |
| --- | --- |
| Before VCN 4, `AMD_IP_VCN_DEC` | `RDECODE_PKT0` register/value pairs write the low/high GPU address to `GPCOM_VCPU_DATA0/1`, then write `command_id << 1` to `GPCOM_VCPU_CMD`. A decode ends by writing `1` to `ENGINE_CNTL`. Register selection changes at VCN 2 and 2.5. |
| VCN 4 and newer, unified `AMD_IP_VCN_ENC` | A 16-byte `ENGINE_INFO` package selects decode, followed by a sized decode-buffer package containing a validity mask and resource addresses. Reference-list and other packages follow when required. |

`ENGINE_INFO` is `0x30000001`; engine types are common `1`, encode `2`, and
decode `3`. Its fourth DWORD is the total engine-package extent in **bytes**,
including the engine-info package. Inner packages start with their byte size
and package type. The decode-buffer address fields are explicitly **high then
low**; that order differs from the writes to `DATA0/1` on the older path.
[Outer header builder][vcn-envelope], [engine constants][vcn-envelope-types],
[resource-address builder][decode-send-command], [address structure][decode-buffer-structure].

For example, a unified creation submission binds initialized GPU-addressable
storage with this structure:

```text
ENGINE_INFO: size=16, type=0x30000001, engine=3, total_bytes
DECODE_BUFFER: package_bytes, type=1, valid_mask,
               message_address_hi/lo, ... session_address_hi/lo, ...

message -> header(CREATE, stream_handle, total_bytes, index_count=1)
           index(CREATE, byte_offset, byte_size)
           create(codec_stream_type, flags, maximum_width, maximum_height)
```

A decode submission binds the compressed stream in a separate buffer and
supplies its byte extent in the message.
Mesa advertises 256-byte stream-address alignment and 128-byte size alignment;
native command-IB alignment is a separate hardware-IP query. Radeonsi pads
and publishes compressed input before issuing decode.
[Creation builder][decode-create], [input publication][decode-input].

The cited Linux self-test additionally emits a `0x30000002` signature package:
16 bytes containing a DWORD checksum and DWORD count, with an additive
checksum over the following package data. The pinned Mesa common builder
emits engine info without that signature.
[Kernel self-test envelope][linux-video-signature],
[Mesa envelope][vcn-envelope].

## Messages, session, and per-picture storage

`rvcn_dec_message_header_t` contains header/total byte sizes, a buffer count,
message type, stream handle, feedback number, and an indexed set of payloads.
Each index describes a message ID, byte offset, byte size, and filled field.
The create payload selects the codec and maximum dimensions. The decode
message combines a generic payload with one codec payload and, as needed,
dynamic-DPB or protected-content metadata.
[Message definitions][decode-message-types], [message construction][decode-message-build].

The generic decode payload supplies dimensions, compressed byte count,
target/DPB/context extents, pitch/height information, address-library mode,
swizzle mode, and plane offsets. Its pitches come from Mesa's surface layout,
whose `surf_pitch` is in format blocks. Plane offsets include the
allocator's tile-swizzle contribution. The allocator's exact representation
is part of the firmware contract.
[Surface field units][surface-pitch], [decode layout builder][decode-layout].

The resource graph is:

| Storage | Access and lifetime |
| --- | --- |
| Session prefix and codec context | Firmware read/write state shared by ordered pictures of one session. Mesa allocates a 128 KiB prefix plus codec-specific context. |
| Per-picture embedded messages/tables | Generic and codec metadata, inverse-quantization/scaling data for AVC/HEVC, segmentation/probability/film-grain data for VP9/AV1, and feedback where used. Remain live through the submitted job. |
| Compressed bitstream | Read by decode; retained through consumption/completion. |
| Decode target | Written by decode; retained through all consumers and any reference role. |
| Reference storage | Read or updated according to the selected DPB tier; survives current-frame consumer completion when later frames reference it. |

The builder binds the session, message, optional DPB, hardware context, codec
tables, compressed data, and target. Radeonsi registers their BO identities
with synchronized read/write usage before recording addresses.
[Binding][video-decode-buffers], [native resource registration][decode-resource-owner].

Mesa reserves **64 DWORDs for creation and 512 DWORDs for a decode command**
(256 bytes and 2 KiB), excluding embedded data, session/context memory, native
padding, and kernel wrapper storage. VP9/AV1 session initialization fills
default probability state. The AV1 sizing function includes fixed maximum-size
scratch terms in addition to the 128 KiB session prefix.
[Decoder constructor][decode-constructor], [session initialization][decode-session-init],
[context sizing][decode-context-size], [AV1 context][av1-context-size].

## Reference storage and output

Mesa names four decode tiers for reference-storage layouts. Windows defines
its own decode tiers separately:

| Mesa tier | Reference storage and access |
| --- | --- |
| 0 | One opaque reference buffer plus decode output. AVC/HEVC and other codec sizing determines the backing extent. |
| 1 | One texture array with dynamic slice indices; available for the older VP9 path. Radeonsi can resize it and copy retained slices. |
| 2 | An array of independently addressed reference textures. The message supplies current/reference addresses and shared layout information. Radeonsi maintains separate DPB textures. |
| 3 | Explicit per-reference Y/UV addresses and layouts, with unified decode-target operation and no separate internal reference representation. Output and reference storage can coincide. |

The capability table adds tier 2 from VCN 3 and tier 3 from VCN 5, with an
additional AV1 firmware predicate. The constructor excludes tier 3 for 12-bit
sessions. Actual callers impose another condition: **tier 3 requires tiled
output**. RADV additionally requires output usage compatible with both decode
destination and DPB. Selecting a linear output for simple buffer loads can
therefore lose the shared-reference path.
[Tier definitions][decode-tier-types], [capability predicates][decode-caps],
[constructor limits][decode-constructor], [radeonsi selection][decode-tier-select],
[RADV selection][radv-tier-select].

For tier 2, the builder maps codec reference IDs to GPU addresses and names a
current reference surface that may differ from the displayed target. Tier 3
instead emits `RDECODE_IB_PARAM_DYNAMIC_REFLIST_BUFFER` (`0xC`), with a
header and per-reference index, plane addresses, pitches, aligned heights,
and sizes; it sets the unified-target flag.
[Dynamic reference construction][decode-reference-build],
[radeonsi DPB allocation][decode-dpb-owner].

For AV1 film grain, radeonsi applies grain to the decode target while retaining
an ungrained reference; its tier-3 caller supplies a separate surface when a
film-grain target is present. A model consuming the ungrained reference receives
different pixels from one consuming the displayed target.
[Film-grain caller][decode-film-grain].

## Submission and retirement

A native caller's lifecycle is:

1. Discover the codec/format/firmware contract, parse sequence requirements,
   select the DPB representation, and allocate bounded session and picture
   pools in the submitting VM.
2. Initialize session tables and submit creation. Serialize stateful session
   use; a second independent stream owns independent session state.
3. For each picture, retain required reference identities, publish compressed
   data and metadata, register resources, and submit after producer/reference
   dependencies. Decode order follows the codec's graph.
4. Publish completion to preprocessing/display consumers through native
   dependencies and establish visibility. Check codec output status separately
   from native job status.
5. Release per-picture input/metadata after their final job, retain references
   until the stream releases them, and retain consumer-visible output until its
   final consumer. Drain or replace the session without recycling accepted-work
   storage early.

[Radeonsi initialization and creation][decode-init-owner],
[picture submission][decode-submit-owner], and
[shared completion contracts](../media.md#native-dependencies-and-completion)
show the implementation boundaries.

The protocol defines a DESTROY message, but the pinned common VCN constructor
installs no destroy command and has a zero destroy-command bound; radeonsi
skips command emission in that case. The host decoder object and its resources
still need retirement.
[VCN constructor][decode-constructor], [destroy dispatch][decode-destroy-owner].

For protected decode, the builder carries encryption metadata and can use a
separate protected context allocation. Submission and consumer access require
the platform's admitted protected-resource path.
[Protected metadata construction][decode-protected],
[protected session allocation][decode-init-owner].

[decode-caps]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_video.c#L269-387
[kernel-video-caps]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_video.c#L670-719
[radv-decode]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/vulkan/radv_video.c#L1970-2112
[rocdecode-parser]: https://github.com/ROCm/rocm-systems/blob/b679bae80581b21a864a858fb0c85c3fb991991a/projects/rocdecode/api/rocdecode/rocparser.h#L243-L285
[vcn-envelope]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_vcn.c#L10-31
[vcn-envelope-types]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_vcn.h#L29-40
[decode-send-command]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_vcn_dec.c#L960-1029
[decode-buffer-structure]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_vcn_dec.h#L453-487
[decode-create]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_vcn_dec.c#L1060-1107
[decode-input]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/si_video_dec.c#L345-376
[linux-video-signature]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_vcn.c#L763-L791
[decode-message-types]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_vcn_dec.h#L493-565
[decode-message-build]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_vcn_dec.c#L1808-1921
[surface-pitch]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_surface.h#L236-251
[decode-layout]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_vcn_dec.c#L1923-1962
[video-decode-buffers]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_vcn_dec.c#L2182-2214
[decode-resource-owner]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/si_video_dec.c#L106-119
[decode-constructor]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_vcn_dec.c#L2330-2418
[decode-session-init]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_vcn_dec.c#L1045-1057
[decode-context-size]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_vcn_dec.c#L781-867
[av1-context-size]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_vcn_dec.c#L191-235
[decode-tier-types]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_video.h#L32-36
[decode-tier-select]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/si_video_dec.c#L309-344
[radv-tier-select]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/vulkan/radv_video.c#L1947-1967
[decode-reference-build]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_vcn_dec.c#L2027-2110
[decode-dpb-owner]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/si_video_dec.c#L417-495
[decode-film-grain]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/si_video_dec.c#L1190-1216
[decode-init-owner]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/si_video_dec.c#L163-259
[decode-submit-owner]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/si_video_dec.c#L1341-1391
[decode-destroy-owner]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/si_video_dec.c#L272-306
[decode-protected]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_vcn_dec.c#L2119-2180
