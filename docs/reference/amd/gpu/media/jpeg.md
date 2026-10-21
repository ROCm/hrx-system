# JPEG decoding

[Media engines](../media.md) describes the shared discovery, ownership, and
compute handoff contracts. This chapter describes the dedicated JPEG engine
and its Linux user-command protocol. Windows JPEG/MJPEG uses the
[video API contracts](windows.md).

AMD's fixed-function JPEG engine reads a compressed image and writes decoded
planes into GPU-addressable memory. Its command processor consumes `PACKETJ`
register operations on a dedicated JPEG queue. A compute consumer can use the
decoded allocation directly when it has the surface layout, a native execution
dependency, and the required memory visibility. The JPEG indirect buffer, the
kernel's ring wrapper, and the compressed image are three different inputs with
different owners. [Mesa's decoder][jpeg-builder] and the [Linux ring implementation][linux-ring]
show those boundaries.

## Discovery and applicability

On Linux, `AMDGPU_HW_IP_VCN_JPEG` is hardware-IP selector **8**.
`AMDGPU_INFO_HW_IP_INFO` reports the available logical ring mask, engine
version, discovery version, and indirect-buffer alignment.
`AMDGPU_INFO_HW_IP_COUNT` is a separate count. In the cited driver the JPEG
count includes instances multiplied by rings, while the available-ring mask is
capped by the number of context entities. Submission uses the legal ring
indices from that mask. The JPEG discovery version comes from `UVD_HWIP`.
[Engine selector][linux-uapi], [query result][linux-info-uapi], [IP information][linux-discovery],
[IP counts][linux-count].

Mesa selects three JPEG register layouts. These are its software labels,
`RDECODE_JPEG_VER_1`, `_2`, and `_3`, not physical IP revision numbers:

| Mesa predicate, evaluated in this order | Register layout |
| --- | --- |
| `vcn_ip_version >= VCN_5_0_0` or `== VCN_4_0_3` | Layout 3: relocated pitch/address registers and explicit plane-offset registers. |
| Otherwise `vcn_ip_version >= VCN_2_0_0` | Layout 2: indirect plane offsets through `UVD_JPEG_INDEX` / `UVD_JPEG_DATA`. |
| Otherwise, on the VCN JPEG path | Layout 1: older SOC15 register addresses and context-index reset/wait sequences. |

The [constructor][jpeg-constructor] selects the layout; the separate
[capability function][jpeg-caps] determines which features Mesa exposes:

| Property | Mesa's initial JPEG capability table at the cited revision |
| --- | --- |
| Dimensions | Minimum 64 × 64; maximum 8192 × 8192 before VCN 2, 16384 × 16384 from VCN 2. |
| Compressed stream | Address aligned to 256 bytes; byte extent rounded to 128 bytes. |
| Output | NV12, packed Y8/U8/Y8/V8 4:2:2, and Y8 monochrome; separate Y/U/V 4:4:4 and 4:4:0 planes from VCN 2. |
| Crop and RGB/RGBA output | Exactly VCN 4.0.3, 5.0.1, or 5.0.2 in this capability function. |
| Reference storage | Zero decoded-picture-buffer slots and zero active references. |

Successful kernel video-capability queries further intersect codec validity
and replace the maximum width/height, including for JPEG.
[Kernel capability reconciliation][kernel-video-caps].

The RGB/crop capability predicate is narrower than the layout-3 predicate.
Mesa installs these capabilities when a JPEG queue is available.
Its JPEG session parameters select 8-bit samples, and rocJPEG's
parser admits baseline SOF0 while rejecting other frame coding modes, including
progressive JPEG. [Mesa picture parameters][jpeg-picture],
[rocJPEG marker admission][rocjpeg-parser].

## Input and output storage

The compressed input contains JPEG markers and tables as well as entropy-coded
data. Mesa's VA frontend can construct SOI, DQT, DHT, DRI, SOF0, and SOS from
picture parameters when its input lacks the header, and appends EOI. CPU header
parsing supplies dimensions, sampling, and table metadata; the engine still
performs entropy decoding and image reconstruction. [Header construction][jpeg-headers]
and [bitstream assembly][jpeg-bitstream] show what reaches the decoder.

Before submission, radeonsi rounds the compressed extent to 128 bytes and
zero-fills the padding. Its command builder adds the compressed BO for reads
and the output BO for read/write use. The entire padded input must be backed
and visible to the engine. The command's stream pointer uses the padded
extent in DWORDs, not the original file length. [Input publication and resource binding][jpeg-resources].

Output storage has a format, pitches, plane offsets, and a tiling mode. A
resident NV12 image has a Y plane and an interleaved UV plane; it need not pass
through an RGB intermediate before a compute kernel samples or resizes it.
The pitch and allocated height determine storage, including padding. A
consumer receives the layout alongside the DMA-BUF handle or GPU address.

Mesa's JPEG builder takes one output base address and per-plane surface
offsets. It combines each offset with the surface's tile-swizzle bits and
programs the actual address/tiling mode. A linear consumer therefore requires
a linear surface or an explicit interpretation/conversion of the native
layout. [Surface interpretation and output registers][jpeg-builder].

For the crop-capable path, the builder rounds X and Y **down** to multiples of
16 and rounds width and height **up** to multiples of 16. Crop positions and
extents occupy low/high 16-bit fields in `UVD_JPEG_ROI_CROP_POS_START` and
`UVD_JPEG_ROI_CROP_POS_STRIDE`. These registers select an aligned output region.
RGB conversion uses the builder's matrix and upsampling coefficients.
[Crop and format conversion][jpeg-builder].

## PACKETJ representation

A register operation consists of two 32-bit DWORDs. Mesa's `RDECODE_PKTJ` and
Linux's `PACKETJ` agree on this header layout:

| Word | Bits | Field |
| --- | --- | --- |
| 0 | 17:0 | Register index, encoded directly rather than as a byte address. |
| 0 | 23:18 | Reserved; zero in the cited user command protocol. |
| 0 | 27:24 | Condition. |
| 0 | 31:28 | Packet type. |
| 1 | 31:0 | Register value or wait mask, according to the operation. |

[Mesa encoding][packet-layout] and [Linux encoding][linux-packet-layout].

For the Linux JPEG 4.0 and 4.0.5 ring implementations, the kernel copies and
parses the user indirect buffer. Its parser admits:

| Header | Operation admitted by the parser |
| --- | --- |
| Type 0, condition 0 | Register write. |
| Type 3, condition 3 | Register wait using the programmed reference and mask. |
| Type 6, header exactly `0x60000000` | NOP; Mesa supplies a zero data DWORD. |

Writes and waits are restricted to register indices `0x4000` through `0x41c2`,
excluding the atomic-register interval `0x4120` through `0x412a`. Other packet
types, nonzero reserved bits, or other condition combinations are rejected.
[Parser and allowed ranges][linux-parser], [range constants][linux-ranges],
[4.0 ring][linux-ring], [4.0.5 ring][linux-ring-405], [command copying][linux-command-copy].

The kernel's own ring wrapper uses additional types for indirect-buffer
execution, memory completion writes, and traps. Those packets are outside this
user-IB contract. [Kernel IB invocation and fences][linux-wrapper].

## Modern register sequence

The following registers illustrate the layout-2/layout-3 differences. The
indices are packet register indices; paired address registers carry the low
and high halves of a 64-bit GPU byte address.

| Native register, Mesa `vcnip` spelling | Layout 2 index | Layout 3 index | Value used by the builder |
| --- | --- | --- | --- |
| `UVD_JPEG_CNTL` | `0x4000` | `0x4000` | `0x6` starts decode, with optional crop/conversion bits; `0x4` stops it. |
| `UVD_JPEG_DEC_SOFT_RST` / `_1` | `0x402f` | `0x4051` | Reset request; bit 16 is the observed reset acknowledgement. |
| `UVD_LMI_JPEG_READ_64BIT_BAR_LOW/HIGH` / `_1` | `0x40e0` / `0x40e1` | `0x40b2` / `0x40b3` | Compressed-input base. |
| `UVD_JPEG_RB_WPTR` | `0x4002` | `0x4002` | Padded compressed byte extent divided by 4. |
| `UVD_JPEG_PITCH`, `UVD_JPEG_UV_PITCH` / `_1` | `0x401f` / `0x4020` | `0x4043` / `0x4044` | YUV byte pitch divided by 16; layout-3 RGB conversion uses unshifted byte pitch. |
| `UVD_LMI_JPEG_WRITE_64BIT_BAR_LOW/HIGH` / `_1` | `0x40e2` / `0x40e3` | `0x40b4` / `0x40b5` | Output-surface base. |
| `UVD_JPEG_OUTBUF_WPTR` / `_1` | `0x401d` | `0x4041` | Bit 0 participates in the output-idle wait. |
| `UVD_JRBC_IB_COND_RD_TIMER`, `UVD_JRBC_IB_REF_DATA` | `0x408e` / `0x408f` | `0x408e` / `0x408f` | Wait control `0x01400200` and the comparison reference. |

[Register definitions][jpeg-registers] supply the indices; the
[decode builder][jpeg-builder] supplies the values and units. The timer literal
is the emitted control word, not a documented duration in microseconds.

After the caller has validated the image, selected a supported output layout,
and made its mapped resources available, Mesa emits this sequence for layouts
2 and 3:

1. Assert soft reset. Program the wait timer and reference `1 << 16`, then
   wait on the reset register with mask `1 << 16`. Deassert reset and wait
   again with reference zero.
2. Set the input base, `UVD_JPEG_RB_BASE = 0`, and
   `UVD_JPEG_RB_SIZE = 0xfffffff0`. Set `UVD_JPEG_RB_WPTR` to the padded
   DWORD count. The fixed RB size literal is not the input allocation size
   and supplies no bounds validation for that allocation.
3. Set pitches, address mode, tiling, output base, and plane offsets. Layout 2
   uses `UVD_JPEG_INDEX` / `UVD_JPEG_DATA`; layout 3 uses
   `UVD_JPEG_LUMA_BASE0_0`, `UVD_JPEG_CHROMA_BASE0_0`, and
   `UVD_JPEG_CHROMAV_BASE0_0`. Program admitted crop/conversion controls.
4. Initialize output control and read pointer, enable error interrupts with
   `UVD_JPEG_INT_EN = 0xfffffffe`, and write the start value to `UVD_JPEG_CNTL`.
5. Set the wait reference to the input DWORD count and wait on
   `UVD_JPEG_RB_RPTR` with mask `0xffffffff`. This observes input consumption.
6. Set the reference to `0xffffffff` and wait on `UVD_JPEG_OUTBUF_WPTR` with
   mask `1`. The RGB path additionally waits on `UVD_JPEG_INT_STAT` with its
   conversion completion mask. Write `UVD_JPEG_CNTL = 0x4` to stop.

These waits execute on the JPEG engine. Scheduling, VM setup, and the externally
observable fence belong to the surrounding kernel job. Layout 1 has different
reset/context-index and LMI-drop sequences.
[Complete JPEG builder][jpeg-builder], [kernel wrapper][linux-wrapper].

## Linux submission, completion, and reuse

The native flow is a DRM command submission on the render node, using a
context, mapped BOs, and a `drm_amdgpu_cs_chunk_ib` whose IP selector is
`AMDGPU_HW_IP_VCN_JPEG`. The chunk carries the GPU address, **byte** length,
IP instance, and logical ring. The cited driver reports a 256-byte start
alignment and a 64-byte length alignment for JPEG IBs. These are distinct
from the compressed stream's 256/128-byte requirements. Mesa pads JPEG commands
with NOP pairs to the native IP alignment. [UAPI][linux-uapi],
[native discovery][linux-discovery], [Mesa padding][jpeg-padding].

The complete ownership sequence is:

1. Allocate/map compressed input, output planes, and command storage in the
   submitting DRM VM. Prepare the BO residency list, or satisfy the native
   always-valid-VM contract used in place of individual BO-list entries.
   Publish input and commands before submission.
2. Submit the JPEG IB with producer dependencies and the intended completion
   signal. DRM CS supports syncobj input/output and timeline wait/signal
   chunks. Mesa's winsys composes these with its resource tracking.
3. The kernel copies and validates the IB on the parsed path, schedules the
   job, and emits the engine wrapper. The wrapper establishes VMIDs, flushes
   the media VM translation path, invokes the IB, and emits the native fence
   and interrupt.
4. A consumer waits through an available native dependency mechanism and
   acquires the producer's writes for its own memory path.
5. Keep accepted-work resources alive through their native completion, and
   keep the decoded surface and mappings alive through the consumer's final
   use. Reusing the image after JPEG completion alone races a still-running
   compute consumer. Submission failure, asynchronous device failure, and
   successful completion remain separate outcomes.

[Mesa native submission][mesa-submission], [BO ownership][mesa-bo-list],
[DRM dependencies][linux-dependencies],
[command copying][linux-command-copy], [VM flush][linux-vm-flush], and
[cross-engine visibility](../../interop/external-memory.md) supply the respective
parts of this sequence.

The cited JPEG rings set `no_user_fence`; DRM rejects a user-fence chunk on
such a ring. Completion uses DRM job fences and syncobj dependencies.
Mesa also advertises no arbitrary memory-write operation for its JPEG queue.
KFD's queue-type UAPI has no JPEG user queue.
[Ring capabilities][linux-ring], [user-fence rejection][linux-user-fence],
[Mesa queue capabilities][queue-caps], [KFD queue types][kfd-queues].

## Working-set shape

Mesa's JPEG constructor budgets at most **128 DWORDs, or 512 bytes**, for its
decode command and requires no GPU session/embedded buffer. This excludes
native alignment, kernel wrapper storage, context state, and pooled input and
output allocations. It also does not allocate an image-sized coefficient
array. JPEG's zero-reference contract allows Mesa to use separate decoder
contexts for parallel independent images. [Constructor][jpeg-constructor],
[capabilities][jpeg-caps], [multi-instance caller][jpeg-instances].

[Pipeline memory accounting](../media.md#memory-and-traffic) includes padded
images, native queue storage, and surfaces retained by in-flight consumers.

[jpeg-builder]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_vcn_dec.c#L2470-2751
[linux-ring]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/jpeg_v4_0.c#L759-L789
[linux-uapi]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/include/uapi/drm/amdgpu_drm.h#L934-L1068
[linux-info-uapi]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/include/uapi/drm/amdgpu_drm.h#L1542-L1561
[linux-discovery]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_kms.c#L557-L634
[linux-count]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_kms.c#L712-L759
[jpeg-constructor]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_vcn_dec.c#L2761-2852
[jpeg-caps]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_video.c#L539-568
[kernel-video-caps]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_video.c#L670-719
[jpeg-picture]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/si_video_dec.c#L811-860
[rocjpeg-parser]: https://github.com/ROCm/rocm-systems/blob/b679bae80581b21a864a858fb0c85c3fb991991a/projects/rocjpeg/src/rocjpeg_parser.cpp#L151-L201
[jpeg-headers]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/frontends/va/picture_mjpeg.c#L140-279
[jpeg-bitstream]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/frontends/va/decode.c#L199-264
[jpeg-resources]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/si_video_dec.c#L345-401
[packet-layout]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_vcn_dec.h#L29-35
[linux-packet-layout]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/soc15d.h#L74-L95
[linux-parser]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_jpeg.c#L546-L609
[linux-ranges]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_jpeg.h#L28-L39
[linux-ring-405]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/jpeg_v4_0_5.c#L817-L849
[linux-command-copy]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_cs.c#L1031-L1089
[linux-wrapper]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/jpeg_v2_0.c#L503-L610
[jpeg-registers]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_vcn_dec.h#L217-274
[jpeg-padding]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/winsys/amdgpu/drm/amdgpu_cs.cpp#L2255-2277
[mesa-submission]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/winsys/amdgpu/drm/amdgpu_cs.cpp#L1298-1470
[mesa-bo-list]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/winsys/amdgpu/drm/amdgpu_cs.cpp#L1930-1974
[linux-dependencies]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_cs.c#L645-L664
[linux-vm-flush]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/jpeg_v2_0.c#L640-L653
[linux-user-fence]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_cs.c#L353-L374
[queue-caps]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_video.c#L613-660
[kfd-queues]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/include/uapi/linux/kfd_ioctl.h#L61-L66
[jpeg-instances]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/mm/si_video_dec.c#L1559-1573
