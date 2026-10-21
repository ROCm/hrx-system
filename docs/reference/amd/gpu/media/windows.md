# Windows media and compute

Windows exposes video decoding, processing, motion estimation, and encoding
through D3D video interfaces. The driver translates those operations into its
native engine protocol. A resident pipeline shares D3D resources and queue
dependencies with compute. [Media engines](../media.md) describes the common
ownership and memory accounting.

## Interfaces and admission

| Interface | What the public contract supplies |
| --- | --- |
| D3D11/D3D12 video decode | Codec profiles, picture parameters, compressed input, references, output resources, and status queries. |
| D3D12 video encode | Encoder/session objects, input and reconstructed resources, reference control, bounded bitstream storage, and resolved feedback. |
| D3D12 video process / motion estimation | Separately queried operations and their input/output resources. |
| AMD Advanced Media Framework (AMF) | Asynchronous media components, codec properties/capabilities, native resource wrapping, and output ownership. |
| AMD Windows kernel-mode interop (WKMI) | The cited public compute/transfer interfaces; they do not define a raw VCN or JPEG submission ABI. |

Microsoft introduces D3D12 H.264/HEVC encoding with Windows 11/WDDM 3.0 and
AV1 encoding with Windows 11 24H2/WDDM 3.2. The JPEG/MJPEG DXVA picture
contract is documented for Windows 11 24H2 and D3D11/D3D12. Driver capability
queries determine profile and format support on each adapter.
[Encode API availability][d3d12-encode-availability],
[JPEG picture contract][dxva-mjpeg].

Decode admission queries the node, profile/configuration, dimensions, output
format, frame rate, and bitrate. Returned support flags, configuration flags,
and tier affect resource construction. Encode admission includes codec, input
format, codec configuration, GOP, rate control, resolutions, and maximum DPB
references; it returns validation flags and per-resolution limits.
[Decode support query][d3d12-support],
[encode support query][d3d12-encode-support].

## Decoding

The pinned headers define H.264, HEVC, VP9, AV1, and JPEG/MJPEG profile GUIDs,
including JPEG 4:2:0, 4:2:2, and 4:4:4 variants. A submitted decode carries
picture arguments, a compressed-bitstream resource with byte offset and size,
reference resources/subresources and associated decoder heaps, and an explicit
output resource. For JPEG, `DXVA_PicParams_MJPEG` includes sampling/table data,
restart interval, and scan locations/sizes.
[Profile identities][d3d12-profiles], [decode arguments][d3d12-decode],
[JPEG parameters][dxva-mjpeg].

When a decoder reports `REFERENCE_ONLY_ALLOCATIONS_REQUIRED`, its reference
resources use `D3D12_RESOURCE_FLAG_VIDEO_DECODE_REFERENCE_ONLY` and cannot
also serve as shader-readable model input. D3D12 decode tiers 1 and 2 may require
this; tier 3 must not. These D3D12 tier numbers are independent of
[Mesa's VCN reference tiers](video-decode.md#reference-storage-and-output).
An encode-reference-only resource similarly belongs exclusively to encode.
[Reference-only resource contract][d3d12-resource-flags].

The caller consequently distinguishes displayed/consumed output from retained
decode references, even when a particular configuration permits the same
resource to serve both. Decoder-heap size queries expose driver memory in
addition to visible frame allocations. Decode statistics report codec status
and affected macroblocks separately from queue completion.
[Heap size and decode statistics][d3d12-decode-storage].

## Encoding

An encoder consumes an input resource and picture/sequence controls, reads
the declared references, and writes three distinct products: compressed bytes,
a reconstructed picture, and hardware-layout metadata. The caller owns GOP
and reference management, stream headers, and muxing. The reconstructed
picture predicts later frames. [Encode resources and controls][d3d12-encode],
[API responsibility boundary][d3d12-encode-availability].

`ResolveEncoderOutputMetadata` converts the opaque metadata into the public
layout. That result includes encode error flags, written byte count, subregion
count, and statistics. Resource-requirement queries supply alignment and
maximum metadata-size constraints. Allocated output capacity, a successful
fence, and a nonempty valid bitstream are separate observations.
[Metadata representation and resolve][d3d12-encode-metadata],
[resource requirements][d3d12-encode-requirements].

Mesa's D3D12 encoder demonstrates the real flow: its video queue waits for
input/header work, records encode, transitions metadata from encode-write to
encode-read, resolves it, and returns tracked references to their subsequent
state. The barriers describe resource access; queue dependencies order work
between engines.
[Input queue dependency][mesa-d3d12-encode-wait],
[encode, barriers, and metadata resolve][mesa-d3d12-encode-resolve].

The documented `EncodeFrame` contract retains the encoder heap through every
in-flight use and allows only one ordered execution flow per encoder/heap.
Recording and submission order must match execution order, including when
several CPU threads record work. Permitted resolution changes can coexist
in flight within the heap's supported resolution list. Resources retain their
layouts and references through all in-flight uses. Independent encoder/heap
instances provide separate stateful flows.
[Object lifetime and serialization][d3d12-encode-lifetime].

Draining an application pipeline includes all accepted frames and produced
bitstreams. Encoded output remains live through its consumer; reconstructed
pictures remain live through later reference uses. AMF exposes an explicit
drain operation for its asynchronous component, while a D3D12 caller manages
its own submission/output bookkeeping.
[AMF component lifecycle][amf-component].

## Processing and sharing

Video-processing support queries include input/output formats, color spaces,
frame rates, scaling, and feature/filter flags. Decode conversion has a
separate query. Motion estimation has its own estimator, input/reference
resources, motion-vector heap, and resolve operation. The driver selects the
implementation for each admitted operation.
[Processing support][d3d12-process], [decode conversion][d3d12-conversion],
[motion estimation][d3d12-motion].

Within D3D12, a decode-to-model flow is:

1. Allocate admitted decode output/reference resources and create the
   supported plane views needed by compute. Publish bitstream and picture
   metadata; retain all resources through their users.
2. Execute decode with the resources in their required video access states,
   perform the producer side of the inter-queue state handoff, and signal a
   fence value on the decode queue.
3. Queue a compute wait on that fence, perform the consumer side of the state
   handoff, and sample the planes into the final model tensor. Signal completion
   of that compute use.
4. Recycle an output slot only after compute and any remaining codec reference
   uses have finished. Reference-only storage, where required, remains separate.

The reverse flow publishes generated/converted input, signals its producer
fence, and queues an encode wait before encode-read access. Encode and metadata
resolve finish before the bitstream consumer reads the reported valid extent.
`ID3D12CommandQueue::Wait` queues a GPU wait and returns immediately; these
flows need not contain a per-frame application CPU wait. Resource states and
barriers still describe the actual media/compute access transition.
[Video resource states][d3d12-video-states], [queue wait semantics][d3d12-wait].

Sharing with another device/API uses separate handles for resources and fences.
`CreateSharedHandle` can export a supported resource/heap or fence. The importer
must understand the native allocation, layout, permitted access, and shared-fence
protocol. `GetCopyableFootprints` describes a texture's layout when copied into
a buffer, not its physical tiled texture layout.
[Shared handles][d3d12-sharing], [copy footprints][d3d12-footprints],
[external memory and synchronization](../../interop/external-memory.md).

rocDecode's Windows path uses a linear staging buffer. Each `GetVideoFrame`
copies the decoded D3D12 texture planes into that buffer using copyable
footprints. Its copy helper submits and waits for a D3D12 fence. HIP then
imports/maps the staging buffer's shared handle, caching that import across
frames. The cached mapping removes repeated setup; the per-frame copy and
completion wait remain.
[Frame mapping][rocdecode-interop], [staging copy and wait][rocdecode-staging].

`ID3D12Fence::GetCompletedValue` returns `UINT64_MAX` when the device is removed;
this value indicates device failure.
Native device failure and codec error feedback remain separate from resource
retirement and any stream restart decision.
[Fence failure contract][d3d12-fence-failure].

## AMF and the native boundary

AMF provides decoder and encoder components over the driver runtime. Its
context can wrap existing D3D11/D3D12 resources. D3D12 interop metadata names
the resource state, fence object, fence value, and decoded-resource state
restoration. [Native context wrapping][amf-context],
[D3D12 metadata][amf-d3d12].

The decoder's surface-copy property defaults to false; its required surface
pool still includes retained reference and presentation storage. The encoder
sample creates a native context, submits input asynchronously, handles input
backpressure, drains, and consumes output separately. Codec capability queries
remain necessary for the chosen format, resolution, quality, and reference
configuration. Using AMF adds a runtime dependency on its implementation.
[Decoder properties][amf-decoder], [encoder caller][amf-encoder-caller],
[component capabilities and lifecycle][amf-component].

The cited public WKMI queue definitions cover PM4, AQL, and SDMA, not JPEG/VCN.
Direct native use needs a documented engine/context identity, command
representation, allocation/residency and layout rules, and completion/error
ownership. D3D video plus supported resource/fence sharing already exposes
these operations at an API boundary; reusing them from a different compute
transport still requires an implemented import and ordering contract in that
transport.
[Public WKMI boundary][wkmi-media-boundary].

[d3d12-encode-availability]: https://learn.microsoft.com/en-us/windows-hardware/drivers/display/video-encoding-d3d12
[dxva-mjpeg]: https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/dxva/ns-dxva-dxva_picparams_mjpeg
[d3d12-support]: https://github.com/microsoft/DirectX-Headers/blob/461ef5138bd30922b44fbad77c4e47662812d2d5/include/directx/d3d12video.idl#L289-L350
[d3d12-encode-support]: https://github.com/microsoft/DirectX-Headers/blob/461ef5138bd30922b44fbad77c4e47662812d2d5/include/directx/d3d12video.idl#L2696-L2719
[d3d12-profiles]: https://github.com/microsoft/DirectX-Headers/blob/461ef5138bd30922b44fbad77c4e47662812d2d5/include/directx/d3d12video.idl#L1453-L1489
[d3d12-decode]: https://github.com/microsoft/DirectX-Headers/blob/461ef5138bd30922b44fbad77c4e47662812d2d5/include/directx/d3d12video.idl#L473-L519
[d3d12-resource-flags]: https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_resource_flags
[d3d12-decode-storage]: https://github.com/microsoft/DirectX-Headers/blob/461ef5138bd30922b44fbad77c4e47662812d2d5/include/directx/d3d12video.idl#L449-L470
[d3d12-encode]: https://github.com/microsoft/DirectX-Headers/blob/461ef5138bd30922b44fbad77c4e47662812d2d5/include/directx/d3d12video.idl#L3036-L3167
[d3d12-encode-metadata]: https://github.com/microsoft/DirectX-Headers/blob/461ef5138bd30922b44fbad77c4e47662812d2d5/include/directx/d3d12video.idl#L3097-L3167
[d3d12-encode-requirements]: https://github.com/microsoft/DirectX-Headers/blob/461ef5138bd30922b44fbad77c4e47662812d2d5/include/directx/d3d12video.idl#L2774-L2789
[mesa-d3d12-encode-wait]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/d3d12/d3d12_video_enc.cpp#L99-141
[mesa-d3d12-encode-resolve]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/d3d12/d3d12_video_enc.cpp#L4317-4418
[d3d12-encode-lifetime]: https://learn.microsoft.com/en-us/windows/win32/api/d3d12video/nf-d3d12video-id3d12videoencodecommandlist2-encodeframe
[amf-component]: https://github.com/GPUOpen-LibrariesAndSDKs/AMF/blob/57ee0cfb0e1a08d9041f77435653794c1b012be0/amf/public/include/components/Component.h#L114-L134
[d3d12-process]: https://github.com/microsoft/DirectX-Headers/blob/461ef5138bd30922b44fbad77c4e47662812d2d5/include/directx/d3d12video.idl#L590-L615
[d3d12-conversion]: https://github.com/microsoft/DirectX-Headers/blob/461ef5138bd30922b44fbad77c4e47662812d2d5/include/directx/d3d12video.idl#L413-L447
[d3d12-motion]: https://github.com/microsoft/DirectX-Headers/blob/461ef5138bd30922b44fbad77c4e47662812d2d5/include/directx/d3d12video.idl#L1000-L1052
[d3d12-video-states]: https://github.com/microsoft/DirectX-Headers/blob/461ef5138bd30922b44fbad77c4e47662812d2d5/include/directx/d3d12.idl#L2038-L2043
[d3d12-wait]: https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12commandqueue-wait
[d3d12-sharing]: https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-createsharedhandle
[d3d12-footprints]: https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-getcopyablefootprints
[rocdecode-interop]: https://github.com/ROCm/rocm-systems/blob/b679bae80581b21a864a858fb0c85c3fb991991a/projects/rocdecode/src/rocdecode/roc_decoder.cpp#L125-L254
[rocdecode-staging]: https://github.com/ROCm/rocm-systems/blob/b679bae80581b21a864a858fb0c85c3fb991991a/projects/rocdecode/src/rocdecode/vaapi/d3d12_interop.cpp#L372-L470
[d3d12-fence-failure]: https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12fence-getcompletedvalue
[amf-context]: https://github.com/GPUOpen-LibrariesAndSDKs/AMF/blob/57ee0cfb0e1a08d9041f77435653794c1b012be0/amf/public/include/core/Context.h#L109-L186
[amf-d3d12]: https://github.com/GPUOpen-LibrariesAndSDKs/AMF/blob/57ee0cfb0e1a08d9041f77435653794c1b012be0/amf/public/include/core/D3D12AMF.h#L39-L45
[amf-decoder]: https://github.com/GPUOpen-LibrariesAndSDKs/AMF/blob/57ee0cfb0e1a08d9041f77435653794c1b012be0/amf/public/include/components/VideoDecoderUVD.h#L43-L89
[amf-encoder-caller]: https://github.com/GPUOpen-LibrariesAndSDKs/AMF/blob/57ee0cfb0e1a08d9041f77435653794c1b012be0/amf/public/samples/CPPSamples/SimpleEncoder/SimpleEncoder.cpp#L153-L286
[wkmi-media-boundary]: https://github.com/ROCm/rocm-systems/blob/6b7847ef03b76c004f57b8cf40cdef9621ee6f53/shared/amdgpu-windows-interop/wkmi/wkmi.h#L137-L322
