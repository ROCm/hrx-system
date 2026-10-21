# AMD hardware reference

This is an **unofficial guide**, assembled through best-effort reverse
engineering and distillation of publicly available repositories and
documentation, together with our own experiments. It is not an official AMD
specification and uses no confidential or non-public vendor material.

Correctness and completeness are not guaranteed. Citations and experimental
results provide evidence for the specific behavior and conditions they cover;
they do not establish correctness on other hardware, firmware, driver or
software configurations.

This reference describes AMD hardware and firmware programming: command and
register representations, execution ordering, memory visibility, resource
lifetime, and architecture-specific behavior. Native driver interfaces and
upstream runtime implementations supply the context needed to use those
mechanisms on a particular platform.

| Area | Contents |
| --- | --- |
| [GPU](gpu/README.md) | Command processors, transfer and media engines, and their memory and signaling protocols. |
| [XDNA](xdna/README.md) | AI Engine array execution, DMA, interconnects, firmware commands, and observation. |
| [CPU, GPU and NPU interop](interop/README.md) | Every directed memory handoff, external API sharing, resident credits, composed pipelines and final reuse. |
| [Sources](sources.md) | Architecture specifications, immutable implementation revisions, and the role of each source. |
| [Writing guide](STYLE.md) | Chapter structure, terminology, evidence, and citation conventions. |

Each mechanism carries its own architecture, firmware, transport, and memory
conditions. A compiler target, physical engine revision, operating-system
interface, and runtime policy describe different parts of that applicability.
Where sources disagree, the affected fields and source-specific interpretations
remain explicit.

## Programming tasks

The family indexes organize the hardware mechanisms. These routes start from
the operation a native caller needs to perform; packet and register names also
provide exact search terms within the tree.

| Task | Starting points |
| --- | --- |
| Identify the compiler target, physical GC or SDMA IP, and native transport | [Architecture identity and discovery](gpu/architectures.md). |
| Select a legal wave32 or wave64 program mode | [Wavefront modes and target restrictions](gpu/architectures.md#wavefront-modes), [PM4 launch and waves-per-workgroup policy](gpu/pm4/dispatch.md#register-binding-and-launch). |
| Select eligible compute resources and queue priority | [CU/WGP affinity, harvested SE/SH/XCC mapping, `COMPUTE_STATIC_THREAD_MGMT_SE*` and native priority translation](gpu/scheduling.md). |
| Join workgroups inside a cooperative dispatch | [Shared cooperative queues, `ALLOC_QUEUE_GWS`, occupancy and grid barrier memory scopes](gpu/cooperative.md). |
| Share input traffic across a workgroup cluster | [CDNA5 cluster geometry, `CLUSTER_LOAD`, multicast masks, completion and LDS reuse](gpu/clusters.md). |
| Reserve, publish and reuse queue storage | [PM4 ring frontiers](gpu/pm4/publication.md), [AQL header and doorbell publication](gpu/aql/publication.md), [SDMA reservation and ordered commit](gpu/sdma/publication.md). |
| Populate an AQL metadata-prefetch ring | [Paired slots, four headers, kernarg preload and `CP_HQD_KD_CNTL`](gpu/aql/metadata.md). |
| Wake a host when GPU work changes a signal | [Native mailbox/event representation, interrupt decoding and check-to-sleep ordering](gpu/notifications.md). |
| Submit Windows GPU work and observe native completion | [WDDM command acceptance, `RingBufferFenceRelease`, monitored fences and native GPU waits](gpu/wddm.md). |
| Launch a compiled GPU program | [PM4 `SET_SH_REG` and `DISPATCH_DIRECT`](gpu/pm4/dispatch.md), [AQL kernel dispatch and descriptors](gpu/aql/dispatch.md). |
| Bind shared workgroup storage | [PM4 `LDS_SIZE`](gpu/pm4/lds.md), [AQL group storage](gpu/aql/dispatch.md#static-and-dynamic-group-storage). |
| Stream tensor tiles through LDS | [TDM descriptor fields, stride units, padding and row gather/scatter](gpu/tensor-memory.md), [asynchronous completion and ready/empty slot reuse](gpu/async-memory.md). |
| Size and reclaim private scratch backing | [AQL `COMPUTE_TMPRING_SIZE`, physical slots and firmware ownership](gpu/aql/scratch.md). |
| Size compute context-save storage and preserve suspended work | [CWSR, `ctx_save_restore_size`, per-XCC headers and native removal](gpu/context-save.md). |
| Determine which agents can access a memory pool | [Pool grain, per-agent access and SVM host access](gpu/recipes/host-device.md#pool-grain-agent-access-and-svm). |
| Interpret SVM prefetch completion and reuse its resources | [Page residency and native results](gpu/recipes/host-device.md#prefetch-completion-and-page-residency), [accepted work and signal reuse](gpu/recipes/host-device.md#accepted-work-and-signal-reuse). |
| Make a producer's writes visible to its consumer | [GPU cache controls](gpu/pm4/cache.md), [CPU/GPU handoffs](gpu/recipes/host-device.md), [all six CPU/GPU/NPU directions](interop/README.md). |
| Retain repeatedly used GPU data in cache | [Temporal hints, persisting-size requests, native admission and HIP access windows](gpu/cache-residency.md). |
| Consume inbound RDMA writes | [Host visibility, `hipDeviceFlushGPUDirectRDMAWrites`, HDP mapping and independent GPU acquisition](interop/rdma.md). |
| Copy or fill memory and wait for completion | [SDMA packet index](gpu/sdma/README.md), [PM4 `DMA_DATA`](gpu/pm4/dma.md), [SDMA upload → AQL dispatch → SDMA download](gpu/recipes/README.md#sdma-upload-aql-dispatch-and-sdma-download). |
| Decode images/video into model input or encode generated frames | [Media engines and compute handoff](gpu/media.md), [JPEG packets](gpu/media/jpeg.md), [video decode](gpu/media/video-decode.md), [video encode](gpu/media/video-encode.md), and [Windows media](gpu/media/windows.md). |
| Reuse host input after an upload | [Source consumption, `hipMemcpySrcAccessOrder`, staging lifetime and destination completion](gpu/recipes/host-device.md#source-consumption-and-copy-completion). |
| Replace or reuse commands and executable storage | [PM4 indirect buffers](gpu/pm4/command-buffers.md), [AQL command carriers](gpu/aql/transfers.md), [AQL executable lifetime](gpu/aql/dispatch.md#executable-publication-and-final-use), [SDMA command buffers](gpu/sdma/command-buffers.md). |
| Configure an NPU transfer or split/join flow | [Tile DMA descriptors and task tokens](xdna/dma.md), [stream switches and multicast](xdna/interconnects.md), [pipeline ownership](interop/pipelines.md). |
| Exchange resident GPU/NPU payloads and return credits | [GPU/NPU ready and completion edges](gpu/recipes/gpu-npu.md#resident-programs-and-per-generation-ownership), [slot generations and drain](interop/pipelines.md). |
| Share Vulkan or D3D12 resources with a native consumer | [External memory, handle ownership and dependency objects](interop/external-memory.md). |
| Measure dispatch, transfer or tile execution | [GPU clocks and counter ownership](gpu/observability.md), [PM4 timing bracket](gpu/pm4/timing.md#timestamp-visibility-and-storage-lifetime), [SDMA timestamps](gpu/sdma/timing.md), [XDNA timers, counters and trace](xdna/observability.md). |
