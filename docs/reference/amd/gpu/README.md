# GPU programming

AMD GPUs expose command-processing, transfer, and media engines with distinct packet
representations, execution boundaries, and cache responsibilities. A command
stream combines these mechanisms with the native queue's publication and
resource-lifetime protocol.

| Area | Topics |
| --- | --- |
| [Architecture and transport](architectures.md) | Physical IP, compiler targets, wave32/wave64 availability, native queues, and submission ownership. |
| [Native compute queue contexts](queue-context.md) | MQD/HQD layouts, ring and pointer units, direct loading versus CP/MES scheduling, per-XCC images, metadata and storage lifetime. |
| [Compute affinity and queue priority](scheduling.md) | CU/WGP mask units, harvested SE/SH/XCC placement, `COMPUTE_STATIC_THREAD_MGMT_SE*`, native priority encodings and update ownership. |
| [Cooperative execution and grid synchronization](cooperative.md) | Shared cooperative queues, `ALLOC_QUEUE_GWS`, firmware admission, occupancy, hidden grid state and GWS/atomic barrier scopes. |
| [Compute context save and restore](context-save.md) | CWSR, `ctx_save_restore_size`, per-XCC storage, control-stack inspection, suspension and final ownership. |
| [Shader memory publication](shader-memory.md) | Resident release/acquire sequences, `s_waitcnt`, `buffer_inv`, `global_wb`, cache scopes, and wave/workgroup joins across GCN, CDNA and RDNA. |
| [Cache residency and replacement policy](cache-residency.md) | Temporal hints, `HSA_AMD_AGENT_INFO_MAX_PERSISTING_L2_CACHE_SIZE`, native size-request admission and HIP access-window semantics. |
| [Asynchronous global/LDS transfers](async-memory.md) | CDNA5 `ASYNCcnt`, `TENSORcnt`, TDM descriptor notifications, LDS split-barrier phases, and producer/consumer storage reuse. |
| [Tensor descriptors and LDS layouts](tensor-memory.md) | CDNA5 TDM address/stride units, 1D–5D fields, iteration, padding, row gather/scatter, descriptor advancement and per-wave issue. |
| [Workgroup clusters and multicast](clusters.md) | CDNA5 `CLUSTER_LOAD`, `workgroup_mask`, per-requestor completion, cluster barrier membership, extended AQL geometry and local LDS reuse. |
| [Native signals and host notification](notifications.md) | `event_mailbox_ptr`, KFD event ages, interrupt decoding, sleeping waits and notification-storage lifetime. |
| [Windows submission and fences](wddm.md) | WDDM context and hardware-queue submissions, accepted work, monitored-fence width, native progress and final storage use. |
| [PM4](pm4/README.md) | Compute dispatch, memory commands, cache control, and command-buffer execution. |
| [AQL](aql/README.md) | Packet publication, dispatch, dependencies, vendor command carriers, and profiling. |
| [SDMA](sdma/README.md) | Transfer-engine operations and signaling protocols. |
| [Media engines](media.md) | [JPEG `PACKETJ`](media/jpeg.md), [VCN decode](media/video-decode.md), [VCN encode](media/video-encode.md), and [Windows video](media/windows.md); shared surfaces, reference lifetimes, dependencies, and memory/traffic accounting. |
| [Programming recipes](recipes/README.md) | Host/device, cross-engine, and local-memory producer/consumer flows. |
| [CPU, GPU and NPU interop](../interop/README.md) | External sharing, directed device handoffs, resident pipelines and storage reuse. |
| [Timing and counters](observability.md) | Shader cycles (`SHADER_CYCLES`, `S_MEMTIME`), fixed-frequency `REALTIME`, timestamp conversion, counter ownership, and profiling interference. |
| [Programmable counter sampling](profiling/counter-sampling.md) | Workload participation, separate observation queues, running samples, register-read protocols, completion, and bounded result reuse. |
| [Streaming performance monitoring](profiling/spm.md) | SPM sample layouts, SCLK/REFCLK intervals, RLC rings, vendor KFD host-buffer rotation, device-scoped start/stop, per-XCC extents, loss and completed storage use. |
| [Program-counter sampling](profiling/pc-sampling.md) | Native trigger methods, trap records, per-XCC buffer rotation, reservation versus publication, loss and final trap users. |
| [Shader thread traces](profiling/thread-trace.md) | SQTT selection, finite drain, live GFX9/GFX12 buffer replacement, hardware-slot phase, per-XCC clock anchors, overflow and decoder context. |
| [Firmware dispatch logs](profiling/dispatch-log.md) | Vendor KFD `DLOG` stream ABI versions, native MQD fields, start/end records, overwrite detection and its limits, queue identity and mapped-storage lifetime. |

Compiler target, physical graphics/compute IP, SDMA IP, firmware, and native
transport identify different parts of a programming contract. The operation
chapters retain those conditions beside the packet fields and sequences they
govern.
