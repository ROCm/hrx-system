# Primary sources

The reference combines architecture and ABI definitions with firmware-facing
packet layouts, native drivers, and upstream consumers. Each chapter cites the
specific definition, builder, caller, or resource owner that supports a claim.

| Source | Revision | Role |
| --- | --- | --- |
| [PAL][pal] | `c5e800072a32f68b6ccc4422936d96167c6e0728` | Generation-specific packet definitions, command builders, cache transitions, conditional compute blocks and Boolean sampling, and command-storage ownership. |
| [XGL][xgl] | `e9782eb33ce5e5e4ed2e339542a28c1b933624b4` | Vulkan conditional-rendering and buffer-update translation, application-derived command-buffer reuse flags and the PAL submission owner. |
| [Mesa][mesa] | `0ba4b08edc65075e9346d20d5310261939aaaf48` | RADV/radeonsi command composition, memory policy, synchronization, MEC predicate normalization and guarded command extents, and native submission. |
| [Mesa counter, barrier and shader policy][mesa-counter-barriers] | `44cc4ca677a4752a10c14194289bde5a6468675e` | GFX12 performance-query admission, event selection and SQG sample ordering; consumer-stage selection for graphics PWS waits; compiler-selected MEM_ORDERED mode, linked-program requirements and per-SA compute affinity. |
| [ROCm systems][rocm] | `8d57824901ffa7d961c00a37d055a108723b93ca` | ROCr queue/signal/copy protocols, HIP/CLR consumers, native queue construction, and profiling. |
| [Archived AMD Compute ABI][legacy-compute-abi] | `572ac3831ac5d30ac82f743d489a90bffba14a59` | Historical native signal object fields, used only where current ROCr representation and behavior corroborate them. The document is obsolete for current compiler and code-object ABI. |
| [ROCm sampling readback][rocm-sampling] | `e53154361af6ec17954c64c020e7c66701669ef4` | PC-sampling DMA chunk bounds, aggregate and record extents, profiler-selected buffer sizes, circular-buffer publication and the shared command-buffer owner. |
| [ROCm profiling protocols][rocm-profiling] | `eab38980a6f5cc0e4b17ec5437345e7680bbc936` | Dispatch-log stream ABI v4 consumer, kept separate from the version-3 native driver below; PC-sampling phase, command ownership, atomic scope, and record validity; SPM interval clocks, producer/collector ordering, and drain policy; SQTT replacement, overflow, and per-XCC anchors; repeated native clock sampling, counter power policy, GFX11 counter reads, separate observation queues, and result reuse. |
| [Tinygrad AMD runtime][tinygrad] | `e8308c08f77148dce3bfd141cbdfb1be66329a45` | Compute counter programming and reads, native profile-mode policy, physical WGP selection, dispatch sampling, and bounded result collection. |
| [ROCm XIO][xio] | `cbe97e6392066bef7901121965ffadad19404da4` | Device-authored SDMA packets, reservation/commit protocols and native queue-resource ownership; compared with the related rocSHMEM Anvil implementation in ROCm systems. |
| [ROCm native memory and engine policy][rocm-native] | `f9ba16bbe70e365b2f59b268e847bef19ad9db6e` | BO-backed queue storage, AQL metadata publication and scratch reclamation, signal/event notification and host waits, pageable host SVM allocation, context-save layout and ownership, DRM import/mapping ownership, CLR peer-engine selection, broadcast/multicast copy selection and completion ownership, indirect payload-address copies and HIP pointer-holder admission, buffer-exchange layouts and both-operand lifetimes, compute affinity, native queue priority, cooperative launch admission/ownership, and signal-backed AQL stream waits. |
| [ROCm copy, cache and visibility policy][rocm-runtime-policy] | `105dd4ff35798f95646353bc08f6c885416ae17e` | HIP admission of device-to-device swap and indirect copies, operand-based executor selection, batch counts and descriptor-control propagation, CLR's indirect-copy ISA predicate, inbound RDMA visibility and the HDP mapping owner, SVM dependency publication, asynchronous results and accepted-work ownership, SYSTEM-scoped batch-memory dispatch, persisting-cache requests and HIP access-window admission. |
| [ROCm compiler and device libraries][rocm-device-libs] | `8cd9ac8c8f12ab07e92229ea9d690b49e866b8ec` | OCKL hidden grid state, GWS versus atomic barrier selection, split arrival/wait, workgroup/agent/system fence composition, stream-memory polling and masked predicates, OpenCL child handles and device scheduler, and target-selected shader-cycle versus steady-time reads. |
| [Linux][linux] | `50d05c7c76c96b90462f24debacca971d2e86713` | Native UAPI, queue descriptors, memory mappings, engine emitters, and driver resource lifetimes. |
| [Linux profiling resume correction][linux-profile-resume] | `f931c54b241ce2f36bfc34955aec43a188276b8d` | Restoring native stable-pstate controls after runtime power transitions while the software profile level remains selected. |
| [Linux memory transports][linux-memory-transports] | `fe2ec83746e501645709761605c2464a44fd2929` | Exporter runtime PM references and their relation to P2P eligibility and memory placement; KFD's noncached HDP remap, native availability and host flush sequence; SVM prefetch placement, partial migration and native completion; VM ioctl and cache-topology counterparts to runtime size requests. |
| [ROCm AMDGPU native driver][rocm-amdgpu] | `820212794d184710298efcecce42c0215bfc9c6a` | Render-node VM ioctl, cache-topology publication, MES V12.1 partition-level scheduler coordination, native SPM buffer rotation, PC-sampling source control, and firmware dispatch-log streams. Compared with [the `releases/therock-7.14` revision][rocm-amdgpu-therock] `5081d704e60e807e9541c355c53df57d88ea8cde` for persisting-cache admission. |
| [Vulkan specification][vulkan] | `01aaacd99480487bf63830959513c5ca8ceb996d` | External memory and semaphore capabilities, handle ownership, host-pointer imports, buffer and address-range bounds, host-update capture, resource ownership transfers, host visibility, and performance-query lifetimes. |
| [LLVM ABI and memory model][llvm-abi] | `6e714c8d91116794cb699cdf80c26afe9cda3ef3` | Kernel descriptors, initial registers, wave-mode availability and target-feature validation, address spaces, shader memory ordering, availability/visibility, the matching cache-control emitter and target predicates, and MLIR's LDS barrier representation and initialization contract. |
| [LLVM compiler implementation][llvm-compiler] | `6dfe1677ab8dffbc6ec13d53a1e0215d75147689` | Executable fetch padding, dispatch inputs, partial workgroups, target feature selection, and XNACK replay-clause register retention. |
| [MLIR tensor descriptor lowering][mlir-tdm] | `906e3ec0647aef01363973edf3cb0104a812589a` | Gather-mode and index-width construction, compared with the ISA field table and other descriptor producers. |
| [Triton CDNA5 transfer lowering][triton-cdna5] | `8262c9a91a1d6828ad4f36437fa0046daa67720d` | Asynchronous and tensor transfer counts, descriptor notification, LDS split barriers and a workgroup producer/consumer reuse protocol. |
| [Composable Kernel tensor descriptors][ck-tdm] | `b51e8921db0e11a0a727032d7adcad94aabd8bdd` | Raw padding fields, iteration representation and gather-index packing. |
| [aiter Opus tensor descriptors][aiter-tdm] | `42c7e4a817e62bdb2052664c0aa8e6d78b176755` | Gather-mode encoding, index packing and element-based padding configuration with an explicit caller. |
| [ROCm target identities][therock-targets] | `1cc8ec570e9f5c720de9ce52dc485228c2df0274` | CDNA5 product-to-compiler-target mapping, kept separate from native GC and SDMA IP. |
| [AMD atomic-operation tables][legacy-rocm] | `85a16825737e43a14ff431754b359380e78062a7` | Architecture-specific atomic operation tables and their separate PCIe-route interpretations. |
| [Windows DDI][windows-ddi] | `7515063cea4c9e98db6a92986c5b4ddb0463fd16` | Native submission, monitored fences, mapping, residency, and destruction contracts. |
| [Windows driver model guides][windows-guides] | `2d03b8b58143ec94b2ae56e2ddc11ec64133fb2c` | Monitored-fence mapping and wait ownership, native GPU-fence enablement, storage and notification ordering. |
| [XDNA driver][xdna] | `8dfda66f67a84aecf26cf68336efc9e4cc1756c3` | Array contexts, firmware command envelopes, native completion, power management, and diagnostic access. |
| [AI Engine driver][aie] | `2855a032366e3d19dab893e7c263b14bb920cd64` | Register descriptions, timers, event counters, stream switches, DMA, and trace configuration. |
| [AI Engine binary utilities][aiebu] | `e82e28cbb237dcfd6c3029d85dc604515367ee24` | AIE2 transaction encoding, AIE2PS CERT controller instructions and task-wait ownership, and ELF platform identities. |
| [MLIR-AIE][mlir-aie] | `c69fb4c8f2fb853d5ca62d19f829796d3ae4ba34` | Array register data, trace-flow construction, and trace decoding. |
| [MLIR-AIE and IRON dataflow][mlir-aie-dataflow] | `41fa359ea1f66f7e5c572f8d0cc8c7646262adf5` | Object FIFOs, task ownership, descriptor allocation, DMA lowering, and stream routes. |
| [XRT runtime][xrt] | `ecad6cf22171ffec754fdd36afc2ae200af5c3a6` | Buffer synchronization delegation, ZYNQ/ZOCL GMIO channel completion, and managed external-buffer task ownership. |
| [AMD validation-data archive][vtd] | `fa945e489497397269cbb912e3e7cce33d1935ff` | Strix firmware trace schema, nanosecond timestamps, event attribution, and record framing. |
| [XDP][xdp] | `03ba80bf6c4942f51eebc71f7d154d9254426396` | Array profiling, trace storage and offload, and host/device timeline consumers. |
| [DynamicDispatch][dynamic-dispatch] | `b3051f03e20aab237cda3bbe4cd2081f76b72b06` | Firmware transaction construction and timestamp-marker encoding. |

HSA's [System Architecture 1.2][hsa-system] defines standard AQL packets,
publication, signal, and fence-scope semantics. Its [Programmer's Reference
Manual 1.2][hsa-prm] supplies work-item and workgroup execution rules. AMD's
[GPU architecture documentation index][gpu-manuals] locates individual ISA
manuals; shader instructions and command-processor packets have separate
representations and owners.

AMD's [Sea Islands 3D/Compute Register Reference Guide, revision 1.0,
19 September 2012][cik-registers], pp. 196–200, describes legacy dispatch
geometry and initiator controls. The [dispatch chapter](gpu/pm4/dispatch.md#initiator-field-family)
separates those meanings from later source layouts and actual caller policy;
the older cache and context fields are not a substitute for a newer native
publication or scheduling protocol.

The [RDNA4 ISA guide, 7 April 2025][rdna4-isa], §5.7 and Table 26, supplies
the architecture's dependency-counter rules used in the
[shader wait-mode discussion](gpu/pm4/dispatch.md#shader-wait-counter-mode-mem_ordered).

The [CDNA5 ISA guide, 27 July 2026][cdna5-isa], covers cache policy in
§§4.1.1–2, barriers in §5.6, dependency counters in §5.7, asynchronous memory
in §10.8, tensor movement in §10.11 and LDS split barriers in §11.2.2. The
[architecture map](gpu/architectures.md#cdna5-and-gfx1250) connects its naming
to public compiler and native-driver sources.

GPUOpen's [6 August 2026 machine-readable ISA archive][isa-xml] contains
`amdgpu_isa_cdna5.xml`, whose document revision is 3 August 2026 and schema
version is 1.2.0. Its instruction definitions provide a second representation
of the cache and wait operations; the descriptions and manual are both AMD
sources, rather than independent hardware observations.

AMD's [Micro Engine Scheduler specification, April 2024][mes-manual],
introduced by GPUOpen as an RDNA3 scheduling overview, describes the native
MES API and its scheduling fields. Its `ADD_QUEUE` flags corroborate the
[cooperative-launch discussion](gpu/cooperative.md#what-reaches-the-scheduler);
that revision does not establish the behavior of later MES implementations.

Microsoft's public [D3D12 sharing][d3d12-sharing] and [queue synchronization][d3d12-sync]
documentation supplies the API contracts for shared heaps, resources and
fences. Those object and execution contracts complement the native Windows
DDI; neither defines a device-specific NPU import protocol. The external-memory
chapter cites the particular creation, residency, mapping and completion APIs.

The AI Engine chapters also use AMD's AM020 and AM027 architecture manuals,
the AM025 register manual, UG1079 kernel coding guide, UG1603 revision 2026.1
for managed GMIO and external-memory dataflow, and the versioned
2025.1/2026.1 intrinsic references.
Each chapter identifies the relevant document, section, architecture, and
toolchain version.
Those architecture manuals do not by themselves establish how a native host
driver exposes the same mechanism.

Arm's [AMBA AXI and ACE specification IHI 0022F.b][axi4] defines the AXI4
transaction attributes, response ordering and single-copy atomicity used in
the array's external-memory discussion. The array and host integration
determine where those protocol observations meet another device's memory
and cache path.

Several consumers can share the same generated packet definition or delegate
to the same runtime. Agreement along that chain corroborates the caller flow;
it is not independent evidence about the underlying representation. A driver
submission wrapper can supply operations absent from the command buffer it
executes. The corresponding chapter identifies those boundaries.

[pal]: https://github.com/GPUOpen-Drivers/pal/tree/c5e800072a32f68b6ccc4422936d96167c6e0728
[xgl]: https://github.com/GPUOpen-Drivers/xgl/tree/e9782eb33ce5e5e4ed2e339542a28c1b933624b4
[mesa]: https://gitlab.freedesktop.org/mesa/mesa/-/tree/0ba4b08edc65075e9346d20d5310261939aaaf48
[mesa-counter-barriers]: https://gitlab.freedesktop.org/mesa/mesa/-/tree/44cc4ca677a4752a10c14194289bde5a6468675e
[rocm]: https://github.com/ROCm/rocm-systems/tree/8d57824901ffa7d961c00a37d055a108723b93ca
[legacy-compute-abi]: https://github.com/ROCm/ROCm-ComputeABI-Doc/blob/572ac3831ac5d30ac82f743d489a90bffba14a59/AMDGPU-ABI.md
[rocm-sampling]: https://github.com/ROCm/rocm-systems/tree/e53154361af6ec17954c64c020e7c66701669ef4
[rocm-profiling]: https://github.com/ROCm/rocm-systems/tree/eab38980a6f5cc0e4b17ec5437345e7680bbc936
[tinygrad]: https://github.com/tinygrad/tinygrad/tree/e8308c08f77148dce3bfd141cbdfb1be66329a45
[xio]: https://github.com/ROCm/rocm-xio/tree/cbe97e6392066bef7901121965ffadad19404da4
[rocm-native]: https://github.com/ROCm/rocm-systems/tree/f9ba16bbe70e365b2f59b268e847bef19ad9db6e
[rocm-runtime-policy]: https://github.com/ROCm/rocm-systems/tree/105dd4ff35798f95646353bc08f6c885416ae17e
[rocm-device-libs]: https://github.com/ROCm/llvm-project/tree/8cd9ac8c8f12ab07e92229ea9d690b49e866b8ec
[linux]: https://github.com/torvalds/linux/tree/50d05c7c76c96b90462f24debacca971d2e86713
[linux-profile-resume]: https://github.com/torvalds/linux/commit/f931c54b241ce2f36bfc34955aec43a188276b8d
[linux-memory-transports]: https://github.com/torvalds/linux/tree/fe2ec83746e501645709761605c2464a44fd2929
[rocm-amdgpu]: https://github.com/ROCm/amdgpu/tree/820212794d184710298efcecce42c0215bfc9c6a
[rocm-amdgpu-therock]: https://github.com/ROCm/amdgpu/tree/5081d704e60e807e9541c355c53df57d88ea8cde
[vulkan]: https://github.com/KhronosGroup/Vulkan-Docs/tree/01aaacd99480487bf63830959513c5ca8ceb996d
[llvm-abi]: https://github.com/llvm/llvm-project/tree/6e714c8d91116794cb699cdf80c26afe9cda3ef3
[llvm-compiler]: https://github.com/llvm/llvm-project/tree/6dfe1677ab8dffbc6ec13d53a1e0215d75147689
[mlir-tdm]: https://github.com/llvm/llvm-project/tree/906e3ec0647aef01363973edf3cb0104a812589a
[triton-cdna5]: https://github.com/triton-lang/triton/tree/8262c9a91a1d6828ad4f36437fa0046daa67720d
[ck-tdm]: https://github.com/ROCm/composable_kernel/tree/b51e8921db0e11a0a727032d7adcad94aabd8bdd
[aiter-tdm]: https://github.com/ROCm/aiter/tree/42c7e4a817e62bdb2052664c0aa8e6d78b176755
[therock-targets]: https://github.com/ROCm/TheRock/tree/1cc8ec570e9f5c720de9ce52dc485228c2df0274
[legacy-rocm]: https://github.com/ROCm/legacy-rocm-build/tree/85a16825737e43a14ff431754b359380e78062a7
[windows-ddi]: https://github.com/MicrosoftDocs/windows-driver-docs-ddi/tree/7515063cea4c9e98db6a92986c5b4ddb0463fd16
[windows-guides]: https://github.com/MicrosoftDocs/windows-driver-docs/tree/2d03b8b58143ec94b2ae56e2ddc11ec64133fb2c
[d3d12-sharing]: https://learn.microsoft.com/en-us/windows/win32/direct3d12/shared-heaps
[d3d12-sync]: https://learn.microsoft.com/en-us/windows/win32/direct3d12/user-mode-heap-synchronization
[xdna]: https://github.com/amd/xdna-driver/tree/8dfda66f67a84aecf26cf68336efc9e4cc1756c3
[aie]: https://github.com/Xilinx/aie-codegen/tree/2855a032366e3d19dab893e7c263b14bb920cd64
[aiebu]: https://github.com/Xilinx/aiebu/tree/e82e28cbb237dcfd6c3029d85dc604515367ee24
[mlir-aie]: https://github.com/Xilinx/mlir-aie/tree/c69fb4c8f2fb853d5ca62d19f829796d3ae4ba34
[mlir-aie-dataflow]: https://github.com/Xilinx/mlir-aie/tree/41fa359ea1f66f7e5c572f8d0cc8c7646262adf5
[xrt]: https://github.com/Xilinx/XRT/tree/ecad6cf22171ffec754fdd36afc2ae200af5c3a6
[vtd]: https://github.com/Xilinx/VTD/tree/fa945e489497397269cbb912e3e7cce33d1935ff
[xdp]: https://github.com/Xilinx/XDP/tree/03ba80bf6c4942f51eebc71f7d154d9254426396
[dynamic-dispatch]: https://github.com/amd/DynamicDispatch/tree/b3051f03e20aab237cda3bbe4cd2081f76b72b06
[hsa-system]: https://hsafoundation.com/wp-content/uploads/2021/02/HSA-SysArch-1.2.pdf
[hsa-prm]: https://hsafoundation.com/wp-content/uploads/2021/02/HSA-PRM-1.2.pdf
[gpu-manuals]: https://gpuopen.com/amd-gpu-architecture-programming-documentation/
[cik-registers]: https://www.x.org/docs/AMD/old/CIK_3D_registers_v2.pdf#page=196
[rdna4-isa]: https://gpuopen.com/download/rdna4-instruction-set-architecture.pdf
[cdna5-isa]: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna5-instruction-set-architecture.pdf
[isa-xml]: https://gpuopen.com/download/AMD_GPU_MR_ISA_XML_2026_08_06.zip
[mes-manual]: https://gpuopen.com/download/documentation/micro_engine_scheduler.pdf
[axi4]: https://documentation-service.arm.com/static/5f915bbcf86e16515cdc3b23
