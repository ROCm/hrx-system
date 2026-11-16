# Use Vulkan drivers as native-code oracles

Vulkan drivers compile SPIR-V into native GPU programs. Inspecting the program
behind a selected compute pipeline makes an optimized Vulkan application useful
in two distinct roles:

- a performance reference for one production operation; and
- a compiler oracle for how SPIR-V specialized into native instructions.

A driver-generated program is not a source-of-truth schedule. The useful
evidence is the shader actually selected for an exact dispatch, joined to its
input program, launch geometry, native listing, and measured boundary. A shader
that merely resembles the operation or targets the same GPU is not the oracle.

## Freeze the reference dispatch

Start from the [production witness](../agent-driven-kernel-development.md#freeze-one-production-witness),
then record the Vulkan-specific identity:

| Identity | Evidence |
| --- | --- |
| Runtime | Application revision, executable and loaded-library hashes, Vulkan device, driver implementation, and driver build. |
| Pipeline | Pipeline or shader identity, specialization constants, descriptor layout, push constants, and source or generated SPIR-V. |
| Dispatch | Exact `vkCmdDispatch` grid, local size, subgroup size, and every dispatch included in the semantic cut. |
| Data | Parameter checkpoint and format, boundary inputs, layouts, residency, and cache policy. |
| Score | Uninstrumented measured boundary, time domain, warmup, batching, interleaving, and machine-ownership policy. |

The physical dispatch is part of the join key. A runtime may select different
pipelines on the two sides of a shape cliff, route a nominally identical graph
through different packing kernels, or use several shader variants with the same
entry-point name.

Trace or instrument pipeline creation and `vkCmdDispatch` in the reference
runtime when its own diagnostics do not expose that join. Record the complete
grid at the call boundary, then associate the command buffer with its submission
and request phase. A recorded dispatch may belong to an unsubmitted command
buffer or startup tuning rather than the measured request. Debugger register
names are host-ABI details, so a saved breakpoint script also records its
debugger, architecture, and calling
convention rather than presenting one register sequence as portable Vulkan
instrumentation.

## Capture native code from the selected pipeline

SPIR-V is the input to the driver compiler. Its operations, vector widths, and
matrix types do not establish the final instruction mix or register allocation.
Keep the SPIR-V and its specialization beside the native program so that each
compiler boundary can be examined independently.

When supported, `VK_KHR_pipeline_executable_properties` exposes executable
properties, statistics, and internal representations. Enable
`pipelineExecutableInfo` on the device and the corresponding
`VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR` and
`VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR` flags when
creating the pipeline. Query the resulting executable with
`vkGetPipelineExecutablePropertiesKHR`, `vkGetPipelineExecutableStatisticsKHR`,
and `vkGetPipelineExecutableInternalRepresentationsKHR`. A driver may expose
statistics while returning no internal representations; an empty representation
list is not an assembly listing. [Vulkan executable inspection][pipeline-inspection]

`VK_KHR_pipeline_binary` provides another capture boundary. Enable the
`pipelineBinaries` device feature and create the pipeline with
`VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR` in a
`VkPipelineCreateFlags2CreateInfo` chain. Its flags replace the ordinary create
flags, so include the original flags and any executable-inspection flags there
too. Obtain the binary objects with `vkCreatePipelineBinariesKHR` and retrieve
their keys and bytes with `vkGetPipelineBinaryDataKHR`. These bytes use a
driver-defined format. Preserve the original payload and driver identity, and
use a decoder for that format before treating any extracted section as native
instructions. The capture preserves the original shader, entry point,
specialization, pipeline layout, subgroup controls, and optimization settings.
[Vulkan pipeline binaries][pipeline-binary]

Capture and timing remain separate runs. Native listings establish emitted
operations; register and shared-memory counts constrain occupancy. Achieved
occupancy, active lanes, cache behavior, and pipeline utilization require
measurements from a supported profiler with the necessary counter permissions.

### Mesa RADV

RADV exposes the ACO-generated AMDGPU program and shader statistics.

Run a short capture with Mesa's disk shader cache disabled and RADV shader
debugging enabled:

```shell
MESA_SHADER_CACHE_DISABLE=true \
RADV_DEBUG=shaders \
reference-command reference-arguments \
  >reference.stdout \
  2>reference.radv.log
```

This is an evidence run, not a score run. Disabling the cache recompiles
pipelines; printing shaders adds I/O and can change process behavior. Collect
performance from a separately pinned, uninstrumented run with the same
executable, workload, pipeline-selection state, and semantic boundary.

Large applications may print many internal and application shaders. Match the
selected native listing back to the pipeline and dispatch using runtime-owned
pipeline statistics, a capture-layer identity, or a temporary source
instrumentation point. Name the filter or patch in the witness. Searching the
log for an appealing matrix instruction does not establish selection.

Keep these RADV products together:

- the generated or captured SPIR-V and its hash;
- the complete ACO listing and shader statistics;
- local size, subgroup size, SGPRs, VGPRs, LDS, scratch, spills, and code size;
- pipeline specialization and descriptor facts; and
- the exact physical grid observed at `vkCmdDispatch`.

The [Mesa RADV documentation](https://docs.mesa3d.org/drivers/radv.html)
describes the SPIR-V to NIR to ACO compiler path. Mesa's
[environment-variable reference](https://docs.mesa3d.org/envvars.html)
owns the disk-cache control; record the Mesa revision because debug controls
are not an application ABI.

### NVIDIA driver

The driver can expose resource statistics through executable inspection even
when it exposes no internal representations. A captured pipeline binary may
contain an `NVuc` shader object inside a compressed driver payload. After decoding
the object and locating its native code section, `nvdisasm` can disassemble the
raw instructions for an explicitly selected architecture. For an SM89 program:

```shell
nvdisasm --binary SM89 shader-code.bin > shader.sass
```

`shader-code.bin` contains only the native code section, not the NVuc header or
original Vulkan binary payload. Direct NVuc input depends on the tool version
and object format; successful decoding of one object does not qualify another.
The container layout is driver-specific; retain the decoder revision, supported
format, original bytes, and extracted-object hash. The
[nvcachetools implementation][nvcachetools] documents one such object layout.
Cross-check code-section sizes against executable statistics and disassemble
for the captured GPU architecture. Unsupported formats remain unqualified.

For CUDA reference implementations, `cuobjdump` extracts native cubins and
resource usage from the executable or shared library; `nvdisasm` also provides
control-flow and register-liveness views. Select the cubin matching the executing
architecture and identify the launched function before comparing it with a
Vulkan shader. [NVIDIA binary utilities][binary-utilities]

Nsight Graphics provides Vulkan shader resource, instruction-mix, and stall
analysis. Its full assembly view is documented as requiring the Pro build;
binary extraction and disassembly are a separate route.
[Nsight Graphics Shader Profiler][shader-profiler]

## Reconstruct the algorithm, not one headline fact

Native code is most useful after reconstructing its ownership and traffic. Fill
out the same worksheet for the reference and Loom candidate:

| Property | Question |
| --- | --- |
| Output ownership | How many outputs does one workgroup, subgroup, wave, and lane publish? |
| Reduction ownership | Which lanes own K, token, head, or expert slices, and where are partials combined? |
| Lane cohorts | Which lanes cooperate on a load, matrix fragment, reduction, or output? |
| Workgroup shape | How many subgroups are present, and which dimensions create independent work? |
| Collectives | What is the scope and order of shuffles, reductions, broadcasts, and scans? |
| Shared memory | Which objects use it, which values cross subgroups, and which barriers protect reuse? |
| Matrix work | What fragment form, instruction count, accumulator chains, and packing surround each matrix instruction? |
| Global memory | What address form and packet width does each lane issue, and what transactions does the whole wave create? |
| Waits | Which dependencies cause waits, clauses, barriers, or full drains? |
| Physical grid | How does the runtime map the production shape onto dispatches? |

“The shader uses a larger subgroup,” “it uses a matrix instruction,” and “it
uses fewer registers” are observations, not schedules. They become hypotheses
only after the worksheet names the work those choices organize.

Changing Loom's subgroup size in isolation is therefore not a reference-matching
experiment. The corresponding ownership, lane cohorts, collective scopes,
fragment layouts, shared-memory exchanges, workgroup shape, and independent
accumulator chains form one schedule bundle. Reducing registers or shared memory
may create occupancy headroom, but the bundle must then fill that headroom with
useful work and latency overlap. Selected partial variants explain the
interaction; none is a promise that one target fact produces a fixed percentage
improvement.

Inspect address expressions as wave-level access patterns. Widening one lane's
packet can make neighboring lanes touch distant rows and replace a coalesced
transaction with many independent transactions. Likewise, higher modeled
occupancy can accompany less independent work or more synchronization. The
native listing constrains the explanation; it does not replace physical
measurement.

## Compare Vulkan and native AMDGPU compilation

On a supported AMD GPU, targetless Loom source can be exercised through the
Vulkan/SPIR-V path and the native AMDGPU path without forking the algorithm:

```shell
loom-test kernel.loom \
  --case=@production_cut_case \
  --device=vulkan

loom-test kernel.loom \
  --case=@production_cut_case \
  --device=amdgpu
```

Use the same checked benchmark catalog for controlled physical comparison:

```shell
loom-benchmark kernel.loom \
  --benchmark=@production_bucket \
  --device=vulkan \
  --output-format=jsonl \
  --output=vulkan.jsonl

loom-benchmark kernel.loom \
  --benchmark=@production_bucket \
  --device=amdgpu \
  --output-format=jsonl \
  --output=amdgpu.jsonl
```

These two runs prove portability and produce backend-local measurements. They
are not automatically a causal backend comparison: device timing mechanisms,
submission paths, instrumentation perturbation, and executable boundaries must
also match under the [benchmark timing contract](../benchmark.md#collect-instrumented-device-timing-outside-the-score).

The SPIR-V path answers whether Loom expresses enough structure for a mature
Vulkan compiler to recover a good program. The native path exposes the
lower-level schedule and target-specific mechanisms Loom can control directly.
When the same source lowers successfully through both, differences in address
forms, waits, clauses, allocation, and traffic become focused compiler
questions rather than reasons to maintain separate model implementations.

## Turn a mismatch into a compiler question

Compare the reference worksheet with Loom's compile report and native artifact. A
useful candidate record names one missing mechanism:

```text
Witness:              exact reference dispatch and production shape
Reference mechanism:  ownership, traffic, or dependency pattern in the selected native code
Loom mechanism:       corresponding report and native-artifact evidence
Expected delta:       the one compiler or source change that should close it
Correctness gates:    Vulkan, native AMDGPU, and captured-boundary checks
Physical gate:        one discriminating uninstrumented workload
```

If valid targetless source cannot lower through SPIR-V, reduce the rejection to
a standalone compiler packet. If the source lowers but the selected mechanism
is poor, preserve the report and native delta. A model-local intrinsic or
mnemonic that bypasses the missing generic behavior hides the portability
question instead of answering it.

## Know what native-code evidence proves

| Evidence | Supported conclusion | Unsupported promotion |
| --- | --- | --- |
| Selected shader plus dispatch trace | This native program executed for this reference witness. | Every nearby shape or pipeline uses it. |
| Native listing and driver statistics | The selected compiler emitted these instructions and reported these resource counts. | The schedule is numerically correct or faster. |
| Matched source-to-output check | The two paths agree on this numerical boundary. | They have the same algorithm or physical traffic. |
| Calibrated device timing | This matched boundary was faster under this timing contract. | One isolated dispatch explains end-to-end speed. |
| Similar candidate and reference native listings | The programs share visible mechanisms. | Their dynamic requests, waits, or cache behavior are equivalent. |

The completed oracle packet contains the pinned runtime witness, capture method,
SPIR-V, selected native program, launch facts, uninstrumented reference
score, Loom reports and artifacts, correctness results, and the remaining
mechanism-level delta. That packet is durable enough for another agent to
challenge the interpretation rather than trusting a screenshot or a remembered
headline number.

[pipeline-inspection]: https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_pipeline_executable_properties.html
[pipeline-binary]: https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_pipeline_binary.html
[nvcachetools]: https://github.com/therontarigo/nvcachetools
[binary-utilities]: https://docs.nvidia.com/cuda/cuda-binary-utilities/
[shader-profiler]: https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html
