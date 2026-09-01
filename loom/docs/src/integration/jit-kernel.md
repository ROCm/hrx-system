# Embed kernel JIT compilation

**Example files:** [`loom/docs/examples/integration/jit-kernel/`](https://github.com/ROCm/hrx-system/tree/main/loom/docs/examples/integration/jit-kernel)

The public `loomc` API exposes each compiler boundary as an in-memory operation.
An application can load authored text or bytecode, link a root from reusable
libraries, specialize it for the selected device, emit executable bytes, and
evaluate the matching launch configuration without invoking a subprocess or
round-tripping through temporary artifacts.

This chapter follows one complete AMDGPU example. AMDGPU appears only where the
embedding creates a target environment and profile and where it requests an
HSACO. Source loading, compilation, diagnostics, launch evaluation, artifact
ownership, and concurrency use the same core API for every target package.

In this chapter, you will learn:

- which compiler objects are reusable and which remain worker-local;
- how one compile invocation specializes an exported kernel for a profile;
- why the launch-config companion and executable are coupled products;
- how workload values become a concrete launch without becoming device
  arguments; and
- where the compiler handoff ends and the target runtime begins.

## Follow one kernel through the boundary

The source gives the host launch-config program one workload value and
gives the device body no arguments:

**Source:** [`loom/docs/examples/integration/jit-kernel/kernel.loom`](https://github.com/ROCm/hrx-system/blob/main/loom/docs/examples/integration/jit-kernel/kernel.loom)

```loom title="kernel.loom"
--8<-- "examples/integration/jit-kernel/kernel.loom"
```

For an element count of 1009, the launch region computes 16 workgroups of 64
invocations. `%element_count` is not a push constant: it is absent from the
`launch()` signature and is consumed only by the compiled host companion. The
empty device body is deliberate here—the example isolates the integration
lifecycle, while the programming guide's [kernel
chapter](../guide/kernels-and-launch.md) develops real device computation and
the independent launch ABI.

The compiler path produces one immutable kernel product from this definition:

| Product field | Role |
| --- | --- |
| Target-native artifact | Contains the specialized executable kernel entry. |
| VM launch-config artifact | Contains a pure host function that maps authored workload arguments to the complete physical launch. |
| Root projection | Carries the exact artifact and function ordinals joining those two facets. |

The product builder compiles and emits both artifacts from one closed function
version. Loading code follows the root ordinals directly; it does not scan
artifact lists, compare export names, or run a second target emission.

## Prepare immutable state once

A target package creates the target environment registered with the context:

**Source:** [`loom/docs/examples/integration/jit-kernel/amdgpu/jit_kernel.c`](https://github.com/ROCm/hrx-system/blob/main/loom/docs/examples/integration/jit-kernel/amdgpu/jit_kernel.c)

```c
--8<-- "examples/integration/jit-kernel/amdgpu/jit_kernel.c:target-context"
```

The target environment and context are immutable after creation. The example
also creates one compiler, one target profile, and one prepared target
pipeline. Those objects can be shared across independent workers. Each worker
uses its own `loomc_workspace_t`:

**Source:** [`loom/docs/examples/integration/jit-kernel/amdgpu/jit_kernel.c`](https://github.com/ROCm/hrx-system/blob/main/loom/docs/examples/integration/jit-kernel/amdgpu/jit_kernel.c)

```c
--8<-- "examples/integration/jit-kernel/amdgpu/jit_kernel.c:profile-pipeline"
```

The example's build rule assembles the checked-in `.loom` file into ordinary
`.loombc`, then passes that bytecode path to the executable. A model loader
normally calls `loomc_source_create` with borrowed, copied, or externally owned
`.loombc` bytes instead. The request, compiler, and product lifecycle remains
filesystem-free.

Target profiles are ordinary prepared objects, not process-global compiler
flags. A runtime adapter may derive an exact profile from a live HSA, Vulkan, or
HAL device; an offline builder may construct a generic profile such as
`gfx11-generic` explicitly.

## Name the immutable product request

Product requests use exact source ordinals, not symbol strings. The example
indexes its bytecode once, resolves the public kernel name at that external
discovery boundary, and creates one host-launchable root:

**Source:** [`loom/docs/examples/integration/jit-kernel/amdgpu/jit_kernel.c`](https://github.com/ROCm/hrx-system/blob/main/loom/docs/examples/integration/jit-kernel/amdgpu/jit_kernel.c)

```c
--8<-- "examples/integration/jit-kernel/amdgpu/jit_kernel.c:request"
```

`LOOMC_KERNEL_ROOT_GOAL_HOST_LAUNCHABLE` requests both the physical executable
entry and its authored host launch policy. A parent command program that
already owns physical launch geometry instead requests an executable-entry root
and does not pull the VM launch artifact into that product.

## Build the coupled kernel product

Product construction receives the immutable request, selected pass program,
per-function target specialization, and target-native emission options:

**Source:** [`loom/docs/examples/integration/jit-kernel/amdgpu/jit_kernel.c`](https://github.com/ROCm/hrx-system/blob/main/loom/docs/examples/integration/jit-kernel/amdgpu/jit_kernel.c)

```c
--8<-- "examples/integration/jit-kernel/amdgpu/jit_kernel.c:compile"
```

`loomc_target_specialization_t` binds one function version to one complete
target profile for this invocation. Other requested roots can select other
profiles in the same product operation. Configuration bindings use the same
per-invocation boundary, so a compiler and pass program remain reusable across
model configurations and autotuning candidates.

A heterogeneous module can instead use `loomc_target_binding_t` to bind an
authored `target.decl` context to a profile. Every function carrying that target
declaration is seeded from the binding, and retained semantic callees inherit
the context during callgraph specialization. Direct function rows remain useful
for default host roots; the two forms can share one invocation as long as they
do not assign the same function.

`loomc_kernel_product_build_request` deserializes the immutable request into
worker-local IR, specializes and lowers its complete closure, emits the native
artifact, and emits one shared launch companion when any root is
host-launchable. The returned product retains only immutable artifacts and root
rows; it retains no request, workspace, mutable IR, or compiler analysis state.

## Load and evaluate the host companion

The loader is supplied by the optional `loomc/target/vm` package. Core
`loomc` defines the target-independent launch result and can compile other
products without linking the VM runtime.

The kernel root names the launch artifact and function by exact ordinal.
Loading the artifact produces an independently retained program, and
`loomc_vm_launch_config_program_function_at` binds the function without a
string lookup:

**Source:** [`loom/docs/examples/integration/jit-kernel/amdgpu/jit_kernel.c`](https://github.com/ROCm/hrx-system/blob/main/loom/docs/examples/integration/jit-kernel/amdgpu/jit_kernel.c)

```c
--8<-- "examples/integration/jit-kernel/amdgpu/jit_kernel.c:launch-config"
```

The product and program use separate ordinal spaces. The root row is the
one-time compiler-owned join between them; the returned program-local token is
the hot-path callable. Name lookup remains available for discovery of
independently authored VM artifacts, but a kernel-product consumer does not
need it.

Workload arguments are positional raw scalar bits. `index` and `offset` consume
all 64 bits; narrower integers and floating-point values use the least
significant bits matching their declared type. Invocation validates the
argument count, types, and authored facts before returning:

- workgroup count;
- workgroup size;
- workgroup-cluster size;
- subgroup size; and
- total workgroup-local storage required by the compiled kernel.

The application consumes this result. It does not duplicate ceiling division,
guess subgroup width, or recover shared-memory use from a kernel-name
convention.

## Consume the target-native artifact

The same root row names the target-native artifact and its artifact-local
function ordinal. A JIT passes `artifact->contents` and that ordinal to the
selected runtime loader before releasing the product. An offline packager can
write the same immutable artifact to a file without changing product
construction.

The example prints the two facts an embedding carries into runtime
loading:

```text
workload=1009 workgroups=16x1x1 workgroup_size=64x1x1
artifact=guide_jit_kernel.hsaco format=amdgpu-hsaco bytes=<target-dependent>
```

Artifact size is target- and compiler-version-dependent; it is not part of the
interface. The format, public export, device ABI, and evaluated launch contract
are.

## Hand off to the target runtime

`loomc` produces and describes executable code. The selected runtime owns
device allocation, executable loading, argument and buffer binding, queue
submission, synchronization, and teardown.

One cached kernel version therefore retains:

1. the runtime's loaded executable and function token;
2. the loaded `loomc_vm_launch_config_program_t` and matching function token;
   and
3. any application metadata needed to identify the cached logical kernel.

Artifact and function ordinals are consumed while preparing that cached
version. They are not looked up again for each dispatch.

At issue time, the application evaluates workload arguments, serializes the
separate device launch arguments according to the selected ABI, binds device
resources, and submits the returned workgroup count with the executable entry.
It maps `workgroup_storage_bytes` according to the target adapter rather than
assuming it is an API's additional dynamic-shared-memory field.

The repository carries complete runtime handoffs for three integration
shapes:

- [raw HSA and AMDGPU HSACO](https://github.com/ROCm/hrx-system/blob/main/loom/binding/c/example/emit_amdgpu_hsa.c);
- [raw Vulkan and SPIR-V](https://github.com/ROCm/hrx-system/blob/main/loom/binding/c/example/emit_spirv_vulkan.c); and
- [IREE HAL and SPIR-V](https://github.com/ROCm/hrx-system/blob/main/loom/binding/c/example/emit_spirv_iree_hal.c).

Those examples are longer because they own real runtime discovery, allocation,
loading, and synchronization. That code is not compiler boilerplate and does
not belong in a target-independent `loomc` helper.

## Respect the lifetime and concurrency split

| Object | Sharing contract |
| --- | --- |
| `loomc_source_t` | Immutable and shareable after creation. |
| Target environment, context, profile, compiler, pass program, frozen link index | Prepared state shared across independent invocations according to each header's contract. |
| `loomc_workspace_t` | Mutable scratch owned by one active worker. |
| `loomc_request_t` | Immutable source-root selection shared independently of compiler workers. |
| `loomc_product_t` | Immutable successful result owning coupled artifacts and root projections. |
| `loomc_result_t` | Operation diagnostics; independent of the successful product lifetime. |
| `loomc_vm_launch_config_program_t` | Retained cached program; invocations on one handle do not overlap without external synchronization. |
| Runtime executable | Owned and synchronized by the target runtime integration. |

The status/result split is equally important. A non-OK `loomc_status_t` means
API misuse or an infrastructure failure prevented the operation from producing
a normal result. An OK status can still return a compiler result containing
structured diagnostics and a failed result state. Inspect both boundaries;
never treat “the API call returned” as “the source program compiled.”

The complete source for this chapter lives in the
[`jit-kernel` example](https://github.com/ROCm/hrx-system/tree/main/loom/docs/examples/integration/jit-kernel).
The generated [`loomc` C API reference](../reference/c-api/index.md) remains the
authoritative contract for every descriptor, handle, status, and artifact used
here.
