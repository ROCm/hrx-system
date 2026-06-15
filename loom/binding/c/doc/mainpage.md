# Loom C API

The `loomc` library is the public C ABI for embedding the Loom compiler in
native language drivers, JITs, autotuners, packaging tools, and runtime
executable caches. The API is shaped around in-memory sources, reusable
compiler/linker/index handles, caller-owned workspaces, structured diagnostics,
and in-memory artifacts.

Core `loomc` headers expose C types defined by the Loom API. Optional target
and runtime integration headers adapt to host ecosystems such as HSA, Vulkan,
and IREE HAL without making those ecosystems part of the core ABI.

## Package And Header Map

The narrow embedding shape is `#include <loomc/loomc.h>` plus the target
package that produces the artifact your runtime loads. The core package keeps
source, link, compile, emit, result, diagnostic, and artifact ownership
independent from any GPU runtime.

| Package or header | Provides | External headers |
| --- | --- | --- |
| `//loom/binding/c:loomc`, `loomc/loomc.h` | Core source, module, linker, compiler, target, result, diagnostic, and artifact APIs. | None beyond the C/C++ runtime headers used by the public ABI. |
| `loomc/iree.h` | Header-only adapters between Loom status/allocator/string types and IREE base types. | IREE base headers. |
| `//loom/binding/c/target/amdgpu`, `loomc/target/amdgpu.h` | AMDGPU target environment, processor profiles, and HSACO emission. | No HSA, HIP, ROCm, or IREE HAL headers. |
| `//loom/binding/c/target/spirv`, `loomc/target/spirv.h` | SPIR-V target environment, profiles, and SPIR-V emission. | No Vulkan or IREE HAL headers. |
| `//loom/binding/c/target/spirv/vulkaninfo`, `loomc/target/spirv/vulkaninfo.h` | Saved Vulkan/GPUInfo profile import for offline or cached SPIR-V targeting. | No Vulkan loader or Vulkan SDK headers. |
| `//loom/binding/c/target/spirv/vulkan`, `loomc/target/spirv/vulkan.h` | Raw Vulkan device/profile adapter for SPIR-V targeting. | Vulkan headers; the caller owns the loader, physical device, and function table. |
| `//loom/binding/c/target/iree_hal`, `loomc/target/iree_hal.h` | Runtime target routing from IREE HAL devices into Loom target profiles. | IREE HAL headers. |
| `//loom/binding/c/target/spirv/iree_hal`, `loomc/target/spirv/iree_hal.h` | SPIR-V targeting through IREE HAL Vulkan device information. | IREE HAL headers. |
| `//loom/binding/c:loomc_full` | Convenience package for the current core, SPIR-V, Vulkan, VulkanInfo, and IREE HAL binding slice. | The union of the headers above for that slice. |

`loomc_full` is a packaging convenience for environments that already accept
the broader dependency surface. Tight runtime integrations usually link
`//loom/binding/c:loomc` and the one target package they need. The AMDGPU path
is intentionally a separate package today so raw HSA/HRX-style integrations can
embed Loom without pulling in Vulkan or IREE HAL.

Installed headers keep the same component shape under `loomc/`. Static and
dynamic library packaging should preserve that boundary: link the core Loom C
API plus selected target packages for dependency-light embeddings, or link the
full convenience package when the application already carries the SPIR-V,
Vulkan, and IREE HAL dependency set. Runtime loader libraries remain an
application choice; raw HSA and raw Vulkan examples load/query those runtimes
outside of `loomc`.

## In-Memory Pipeline

Loom's embedding pipeline is filesystem-optional. File constructors and artifact
write helpers exist for command line tools, caches, and release packaging, but a
JIT can keep every important edge in process memory:

1. Create immutable `loomc_source_t` handles from `.loom` text, `.loombc`
   bytes, generated source text, or externally owned buffers.
2. Parse, index, or link those sources into `loomc_module_t` handles. A frozen
   link index can be shared across many specializations of the same library.
3. Reuse prepared pass programs, linkers, compilers, and target profiles across
   invocations.
4. Use a caller-owned `loomc_workspace_t` as per-worker scratch while compiling
   or emitting.
5. Inspect the returned `loomc_result_t` for state, diagnostics, reports, and
   artifacts.
6. Load executable artifact bytes directly into the host runtime, or copy/write
   them into the application's cache.

That shape supports both offline and online specialization. An application can
pre-link authored libraries into `.loombc`, then at runtime create a tiny
generated source containing a `kernel.def`, constants, or dispatch parameters,
link it against the frozen library index, and emit a target artifact without
round-tripping through temporary files.

## Contract Summary

- `loomc_status_t` reports API and infrastructure failures that prevent an
  operation result from being produced.
- `loomc_result_t` and operation-specific result handles report compiler,
  linker, configuration, and source-program outcomes through state,
  diagnostics, reports, and artifacts.
- `loomc_source_t` is an immutable source handle with explicit borrowed, copied,
  or externally owned storage.
- `loomc_workspace_t` is mutable scratch for one worker at a time and is reset
  between operations or phases.
- Prepared pass programs, compiler, linker, and frozen link-index handles are
  intended to be immutable and reusable across many invocations.

## Binding Ownership Model

The C API uses explicit reference ownership so C++, Rust, Python, and other
bindings can present their native lifetime model without changing the compiler
contract.

- Retained handle types map naturally to RAII destructors, Rust `Drop`, Python
  finalizers, or context managers. A binding that moves a handle into another
  owner transfers the retained reference; a binding that shares the handle with
  another owner calls the matching retain function when the type provides one.
- `loomc_status_t` is not a retained object, but every non-OK status is
  consumed exactly once by `loomc_status_free`, `loomc_status_consume_code`, or
  `loomc_status_join`. Bindings normally convert status failures into native
  infrastructure exceptions or error values before exposing any operation
  result.
- `loomc_result_t` is the lifetime root for diagnostics and artifacts returned
  by an operation. Borrowed diagnostic and artifact pointers remain valid until
  the result is released. Bindings can either keep the parent result alive for
  child views or copy diagnostics/artifact bytes into language-owned values.
- `loomc_string_view_t` and `loomc_byte_span_t` are borrowed byte ranges, not
  NUL-terminated strings. Bindings copy them when a language object must outlive
  the source, result, artifact, or caller buffer that owns the bytes.
- `loomc_source_t` handles are immutable and may borrow, copy, or externally
  release source bytes. Borrowed source contents must outlive the source handle;
  copied source contents are owned by the source.
- `loomc_workspace_t` is mutable scratch for one worker at a time. Reusing a
  workspace avoids per-invocation allocation churn, while immutable sources,
  results, frozen indexes, pass programs, compilers, and target profiles can be
  shared across workers according to their header contracts.

## Example Index

The examples under `loom/binding/c/example/` are intentionally small end-to-end
embedding programs:

- `compile_text.c` compiles in-memory Loom text and consumes diagnostics.
- `link_modules.c` links multiple sources and turns module artifacts back into
  source handles.
- `emit_amdgpu_offline.c` emits AMDGPU artifacts without a runtime loader.
- `emit_amdgpu_hsa.c` queries HSA directly and launches a Loom-produced HSACO
  without IREE HAL. It can start from its embedded source or from a `.loom` or
  `.loombc` path supplied on the command line.
- `emit_spirv_offline.c` emits SPIR-V artifacts without a runtime loader.
- `emit_spirv_vulkan.c` queries Vulkan directly and launches a
  Loom-produced SPIR-V module without IREE HAL.
- `emit_spirv_iree_hal.c` shows the optional HAL-specific SPIR-V route when an
  application already has an IREE HAL device.

The public headers are the primary API documentation. Contributor conventions
for those comments live in `loom/binding/c/doc/STYLE_GUIDE.md`.
