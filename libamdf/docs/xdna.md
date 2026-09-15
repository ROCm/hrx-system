# XDNA native execution

libamdf supplies device admission, scoped memory, addresses, and native queue
submission. A HAL supplies instruction bytes, ELF loading and relocation, tile
programs, and scheduling. ARRAY and CONTROL are image-layer concepts; neither
appears in the driver API.

## One caller flow

The caller passively discovers an endpoint, checks its instruction and context
capabilities, then explicitly creates a device and context. Ordinary data comes
from the instance's system-memory scope with that live device in its access set.
Instruction storage comes from the context's private scope with EXECUTE access.
Both use `memory_create`, explicit host mappings, and cached address queries.

The HAL writes and publishes its instruction ranges once. Each native queue
submission identifies `{memory, access_ordinal, byte_offset, byte_length}`.
libamdf borrows that range through retirement and supplies only the mandatory
native transport packet. It neither copies the instruction stream nor patches
application arguments. A cold initialization range and a reusable execution
range can occupy the same allocation without additional driver objects.

Ordinary data buffers have no per-submission BO list. Allocation establishes
their native mappings and residency. The caller maintains visibility, ordering,
and lifetime, including references followed by device-side streaming after a
kernel submission retires. Completion does not discover those references.

For CPU/NPU interchange, the caller queries `memory_query_pair_info` with its
concrete host mapping and device access plus queue-family ordinal. The result
selects the host publication or invalidation operation; XDNA DMA requires no
additional device cache transition. The caller publishes inputs before use and
acquires outputs after the program finishes the relevant DMA and its ordering
edge completes. The query itself neither flushes caches nor orders execution.

## Native requirements

| Boundary | Linux modern DRM | Windows MCDM |
| --- | --- | --- |
| Context admission | Native hardware context and negotiated execution support. | Driver-specific admission metadata; wire ABI is resolved independently of PCI identity. |
| Instruction storage | Context-qualified DEV backing, with sizes and alignment from its scope. | One 64 MiB native aperture per context; a reserved 32 KiB bootstrap prefix is excluded from the caller's usable range. |
| Instruction submission | Mandatory DRM execution record and command BO referencing the caller's instruction range. | Mandatory native transport record and transaction-interpreter packet referencing that range. |
| Ordinary data addresses | Firmware and shim-DMA address interpretations. | Firmware and shim-DMA address interpretations for standard system backing. |
| Private addresses | The scope reports available firmware and DMA interpretations. | Firmware address returned by native allocation; a private shim-DMA interpretation is not advertised. |

Profiles distinguish backing granularity from address alignment. In particular,
Windows standard backing is rounded to 64 KiB, while KMT mappings guarantee
4 KiB address alignment. Native mapping bounds account for the target's shim-DMA
translation so the complete range fits every advertised address interpretation.

The Windows metadata ABI qualified for KMD 32.0.203.240 uses a 272-byte context
record and an 88-byte submission prefix. Context creation needs no application
xclbin or kernel-description payload. The separately tagged older ABI retains
its matching compatibility envelope and 104-byte submission prefix. A successful
private query which writes no response is not an ABI tag; unrecognized schemas
are rejected before context construction. Native context ID zero is valid.

Windows initialization loads a target-selected native bootstrap independently
of application code. The NPU4 bootstrap only asserts four core resets; it does
not install application DMA routes or a schedule. The native bootstrap UUID
identifies that setup, not the caller's executable. This mandatory provider
setup is distinct from any application PDI a HAL may construct. No public PDI,
program, lane, or argument-patching object is required.

## Submitting a prepared range

The queue and instruction memory below come from the same context. The caller
has already queried alignment and size limits, obtained EXECUTE access, written
its target-native bytes, and performed the required host publication. The
descriptor is consumed during the call; accepted instruction bytes remain
immutable until retirement.

```c
#include "amdf/xdna.h"

amdf_status_t publish_instructions(
    const amdf_xdna_api_t* xdna, amdf_kernel_queue_t* queue,
    amdf_memory_t* instructions, uint32_t access_ordinal,
    uint64_t byte_offset, uint64_t byte_length, uint64_t* out_submission) {
  const amdf_xdna_kernel_command_t command = {
      .memory = instructions,
      .access_ordinal = access_ordinal,
      .byte_offset = byte_offset,
      .byte_length = byte_length,
  };
  const amdf_xdna_kernel_queue_submission_info_t submit = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO,
      .structure_size = sizeof(submit),
      .command_count = 1,
      .commands = &command,
  };
  return xdna->kernel_queue_submit(queue, &submit, out_submission);
}
```

NPU4 and NPU5 queues admit one instruction range per submission and one
unretired submission per queue. The publication call performs no allocation,
instruction parsing, relocation, argument resolution, native submission retry,
sleep or host wait.
The queue preallocates its mandatory native packet storage. Multiple contexts
can independently own backing for different resident programs or queues.

The returned submission number identifies accepted work. A caller can observe
progress with `kernel_queue_query_status` or wait with
`kernel_queue_wait(queue, submission, AMDF_TIMEOUT_INFINITE, 0)`. A successful
wait establishes native retirement, including command-result inspection. A
timeout or wait error is not cancellation and does not by itself release the
instruction borrow; the status query reports retirement separately from sticky
terminal failure.

The [canonical ELF consumer](../../experimental/xdna/cts/execution_test.cc)
shows the complete flow, including target selection, image loading, relocation,
cold initialization, reusable execution, numerical checks and teardown. The
ELF decoder and materializer live in the runtime image layer; libamdf receives
only the prepared native range. Reusing that range does not repeat image
loading or require an indirect data-buffer list.

## Ownership

A queue borrows its context. Private memory also borrows its context; ordinary
memory borrows its explicitly requested devices. The HAL releases mappings and
memory after final use and before their owners. There is no hidden retention,
allocation registry, or library suballocator. Several live contexts can own
independent private backing even when their firmware addresses are numerically
equal.

Hardware facts and native ABI facts have separate owners. Endpoint profiles hold
topology, limits, instruction format, and address translations. The platform
provider qualifies its native services. A known hardware identity does not make
an unknown driver schema compatible, nor does one native firmware command imply
support for another. Capability checks describe the implemented combination.

The [memory fabric](memory.md) describes the shared scope, address, visibility,
and lifetime contracts used by GPU and XDNA callers.
