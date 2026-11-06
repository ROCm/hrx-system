# Qualification model

The unit of qualification is a behavior under explicit preconditions. It
includes command fields, queue format and publication mode, memory placement
and cache policy, producer and consumer, topology, native transport, driver
and relevant firmware. An opcode or compiler target name alone is too broad.

## Evidence chain

| Evidence | What it establishes |
| --- | --- |
| Public specification | The documented semantics for the stated revision and execution model. |
| Pinned implementation | What a specific consumer emits, including its version predicates and workarounds. |
| Independent corroboration | Agreement with another implementation or normative definition; shared generated headers are one source. |
| Host encoding case | Exact packet bytes, units, layout, reserved constants and legal operand partitions. |
| Native operation case | Independently observed behavior on one recorded deployment. |
| Composition case | The release, ordering, acquire, observation and lifetime edges of a real caller. |

Each case record carries the relevant source revision/path/symbol, field
meanings and constraints, architecture and firmware predicates, ownership
requirements and unresolved evidence boundaries. Conflicting source behavior
stays visible until the controlling predicate is established. Runtime policy
limits and hardware representation limits occupy separate fields.

## Observing the behavior

`GpuCommandTest` borrows the corpus's cached device and owns each case's
allocations, host mappings and queues. It performs no payload cache maintenance
and emits no synchronization packets. Engine cases choose those operations.
Family selection matches required packet fields, semantic cache operations
and transition domains separately before activation.
Case cleanup releases producer mappings and successfully destroys all queues
before releasing reachable memory; failure preserves the remaining resources.

User-published queue completion uses coherent system memory and naturally
aligned x86-64 loads/stores. Command sequences establish execution completion
before the host observes the completion word. AQL polls the signal value after
the command processor decrements it; PM4 and SDMA emit explicit completion
writes. Read-index progress is checked separately before queue teardown.

Kernel-published recipes retire the exact native submission point. Cases with
an in-band completion marker observe payload before that native wait; cases
without one use the native wait for completion as well. Their result records
identify the publication mode. No payload maintenance is hidden in either
transport helper.

Completion occupies bytes disjoint from the copied or shader-produced payload.
Host polling performs no cache maintenance. After observing completion, a
composition case snapshots the final consumer's output before other payload
diagnostics, consumed-frontier polling or queue destruction. This prevents a
native progress operation, cleanup flush or intermediate host wait from
supplying the edge under test.
The conservative PM4 system barrier used by the composition cases is a sufficient
recipe, not a measurement of the least expensive possible release or acquire.

An exact data oracle and changing input expose stale results. Guard words
expose transfer-length mistakes. Ring reuse follows consumption; signal,
descriptor, code and payload reuse follow their respective final users.
Coherence does not create an execution dependency, and a host wake is not a
retirement certificate.

### Portable conditional-branch operands

The PM4 conditional graph cases require `reference & ~mask == 0`. That
precondition makes unsigned comparisons equivalent whether an engine masks
only the memory operand or both memory and reference. The command writer
preserves the supplied fields; it does not silently change the predicate.
Every comparison function uses full, partial and zero masks, with changing
ignored memory bits and a pre-masked reference.

Native observations on tested gfx1100 and gfx1151 deployments distinguish
the two mask behaviors when the reference has bits outside the mask. Such
operands are outside this portable recipe. Neither a passing case under the
precondition nor a compiler target name establishes arbitrary-reference
semantics for another driver/firmware pairing.

## Required witnesses and deployment identity

An ordinary discovery invocation can skip an unavailable native service. The
passive enumeration example prints endpoint IDs alongside native identities
without activating a device:

```sh
build_tools/bin/iree-bazel-run --//libamdf/config:enabled=true \
  //libamdf/examples:enumerate
```

Set `GPU_ENDPOINT_ID` to the selected endpoint's two hexadecimal words, including
the colon, and `GPU_TARGET` to its reported GFX target. A qualification invocation
names that endpoint, its expected target and the required cases explicitly:

```sh
build_tools/bin/iree-bazel-test --config=asan \
  --//libamdf/config:enabled=true \
  //libamdf/cts/gpu/recipes:recipes_dynamic \
  --test_arg="--amdf_gpu_endpoint_id=${GPU_ENDPOINT_ID}" \
  --test_arg="--amdf_gpu_target=${GPU_TARGET}" \
  --test_arg=--amdf_require_test=MemoryPairRecipeTest.ConcreteCoherentHostAndTwoQueueHandoff \
  --test_arg=--amdf_require_test=MemoryPairRecipeTest.ProfileCoherentHostAndTwoQueueHandoff
```

Repeated `--amdf_gpu_target` arguments admit any listed target, allowing one
qualification invocation to run on heterogeneous members of a GPU family. CI
expands build-family selectors through the shared target catalog before passing
these exact targets to CTS.

Both selectors run before native activation. An explicit endpoint absent from
GPU discovery fails the case; it never selects another device. A present
endpoint that lacks the case's target or native service remains a skip, which
fails qualification when that case is required. A required case that is
absent, filtered, skipped, unexecuted or failed makes the process fail. Repeated
requirements apply within that executable; different corpora have different
required-case lists. GoogleTest XML records the selected compiler target, ASIC
revision and XCC count, together with the opaque endpoint ID and its native
correlation identity. Linux records the character-device major/minor; Windows
records the adapter LUID and physical-adapter index. The exact endpoint selector
distinguishes devices with the same compiler target. The recorded native
identity connects preflight and resource reservation to the selected endpoint
and the contemporaneous deployment inventory; neither identity is a persistent
identifier across reboot or device replacement. A `none` native identity leaves
that correlation unestablished.

A complete record contains library/source revision, binary identity, compiler
and build flags, case names and parameters, provider and native lifetime,
physical device/GC/SDMA identities, relevant firmware, OS/kernel/driver, memory
profiles and cache classes, raw result/XML, cleanup result and observed driver
health. Facts absent from the public API come from the deployment inventory;
they cannot silently become predicates evaluated by a runtime client.
The loaded AMDGPU module identity is separate from the running kernel release:
an independently installed module need not share that kernel's source lineage.
HBM results also record `mtype_local`, compute/memory partition configuration
and XCC count. A pinned reference source is not a substitute for the deployed
module's build identity.

A catalog entry or newly compiled case establishes source membership. Native
coverage remains attached to the exact required-case list and frozen source
and artifact identities of each execution; it does not move with the branch.

Performance records use optimized, uninstrumented builds and report submit-only
and end-to-end measurements separately. Correctness tests use explicit valid-work
completion and the outer test harness's hang detection; elapsed test duration
is not a command-latency benchmark.

[Timing and performance counters](../../../docs/reference/amd/gpu/observability.md)
are independently qualified observation surfaces. Timestamp ordering and
visibility, clock conversion, counter ownership, reset/read/stop semantics and
instrumentation effects each need evidence before their measurements can
describe a HAL workload.
