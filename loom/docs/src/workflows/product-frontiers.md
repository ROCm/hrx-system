# Compose recursive product frontiers

A product frontier closes one parent artifact while leaving each child
implementation as an explicit requirement. This lets one command program become
portable command bytes without embedding every kernel body, and it lets each
required kernel compile or resolve from a cache independently.

The frontier is a normal selective-link boundary. It does not introduce a fact
sidecar, generated source format, or graph-specific package. Parent and child
products remain ordinary Loom modules and artifacts, so the same formatter,
linker, compiler, diagnostics, and cache keys apply at every level.

## Start from requester ownership

Positional inputs are the requester. Their selected exports state what the
output deliberately publishes. Libraries contribute definitions and template
implementations that may satisfy that requester, but a public definition in a
library is not automatically re-exported. When selected only as a dependency,
it becomes private in the linked product.

This distinction lets a reusable catalog advertise many public definitions
without leaking them through every application that uses one. Select a library
definition as its own root when it should also be an output; supplying the
library is only permission to resolve dependencies from it.

Structural facets make the same rule useful inside a symbol. A
[`kernel.def`](../guide/kernels-and-launch.md) provides three independently
selectable parts:

| Facet | Parent use |
| --- | --- |
| Contract | Names workload and device arguments and establishes the kernel identity. |
| Configuration | Pure computation mapping workload values to physical launch geometry. |
| Implementation | Device computation compiled for the execution target. |

A `kernel.launch` needs the contract and configuration because it starts from a
logical workload. A `kernel.dispatch` already has a physical workgroup count,
so it needs only a configured entry contract. That is why an authored
`kernel.entry.decl` can represent a HIP, SPIR-V, precompiled, or otherwise
host-provided executable entry without pretending Loom has its source.

Command-product construction follows command implementations and the contract
and configuration facets of launched kernels. It stops before kernel
implementation facets. The parent can therefore finish without reading device
bodies, while an optional child-product operation opens only the implementations
needed by surviving launch sites.

## Run a heterogeneous product workflow

The checked example combines all three useful provider shapes. The requester
owns a local kernel, launches a command program supplied by another source
module, and directly dispatches an external entry:

```loom title="requester.loom"
--8<-- "examples/product-frontiers/requester.loom"
```

The linked command library declares the logical kernel it needs without owning
that kernel's source:

```loom title="stage.loom"
--8<-- "examples/product-frontiers/stage.loom"
```

The provider library supplies the kernel and two schedule alternatives. Its
bytecode representation is deliberately mixed with the other text inputs in
the workflow:

```loom title="providers.loom"
--8<-- "examples/product-frontiers/providers.loom"
```

Run the exact formatting, bytecode packaging, selective linking, command
compilation, request decoding, and result checks with:

```shell
loom/docs/examples/product-frontiers/run.sh build/product-frontiers
```

The result has one portable command artifact and four executable-entry
requirements:

| Entry requirement | Surviving distinction | Child source product |
| --- | --- | --- |
| `@local` | One local launch class | `kernel-0.loombc` |
| `@classified` | `%size = 64` selects `@small` | `kernel-1.loombc` |
| `@classified` | `%size = 256` selects `@large` | `kernel-2.loombc` |
| `@external` | Direct physical dispatch | None; the host binds the entry. |

The two `@classified` requests come from one kernel definition. They are split
because the launch-site facts select different generated programs, not merely
because the argument tuples differ. Each request is ordinary rooted Loom IR:
the selected template call and the assumptions that justify it travel with the
module. A target compiler can consume it directly, and a content-addressed host
can cache it at the same boundary as any other Loom input.

The external dispatch remains equally real in the parent artifact. Its missing
source is not a degraded request or a failed lookup; it is a binding requirement
whose implementation ownership was explicitly outside this source universe.

## Recurse without coupling products

The same ownership rule composes through deeper systems:

```text
host or VM product
  -> command-program requirement
       -> portable command product
            -> kernel-entry requirement
                 -> kernel source request or external executable binding
```

A host product may stop at command contracts, a command product stops at kernel
implementations, and a kernel product may stop at target ABI imports. At each
frontier, the parent records what must be bound and the child remains an
independent compilation or cache lookup. The checked workflow currently
exercises the command-to-kernel frontier; the hierarchy does not require a
different request representation when a host-to-command producer is added.

Specialization happens before a frontier publishes child work. Dead command
branches create no kernel requests, and repeated launch sites share a request
when their decision traces select the same semantic class. The parent therefore
publishes the smallest known set of child products without compiling candidates
merely to discover afterward that their binaries match.

## Preserve ownership and concurrency

The native `loomc` path exposes these lifetimes directly:

| Object | Ownership and sharing |
| --- | --- |
| Frozen link index | Retains indexed source or module providers and may be shared across workers. |
| Workspace | Mutable invocation scratch owned by one active worker. |
| Command product | Owns serialized root bytes, copied names, entry requirements, and resource projections; it retains no source module, link index, compiler plan, or analysis state. |
| Kernel request module | Transfers independently to the request callback and follows ordinary module ownership. |

Omitting the request callback is the body-blind path: command construction
leaves every kernel implementation unopened and returns only executable-entry
requirements. Supplying it classifies surviving source-backed launch sites and
publishes at most one child module for each live semantic class. Publication is
provisional until the parent operation returns an OK status with a succeeded
result; an embedding can stage accepted children concurrently and commit them
to its cache only when the parent succeeds.

The returned command product never becomes a lifetime root for child modules.
That independence is what permits target compilation, remote lookup, and
runtime executable loading to overlap parent construction without retaining a
large compiler graph.

## Keep failures at the frontier that owns them

The boundaries distinguish unsupported source from deliberately external work:

| Failure | Owning contract |
| --- | --- |
| A reachable `kernel.launch` has no configuration facet | The logical launch cannot derive physical counts; provide the defining kernel or repair its declaration/library dependency. |
| A reachable exact declaration has no provider in a closed link | The source universe is incomplete; add its declared library or deliberately emit a partial link. |
| A `kernel.dispatch` targets a bodyless `kernel.entry.decl` | Valid external binding; the parent records the requirement without asking for source. |
| A request callback or artifact write fails | Infrastructure failure cancels the parent operation; already transferred request modules remain owned by their recipients. |
| Authored facts, signatures, or providers conflict | The operation returns a normal failed compiler result with diagnostics at the source contract. |

Use [`loom-link --print-plan`](link-and-package.md#inspect-before-linking) to
inspect requester roots and provider selection before materialization. Use
[`loom-compile --backend=command`](compile-artifacts.md#emit-portable-command-programs)
to emit the parent artifacts and optional child requests from an already linked
catalog.
