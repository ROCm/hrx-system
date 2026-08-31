# Integrate Loom

Loom can stop at a checked source library, a root-selected bytecode
module, a target-native executable, or an in-process JIT. The application
chooses that boundary according to when it learns the model, shape,
configuration, and physical device facts that matter.

## Choose where specialization happens

| Integration boundary | Facts available | Application responsibility |
| --- | --- | --- |
| Ahead-of-time artifact build | Release configuration and a generic or exact target profile. | Package the executable and load it through the matching runtime. |
| Model-load JIT | Model configuration, linked kernel catalogs, and the selected device. | Compile profitable roots, load the artifacts, and retain them with the model plan. |
| Invocation-specialized JIT | Exact workload facts not known at model load. | Select or compile a cached version without putting compilation on every unchanged invocation. |
| Command-program materialization | Kernel versions, resource requirements, and reusable subgraph structure. | Bind replaceable resources and issue the prepared schedule through its runtime adapter. |

These are deployment policies over the same modules. A `.loombc` catalog does
not commit to offline compilation, and a JIT does not require generated source
or temporary files. Moving the boundary changes when facts become exact; it
does not introduce another kernel language or library representation.

## Keep ownership visible

A native embedding normally has four lifetime layers:

| Lifetime | Typical state |
| --- | --- |
| Process or compiler service | Target environments, contexts, frozen link indexes, compilers, and prepared pass programs. |
| Worker | One mutable workspace and the invocation-local modules being linked, compiled, or emitted. |
| Cached version | Executable bytes, a loaded launch-config companion, public export bindings, and any application cache metadata. |
| Issue | Workload scalar bits, launch arguments, buffer bindings, and synchronization edges. |

That split is the reason an embedding can specialize many independent kernels
in parallel without rebuilding compiler setup or sharing mutable scratch.
Immutable prepared objects are reused; workspaces and modules belong to one
active invocation; runtime artifacts outlive the compiler operation that
created them only after the application loads or copies them.

## Integration guides

- [Compose modules in memory](module-composition.md) shows both declaration-only
  and logical-provider composition over caller-owned source bytes and one
  reusable frozen link index.
- [Embed command and kernel product frontiers](product-frontier.md) shows how a
  portable command product publishes independently compilable kernel requests
  while preserving external executable requirements.
- [Schedule concurrent JIT work](jit-task-pool.md) shows how generic tasks,
  cache-before-submit policy, worker-local workspaces, and independent queues
  on one optional shared pool compose without making scheduling part of the
  compiler API.
- [Embed kernel JIT compilation](jit-kernel.md) follows source through target
  specialization, launch evaluation, native artifact emission, and the runtime
  handoff using the public `loomc` API.
- [Link and package modules](../workflows/link-and-package.md) covers the
  installed-tool workflow for reusable `.loombc` catalogs and selected roots.
- [Compile artifacts](../workflows/compile-artifacts.md) covers ahead-of-time
  executable, manifest, and report products.
- The generated [`loomc` reference](../reference/c-api/index.md) defines the
  exact ownership, lifetime, threading, and failure contracts for every API.
