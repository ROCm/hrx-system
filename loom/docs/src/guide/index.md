# Programming guide

The programming guide explains Loom by the boundary that owns each decision.
It is written for systems and ML engineers who are comfortable reading a
kernel.

Start with the [source-to-artifacts walkthrough](../getting-started/source-to-artifacts.md)
if modules, templates, launch configuration, checks, and command programs are
all new. Use the chapters here to understand why a source construct exists and
when it is the right construct. Use the [generated
reference](../reference/index.md) when you need the exhaustive syntax or field
contract for one operation.

## The language progression

Loom source grows through a sequence of ownership boundaries. Later boundaries
compose the earlier ones rather than replacing them with a separate kernel,
graph, or deployment language.

1. **Source modules and canonical text** establish independently verifiable
   files, symbols, declarations, comments, bytecode, and deterministic source.
   Begin with [Source modules and canonical text](source-modules.md).
2. **Values, types, and shapes** distinguish logical coordinates, byte offsets,
   scalars, vectors, tensors, buffers, views, and physical encodings. The
   generated [type reference](../reference/types/index.md) is the exact catalog.
   Continue with [Values, types, and shapes](values-types-shapes.md).
3. **Functions and structured control flow** define typed host or device
   callables. Exact calls name one implementation;
   templates provide interchangeable implementations of a contract. Continue
   with [Functions and structured control flow](functions-and-control.md).
4. **Buffers, views, and memory** separate opaque storage identity from typed
   logical access, then make memory spaces, layouts, vector transfers, and
   synchronization explicit. Continue with [Buffers, views, and
   memory](buffers-views-memory.md).
5. **Vectors and structured compute** express lane-wise arithmetic, reductions,
   encoded numeric interpretation, and matrix contractions without committing
   reusable computation to an instruction set. Continue with [Vectors and
   structured compute](vectors-and-structured-compute.md).
6. **Kernels and launch configuration** add the dispatch boundary. A kernel
   owns both the mapping from workload to physical launch and the arguments
   carried into its device body. Continue with [Kernels and launch
   configuration](kernels-and-launch.md).
7. **Facts and specialization** state what the author, composition root, and
   target know. Configuration constraints, assumptions, target facts, and
   provider selection make that information useful before lowering. Continue
   with [Facts and specialization](facts-and-specialization.md).
8. **Checks and benchmarks** make correctness workloads executable and let
   performance rows select those same proven workloads instead of rebuilding
   them in a harness. Continue with [Checks and
   benchmarks](checks-and-benchmarks.md).
9. **Command programs** compose launchable kernels, resources, and scheduling
   into reusable subgraphs while preserving the same linking and specialization
   model. Continue with [Command programs](command-programs.md).

The generated [dialect reference](../reference/dialects/index.md) is organized
alphabetically because it is an inventory. The progression above is organized
semantically because it is a guide.

## Different pages answer different questions

| Need | Best starting point |
| --- | --- |
| Learn by building an operation | The ordered tutorials, where every lesson owns complete checked source, correctness, and a benchmark. |
| Understand a language decision | This programming guide. |
| Run a tool or diagnose a failure | A workflow page, with concise public commands before advanced evidence. |
| Package or embed Loom | Integration guides and the generated [`loomc` C API](../reference/c-api/index.md). |
| Set up or tune one architecture | Its target guide; reusable language chapters remain target-independent. |
| Look up exact syntax | The generated [language reference](../reference/index.md). |

## Reading source at library scale

A reusable motif is usually a `func.def` or `template.def`: it carries an
algorithm or physical representation contract but no launch ABI. A concrete
`kernel.def` composes motifs behind a workload and launch contract. A
`command.program.def` composes kernels behind a reusable scheduling and resource
contract.

This layering is also the library organization model. Format decoders,
reductions, tiles, and other motifs remain reusable across many kernels. Kernel
packages own concrete operation ABIs. Model packages own only the composition
that is genuinely model-specific. The optional
[`hrx-loom-kernels`](https://github.com/ROCm/hrx-loom-kernels) repository is the
growing standard-library corpus; the language and tool contracts remain here.
