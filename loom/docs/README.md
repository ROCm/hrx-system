# Loom documentation

`loom/docs/` owns the Loom programming guide and the inputs used to assemble
its published reference. It is documentation build infrastructure, not a
shipped Loom command-line surface.

Add the optional, pinned documentation toolchain to the repository environment:

```bash
python dev.py setup --docs
```

This installs the hash-locked Python requirements and the pinned Bazelisk and
Doxygen binaries into `.venv`; it does not depend on distribution packages.
The compiler-backed guide examples use Bazelisk to build the exact Loom tools
from the current checkout before documentation is assembled.

Build the strict static site:

```bash
python dev.py docs build
```

The result is written to `build/loom-docs/site/`. Serve it with live reload
while editing:

```bash
python dev.py docs serve
```

Hand-authored pages live under `src/`. Executable guide programs live under
`examples/` so the compiler and execution-test infrastructure can validate the
same source readers see. Generated dialect reference pages and C API output are
assembled only in the build tree; generated Markdown and HTML are not checked
in. Set `LOOM_DOCS_WORK_DIR` to relocate that isolated build tree when needed.

## Content architecture

Each public page belongs to one reader journey. Keeping those journeys separate
is what lets the guide grow without turning one authoring page into a language
manual, tool reference, target cookbook, and fixture inventory at once.

| Lane | Reader question | Content contract |
| --- | --- | --- |
| Start | What is Loom, how do I acquire it, and what is its programming model? | A short route from installation to the first complete program. |
| Tutorials | How do I build increasingly substantial workloads? | An ordered sequence of complete operations, each adding a bounded set of concepts. |
| Programming guide | What does this language construct mean, and which boundary owns it? | Concept-oriented chapters organized from modules and values through kernels, specialization, checks, and command programs. |
| Workflows | How do I format, test, benchmark, compile, inspect, or debug a program? | Task-oriented public tool invocations with their input, output, and failure contracts. |
| Integration | How do I package libraries or embed AOT/JIT compilation? | Real caller flows for bytecode libraries, the `loomc` API, launch evaluation, artifacts, and caches. |
| Targets | What changes for this architecture or runtime? | Target-owned setup, profiles, capabilities, execution paths, and evidence interpretation. |
| Reference | What is the exact syntax or API contract? | Generated dialect/type/attribute pages and declaration-owned C API documentation. |
| Contributing | How do I build Loom or change its documentation? | Repository tooling and compiler-development workflows kept out of the installed-user path. |

The public navigation grows with complete content rather than publishing empty
sections. The lane assignment still applies before a page is linked into the
site: target-specific commands do not migrate into a language chapter merely
because its target page has not been written yet.

## Page contracts

Tutorials follow one stable lesson shape:

1. The operation and the concepts the reader will learn.
2. One complete source file included from `examples/`.
3. The algorithm and its Loom representation.
4. Executable correctness cases and the public test command.
5. Benchmark rows and the public benchmark command.
6. Compiler evidence only when that evidence is the lesson.
7. A concise recap and the next operation in the progression.

Programming-guide chapters explain one ownership boundary, use small excerpts
from checked source, give decision guidance and characteristic failure modes,
and link to generated reference pages for exhaustive syntax. They do not repeat
operation inventories.

Workflow pages begin with the concise installed-tool path. Advanced reports,
intermediate IR, native output, and structured data appear in the workflow that
interprets them rather than becoming mandatory flags in every example.

Target pages own platform prerequisites and target-specific source or compiler
evidence. Reusable tutorial and guide source remains targetless until an
algorithm genuinely requires a target fact.

## Executable examples

Each example package owns one `run.sh` that accepts a canonical target name and
prints every public command before executing it. The script is both the reader's
lesson and the CI entry point. Shared shell code owns only target setup and
common command presentation.

A package-local `.doc-generate.sh` reuses `run.sh`, validates the products needed
by its pages, and stages generated snippets under the documentation build tree.
Generated artifacts never become checked-in source, and documentation builds do
not reach into repository-private test fixture formats to manufacture a public
workflow.
