# Authoring and Linking

This directory is a checked miniature of the library flow used by larger model
ports. The root workload lives in one file, reusable implementations live in a
provider library, and the command line tools decide whether to link text,
package bytecode, preserve checks, strip checks, or compile a selected target
artifact.

Command blocks use short tool names and paths relative to this directory. From
the repository root, run the same tools through `dev.py` and pass root-relative
paths:

```bash
python dev.py bazel run //loom/src/loom/tools/loom-link:loom-link -- \
  loom/src/loom/test/corpus/authoring/linking/root.loom \
  --library=loom/src/loom/test/corpus/authoring/linking/providers.loom \
  --root=@scale_i32_buffer \
  --print-plan
```

## Files

| File | Role |
| --- | --- |
| `root.loom` | Targetless public kernel for artifact packaging. The kernel asks for `authoring.link.scale_i32` with `template.apply`. |
| `checks.loom` | Correctness case and benchmark rows for the root workload. It declares the callable surface so checks can be linked during authoring without being part of the artifact root file. |
| `providers.loom` | Reusable provider library with AMDGPU target records, gfx-specific `template.def` implementations, a generic fallback, and an unused provider. |
| `linking.test.json` | Production CLI proof for source linking, bytecode-library linking, and AMDGPU artifact compilation from linked bytecode. |

The kernel deliberately has no authored `target(@...)`. It is portable source
until a compile or JIT invocation specializes it. Both the root and provider
modules carry the same `template.decl`; the linker merges that family contract
and never uses a provider path as a matching rule. The provider library carries
`amdgpu.target<gfx1100> @gfx1100` and `amdgpu.target<gfx1200> @gfx1200` records
so target-specialized definitions can state their exact applicability.

A closed link uses the facts available at that link boundary. With
this targetless root, it chooses `@scale_i32_fallback` and omits the target
records and target-specific alternatives. A merge of the root and provider
modules preserves the full explicit universe. Compiling that module with
`--target=gfx1100`
then specializes the requested kernel to a materialized or reused `@gfx1100`
target record and resolves `template.apply<@authoring.link.scale_i32>` against
that durable target.

## Inspecting a Library

List the indexed symbols before deciding what to link:

```bash
loom-link root.loom --library=providers.loom --list-symbols
```

Print the link plan for one root:

```bash
loom-link root.loom --library=providers.loom \
  --root=@scale_i32_buffer \
  --print-plan
```

The plan keeps the root kernel, its family declaration, and the proven fallback
selected for the targetless application. It does not retain the target-specific
definitions, their target records, or the unrelated provider.

During correctness work, make the check case the root and provide the kernel
and implementation library as libraries:

```bash
loom-link checks.loom \
  --library=root.loom \
  --library=providers.loom \
  --root=@scale_i32_buffer_case \
  --print-plan
```

This keeps test/benchmark ownership separate from artifact packaging. The check
module can stay rich while the artifact root module remains easy to package,
strip, and compile.

## Linking Text

Link source files directly while developing:

```bash
loom-link root.loom --library=providers.loom \
  --root=@scale_i32_buffer \
  --output=linked.loom
```

Add `--strip-check` when producing a package for artifact compilation or a JIT
cache entry that should not carry `check.case` and `check.benchmark` symbols:

```bash
loom-link root.loom --library=providers.loom \
  --root=@scale_i32_buffer \
  --strip-check \
  --output=linked.loom
```

Keep checks when the linked module is still an authoring/test artifact. Strip
checks when the linked module is a deployment artifact or when the embedding
program owns correctness and benchmark orchestration.

## Packaging Bytecode

Provider libraries can be packaged once as `.loombc` and reused by many root
modules:

```bash
loom-format --from=text --to=bc \
  --output=providers.loombc \
  providers.loom
```

The linker accepts text and bytecode inputs together:

```bash
loom-link root.loom \
  --library=providers.loombc \
  --root=@scale_i32_buffer \
  --to=bc \
  --output=linked.loombc
```

That link is closed for the facts currently available and therefore
contains the portable fallback. It is the shape an embedding API mirrors when
the current boundary has enough information to choose implementations: add
root sources, add prebuilt `.loombc` libraries, name the roots, and link to text
or bytecode depending on the next stage.

When target facts arrive only at compilation, merge the explicit input universe
instead of prematurely selecting a targetless fallback. Both modules are
positional because both should become part of the output:

```bash
loom-link root.loom providers.loombc \
  --mode=merge \
  --strip-check \
  --to=bc \
  --output=portable.loombc
```

## Compiling an AMDGPU Artifact

Compile the merged bytecode with a function specialization target:

```bash
loom-compile portable.loombc \
  --backend=amdgpu-hal \
  --target=gfx1100 \
  --output=scale_i32.vmfb \
  --emit-target-artifact=scale_i32.hsaco \
  --artifact-manifest=summary \
  --compile-report=summary
```

`--target=gfx1100` does not establish a module-global target. The command-line
driver maps the requested profile to the HAL kernel entries it is compiling,
materializes the exact target record once, and writes that durable target onto
those functions before target-aware passes run. A module may still contain
unrequested functions for other targets.

The loomc C API expresses the same operation with
`loomc_target_specialization_options_t` attached only to
`loomc_compile_options_t.next`. A direct specialization row pairs one function
symbol with one structured target profile. A target binding row instead pairs
one authored `target.decl` symbol with a profile and seeds every function using
that declaration; this is the compact form for heterogeneous command or VM
programs with named target roles. Plain loading and linking preserve all
authored targets and never accept a specialization option; emission consumes
the durable targets in prepared IR and never accepts an override. Embedders can
therefore link a multi-target library once, clone or filter it as appropriate,
and specialize different function versions in later compile invocations.

The artifact manifest is the sidecar a packager or benchmark database should
keep with `scale_i32.hsaco`. The compile report is the per-invocation feedback
channel for status, diagnostics, timings, and analysis summaries requested by
the command line or C API.

## Debugging Selection

Use pass IR dumps during compilation to see which provider was selected for a
specialized function:

```bash
loom-compile portable.loombc \
  --backend=amdgpu-hal \
  --target=gfx1100 \
  --output=scale_i32.vmfb \
  --emit-target-artifact=scale_i32.hsaco \
  --dump-ir-after=select-templates \
  --dump-ir-after=inline-callables \
  --dump-ir-format=jsonl \
  --dump-ir-output=trace/
```

`trace/trace.jsonl` is convenient for agents and scripts. The adjacent
`trace/ir/*.loom` files are the human-readable snapshots. Provider selection
failures should be investigated at the `select-templates` boundary before
debugging lower-level target code.

For a smaller provider-selection query, run `select-templates` through
`loom-opt --pass-report=json` before artifact compilation. Each
`template-selection` detail row names the enclosing function, contract,
selected provider, effective target when known, candidate counts, and the
selection outcome:

```bash
loom-opt linked.loom \
  --pass=select-templates \
  --pass-report=json \
  --output=/tmp/selected.loom \
  2>/tmp/pass-report.json

jq '.invocations[]
  | select(.pass == "select-templates")
  | .details[]
  | select(.category == "template-selection")' /tmp/pass-report.json
```

## Authoring Pressure Points

`func.call @symbol` names one exact helper. Use it for mechanical helpers whose
identity is part of the algorithm.

`template.apply<@family>` names a compile-time implementation demand. Use it
when a library may provide several target-, layout-, or shape-specialized
implementations.

`target(@...)` on a provider is an applicability constraint. It should describe
where that provider is valid, not force every caller to carry target attributes.

Priority breaks ties after signature and predicate filtering. Give the best
specialized provider a higher priority than a generic fallback, and keep
fallback behavior correct so unsupported targets still have a useful path.

Config and shape choices should be explicit source facts. Use `check.param`
values, function arguments, and assumptions so the linker/JIT can specialize by
binding config or choosing roots without regenerating source text for ordinary
cases.
