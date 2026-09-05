# Build libraries and binaries with Bazel

The public Bazel rules name the artifact being built. A
[`loom_library`](#libraries-stay-relocatable) produces reusable Loom bytecode;
`loom_kernel_binary` and `loom_command_binary` close selected roots into
deployment products. The same source graph can therefore stop at a linkable
library or continue into the product required by one application.

## Depend on HRX

HRX publishes Loom as the `loom/` subproject of its root Bzlmod module. An
independent authoring repository using a local HRX checkout declares that
dependency in its root module:

```starlark title="MODULE.bazel"
module(name = "hrx_loom_kernels")

bazel_dep(name = "hrx", version = "0.0.0")

local_path_override(
    module_name = "hrx",
    path = "../hrx-system",
)
```

The dependency's default apparent repository name is `@hrx`, so BUILD files
load the public authoring API from
`@hrx//loom/build_tools/bazel:defs.bzl` and select built-in profiles from
`@hrx//loom/target/...`. The local override changes where Bazel obtains the HRX
module without changing those labels. It also preserves the compiler
co-development loop: editing Loom in the referenced checkout makes the next
kernel build rebuild the affected source tools before consuming them.

The HRX module registers source-built toolchains for each Loom authoring role.
A root module may register a higher-priority implementation of the same public
toolchain types when it consumes released executables instead. The library and
binary rules remain unchanged because they resolve tools by role rather than by
an executable label.

Inside the HRX source tree, `@hrx//loom/...` and `//loom/...` reach the same
packages. The checked examples use the external spelling so their BUILD
declarations can move unchanged into a standalone kernel repository.

The declaration below is exercised by the documentation test suite:

```starlark title="BUILD.bazel"
--8<-- "examples/elementwise-transform/BUILD.bazel:products"
```

The three source libraries form one ordinary dependency graph:

```text
model  ->  kernel  ->  motif
```

`model.loom` declares the kernel it launches, `kernel.loom` declares the
template family it applies, and `motif.loom` contributes eligible template
implementations. Bazel labels state which library artifacts are available;
Loom declarations and template contracts state how symbols compose.

## Libraries stay relocatable

`loom_library` merges its direct `srcs` into `<name>.loombc`. That bytecode is a
single, independently reloadable Loom module, not target-native code and not an
archive of named source files. Unresolved declarations may remain for a later
link boundary.

Dependencies deliberately remain separate. The `:model` bytecode does not
flatten `:kernel` or `:motif`; its `LoomLibraryInfo` carries those modules as an
independent dependency closure. A final product can then index the complete
library universe and materialize only the definitions reachable from its roots.

The rule also performs strict direct-dependency analysis. A source reference
may be satisfied by its own module or by a library named directly in `deps`.
Finding the symbol only through a transitive dependency is an error. In the
example, `model` names `kernel` and `kernel` names `motif`; `model` does not need
to repeat `motif` because it does not reference one of its symbols directly.
This keeps large library graphs gardenable without flattening them.

Build the relocatable model library alone:

```shell
bazel build //loom/docs/examples/elementwise-transform:model
```

The default output is `model.loombc`. Request its schema-versioned dependency
analysis when a build or dependency-gardening tool needs it:

```shell
bazel build //loom/docs/examples/elementwise-transform:model \
  --output_groups=+dependency_reports
```

## Binary roots close deployment products

Both deployment binary rules share the same composition contract:

| Attribute | Meaning |
| --- | --- |
| `srcs` | Direct `.loom` or `.loombc` sources assembled as an implicit relocatable library. |
| `deps` | Direct `loom_library` inputs whose exports may become roots and whose dependencies may satisfy the reachable closure. |
| `roots` | Optional explicit `@symbol` roots. When omitted, every exported symbol from the direct `srcs` and `deps` is a root. |
| `configs` | Compile-time configuration values keyed by `config.decl` symbol name. |

At least one of `srcs` and `deps` must be present. A source-only binary is the
compact spelling for a standalone program. A dependency-only binary is the
normal shape for a reusable library graph. A mixed binary adds small
application-owned sources to established libraries without creating a
one-purpose library target in the BUILD file.

Transitive exports never become roots merely because their library is in the
closure. They are candidates for satisfying reachable declarations and
template applications. This distinction is what lets `:model` bring a large
kernel catalog without compiling every exported kernel in that catalog.

Explicit `roots` replace the default export set. The checked
`elementwise_command` target above demonstrates this by selecting only
`@elementwise_transform` from `:model`.

Root selection happens during linking. Unreachable functions, templates,
kernels, command programs, configuration, checks, and benchmarks are absent
from the closed `.loombc` passed to artifact emission.

Kernel and command binaries also require a typed `target` label. That profile
participates in the selective link itself: target facts are projected before
template selection, so a target-constrained provider can win before
unreachable alternatives are discarded. The same profile then advertises the
compatible product formats used for artifact emission.

## Kernel binaries are loader-ready executables

`loom_kernel_binary` specializes the selected kernel closure for one immutable
target profile and emits one loader-ready file. The profile's canonical
`kernel` format determines the output extension when `format` is omitted. The
example's AMDGPU profile selects `amdgpu-hsaco` and therefore emits
`elementwise_kernel.hsaco`; a SPIR-V profile selecting `spirv-binary` would emit
an `.spv` file through the same rule.

```shell
bazel build //loom/docs/examples/elementwise-transform:elementwise_kernel
```

Reusable source remains targetless. Selecting the profile at the binary
boundary allows the same library to produce a generic GFX11 executable or an
exact architecture-specialized executable without copying its `.loom` files.

## Target and format labels are typed metadata

A target profile carries a `family:selector` identity, every compatible
product-format label, and at most one canonical format for each product. The
binary rules consume that metadata without naming a target family or executable
encoding:

| Attribute | Meaning |
| --- | --- |
| `target` | Required immutable profile used by selective linking and kernel compilation. |
| `format` | Optional exact format for the rule's primary product; omission selects the profile's canonical format. |
| `kernel_format` | Optional exact kernel format for `loom_command_binary`; omission selects the profile's canonical kernel format. |

HRX publishes built-in format labels under `@hrx//loom/product/formats:*` and
profiles under `@hrx//loom/target/...`. A downstream module may instead declare
its own `loom_file_product_format`, `loom_artifact_set_product_format`, and
`loom_target_profile` labels. Those declarations are inert analysis metadata:
they neither enable a target family nor add an emitter to `loom-compile`. The
selected Loom compile toolchain must already implement the requested target and
format, which lets an embedding supply both its own metadata and its own
toolchain without changing the generic rules.

## Command binaries package schedules with their kernels

`loom_command_binary` performs one selective link, lowers every selected
command-program root to a portable artifact, and compiles the reachable kernel
entries for its target profile. The command and kernel formats are resolved
independently because they have different portability and loading contracts:

```shell
bazel build //loom/docs/examples/elementwise-transform:elementwise_command
```

The AMDGPU example's default outputs are:

| Output | Consumer |
| --- | --- |
| `<name>.commands.json` | Maps command symbols to portable artifacts and lists their logical executable-entry requirements. |
| `<name>.commands/*.loomcmd` | One target-neutral command artifact per selected command root. |
| `<name>.kernels.hsaco` | The AMDGPU executable satisfying the manifest's reachable kernel entries. |

The command schedule and device executable remain separate deployment
artifacts because they have different portability and caching boundaries. The
manifest joins them through logical entry symbols; it does not force an
embedding to reverse-engineer either binary format.

## Inspect the closed input and compiler evidence

Binary rules keep their primary runtime products in Bazel's default output
group. Their linked input and compile reports are opt-in evidence products:

```shell
bazel build //loom/docs/examples/elementwise-transform:elementwise_command \
  --output_groups=+linked_modules,+compile_reports
```

`linked_modules` contains the closed `.loombc` used by every emitter for that
binary. `compile_reports` contains the command and kernel reports for a command
binary and the corresponding single report for a kernel binary. This makes it
possible to inspect reachability or compare compiler evidence without changing
the product graph.

## The CLI and in-memory APIs use the same boundaries

The Bazel rules orchestrate the public tools; they do not add a second linkage
model. `loom_library` corresponds to a strict relocatable merge. A kernel or
command binary first performs a root-selected `loom-link --mode=link` with the
selected `--target`, then invokes `loom-compile` on that one closed module with
the resolved `--product`, `--format`, and `--target`. A command binary emits its
portable command product and target-specific kernel product from the same
linked input.

An embedding can construct the same explicit library universe with the
[`loomc` API](../integration/module-composition.md), select roots, and retain or
emit the resulting module entirely in memory. Bazel labels and CLI paths are
frontend identities for artifacts and diagnostics; neither becomes a Loom
symbol namespace or causes the compiler to search a filesystem.

[Link and package modules](link-and-package.md) gives the equivalent
command-line composition workflow. [Compile artifacts](compile-artifacts.md)
documents kernel, command, and module products directly.
