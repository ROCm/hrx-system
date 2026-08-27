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

## Binary roots close one product

All three binary rules share the same composition contract:

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
unreachable alternatives are discarded. The same profile then drives device
artifact emission. VM binaries have no device `target` attribute and keep this
link boundary targetless.

## Kernel binaries are loader-ready executables

`loom_kernel_binary` specializes the selected kernel closure for one immutable
target profile and emits `<name>.hsaco` for the current AMDGPU product. The
example's profile label carries the typed family and the target selector
`gfx11-generic`; it is a Bazel target, not a source file or filename-like
configuration blob.

```shell
bazel build //loom/docs/examples/elementwise-transform:elementwise_kernel
```

Reusable source remains targetless. Selecting the profile at the binary
boundary allows the same library to produce a generic GFX11 executable or an
exact architecture-specialized executable without copying its `.loom` files.

## Command binaries package schedules with their kernels

`loom_command_binary` performs one selective link, lowers every selected
command-program root to a portable artifact, and compiles the reachable kernel
entries for its target profile:

```shell
bazel build //loom/docs/examples/elementwise-transform:elementwise_command
```

Its default outputs are:

| Output | Consumer |
| --- | --- |
| `<name>.commands.json` | Maps command symbols to portable artifacts and lists their logical executable-entry requirements. |
| `<name>.commands/*.loomcmd` | One target-neutral command artifact per selected command root. |
| `<name>.kernels.hsaco` | The AMDGPU executable satisfying the manifest's reachable kernel entries. |

The command schedule and device executable remain separate deployment
artifacts because they have different portability and caching boundaries. The
manifest joins them through logical entry symbols; it does not force an
embedding to reverse-engineer either binary format.

## VM binaries contain authored VM functions

`loom_vm_binary` links functions authored for a `vm.target` and emits
`<name>.vmfb`. VM images do not embed a module name; the host assigns a name
when it loads the image into its runtime environment.

```shell
bazel build //loom/docs/examples/elementwise-transform:elementwise_vm

iree-dump-module \
  bazel-bin/loom/docs/examples/elementwise-transform/elementwise_vm.vmfb
```

The result is an executable VM bytecode module; the dump includes
`@double_i32` as a public export. Runtime applications assign the module name
while loading the image and invoke exports through the generic VM API.
This product consumes targets authored on the VM functions and has no `target`
attribute. It does not infer or bundle command-program and device artifacts:
those are explicit `loom_command_binary` products with their own target profile
and runtime ownership.

## Inspect the closed input and compiler evidence

Binary rules keep their primary runtime products in Bazel's default output
group. Their linked input and compile reports are opt-in evidence products:

```shell
bazel build //loom/docs/examples/elementwise-transform:elementwise_command \
  --output_groups=+linked_modules,+compile_reports
```

`linked_modules` contains the closed `.loombc` used by every emitter for that
binary. `compile_reports` contains the command and kernel reports for a command
binary and the corresponding single report for kernel and VM binaries. This
makes it possible to inspect reachability or compare compiler evidence without
changing the product graph.

## The CLI and in-memory APIs use the same boundaries

The Bazel rules orchestrate the public tools; they do not add a second linkage
model. `loom_library` corresponds to a strict relocatable merge. A kernel or
command binary first performs a root-selected `loom-link --mode=link` with the
selected `--target-profile`, then invokes the appropriate `loom-compile`
backend on that one closed module. A VM binary performs the same selective
link without a device profile. The command product invokes the command and
AMDGPU emitters over the same linked input.

An embedding can construct the same explicit library universe with the
[`loomc` API](../integration/module-composition.md), select roots, and retain or
emit the resulting module entirely in memory. Bazel labels and CLI paths are
frontend identities for artifacts and diagnostics; neither becomes a Loom
symbol namespace or causes the compiler to search a filesystem.

[Link and package modules](link-and-package.md) gives the equivalent
command-line composition workflow. [Compile artifacts](compile-artifacts.md)
documents the kernel, command, and VM emitters directly.
