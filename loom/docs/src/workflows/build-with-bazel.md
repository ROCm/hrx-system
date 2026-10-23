# Build libraries and binaries with Bazel

The public Bazel rules name the artifact being built. A
[`loom_library`](#libraries-stay-relocatable) produces reusable Loom bytecode;
`loom_kernel_binary` closes selected kernel roots into a loader-ready
executable. The same source graph can therefore stop at a linkable library or
continue into the executable required by one application.

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

## Binary roots close one kernel executable

`loom_kernel_binary` uses the same composition contract as the library rules:

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

Explicit `roots` replace the default export set.

Root selection happens during linking. Unreachable functions, templates,
kernels, command programs, configuration, checks, and benchmarks are absent
from the closed `.loombc` passed to artifact emission.

Kernel binaries also require a typed `target` label. That profile
participates in the selective link itself: target facts are projected before
template selection, so a target-constrained provider can win before
unreachable alternatives are discarded. The same profile then drives device
artifact emission.

## Kernel binaries are loader-ready executables

`loom_kernel_binary` specializes the selected kernel closure for one immutable
target profile and emits one artifact named by `out`. The compiler chooses the
canonical kernel format for that target. The example therefore names
`elementwise_kernel.hsaco` for its `gfx11-generic` AMDGPU profile, while a rule
using `//loom/target/spirv:vulkan1.3+bda` can name an `.spv` output without
changing the base rule.

`out` controls only the Bazel artifact path; the compiler never infers the
format from its suffix. When omitted, the artifact has the extensionless rule
name. This is useful for higher-level macros that fan out target products but
do not own target-specific filename conventions.

```shell
bazel build //loom/docs/examples/elementwise-transform:elementwise_kernel
```

Reusable source remains targetless. Selecting the profile at the binary
boundary allows the same library to produce a generic GFX11 executable or an
exact architecture-specialized executable without copying its `.loom` files.

## Inspect the closed input and compiler evidence

Binary rules keep their primary runtime products in Bazel's default output
group. Their linked input and compile reports are opt-in evidence products:

```shell
bazel build //loom/docs/examples/elementwise-transform:elementwise_kernel \
  --output_groups=+linked_modules,+compile_reports
```

`linked_modules` contains the closed `.loombc` used by the emitter for that
binary. `compile_reports` contains the corresponding compiler report. This
makes it possible to inspect reachability or compare compiler evidence without
changing the product graph.

## Test execution and compiler profiles together

`loom_test` keeps execution and offline compiler qualification beside the same
authored sources:

```starlark
load("@hrx//loom/build_tools/bazel:defs.bzl", "loom_test")
load("@hrx//loom/target:execution_profiles.bzl", "GPU_HARDWARE_PROFILES")

loom_test(
    name = "address_tests",
    srcs = ["address_tests.loom"],
    execution_profiles = GPU_HARDWARE_PROFILES,
    compile_targets = [
        "@hrx//loom/target/amdgpu:gfx942",
        "@hrx//loom/target/amdgpu:gfx1151",
    ],
)
```

The sources own `check.case` or `check.benchmark` roots; `deps` contribute only
reachable definitions. Execution and compilation consume the same linked test
module. The owning test target includes both phases, plus lint and format checks
for its authored `.loom` files. Source checks run once regardless of workload
or profile count and do not inherit execution profiles' device requirements. Each
`_execute_<profile>_test` child requires only its execution profile's device,
while `_compile_gfx942` and `_compile_gfx1151` run the offline compiler on the
host. Compilation succeeds
only when final artifacts are produced; it does not execute numerical checks.

`GPU_HARDWARE_PROFILES` selects native AMDGPU and Vulkan execution independently.
Use `VM_REFERENCE_PROFILE` from `@hrx//loom/target/vm:execution_profiles.bzl`
for reference function execution. An empty `execution_profiles` list creates
no execution checks; at least one execution or compiler profile is required.
Adding profiles creates separate results without importing or linking the
source closure again. Device availability never gates a sibling compiler check.

### Share a corpus across test packages

`loom_test_module` gives a corpus its own source and fixture ownership. A target
package can consume that module with `loom_test(module = ...)`, applying its
execution profiles without repeating the import or link:

```starlark
# In the corpus package:
loom_test_module(
    name = "copy_cases",
    srcs = ["copy.cxx", ":copy_checks.loom"],
    data = [":reference_arrays"],
    visibility = ["//visibility:public"],
)

# In a target qualification package:
loom_test(
    name = "copy_test",
    module = "//corpus:copy_cases",
    execution_profiles = [AMDGPU_HARDWARE_PROFILE],
)
```

Load both rules from `@hrx//loom/build_tools/bazel:defs.bzl` and the AMDGPU
profile from `@hrx//loom/target/amdgpu:execution_profiles.bzl`. Source-provider
options and runtime fixtures belong on the module. Configuration bindings,
case selection, and compiler or execution profiles belong on each consumer.
The shared module builds independently of device availability, and its runtime
fixtures follow it into every consumer. `module` is exclusive with source and
import arguments on `loom_test`.

The corpus owns its lint and format checks in `//corpus:copy_cases_test`.
Recursive test discovery includes that source suite; consumers such as
`copy_test` contain only their selected compiler and execution checks. This keeps
source checks independent of how many target packages consume the module.
The checks retain the source owner's tags and source-provider constraints.
Modules with no authored `.loom` inputs do not declare a source-check suite.

### Workload variants share one source owner

`configs` binds compile-time values for compiler qualification, correctness,
and benchmark smoke. `case` selects cases for the two numerical runners;
compiler qualification still covers the entire owned module. These settings
belong to the workload, while execution profiles own device selection,
instrumentation, and resource requirements.

Use named `variants` when the same program needs several configurations:

```starlark
--8<-- "examples/guide/functions-and-control/BUILD.bazel:workload_variants"
```

Each row overrides keys in the common `configs` mapping and may override the
common `case` selector. Configuration values are strings, as on binary rules.
Omitting `variants` creates one default workload; supplying it names the
complete set, with no extra default row. An empty mapping is an authoring error.

Each workload has independent compiler and execution children, named
`<name>_<variant>_compile_<target>` and
`<name>_<variant>_execute_<profile>_test`. The source import and linked module
remain shared across every row and profile. `args` remains correctness-only;
`--config` and `--case` are rejected there and in execution profiles so the two
numerical runners cannot accidentally select different workloads.

### Compiler fixtures retain their source assertions

Compiler fixtures have the same alongside option. `loom_check_test` accepts a
`compile_targets` list; `loom_check_test_suite` accepts a map from existing
source paths to profile lists. Each source case uses the ordinary input
provider and diagnostic annotations. Compilation
does not compare RUN goldens or inherit their execution skips and expected
failures. Profile labels supply typed compiler identities, independently of
execution resource requirements.

## The CLI and in-memory APIs use the same boundaries

The Bazel rules orchestrate the public tools; they do not add a second linkage
model. `loom_library` corresponds to a strict relocatable merge. A kernel
binary first performs a root-selected `loom-link --mode=link` with the selected
`--target`, then invokes `loom-compile` on that one closed module.

An embedding can construct the same explicit library universe with the
[`loomc` API](../integration/module-composition.md), select roots, and retain or
emit the resulting module entirely in memory. Bazel labels and CLI paths are
frontend identities for artifacts and diagnostics; neither becomes a Loom
symbol namespace or causes the compiler to search a filesystem.

[Link and package modules](link-and-package.md) gives the equivalent
command-line composition workflow. [Compile artifacts](compile-artifacts.md)
documents kernel and module emitters directly.
