# Compile artifacts

`loom-compile` specializes one verified `.loom` or `.loombc` module and emits an
artifact for a selected backend. It is the offline form of the same parse,
link, configure, specialize, lower, and emit operations available through the
`loomc` API.

The shortest useful invocation names the input, backend, target profile, and
artifact path. Reports, manifests, and IR traces are optional evidence products,
not boilerplate required by every compile.

## Compile for a HAL loader

Compile one targetless kernel for the generic GFX11 profile:

```shell
loom-compile kernel.loom \
  --backend=amdgpu-hal \
  --target=amdgpu:gfx11-generic \
  --output=kernel.hsaco
```

Targets use `family:selector` syntax. The family selects one target provider
linked into the tool, and the remainder is interpreted only by that provider.
A build without the requested family fails instead of enabling or loading it
implicitly.

The primary output is the executable byte sequence consumed by the selected
HAL loader. For AMDGPU that representation is currently an HSACO. The source
module can contain checks, reference functions, template providers, and other
authoring support; the HAL backend materializes kernel entries and their
dependency closures into the executable library.

Backend availability is a property of the installed Loom tool. Target pages
own supported profile names; the compile workflow remains the same across
backends. Offline emission is independent of runtime drivers: a Loom build with
SPIR-V targeting and emission can produce SPIR-V artifacts without the Vulkan
HAL, while creating a device and executing those artifacts still requires the
Vulkan driver.

## Choose generic or exact specialization

A generic profile retains portability within one target family:

```shell
loom-compile kernel.loom \
  --backend=amdgpu-hal \
  --target=amdgpu:gfx11-generic \
  --output=kernel-gfx11.hsaco
```

An exact profile exposes the features and limits of one physical target:

```shell
loom-compile kernel.loom \
  --backend=amdgpu-hal \
  --target=amdgpu:gfx1151 \
  --output=kernel-gfx1151.hsaco
```

`--target` specializes every materialized kernel entry before the compile
pipeline. An authored target remains a compatibility requirement: a generic
GFX11 entry can specialize to `gfx1151`, while an entry constrained to an
incompatible family fails. A HAL compile only needs `--target` when a
materialized root does not already carry an authored target.

Reusable libraries generally omit authored targets. Their source can then
specialize for the target selected by the application, benchmark, or artifact
build rather than fragmenting into target-specific copies.

## Select roots from a catalog

When `--root` is omitted, a HAL backend compiles every kernel entry and its
dependency closure. Select one or more entries from a catalog by repeating the
flag:

```shell
loom-compile catalog.loombc \
  --root=@prefill \
  --root=@decode \
  --backend=amdgpu-hal \
  --target=amdgpu:gfx11-generic \
  --output=qwen-kernels.hsaco
```

Root selection is reachability, not name filtering after compilation. Unused
functions, providers, configurations, checks, and kernels are absent from the
materialized compile module.

Use [`loom-link`](link-and-package.md#link-transitive-dependencies-incrementally)
first when several independently shipped modules must be composed. Use
`--root` directly when one linked catalog already contains the complete
declared dependency graph.

## Bind compile-time configuration

Bind the `config.decl` values that describe this artifact:

```shell
loom-compile kernel.loombc \
  --root=@decode \
  --backend=amdgpu-hal \
  --target=amdgpu:gfx1151 \
  --config=model.hidden_size=4096 \
  --config=model.head_count=32 \
  --output=decode.hsaco
```

JSON and JSONC files carry larger configuration objects:

```shell
loom-compile kernel.loombc \
  --backend=amdgpu-hal \
  --target=amdgpu:gfx1151 \
  --config-file=model-config.jsonc \
  --output=model-kernels.hsaco
```

Configuration materializes before root dependency walking and the pass
pipeline. Specialization can therefore select providers and remove unreachable
paths before later compilation work. Nested JSON keys flatten with `.`
separators; explicit bindings not referenced by the module are ignored.

## Emit portable command programs

The command backend prepares selected command-program roots and emits one
portable artifact per root:

```shell
loom-compile model.loombc \
  --root=@elementwise_transform \
  --backend=command \
  --output=commands.json \
  --emit-command-artifacts=commands/ \
  --emit-kernel-requests=kernel-requests/
```

`commands.json` maps each command symbol to a `.loomcmd` file and records the
logical kernel entries that artifact requires. The files under `commands/`
contain target-neutral resource bindings, schedule waves, dispatches, and
executable slots. They do not embed a device executable. Source-backed entries
also name ordinary `.loombc` modules under `kernel-requests/`. Launch sites are
partitioned only by decisions that change their generated kernels, so repeated
sites and separate command roots share a request when their semantic class is
the same. External bodyless entries remain plain binding requirements.

Compile each manifest-listed source request independently through the target
backend. For example, the first request can be compiled with:

```shell
loom-compile kernel-requests/kernel-0.loombc \
  --backend=amdgpu-hal \
  --target=amdgpu:gfx11-generic \
  --output=kernel-0.hsaco
```

Each request is already closed around one semantic kernel class while retaining
the ordinary target-selection surface. The embedding compiles or cache-resolves
those requests concurrently, then binds each resulting executable entry to the
manifest ordinal that named it. The manifest is the parent commit point: request
files emitted by a failed command compilation are not a usable artifact set.

The public
[`loom_command_binary`](build-with-bazel.md#command-binaries-package-schedules-with-their-kernels)
rule retains the static packaging workflow. It compiles the linked kernel module
as one executable and exposes it with the manifest and portable artifacts as one
Bazel product target.

Target-owned emitters can also expose intermediate deployment formats directly.
For example, an installation with the LLVM IR emitter can write textual or
bitcode artifacts:

```shell
loom-compile kernel.loom --backend=llvmir-text --output=kernel.ll
loom-compile kernel.loom --backend=llvmir-bitcode --output=kernel.bc
```

## Emit a target-native sidecar

A HAL backend may have both a loader-ready representation and a target-native
artifact. Request both when an integration needs the primary loader product and
tooling needs the native object:

```shell
loom-compile kernel.loom \
  --backend=amdgpu-hal \
  --target=amdgpu:gfx11-generic \
  --output=kernel.executable \
  --emit-target-artifact=kernel.hsaco
```

The two byte sequences may be identical. AMDGPU currently uses HSACO for both;
the separate output contract still matters for backends whose loader container
and native artifact differ.

## Emit an artifact manifest

An artifact manifest describes the loader product without asking a consumer to
reverse-engineer it:

```shell
loom-compile kernel.loom \
  --backend=amdgpu-hal \
  --target=amdgpu:gfx11-generic \
  --output=kernel.hsaco \
  --artifact-manifest=summary
```

With a filesystem artifact output, the manifest path defaults to
`kernel.hsaco.manifest.json`. Name it explicitly when packaging has a fixed
layout:

```shell
loom-compile kernel.loom \
  --backend=amdgpu-hal \
  --target=amdgpu:gfx11-generic \
  --output=kernel.hsaco \
  --artifact-manifest=details \
  --emit-artifact-manifest=kernel.manifest.json
```

Summary manifests expose the stable loader-facing inventory. Details and
analysis modes add progressively richer target metadata. The manifest answers
which functions, exports, bindings, launch sizes, constants, and target facts
are present; it does not explain why the compiler chose them.

```shell
jq '.functions[] | {name, target, workgroup_size}' kernel.manifest.json
```

## Keep compiler evidence separate

The primary artifact is what a runtime loads. The target-native sidecar is what
target tooling consumes. The manifest describes the emitted interface. A
compile report records compiler and emitted-code evidence. An IR trace records
the program at selected pipeline boundaries.

Generate each product only for the consumer that needs it. Routine application
builds can stop at the artifact; tuning runs continue with
[Read compile reports](compile-reports.md).
