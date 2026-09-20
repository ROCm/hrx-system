# Compile artifacts

`loom-compile` specializes one verified `.loom` or `.loombc` module and emits an
artifact for selected roots. It is the offline form of the same parse, link,
configure, specialize, lower, and emit operations available through the `loomc`
API.

Selected roots infer exactly one product: `kernel`, `command`, or `module`.
With explicit `--root` values, `--product` is an optional assertion on that
result and never reinterprets the roots. With no roots, an explicit product
selects its canonical root set. `--target=family:selector` supplies a target
when the roots do not already name one. `--format` requests an exact output
encoding; when it is omitted, the tool requires one canonical compatible
format. Reports, manifests, and IR traces are optional evidence products, not
boilerplate required by every compile.

## Compile a loadable kernel

Compile one targetless kernel for the generic GFX11 profile:

```shell
loom-compile kernel.loom \
  --target=amdgpu:gfx11-generic \
  --output=kernel.hsaco
```

Targets use `family:selector` syntax. The family selects one target provider
linked into the tool, and the remainder is interpreted only by that provider.
A build without the requested family fails instead of enabling or loading it
implicitly.

The kernel roots and AMDGPU target select the canonical `amdgpu-hsaco` format.
Pass `--format=amdgpu-hsaco` when the exact encoding is part of the invocation's
contract. The primary output is the executable byte sequence consumed by the
selected HAL loader. The source module can contain checks, reference functions,
template providers, and other authoring support; the selected format
materializes kernel entries and their dependency closures into the executable
library.

Format and target availability are properties of the installed Loom tool.
Target pages own supported profile names; the compile workflow remains the same
across target families and formats. Offline emission is independent of runtime
drivers: a Loom build with SPIR-V targeting and emission can produce SPIR-V
artifacts without the Vulkan HAL, while creating a device and executing those
artifacts still requires the Vulkan driver.

## Choose generic or exact specialization

A generic profile retains portability within one target family:

```shell
loom-compile kernel.loom \
  --target=amdgpu:gfx11-generic \
  --output=kernel-gfx11.hsaco
```

An exact profile exposes the features and limits of one physical target:

```shell
loom-compile kernel.loom \
  --target=amdgpu:gfx1151 \
  --output=kernel-gfx1151.hsaco
```

`--target` specializes every materialized kernel entry before the compile
pipeline. An authored target remains a compatibility requirement: a generic
GFX11 entry can specialize to `gfx1151`, while an entry constrained to an
incompatible family fails. A kernel compile only needs `--target` when a
materialized root does not already carry an authored target.

Reusable libraries generally omit authored targets. Their source can then
specialize for the target selected by the application, benchmark, or artifact
build rather than fragmenting into target-specific copies.

## Select roots from a catalog

When `--root` is omitted, public or retained command programs take precedence;
otherwise the tool selects every kernel entry, or the whole module when neither
kind exists. Select one or more entries from a catalog by repeating the flag.
Every selected root must infer the same product:

```shell
loom-compile catalog.loombc \
  --root=@prefill \
  --root=@decode \
  --target=amdgpu:gfx11-generic \
  --output=qwen-kernels.hsaco
```

Root selection is reachability, not name filtering after compilation. Unused
functions, providers, configurations, checks, and kernels are absent from the
materialized compile module. In a mixed command-and-kernel catalog, name the
kernel roots explicitly or pass `--product=kernel` without roots to select all
kernel entries. A product never changes the meaning of explicit roots.

Use [`loom-link`](link-and-package.md#link-transitive-dependencies-incrementally)
first when several independently shipped modules must be composed. Use
`--root` directly when one linked catalog already contains the complete
declared dependency graph.

## Bind compile-time configuration

Bind the `config.decl` values that describe this artifact:

```shell
loom-compile kernel.loombc \
  --root=@decode \
  --target=amdgpu:gfx1151 \
  --config=model.hidden_size=4096 \
  --config=model.head_count=32 \
  --output=decode.hsaco
```

JSON and JSONC files carry larger configuration objects:

```shell
loom-compile kernel.loombc \
  --target=amdgpu:gfx1151 \
  --config-file=model-config.jsonc \
  --output=model-kernels.hsaco
```

Configuration materializes before root dependency walking and the pass
pipeline. Specialization can therefore select providers and remove unreachable
paths before later compilation work. Nested JSON keys flatten with `.`
separators; explicit bindings not referenced by the module are ignored.

## Emit portable command programs

The command product prepares selected command-program roots and emits one
portable artifact per root. Command roots select the canonical `loom-command`
format, which may also be stated explicitly:

```shell
loom-compile model.loombc \
  --root=@elementwise_transform \
  --format=loom-command \
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

Compile each manifest-listed source request independently for its target. For
example, the first request can be compiled with:

```shell
loom-compile kernel-requests/kernel-0.loombc \
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
loom-compile kernel.loom --format=llvmir-text --output=kernel.ll
loom-compile kernel.loom --format=llvmir-bitcode --output=kernel.bc
```

## Emit a WebAssembly module

An installation with Wasm enabled can compile ordinary functions into a binary
module. Save this as `sum.loom`:

```loom
wasm.target<simd128> @target

func.def public target(@target) @sum_to(%end: index) -> (index) {
  %begin = index.constant 0 : index
  %step = index.constant 1 : index
  %terminal, %sum = scf.while(%before = %begin : index, %total = %begin : index) -> (index, index) {
    %more = index.cmp ult, %before, %end : index
    scf.condition %more, %before, %total : i1, index, index
  } do(%position: index, %partial: index) {
    %next_sum = index.add %partial, %position : index
    %next = index.add %position, %step : index
    scf.yield %next, %next_sum : index, index
  }
  func.return %sum : index
}
```

Compile it with the default pipeline, which preserves Wasm's structured control
flow:

```shell
loom-compile sum.loom --format=wasm-binary --output=sum.wasm
```

Public functions become exports. For example, a JavaScript host can instantiate
the artifact and call the sum for a nonnegative end value:

```javascript
const bytes = await (await fetch('sum.wasm')).arrayBuffer();
const {instance} = await WebAssembly.instantiate(bytes);
console.log(instance.exports.sum_to(4)); // 6
```

Wasm uses 32-bit `index` and `offset` carriers. Address casts connect these types
with `i32`; converting an offset to an index requires a value no greater than
`2147483647`. An `i64` input can also cast to `index` when its proven range fits
`[-2147483648, 2147483647]`, or to `offset` when it fits `[0, 4294967295]`.
These casts require facts about the input; an assumption on the cast's result
does not establish that the input fits. Ordinary arithmetic can prove the range,
or `scalar.assume` can state a guarantee made by the caller.

Scalar `i32`, `i64`, `f32`, and `f64` memory accesses support static and dynamic
view origins, bounded indices, and dynamic strides. Wide integer origins can
combine with element indices and static byte prefixes; the complete relative
byte offset must fit within 32 bits. Modules using memory define one private
64 KiB linear memory; buffer parameters are byte addresses into it, and exported
functions provide the host's access to that memory.

## Emit a target-native sidecar

A loadable kernel format may have both a loader-ready representation and a
target-native artifact. Request both when an integration needs the primary
loader product and tooling needs the native object:

```shell
loom-compile kernel.loom \
  --format=amdgpu-hsaco \
  --target=amdgpu:gfx11-generic \
  --output=kernel.executable \
  --emit-target-artifact=kernel.hsaco
```

The two byte sequences may be identical. AMDGPU currently uses HSACO for both;
the separate output contract still matters for formats whose loader container
and native artifact differ.

## Emit an artifact manifest

An artifact manifest describes the loader product without asking a consumer to
reverse-engineer it:

```shell
loom-compile kernel.loom \
  --format=amdgpu-hsaco \
  --target=amdgpu:gfx11-generic \
  --output=kernel.hsaco \
  --artifact-manifest=summary
```

With a filesystem artifact output, the manifest path defaults to
`kernel.hsaco.manifest.json`. Name it explicitly when packaging has a fixed
layout:

```shell
loom-compile kernel.loom \
  --format=amdgpu-hsaco \
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
