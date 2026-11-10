# Explicit rocjitsu runtime

The libamdf adapter compiles against five pinned public C headers. The simulator
is loaded at runtime; its C++ implementation and runtime files are not linker
inputs to libamdf or its consumers. The pin is declared in
[`deps.MODULE.bazel`](../deps.MODULE.bazel). CMake uses the same declarations
through `MODULE.cmake.lock` and `rocjitsu.cmake`.

`@rocjitsu_runtime//:runtime` supplies a separately built package as execution
data. Select its root with
`--repo_env=IREE_ROCJITSU_RUNTIME_PATH=/absolute/path/to/package`. The repository
is lazy: native builds do not require a simulator package.

The package preserves the simulator's installed layout:

```text
runtime.json
lib/librocjitsu.so
lib/<non-glibc ELF dependencies>
libexec/rocjitsu/rocjitsu-memory
```

The memory helper is an executable owned by the simulator. It is required for
memory backing, and the simulator resolves it relative to its loaded library.
It runs without the application's loader environment. Its compiler runtime
dependencies must be statically linked or resolved from the package through its
own relative loader paths. Glibc and the ELF interpreter belong to the execution
platform and are not packaged.

`runtime.json` describes the actual build and SHA256 digests of every payload:

```json
{
  "format": 2,
  "revision": "<full pinned rocm-systems commit>",
  "host_os": "linux",
  "host_cpu": "x86-64-v3",
  "interposer": false,
  "host_fault_recovery": false,
  "sanitizer": "address",
  "files": {
    "lib/librocjitsu.so": "<sha256>",
    "lib/libstdc++.so.6": "<sha256>",
    "lib/libgcc_s.so.1": "<sha256>",
    "libexec/rocjitsu/rocjitsu-memory": "<sha256>"
  }
}
```

`host_cpu` is `x86-64` or `x86-64-v3`. `sanitizer` is `none` for an ordinary
build or `address` for an ASan build; an ASan simulator requires an ASan consumer.
Both build variants disable `ROCJITSU_ENABLE_INTERPOSER` and
`RJ_ENABLE_HOST_FAULT_RECOVERY`. The manifest records these build requirements;
checksums and ELF closure validation cannot establish how a library was compiled.
The package producer must qualify that configuration.

The runtime repository checks the pin, payload checksums, helper executability,
and complete non-glibc ELF closure, then preserves the layout in runfiles. It
rejects missing dependencies and dependencies resolved outside the package.

`rocjitsu_executable` in `invocation.bzl` binds this runtime and an ordered list
of topology configurations to an ordinary executable's existing library and
configuration options. The executable retains the caller's compilation
configuration; the simulator is added only as execution data. The rule checks
that an ASan runtime has an ASan consumer. An ordinary runtime supports ordinary
and ASan consumers.

The configurations under `profiles/` describe complete multi-GPU topologies.
They have no simulator tick limit: the consumer owns shutdown, and the outer
test harness detects hangs.

The invocation sets its loader search path to the package's library directory,
checks the package's host CPU requirement, and prepares a descriptor soft limit
of at least 8,192 before exec. This prevents the simulator from adjusting that
process limit during initialization. It preserves arguments and inherited signal
dispositions, and rejects overrides of fixed bootstrap options. Consumers with
flag files or aliases can also receive a flag-lock option for their own parser
to enforce. Library ownership and VM creation remain in the consumer.

The manual `//build_tools/third_party/rocjitsu:runtime_test` executes passive
discovery and two complete VM lifecycles through the declared runtime. It checks
that discovery does not activate the helper, activation does, teardown releases
its lease and unloads the library, and host signal handlers remain intact. Run
it with the sanitizer configuration matching the selected package, for example:

```sh
iree-bazel-test --config=asan \
  --repo_env=IREE_ROCJITSU_RUNTIME_PATH=/absolute/path/to/package \
  //build_tools/third_party/rocjitsu:runtime_test
```
