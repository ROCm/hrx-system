# Run your first Loom kernel

**Example files:**
[`loom/docs/examples/getting-started/first-kernel/`](https://github.com/ROCm/hrx-system/tree/main/loom/docs/examples/getting-started/first-kernel)

This quickstart takes one source file from readable kernel code to checked
AMDGPU execution and a real benchmark. The example computes SAXPY,
`y = alpha * x + y`, over a runtime-sized `f32` vector.

The same file owns three related things:

- the kernel and its launch geometry;
- deterministic correctness samples and expected results; and
- a named benchmark that selects one of those proven samples.

There is no generated host harness or separate benchmark program. Loom's tools
compile the kernel for the selected device, materialize the tensors declared by
the case, launch the work, check the result, and measure the same workload when
requested.

This page assumes the Loom tools are on `PATH` and that `--device=amdgpu` is
available. [Acquiring Loom](acquiring-loom.md) describes the current
source-build and Bazel setup.

## Create one source file

Save the following as `saxpy.loom`:

```loom title="saxpy.loom"
--8<-- "generated/examples/getting-started/first-kernel/saxpy.loom"
```

The [`kernel.def`](../reference/dialects/kernel/ops/def.md) has two regions. The
first maps `%element_count` to a physical launch using 256 work-items per
workgroup. The second is the device body. Each work-item computes one element
index, guards the partial final workgroup, loads `x` and `y`, performs the fused
multiply-add, and stores the result back to `y`.

`%element_count` appears in both signatures for different reasons. The value
before `launch` is workload input used by the host-side launch calculation. The
value inside `launch(...)` is an explicit device argument used for bounds and
buffer views. Loom does not silently turn one into the other.

The [`check.case`](../reference/dialects/check/ops/case.md) expands to six
deterministic samples. Sizes 255, 256, and 257 cover both sides of the workgroup
boundary; 1009 exercises a larger tail; and 65536 supplies the benchmark
workload. Every sample fills `x` with 2, initializes `y` to 3, launches with
`alpha = 4`, and requires every result to equal 11.

The final [`check.benchmark`](../reference/dialects/check/ops/benchmark.md) does
not duplicate that setup. It gives the 65536-element assignment a stable
workload name and reuses the case's inputs, launch, and expectation.

## Check the source before compiling

`loom-format --check` parses and verifies the complete program and rejects noncanonical text:

```shell
loom-format --check saxpy.loom
```

This is more than whitespace checking. Invalid SSA use, mismatched types, and
malformed launch signatures fail at this boundary. The test and benchmark
runners then plan concrete parameter samples and reject assignments that do
not select a valid case sample.

## Execute every correctness sample on AMDGPU

Run the checked program directly:

```shell
iree-test-loom saxpy.loom --device=amdgpu >saxpy-test.json
```

The runner discovers `@saxpy_f32_case`, plans its six parameter assignments,
compiles the reachable kernel for the selected AMDGPU device, allocates and
initializes each tensor, executes the launch, and evaluates
`check.expect.equal`. It returns a failing process status if any sample fails
and writes the structured report before exiting.

The report distinguishes cases from their concrete samples. This case has one
authored definition and six executions; a failure records the exact
`element_count` and stable sample ordinal needed to reproduce it.

Select one sample while editing without changing the source:

```shell
iree-test-loom saxpy.loom \
  --device=amdgpu \
  --case=@saxpy_f32_case \
  --sample=3
```

Sample 3 is the 257-element tail. The ordinal comes from the deterministic
parameter plan rather than hidden random state.

## Benchmark the proven workload

Measure the named 64K dispatch workload:

```shell
iree-benchmark-loom saxpy.loom \
  --device=amdgpu \
  --benchmark=@saxpy_f32_64k \
  --measure=dispatch_complete \
  --batch-size=64 \
  --output=saxpy-benchmark.json
```

The benchmark runner executes the selected case and verifies its expected
result before accepting timing evidence. `dispatch_complete` measures host
submission through device completion for a prepared batch, and the report
normalizes the result per logical SAXPY operation. The batch size remains part
of the recorded measurement policy instead of becoming source metadata.

Benchmark with an optimized, uninstrumented tool build on an otherwise quiet
device. The JSON report carries device identity, parameter values, timing
policy, score distribution, and warnings needed to interpret the number.

The checked-in example wraps the same steps and also emits a generic gfx11
deployment artifact:

```shell
loom/docs/examples/getting-started/first-kernel/run.sh \
  gfx11-generic build/first-kernel/gfx11-generic
```

The script prints each public command before executing it and leaves the test,
benchmark, plan, and native artifact reports together in the output directory.

## Put the source under Bazel

An authoring repository can give the same file automated format, lint, plan,
correctness, and benchmark-smoke coverage with one rule:

```starlark title="BUILD.bazel"
--8<-- "generated/examples/getting-started/first-kernel/BUILD.bazel:kernel-library"
```

The built-in AMDGPU execution profile owns the compiler and driver
requirements, `--device=amdgpu` argument, GPU resource serialization, and
stable query tags. The source remains target-independent; the profile chooses
where this test instance runs.

Build the reusable bytecode library:

```shell
bazel build :saxpy
```

Run its generated test suite on a configured AMDGPU machine:

```shell
bazel test :saxpy_test
```

`saxpy_test` performs source lint and canonical-format checks, plans every
benchmark without a device, emits and inspects the generic gfx11 deployment
artifact, executes all correctness cases on AMDGPU, and runs every benchmark
once with no warmups as a smoke test. Full repeated timing remains the explicit
`iree-benchmark-loom` command above; CI should prove that the workload remains
correct and executable without turning shared runners into performance
laboratories.

`loom_kernel_library` links the authored file into reusable `.loombc` and builds
a separate test module that retains the root library's checks. Deployment
linking selects exported production roots such as `@saxpy_f32` and strips
`check.case` and `check.benchmark` records before target compilation. Test code
therefore stays next to the kernel without entering the native artifact.

## Follow the boundary that matters next

You now have one portable source file that formats, verifies, executes across
tail sizes, names a benchmark workload, participates in automated AMDGPU
testing, and remains linkable as a library.

- [Kernels and launch configuration](../guide/kernels-and-launch.md) explains
  the workload, launch, and device-argument split.
- [Checks and benchmarks](../guide/checks-and-benchmarks.md) develops richer
  generators, oracles, comparisons, and sample plans.
- [Benchmark checked work](../workflows/benchmark.md) covers timing modes, data
  reuse, profiling, and interleaved comparisons.
- [Build libraries and binaries with Bazel](../workflows/build-with-bazel.md)
  grows this one-file rule into reusable libraries and deployment artifacts.
- [From source to artifacts](source-to-artifacts.md) follows a larger
  multi-module program through linking, target specialization, command
  programs, and compiler reports.
