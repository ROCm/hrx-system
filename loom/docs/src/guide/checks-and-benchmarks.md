# Checks and benchmarks

Loom correctness cases are programs, not runner metadata. A `check.case`
constructs values, selects deterministic samples, invokes kernels or reference
oracles, and states expectations in the same linked symbol graph as the code it
tests. A `check.benchmark` names a slice of that proven workload rather than
recreating setup in a second harness.

In this chapter, you will learn:

- how one case expands into deterministic concrete samples;
- how tensors cross from host-side materialization into kernel buffer bindings;
- where expected values and comparison policy belong;
- why requirements, configuration, and case parameters are different domains;
  and
- how benchmark records preserve workload identity while tools own timing
  policy.

## A case is an isolated SSA program

[`check.case`](../reference/dialects/check/ops/case.md) owns one isolated region
terminated by [`check.return`](../reference/dialects/check/ops/return.md). Values
inside the region flow through five semantic stages:

1. **Requirements** decide whether the environment can execute the case.
2. **Parameters** expand the case into concrete samples.
3. **Value sources** materialize inputs, outputs, fixtures, and expected values.
4. **Invocations** call an oracle provider or launch one or more kernels.
5. **Expectations** compare resulting values or require structured runtime
   events.

The stages are an ownership model, not mandatory source blocks. Ordinary SSA
dependencies determine the actual order. A case may launch several kernels,
feed one output into another launch, compare several results, and write failure
artifacts without moving that schedule into host test code.

`public` makes a case visible to test discovery:

```loom
check.case public @decode_tail_case {
  // Requirements, samples, values, work, and expectations.
  check.return
}
```

Cases and benchmark records are test-only symbols. Production motifs and
kernels remain in production libraries; a sibling test module owns the records
that exercise them.

## Keep the production symbol and test program separate

The checked example for this chapter places a small production kernel in one
module:

```loom title="kernel.loom"
--8<-- "examples/guide/checks-and-benchmarks/kernel.loom"
```

Its test module declares the exact dependency, constructs a deterministic seed
sweep, launches the kernel, and compares the mutated output tensor:

```loom title="tests.loom"
--8<-- "examples/guide/checks-and-benchmarks/tests.loom"
```

The [`kernel.decl`](kernels-and-launch.md#declare-and-launch-the-complete-contract)
is part of the test module's source contract. Supplying the production archive
can satisfy that declaration; it cannot make an undeclared launch valid. The
test archive therefore verifies and plans independently while the production
archive remains free of check records and private wrapper entries.

This is the scalable library shape: `motif/` and `kernel/` packages publish
reusable archives, while their sibling `test/` packages own private wrappers,
cases, and benchmark records. The public Bazel `loom_test_library` rule formats
the authored source, plans its benchmarks, and expands target execution
profiles without changing the Loom program.

## Parameters define the sample space

Parameter operations produce ordinary scalar SSA values and a deterministic
sample plan:

| Operation | Sample domain |
| --- | --- |
| [`check.param.choice`](../reference/dialects/check/ops/param-choice.md) | An explicit set of integer or index values. |
| [`check.param.range`](../reference/dialects/check/ops/param-range.md) | A linear or power-of-two sampling of a closed interval. |
| [`check.param.seed`](../reference/dialects/check/ops/param-seed.md) | A fixed count of deterministic random-generator seeds. |

Several parameter operations form a Cartesian sample space. The planner assigns
stable ordinals in source order, and runners may select one ordinal or apply an
explicit maximum when the full product is not the intended evidence.

The optional `name` is the stable key used by benchmark assignments and
structured reports:

```loom
%row_count = check.param.choice values([17, 64, 193]) name("row_count") : index
%seed = check.param.seed base(1000) count(4) name("seed") : i64
```

Parameter values are runtime case inputs. They do not create compiled artifact
versions. A `config.decl` instead states a value fixed for one specialization;
a command-line timing flag states measurement policy. Keeping these domains
separate allows one compiled candidate to run many case samples and lets the
same named workload be measured under several policies.

## Materialize reproducible values in source

Value-source operations make a case self-contained:

| Source | Intended use |
| --- | --- |
| [`check.literal`](../reference/dialects/check/ops/literal.md) | One typed scalar reproducer or expected value. |
| [`check.generate.fill`](../reference/dialects/check/ops/generate-fill.md) | A tensor or scalar filled with a static payload. |
| [`check.generate.iota`](../reference/dialects/check/ops/generate-iota.md) | Deterministic arithmetic sequences, optionally periodic. |
| [`check.generate.random.uniform`](../reference/dialects/check/ops/generate-random-uniform.md) | Seeded values over a static numeric interval. |
| [`check.file.read.npy`](../reference/dialects/check/ops/file-read-npy.md) | A typed external fixture whose exact bytes matter. |

Generators accept dynamic tensor dimensions from case parameters:

```loom
%input = check.generate.iota offset(1.0) step(1.0) : tensor<[%row_count]xf32>
%output = check.generate.fill value(0.0) : tensor<[%row_count]xf32>
```

A check `tensor` is a typed materialized value. Passing it to a kernel buffer
argument binds its storage for that launch; writes through an output binding are
visible to later invocations and expectations in the case. The tensor retains
shape and element type for planning and comparison while the kernel continues
to receive the opaque `buffer` ABI it declared.

[`check.tensor.view`](../reference/dialects/check/ops/tensor-view.md) creates a
typed alias over a byte subspan when several logical bindings intentionally
share one fixture. Alias that is not part of the production workload should not
be manufactured merely to reduce test allocation.

Random generation is reproducible because the seed is explicit SSA input. In
the checked copy case, `%input` and `%expected` use the same sampled seed and
therefore materialize the same values. A failed sample report carries that seed
assignment directly; no hidden process RNG state is needed to reproduce it.

## Make the oracle and comparison policy explicit

Expected values can be generated directly, read from a fixture, produced by an
earlier launch, or returned by a configured
[`check.oracle.call`](../reference/dialects/check/ops/oracle-call.md) provider.
The oracle operation names its provider, callee contract, inputs, result types,
and provider-specific attributes. The runner or embedding supplies the
provider implementation; unavailable providers are planning failures unless a
case requirement deliberately excludes that environment.

The expectation operation records what equality means:

| Expectation | Contract |
| --- | --- |
| [`check.expect.equal`](../reference/dialects/check/ops/expect-equal.md) | Typed value equality. |
| [`check.expect.bitwise`](../reference/dialects/check/ops/expect-bitwise.md) | Exact payload-bit identity. |
| [`check.expect.close`](../reference/dialects/check/ops/expect-close.md) | Floating-point comparison with explicit absolute tolerance, relative tolerance, and NaN policy. |
| [`check.expect.shape`](../reference/dialects/check/ops/expect-shape.md) | Exact static and dynamic dimensions. |
| [`check.expect`](../reference/dialects/check/ops/expect.md) | A named pluggable validator with typed actual and expected values. |
| [`check.expect.event`](../reference/dialects/check/ops/expect-event.md) | A provider-specific structured runtime event such as a sanitizer report. |

Tolerance belongs in source because it is part of the operation's numeric
contract. A benchmark or CI invocation may select a case, but it must not make a
failing result pass by changing comparison thresholds out of band.

[`check.file.write.npy`](../reference/dialects/check/ops/file-write-npy.md) can
capture an actual value always or only on failure. This is an evidence output,
not the oracle: checked source still states how success is decided.

## Requirements explain unavailable evidence

[`check.requires`](../reference/dialects/check/ops/requires.md) names a provider
capability needed to execute the case. An unmet requirement skips the case with
a structured reason. [`check.skip_if`](../reference/dialects/check/ops/skip_if.md)
names an exceptional environment predicate that makes execution invalid or
unrepresentative.

Requirements do not specialize the program and a skip is not a pass. Target
requirements on function or kernel providers decide whether an implementation
is valid during compilation. Check requirements decide whether this environment
can produce the requested runtime evidence. Reports keep skipped, passed,
failed, and unplannable samples distinct.

Use positive requirements for the normal capability contract. `skip_if` is for
exceptional conditions such as a fixture exceeding available device memory,
not as a growing architecture allowlist.

## A benchmark names proven work

[`check.benchmark`](../reference/dialects/check/ops/benchmark.md) references one
case and optionally binds named parameters:

```loom
check.benchmark<@copy_i32_case> @copy_i32_sweep

check.benchmark<@copy_i32_case> @copy_i32_seed_1002 {seed = 1002}
```

The first record selects the case's complete seed sweep. The second names one
representative sample. Both reuse its value generation, kernel launch,
expectation, and correctness contract. Several benchmark names that resolve to
the same case assignments identify the same physical work item and can be
deduplicated by the benchmark planner.

The record does not mark a timer region or encode warmup, duration, batching,
profiling, cache reuse, or target selection. The benchmark tool chooses whether
to measure the complete case end to end or a prepared dispatch-completion
schedule, and it records that policy with the result. Correctness runs before
timing; a failed sample produces failure evidence instead of a score.

This division keeps source workload identity stable. One benchmark record can
be compared across targets, sanitizer configurations, compiler versions, and
measurement policies without cloning the program or silently changing its
inputs.

## Keep every failure in its owning domain

| Symptom | Contract to inspect |
| --- | --- |
| A case unexpectedly has thousands of samples | Parameter generators formed a larger Cartesian product than intended. |
| A benchmark assignment is ignored or rejected | The parameter lacks the matching stable `name` or the value is outside its domain. |
| A kernel symbol exists in a library but the launch is unresolved | The test module omitted the exact `kernel.decl`. |
| Expected random values cannot be reproduced | Seed state was hidden outside the check program. |
| A floating-point result passes one harness and fails another | Tolerance or NaN policy was not owned by the checked source. |
| A target-specific case appears green elsewhere | An unmet requirement was counted as a pass instead of a structured skip. |
| A benchmark has a number but correctness failed | The harness timed work without enforcing the case gate. |

[Test correctness](../workflows/test-correctness.md) gives the concise
`iree-test-loom` commands for case and sample selection, devices, sanitizers,
and structured reports. [Benchmark checked work](../workflows/benchmark.md)
covers dry-run planning, end-to-end and dispatch-completion measurement,
interleaved comparisons, profiling, and artifact bundles.

Continue with command programs: the same kernel launches and typed resources
can become a reusable model subgraph while checks continue to exercise the
program through its public schedule.
