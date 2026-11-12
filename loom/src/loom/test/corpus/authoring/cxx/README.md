# Loom C++ Authoring Examples

These programs show complete kernels and kernel libraries authored through the
optional Loom C++ importer. They keep algorithms, configuration, target
selection, schedules, and independent correctness checks in readable C++ while
publishing ordinary Loom test modules for target packages to qualify.

This is a curated authoring corpus rather than a general C++ test directory.
Focused source-language behavior and diagnostics live with the importer;
instruction-selection checks and execution matrices live with their target.
Compiler benchmarks use private copies of their inputs, so editing a scenario
or oracle here never changes the source bytes measured by a benchmark.

## Programs

| Program | Authoring contract |
| --- | --- |
| `flash_attention.cxx` | Online softmax attention with shared key/value tiles, subgroup reductions, partial tiles, and an independent f32 oracle. |
| `q4k_q8_swiglu.cxx` | Serialized Q4_K and Q8_1 records, vectorized decode/dot/reduction, runtime configuration, and a scalar record oracle. |
| `iq4xs_gate_up.cxx` | IQ4_XS gate/up projection with configurable asynchronous staging, structured schedules, and serialized-format checks. |
| `nvfp4_matrix*` | One NVFP4 matrix root linked with interchangeable C++ and authored-Loom providers selected by subgroup width. |
| `q8s32_specialization*` | One Q8S32 root linked with interchangeable C++ and authored-Loom providers selected by declared vector shape, target, and dynamic configuration predicates. |

The package deliberately publishes modules without choosing a device. Target
packages attach compile targets, execution profiles, configuration variants,
and access instrumentation to the same module. This keeps the C++ program
reusable across CPU, GPU, NPU, and future target qualification wherever its
operations are supported.
