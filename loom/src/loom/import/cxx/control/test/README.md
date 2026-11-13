# C++ control-flow semantics

These tests import C++ into verified High IR. Each case pairs a small source
construct with its complete expected program, including operand wiring,
structured regions, state carried across joins, and observable effects.

| Contract | Fixtures |
| --- | --- |
| Conditional values and early branch returns | `conditionals` |
| Contextual Boolean conversion, short circuiting, and guarded effects | `logical` |
| Condition-variable initialization, scope, mutation, and destruction | `condition_declarations` |
| Compile-time branch selection and runtime initializer effects | `constexpr_selection` |
| Pre-test and post-test loops, recurrences, and mutable conditions | `loops` |
| Bounded array and class range iteration through checked begin/end operations | `range_loops` |
| Counted intervals, integer wrap boundaries, and native address domains | `counted_loops`, `address_counted_loops` |
| Unbraced loop bodies and counted-loop admissibility | `loop_bodies` |
| Loop schedule operands retained from source | `loop_schedules` |
| Break, continue, nested exits, and scalar/vector/pointer state | `loop_exits` |
| Returns across loops, aggregate results, and memory effects | `structured_returns` |
| Switch selection, fallthrough, and interaction with loop exits | `switches` |
| Source rejection and deep-expression admission | `control_invalid`, `deep_conditionals` |
| Source construction, partial records, independent copies, and mandatory first loop bodies | `construction`, `construction_ssa` |
| Semantic carrier families, native storage preservation, and unavailable-value diagnostics | `construction_carriers`, `construction_native`, `construction_invalid` |
| Cleanup of a proven counted recurrence | `condition_loop_cleanup` |

Related source cases share a fixture through `// ====`. `// ----` introduces
the exact expected output. Diagnostic cases use source annotations; constexpr
acceptance cases use `static_assert` for the evaluated result. A successfully
verified module by itself does not establish runtime expression semantics.

```sh
iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/control/test:compiler_test

iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/control/test:logical --test_arg=--update
```

The shared [loom-check workflow](../../../../tools/loom-check/README.md)
generates and validates canonical expectations. Reviewing an update means
checking which region owns each effect, which values reach each join, and which
condition or step executes after an exit. Target lowering and program execution
have their own test owners.
