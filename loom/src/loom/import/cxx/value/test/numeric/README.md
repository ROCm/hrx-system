# Numeric source semantics

These fixtures check the values imported from ordinary C++ expressions,
builtins, and Loom's numeric facades. Each `// ====` case has a focused source
program and exact expected IR, or source diagnostics and `static_assert`
assertions when acceptance is the contract.

| Contract | Fixtures |
| --- | --- |
| Promotions, signedness, shifts, and narrowing updates | `scalar`, `integer_operations`, `integer_widening`, `increments` |
| Width-preserving bit counts and argument evaluation | `integer_intrinsics` |
| Floating type identity, constants, and conversions | `float16`, `bfloat16`, `float8`, `numeric_conversion` |
| Precision at the arithmetic result | `narrow_float_subtraction`, `rounding` |
| Intrinsic result fields and exponent output arguments | `exponent` |
| Standard-library math, builtin precision, diagnostics, and C++ export | `math_builtins`, `math_library`, `math_builtins_invalid`, `math_export` |
| Math permissions and lexical scope | `fastmath`, `scalar_templates`, `logarithms`, `reciprocal` |
| Value and object representation bits | `bitcast`, `bitcast_payloads`, `bitcast_folding`, `constexpr_bitcast` |
| Infinity/huge-value builtins, NaN payloads, and their source admission | `floating_constants`, `floating_constants_invalid` |

For example, testing `input++` requires the expected call operand to be the old
value and later uses to receive the updated value. A test of narrow subtraction
followed by widening checks both operations and their intermediate type. Seeing
only an increment or subtraction mnemonic would not establish either contract.

```sh
iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/value/test/numeric:compiler_test

iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/value/test/numeric:exponent --test_arg=--update
```

The shared [loom-check workflow](../../../../../tools/loom-check/README.md)
owns expectation updates. Review the complete generated values, conversions,
effects, and regions before accepting an update. Numerical execution and
hardware lowering have separate owners; these fixtures stop at verified IR.
