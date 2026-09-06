# IREE Deterministic Math

`iree/math` provides compact numerical primitives whose result mappings can be
shared by the runtime and compiler. It is not a host `libm` wrapper: each
public function names one stable operation, format, and numerical contract so
constant evaluation and dynamic execution can agree across hosts.

The initial f32 operations model machine-shaped leaves used by ML target
recipes:

```c
float y0 = iree_math_exp2_f32_approx(x);
float y1 = iree_math_log2_f32_approx(x);
float y2 = iree_math_reciprocal_f32_approx(x);
float y3 = iree_math_rsqrt_f32_approx(x);
float y4 = iree_math_sqrt_f32_approx(x);
float y5 = iree_math_sin_turns_f32_approx(x);
float y6 = iree_math_cos_turns_f32_approx(x);
```

`approx` does not grant the implementation a tolerance. It identifies one
frozen IREE result mapping selected for a machine-like accuracy and denormal
contract. Changing any result payload requires a new named contract and, when
encoded in bytecode, a new selector. The public headers document each mapping's
NaN, infinity, signed-zero, denormal, and accuracy behavior.

## Execution Profile

Leaf functions contain no allocation, status handling, synchronization,
`errno`, or floating-environment transitions. The caller establishes the
documented non-trapping rounding and denormal profile once around its execution
scope. This lets a VM invocation or compiler evaluation segment pay for at most
one host FPU transition instead of one transition per scalar operation.

Package-owned compile options prevent an embedding application's unrestricted
fast-math flags from changing the frozen mappings. Contraction remains enabled
where exhaustive testing proves it bit-invariant.

## Organization

Operations are grouped by algorithm family and provenance:

- `exponential.h` owns base-two exponential and logarithm.
- `roots.h` owns reciprocal, reciprocal square root, and square root.
- `trigonometry.h` owns turns-based sine and cosine.

All clusters form one `:math` build target. Keeping their implementations in
separate translation units preserves readable ownership and ordinary
dead-stripping without multiplying build graph actions. `float_bits.h` is a
private representation helper, not another public API.

New operations require a concrete compiler or runtime consumer, an explicit
payload contract, independent accuracy evidence, cross-platform bit vectors,
and measured latency and closure size. Target-specific approximations belong
with their target unless they are selected as the portable IREE mapping.
