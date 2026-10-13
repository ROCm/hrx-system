# Reusable numerical motifs

`ggml/` implements contractions and quantization over GGML block formats.
`tensor/` supplies attention, linear algebra, convolution, normalization,
quantization, random generation and image transforms.

Helpers receive dimensions, layouts, storage choices and operands through SSA
values or template arguments. Concrete model leaves supply configuration,
targets and launch geometry. This keeps a reusable projection independent of a
particular model's configuration namespace while allowing the JIT to specialize
the entire call chain.

The tests beside each family use actual production helpers. They include scalar
oracles, exact integer contractions, and comparisons with smaller cooperative
or single-wave implementations. Hardware and access-sanitizer profiles are
separate executions; performance measurement uses optimized unsanitized builds.

Some larger shape sweeps are authored as directly selectable `.loom` cases.
For example, the tiled BF16 projection case binds compiler dimensions and the
matching tensor sample explicitly:

```sh
iree-test-loom experimental/loom_serve/motifs/tensor/tests/linear_tiled.loom \
  --library=experimental/loom_serve/motifs/tensor/linear.loom \
  --library=experimental/loom_serve/motifs/tensor/linear_tiled.loom \
  --device=amdgpu --target=amdgpu:gfx1151 \
  --case=@linear_bf16_6144_6144 --sample=0 \
  --config=projection_test.rows=16 \
  --config=projection_test.inputs=6144 \
  --config=projection_test.outputs=6144 \
  --config=projection_test.row_tile_group_size=8 \
  --config=projection_test.column_major=0
```

The target must match the execution device. The model JSON provider list is a
linking input, not a promise to compile every unused helper in this directory.
Tests and compiler reports can select narrower roots to inspect one mechanism.

The shared-head attention adaptations retain their upstream attribution and
license in [LICENSE.krea2-hrx](tensor/LICENSE.krea2-hrx).
