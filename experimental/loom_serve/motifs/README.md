# Shared numerical motifs

These sources own reusable numerical operations, not checkpoint keys or model
graphs. Models bind their operands and compose them into kernels and command
programs. The serving runtime has no registry of these operations: the source
JIT resolves reachable definitions from each model's source catalog.

[`tensor/`](tensor) contains projections, attention, convolution, normalization,
rotary transforms, embeddings, diffusion updates, and image layout operations.
Its templates take dimensions, layouts, buffer bindings, and specialization
choices as SSA operands; none reads model configuration. A caller can specialize
those operands without changing the device ABI or recompiling the server.
[`ggml/`](ggml) owns packed weight/activation formats and contraction schedules.
Its reusable bodies likewise take explicit operands; concrete kernel entry
points bind the `ggml.*` capacity configuration.

The implementations retain their numerical and launch contracts. They currently
use AMDGPU execution, including GFX11 wave32 matrix layouts where stated; shared
ownership does not imply arbitrary shapes, storage formats, or targets. Each
definition documents its required extents, alignment, physical layout, aliasing,
and rounding boundaries. Model wrappers choose a valid implementation for their
shape instead of making the helper infer a model from a buffer or global config.

For example, Krea's [projection wrapper](../models/krea/projection.loom) supplies
its dimensions to [`tensor/linear.loom`](tensor/linear.loom). The
[Q5 composition check](ggml/tests/q5_specialization.loom) instantiates the same
packed contraction bodies with independent capacity bounds in one module.
Model-owned checks retain checkpoint and graph assertions; the `tests/` folders
here contain checks of the reusable operations, including independent VM oracles
for convolution and image attention.

Each package publishes a `:sources` filegroup. Model filegroups depend on these
so Bazel runfiles preserve the complete source-JIT closure. This is source data,
not an ahead-of-time kernel artifact. Source catalogs list shared paths relative
to the model directory, such as `../../motifs/tensor/linear.loom`; a standalone
distribution preserves that directory layout. The tensor check scripts resolve
their helper sources relative to their own location and accept an explicitly
built `iree-test-loom` executable through `--checker`.
