# XDNA execution fixtures

`mul_i32.loom` and `add_i32.loom` produce the native images used by the loader
and execution CTS. Their exports multiply or add sixteen little-endian i32
pairs, retaining each result's low 32 bits. Both have three buffer bindings:
read-only lhs, read-only rhs and write-only output, each 64 bytes.

All images use one column and six rows with a resident compute worker. The
establishing invocation loads code and configures the array; its continuation
reuses that state only when the caller separately guarantees its residency.
Independent time-sliced native submissions use the establishing invocation
because context lifetime does not guarantee retained tile state. Each finite
invocation completes at the output DMA wait.
Load ranges splice shared file bytes into final command backing, and declared
binding relocations update both invocation ranges.

Regenerate from the repository root after building `loom-compile`:

```sh
iree-bazel-run //loom/src/loom/tools/loom-compile -- \
  runtime/src/iree/hal/drivers/amd/xdna/image/testdata/mul_i32.loom \
  --root=@mul_i32 --target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11 \
  --format=xdna \
  --output=runtime/src/iree/hal/drivers/amd/xdna/image/testdata/mul_i32.xdna
iree-bazel-run //loom/src/loom/tools/loom-compile -- \
  runtime/src/iree/hal/drivers/amd/xdna/image/testdata/mul_i32.loom \
  --root=@mul_i32 --target=amd.xdna.aie2p:amd.xdna.strix.17f0_10 \
  --format=xdna \
  --output=runtime/src/iree/hal/drivers/amd/xdna/image/testdata/mul_i32_npu4.xdna
```

Generate the addition images with the same commands, replacing `mul_i32` with
`add_i32` in the input, root and output names. The compiler produces intact
independent programs; the execution CTS does not patch instruction encodings.

`sparse_copy_i32.loom` is the compiler-to-loader witness for a preserved unused
middle launch binding. Its Halo image has three dense ABI slots whose kinds are
BUFFER, NONE and BUFFER; dynamic relocations reference only the two live slots.
Regenerate it with:

```sh
iree-bazel-run //loom/src/loom/tools/loom-compile -- \
  runtime/src/iree/hal/drivers/amd/xdna/image/testdata/sparse_copy_i32.loom \
  --root=@sparse_copy_i32 \
  --target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11 \
  --format=xdna \
  --output=runtime/src/iree/hal/drivers/amd/xdna/image/testdata/sparse_copy_i32.xdna
```

Strix and Krackan share the NPU4 execution-profile identity; Halo uses its own
profile. The native CTS selects the matching intact image from the endpoint,
checks exact numerical results with changing inputs, and exercises independent
context lifetimes and caller-owned shared data. A prequeued multiplication and
addition pair uses different binding addresses and immutable slices of one
instruction allocation. Addition consumes multiplication's output without a
host wait or data copy between them. Repeated pairs exercise program rotation
back to multiplication and reuse of both queue slots. Numerical checks
distinguish stale programs, stale bindings and stale output. Host loader tests
exercise segmented source ownership and malformed external metadata through
the actual image reader. They do not decode or emulate tile instructions.
