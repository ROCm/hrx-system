# XDNA image compatibility fixture

`mul_i32.xdna` is the exact Loom-generated vector multiplication image from the
native driver smoke package. It targets AIE2P Strix Halo
`amd.xdna.strix_halo.17f0_11`, with one compute core at context-relative `(0, 2)`
in a one-column, six-row partition.

The entry `mul_i32` has three 64-byte buffer bindings: read-only lhs, read-only
rhs, and write-only output. It multiplies sixteen little-endian i32 elements,
retaining each product's low 32 bits. The image contains the tile program,
ordered ARRAY initialization, finite CONTROL dispatch, three typed address
relocations, and an output DMA completion wait.

These exact bytes completed native Linux XDNA execution with all sixteen
results checked bit-for-bit and explicit resource teardown. The fixture tests
the compiler/loader ABI without rebuilding the compiler or requiring a device.
It does not establish concurrent or repeated dispatch behavior.

SHA-256:
`2085441fa886afc38ee8e850e6bcc5baeeb08cddaf8fe31c8fa11aa2c06cb3e7`

`mul_i32_npu4.xdna` is a separate canonical Loom image for Strix NPU4
`amd.xdna.strix.17f0_10`. Its entry, context geometry and binding arithmetic
match the contract above. A 270-byte resident worker consumes successive FIFO
records; ARRAY initializes it once and finite CONTROL invocations complete at
the output DMA wait. Binding addresses are cold relocations. The intact image
uses profile ID `0x5354524958000001`, revision 1, rather than Halo's identity.
Both images exercise exact device admission and native lowering in the consumer
tests. Host-side fixture coverage alone does not establish native execution.

Producer: Loom commit `4f5d05550c78b94dbd691baaa3a1beafec8c0d5d`.
SHA-256:
`216a871bb644695b0bad35cb52a0d8a59b529973042a30e4d9abfc58f394ee04`.
