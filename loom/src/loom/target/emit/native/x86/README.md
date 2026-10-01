# Scalar x86 native functions

`loom-compile` can compile ordinary High functions into an ELF relocatable
object for an independent C or C++ linker input:

```sh
loom-compile functions.loom --target=x86:scalar --output=functions.o
cc caller.c functions.o -o caller
```

For example, this function exports the C signature
`uint64_t mix(uint64_t input, uint64_t delta)`:

```loom
func.def public @mix(%input: i64, %delta: i64) -> (i64) {
  %sum = scalar.addi %input, %delta : i64
  %result = scalar.xori %sum, %input : i64
  func.return %result : i64
}
```

The target selects SysV AMD64 by default, independently of the compiler host.
The explicit function spelling is
`abi(object_function, {calling_convention = "sysv"})`. The ELF product accepts
up to six scalar register arguments and one result: `i32`, `i64`, and 64-bit
buffer addresses, indices, or offsets. Unused arguments retain their ABI slots.
The x86 call policy requires inlining, and allocation must produce a spill-free
scalar frame. Other boundaries are diagnosed before an object is returned.

Authored Low uses the same contract. When a logical boundary is not determined
by its physical register carriers, `abi_layout({signature = (...) -> (...)})`
retains that distinction across serialization. A Boolean in GPR32 consequently
cannot be admitted as an ordinary C `uint32_t` parameter by accident.

The architecture provider owns target identity and selects the object emitter.
ABI classification supplies fixed argument locations and reserves RSP before
the shared scheduler and allocator run. Native instruction preparation consumes
their final packets, edge transport, and CFG block ordinals, retaining actual
GPR writes, including implicit writes and allocation-generated moves. This
determines the minimal callee-preserved register envelope before byte encoding.
All returns reach its common restore sequence.

The encoder consumes concrete operands and generated opcode facts. It has no
IR, allocation, or calling-convention decisions to recover. The native object
writer joins code contributions and owns ELF tables. Only emitted bytes survive
between functions; scheduling and allocation scratch is reclaimed immediately.
The returned artifact is independent of all compiler storage.

`test/function.loom-test` checks which registers require preservation, including
writes introduced by allocation transport. Focused writer tests verify save and
restore order and branch destinations from prepared native instructions;
instruction encoding has its own byte-level tests. `callable_link_test` links a
Loom-generated object into a normal host test with independent arithmetic and
memory references; the compiler exits before linking begins.
