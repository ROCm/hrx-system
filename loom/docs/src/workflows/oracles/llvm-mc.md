# Verify instruction encoding with LLVM MC

LLVM MC is an independent parser, encoder, and disassembler for native
instruction packets. It is a fast oracle when a Loom target adds an instruction,
an operand form, or support for a new GPU before physical hardware is available.

The strongest claim from this workflow is deliberately narrow:

> For this LLVM revision, target triple, processor, feature set, assembly, and
> packet sequence, LLVM emits the same bytes found in the Loom artifact.

Byte agreement proves target selection and native encoding. It does not prove
hardware semantics, numerical correctness, scheduling quality, or performance.

## Qualify the LLVM binary first

Record the complete tool identity and confirm that the processor is actually
known:

```shell
llvm-mc --version
llvm-mc --triple=amdgcn-amd-amdhsa --mcpu=help
```

The selected processor must appear in the supported-processor list. An “unknown
processor” warning, ignored `--mcpu`, fallback target, or unsupported-instruction
diagnostic invalidates the oracle. A system LLVM and a toolchain-bundled LLVM
with similar version strings may carry different AMDGPU target tables, so the
evidence records the resolved paths and hashes of `llvm-mc`, `llvm-objdump`, and
`llvm-readobj` as well as their version output.

The LLVM command guides define the public contracts for
[`llvm-mc`](https://llvm.org/docs/CommandGuide/llvm-mc.html),
[`llvm-objdump`](https://llvm.org/docs/CommandGuide/llvm-objdump.html), and
[`llvm-readobj`](https://llvm.org/docs/CommandGuide/llvm-readobj.html).

## Emit the Loom artifact and provenance

Compile the exact kernel entry and retain both compiler and object metadata:

```shell
loom-compile kernel.loom \
  --root=@kernel \
  --backend=amdgpu-hal \
  --target=gfx1170 \
  --output=kernel.hsaco \
  --artifact-manifest=details \
  --emit-artifact-manifest=kernel.manifest.json \
  --compile-report=details \
  --compile-report-output=kernel.report.json

llvm-readobj \
  --file-headers \
  --symbols \
  --notes \
  kernel.hsaco \
  >kernel.readobj.txt

llvm-objdump \
  --disassemble \
  --mcpu=gfx1170 \
  kernel.hsaco \
  >kernel.disasm.txt
```

The manifest and object headers establish the emitted target and exported
symbol. The compile report establishes the selected compiler mechanism. The
disassembly supplies the exact packets and register operands that reached the
code object. Preserve the unmodified HSACO; rendered text alone loses the byte
source and object metadata.

## Reduce the question to selected packets

Create a small `oracle.s` containing only the instructions whose encoding is in
question. Keep the final artifact's register operands, modifiers, literals, and
order:

```asm
.text
v_cvt_pk_f32_fp8_e32 v[6:7], v4.l
v_wmma_f32_16x16x16_bf16 v[24:31], v[8:11], v[16:19], 0
```

Assemble it once for the human-readable encodings and once as an object:

```shell
llvm-mc \
  --triple=amdgcn-amd-amdhsa \
  --mcpu=gfx1170 \
  --show-encoding \
  oracle.s \
  >oracle.encodings.txt

llvm-mc \
  --triple=amdgcn-amd-amdhsa \
  --mcpu=gfx1170 \
  --filetype=obj \
  oracle.s \
  -o oracle.o

llvm-objdump \
  --disassemble \
  --mcpu=gfx1170 \
  oracle.o \
  >oracle.disasm.txt
```

`--show-encoding` exposes LLVM's packet bytes directly. The object round trip
checks that the same bytes decode to the expected instructions under the same
processor. Compare those ordered bytes with the corresponding range in
`kernel.hsaco`, not with a hand-transcribed hexadecimal word.

For example, LLVM prints packet bytes in address order:

```text
v_cvt_pk_f32_fp8_e32 v[6:7], v4.l
  encoding: [0x04,0xdd,0x0c,0x7e]
```

An object dumper may render the same little-endian bytes as the word
`7E0CDD04`. That is agreement, not reversal of the packet. Retaining both forms
prevents byte-order confusion in reviews and generated checks.

## Choose the smallest independent boundary

The oracle source includes enough context to answer the question and no more:

| Question | Oracle boundary |
| --- | --- |
| One operand form or modifier | One complete instruction packet. |
| A literal or extension word | The instruction and every encoded extension word. |
| A hazard-sensitive sequence | The exact ordered instructions from the final artifact. |
| A branch or symbol reference | A relocatable object with labels and its relocation records. |
| Kernel ABI or target metadata | The emitted HSACO plus `llvm-readobj` headers, notes, and symbols. |

Isolated packets deliberately omit register allocation and scheduling context.
They cannot explain why a compiler chose those registers, whether dependencies
need intervening waits, or whether the sequence occupies a hot loop. Sequence
evidence comes from the final Loom disassembly; LLVM MC verifies that the
selected textual sequence maps to the expected bytes.

Relocations, labels, kernel descriptors, and metadata require object-level
evidence. Copying a branch or symbolic operand into a label-free packet can
change or erase the property being verified.

## Keep the evidence classes separate

| Evidence | Supported conclusion | Next required gate |
| --- | --- | --- |
| Loom report selects the intended instruction family. | The compiler chose the expected mechanism. | Inspect the emitted artifact. |
| Loom disassembly contains the intended packet sequence. | The artifact carries that native sequence. | Independently assemble it. |
| LLVM MC emits identical bytes. | LLVM and Loom agree on target encoding. | Execute a numerical witness or semantic model. |
| Hardware or a qualified semantic model passes. | The sequence implements the tested semantics on that target. | Measure a production boundary. |
| Controlled physical timing wins. | The selected program improves that matched workload. | Integrate and measure the complete cut. |

An LLVM MC success is especially valuable before hardware exists because it
can reject parser, operand, target-selection, and packet-layout mistakes early.
The absence of hardware remains explicit. Modeled occupancy, a valid ELF, and
byte-perfect instructions are compiler evidence, not a synthetic performance
claim.

## Preserve a qualification packet

A durable packet contains:

- the minimized Loom witness and exact compile command;
- the Loom compiler identity, report, manifest, HSACO, and disassembly;
- `oracle.s`, LLVM path and version evidence, encodings, object, and
  disassembly;
- a byte-by-byte comparison naming every selected packet;
- the numerical or modeled execution result when one exists; and
- an explicit statement of the first unavailable evidence class.

That last statement matters. “Encoding qualified; physical hardware
unavailable” is a useful completed result. It prevents a future agent from
rerunning the same encoder archaeology while keeping hardware correctness and
performance as real, visible gates.
