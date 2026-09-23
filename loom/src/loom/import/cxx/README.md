# C and C++ source import

The optional native importer translates a source translation unit into a
verified Loom module. It uses the standalone cxx parser and semantic frontend;
the importer has no LLVM or Python runtime dependency. The result is ordinary
editable High IR, retaining source names, locations, structured loops, shared
helpers, and concrete template instances.

Enable `--config=loom-importer-cxx` with Bazel or `-DLOOM_IMPORT_CXX=ON` with
CMake. The importer is disabled by default. Its C++23 and exception requirements
are confined to the frontend implementation.

```sh
iree-bazel-run --config=loom-importer-cxx \
  //loom/src/loom/tools/loom-import-cxx -- \
  --I=include --std=c++26 --root=attention attention.cxx
```

`loom-import-cxx` imports one translation unit, runs canonicalization, common
subexpression elimination and dead-code elimination, and prints Loom text.
`--cleanup=false` exposes the direct import. `--to=bc --output=module.loombc`
produces normal Loom bytecode for the existing compilation and linking tools.

## Explicit 16-bit floating-point types

`<stdfloat>` provides `std::float16_t` (IEEE FP16) and `std::bfloat16_t` (BF16).
These are numeric types: initializers, arithmetic, helper arguments, record
fields, vector elements, and pointed-to storage retain the selected format.

```cpp
#include <stdfloat>
using std::bfloat16_t;

bfloat16_t scale(bfloat16_t value) {
  bfloat16_t factor = 5.0f;
  return value * factor;
}

void convert(const float* input, bfloat16_t* output) {
  output[0] = input[0];
}
```

The multiplication imports as `scalar.mulf ... : bf16`; the assignment from
`float` imports as `scalar.fptrunc ... : f32 to bf16`. The same types can be
elements of explicit vectors or `loom::type::view` values. Target compilation
selects native instructions or legalization appropriate to that format.

The underlying spellings `_Float16` and `__bf16` also work directly in C and
C++. Both occupy two bytes with two-byte scalar alignment. Literal suffixes
`f16`/`F16` and `bf16`/`BF16` select the corresponding type. The C++ feature
macros `__STDCPP_FLOAT16_T__` and `__STDCPP_BFLOAT16_T__` are defined.
Narrow constant conversions and arithmetic round to nearest with ties to even,
including subnormal values. Floating-point operation flags and the selected
target's math policy still govern compiled arithmetic.

FP16 and BF16 have different ranges and precision. Mixed arithmetic requires
an explicit cast; conversion preserves the numeric value subject to destination
rounding. For example, `static_cast<std::float16_t>(value)` on a BF16 value
imports an exact extension to F32 followed by truncation to FP16. Bit
reinterpretation remains an explicit `__builtin_bit_cast` operation.

The header contains only the two aliases and include guards, with no transitive
includes. Embedded lookup and explicit include directories expose the same
header. Mathematical operations remain in the separately included
`<loomcxx/scalar.h>`.

## Explicit 8-bit floating-point types

`<loomcxx/numeric.h>` provides `loom::type::float8_e4m3fn_t` and
`loom::type::float8_e5m2_t`. Both occupy one byte with one-byte scalar alignment.
They are numeric types for typed storage, explicit vectors, record fields and
ordinary C++ conversions:

```cpp
#include <loomcxx/numeric.h>
#include <stdfloat>
namespace loomt = loom::type;

void quantize(const float* input, loomt::float8_e4m3fn_t* output,
              unsigned index) {
  output[index] = input[index];
}

std::bfloat16_t dequantize(loomt::float8_e4m3fn_t value) {
  return value;
}
```

The store retains an `f8E4M3` element type and one-byte stride. Quantization
imports as `scalar.fptrunc`; dequantization imports as `scalar.extf` to BF16.
The source types expose Loom's existing format contracts:

| C++ type | High element | Conversion policy |
| --- | --- | --- |
| `float8_e4m3fn_t` | `f8E4M3` | Ties to even; finite overflow and infinities saturate to ±448; NaNs remain NaNs. |
| `float8_e5m2_t` | `f8E5M2` | IEEE ties to even, including infinity and NaN; largest finite magnitude is 57344. |

Both preserve signed zero and subnormals. Same-format arithmetic retains the
format. Mixing the two FP8 types requires an explicit numeric cast, which
extends exactly to F32 and then rounds to the destination. Wider FP16, BF16,
F32 or F64 operands select that wider type. List initialization rejects
narrowing conversions.

The fundamental spellings `__float8_e4m3fn` and `__float8_e5m2` also work in C.
Ordinary floating literals initialize them; no FP8 literal suffix is required.
These spellings identify signed-zero formats, not the distinct FNUZ encodings.
The optional header contains only aliases, documentation and guards, with no
transitive includes. Scalar math templates admit these types when used without
adding per-format declarations to every import.

## Numeric vector conversion

`__builtin_convertvector(value, DestinationType)` converts each lane numerically
while preserving the number of lanes. It works in C and C++, needs no header,
and accepts template-dependent destination types and constant expressions.

For example, load packed FP8 values, compute in F32, and round back to FP8:

```cpp
#include <loomcxx/numeric.h>
using Float8x4 =
    loom::type::float8_e4m3fn_t __attribute__((ext_vector_type(4)));
using Float4 = float __attribute__((ext_vector_type(4)));

Float8x4 scale(Float8x4 value, float factor) {
  auto wide = __builtin_convertvector(value, Float4);
  return __builtin_convertvector(wide * factor, Float8x4);
}
```

The imported computation retains vector operations throughout:

```loom
func.def public @scale(%value: vector<4xf8E4M3>, %factor: f32) -> (vector<4xf8E4M3>) {
  %wide = vector.extf %value : vector<4xf8E4M3> to vector<4xf32>
  %factors = vector.splat %factor : vector<4xf32>
  %scaled = vector.mulf %wide, %factors : vector<4xf32>
  %result = vector.fptrunc %scaled : vector<4xf32> to vector<4xf8E4M3>
  func.return %result : vector<4xf8E4M3>
}
```

Integer conversions retain source signedness: widening `unsigned char` lanes
uses `vector.extui`, widening `signed char` uses `vector.extsi`, and conversion
to floating point selects `vector.uitofp` or `vector.sitofp`. Float-to-integer
conversion truncates toward zero; the input must be finite and its truncated
value representable in the destination integer type. Constant evaluation
rejects values outside that domain.

Distinct floating formats with equal storage widths, such as FP16 and BF16 or
the two FP8 types, convert through an exact F32 vector intermediate. Numeric
conversion keeps every lane and rounds to the destination format. C-style
vector casts and `__builtin_bit_cast` continue to reinterpret equal-sized
objects. The importer leaves vector legalization and instruction selection to
the target compiler.

## Packed scalar and vector bit casts

`__builtin_bit_cast(DestinationType, value)` reinterprets equal-width scalar
and vector payloads. A packed word can become byte, FP8, or FP16 lanes for
vector computation, then return to a scalar word for storage:

```cpp
using Bytes4 = unsigned char __attribute__((ext_vector_type(4)));

unsigned increment_bytes(unsigned word) {
  auto bytes = __builtin_bit_cast(Bytes4, word);
  return __builtin_bit_cast(unsigned, bytes + Bytes4{1, 2, 3, 4});
}
```

Lane zero occupies the least-significant bits. Each byte addition wraps in its
own lane. The imported value stays in SSA: a scalar endpoint uses a one-lane
`vector.from_elements` or `vector.extract` around `vector.bitcast`. The target
compiler can realize those representation changes as register aliases. Floating
bit casts preserve the payload, including signed zero and NaN encodings; use
`__builtin_convertvector` for numeric conversion.

The same operation supports scalar-to-scalar and vector-to-vector values.
Pointers, aggregates and padded vectors produce source diagnostics. Runtime
casts between `bool` and byte-sized values are also rejected: Loom represents
`bool` as an `i1` predicate rather than its C++ object byte.

### Constant bit casts

Bit casts also work in `constexpr` initializers and `static_assert`. Packed
weights, scales and codebooks can retain their published object encodings in
source while supplying typed values to ordinary arithmetic:

```cpp
using Fp8x4 = __float8_e4m3fn __attribute__((ext_vector_type(4)));
using Float4 = float __attribute__((ext_vector_type(4)));
constexpr Fp8x4 kScale = __builtin_bit_cast(Fp8x4, 0x403c3830u);
static_assert(kScale[0] == 0.5f && kScale[3] == 2.0f);

void apply_scale(const Float4* input, Float4* output) {
  *output = *input * __builtin_convertvector(kScale, Float4);
}
```

The direct import retains typed FP8 constants, a vector extension and a vector
multiply. Ordinary cleanup folds the extension to F32 constants; there is no
runtime decoding of the packed word. Fixed unpadded vectors can also regroup
larger constant vectors, such as four integer words into a sixteen-element FP8
table.

Constant evaluation retains object bits for FP16, BF16, both FP8 formats, F32
and F64. Copies, same-type conversions, unary sign changes and template
arguments preserve signed zero and NaN payloads. Numeric arithmetic and
conversion apply the destination format's rounding rules.

```cpp
constexpr float kPayload = __builtin_bit_cast(float, 0x7f812345u);
static_assert(__builtin_bit_cast(unsigned, +kPayload) == 0x7f812345u);
static_assert(__builtin_bit_cast(unsigned, -kPayload) == 0xff812345u);
float payload() { return kPayload; }
```

Finite values import as ordinary numeric constants. Exact NaN values import as
integer constants followed by `scalar.bitcast`, retaining their representation
through Loom transformations. Numeric-only config and check metadata cannot
carry NaN payloads and diagnose such values; an integer configuration value
can carry the encoding for a bit cast in the function body.

The constant evaluator admits integral and enum scalars up to 64 bits and
unpadded vectors of those integers or supported floats. Scalar `bool` object
bytes must encode zero or one. Pointer, aggregate, long-double, padded-vector
and packed-Boolean-vector representations are rejected during constant
evaluation because their object layout is outside this encoding contract.

## Compiler tests

When the importer is enabled, `loom-check` accepts `.cxx-test` files through the
same case, diagnostic, comparison, and update machinery as `.loom-test` files.
Each case is an independent translation unit. Roundtrip imports C++ and checks
canonical Loom output; pass, verify, format, emit, and report modes consume that
same module.

Files group cases with one shared `RUN` mode declared at the top.

```cpp
// RUN: with-checks pass canonicalize,cse,dce
// INPUT: cxx root=twice std=c++23
int twice(int value) { return value * 2; }
// ----
// CHECK: func.def public @twice*
// CHECK: *scalar.shli*
// CHECK-NOT: *scalar.muli*
```

Diagnostic fixtures use `verify`:

```cpp
// RUN: verify
// INPUT: cxx
long distance(int* left, int* right) {
  // ERROR@+1: LOWERING/059 "pointer arithmetic"
  return left - right;
}
```

Exact goldens occupy the `// ----` section without `with-checks`. `loom-check --update file.cxx-test`
writes that section, inserting it when absent, and preserves the source. A subsequent run checks the updated output. Hand-authored
CHECK patterns are preserved by updates. Diagnostic annotations use the normal
structured ERROR/WARNING/REMARK syntax; the JSON report includes suggested
annotation edits when expectations differ.

The first case's `RUN` and `INPUT` establish file defaults. `INPUT` selects the
source format and options independently of `RUN`; a later `INPUT` overrides
the default for that case. With no `INPUT`, the filename selects C++ with the
importer's normal C++26/LP64 defaults. `--input-format=cxx` selects C++ for
stdin or another filename. `--list-input-formats` reports the formats linked
into that binary. An importer-disabled binary rejects `.cxx-test` and explicit
`cxx` input.

Options use whitespace-separated `key=value` tokens. Single or double quotes
preserve spaces within a token, and quotes/backslashes can be escaped inside
quotes. The C++ provider accepts:

| Option | Meaning |
| --- | --- |
| `std=c++23` | Source language standard, including supported C spellings. |
| `triple=...` | Source ABI triple. |
| `data-model=lp64` | `lp64`, `llp64`, or `ilp32` integer and pointer widths. |
| `root=entry` | Exported source function; repeat for multiple roots. |
| `I=include` | User include directory, relative to the test file; repeatable. |
| `isystem=include` | System include directory, relative to the test file; repeatable. |
| `D=COUNT=8` | Macro definition; repeatable. `D=ENABLED` defines it to `1`. |
| `approximate-functions=true` | Permit approximate mathematical functions. |
| `builtin-includes=false` | Use explicit include paths without embedded headers. |

Source locations use physical, case-relative lines. `#line` controls the presumed
values of `__LINE__` and `__FILE__`; locations and diagnostics still refer to the
admitted source bytes. Builtin macro replacements retain the range of the macro
use, independently of the replacement spelling: the value `100` from `__LINE__`
points at the eight characters of `__LINE__`. Nested expansions retain the
frontend's macro-body use site or argument-token range.
Stringized and pasted tokens use the macro invocation's range; their generated
spelling need not occur contiguously in the source.

Header locations retain their own filenames and coordinates; an ERROR in the
main test cannot match a header diagnostic at a coincident line. Source snapshots
remain available after import for pass diagnostics, including headers.
`--source-prefix-map` changes displayed
names without changing relative include lookup; mappings that merge distinct
admitted source identities are rejected. Full-line test separators and
directives are reserved by the shared envelope, including inside multiline
source constructs. Other C++ comments, macro continuations, and raw strings
remain untouched.

The importer-owned integration suite is
`//loom/src/loom/import/cxx/tooling/test:test`. Authored source lowering goldens
run through `//loom/src/loom/import/cxx/test:compiler_test`; native API tests
cover ownership, source providers, and failure propagation. The
[execution corpus](test/README.md) uses ordinary `loom_test` targets with
independent numerical oracles.

## Callable symbol names

Public functions use readable Loom names: `ticks32` becomes `@ticks32`, and
`device::ticks32` becomes `@device.ticks32`. C-linkage functions keep their
unqualified C name. Private helpers have module-local names that can be
disambiguated as functions become reachable.

The leading `loom::symbol` attribute chooses an exact name independently of
the C++ spelling:

```cpp
namespace arithmetic {
[[loom::symbol("math.square.u32")]] unsigned square(unsigned);
unsigned square(unsigned value) { return value * value; }

[[loom::symbol("math.square.f32")]] float square(float value) {
  return value * value;
}
}

[[loom::kernel, loom::symbol("kernels.square"),
  loom::workgroup_size(1, 1, 1)]]
void square_kernel(unsigned* output, const unsigned* input) {
  output[0u] = arithmetic::square(input[0u]);
}
```

This exports `@math.square.u32`, `@math.square.f32` and `@kernels.square`.
Authored Loom can declare and call the typed exported functions and resolve
them against the imported library. The linker still checks argument and result
types. Generated kernel configuration keys use the chosen name, such as
`@kernels.square.workgroup_count.x`.

The same attribute applies to declarations, definitions, kernels, check cases
and benchmarks. One annotation on any canonical redeclaration supplies the
name; repeated identical redeclarations agree, and conflicting names diagnose.
Source calls and `--root` selection still use C++ names. The attribute changes
neither visibility nor reachability: a selected root remains public, a reached
helper remains private, and an unused helper is omitted. Ordinary called
functions still require C++ definitions during import.

Public overloads, operator functions and concrete template specializations
require explicit names. An attribute on a primary template, member, local or
parameter is rejected. Intrinsic bindings (`loom::op` and `loom::assume`) have
no function symbol to rename. Names contain ASCII letters, digits, `_`, `$`, `.`
or `-`, with no leading `@`. Functions and configuration values share one Loom
namespace; conflicting exact names diagnose instead of receiving an automatic
suffix.

## Kernel pointer contracts

Kernel pointer parameters can state the byte alignment supplied by their caller:

```cpp
using Words = unsigned __attribute__((vector_size(64)));

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void copy_block([[loom::assume_aligned(64)]] const unsigned* input,
                [[loom::assume_aligned(64)]] unsigned* output) {
  *reinterpret_cast<Words*>(output) =
      *reinterpret_cast<const Words*>(input + 16);
}
```

The annotation promises alignment of the incoming pointer's address. Its
argument is a positive power-of-two integer constant, and the attribute goes
before the parameter type. Consistent declarations may repeat the contract;
the definition inherits it even when parameter names differ. The importer
emits `buffer.assume.alignment` at kernel entry, where the pointer has zero
byte offset into its buffer binding.

The sixteen-word displacement above preserves 64-byte alignment. Advancing one
word instead guarantees only four-byte alignment, and assigning a different
pointer to the parameter does not transfer the entry promise to that value.
The annotation performs no allocation, realignment, or runtime check. Its
guarantee must match the caller's buffers; the
[memory guide](../../../../docs/src/guide/buffers-views-memory.md#aligned-bases-enable-wide-transfers)
shows the corresponding High IR contract.

Ordinary helper pointers carry an additional byte origin. Their parameter
annotations currently diagnose because alignment of the combined address
requires an origin-aware contract; strengthening the backing buffer alone would
be incorrect for an aligned interior pointer.

Kernel pointers can also mark their incoming buffer roots as mutually
non-overlapping with `[[loom::noalias]]`:

```cpp
#include <loomcxx/kernel.h>

[[loom::kernel, loom::workgroup_size(256, 1, 1),
  loom::workgroup_count(3, 64, 1)]]
void gather_rows([[loom::noalias]] const float* weights,
                 [[loom::noalias]] const unsigned* row_ids,
                 [[loom::noalias]] float* output) {
  unsigned row = loom::workgroup_id.y;
  unsigned column = loom::workgroup_id.x * 256u + loom::workitem_id.x;
  unsigned source_row = row_ids[row];
  loom::assume(source_row < 65536u);
  output[row * 768u + column] = weights[source_row * 768u + column];
}
```

Each marked parameter produces `buffer.assume.noalias` at kernel entry. Distinct
marked roots promise disjoint storage; unmarked parameters can still alias
them. Aliases and interior pointers derived from a root retain its identity.
Reassigning a parameter does not attach the entry promise to its replacement.
This enables existing memory-dependence analysis and, when the address and
memory-stability facts permit it, uniform loads such as the row-ID read above.

The attribute has no arguments and composes with alignment, for example
`[[loom::noalias, loom::assume_aligned(64)]] const float* input`. It introduces
no runtime check, allocation, alignment or nonnull guarantee. Declarations and
definitions reconcile contracts by parameter position, including concrete
kernel template instances.

This is Loom's buffer-root contract. C `restrict` and C++ `__restrict__` describe
accesses during a lexical scope and can permit two read-only pointers to share
storage. They currently contribute no alias facts during import. Applying
`loom::noalias` to ordinary helper parameters diagnoses: helper calls need
invocation-scoped alias contracts, including when a helper receives two
non-overlapping slices of one backing buffer.

## Named configuration values

An attributed `extern const` scalar declares a Loom specialization input:

```cpp
[[loom::config("tuning.factor")]] extern const unsigned factor;

unsigned scale(unsigned value) { return value * factor; }
```

The declaration imports as `config.decl @tuning.factor : i32`, and each read
becomes `config.get @tuning.factor : i32`. Unresolved reads survive ordinary
cleanup and bytecode serialization. Import once, then specialize fresh copies
of that module with different settings through the existing Loom config API
or command-line inputs. `iree-test-loom` and `iree-benchmark-loom` both apply
`--config=tuning.factor=5` when compiling called functions as well as kernels.

Config values work in ordinary arithmetic, branches, loop bounds and scheduling
attributes:

```cpp
[[loom::config("tuning.unroll")]] extern const unsigned unroll;
[[loom::config("tuning.depth")]] extern const unsigned depth;

unsigned sum(const unsigned* input, unsigned count) {
  unsigned total = 0;
  [[loom::unroll(unroll), loom::pipeline(depth)]]
  for (unsigned index = 0; index < count; ++index) {
    total += input[index];
  }
  return total;
}
```

An attributed constant definition supplies an exact value:

```cpp
[[loom::config("tuning.factor")]] const unsigned factor = 5;
```

This imports as `config.def @tuning.factor = 5 : i32`. A provider translation
unit can contain only definitions, with no functions. Embedders import it with
`loomc_module_import_cxx` and pass the resulting module as
`loomc_compile_options_t::config_module`; that provider is borrowed and remains
immutable. `loom-link --mode=merge` also combines declarations and definitions.
Supplying a provider as an ordinary rooted-link library does not bind configs.

A source definition fixes that setting throughout its translation unit,
including earlier reads and other source names using the same key. It can also
participate in C++ constant evaluation. The initializer is an exact definition,
not an overridable default: derived constants may already be folded during
parsing. Reusable kernels include declaration-only headers and receive their
config definitions at Loom compilation. Each specialization starts from the
unresolved module again.

Bindings require a leading `[[loom::config("key")]]` attribute on every
declaration and definition, including redeclarations. Keys are explicit,
nonempty Loom symbol spellings without `@`. Different source names with the same
key share one binding; their source types and any exact definitions must agree. Boolean, integer,
floating-point and enum scalars retain their source representations. Mutable,
volatile, thread-local, local, member, pointer and aggregate bindings produce
source diagnostics. Configs are values without addressable storage.

An unresolved setting is not a C++ constant expression. Template arguments,
`constexpr` initializers, `static_assert` and fixed type extents still require
source-known values; use source definitions and reimport when changing C++
types or template instantiations. Ordinary `if` remains available for Loom
specialization. `if constexpr` selects from source-known constants, including
template arguments and exact config definitions; unresolved config declarations
cannot select a C++ branch.

## Compile-time branches

`if constexpr (expression)` imports only the source-selected arm. Template
specializations can choose different operations, helper calls, and return types
without emitting a runtime condition. A discarded arm contributes no writes,
returns or continues to Loom control analysis, so it cannot prevent a counted
loop from retaining its schedule.

```cpp
template <bool Filter>
unsigned sum(unsigned count) {
  unsigned total = 0;
  [[loom::unroll(2)]]
  for (unsigned index = 0; index < count; ++index) {
    if constexpr (Filter) {
      if (index & 1u) continue;
    }
    total += index;
  }
  return total;
}
```

`sum<false>` imports as an unconditional scheduled loop. `sum<true>` retains
the per-iteration filter. CXX performs required-constant checking when parsing
or instantiating the source, and the importer consumes that selection directly.
A nonconstant condition produces a source diagnostic, including an
instantiation location when the failure occurs in a template.

Both ordinary `if` and `if constexpr` support an initializer before the
semicolon. It executes once before the selected path, including when a false
constexpr condition has no `else`:

```cpp
if (unsigned original = value++; selected) return value + original;
if constexpr (++visits; false) { /* Discarded. */ }
```

The condition itself can declare its scalar decision variable. Its name remains
available in both arms; `if constexpr` requires the variable read to be a
constant expression:

```cpp
if (unsigned remaining = count - offset) return remaining;
if constexpr (const unsigned width = 4) return width;
```

`while` and `for` also admit scalar decision declarations. Each check creates
a new value, including the final false check. The body and the `for` increment
can read or update it:

```cpp
unsigned total = 0;
for (unsigned remaining = count; unsigned chunk = remaining;
     remaining -= chunk) {
  if (chunk > 4) chunk = 4;
  total += chunk;
}
```

These loops lower to `scf.while` with named decision values. Initializers run
in the preheader and after each body/increment, preserving side effects and
`continue` sequencing.

Discarded template arms are not instantiated. Outside templates, both arms
remain subject to C++ source checking even though only one is imported.

## Typed views and layouts

`<loomcxx/view.h>` exposes Loom views without hiding their shape and address
mapping behind a pointer intrinsic. Static extents remain template arguments;
`loom::type::dynamic` marks each extent supplied when the view is formed. Layout
values live in the matching `loom::encoding` namespace and can be dense or use
runtime element strides.

```cpp
#include <loomcxx/view.h>

namespace loomt = loom::type;
using rows32 = loomt::view<const float, loomt::dynamic, 32>;

rows32 suffix(rows32 source, unsigned first, unsigned remaining) {
  return loom::view::subview(source, {first, 0}, {remaining});
}

void copy_element(const float* input, float* output, unsigned rows,
                  unsigned input_stride, unsigned row, unsigned column) {
  auto input_layout = loom::encoding::layout::strided(input_stride, 1u);
  auto output_layout = loom::encoding::layout::dense<2>();
  auto source = loom::buffer::view<loomt::dynamic, 32>(
      input, {rows}, input_layout);
  auto destination = loom::buffer::view<loomt::dynamic, 32>(
      output, {rows}, output_layout);
  auto tail = suffix(source, row, rows - row);
  loom::view::store(loom::view::load(tail, 0, column), destination, row,
                    column);
}
```

The deduced `subview` overload retains the source's static/dynamic extent
pattern. An explicit `subview<loom::type::dynamic, 16>(...)` selects a different
result pattern while still deducing the element and source shape. Dimensions
are supplied in source-axis order, with no slot for an axis already fixed by
the type. A `view<const T, ...>` supports loads; ordinary C++ template deduction
rejects attempts to store through it.

The facade types have real, trivially copyable C++ object representations, so
copies, overload resolution, `sizeof`, and data-model-dependent layout remain
source-language facts. Import projects each value into the facts Loom needs:
dynamic extents, one first-class layout, and the dependent view. Calls and
structured branches reserve all destination identities before constructing the
dependent view type. A helper returning `rows32` therefore has the shape:

```loom
func.def @suffix(...) -> (%rows: index, %layout: encoding<layout>,
                          view<[%rows]x32xf32, %layout>)
```

Every call result and control-flow join names its own extent and layout instead
of retaining references to values inside the callee or one incoming branch.

Compile-time policies can choose the shape type itself:

```cpp
template <bool Strided>
auto make_view(const float* input) {
  if constexpr (Strided) {
    auto layout = loom::encoding::layout::strided(11u, 1u);
    return loom::buffer::view<loomt::dynamic, 8>(input, {3}, layout);
  } else {
    auto layout = loom::encoding::layout::dense<2>();
    return loom::buffer::view<3, 8>(input, {}, layout);
  }
}

template <class View>
float read_view(View source) {
  return loom::view::load(source, 2, 3);
}
```

`make_view<false>` returns a statically shaped `3x8` view with a dense layout;
`make_view<true>` returns a dynamic-row view with an explicit row stride. Each
specialization deduces its own return type. The generic reader accepts either
type, and Loom retains the shape and layout facts through the helper calls.

## Integer bit counts

`<loomcxx/scalar.h>` exposes `ctlzi`, `cttzi`, and `ctpopi` for non-boolean
integer types. These count leading zeros, trailing zeros, and set bits in the
source type's representation. The result keeps the input type; narrow operands
are not implicitly widened before counting. For zero input, the zero counts
return the type's bit width, and the population count returns zero.

```cpp
#include <loomcxx/scalar.h>

unsigned first_ready_lane(unsigned mask) {
  return loom::scalar::cttzi(mask);
}
```

This imports as `scalar.cttzi ... : i32`, using the same shared target lowering
as authored High. Signed values count representation bits. Floating-point math
permissions do not apply to these integer operations.

## Integer atomics

`<loomcxx/atomic.h>` exposes scalar atomic observations and updates through typed
pointers. The kind, ordering, and scope use the same names as High IR, with
explicit template arguments. Pointer arithmetic and helper calls preserve the
original allocation and byte origin.

```cpp
#include <loomcxx/atomic.h>
#include <loomcxx/kernel.h>

using loom::atomic::kind;
using loom::atomic::ordering;
using loom::atomic::scope;

[[loom::kernel, loom::workgroup_size(32, 1, 1),
  loom::workgroup_count(8, 1, 1)]]
void claim_slots(unsigned* counter, unsigned* slots) {
  // The caller initializes the counter to zero and provides 256 slots.
  unsigned ticket = loom::view::atomic::rmw<
      kind::addi, ordering::relaxed, scope::device>(1u, counter);
  slots[ticket] = 1;
}

unsigned replace(unsigned* pointer, unsigned expected, unsigned replacement) {
  return loom::view::atomic::cmpxchg<
      ordering::acq_rel, ordering::acquire, scope::device>(
      expected, replacement, pointer);
}
```

The atomic operations import directly as:

```loom
%cell = buffer.view %counter[%counter_byte_offset] : buffer -> view<1xi32>
%ticket = view.atomic.rmw<addi> %increment, %cell[0] {ordering = relaxed, scope = device} : i32, view<1xi32> -> i32
%old = view.atomic.cmpxchg %expected, %replacement, %cell[0] {failure_ordering = acquire, scope = device, success_ordering = acq_rel} : i32, view<1xi32> -> i32
```

`rmw<kind, ordering, scope>(value, pointer)` returns the old memory value.
`reduce<kind, ordering, scope>(value, pointer)` performs an atomic update without
a result. Integer kinds include exchange, addition, subtraction, bitwise
AND/OR/XOR, and signed or unsigned minimum/maximum. Minimum/maximum signedness
must agree with the source integer type; the High payload itself is signless.

`cmpxchg<success, failure, scope>(expected, replacement, pointer)` returns the
old value on both success and failure. It writes exactly when the old bits
equal `expected`, without spurious failure or an output parameter. Failure
ordering cannot release or be stronger than success ordering. The declaration
binding validates this pair using the same contract as High IR.

`load<ordering, scope>(pointer)` observes an object without changing it;
`store<ordering, scope>(value, pointer)` publishes a value without reading the
old one. Loads accept relaxed, acquire, or sequentially consistent ordering.
Stores accept relaxed, release, or sequentially consistent ordering.

```cpp
unsigned consume(const unsigned* ready, const unsigned* payload) {
  while (loom::view::atomic::load<ordering::relaxed, scope::system>(ready) == 0) {}
  loom::buffer::fence<ordering::acquire, scope::system>();
  return *payload;
}

void publish(unsigned* ready, unsigned* payload, unsigned value) {
  *payload = value;
  loom::view::atomic::store<ordering::release, scope::system>(1u, ready);
}
```

The caller supplies shared storage with the requested synchronization domain
and keeps the payload live and unchanged until the consumer finishes. The
consumer's load and fence become `view.atomic.load` and `buffer.fence`; using
an acquire load directly also orders the payload read. These operations work
in ordinary functions. A fence orders the executing thread's memory accesses;
it does not rendezvous with other invocations or complete asynchronous DMA.
Fence ordering is acquire, release, acquire-release, or sequentially consistent.

Pointers identify live, naturally aligned, non-boolean integer objects.
Loads accept const pointers, and updates require mutable storage. Volatile
pointers are accepted; every atomic load remains observable even when its
result is discarded. These bindings preserve the chosen scope; the target
diagnoses unsupported widths, memory spaces, and synchronization contracts.
Native support follows the corresponding High operation. The importer does not
choose cache policies, insert host locks, or replace observations with
read-modify-write operations.

GCC atomic builtins use the same High operations in C and C++ without a facade
header. The supported integer forms are `__atomic_load_n`, `__atomic_store_n`,
`__atomic_exchange_n`, `__atomic_compare_exchange_n`, and both
`__atomic_fetch_<op>` and `__atomic_<op>_fetch` for `add`, `sub`, `and`, `or`, and
`xor`. `__atomic_thread_fence` supplies a standalone fence. These calls preserve
system scope; the typed facade also offers explicit narrower scopes.

```c
unsigned reserve(unsigned* next, unsigned count) {
  return __atomic_fetch_add(next, count, __ATOMIC_RELAXED);
}

bool replace(unsigned* state, unsigned* expected, unsigned replacement) {
  return __atomic_compare_exchange_n(state, expected, replacement, false,
                                     __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}
```

Fetch operations return either the observed value or the updated value computed
from it, without reloading storage. Signed integer atomic arithmetic wraps.
Compare-exchange returns success and writes the observed value to `*expected`
only on failure; success leaves expected storage untouched. All value arguments,
including `weak`, are evaluated once. High uses strong compare-exchange for
either weak value. The expected pointer uses ordinary addressable storage,
such as an incoming pointer or array element.

Ordering arguments must be pure integer constants; use the predefined
`__ATOMIC_*` values. `__ATOMIC_CONSUME` uses acquire semantics, and a relaxed
thread fence has no effect. Runtime orderings, target-specific modifier bits,
generic by-reference forms, NAND, and storage outside the non-boolean integer
subset diagnose at import. Native width, memory-space, and scope support follow
the corresponding High operation.

## Subgroup cooperation

`<loomcxx/kernel.h>` exposes subgroup topology, votes, and broadcasts as typed
High operations. `subgroup_id()`, `subgroup_count()`, `subgroup_size()`, and
`subgroup_lane_id()` return unsigned coordinates. The execution width includes
inactive lanes and can exceed the number of invocations in a partial subgroup.

Votes observe the active invocations at the call. `subgroup_any(predicate)` and
`subgroup_all(predicate)` return booleans. `subgroup_ballot(predicate)` returns
a 64-bit mask, with bit *i* identifying physical lane *i*;
`subgroup_ballot<unsigned>(predicate)` explicitly requests a 32-bit mask.
`subgroup_active_mask<Mask>()` uses the same width contract without a predicate.
The selected mask must cover the target subgroup width.

Broadcasts preserve their scalar or explicit vector value type. A named source
lane must be active; the target determines its range and uniformity requirements.
`subgroup_broadcast_first(value)` selects the first active lane, including
inside divergent control flow; that lane need not be lane zero.

```cpp
#include <loomcxx/kernel.h>
using UInt4 = unsigned __attribute__((ext_vector_type(4)));

[[loom::force_inline]] UInt4 exchange(UInt4 local, unsigned elected_lane) {
  return loom::subgroup_broadcast(local, elected_lane);
}

[[loom::force_inline]] unsigned long long ready_streams(bool ready) {
  return loom::subgroup_ballot(ready);
}
```

These become `kernel.subgroup.broadcast ... : vector<4xi32>, i32` and
`kernel.subgroup.vote.ballot ... : i1 -> i64`. Declarations use ordinary
`[[loom::op("kernel.subgroup.broadcast")]]` bindings; custom names and concrete
template instances use the same admission path. Calls require a kernel body or
a `loom::force_inline` helper. Each argument is evaluated once, and the shared
compiler owns convergence, value facts, and target lowering.

Collectives do not imply memory synchronization. An execution rendezvous names
its memory space, participant scope, and ordering separately:

```cpp
using loom::atomic::ordering;
using loom::atomic::scope;

// After observing a system publication, acquire the payload and let the
// subgroup cooperate on it.
loom::buffer::fence<ordering::acquire, scope::system>();
loom::barrier<loom::memory_space::global, scope::subgroup,
              ordering::acq_rel>();
```

The second call emits one `kernel.barrier<global> scope(subgroup)
ordering(acq_rel)`. Barriers accept subgroup or workgroup scope. Global memory
accepts acquire, release, or acquire-release ordering; workgroup memory requires
acquire-release. They can live in ordinary callable helpers and do not complete
independent asynchronous DMA. `workgroup_barrier()` retains the HIP-style
global-and-workgroup-memory contract. Native payload widths and synchronization
support follow the corresponding High operations; source import does not split
values or insert target-specific instructions.

## Embedded Low assembly

`loom::low::assembly` embeds a typed descriptor-backed instruction fragment in
an ordinary C++ function. A single template exposes the selected descriptor
vocabulary; instructions need no corresponding C++ intrinsic declaration.
Include `<loomcxx/low.h>` and name the representation contract on a tag type:

```cpp
struct [[loom::representation("amdgpu.gfx11.generic.core")]] Contract {};
using Word = float __attribute__((ext_vector_type(1)));

Word pack(Word even, Word odd) {
  return loom::low::assembly<Contract, Word>(R"loom(
      (%even: reg<amdgpu.vgpr>, %odd: reg<amdgpu.vgpr>) -> (reg<amdgpu.vgpr>) {
        %selector = s_mov_b32 0x05040100
        %packed = v_perm_b32 %odd, %even, %selector
        return %packed
      }
  )loom", even, odd);
}
```

The selector takes the low two bytes from each input word. The C++ inputs and
result retain their `vector<1xf32>` semantic types, while the literal describes
physical registers. The byte permutation does not convert floating-point
values to integers: it operates on their bits. Changing `Word` to a one-element
unsigned vector uses the same physical instructions. Source lowering checks
both the width and register class of every argument and result against the
selected target's mapping.

The same interface exposes XDNA's native saturating INT4 pack. This function
clamps 128 signed bytes to `[-8, 7]` and packs adjacent lanes into the low and
high nibbles of 64 output bytes:

```cpp
struct [[loom::representation("amd.xdna.aie2p.core")]] Aie2p {};
using SignedBytes = signed char __attribute__((ext_vector_type(128)));
using Packed = unsigned char __attribute__((ext_vector_type(64)));

Packed saturate_s4(SignedBytes values) {
  return loom::low::assembly<Aie2p, Packed>(R"loom(
      (%values: reg<aie2p.vec256 x4>) -> (reg<aie2p.vec256 x2>) {
        set.pack-size 0
        set.saturation 1
        %packed = vpack.x.signed %values
        return %packed
      }
  )loom", values);
}
```

High `vector.bitpack` packs low bits; this fragment also selects the hardware's
saturation behavior. Its control-register writes and the pack's state reads
are descriptor effects visible to optimization and scheduling. The native
[packing test](../../../../../experimental/xdna/cts/testdata/assembly_pack.cxx)
alternates saturated and unsaturated fragments on the same inputs, within
ordinary C++ vector loads, stores, and loops.

`Contract` selects a representation vocabulary independently of the hardware
profile used for compilation. The first example can be imported with the AMDGPU
descriptors enabled, then specialized to a compatible profile such as `gfx1151`.
The `loomc` API uses the context's target environment; the native importer
accepts an optional `low_asm_environment` in its import options. Ordinary C++
import does not require that environment.

The imported module contains a private `low.func.def ... asm` with the readable
body and a source-typed `low.invoke inline` at the C++ callsite. Existing
inlining, scheduling, register allocation, verification, and emission handle
the fragment. For this function, source-to-Low produces just `s_mov_b32`,
`v_perm_b32`, and the return. Both text and bytecode preserve the descriptor
contract and can be reused after the C++ frontend and source invocation end.

The source argument is a narrow raw string literal. Its formal parameters bind
positionally to the remaining C++ arguments, which are evaluated once. Result
and argument types are SSA scalars or vectors; `void` permits no result.
The literal contains an argument list, optional result list, optional `where`
predicates, and a braced body using the normal Low assembly grammar. It can use
local SSA values and blocks. Surrounding C++ names and module symbols are not
implicit captures. Pointer/view/record arguments require an explicit Low ABI
binding beyond this scalar/vector surface. Scheduling is free; the literal
does not impose a locked instruction order.

Malformed mnemonics and types diagnose inside the original C++ literal.
Physical signature mismatches diagnose at the callsite during target lowering.
The body is verified at import, before it becomes trusted compiler IR.

## Executable checks and benchmarks

Include `<loomcxx/check.h>` to author a correctness case beside its implementation:

```cpp
#include <loomcxx/check.h>

unsigned byte_increment(unsigned input) {
  unsigned char value = (unsigned char)input;
  ++value;
  return value;
}

LOOM_CHECK_CASE(wraps_byte) {
  const auto actual = byte_increment(255u);
  loom::check::expect_equal(actual, 0u);
}
LOOM_CHECK_BENCHMARK(wraps_byte_benchmark, wraps_byte);
```

Save this as `checks.cxx`. The existing test and benchmark tools accept it
directly when the C++ importer and VM target are enabled:

```sh
iree-bazel-run --config=loom-importer-cxx --//loom/config/target:enable=vm \
  //loom/src/loom/tools/iree-test-loom -- checks.cxx

iree-bazel-run --config=loom-importer-cxx --//loom/config/target:enable=vm \
  //loom/src/loom/tools/iree-benchmark-loom -- checks.cxx \
  --iterations=10 --warmup-iterations=1 --output-format=jsonl
```

The importer emits an ordinary function, a `check.case`, and a benchmark
reference. The case's IR is:

```loom
check.case public @wraps_byte {
  %input = check.literal value(255) : i32
  %actual = func.call @byte_increment(%input) : (i32) -> (i32)
  %expected = check.literal value(0) : i32
  check.expect.equal actual(%actual) expected(%expected) : i32
  check.return
}

check.benchmark<@wraps_byte> @wraps_byte_benchmark
```

`LOOM_CHECK_CASE(name)` expands to `[[loom::check_case]] void name()`.
Cases are concrete namespace-scope functions with no parameters and a `void`
result. Namespace qualification and static helper functions work normally.
`LOOM_CHECK_BENCHMARK(name, case_name)` declares a benchmark referencing a case
in the same translation unit. The benchmark runner owns timing and iteration
policy and applies its existing correctness gate.

Case bodies admit scalar constants, initialized automatic `const` or `constexpr`
bindings, direct calls to defined ordinary functions, and terminal
`loom::check::expect_equal` observations. Ordinary calls can return and accept
plain records of scalars, including nested and empty records. Immutable copies
and member reads retain each field's source type and value; expectations compare
individual scalar fields:

```cpp
struct Observation {
  unsigned returned;
  unsigned updated;
  unsigned guard;
};

Observation observe_update();  // Defined in this source or an included header.

LOOM_CHECK_CASE(update_values) {
  const auto actual = observe_update();
  loom::check::expect_equal(actual.returned, 37u);
  loom::check::expect_equal(actual.updated, 13u);
  loom::check::expect_equal(actual.guard, 37u);
}
```

Native kernel checks can generate inputs, launch the kernel, and compare its
storage in the same file:

```cpp
#include <loomcxx/check.h>
#include <loomcxx/kernel.h>

[[loom::kernel, loom::workgroup_size(1, 1, 1),
  loom::workgroup_count(1, 1, 1)]]
void update(unsigned* output, unsigned value) {
  output[0] = value;
}

LOOM_CHECK_CASE(update_values) {
  const auto storage = loom::check::fill<unsigned, 3>(37u);
  const auto middle = loom::check::slice<1>(storage, 1);
  loom::check::launch<update>(middle, 13u);
  loom::check::expect_bitwise(middle, loom::check::fill<unsigned, 1>(13u));
  loom::check::expect_bitwise(loom::check::slice<1>(storage, 0),
                            loom::check::fill<unsigned, 1>(37u));
  loom::check::expect_bitwise(loom::check::slice<1>(storage, 2),
                            loom::check::fill<unsigned, 1>(37u));
  loom::check::expect_event("device", "type", "asan_report", "count", 0);
}
LOOM_CHECK_BENCHMARK(update_benchmark, update_values);
```

For an AMDGPU-enabled build, run this file with
`iree-test-loom native_checks.cxx --device=amdgpu`. The benchmark tool accepts
that same file and device selection. Build-integrated native tests select
`AMDGPU_HARDWARE_PROFILE` or `AMDGPU_ACCESS_PROFILE` from
`//loom/target/amdgpu:execution_profiles.bzl`.

`fill<T, Count>(value)` produces a dense rank-one `tensor<T, Count>` using a
constant payload. Copies of the handle retain the same storage. `const` applies
to the handle; the kernel can update its contents. `slice<Count>(source, offset)`
creates an alias at a constant **element** offset and diagnoses a range outside
its source. Slicing a slice preserves the original storage and accumulated
origin. Tensor element types are unqualified, non-boolean scalars.

`launch<Kernel>(arguments...)` names a kernel declaration directly. Tensor
arguments bind its buffer parameters, including each slice's origin; scalar
arguments match its scalar ABI types. The kernel retains its declared launch
geometry and configurations. Tensors are test data handles, not C++ pointers
that can be passed to ordinary functions. `expect_bitwise` compares equal-typed
tensors exactly, including floating-point payload bits.

These calls produce the same IR as authored Loom checks:

```loom
%storage = check.generate.fill value(37) : tensor<3xi32>
%middle = check.tensor.view %storage offset(4) : tensor<3xi32> -> tensor<1xi32>
%value = check.literal value(13) : i32
kernel.launch @update(%middle, %value) : (tensor<1xi32>, i32)
%expected = check.generate.fill value(13) : tensor<1xi32>
check.expect.bitwise actual(%middle) expected(%expected) : tensor<1xi32>
```

Provider requirements use literal names followed by name/value pairs, for example
`loom::check::require("hal.amdgpu.descriptor_set", "descriptor_set",
"amdgpu.rdna3_5.core")`. Attribute values are literal strings or scalar constant
expressions. `require` emits `check.requires`; `expect_event` uses the same
spelling for `check.expect.event`. The existing requirement and event providers
interpret these attributes, including device selection and sanitizer reports.

Constant expressions are evaluated by the frontend; runtime conversions and
arithmetic belong inside ordinary called functions. After the first expectation,
a case can contain further expectations, inline `fill` and `slice` calls, and an
optional final bare `return`, but no further ordinary calls, launches or bindings.
Mutable locals, branches, loops and pointer operations in the case body produce
source diagnostics during import.

These restrictions describe the harness body. The implementation under test
can use the importer's ordinary control flow, helpers, templates and vector
operations wherever the selected target supports them. Its definition must be
available in the translation unit, directly or through an include. The current
shared testbench materializes case inputs, executes ordinary calls through the
IREE VM function provider and native launches through the HAL provider, then
checks observations. Executing control flow inside the harness itself requires a
shared executor for complete `check.case` bodies; no C++-specific interpreter is
involved.

Build-integrated checks use the same source file:

```python
load("//loom/build_tools/bazel:defs.bzl", "loom_test")
load("//loom/target/vm:execution_profiles.bzl", "VM_REFERENCE_PROFILE")

loom_test(
    name = "checks_test",
    srcs = ["checks.cxx"],
    execution_profiles = [VM_REFERENCE_PROFILE],
)
```

List included project headers in `data`. `inputopts` accepts provider-scoped
settings such as `["cxx:std=c++23 D=COUNT=8 I=include"]`. `loom_test` imports and
links its root-owned cases, runs correctness checks, and runs a single-iteration
benchmark smoke check. Dependency libraries retain their own tests; linking one
does not implicitly add its cases to the root's suite.

## Source input and diagnostic ownership

Repository-authored C++ programs use `.cxx`; compiler expectation fixtures use
`.cxx-test`. Kernels, helpers, check cases and benchmarks can share one `.cxx`
translation unit.

`loom-link`, `iree-test-loom`, `iree-benchmark-loom`, and `iree-run-loom` share
the optional input provider used by `loom-check`. Enabled binaries recognize
`.c`, `.cc`, `.cpp`, and `.cxx`; explicit `--input-format=cxx` handles another
filename or stdin. `--input-options='cxx:std=c++23 D=COUNT=8 I=include'` applies
options to C++ inputs, including `--library` sources. Include directories are
relative to the source file. A `.c` suffix selects the provider; use `std=c11`
or another supported C standard to select C language semantics.

Source inputs can also be linked once and executed as bytecode:

```sh
loom-link checks.cxx --mode=link --include-input-tests \
  --to=bc --output=checks.loombc
iree-test-loom checks.loombc
```

Source diagnostics retain the admitted main file and headers through linking
and target compilation. Module source identities are remapped by the linker;
execution reports attribute failed expectations to the original source range,
including assertions authored in headers. Bytecode retains those locations.
`--source-prefix-map=old=new` changes displayed filenames without changing
include resolution.

The input provider changes admission, not execution semantics. `iree-run-loom`
accepts C++ kernel programs for its HAL workflow; scalar check cases use
`iree-test-loom`. Building without the C++ importer removes C++ input support
from these tools. Other importers can participate through the same optional
provider contract.

## Source constructs

Multiple kernels and ordinary functions can coexist. By default, concrete
definitions with external visibility are exported. Repeated `--root` options
select qualified source function names; their reachable helpers remain private.
Kernel entries export by their symbol name, so a separate Loom module can
resolve its `kernel.decl` through normal library dependencies. An explicit
`export("name")` is needed only when the artifact export name differs.
Overloaded root names require disambiguation in the source. Template helpers
are instantiated by cxx before import. This interface does not yet define an
external C++ ABI for linking separately compiled C++ translation units.

Scoped and unscoped enums preserve their resolved integer representation:

```cpp
enum class Kind : unsigned { solid, water, sky };
unsigned opacity(Kind kind) { return kind == Kind::water ? 128u : 255u; }

enum class Byte : unsigned char { low, high = 255 };
unsigned widen(Byte value) { return static_cast<unsigned>(value); }
```

`Kind` arguments use `i32`, and `Kind::water` becomes a named `%water` constant.
`Byte` uses `i8`; widening it emits `scalar.extui`, while a signed byte enum uses
`scalar.extsi`. Enum pointers retain that element width and its storage stride.
C++ nominal type checking happens before projection, so distinct enum classes
do not become interchangeable simply because they share an IR carrier.

Fixed underlying types are checked for representability, including implicit
enumerator increments. Inferred enums select the first type in the integer
promotion order that contains their complete value range; values above 32 bits
and the full unsigned 64-bit range remain intact. Template-dependent definitions
are resolved when instantiated. Boolean enums use `i1` values. Accessing their
memory requires a byte-storage projection and receives the same diagnostic as
accessing storage through `bool*`; carrying either pointer is supported.

GNU `packed` enums select their smallest signed or unsigned storage while
retaining the promotion selected from their enumerator range. An enum containing
`255` uses `i8` storage and one-byte pointer stride, but promotes to `int` for
arithmetic and overload resolution. Fixed underlying types and scoped enums keep
their declared representation.

```cpp
enum __attribute__((packed)) Byte { last = 255 };
int next(Byte value) { return value + 1; }
```

`next` zero-extends its `i8` argument before the `i32` addition, so `next(last)`
returns `256`.

Ordinary helpers can accept and return aggregate records by value, including
nested records, explicit vectors, interior pointers, and empty records:

```cpp
typedef unsigned u32x4 __attribute__((vector_size(16)));
struct Result {
  u32x4 lanes;
  const unsigned* cursor;
  bool valid;
};

Result advance(Result previous, unsigned count) {
  Result next = previous;
  for (unsigned i = 0; i < count; ++i) {
    next.cursor++;
    next.lanes += u32x4{1u, 2u, 3u, 4u};
    next.valid = !next.valid;
  }
  return count ? next : previous;
}
```

Copies are independent values. Assigning a nested member changes only that
member of the destination; the source copy and untouched siblings retain their
values. Helper and region boundaries transport members in declaration order,
with each pointer retaining both its buffer root and byte origin. C++ padding
is absent from these signatures. Empty records retain source identity and layout
while transporting zero values; a two-scalar record remains distinct from a
pointer. Kernel parameters continue to use the ordinary launch binding ABI and
reject record parameters.

The admitted records are complete aggregates with public fields, trivial copying
and destruction, and no unions or base classes. Fields recursively admit scalar,
vector, pointer, and record values. Braced construction includes normalized
positional, designated and omitted-field initialization, plus default member
expressions that do not access the object under construction. Bitfields,
references, array fields and addresses of automatic records are rejected.
Implicit default construction such as `Result()` and default member expressions
accessing another member require source object initialization phases and receive
explicit diagnostics; substituting aggregate braces would not preserve their
semantics.

Record pointers project named fields from existing storage using the C++ object
layout. Nested fields, field addresses, pointer arithmetic and scalar/vector
updates share the ordinary buffer plus byte-origin representation:

```cpp
struct Particle {
  float position;
  float velocity;
  unsigned flags;
};

void step(Particle* particles, unsigned index, float dt) {
  particles[index].position += particles[index].velocity * dt;
  particles[index].flags |= 1u;
}
```

Here the pointer advances by `sizeof(Particle)` (12 bytes), and the fields use
offsets 0, 4 and 8. Float accesses become `view<1xf32>` loads and stores; the
unsigned field uses `view<1xi32>` with unsigned arithmetic and conversion
operations when signedness matters. Each view retains the same allocation
identity. Natural padding, explicit alignment and concrete template layouts
come from the frontend's completed object layout; field stores leave padding
and neighboring fields untouched. `&particles[index].flags` can pass through an
ordinary helper as a typed pointer.

Memory records admit non-boolean scalar, enum, vector and nested named fields,
including fixed arrays of those types and multidimensional arrays. Array
members decay to borrowed pointers, so ordinary helpers can consume them.
Indexing and pointer-to-array arithmetic preserve each source extent's stride.
The admitted object must be a complete aggregate with trivial copying and
destruction, without bases, unions, bitfields or `no_unique_address`. Stored
pointers, references and Loom view/encoding objects require additional
memory representations and receive source diagnostics. Whole-record loads and
stores also diagnose: the by-value SSA partition is not an object-copy operation.
Volatile record pointers and volatile fields retain observable accesses through
nested members and array elements.

This allows a kernel to consume a block-quantized format directly from its
existing storage. For example, an IQ4_XS block holds 256 nonlinear four-bit
codes, one FP16 base scale, and eight six-bit group scales in 136 bytes:

```cpp
struct IQ4XSBlock {
  // Base multiplier shared by all eight groups.
  _Float16 scale;
  // Two high bits of each group scale code.
  unsigned short scales_high;
  // Two low four-bit scale codes per byte.
  unsigned char scales_low[4];
  // Sixteen packed bytes per 32-value group.
  unsigned char quants[128];
};
static_assert(sizeof(IQ4XSBlock) == 136);

float decode(const IQ4XSBlock* blocks, unsigned block_index, unsigned index,
             const signed char* codebook) {
  auto* block = &blocks[block_index];
  unsigned group = index / 32;
  unsigned lane = index % 32;
  unsigned low = (block->scales_low[group / 2] >> (4 * (group % 2))) & 15;
  unsigned high = (block->scales_high >> (2 * group)) & 3;
  float scale = float(block->scale) * (int(low | (high << 4)) - 32);
  unsigned packed = block->quants[group * 16 + lane % 16];
  unsigned code = (packed >> (4 * (lane / 16))) & 15;
  return scale * codebook[code];
}
```

The block pointer advances by 136 bytes; `scales_low` and `quants` start at
offsets 4 and 8. Their elements use byte loads, the base scale uses an FP16
load, and the codebook uses signed integer-to-float conversion. No record or
array is copied or allocated. Arrays in by-value records and addresses of
automatic records require aggregate object initialization and copy projections.
Automatic scalars, vectors, and fixed scalar arrays use the storage contract
below.

Packed fields use the same typed memory operations, with their exact byte
origins and record strides:

```cpp
struct [[gnu::packed]] Sample {
  unsigned char tag;
  float value;
};

void scale(Sample* samples, unsigned index, float factor) {
  samples[index].value *= factor;
}
```

This advances by five bytes per sample, then loads and stores a `view<1xf32>`
at byte offset one within the record. The importer preserves the allocation
identity and does not assert natural alignment for the float. Shared target
lowering selects legal accesses from the retained alignment facts, including
bytewise accesses when needed. Class packing, member packing and pragma packing
compose with nested records and explicit alignment. A target without an
implementation for the requested memory operation reports that at compilation.

Record layout queries honor GNU `packed`, explicit `aligned(N)`, standard
`alignas`, and `#pragma pack`. Requests stay attached to their declarations
through template specialization. Nested records keep their own padding; packing
an outer record changes where its members begin.

```cpp
template<class Word>
struct Block {
  unsigned short scale;
  Word words[4];
} __attribute__((packed));

unsigned stride() { return sizeof(Block<unsigned>); }             // 18
unsigned payload() { return __builtin_offsetof(Block<unsigned>, words); } // 2
unsigned alignment() { return alignof(Block<unsigned>); }         // 1
```

The functions import as scalar constants. `__attribute__((packed))`,
`[[gnu::packed]]`, underscored spellings, record-level requests and member-level
requests share the same source layout owner. GNU `aligned(N)` raises a packed
member's alignment, while a pragma pack cap limits member alignment. Record
alignment can raise the final stride without changing internal member offsets.
Standard `alignas` retains its own validation rules. Explicit alignment on shared
arrays reaches the workgroup allocation's `align` operand.

Compound `__builtin_offsetof` designators follow nested members and constant
array indices. For example, `__builtin_offsetof(Block<unsigned>, words[2])` is
`10`. Indices may become constant through template instantiation; the complete
offset retains the source `size_t` width.

Bitfield layouts retain actual bit positions, including fields crossing their
declared storage units and zero-width alignment boundaries. Layout queries are
independent of value admission; bitfield memory operations remain unsupported.
Packed base classes, virtual members, Microsoft bitfield ABI layouts, aligned
typedefs, and GNU `aligned` without an explicit argument produce source diagnostics.

Explicit fixed vectors retain their lanes and element widths in High IR:

```cpp
typedef unsigned u32x16 __attribute__((vector_size(64)));

u32x16 replace_depth(u32x16 previous, u32x16 depth) {
  return (previous & 65535u) | (depth & 0xffff0000u);
}
```

This imports as `vector<16xi32>` arguments, vector constants, `vector.andi` and
`vector.ori`. Vector pointers use contiguous `vector.load` and `vector.store`,
with source object sizes determining pointer strides. The
`test/vector_depth.cxx` example combines a vector depth recurrence with a scalar
tail and ordinary pointer reinterpretation.

GNU `vector_size` and Clang `ext_vector_type` forms admit arithmetic, bitwise
operations, shifts, scalar splats, brace initialization and indexed lane reads.
Named brace initializers and typed temporaries such as `u32x16{1u, 2u}` use
the same lane conversions and zero the remaining lanes. Typed temporaries can
appear in returns, call arguments and larger expressions. Vector comparisons
and logical negation produce source-width integer masks containing zero or all ones.
Narrow integer vector arithmetic wraps at its element width; it does not acquire
the scalar language's integer promotions. Equal-size vector casts and
`__builtin_bit_cast` reinterpret the bits. Whole vectors flow through local
values, helper calls, returns, branches and loops.

Admission requires non-boolean lanes and an object layout with no padding beyond
the lanes. Packed boolean vectors and padded three-lane objects need separate
storage projections. Lane assignment, lane addresses, swizzles, vector `&&`/`||`
and vector-conditioned `?:` also produce source diagnostics: they require
additional lvalue or lane-selection projections. Target support remains a
separate compilation boundary. For example, the VM can execute extracted lane
programs but currently rejects vector aggregate transport across loop/branch
arguments; that requires a shared aggregate representation through control flow.

Kernel launch contracts are explicit source attributes:

```cpp
[[loom::kernel, loom::workgroup_count(2, 1, 1),
  loom::workgroup_size(64, 1, 1)]]
void fill(float* output) {
  output[0u] = 42.0f;
}
```

Each exact launch attribute accepts three positive i32 integer constant
expressions. `loom::workgroup_count_range(xmin, xmax, ymin, ymax, zmin, zmax)`
and `loom::workgroup_size_range(...)` instead constrain required config values
with inclusive bounds. For example:

```cpp
[[loom::kernel, loom::workgroup_count_range(1, 4, 1, 1, 1, 1),
  loom::workgroup_size(64, 1, 1)]]
void configured(float* output) { output[0u] = 42.0f; }
```

The importer emits `config.decl @configured.workgroup_count.x` with a
`range(%value, 1, 4)` predicate. Normal Loom config specialization supplies
the exact value and rejects values outside the contract. An absent annotation
also produces three required configs, with no additional range constraint.
Config names use the imported kernel symbol, so overloads remain distinct.
Contracts on leading function declarations carry to the definition; conflicting
redeclarations and multiple contracts for the same dimension group are errors.
Import does not guess launch dimensions or select a physical target.

`loom::assume`, also exposed as HIP's `__builtin_assume`, carries unsigned
integer bounds into Loom's value analysis:

```cpp
constexpr unsigned capacity = 28672;
loom::assume(count < ((capacity / sizeof(unsigned) - 16u - 320u) / 16u + 1u) &&
             channel < 256u);
```

Each `binding < bound` becomes a `scalar.assume` range on the current value.
Parentheses and repeated bindings preserve the same refinements as separate
calls. Narrow unsigned bindings retain their C++ integer promotions before
refinement. Bounds are pure integer constant expressions in `[1, INT32_MAX]`,
including named constants, concrete template arguments, integral casts and
`sizeof`. Source conditions generate no runtime comparisons or branches.
Calls, mutation, volatile reads, overloaded operators and unsupported predicates
produce source diagnostics. The program must satisfy every declared bound.

Counted unsigned `for` loops accept explicit scheduling attributes:

```cpp
[[loom::unroll(3), loom::pipeline(2), loom::schedule("linear")]]
for (unsigned column = 0; column < columns; ++column) {
  total += input[column];
}
```

Factors and depths can also be ordinary integer expressions: local values,
helper parameters, template arguments, arithmetic, and topology values such
as `loom::workgroup_size.x`. The importer reads them once after the `for`
initializer, before loop execution, and retains them as SSA index operands
of `scf.for`. Mutation of a binding inside the loop does not change its
schedule. Source widths, promotions, and signedness are preserved.

```cpp
LOOM_FORCE_INLINE int sum(const int* input, unsigned columns,
                          unsigned factor, unsigned depth) {
  int total = 0;
  [[loom::unroll(factor), loom::pipeline(depth)]]
  for (unsigned column = 0; column < columns; ++column) {
    total += input[column];
  }
  return total;
}
```

A caller can supply values that become constant through Loom inlining or
target specialization. The consuming scheduling pass requires an exact value;
an unresolved factor or depth is a compilation error. Expressions inside the
annotations cannot call functions, mutate bindings, read volatile values, or
invoke overloaded operations. Ordinary statements can compute named values
before the annotation, and `sizeof` operands remain unevaluated.

Pipeline depth counts original iterations; unrolling follows pipelining.
`loom::unroll` without arguments requests full unrolling, and schedule ordering
is `linear`, `interleaved`, or `recurrence`. Unroll factors zero and one and
pipeline depth one keep that part serial. Negative factors and pipeline depths
outside `[1, 65535]` are compilation errors. An annotation on a loop that cannot
be represented as a nonwrapping counted loop is a source error. The cleaned
Loom can also use ordinary config values for schedule exploration without
reimporting C++.

Bounds such as `pixels / 16u` and steps such as `index += (1u << 2)` retain
`scf.for` when their integer expressions are constant. This includes macros,
constexpr bindings, concrete template parameters, integral casts, and `sizeof`.
Constant evaluation preserves source widths and defined unsigned wrapping;
unevaluated operands such as `sizeof(++value)` have no runtime effects.
Prefix and postfix increments have the same counted-loop behavior.

The induction and comparison use the same unsigned-int width. Unit steps may
use a stable runtime scalar bound; larger steps require a constant bound that
leaves room for the final increment without wrapping. Runtime starts preserve
zero-trip and partial-tail behavior. Mutable bounds, effectful expressions,
widened comparisons and potentially wrapping increments keep general-loop
semantics, and explicit scheduling on those loops is diagnosed.

Dynamic offset loops currently compile with linear body ordering. The core
interleaved/recurrence tail construction uses index-only division and
multiplication for those bounds and must preserve the offset domain before
these combinations can lower. Import preserves the requested policy and the
compiler reports the type mismatch; it does not substitute another schedule.

The native API in `import.h` accepts a finalized Loom context and arena block
pool and returns an owned module. Source rejection produces structured
diagnostics and a null module; infrastructure failures return status. The API
runs verification and leaves cleanup policy to its caller. Source buffers,
headers, and the frontend AST can be released as soon as import returns.

The optional public extension `loomc/import/cxx.h` exposes
`loomc_module_import_cxx` to C embedders. It takes an ordinary immutable source
with UNKNOWN format, context, workspace, source configuration and allocator.
The returned module uses the existing compile, emit, link, query and serialize
APIs. A failed source produces a failed result with retained diagnostics;
allocation, provider and API failures return status without partial outputs.
Diagnostics own their source contents, including included headers.

The `//loom/binding/c/example/cxx:jit_amdgpu` example imports two HIP-style
kernels, specializes them to a live AMDGPU profile, emits one HSACO and executes
both through IREE HAL. Its host driver is C; its input kernels are C++.

Includes follow ordinary quoted, user-directory, system-directory and
`include_next` ordering. A source provider can replace filesystem reads with
immutable caller-owned headers, including a shared source cache. Each invocation
owns its mutable preprocessing and semantic state. Header hits and misses are
reused within an invocation. Diagnostic sinks copy any bytes they retain.

The importer embeds `<stdfloat>`, `loomcxx/` and `hip/` source headers by default.
User and system include paths take precedence over the embedded system root. The HIP
facade maps topology, synchronization, scalar half conversions and math onto
the Loom vocabulary; it does not load the HIP runtime or a host SDK.

`--builtin-includes=false --I=loom/src/loom/import/cxx/include` uses the external
headers. Building with `--//loom/config/import/cxx:embed_includes=false`, or
`-DLOOM_IMPORT_CXX_EMBED_INCLUDES=OFF` in CMake, also removes those header
contents from the library and its rebuild dependencies.

Canonical Python op declarations generate constrained function templates in
`loomcxx/scalar.h`, their documentation, and native binding entries. The header
declares each operation once; concrete signatures are admitted only when used,
so supporting another numeric format does not add eagerly parsed overloads.
For example, `loom::scalar::expf(x)` imports `scalar.expf`;
`loom::scalar::approximate::expf(x)` explicitly grants AFN. The HIP spelling
`__expf(x)` is an ordinary inline wrapper around that declaration. An explicit
`[[loom::op("scalar.expf", "afn")]] float custom_exp(float);` declaration uses
the same checked binding. Incorrect arity, types, flags, or attribute arguments
produce source diagnostics.

Deduced scalar calls accept `_Float16`, `__bf16`, `float`, or `double`, with
matching operand types. An explicit template argument requests conversion:
`loom::scalar::mulf<__bf16>(value, 5.0f)` converts the second operand to BF16
before emitting a BF16 multiply. Custom operation templates retain their
declared math permissions across every concrete specialization.

Register lookup and mixed-width integer dots use the same declaration binding,
with independently typed operands and explicit dot signedness:

```cpp
typedef unsigned u32x16 __attribute__((vector_size(64)));
typedef unsigned u32x4 __attribute__((vector_size(16)));
typedef signed char i8x16 __attribute__((vector_size(16)));
typedef int i32x4 __attribute__((vector_size(16)));

[[loom::op("vector.table.lookup")]]
u32x4 lookup(u32x16 table, u32x4 indices);
[[loom::op("vector.dot4i", "s8s8")]]
i32x4 dot(i8x16 coordinates, i8x16 rows, i32x4 translation);

i32x4 transform_points(i8x16 coordinates, i8x16 rows, i32x4 translation) {
  return dot(coordinates, rows, translation);
}
```

The dot imports directly as `vector.dot4i<s8s8>` with `vector<16xi8>` inputs
and a `vector<4xi32>` accumulator/result. Each result adds four adjacent byte
products to its accumulator, wrapping to 32 bits. The `s8s8`, `u8s8`, `s8u8`,
and `u8u8` kinds must match the declared byte signedness. Input vectors have
equal lane counts, with four input lanes per accumulator/result lane.

Ordinary vector subscripting can reach the same register-table operation through
the shared `combine` pass. For example, a four-byte code vector can select from a
16-byte codebook without an operation binding:

```cpp
using Bytes4 = unsigned char __attribute__((ext_vector_type(4)));
using Codes4 = signed char __attribute__((ext_vector_type(4)));
using Codebook16 = signed char __attribute__((ext_vector_type(16)));

Codes4 decode(Codebook16 table, Bytes4 packed) {
  Bytes4 indices = packed & 15;
  return {table[indices[0]], table[indices[1]], table[indices[2]],
          table[indices[3]]};
}
```

The importer CLI and source-lowering pipeline run `combine` before target
legalization. This pass includes ordinary canonicalization and source
representation combines; later `canonicalize` cleanup preserves the legalized
representation. After source cleanup, the body is a byte-vector mask and one
`vector.table.lookup` using that byte vector. On supported AMDGPU targets this
selects three byte permutes. The rewrite preserves C++ promotions when removing
them would change the numeric indices, and also applies to directly authored
Loom extracts and vector construction. Reordered and repeated selectors keep
their original lane mapping.

Lookup supports integer or floating-point tables and integer index vectors.
Its result has the table's C++ element type and the index vector's lane count;
each index must be in the table's range. Calls retain the shared register-table
operation until target legalization chooses a native or portable recipe.
Banked memory lookup needs a memory operation carrying pointer provenance,
effects and placement; a register lookup declaration does not provide those
contracts.

Source standard, predefines, ABI triple, integer/pointer layout, and mathematical
approximation permissions are explicit options. `--std` selects the pinned
frontend's language and version macros; it does not promise historical-standard
conformance. LP64, LLP64 and ILP32 source layouts are independent of the machine
running the importer.

The current translation surface covers scalar and explicit vector arithmetic,
conversions, typed-pointer indexing and arithmetic, aggregate record values,
record field storage, local SSA values, automatic scalar, vector, and array storage,
conditional regions, short-circuit `&&` and `||`, counted and general `for`
loops, `while` and `do/while` loops, fixed workgroup arrays, and direct calls.
Unsupported reachable types and statements produce source diagnostics. Integral
subscripts preserve their source width and signedness. Interior pointers carry a
buffer root and an
object-relative byte offset through helper arguments, returns, conditional
regions, and loop-carried values. Kernel pointer parameters retain their
single-buffer binding ABI. Signed displacements are combined with the current
origin before entering the nonnegative offset domain, so an interior pointer
can move backward within its allocation.

`void*`, qualified void pointers, and pointers to forward-declared objects use
the same representation. Copies, casts, helpers, branches, loops, and SSA record
fields preserve the buffer and byte origin without requiring a pointee layout.
Recovering a supported object type enables ordinary memory access:

```cpp
unsigned read_erased(const void* storage, unsigned byte_offset) {
  auto* bytes = static_cast<const unsigned char*>(storage);
  auto* element = reinterpret_cast<const unsigned*>(bytes + byte_offset);
  return *element;
}
```

Typed object projection and scaled pointer arithmetic require an admitted
storage layout. Carrying an opaque pointer does not enable loads or stores of
unsupported object formats. Builtin `&*pointer` preserves the pointer without
projecting an object, so its pointee may remain incomplete. Function pointers
have no object-pointer representation and produce a source diagnostic.

Pointer addition, subtraction by an integer, unary plus, dereference, address-of
storage elements and automatic scalar/vector/array objects, and prefix/postfix
increments are admitted.
Integer increments update automatic bindings or storage-backed elements and
return the previous or updated value. Pointer increments update automatic
bindings; `*output++ = *input++` preserves both pointer origins. Compound
assignment supports scalar and vector storage through pointers, record fields,
and array elements. The right operand executes before the destination is
resolved, and one resolved address supplies both the load and store. Arithmetic
uses the source promotions and converts back to the element type before storing
or returning:

```cpp
unsigned mark(unsigned* words, unsigned index, unsigned mask) {
  return words[index] |= mask;
}

unsigned advance(unsigned char* counts) {
  return (*counts)++;  // Promotes, increments, then converts back to a byte.
}
```

An automatic object's address identifies the same storage seen by its direct
reads, writes, and aliases. Ordinary helpers can update local state through
pointers, including across branches and loops:

```cpp
[[loom::force_inline]] void increment(unsigned* value) { ++*value; }

unsigned update(unsigned input, bool enabled) {
  unsigned value = input;
  if (enabled) increment(&value);
  return value;
}
```

Automatic storage supports the same non-boolean scalar and vector types as
typed pointer storage. The importer emits `buffer.alloca<private>` at the
object's declaration, using its source size and alignment, and initializes it
with a store preserving its access qualifiers. A declaration without an
initializer leaves storage uninitialized for an output-only helper to write.
Addressed by-value parameters receive a private copy of their incoming value.
Pointers retain the same buffer and byte origin
through copies and borrowed helper returns. Automatic objects retain their C++
lifetimes; returning a pointer does not extend the pointee's lifetime. Taking an
address in an unevaluated operand or a discarded `if constexpr` arm does not
create storage.

Inlining and private-storage promotion belong to the shared compiler. For the
example above they replace the allocation, accesses, and helper call with a
conditional SSA result. Volatile and atomic observations retain memory effects;
addressing an object does not promise promotion. Scalars that need no storage
continue to import directly as SSA values. An aliased loop bound or induction
object uses a general loop so indirect mutations cannot be lost by counted-loop
lowering.

Fixed one-dimensional arrays of non-boolean scalars use the same private storage
contract. Braced and parenthesized element initializers execute in order, with
each element stored before evaluating the next clause. Omitted elements are
value-initialized; a declaration without an initializer emits no stores.
Array bounds can be deduced from the initializer. Decay, element addresses, and
whole-array addresses preserve the same allocation:

```cpp
unsigned guarded_update(unsigned input, bool enabled) {
  unsigned values[3] = {37, input, 41};
  if (enabled) increment(&values[1]);
  return values[0] + values[1] + values[2];
}
```

After inlining, shared promotion can carry these three scalar cells through the
branch without temporary memory. Dynamic indexing and atomic or volatile
observations retain storage where needed. Direct indexing into a declared array
publishes the C++ in-bounds precondition using its fixed extent. This is a source
precondition, not a runtime bounds check.

Conditional expressions and short-circuit operands carry binding updates only
along the executed path. Incrementing vectors or floating-point values requires
additional type projections and produces a source diagnostic. Pointer
differences, comparisons, and truth conversions also produce source diagnostics.
Distinct-root choices import as ordinary buffer values; executing them requires
the selected Loom target to support buffer transport through those control-flow
edges. Objects with constructors, exceptions, and indirect calls need additional
storage and control-flow projections before they can be imported.

Volatile scalar and vector accesses through pointers, automatic objects, and
workgroup arrays become `view.load/store<volatile>` and
`vector.load/store<volatile>`. The qualifier belongs to the accessed object:
a copied pointer or a pointer member
retains its pointee's observation semantics. Discarded reads, including explicit
casts to `void`, remain observable, and repeated accesses stay distinct through
optimization. Ordinary reads retain their usual optimization.

```cpp
unsigned observe(const volatile unsigned* input, volatile unsigned* output) {
  static_cast<void>(*input);
  unsigned first = *input;
  unsigned second = *input;
  *output = first;
  *output = second;
  return first + second;
}
```

Typed views use the same contract. A `loom::type::view<volatile unsigned, ...>`
preserves its element qualifier through copies, helpers and subviews;
`loom::view::load` returns an ordinary scalar and `loom::view::store` accepts
one. A `const volatile` element permits observations but rejects stores.
Volatile supplies observable accesses, without atomicity, synchronization or a
cache-coherence guarantee. Volatile scalar and vector local declarations receive
private storage even when their address is never taken. Volatile by-value
parameters require separating the incoming SSA signature from the qualified
parameter object and produce a source diagnostic. Stored pointer objects
need an object representation for their buffer and byte origin, and automatic
record values need aggregate memory copies; those declarations produce source
diagnostics. Namespace-scope volatile objects require global-storage projection
and cannot fold to their initializer.

`continue` skips the remaining body of the innermost `for`, `while`, or
`do/while`. Updates before the exit survive; a `for` increment and a `do/while`
condition still execute. Nested blocks and conditionals may continue from
either arm, while a shared source tail appears only once in the imported IR.
Counted loops retain `scf.for` and their explicit unroll/pipeline schedules.
This includes sparse copies, filtered pointer streams, and scalar or vector
recurrences. Each iteration uses ordinary conditional regions; there is no
extra loop-carried exit state.

Kernels and ordinary functions can return early through guard chains, nested
blocks and returning `if`/`else` trees. A returning conditional must have at
least one arm that always returns. The remaining source then executes only on
the path that continues, without copying that source into both arms. Returning
helpers retain structured bodies and can inline inside ordinary loops. A return
from within a loop, or a returning conditional with two continuing arms, is
diagnosed: those forms require a shared continuation or scoped exit contract.

Logical operators preserve contextual boolean conversions and evaluate their
right operand only in the selected `scf.if` region. Guarded loads and effectful
helpers are checked by native execution and device access sanitization. Some
compositions still reach core AMDGPU control-flow restrictions: a guarded
`while` condition can produce multiple loop exit edges, and `(a && b) || c`
can produce a shared continuation that the current masked-branch planner
rejects. Import preserves those source semantics; target compilation reports
the unsupported control-flow shape.

## Implementation boundaries

The importer composes five native packages. Each owns declared library targets,
header APIs, and direct API tests that do not link the aggregate importer.

| Package | Contract |
| --- | --- |
| `source/` | One configured frontend invocation, provider and diagnostic handling, immutable facade lookup, and source locations copied into the output module. |
| `value/` | Source type/layout projection, scalar/vector SSA and buffer/origin representations, arithmetic, and memory access construction from already evaluated operands. |
| `control/` | An immutable analysis of ordered source writes, function/iteration exits and fallthrough, and nonwrapping counted-loop eligibility. This package has no IR dependency. |
| `binding/` | Admission and construction for generated operation bindings, named scalar configs, kernel launch contracts, and explicit loop schedules. |
| `symbol/` | Root selection, reachable function identities, deterministic naming, and native function definitions with explicit body contracts. |

`import.cc` owns the native API's validation, exception boundary, module
ownership, and final verification. `translation.cc` composes the packages: it
evaluates expressions, maintains source-symbol-to-SSA bindings, and constructs
structured regions. Projection APIs consume resolved source identities and
explicit native builders; they do not call back into the translation driver.

For example, a cast first evaluates its operand in the driver and then calls
`Scalars::convert` with the SSA value, source and destination types, and source
owner for diagnostics. Function translation consumes `Functions::define`'s
body contract and builds one `ControlFlow` analysis. Loop construction queries
its retained writes and counted-loop proof instead of traversing the body again.
Workgroup allocation similarly retains its typed array view for later accesses.

The source owns AST, symbol, token, and layout storage until translation ends.
The module owns emitted IR, interned names, and copied locations independently
of that source. The public C adapter copies diagnostic source contents before
releasing the frontend. These ownership boundaries also apply when headers come
from a caller-supplied provider.
