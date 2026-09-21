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
  --I=include --std=c++26 --root=attention attention.cpp
```

`loom-import-cxx` imports one translation unit, runs canonicalization, common
subexpression elimination and dead-code elimination, and prints Loom text.
`--cleanup=false` exposes the direct import. `--to=bc --output=module.loombc`
produces normal Loom bytecode for the existing compilation and linking tools.

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

Save this as `checks.cc`. The existing test and benchmark tools accept it
directly when the C++ importer and VM target are enabled:

```sh
iree-bazel-run --config=loom-importer-cxx --//loom/config/target:enable=vm \
  //loom/src/loom/tools/iree-test-loom -- checks.cc

iree-bazel-run --config=loom-importer-cxx --//loom/config/target:enable=vm \
  //loom/src/loom/tools/iree-benchmark-loom -- checks.cc \
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
`loom::check::expect_equal` observations. Arguments and results retain their
source scalar types. Constant expressions are evaluated by the frontend;
runtime conversions and arithmetic belong inside the ordinary called functions.
After the first expectation, a case can contain further expectations and an
optional final bare `return`, but no further invocations or bindings. Mutable
locals, branches, loops, pointers, and kernel launches in the case body produce
source diagnostics during import.

These restrictions describe the harness body. The implementation under test
can use the importer's ordinary control flow, helpers, templates and vector
operations wherever the selected target supports them. Its definition must be
available in the translation unit, directly or through an include. The current
shared testbench materializes case inputs, executes its planned calls through
the IREE VM function provider, then checks observations. Executing control flow
inside the harness itself requires a shared executor for complete `check.case`
bodies; no C++-specific interpreter is involved.

Build-integrated checks use the same source file:

```python
load("//loom/build_tools/bazel:defs.bzl", "loom_test")
load("//loom/target/vm:execution_profiles.bzl", "VM_REFERENCE_PROFILE")

loom_test(
    name = "checks_test",
    srcs = ["checks.cc"],
    execution_profiles = [VM_REFERENCE_PROFILE],
)
```

List included project headers in `data`. `inputopts` accepts provider-scoped
settings such as `["cxx:std=c++23 D=COUNT=8 I=include"]`. `loom_test` imports and
links its root-owned cases, runs correctness checks, and runs a single-iteration
benchmark smoke check. Dependency libraries retain their own tests; linking one
does not implicitly add its cases to the root's suite.

## Source input and diagnostic ownership

`loom-link`, `iree-test-loom`, `iree-benchmark-loom`, and `iree-run-loom` share
the optional input provider used by `loom-check`. Enabled binaries recognize
`.c`, `.cc`, `.cpp`, and `.cxx`; explicit `--input-format=cxx` handles another
filename or stdin. `--input-options='cxx:std=c++23 D=COUNT=8 I=include'` applies
options to C++ inputs, including `--library` sources. Include directories are
relative to the source file. A `.c` suffix selects the provider; use `std=c11`
or another supported C standard to select C language semantics.

Source inputs can also be linked once and executed as bytecode:

```sh
loom-link checks.cc --mode=link --include-input-tests \
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
are resolved when instantiated. Boolean enums use `i1` values; pointers to them
require a byte-storage projection and receive the same diagnostic as `bool*`.

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

Memory records admit non-boolean scalar, enum, vector and nested named fields.
The admitted object must be a complete aggregate with trivial copying and
destruction, without bases, unions, bitfields or `no_unique_address`. Stored
pointers, references, arrays and Loom view/encoding objects require additional
memory representations and receive source diagnostics. Whole-record loads and
stores also diagnose: the by-value SSA partition is not an object-copy operation.
Volatile record pointers and volatile fields retain observable accesses through
nested members.

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
`test/vector_depth.cpp` example combines a vector depth recurrence with a scalar
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

The importer embeds `loomcxx/` and `hip/` source headers by default. User and
system include paths take precedence over the embedded system root. The HIP
facade maps topology, synchronization, scalar half conversions and math onto
the Loom vocabulary; it does not load the HIP runtime or a host SDK.

`--builtin-includes=false --I=loom/src/loom/import/cxx/include` uses the external
headers. Building with `--//loom/config/import/cxx:embed_includes=false`, or
`-DLOOM_IMPORT_CXX_EMBED_INCLUDES=OFF` in CMake, also removes those header
contents from the library and its rebuild dependencies.

Canonical Python op declarations generate the typed `_Float16`, `float` and
`double` overloads in `loomcxx/scalar.h`, their documentation, and native
binding entries. For example, `loom::scalar::expf(x)` imports `scalar.expf`;
`loom::scalar::approximate::expf(x)` explicitly grants AFN. The HIP spelling
`__expf(x)` is an ordinary inline wrapper around that declaration. An explicit
`[[loom::op("scalar.expf", "afn")]] float custom_exp(float);` declaration uses
the same checked binding. Incorrect arity, types, flags, or attribute arguments
produce source diagnostics.

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
record field storage, local SSA values, conditional regions, short-circuit `&&`
and `||`, counted and general `for` loops, `while` and `do/while` loops, fixed
workgroup arrays, and direct calls. Unsupported reachable types and statements
produce source diagnostics. Integral subscripts preserve
their source width and signedness. Interior pointers carry a buffer root and an
object-relative byte offset through helper arguments, returns, conditional
regions, and loop-carried values. Kernel pointer parameters retain their
single-buffer binding ABI. Signed displacements are combined with the current
origin before entering the nonnegative offset domain, so an interior pointer
can move backward within its allocation.

Pointer addition, subtraction by an integer, unary plus, dereference, address-of
an existing storage element, and prefix/postfix increments are admitted.
Integer increments update automatic bindings or storage-backed elements and
return the previous or updated value. Pointer increments update automatic
bindings; `*output++ = *input++` preserves both pointer origins. Compound
assignment supports scalar and vector storage through pointers, record fields
and workgroup arrays. The right operand executes before the destination is
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

Conditional expressions and short-circuit operands carry binding updates only
along the executed path. Incrementing vectors or floating-point values requires
additional type projections and produces a source diagnostic. Pointer
differences, comparisons, truth conversions, and addresses of automatic scalar
locals also produce source diagnostics.
Distinct-root choices import as ordinary buffer values; executing them requires
the selected Loom target to support buffer transport through those control-flow
edges. Objects with constructors, exceptions and indirect calls need additional
storage and control-flow projections before they can be imported.

Volatile scalar and vector accesses through pointers and workgroup arrays
become `view.load/store<volatile>` and `vector.load/store<volatile>`. The
qualifier belongs to the accessed object: a copied pointer or a pointer member
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
cache-coherence guarantee. Automatic scalar objects and automatic record values
use SSA transport and cannot represent volatile object storage; those
declarations produce an explicit source diagnostic. Namespace-scope volatile
objects require global-storage projection and cannot fold to their initializer.

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
