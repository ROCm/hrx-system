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

Multiple kernels and ordinary functions can coexist. By default, concrete
definitions with external visibility are exported. Repeated `--root` options
select qualified source function names; their reachable helpers remain private.
Overloaded root names require disambiguation in the source. Template helpers
are instantiated by cxx before import. This interface does not yet define an
external C++ ABI for linking separately compiled C++ translation units.

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

Factors and depths are positive i32 integer constant expressions, including
concrete template parameters. They become ordinary SSA index operands of
`scf.for`, so the imported program preserves the scheduling choice through
Loom's existing transformations. Pipeline depth counts original iterations;
unrolling follows pipelining. `loom::unroll` without arguments requests full
unrolling, and schedule ordering is `linear`, `interleaved`, or `recurrence`.
Depth or factor one explicitly keeps that part serial. An annotation on a
loop that cannot be represented as a nonwrapping counted loop is a source
error. The cleaned Loom can replace these constants with ordinary config
values when exploring schedules without reimporting C++.

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
conversions, typed-pointer indexing and arithmetic, local SSA values,
conditional regions, short-circuit `&&` and `||`, counted and general `for` loops, `while` and
`do/while` loops, fixed workgroup arrays, and direct calls. Unsupported reachable
types and statements produce source diagnostics. Integral subscripts preserve
their source width and signedness. Interior pointers carry a buffer root and an
object-relative byte offset through helper arguments, returns, conditional
regions, and loop-carried values. Kernel pointer parameters retain their
single-buffer binding ABI. Signed displacements are combined with the current
origin before entering the nonnegative offset domain, so an interior pointer
can move backward within its allocation.

Pointer addition, subtraction by an integer, unary plus, dereference, address-of
an existing storage element, and prefix/postfix increments are admitted.
Integer and pointer increments update an owned automatic binding and return
its previous or updated value; `*output++ = *input++` preserves both pointer
origins. Conditional expressions and short-circuit operands carry binding
updates only along the executed path. Incrementing memory elements, vectors or
floating-point values requires additional lvalue/type projections and produces
a source diagnostic. Pointer differences, comparisons, truth conversions, and
addresses of automatic scalar locals also produce source diagnostics.
Distinct-root choices import as ordinary buffer values; executing them requires
the selected Loom target to support buffer transport through those control-flow
edges. Objects with constructors, exceptions and indirect calls need additional
storage and control-flow projections before they can be imported.

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
| `control/` | An immutable analysis of ordered source writes, return/fallthrough outcomes and nonwrapping counted-loop eligibility. This package has no IR dependency. |
| `binding/` | Admission and construction for generated operation bindings, kernel launch contracts, and explicit loop schedules. |
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
