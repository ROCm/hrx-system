# Source input providers

The input layer separates source format selection from compiler invocation.
`format.h` owns filename suffixes, explicit language names, and per-language
option selection. These immutable descriptors have no loader callbacks, so a
command-line client can select C++ without linking a second compiler into its
executable.

`loom_tooling_input_admit_loomc_module` receives the selected format and admits
text, bytecode, or an optional importer through LoomC. Returned modules and
results retain their source identities and bytes after frontend teardown.
`loom-link` snapshots admitted Loom text or bytecode modules into its reusable
link index. Source languages are translated to Loom modules by their owning
frontends before that boundary; bytecode inputs retain their file buffers for
metadata-first selective linking.

Native compiler tools use `loom_input_provider_t` to pair the same format
metadata with a native loader. `loom_input_module_t` retains the native module,
physical-to-logical source association, and exact admitted bytes. Optional
importers own their metadata and adapters beside their implementation; final
binaries compose the enabled formats independently of target or emit providers.

A provider receives borrowed source bytes, a physical input path, language
options, the parser diagnostic sink, and a source-capture callback. It captures
headers from every admission route before frontend storage expires. Source
rejection emits diagnostics and returns OK with no module; allocation, IO,
option, and sink failures propagate as statuses. Every load result is
deinitialized, including failures. Context and block pool outlive the result.

Physical paths own include lookup. Logical names are applied to diagnostics
and module source tables after capture, preserving source IDs and locations.
Distinct admitted sources cannot share a logical name. The retained resolver
uses module source IDs, so consumers cloning a module retain those assignments.
The linker publishes its input-to-output source mappings while materializing
the result. Execution sessions use those mappings to retain owned snapshots
indexed by the linked module's source IDs, including after input modules are
released. Later compilation consumes the retained mapping rather than matching
filenames or reopening source files.

Compiler clients can freeze an original location as executable data before
releasing this ownership. `loom_func_location_capture` in
[`location_capture.h`](../../ops/func/location_capture.h) consumes a location
ID and this input's resolver, then copies its reachable provenance graph,
structured field spans, and available original source lines into module-owned
semantic attributes:

```c
loom_parameterized_attr_array_t nodes;
IREE_RETURN_IF_ERROR(loom_func_location_capture(
    input.module, original_op->location,
    loom_input_module_source_resolver(&input), scratch_arena, &nodes));
IREE_RETURN_IF_ERROR(loom_func_location_build(
    builder, nodes, loom_type_buffer(), LOOM_LOCATION_UNKNOWN, &location_op));
```

The builder belongs to `input.module` and inserts at the consumer's chosen
point. Capture happens while original field spans and admitted text are still
available; an importer or compiler client chooses the site explicitly. It is
independent of verification and does not turn an operation's debug annotation
into a late runtime lookup. The same construction works for Loom text, C++
headers and macro invocation ranges, or other frontends carrying Loom locations.

The resulting `func.location` value survives serialization, linking, and
debug stripping. It lowers to interned immutable rodata and crosses native VM
imports as an ordinary `buffer`. The compiler-free reader in
[`format/location.h`](../../format/location.h) borrows its buffer span;
retaining the buffer retains the data after compiler, program, and process
teardown. Source names, one-based lines and Unicode code-point columns, field
ranges, tagged/fused provenance, and optional original text are part of the
value's identity. Missing source remains explicit, with no filesystem reads or
printed-IR replacement.

Adding a native importer requires its descriptor and adapter, conditional
composition in each final binary, and importer-owned integration tests. The
shared test rules accept `.<format>-test`; envelopes, diagnostic matching,
output comparison, and updates stay common. Language options belong to the
provider. Source-consuming tools accept `--input-options=format:options` once
per format and select each source by filename unless `--input-format`
overrides that selection. Bytecode remains a native input independently of
source providers. `loom_test` and `loom_library` run the selected source
importer first and pass only resulting Loom modules to `loom-link`. Projecting
a source language into executable check cases is an importer-owned language
contract; the resulting operations use the shared testbench and its existing
invocation providers.
