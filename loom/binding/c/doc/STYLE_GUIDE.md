# Loom C API Documentation Style Guide

This guide describes how public comments in `loom/binding/c/include/loomc`
are written so they work both as readable headers and generated API reference.
The comments are part of the binding contract: they describe observable API
behavior, ownership, lifetimes, threading, and failure semantics without
exposing implementation machinery.

The generated reference is Doxygen-based. Public headers remain the source of
truth, and `loom/binding/c/doc/generate.sh --check` is the local lint shape
for validating that comments match declarations.

## Reader Model

The primary reader is an embedding author building a native driver, JIT,
autotuner, package, or language binding around the Loom C API. They may never
read Loom or IREE internals. The documentation should let that reader answer:

- Which object owns this memory?
- How long does this view remain valid?
- Which calls are safe from multiple threads?
- Which failures return `loomc_status_t`, and which failures appear as result
  diagnostics?
- Which descriptor fields are required, optional, borrowed, copied, or retained?
- Which handles are mutable, immutable, reusable, or tied to a workspace?

Implementation details belong in implementation comments, design notes, or
non-installed representation headers. Public comments describe the shape seen by
callers.

## Comment Syntax

Doxygen-visible public comments use `///` for short declarations and `/** ... */`
for longer comments with sections. The comment sits immediately above the
declaration it documents.

```c
/// Non-owning string view over bytes that need not be NUL-terminated.
typedef struct loomc_string_view_t {
  /// First byte of the string, or `NULL` when the view is empty.
  const char* data;

  /// Number of bytes in `data`.
  loomc_host_size_t size;
} loomc_string_view_t;
```

Struct fields keep their own adjacent comment. A future binding author or LSP
user often sees one field with little context, so the field comment carries the
unit, ownership, or sentinel value when that matters.

## Function Comments

Function comments start with the caller-visible behavior. Longer comments add
parameter, return, ownership, lifetime, thread safety, and error-contract
sections as needed.

```c
/**
 * Creates an immutable source handle.
 *
 * @param options Source format, identity, bytes, and storage policy.
 * @param allocator Host allocator used for source-owned storage.
 * @param out_source Receives one retained source on success.
 *
 * @return OK when the source was created. Invalid argument is returned for
 * malformed descriptors or impossible storage/release combinations.
 *
 * @ownership
 * The caller owns the returned reference and releases it with
 * `loomc_source_release`.
 *
 * @lifetime
 * Borrowed source bytes must outlive the source. Copied source bytes are owned
 * by the source. External source bytes are released through the callback stored
 * in `options`.
 *
 * @thread_safety
 * The returned source is immutable and may be shared across threads.
 */
LOOMC_API_EXPORT loomc_status_t
loomc_source_create(const loomc_source_options_t* options,
                    loomc_allocator_t allocator, loomc_source_t** out_source);
```

For simple accessors, a one-line `///` comment is enough when the return
lifetime is obvious from the surrounding type. When returning a borrowed view,
the comment states the owner:

```c
/// Returns a borrowed identifier owned by `source`.
LOOMC_API_EXPORT loomc_string_view_t
loomc_source_identifier(const loomc_source_t* source);
```

## Standard Sections

These Doxygen aliases are available in `Doxyfile`:

- `@ownership`: reference ownership, allocator ownership, release functions.
- `@lifetime`: borrowed views, workspace-tied storage, persistent handles.
- `@thread_safety`: concurrent call rules, immutable handle sharing, mutable
  workspace exclusivity.
- `@error_contract`: distinction between infrastructure status and
  operation-domain diagnostics.

The plain Doxygen tags used most often are:

- `@param` for every public function parameter when the function takes
  parameters.
- `@return` for status-returning functions and nontrivial accessors.
- `@retval` when individual return values carry policy.
- `@note` for important API facts that are not ownership, lifetime, threading,
  or error-contract rules.
- `@warning` for caller actions that can invalidate memory, introduce races, or
  make cached compilation results unsound.
- `@pre` for preconditions that are part of the API contract.

## Status, Results, And Diagnostics

Status comments make the API/result split explicit. `loomc_status_t` represents
API misuse, allocation failure, cancellation infrastructure, and other failures
that prevent an operation result from being produced. Source, program, link,
compile, and configuration failures are represented by result state plus
diagnostics once the operation completed far enough to report them.

```c
/**
 * Links sources into an opaque module ready for compilation.
 *
 * @error_contract
 * A non-OK status means the API call failed before a link result could be
 * produced. Unresolved symbols, duplicate globals, invalid Loom source, and
 * configuration failures are reported as diagnostics on the returned result.
 */
```

## Threading Language

Threading comments name the handle state:

- Immutable handles may be shared across threads.
- Mutable handles name the single-owner rule.
- Workspaces are caller-owned scratch for one worker at a time.
- Frozen indexes are immutable and reusable by many link operations.
- Results are immutable after creation; retaining/releasing references is safe
  according to the function contract.

```c
/// Creates a mutable workspace for one worker thread at a time.
///
/// @thread_safety
/// Calls that mutate the same workspace require external synchronization.
/// Persistent sources, results, frozen link indexes, compilers, and linkers do
/// not borrow from workspace storage unless a descriptor explicitly says so.
```

## Boundary Language

Public comments use Loom concepts: source, workspace, result, diagnostic,
artifact, link index, compiler, linker, and configuration. IREE, HAL, VM,
parser implementation, pass managers, arenas, and backend-specific runtime
handles appear only in optional adapter headers whose purpose is interop. The
core `loomc/*.h` headers describe stable C ABI behavior in Loom terms.

## Examples

Examples in public comments should be small enough to stay correct during API
review. Larger examples live in `loom/binding/c/example` and can be included
or linked from docs once the API surface settles.

Good inline examples show ownership and cleanup:

```c
loomc_source_t* source = NULL;
loomc_status_t status =
    loomc_source_create(&options, loomc_allocator_system(), &source);
if (!loomc_status_is_ok(status)) {
  loomc_status_free(status);
  return;
}

// Use source.

loomc_source_release(source);
```

## Generated Docs

The local docs command is:

```bash
loom/binding/c/doc/generate.sh
```

The lint shape is:

```bash
loom/binding/c/doc/generate.sh --check
```

Both commands require `doxygen` on `PATH`. Generated output defaults to
`${TMPDIR:-/tmp}/loom-c-api-docs` for local browsing and
`${TMPDIR:-/tmp}/loom-c-api-docs-check` for the check mode.
