# IREE Clang-Tidy Checks

This directory contains the out-of-tree clang-tidy plugin for IREE C/C++
contract and style checks.

The plugin builds against the LLVM installation that provides the `clang-tidy`
binary that loads it. Normal runtime, Loom, and libhrx builds do not require
LLVM development packages.

## Bazel

```bash
python dev.py bazel clang-tidy //runtime/src/iree/base:all
python dev.py bazel clang-tidy --base origin/main
python dev.py bazel clang-tidy --all --profile ci
```

Bazel exposes the matching LLVM install through the optional
`@iree_clang_tidy_llvm` repository. The repository is a stub unless explicitly
enabled. `dev.py bazel clang-tidy` enables it with
`--repo_env=IREE_CLANG_TIDY_LLVM=auto`.

Discovery checks `IREE_CLANG_TIDY_LLVM_CONFIG`, `LLVM_CONFIG`,
`IREE_CLANG_TIDY_LLVM_ROOT`, `IREE_LLVM_ROOT`, `LLVM_ROOT`, and then `PATH`.
Use `IREE_CLANG_TIDY_BINARY` or `IREE_CLANG_TIDY_CLANGXX_BINARY` only when the
tools are not next to the discovered `llvm-config`.

The Bazel action runner uses the same configured C/C++ compile arguments that
feed `dev.py bazel compile-commands`, builds one cacheable action per source
file, and writes a per-source report:

```bash
iree-bazel-test --repo_env=IREE_CLANG_TIDY_LLVM=auto \
  //build_tools/clang_tidy:refcount_checks_test \
  //build_tools/clang_tidy:status_checks_test \
  //build_tools/clang_tidy:trace_checks_test
```

## CMake

```bash
python dev.py cmake configure
python dev.py cmake clang-tidy runtime/src/iree/base/status.c
python dev.py cmake clang-tidy --base origin/main
```

The CMake command builds the plugin in `.tmp/iree-clang-tidy-plugin`,
materializes generated C/C++ compile inputs in the configured CMake build tree,
and runs `run-clang-tidy` against source files using that build tree's
`compile_commands.json`. Select the CMake build tree with `--cmake-build-dir`
or `IREE_CMAKE_BUILD_DIR`. The runner defaults to a capped parallel job count
and can be tuned with `IREE_CLANG_TIDY_JOBS`.

Plugin-only CMake validation is also available:

```bash
cmake -S build_tools/clang_tidy -B .tmp/iree-clang-tidy-plugin \
  -DLLVM_DIR="$(llvm-config --cmakedir)"
cmake --build .tmp/iree-clang-tidy-plugin
ctest --test-dir .tmp/iree-clang-tidy-plugin --output-on-failure
```

## Checks

### `iree-status-discarded`

`iree-status-discarded` diagnoses calls returning `iree_status_t` when the call
is used as a bare expression statement, including `(void)` casts:

```c
some_status_returning_function();
(void)some_status_returning_function();
```

The status result must be returned, stored for later consumption, or explicitly
consumed. The check intentionally does not reason about whether a callee is
infallible; if the function returns `iree_status_t`, the caller owns that result.

### `iree-status-borrowed-parameter`

`iree-status-borrowed-parameter` diagnoses functions that take an
`iree_status_t` parameter by value but only observe it:

```c
static bool is_unavailable(iree_status_t status) {
  return iree_status_is_unavailable(status);
}
```

A by-value `iree_status_t` parameter means the callee participates in ownership.
The callee must return it, store it into an owning destination, consume it,
or transfer it to another owning status API. If the caller still needs the
original after such a call, the caller passes `iree_status_clone(status)` and
keeps owning the original. Observer helpers should expose the non-owning
representation they actually need:

```c
static bool is_unavailable(iree_status_code_t status_code) {
  return status_code == IREE_STATUS_UNAVAILABLE;
}

static iree_status_t append_status(iree_string_builder_t* builder,
                                   const iree_status_t status) {
  return iree_status_format_to(status, append_chunk, builder)
             ? iree_ok_status()
             : iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to format status");
}
```

The check intentionally keeps exceptions narrow. Status primitives that define
the observer API, known status sinks, C++ `iree::Status` formatting internals,
and documented borrowed callback boundaries are modeled explicitly. Ordinary
debug/reporting helpers should use `iree_status_code_t`, `const iree_status_t`,
or `const iree_status_t&` instead of accepting an owned status they do not
consume.

### `iree-status-lifetime`

`iree-status-lifetime` treats local `iree_status_t` variables as owned linear
values. The check diagnoses mechanically provable local lifetime errors:

```c
iree_status_t status = do_work();
(void)status;  // Status leaves scope unconsumed.

iree_status_t status = do_work();
status = do_cleanup();  // Overwrites an owned status.

iree_status_t status = do_work();
iree_status_ignore(status);
return status;  // Uses a consumed status value.

iree_status_t status = iree_ok_status();
status = do_work();  // OK initializer was never used.
```

The accepted terminal actions mirror `runtime/src/iree/base/status.h`: return
the status, store it into an owning destination, consume it with
`iree_status_free`/`iree_status_ignore`/`iree_status_consume_code`, or transfer
it through helpers such as `iree_status_join`, `iree_status_annotate`, and
`iree_status_freeze` while continuing to own the returned status.

`iree_status_ignore` is reserved for failures that were deliberately not
handled. Once a status has been reported through formatting or printing helpers,
the local owner should release it with `iree_status_free` instead:

```c
iree_status_fprint(stderr, status);
iree_status_free(status);
```

An `iree_ok_status()` initializer should represent the real initial state of a
terminal status accumulator. If the next same-scope assignment replaces it
before a branch, loop, observer, cleanup join, or ownership transfer can use
that initial OK value, initialize the variable from the producer directly or use
the appropriate return-if-error helper. The simple adjacent declaration and
assignment form is fixable with `clang-tidy --fix`.

The lifetime model focuses on local ownership states that can be proven from
the AST with high confidence. Straight-line code, block scope, local transfers,
returns, `if` branches, status helper calls, C++ status wrappers, and known
status-owning callback boundaries are checked directly. Loop-carried status
variables and functions with `goto` cleanup paths are treated conservatively so
diagnostics remain high signal.

### `iree-status-transfer-order`

`iree-status-transfer-order` diagnoses full expressions where the same local
`iree_status_t` appears more than once and at least one occurrence transfers
ownership:

```c
return iree_status_join(status, status);
return iree_status_join(status, notify_failure(status));
```

C does not sequence function argument evaluation. In the second example, either
argument can be evaluated first, so one path can consume or replace the status
while the other path still tries to use the old owned value. The reliable shape
is explicit sequencing through owned temporaries:

```c
iree_status_t notify_status = notify_failure(status);
return iree_status_join(status, notify_status);
```

When two independent consumers intentionally need the same failure payload, clone
first and give each owner one status value:

```c
iree_status_t cloned_status = iree_status_clone(status);
iree_status_t notify_status = notify_failure(cloned_status);
return iree_status_join(status, notify_status);
```

The check is local and syntactic by design. It models known status consumers,
status transfer helpers, C++ status wrapper constructors, and status observer
functions. Pure observer expressions such as multiple `iree_status_is_*` checks
do not transfer ownership and are accepted.

### `iree-cpp-designated-initializer`

`iree-cpp-designated-initializer` diagnoses C++ designated initializers. MSVC
does not support this syntax, so C++ aggregate initializers should use IREE's
comment field-label convention instead:

```c++
iree_hal_buffer_params_t params = {
    /*.type=*/IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
    /*.access=*/IREE_HAL_MEMORY_ACCESS_ALL,
    /*.usage=*/IREE_HAL_BUFFER_USAGE_DEFAULT,
};
```

The check is C++-only. C designated initializers remain valid and are not
diagnosed:

```c
iree_hal_buffer_params_t params = {
    .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
    .access = IREE_HAL_MEMORY_ACCESS_ALL,
    .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
};
```

The simple `.field = value` C++ form is fixable with `clang-tidy --fix`. The
fix replaces the designator with `/*.field=*/` and leaves the initializer value
unchanged. When an initializer skips a field, the fix inserts an explicit
zero-initialized placeholder so the positional initializer keeps the original
designated-initializer semantics:

```c++
// Before.
iree_example_t example = {
    .mode = IREE_EXAMPLE_MODE_DEFAULT,
    .name = IREE_SVL("example"),
};

// After, assuming `flags` is declared between `mode` and `name`.
iree_example_t example = {
    /*.mode=*/IREE_EXAMPLE_MODE_DEFAULT,
    /*.flags=*/{},
    /*.name=*/IREE_SVL("example"),
};
```

Macro expansions are diagnosed but not automatically fixed. For macro bodies,
update the macro definition. For macro arguments, rewrite the argument at the
callsite or use zero-initialization plus field assignment when the initializer
is intentionally sparse.

### `iree-lifecycle-naming`

`iree-lifecycle-naming` diagnoses caller-owned output records that use
`allocate` naming but have a matching `deinitialize` cleanup function:

```c
iree_status_t iree_tool_temp_file_allocate(
    iree_string_view_t stem, iree_tool_temp_file_t* out_file);
void iree_tool_temp_file_deinitialize(iree_tool_temp_file_t* file);
```

The paired cleanup contract says the caller owns the record storage and the
function initializes that storage. The acquisition side should use
`initialize` naming even when initialization creates external resources or
allocates backing storage:

```c
iree_status_t iree_tool_temp_file_initialize(
    iree_string_view_t stem, iree_tool_temp_file_t* out_file);
void iree_tool_temp_file_deinitialize(iree_tool_temp_file_t* file);
```

`allocate/free` naming remains the storage-oriented shape for APIs that publish
heap or pool objects through pointer-to-pointer outputs, allocator operations
such as buffer allocation, and arena helpers that allocate backing storage into
descriptors without owning a deinitialize contract.

`initialize` functions that publish a pointer-to-pointer output must name the
caller-owned storage backing that typed view:

```c
iree_status_t iree_vm_stack_initialize(
    iree_byte_span_t storage, iree_vm_invocation_flags_t flags,
    iree_vm_stack_t** out_stack);
```

Without an explicit `storage` or `*_storage` parameter, a pointer-to-pointer
output reads like a callee-owned object factory and should use
`create/destroy/retain/release` or `allocate/free` naming instead.

### `iree-refcount-lifecycle`

`iree-refcount-lifecycle` treats `iree_atomic_ref_count_t` as an object
lifetime primitive, not as a generic atomic counter. A refcounted IREE C object
is anchored by an offset-zero `iree_atomic_ref_count_t ref_count` field. The VM
type-erased reference base uses an explicit `counter` field in
`runtime/src/iree/vm/ref.h` because VM descriptors store the counter offset.
Other structures are not inferred to be refcounted merely because they mention
the primitive.

Anchored refcounted objects should expose retain/release operations with the
normal C ownership contract:

```c
void iree_hal_buffer_retain(iree_hal_buffer_t* buffer);
void iree_hal_buffer_release(iree_hal_buffer_t* buffer);
```

Factories that publish refcounted objects through `out_*` pointer-to-pointer
parameters use create naming, even when the implementation allocates heap
storage internally:

```c
iree_status_t iree_hal_buffer_view_create(
    iree_hal_buffer_t* buffer,
    iree_hal_buffer_view_t** out_buffer_view);
```

`allocate/free` naming is reserved for storage-oriented resources that are not
themselves refcounted objects, and `initialize/deinitialize` naming is reserved
for caller-owned storage passed as a single pointer.

Retain and release operations that mutate the reference count return `void`.
Release functions are null-safe so cleanup code can call them unconditionally
without adding noisy guards:

```c
iree_hal_buffer_release(buffer);
buffer = NULL;
```

In straight-line code, a direct one-argument release statement consumes that
handle spelling until it is reassigned. Later dereferences or repeated releases
of the same handle are diagnosed:

```c
iree_hal_buffer_release(buffer);
buffer->data = NULL;  // The handle was already released.

iree_hal_buffer_release(buffer);
iree_hal_buffer_release(buffer);  // No remaining modeled reference edge.
```

Explicit direct retains in the same block add modeled reference edges, so code
that deliberately drops multiple owned references stays valid:

```c
iree_hal_buffer_retain(buffer);
iree_hal_buffer_release(buffer);
iree_hal_buffer_release(buffer);
```

This rule is intentionally local. It merges the narrow branch shape where both
arms directly release the same handle, and assignment resets the handle state:

```c
if (condition) {
  iree_hal_buffer_release(buffer);
} else {
  iree_hal_buffer_release(buffer);
}
buffer->data = NULL;  // The handle was released on every branch.

iree_hal_buffer_release(buffer);
buffer = NULL;

iree_hal_buffer_release(buffer);
buffer = replacement_buffer;
```

Partial branch releases do not poison the outer handle state because the local
checker only reports state it can prove on every path through that statement.

If a release drops one retained edge while another owner keeps the object alive,
the code should still avoid reading through the released handle spelling. Take
needed scalar values before the release, release after the last read, or model
the retained edge explicitly with a direct retain/release pair.

The decrement result carries the last-reference decision and must be checked:

```c
if (iree_atomic_ref_count_dec(&buffer->ref_count) == 1) {
  iree_hal_buffer_destroy(buffer);
}
```

Additional atomic counters in refcounted objects should use explicit atomic
integer types such as `iree_atomic_uint32_t`, with names and comments describing
the synchronization contract. Reusing `iree_atomic_ref_count_t` for queue depth,
pending work, or latch state hides the difference between object lifetime and
ordinary atomic coordination.

### `iree-trace-zone-balance`

`iree-trace-zone-balance` treats trace zones as scoped C resources. A terminal
path inside an active zone must end the zone before leaving the function. A
status-returning helper used inside an active zone must use the
trace-zone-aware form so the failure path ends the zone:

```c
IREE_TRACE_ZONE_BEGIN(z0);
IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, do_work());
IREE_TRACE_ZONE_END(z0);
return iree_ok_status();
```

Plain early-return helpers inside an active zone are diagnosed because their
failure path skips the end macro:

```c
IREE_TRACE_ZONE_BEGIN(z0);
IREE_RETURN_IF_ERROR(do_work());  // Use IREE_RETURN_AND_END_ZONE_IF_ERROR.
IREE_TRACE_ZONE_END(z0);
return iree_ok_status();
```

Plain source returns and known return macros such as `HIP_RETURN_ERROR` are
also diagnosed when they leave an active zone:

```c
IREE_TRACE_ZONE_BEGIN(z0);
return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "bad input");
```

Falling off the end of a function with an active zone is diagnosed for the same
reason: a zone is a scoped C resource and the function must close it on every
terminal path.

The reliable shapes are to use `IREE_RETURN_AND_END_ZONE` for immediate return
expressions, `IREE_RETURN_AND_END_ZONE_IF_ERROR` for conditional status helper
returns, or an explicit `IREE_TRACE_ZONE_END` immediately before returning a
status that is already carried in a local variable.

End macros using statically named zones are checked against the active zone
stack. Ending a zone after this function has already ended it, ending an outer
zone while an inner zone is active, or passing the wrong static zone ID to a
return-and-end helper is diagnosed. Dynamic zone ID carriers such as
`iree_zone_id_t zone_id` are treated conservatively because they often model an
explicit ownership transfer across helper boundaries.

The check uses the preprocessor's macro expansion stream so disabled tracing,
HRX wrapper macros, and multi-line helper invocations are modeled by the macro
the developer wrote rather than by the helper's implementation detail. Return
statements generated inside macro bodies are not diagnosed independently; known
return macros are checked at the macro expansion site. Ambiguous block-local,
loop, and switch zone balance is handled conservatively to keep this check high
signal.
