# HIP Passthrough Interception Library

A transparent proxy library for AMD HIP that allows intercepting, logging,
and tracing HIP API calls without modifying the target application.

## Overview

This library replaces `libamdhip64.so` at link/load time, forwarding all HIP
calls to the real HIP library while optionally routing them through a
pluggable **interceptor** library. This enables:

- **API call logging** — trace every HIP call with timestamps and thread IDs.
- **Buffer tracing** — track memory allocations and dump buffer contents
  before/after kernel launches.
- **Custom instrumentation** — write your own interceptor to observe or modify
  HIP behaviour without touching application code.

## Architecture

```
┌─────────────┐       ┌──────────────────┐       ┌──────────────────┐
│ Application │──────▸│ libhip_intercept │──────▸│ Real libamdhip64 │
│             │       │  (passthrough)   │       │   (backend)      │
└─────────────┘       └────────┬─────────┘       └──────────────────┘
                               │
                               ▼ (optional)
                      ┌──────────────────┐
                      │   Interceptor    │
                      │  (logging, etc.) │
                      └──────────────────┘
```

1. **`libhip_intercept.so`** is loaded in place of the real HIP library.
2. On first use it loads the real HIP library from `HIP_PASSTHROUGH_BACKEND_LIB`.
3. All HIP symbols are resolved into a `hip_function_table_t`.
4. If `HIP_PASSTHROUGH_INTERCEPTOR` is set, the interceptor library is loaded
   and given the real function table. The interceptor returns its own table of
   wrapper functions that are used for all subsequent calls.

## Build Targets

| Target                | Output                     | Description |
|-----------------------|----------------------------|-------------|
| `hip_intercept`       | `libhip_intercept.so`      | Main passthrough library — drop-in replacement for `libamdhip64.so`. |
| `hip_logging`         | `libhip_logging.so`        | Interceptor that logs all HIP API calls. |
| `hip_buffer_tracer`   | `libhip_buffer_tracer.so`  | Interceptor that tracks allocations and dumps buffer contents around kernel launches. |
| `hip_noop`            | `libhip_noop.so`           | No-op interceptor — all calls pass through directly. Useful as a template. |

## File Reference

| File | Purpose |
|------|---------|
| `hip_intercept.c` | Full passthrough library with stubs for the complete HIP API surface (~3500 lines, generated). |
| `passthrough.c` | Cleaner passthrough implementation covering the core HIP API via the function table. |
| `simple_passthrough.c` | Minimal passthrough that loads the real library with `RTLD_GLOBAL` and logs a small set of functions. |
| `hip_function_table.h` | Defines `hip_function_table_t`, all HIP type aliases, function pointer typedefs, and the interceptor interface. |
| `hip_logging.c` | Logging interceptor — wraps every function in the table with timestamped log output. |
| `hip_buffer_tracer.c` | Buffer-tracing interceptor — tracks `hipMalloc`/`hipFree`, dumps buffer hex/hash around kernel launches. |
| `stubs_generated.c` | Auto-generated stubs forwarding every HIP symbol through `dlsym` on the backend library. |
| `interceptors/passthrough_interceptor.c` | Minimal no-op interceptor (returns `NULL` → use real table). Good starting point for custom interceptors. |
| `interceptors/logging_interceptor.c` | Standalone logging interceptor with `HIP_LOG_FILE`/`HIP_LOG_LEVEL` support. |
| `add_logging.py` | Script to transform `hip_intercept.c` to inject `pt_log()` calls into every pass-through stub. |
| `passthrough.map` | Linker version script exporting the core HIP symbols. |
| `passthrough_full.map` | Full version script matching `libamdhip64.so` symbol versions (hip\_4.2 through hip\_7.2). |

## Quick Start

### Building

```bash
cd iree-stream
mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja hip_intercept hip_logging hip_buffer_tracer
```

### Running with the passthrough

```bash
# Point to the real HIP library
export HIP_PASSTHROUGH_BACKEND_LIB=/opt/rocm/lib/libamdhip64.so

# (Optional) Load a logging interceptor
export HIP_PASSTHROUGH_INTERCEPTOR=/path/to/build/libhip_logging.so

# Preload or symlink the passthrough library
LD_PRELOAD=/path/to/build/libhip_intercept.so ./my_hip_app
```

## Environment Variables

### Core (libhip\_intercept)

| Variable | Description |
|----------|-------------|
| `HIP_PASSTHROUGH_BACKEND_LIB` | **Required.** Path to the real `libamdhip64.so`. |
| `HIP_PASSTHROUGH_INTERCEPTOR` | Path to an interceptor `.so` to load (optional). |

### Logging Interceptor (libhip\_logging)

| Variable | Default | Description |
|----------|---------|-------------|
| `HIP_LOG_FILE` | stderr | Path to log output file. |
| `HIP_LOG_LEVEL` | `2` | `0`=off, `1`=errors only, `2`=all calls, `3`=verbose. |

### Buffer Tracer Interceptor (libhip\_buffer\_tracer)

| Variable | Default | Description |
|----------|---------|-------------|
| `HIP_TRACE_FILE` | stderr | Path to trace output file. |
| `HIP_TRACE_LEVEL` | `2` | `0`=off, `1`=errors, `2`=calls, `3`=buffers, `4`=verbose. |
| `HIP_TRACE_SYNC` | `0` | `1` = synchronize device before/after every kernel launch. |
| `HIP_TRACE_DUMP` | `0` | `1` = dump buffer hex contents, `2` = dump FNV-1a hash only. |
| `HIP_TRACE_DUMP_MAX` | `1024` | Max bytes to dump per buffer. |
| `HIP_TRACE_KERNEL_FILTER` | (all) | Only trace kernels whose name contains this substring. |
| `HIP_TRACE_KERNEL_COUNT` | `0` | Only trace the first N kernel launches (`0` = unlimited). |
| `HIP_TRACE_KERNEL_FULL_DUMP` | (none) | Colon-separated list of kernel name substrings for full (unlimited size) buffer dumps. |

## Writing a Custom Interceptor

An interceptor is a shared library that exports two functions:

```c
#include "passthrough/hip_function_table.h"

// Required: called once at init time.
// Receives the table of real HIP functions.
// Return your own table to intercept calls, or NULL to use the real table.
hip_function_table_t* hip_interceptor_init(hip_function_table_t* real_functions);

// Optional: called at library unload for cleanup.
void hip_interceptor_shutdown(void);
```

A typical interceptor copies the real table, replaces selected function
pointers with wrapper functions, and returns the modified table:

```c
static hip_function_table_t* g_real = NULL;
static hip_function_table_t  g_wrapper = {0};

hip_function_table_t* hip_interceptor_init(hip_function_table_t* real_functions) {
    g_real = real_functions;
    memcpy(&g_wrapper, g_real, sizeof(g_wrapper));

    // Replace only the functions you care about:
    g_wrapper.hipMalloc = my_wrap_hipMalloc;
    g_wrapper.hipFree   = my_wrap_hipFree;

    return &g_wrapper;
}
```

See `interceptors/passthrough_interceptor.c` for a minimal template and
`interceptors/logging_interceptor.c` for a working example.
