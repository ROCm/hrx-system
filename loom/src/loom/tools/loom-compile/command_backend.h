// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Portable command artifact emission for loom-compile.

#ifndef LOOM_TOOLS_LOOM_COMPILE_COMMAND_BACKEND_H_
#define LOOM_TOOLS_LOOM_COMPILE_COMMAND_BACKEND_H_

#include "iree/base/api.h"
#include "loom/error/diagnostic.h"
#include "loom/tooling/execution/session.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_compile_command_backend_options_t {
  // Directory receiving canonical root artifact files.
  iree_string_view_t artifact_directory;

  // Path receiving the artifact-set manifest.
  iree_string_view_t manifest_path;

  // Final sink receiving materialized command diagnostics.
  loom_diagnostic_sink_t diagnostic_sink;

  // Original-source resolver for linked command operations.
  loom_source_resolver_t source_resolver;

  // Maximum command diagnostics to emit before stopping.
  uint32_t max_errors;
} loom_compile_command_backend_options_t;

// Emits all retained public command roots from an expanded-source module.
//
// Root artifacts are written to |options->artifact_directory| using canonical
// ordinal filenames. The schema-versioned manifest at |manifest_path| maps
// root symbols to those files and records the shared executable-entry table.
// Kernel implementation bodies are not opened or emitted by this backend.
//
// Source contract failures emit diagnostics, leave |out_emitted| false, and
// return OK. Infrastructure and filesystem failures return a non-OK status.
iree_status_t loom_compile_command_backend_emit(
    loom_run_session_t* session, loom_run_module_t* run_module,
    const loom_compile_command_backend_options_t* options, bool* out_emitted,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_COMPILE_COMMAND_BACKEND_H_
