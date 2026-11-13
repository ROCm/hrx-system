// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Diagnostics at indexed input identities, before IR materialization.

#ifndef LOOM_LINK_INDEX_DIAGNOSTICS_H_
#define LOOM_LINK_INDEX_DIAGNOSTICS_H_

#include "loom/error/emitter.h"
#include "loom/error/source.h"
#include "loom/link/module_index.h"

#ifdef __cplusplus
extern "C" {
#endif

// Looks up a borrowed exact snapshot by indexed input and source ID. An invalid
// source ID denotes unavailable text. Strings outlive the materialization.
typedef loom_source_entry_t (*loom_link_source_entry_fn_t)(
    void* user_data, const loom_link_module_index_module_t* module,
    loom_source_id_t source_id);

typedef struct loom_link_diagnostic_environment_t {
  // Destination for structured diagnostics, optional without changing
  // rejection.
  iree_diagnostic_emitter_t emitter;
  // Optional source-ID lookup for exact source excerpts.
  loom_link_source_entry_fn_t source_entry;
  // Embedding state borrowed by source_entry.
  void* user_data;
} loom_link_diagnostic_environment_t;

// Emits at |symbol|'s definition with optional |related| definition provenance.
// NULL symbols represent an invocation-wide error with no authored location.
iree_status_t loom_link_index_emit_diagnostic(
    const loom_link_module_index_t* index,
    loom_link_diagnostic_environment_t environment,
    const loom_link_module_index_symbol_t* symbol,
    const loom_link_module_index_symbol_t* related,
    const loom_error_def_t* error, const loom_diagnostic_param_t* parameters,
    iree_host_size_t parameter_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_LINK_INDEX_DIAGNOSTICS_H_
