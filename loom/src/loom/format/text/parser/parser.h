// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PARSER_PARSER_H_
#define LOOM_FORMAT_TEXT_PARSER_PARSER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/diagnostic.h"
#include "loom/format/text/low_asm.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Options controlling parser behavior.
typedef struct loom_text_parse_options_t {
  // Diagnostic sink for emitting errors, warnings, and remarks.
  // If fn is NULL, diagnostics are silently dropped.
  loom_diagnostic_sink_t diagnostic_sink;

  // Maximum number of errors before the parser stops. 0 = unlimited.
  // Default: 20. When the limit is reached, ERR_PARSE_012 is emitted
  // and parsing halts.
  uint32_t max_errors;

  // Optional environment used by low.asm region syntax. The environment is
  // parse-time context only: parsed IR contains canonical low ops with
  // descriptor keys, not a persistent reference to the selected set.
  loom_text_low_asm_environment_t low_asm_environment;
} loom_text_parse_options_t;

// Parses loom IR text into an in-memory module.
//
// Single-pass, table-driven: recursive descent for structure (module,
// functions, regions, blocks), format element walker for op interiors
// (same .rodata tables the printer uses). Two arenas: module arena
// (persistent IR), parser arena (transient scope/accumulator storage).
//
// Security model: input is untrusted. Every token, integer, and index
// is validated before use. Failures produce structured diagnostics
// through |options->diagnostic_sink| and never abort or crash.
//
// Error recovery: on op parse failure, sync to the next newline and
// continue. On region parse failure, sync to the matching '}' and
// continue. The parser continues after recoverable errors up to
// |options->max_errors| (default 20).
//
// The source buffer must remain valid for the duration of the call.
// The context must be finalized (all dialects registered). The block
// pool provides arena storage for the module and parser temporaries.
// |options| may be NULL for defaults (no sink, 20 error limit).
//
// On a clean parse, |*out_module| is set to a freshly allocated module
// owned by the caller (free with loom_module_free()).
//
// When parse errors are emitted through the diagnostic sink, |*out_module|
// is NULL and the function still returns iree_ok_status() — parse errors
// are not infrastructure failures. The caller checks |*out_module| to
// distinguish success from parse failure.
//
// Returns non-ok status only for infrastructure failures (OOM, missing
// dialect registrations). In that case |*out_module| is also NULL.
iree_status_t loom_text_parse(iree_string_view_t source,
                              iree_string_view_t filename,
                              loom_context_t* context,
                              iree_arena_block_pool_t* block_pool,
                              const loom_text_parse_options_t* options,
                              loom_module_t** out_module);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_FORMAT_TEXT_PARSER_PARSER_H_
