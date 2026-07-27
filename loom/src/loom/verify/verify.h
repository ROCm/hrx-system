// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Single-pass O(N) IR verifier for loom modules.
//
// ==========================================================================
// Overview
// ==========================================================================
//
// The verifier walks every operation in a module exactly once, checking:
//
//   Structural consistency
//     Operand/result/attr/region counts match the op vtable.
//     Variadic fields satisfy minimum counts.
//     Terminators present where required.
//     Single-block regions contain exactly one block.
//
//   Type constraints
//     Each operand and result type satisfies its declared constraint
//     (TILE, INTEGER, FLOAT, etc.) from the vtable's field descriptors.
//
//   Semantic constraints (table-driven)
//     SameType, SameElementType, SameEncoding, SameShape, RanksMatch,
//     OffsetCountMatchesRank, DimIndexInBounds, AllShapesMatch,
//     BlockArgCount, BlockArgsMatchElementTypes,
//     YieldCountMatchesResults, YieldTypesMatchResults.
//     Checked by a single interpreter walking per-op constraint tables.
//
//   SSA dominance
//     Every use of a value is dominated by its definition. In loom's
//     structured IR, this means: defined in the same block before the
//     use, or defined in an ancestor block/region.
//
//   Linear ownership (tied results)
//     Operands consumed by tied results are not used after the
//     consuming op. Tied result indices are in range and refer to
//     valid operands.
//
//   Symbol references
//     Every symbol reference (@name) resolves to a symbol in the
//     module's symbol table. The symbol kind matches the usage
//     context (e.g., func.call targets must be functions).
//
//   Op-specific verification (escape hatch)
//     Ops with a verify callback on their vtable get that callback
//     invoked after all table-driven checks pass.
//
// ==========================================================================
// Single-pass design
// ==========================================================================
//
// The verifier makes exactly one walk over the IR: module body →
// blocks → ops, recursing into regions. Total cost is O(N) where N
// is the number of ops. There are no separate passes for dominance,
// type checking, or constraint verification — everything is checked
// as each op is visited.
//
// SSA scope tracking uses a bitset (one bit per value_id in the
// module) with a watermark stack for region entry/exit:
//
//   Enter region: push current defined-list watermark.
//   Process ops:  set defined bit for each result, check operands.
//   Exit region:  pop watermark, clear bits for values defined inside.
//
// This gives O(1) dominance checks (bit test) and O(values-in-region)
// cleanup on region exit, which is linear in the total IR size.
//
// ==========================================================================
// Diagnostics
// ==========================================================================
//
// All verifier errors are structured: each error site produces a
// loom_diagnostic_t with a typed error definition, typed parameters,
// and the emitter tag LOOM_EMITTER_VERIFIER. Sinks render the
// human-readable message from the error definition and params.
//
// Source ranges for caret underlining come from the op's location resolved
// through an optional source resolver callback. When no resolver is provided
// (or the location is unknown), the verifier falls back to canonical printed
// IR when an op is available and tags the range provenance accordingly. If no
// op location can be reconstructed, the diagnostic still carries its
// structured identity and explicit `unavailable_source` provenance.
//
// Diagnostics are emitted through a callback sink (same pattern as
// the parser). The verifier continues after errors to report as many
// issues as possible in a single pass. A max_errors limit prevents
// runaway output on badly malformed IR.
//
// ==========================================================================
// Source resolution
// ==========================================================================
//
// The verifier separates location *provenance* (on the op) from
// source text *availability* (in the pipeline). An op's location says
// "I came from model.loom:42:3". The source resolver says "here's
// the text of model.loom so you can underline it."
//
// The common case: the parser consumes source text, builds IR with
// file locations on each op, and the first verifier pass runs while
// the source buffer is still alive. The parser hands the source
// buffer(s) to the verifier via the resolver. After passes that
// transform the IR, the op locations still reference the original
// source positions — as long as the source buffers are retained,
// later verification passes get carets too.
//
// For linked modules with ops from multiple source files, the
// resolver carries a table of (source_id, buffer) entries so each
// op's location resolves against the correct source.
//
// When the resolver can't find original source text for an op, the
// verifier falls back to printing the op and using the printed
// representation as the diagnostic source. This guarantees every
// diagnostic has a source line with carets — even for ops created
// by passes that never existed in any input file.
//
// ==========================================================================
// Usage
// ==========================================================================
//
//   // Verification without carets (e.g., programmatic consumer):
//   loom_verify_options_t options = {
//       .sink = my_sink,
//       .max_errors = 20,
//   };
//
//   // Verification with carets (the common compilation path):
//   loom_source_entry_t sources[] = {
//       {0, source_text, filename},
//   };
//   loom_source_table_resolver_t resolver_data = {
//       .entries = sources,
//       .count = 1,
//   };
//   loom_verify_options_t options = {
//       .sink = my_sink,
//       .max_errors = 20,
//       .source_resolver = {loom_source_table_resolve, &resolver_data},
//   };
//
//   loom_verify_result_t result;
//   iree_status_t status = loom_verify_module(module, &options, &result);
//   // status is ok even if verification found errors (check result).
//   // status is non-ok only on internal verifier failures (OOM, etc.).
//   if (result.error_count > 0) { /* module is invalid */ }

#ifndef LOOM_VERIFY_VERIFY_H_
#define LOOM_VERIFY_VERIFY_H_

#include "iree/base/api.h"
#include "loom/error/diagnostic.h"
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Source resolution
//===----------------------------------------------------------------------===//

// Resolves an op's location to a source range for caret rendering.
// Returns true if the location could be resolved (source text is
// available and the location points into it). When false, the
// verifier emits diagnostics without source ranges.
typedef bool (*loom_source_resolver_fn_t)(void* user_data,
                                          const loom_module_t* module,
                                          loom_location_id_t location,
                                          loom_source_range_t* out_range);

// A source resolver: callback + context. Passed to the verifier at
// creation time. The verifier resolves op locations through this.
typedef struct loom_source_resolver_t {
  loom_source_resolver_fn_t fn;
  void* user_data;
} loom_source_resolver_t;

// Resolves a location through the resolver. If fn is NULL, returns
// false (no source available).
static inline bool loom_source_resolve(loom_source_resolver_t resolver,
                                       const loom_module_t* module,
                                       loom_location_id_t location,
                                       loom_source_range_t* out_range) {
  if (resolver.fn) {
    return resolver.fn(resolver.user_data, module, location, out_range);
  }
  return false;
}

// A single source buffer entry: maps a source_id to the text that
// was parsed to produce ops referencing that source.
typedef struct loom_source_entry_t {
  loom_source_id_t source_id;
  iree_string_view_t source;
  iree_string_view_t filename;
} loom_source_entry_t;

// Table of source buffers for the built-in resolver. In the common
// case (single parse → verify), this has one entry. After linking
// modules from multiple source files, one entry per file.
typedef struct loom_source_table_resolver_t {
  const loom_source_entry_t* entries;
  iree_host_size_t count;
} loom_source_table_resolver_t;

// Built-in source resolver that resolves file locations by looking up
// the source_id in a table of source buffers. Pass a
// loom_source_table_resolver_t* as user_data.
bool loom_source_table_resolve(void* user_data, const loom_module_t* module,
                               loom_location_id_t location,
                               loom_source_range_t* out_range);

//===----------------------------------------------------------------------===//
// Verification options
//===----------------------------------------------------------------------===//

typedef struct loom_verify_options_t {
  // Diagnostic sink for error/warning/note messages. If NULL,
  // diagnostics are silently counted but not emitted.
  loom_diagnostic_sink_t sink;

  // Maximum number of errors before aborting the walk. The verifier
  // attempts to report as many errors as possible per pass, but stops
  // after this many to avoid flooding output on badly malformed IR.
  // 0 = unlimited (report all errors).
  uint32_t max_errors;

  // Optional source resolver for caret diagnostics. When fn is
  // non-NULL, the verifier calls this to resolve op locations into
  // source ranges for caret underlining. When fn is NULL, diagnostics
  // carry the message and error identity without source position.
  loom_source_resolver_t source_resolver;
} loom_verify_options_t;

//===----------------------------------------------------------------------===//
// Verification result
//===----------------------------------------------------------------------===//

typedef struct loom_verify_result_t {
  uint32_t error_count;
  uint32_t warning_count;
} loom_verify_result_t;

//===----------------------------------------------------------------------===//
// Verification entry points
//===----------------------------------------------------------------------===//

// Verifies an entire module in a single O(N) pass.
//
// Returns iree_ok_status() even if verification errors are found —
// check result->error_count for the number of errors. Returns a
// non-ok status only on internal verifier failures (out of memory,
// missing vtable registrations, etc.).
//
// The module must have a valid context with registered op vtables.
iree_status_t loom_verify_module(const loom_module_t* module,
                                 const loom_verify_options_t* options,
                                 loom_verify_result_t* out_result);

// Verifies a single function within a module.
//
// Checks everything loom_verify_module checks for ops within the
// function body, plus function-level invariants (signature
// consistency, named dim validity). Does NOT check module-level
// properties (symbol table completeness, etc.).
//
// Useful for incremental verification after a pass modifies one
// function.
iree_status_t loom_verify_function(const loom_module_t* module,
                                   loom_func_like_t function,
                                   const loom_verify_options_t* options,
                                   loom_verify_result_t* out_result);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_VERIFY_VERIFY_H_
