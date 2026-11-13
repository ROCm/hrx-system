// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structural, type, SSA-availability and ownership verification for Loom IR.
//
// ==========================================================================
// Overview
// ==========================================================================
//
// The verifier checks each operation and recursively verifies its regions:
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
//     YieldCountMatches, YieldTypesMatch.
//     Checked by a single interpreter walking per-op constraint tables.
//
//   SSA dominance
//     Every use of a value is dominated by its definition. In loom's
//     structured IR, values are visible after their definition and in nested
//     regions. In multi-block regions, the defining block must also dominate
//     the reachable use block, independently of physical block order. Dead
//     blocks follow the text serialization order: reachable definitions and
//     earlier dead definitions are available, but later dead definitions are
//     not.
//
//   Linear ownership transfers
//     Operands consumed by tied or moved results and owning managed callable
//     fields are not used after the consuming op, including through required
//     nonwriting storage aliases. Each consuming field needs an independent
//     owner; borrowed callable fields preserve the original owner.
//     Independent copies and the consuming operation's results have separate
//     ownership. Mutually exclusive paths may each consume the same owner;
//     recurrence is cut by the original owner's definition, not an alias.
//     Tied result indices are in range and refer to valid operands.
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
// Analysis and scope ownership
// ==========================================================================
//
// Canonical type and symbol facts are prepared before the operation walk.
// A one-block region needs no graph for dominance; consumption extracts its
// CFG only when required. Each multi-block region builds one graph and
// dominator tree, shared with ownership verification. Graph
// extraction is linear in the region's operations and successor edges;
// dominance construction takes O(B + E log B) time and O(B) additional arena
// space for B blocks and E edges. Dialect callbacks and ownership queries have
// their own costs; this is not a whole-verifier constant-work-per-op guarantee.
// Consumers are grouped by required storage owner, and each consumed family's
// use lists are distributed once into retained region/block summaries. Each
// touched region checks the union of its consumers' reachable paths, visiting
// a block/edge at most once per owner. Existing component and DFS facts bound
// that search to possible observations. Nested regions propagate observations
// and returning consumption to their parent; declared repeated and exiting
// execution preserve the dynamic owner's lifetime across region boundaries.
//
// SSA scope tracking uses a definition-depth table (one byte per value_id in
// the module) and definition-stack watermarks:
//
//   Enter region: push current defined-list watermark.
//   Enter block:  restore the immediate dominator's ending watermark.
//   Process ops:  record definition depth for each result, check operands.
//   Exit region:  pop watermark, clear depths for values defined inside.
//
// Blocks are visited in dominator-tree preorder, preserving only ancestor
// definitions between reachable blocks. Unreachable blocks can reference direct
// definitions from reachable blocks and earlier unreachable blocks in their
// region, including for continuations made dead by non-returning calls. A
// linear declaration inventory is retained only for regions with unreachable
// blocks. Dead blocks are visited in physical order, matching their
// serialization order. Their local definitions still become visible in
// operation order; nested regions retain their lexical boundaries. CFG depth
// does not consume the nested-region scope limit or the C call stack. Every
// definition is pushed and removed once, and each operand availability check is
// O(1).
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
// representation in diagnostic.origin for carets, explicitly labeled as IR.
// diagnostic.source_location retains the recorded original filename and
// coordinates independently of that fallback.
//
// ==========================================================================
// Usage
// ==========================================================================
//
//   // Verification with locations and printed IR (no original source bytes):
//   loom_verify_options_t options = {
//       .sink = my_sink,
//       .max_errors = 20,
//   };
//
//   // Verification with original source excerpts and carets:
//   loom_source_entry_t sources[] = {
//       {0, source_text, filename},
//   };
//   loom_source_table_resolver_t resolver_data = {
//       .module = module,
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
#include "loom/error/source.h"
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Verification options
//===----------------------------------------------------------------------===//

typedef struct loom_verify_options_t {
  // Diagnostic sink for error/warning/note messages. If NULL,
  // diagnostics are silently counted but not emitted.
  loom_diagnostic_sink_t sink;

  // Maximum number of errors before aborting the walk. The verifier
  // attempts to report as many errors as possible per pass, but stops
  // after this many to avoid flooding output on badly malformed IR. The limit
  // also bounds diagnostics and source rendering within one operation, even
  // when no sink is installed. No further diagnostics are counted or emitted
  // after reaching the error limit.
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

// Verifies selected functions with one shared module-fact setup. The functions
// and module remain immutable for the call; results and the error limit span
// the entire batch. This has the same scope as loom_verify_function and avoids
// repeating module-wide symbol/encoding preparation for each function.
iree_status_t loom_verify_functions(const loom_module_t* module,
                                    const loom_func_like_t* functions,
                                    iree_host_size_t function_count,
                                    const loom_verify_options_t* options,
                                    loom_verify_result_t* out_result);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_VERIFY_VERIFY_H_
