// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Clone analysis over parsed Loom modules.
//
// The analyzer finds operation sequences that are written more than once so
// authors can replace the copies with one motif (`func.def inline` or a
// `template.def` provider reached through `template.apply`). It is a read-only
// static analysis: modules are borrowed, never mutated, and results never feed
// compilation or template provider selection.
//
// Sequences are contiguous operations of one block. Two sequences are clones
// when they are equal up to SSA renaming:
//   - Each operation prints identically once SSA names, block labels and
//     comments are removed. Symbols, strings, constants, attributes and types
//     therefore compare by their spelled content, independent of the module
//     that interned them.
//   - Every SSA reference (operands, values embedded in result and block
//     argument types, and values embedded in attributes) resolves to the same
//     window-relative identity: the Nth value defined inside the sequence, or
//     the Nth distinct value flowing in from outside it. Operands of
//     commutative operations compare as a multiset.
//   - Nested regions compare recursively, so a whole `scf.if` matches as a
//     unit, and sequences inside nested blocks are candidates too.
//
// Module top-level blocks are not candidates: repeated `template.decl` and
// `func.decl` headers are required redeclarations, not reusable code.
//
// Clone size counts every operation in the sequence including nested ones.
// Findings are maximal: a class whose occurrences all lie inside occurrences
// of a larger reported class is dropped.

#ifndef LOOM_TOOLS_LOOM_ANALYZE_CLONE_ANALYSIS_H_
#define LOOM_TOOLS_LOOM_ANALYZE_CLONE_ANALYSIS_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Stable rule identities. Spellings are part of the tool's output contract.
typedef enum loom_analyze_rule_e {
  // A sequence of at least min_op_count operations appears at least
  // min_occurrence_count times.
  LOOM_ANALYZE_RULE_DUPLICATE_SEQUENCE = 0,
  // A sequence equals the complete body of an existing func.def or
  // template.def (excluding its terminator) and can call that motif instead.
  LOOM_ANALYZE_RULE_REIMPLEMENTED_MOTIF = 1,
  LOOM_ANALYZE_RULE_COUNT_,
} loom_analyze_rule_t;

// Returns the stable rule ID, e.g. "loom-analyze/duplicate-sequence".
iree_string_view_t loom_analyze_rule_id(loom_analyze_rule_t rule);

// Returns a one-sentence rule description for help and SARIF metadata.
iree_string_view_t loom_analyze_rule_description(loom_analyze_rule_t rule);

// Comparison relaxations, combinable as a bitmask.
enum loom_analyze_parameterization_bits_e {
  // Operations compare exactly up to SSA renaming.
  LOOM_ANALYZE_PARAMETERIZE_NONE = 0,
  // `*.constant` operations additionally compare without their literal value,
  // treating differing constants as motif parameters.
  LOOM_ANALYZE_PARAMETERIZE_CONSTANTS = 1u << 0,
  // Element types (i16, f32, bf16, the f32 of vector<4xf32>, ...) compare as
  // one placeholder. Shapes, layouts and type constructors still compare
  // exactly. Loom motifs cannot be generic over element types today, so such
  // findings are reported but never applicable.
  LOOM_ANALYZE_PARAMETERIZE_TYPES = 1u << 1,
  LOOM_ANALYZE_PARAMETERIZE_ALL_ = (1u << 2) - 1,
};
typedef uint32_t loom_analyze_parameterization_t;

// Parses a comma-separated list of "none", "constants" and "types".
iree_status_t loom_analyze_parse_parameterization(
    iree_string_view_t value, loom_analyze_parameterization_t* out_value);

// Returns "", "constants", "types" or "constants,types".
iree_string_view_t loom_analyze_parameterization_name(
    loom_analyze_parameterization_t value);

typedef struct loom_analyze_options_t {
  // Minimum clone size in operations, counting nested operations.
  uint32_t min_op_count;
  // Minimum number of non-overlapping occurrences reported as a clone.
  uint32_t min_occurrence_count;
  // Operation comparison relaxation.
  loom_analyze_parameterization_t parameterization;
} loom_analyze_options_t;

// Initializes |out_options| with the documented CLI defaults.
void loom_analyze_options_initialize(loom_analyze_options_t* out_options);

// One SSA value crossing an occurrence boundary, as authored in its source.
typedef struct loom_analyze_binding_t {
  // Authored value name without the '%' sigil, borrowed from the module; empty
  // for unnamed values.
  iree_string_view_t name;
  // Printed value type, owned by the analyzer. Types embedding SSA values
  // spell this occurrence's names, e.g. view<[%span]xi16>.
  iree_string_view_t type;
} loom_analyze_binding_t;

typedef struct loom_analyze_occurrence_t {
  // Source file containing the occurrence, borrowed from its module.
  iree_string_view_t file;
  // One-based start line and column of the first operation, or zero.
  uint32_t start_line;
  uint32_t start_column;
  // One-based end line and exclusive end column of the last operation's
  // source range, or zero. Together with the start this is the exact text a
  // rewriter replaces with a call.
  uint32_t end_line;
  uint32_t end_column;
  // Values flowing into the occurrence, in canonical motif parameter order:
  // entry j of every occurrence of a finding binds the same parameter.
  loom_analyze_binding_t* inputs;
  uint32_t input_count;
  // Values defined by the occurrence and used after it, in definition order:
  // the names a replacing call must bind, one per motif result.
  loom_analyze_binding_t* outputs;
  uint32_t output_count;
  // Symbol of the enclosing top-level definition, e.g. "ggml_cpy_f32", or
  // empty when the occurrence is not inside a named definition.
  iree_string_view_t symbol;
  // True when the occurrence is inside a kernel.def launch-config region,
  // where function and template calls are not available today.
  bool in_launch_config;
  // True when the occurrence is the complete body of the reported motif.
  bool is_motif_definition;
} loom_analyze_occurrence_t;

typedef struct loom_analyze_finding_t {
  // Rule that produced the finding.
  loom_analyze_rule_t rule;
  // Operations per occurrence, including nested operations.
  uint32_t op_count;
  // Stable content fingerprint of the canonical sequence. It is independent of
  // file names, SSA names and process state, so it identifies the same clone
  // across runs for baselines and SARIF partial fingerprints.
  uint64_t fingerprint;
  // Relaxations the copies actually needed to match: zero for exact clones
  // even when relaxations were enabled.
  loom_analyze_parameterization_t parameterized_by;
  // False when no Loom motif can replace the copies today: they differ in
  // element types. Inapplicable findings have zero removable_op_count and
  // never join the plan.
  bool applicable;
  // Operations the refactor would remove if motifs could be generic over the
  // differing element types; equals removable_op_count when applicable.
  uint64_t type_generic_removable_op_count;
  // Distinct values flowing into the sequence: the parameters a motif
  // replacing it takes (before any constant parameterization).
  uint32_t input_count;
  // Top-level constants whose values differ between copies. They stay at each
  // call site and are passed to the motif. Zero unless constants are
  // parameterized.
  uint32_t call_site_constant_count;
  // Values defined by the sequence and used after it: the results a motif
  // replacing it returns. Measured on the first occurrence.
  uint32_t output_count;
  // Estimated operations removed by the refactor the rule suggests, counting
  // only callable copies (launch-config copies cannot become calls). For a
  // duplicate sequence with c callable copies: each becomes one call and one
  // motif body is added, c * op_count - (op_count + c). Each of the k
  // call-site constants stays in every copy and leaves the motif body:
  // c * op_count - (op_count - k + c * (k + 1)). Copies differing in a
  // constant nested in a region remove nothing, since that constant cannot
  // stay at the call site. For a reimplemented motif the body already exists:
  // each callable copy saves op_count - 1. Never negative.
  uint64_t removable_op_count;
  // True when the finding belongs to the analyzer's refactor plan: a greedy
  // selection, largest removable_op_count first, of findings whose callable
  // copies cover disjoint operations. Selected findings are independent edits
  // that can be applied together in any order.
  bool selected;
  // Motif symbol for REIMPLEMENTED_MOTIF findings; empty otherwise.
  iree_string_view_t motif_symbol;
  // Occurrences sorted by file, line and column. For REIMPLEMENTED_MOTIF the
  // motif definition is included and flagged by is_motif_definition.
  loom_analyze_occurrence_t* occurrences;
  // Number of entries in occurrences.
  uint32_t occurrence_count;
} loom_analyze_finding_t;

typedef struct loom_analyze_summary_t {
  // Modules analyzed.
  uint32_t module_count;
  // Operations inside definitions (excludes module top-level declarations).
  uint64_t op_count;
  // Distinct operations covered by at least one reported occurrence.
  uint64_t duplicated_op_count;
  // Sum of removable_op_count over selected findings: the operations removed
  // by applying the whole refactor plan.
  uint64_t removable_op_count;
  // Number of selected findings.
  uint32_t selected_count;
  // Additional operations removable if motifs could be generic over element
  // types: a second greedy pass over type-variant findings, restricted to
  // operations the plan leaves untouched. Zero unless types are parameterized.
  uint64_t type_generic_removable_op_count;
} loom_analyze_summary_t;

typedef struct loom_analyze_result_t {
  // Findings sorted by first occurrence location, then decreasing size.
  loom_analyze_finding_t* findings;
  // Number of entries in findings.
  iree_host_size_t finding_count;
  // Whole-input statistics.
  loom_analyze_summary_t summary;
} loom_analyze_result_t;

typedef struct loom_analyze_analyzer_t loom_analyze_analyzer_t;

// Creates an analyzer. Options are copied.
iree_status_t loom_analyze_analyzer_create(
    const loom_analyze_options_t* options, iree_allocator_t allocator,
    loom_analyze_analyzer_t** out_analyzer);

// Releases the analyzer, including any result it produced.
void loom_analyze_analyzer_free(loom_analyze_analyzer_t* analyzer);

// Indexes one verified module. |module| is borrowed and must stay alive and
// unmodified until the analyzer is freed; results borrow its source names.
iree_status_t loom_analyze_analyzer_add_module(
    loom_analyze_analyzer_t* analyzer, const loom_module_t* module);

// Runs clone detection over all added modules. |out_result| is owned by the
// analyzer and remains valid until it is freed. May be called once.
iree_status_t loom_analyze_analyzer_run(
    loom_analyze_analyzer_t* analyzer,
    const loom_analyze_result_t** out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_ANALYZE_CLONE_ANALYSIS_H_
