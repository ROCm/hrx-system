// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_SYMBOL_INLINE_CALLABLES_H_
#define LOOM_TRANSFORMS_SYMBOL_INLINE_CALLABLES_H_

#include "loom/analysis/symbol_references.h"
#include "loom/pass/types.h"
#include "loom/target/function_version.h"

#ifdef __cplusplus
extern "C" {
#endif

// Establishes applicability for one template call in the borrowed snapshot.
// An unproved call stays intact, including when its enclosing body is cloned
// or moved. Allocation/analysis failures propagate separately from eligibility.
typedef struct loom_inline_callables_eligibility_callback_t {
  // Called once per reachable template call during plan construction.
  iree_status_t (*fn)(void* user_data, loom_symbol_id_t source_symbol_id,
                      loom_op_t* call_op, bool* out_eligible);
  // Borrowed state owned by the producer of applicability decisions.
  void* user_data;
} loom_inline_callables_eligibility_callback_t;

// Inputs for one transient call plan. All snapshots describe the same module
// and remain valid until execution completes; symbols must not be compacted
// between construction and execution.
typedef struct loom_inline_callables_plan_options_t {
  // Borrowed references before any application-to-call rewrites.
  const loom_symbol_reference_table_t* references;
  // Borrowed function versions for target policy and consuming-inline cleanup.
  const loom_target_function_version_snapshot_t* target_versions;
  // Optional byte per symbol restricting planning to live source definitions.
  // NULL includes all definitions. Module-root references are always included.
  const uint8_t* live_symbols;
  // Optional bit per family with remaining template.apply demands. Such a
  // family's providers must remain available for subsequent selection.
  const uint64_t* demanded_families;
  // Applicability producer for existing template calls. With no callback,
  // template calls remain intact; ordinary call policy is unaffected.
  loom_inline_callables_eligibility_callback_t template_eligibility;
  // Capacity for newly selected calls appended before execution.
  iree_host_size_t additional_call_capacity;
  // Applies target emission requirements in addition to authored call policy.
  bool target_policy;
} loom_inline_callables_plan_options_t;

// Arena-owned call plan shared with the driver that establishes applicability.
typedef struct loom_inline_callables_plan_t loom_inline_callables_plan_t;

// Builds the common inline plan using producer-owned references and facts.
// Planning borrows |options|' snapshots and allocates in |pass->arena|. It does
// not mutate the module. The caller may then replace selected applications
// with exact calls and append them without rebuilding the reference snapshot.
iree_status_t loom_inline_callables_plan_create(
    loom_pass_t* pass, loom_module_t* module,
    const loom_inline_callables_plan_options_t* options,
    loom_inline_callables_plan_t** out_plan);

// Appends an applicable template call replacing an application in the original
// snapshot. Its selection owner supplies the original containing symbol and
// selected provider directly. Each call consumes one reserved capacity slot;
// no lookup, applicability reevaluation, or allocation occurs here.
void loom_inline_callables_plan_append(loom_inline_callables_plan_t* plan,
                                       loom_symbol_id_t source_symbol_id,
                                       loom_symbol_id_t target_symbol_id,
                                       loom_op_t* call_op);

// Checks required-edge cycles and body legality, then executes the existing
// clone/consume scheduler. Emits diagnostics through the owning pass and marks
// that pass changed on mutation. The caller owns pruning and symbol compaction
// after this returns; the plan and borrowed snapshots are then no longer used.
iree_status_t loom_inline_callables_plan_execute(
    loom_inline_callables_plan_t* plan);

// Returns the callable inliner pass metadata.
const loom_pass_info_t* loom_inline_callables_pass_info(void);

// Creates an inline-callables pass instance.
iree_status_t loom_inline_callables_create(loom_pass_t* pass,
                                           iree_string_view_t options);

// Runs module-level required callable inlining. Command program calls expand
// with their serial ordering intact. Template calls stay intact;
// select-templates{rewrite=inline} owns their applicability and expansion.
iree_status_t loom_inline_callables_run(loom_pass_t* pass,
                                        loom_module_t* module);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_SYMBOL_INLINE_CALLABLES_H_
