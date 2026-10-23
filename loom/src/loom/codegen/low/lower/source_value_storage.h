// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-value materialization demand analysis.
//
// Selected source plans are walked backward once to retain every source value
// required by target-Low emission. Fact-derived references can invalidate a
// selected graph when that graph does not bind the referenced source result;
// callers then refine selection against the retained demand bits without
// rematching source IR.

#ifndef LOOM_CODEGEN_LOW_LOWER_SOURCE_VALUE_STORAGE_H_
#define LOOM_CODEGEN_LOW_LOWER_SOURCE_VALUE_STORAGE_H_

#include "loom/codegen/low/lower/rules.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |source_op| must emit independently of result demand.
bool loom_low_lower_source_value_storage_op_requires_emission(
    const loom_op_t* source_op, loom_trait_flags_t traits);

// Returns true when a source identity also aliases its target storage. Fact
// refinements preserve storage; an explicit value-alias target contract may own
// a physical realization at its boundary.
bool loom_low_lower_source_value_storage_op_is_identity(
    const loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_trait_flags_t traits);

// Returns true when |source_value_id| needs a materialized target-Low result.
bool loom_low_lower_source_value_storage_result_required(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id);

// Returns true when |rule| binds every fact-referenced result in its source
// graph. False disables the rule while refining retained graph selection.
bool loom_low_lower_source_value_storage_rule_preserves_fact_demands(
    const loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_op_t* const* source_nodes, uint8_t source_node_count,
    const loom_low_lower_rule_t* rule);

// Marks source-value demands for one structural operation encountered during
// source-plan preparation.
void loom_low_lower_source_value_storage_mark_structural_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_trait_flags_t traits);

// Propagates source-value storage demands backward through selected plans and
// marks pure plans with no materialized result as elided.
void loom_low_lower_source_value_storage_analyze(
    loom_low_lower_context_t* context);

// Returns true when every selected rule preserves current fact-derived storage
// demands.
bool loom_low_lower_source_value_storage_plans_preserve_fact_demands(
    const loom_low_lower_context_t* context);

// Resets transient materialization state before selection refinement while
// preserving function-boundary and fact-derived requirements.
void loom_low_lower_source_value_storage_prepare_refinement(
    loom_low_lower_context_t* context);

// Recomputes source demands after a deferred target replacement without
// rerunning rule selection. Newly demanded producers are reactivated, claimed
// results remain elided, and the reverse walk propagates the new demands.
void loom_low_lower_source_value_storage_refine_selected_plans(
    loom_low_lower_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_SOURCE_VALUE_STORAGE_H_
