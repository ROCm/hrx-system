// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/diagnostics.h"

#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"

static iree_status_t loom_low_allocation_emit_predicted_spills(
    const loom_low_allocation_table_t* table,
    iree_diagnostic_emitter_t emitter) {
  for (iree_host_size_t i = 0; i < table->spill_plan_count; ++i) {
    const loom_low_allocation_spill_plan_t* spill_plan = &table->spill_plans[i];
    const loom_low_allocation_assignment_t* assignment =
        &table->assignments[spill_plan->assignment_index];
    loom_diagnostic_param_t params[] = {
        loom_param_string(loom_low_diagnostic_target_key(&table->target)),
        loom_param_string(loom_low_diagnostic_export_name(&table->target)),
        loom_param_string(loom_low_diagnostic_config_key(&table->target)),
        loom_param_string(loom_low_diagnostic_function_name(
            table->module, table->function_op)),
        loom_param_string(loom_low_diagnostic_value_name(table->module,
                                                         spill_plan->value_id)),
        loom_param_string(loom_low_diagnostic_value_class_name(
            table->target.descriptor_set, assignment->value_class)),
        loom_param_u32(spill_plan->byte_size),
        loom_param_u32(spill_plan->store_count),
        loom_param_u32(spill_plan->reload_count),
        loom_param_string(IREE_SV("register-budget-conflict")),
    };
    loom_diagnostic_emission_t emission = {
        .op = loom_low_diagnostic_value_origin_op(
            table->module, spill_plan->value_id, table->function_op),
        .error = LOOM_ERR_BACKEND_008,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    IREE_RETURN_IF_ERROR(iree_diagnostic_emit(emitter, &emission));
  }
  return iree_ok_status();
}

static iree_string_view_t loom_low_allocation_copy_decision_name(
    loom_low_allocation_copy_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_ALLOCATION_COPY_COALESCED:
      return IREE_SV("accepted");
    case LOOM_LOW_ALLOCATION_COPY_MATERIALIZED:
      return IREE_SV("rejected");
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t loom_low_allocation_coalescing_constraint(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_copy_decision_t* copy_decision) {
  const loom_low_allocation_assignment_t* source_assignment =
      &table->assignments[copy_decision->source_assignment_index];
  const loom_low_allocation_assignment_t* result_assignment =
      &table->assignments[copy_decision->result_assignment_index];
  if (!loom_low_allocation_assignment_is_register_like(source_assignment)) {
    return IREE_SV("source-not-register-like");
  }
  if (!loom_low_allocation_assignment_is_register_like(result_assignment)) {
    return IREE_SV("result-not-register-like");
  }
  if (loom_low_allocation_storage_assignment_locations_share(
          table->target.descriptor_set, source_assignment, result_assignment)) {
    return IREE_SV("assigned-locations-match");
  }
  return IREE_SV("assigned-locations-differ");
}

static iree_status_t loom_low_allocation_emit_copy_decisions(
    const loom_low_allocation_table_t* table,
    iree_diagnostic_emitter_t emitter) {
  for (iree_host_size_t i = 0; i < table->copy_decision_count; ++i) {
    const loom_low_allocation_copy_decision_t* copy_decision =
        &table->copy_decisions[i];
    loom_diagnostic_param_t params[] = {
        loom_param_string(loom_low_diagnostic_target_key(&table->target)),
        loom_param_string(loom_low_diagnostic_export_name(&table->target)),
        loom_param_string(loom_low_diagnostic_config_key(&table->target)),
        loom_param_string(loom_low_diagnostic_function_name(
            table->module, table->function_op)),
        loom_param_string(loom_low_diagnostic_value_name(
            table->module, copy_decision->source_value_id)),
        loom_param_string(loom_low_diagnostic_value_name(
            table->module, copy_decision->result_value_id)),
        loom_param_string(
            loom_low_allocation_copy_decision_name(copy_decision->kind)),
        loom_param_string(
            loom_low_allocation_coalescing_constraint(table, copy_decision)),
    };
    loom_diagnostic_emission_t emission = {
        .op = loom_low_diagnostic_value_origin_op(
            table->module, copy_decision->result_value_id, table->function_op),
        .error = LOOM_ERR_BACKEND_006,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    IREE_RETURN_IF_ERROR(iree_diagnostic_emit(emitter, &emission));
  }
  return iree_ok_status();
}

static iree_string_view_t loom_low_allocation_placement_cause_name(
    loom_low_placement_cause_t cause) {
  switch (cause) {
    case LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT:
      return IREE_SV("tied-result");
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY:
      return IREE_SV("low.copy");
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE:
      return IREE_SV("low.slice");
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT:
      return IREE_SV("low.concat");
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH:
      return IREE_SV("low.br");
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t loom_low_allocation_placement_relation_kind_name(
    loom_low_placement_relation_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE:
      return IREE_SV("same-storage");
    case LOOM_LOW_PLACEMENT_RELATION_SUBRANGE:
      return IREE_SV("subrange");
    case LOOM_LOW_PLACEMENT_RELATION_CONTIGUOUS_PART:
      return IREE_SV("contiguous-part");
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t loom_low_allocation_placement_relation_weight_name(
    loom_low_placement_relation_flags_t flags) {
  if (iree_any_bit_set(flags, LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD)) {
    return IREE_SV("hard");
  }
  if (iree_any_bit_set(flags, LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED)) {
    return IREE_SV("preferred");
  }
  return IREE_SV("unknown");
}

static bool loom_low_allocation_placement_relation_range_fits(
    const loom_low_allocation_assignment_t* assignment, uint32_t unit_offset,
    uint32_t unit_count) {
  return unit_offset <= assignment->unit_count &&
         unit_count <= assignment->unit_count - unit_offset;
}

static iree_string_view_t loom_low_allocation_placement_decision_reason(
    const loom_low_allocation_table_t* table,
    const loom_low_placement_relation_t* relation,
    const loom_low_allocation_assignment_t* source_assignment,
    const loom_low_allocation_assignment_t* result_assignment,
    bool* out_accepted) {
  *out_accepted = false;
  if (source_assignment == NULL) {
    return IREE_SV("source-unassigned");
  }
  if (result_assignment == NULL) {
    return IREE_SV("result-unassigned");
  }
  if (!loom_low_allocation_placement_relation_range_fits(
          source_assignment, relation->source_unit_offset,
          relation->unit_count) ||
      !loom_low_allocation_placement_relation_range_fits(
          result_assignment, relation->result_unit_offset,
          relation->unit_count)) {
    return IREE_SV("relation-exceeds-assignment");
  }
  if (loom_low_allocation_storage_assignment_subranges_equal(
          table->target.descriptor_set, result_assignment,
          relation->result_unit_offset, source_assignment,
          relation->source_unit_offset, relation->unit_count)) {
    *out_accepted = true;
    return IREE_SV("assigned-locations-match");
  }
  if (!loom_low_allocation_storage_assignment_classes_share(
          table->target.descriptor_set, result_assignment, source_assignment)) {
    return IREE_SV("storage-classes-differ");
  }
  if (loom_low_allocation_storage_assignment_subranges_overlap(
          table->target.descriptor_set, result_assignment,
          relation->result_unit_offset, source_assignment,
          relation->source_unit_offset, relation->unit_count)) {
    return IREE_SV("assigned-locations-overlap");
  }
  return IREE_SV("assigned-locations-differ");
}

static iree_status_t loom_low_allocation_emit_placement_decisions(
    const loom_low_allocation_table_t* table,
    iree_diagnostic_emitter_t emitter) {
  for (iree_host_size_t i = 0; i < table->placement.relation_count; ++i) {
    const loom_low_placement_relation_t* relation =
        &table->placement.relations[i];
    const loom_low_allocation_assignment_t* source_assignment =
        loom_low_allocation_assignment_for_value_ordinal(
            table, relation->source_ordinal, NULL);
    const loom_low_allocation_assignment_t* result_assignment =
        loom_low_allocation_assignment_for_value_ordinal(
            table, relation->result_ordinal, NULL);
    bool accepted = false;
    const iree_string_view_t reason =
        loom_low_allocation_placement_decision_reason(
            table, relation, source_assignment, result_assignment, &accepted);
    const iree_string_view_t value_class_name =
        result_assignment
            ? loom_low_diagnostic_value_class_name(
                  table->target.descriptor_set, result_assignment->value_class)
        : source_assignment
            ? loom_low_diagnostic_value_class_name(
                  table->target.descriptor_set, source_assignment->value_class)
            : IREE_SV("<unknown>");
    const loom_value_id_t source_value_id = loom_low_placement_value_id(
        &table->placement, relation->source_ordinal);
    const loom_value_id_t result_value_id = loom_low_placement_value_id(
        &table->placement, relation->result_ordinal);
    const loom_op_t* diagnostic_op =
        relation->op ? relation->op : table->function_op;
    loom_diagnostic_param_t params[] = {
        loom_param_string(loom_low_diagnostic_target_key(&table->target)),
        loom_param_string(loom_low_diagnostic_export_name(&table->target)),
        loom_param_string(loom_low_diagnostic_config_key(&table->target)),
        loom_param_string(loom_low_diagnostic_function_name(
            table->module, table->function_op)),
        loom_param_string(loom_op_name(table->module, diagnostic_op)),
        loom_param_string(
            loom_low_diagnostic_value_name(table->module, source_value_id)),
        loom_param_string(
            loom_low_diagnostic_value_name(table->module, result_value_id)),
        loom_param_string(value_class_name),
        loom_param_string(
            loom_low_allocation_placement_cause_name(relation->cause)),
        loom_param_string(
            loom_low_allocation_placement_relation_kind_name(relation->kind)),
        loom_param_string(loom_low_allocation_placement_relation_weight_name(
            relation->flags)),
        loom_param_string(accepted ? IREE_SV("accepted") : IREE_SV("rejected")),
        loom_param_string(reason),
        loom_param_u32(relation->result_unit_offset),
        loom_param_u32(relation->source_unit_offset),
        loom_param_u32(relation->unit_count),
    };
    loom_diagnostic_emission_t emission = {
        .op = diagnostic_op,
        .error = LOOM_ERR_BACKEND_043,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    IREE_RETURN_IF_ERROR(iree_diagnostic_emit(emitter, &emission));
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_diagnostics_emit(
    const loom_low_allocation_table_t* table,
    loom_low_allocation_diagnostic_flags_t flags,
    iree_diagnostic_emitter_t emitter) {
  IREE_ASSERT_ARGUMENT(table);
  if (iree_any_bit_set(flags,
                       LOOM_LOW_ALLOCATION_DIAGNOSTIC_PREDICTED_SPILLS)) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_emit_predicted_spills(table, emitter));
  }
  if (iree_any_bit_set(flags, LOOM_LOW_ALLOCATION_DIAGNOSTIC_COPY_DECISIONS)) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_emit_copy_decisions(table, emitter));
  }
  if (iree_any_bit_set(flags,
                       LOOM_LOW_ALLOCATION_DIAGNOSTIC_PLACEMENT_DECISIONS)) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_emit_placement_decisions(table, emitter));
  }
  return iree_ok_status();
}
