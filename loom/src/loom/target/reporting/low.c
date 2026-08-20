// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/reporting/low.h"

#include <string.h>

#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/packet.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/low/kernel.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"
#include "loom/target/reporting/low_allocation.h"
#include "loom/target/reporting/low_mix.h"
#include "loom/target/reporting/low_names.h"
#include "loom/util/cfg_graph.h"
#include "loom/util/fact_table.h"

static bool loom_target_compile_report_facts_have_finite_range(
    loom_value_facts_t facts) {
  return facts.range_lo != INT64_MIN && facts.range_hi != INT64_MAX;
}

static loom_target_compile_report_memory_interval_t
loom_target_compile_report_source_interval(
    const loom_low_byte_interval_t* source_interval) {
  const loom_low_byte_interval_precision_flags_t required_flags =
      LOOM_LOW_BYTE_INTERVAL_PRECISION_BEGIN_RANGE |
      LOOM_LOW_BYTE_INTERVAL_PRECISION_END_RANGE;
  if (!iree_all_bits_set(source_interval->precision_flags, required_flags) ||
      !loom_target_compile_report_facts_have_finite_range(
          source_interval->begin_facts) ||
      !loom_target_compile_report_facts_have_finite_range(
          source_interval->end_facts)) {
    return (loom_target_compile_report_memory_interval_t){0};
  }

  loom_target_compile_report_memory_interval_t target_interval = {
      .flags = LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_BEGIN_RANGE |
               LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_END_RANGE,
      .begin_min_bytes = source_interval->begin_facts.range_lo,
      .begin_max_bytes = source_interval->begin_facts.range_hi,
      .end_min_bytes = source_interval->end_facts.range_lo,
      .end_max_bytes = source_interval->end_facts.range_hi,
  };
  if (iree_all_bits_set(source_interval->precision_flags,
                        LOOM_LOW_BYTE_INTERVAL_PRECISION_EXACT_LENGTH)) {
    int64_t exact_length = 0;
    if (iree_checked_sub_i64(source_interval->end_facts.range_lo,
                             source_interval->begin_facts.range_lo,
                             &exact_length) &&
        exact_length > 0) {
      target_interval.flags |=
          LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_EXACT_LENGTH;
      target_interval.exact_length_bytes = (uint64_t)exact_length;
    }
  }
  if (iree_all_bits_set(source_interval->precision_flags,
                        LOOM_LOW_BYTE_INTERVAL_PRECISION_BEGIN_EXPR)) {
    target_interval.flags |=
        LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_BEGIN_EXPR;
    target_interval.begin_expr_id = source_interval->begin_expr_id;
  }
  if (iree_all_bits_set(source_interval->precision_flags,
                        LOOM_LOW_BYTE_INTERVAL_PRECISION_END_EXPR)) {
    target_interval.flags |=
        LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_END_EXPR;
    target_interval.end_expr_id = source_interval->end_expr_id;
  }
  return target_interval;
}

static_assert(
    LOOM_LOW_LOWER_MEMORY_BANK_SERVICE_PHASE_CAPACITY ==
        LOOM_TARGET_COMPILE_REPORT_BANK_SERVICE_PHASE_CAPACITY,
    "source-low and compile-report bank-service phase capacities must match");

static loom_target_compile_report_bank_service_t
loom_target_compile_report_bank_service(
    const loom_low_lower_memory_bank_service_report_t* source) {
  loom_target_compile_report_bank_service_t target = {
      .proof = source->proof,
      .classification = source->classification,
      .model_key = source->model_key,
      .model_revision = source->model_revision,
      .model_evidence = source->model_evidence,
      .request_policy = source->request_policy,
      .lane_address_proof = source->lane_address_proof,
      .active_lane_proof = source->active_lane_proof,
      .base_residue_proof = source->base_residue_proof,
      .unknown_reason = source->unknown_reason,
      .wave_size = source->wave_size,
      .bank_count = source->bank_count,
      .bank_word_byte_count = source->bank_word_byte_count,
      .packet_word_count = source->packet_word_count,
      .phase_count = source->phase_count,
      .base_residue_count = source->base_residue_count,
      .required_rounds = source->required_rounds,
      .uncontended_rounds = source->uncontended_rounds,
      .extra_rounds = source->extra_rounds,
      .maximum_request_multiplicity = source->maximum_request_multiplicity,
  };
  for (uint8_t i = 0; i < source->phase_count; ++i) {
    target.phase_lane_counts[i] = source->phase_lane_counts[i];
    target.phase_required_rounds[i] = source->phase_required_rounds[i];
  }
  return target;
}

static_assert(
    LOOM_LOW_LOWER_MEMORY_SUBGROUP_ACCESS_TERM_CAPACITY ==
        LOOM_TARGET_COMPILE_REPORT_SUBGROUP_ACCESS_TERM_CAPACITY,
    "source-low and compile-report subgroup term capacities must match");

static loom_target_compile_report_subgroup_access_t
loom_target_compile_report_subgroup_access(
    const loom_low_lower_memory_subgroup_access_report_t* source) {
  loom_target_compile_report_subgroup_access_t target = {
      .proof = source->proof,
      .lane_address_proof = source->lane_address_proof,
      .active_lane_proof = source->active_lane_proof,
      .lane_mapping = source->lane_mapping,
      .interval_coverage = source->interval_coverage,
      .unknown_reason = source->unknown_reason,
      .subgroup_size = source->subgroup_size,
      .lane_term_count = source->lane_term_count,
      .per_lane_packet_byte_count = source->per_lane_packet_byte_count,
      .linear_lane_byte_stride = source->linear_lane_byte_stride,
      .subgroup_requested_byte_count = source->subgroup_requested_byte_count,
      .subgroup_unique_byte_count = source->subgroup_unique_byte_count,
      .subgroup_span_byte_count = source->subgroup_span_byte_count,
      .maximum_adjacent_lane_delta_bytes =
          source->maximum_adjacent_lane_delta_bytes,
      .maximum_uncovered_byte_gap_bytes =
          source->maximum_uncovered_byte_gap_bytes,
      .distinct_lane_address_count = source->distinct_lane_address_count,
      .contiguous_region_count = source->contiguous_region_count,
  };
  for (uint8_t i = 0; i < source->lane_term_count; ++i) {
    target.lane_terms[i] = (loom_target_compile_report_subgroup_access_term_t){
        .divisor = source->lane_terms[i].divisor,
        .modulus = source->lane_terms[i].modulus,
        .byte_stride = source->lane_terms[i].byte_stride,
    };
  }
  return target;
}

static iree_string_view_t
loom_target_compile_report_low_frame_emitted_function_name(
    const loom_low_emission_frame_t* frame) {
  const loom_target_bundle_t* bundle =
      loom_low_resolved_target_bundle(&frame->target);
  const iree_string_view_t export_symbol = bundle->export_plan->export_symbol;
  if (!iree_string_view_is_empty(export_symbol)) {
    return export_symbol;
  }
  return loom_low_diagnostic_function_name(frame->module, frame->function_op);
}

static void loom_target_compile_report_record_low_frame_identity(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame) {
  const loom_target_bundle_t* bundle =
      loom_low_resolved_target_bundle(&frame->target);
  report->function_name =
      loom_target_compile_report_low_frame_emitted_function_name(frame);
  report->lowered_symbol =
      loom_low_diagnostic_function_name(frame->module, frame->function_op);
  report->target_bundle_name = bundle->name;
  report->target_snapshot_name = bundle->snapshot->name;
  report->target_export_name = bundle->export_plan->name;
  report->target_export_symbol = bundle->export_plan->export_symbol;
  report->target_config_name = bundle->config->name;
}

static void loom_target_compile_report_record_move_cause_if_nonzero(
    loom_target_compile_report_t* report,
    loom_target_compile_report_move_cause_t cause, uint64_t packet_count,
    uint64_t unit_count) {
  if (packet_count == 0 && unit_count == 0) {
    return;
  }
  loom_target_compile_report_record_move_cause(report, cause, packet_count,
                                               unit_count);
}

static uint64_t loom_target_compile_report_value_register_unit_count(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (module == NULL || value_id >= module->values.count) {
    return 0;
  }
  const loom_type_t type = loom_module_value_type(module, value_id);
  return loom_low_type_is_register(type)
             ? loom_low_register_type_unit_count(type)
             : 0;
}

static uint64_t loom_target_compile_report_result_register_unit_count(
    const loom_module_t* module, const loom_liveness_analysis_t* liveness,
    const loom_low_schedule_node_t* node) {
  uint64_t unit_count = 0;
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  for (uint16_t i = 0; i < node->result_count; ++i) {
    const loom_value_ordinal_t result_ordinal = result_ordinals[i];
    IREE_ASSERT(result_ordinal < liveness->value_count,
                "verified schedule node result ordinal must fit liveness "
                "value domain");
    unit_count += loom_target_compile_report_value_register_unit_count(
        module, liveness->value_ids[result_ordinal]);
  }
  return unit_count;
}

static void loom_target_compile_report_record_edge_copy_moves(
    loom_target_compile_report_t* report,
    const loom_low_allocation_table_t* allocation) {
  uint64_t packet_count = 0;
  uint64_t unit_count = 0;
  for (iree_host_size_t i = 0; i < allocation->edge_copy_group_count; ++i) {
    const loom_low_allocation_edge_copy_group_t* group =
        &allocation->edge_copy_groups[i];
    if (group->move_group.moves.count == 0) continue;
    ++packet_count;
    unit_count += group->move_group.moves.count;
  }
  loom_target_compile_report_record_move_cause_if_nonzero(
      report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_BRANCH_EDGE, packet_count,
      unit_count);
}

static void loom_target_compile_report_record_packet_moves(
    loom_target_compile_report_t* report,
    const loom_low_allocation_table_t* allocation) {
  uint64_t copy_packet_count = 0;
  uint64_t copy_unit_count = 0;
  uint64_t move_packet_count = 0;
  uint64_t move_unit_count = 0;
  uint64_t slice_packet_count = 0;
  uint64_t slice_unit_count = 0;
  uint64_t concat_packet_count = 0;
  uint64_t concat_unit_count = 0;
  for (iree_host_size_t i = 0; i < allocation->packet_move_group_count; ++i) {
    const loom_low_allocation_packet_move_group_t* group =
        &allocation->packet_move_groups[i];
    switch (group->cause) {
      case LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY:
        ++copy_packet_count;
        copy_unit_count += group->move_group.moves.count;
        break;
      case LOOM_LOW_PLACEMENT_CAUSE_LOW_MOVE:
        ++move_packet_count;
        move_unit_count += group->move_group.moves.count;
        break;
      case LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE:
        ++slice_packet_count;
        slice_unit_count += group->move_group.moves.count;
        break;
      case LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT:
        ++concat_packet_count;
        concat_unit_count += group->move_group.moves.count;
        break;
      default:
        IREE_ASSERT_UNREACHABLE(
            "packet move group must have a packet-local placement cause");
        break;
    }
  }
  loom_target_compile_report_record_move_cause_if_nonzero(
      report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_COPY, copy_packet_count,
      copy_unit_count);
  loom_target_compile_report_record_move_cause_if_nonzero(
      report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_MOVE, move_packet_count,
      move_unit_count);
  loom_target_compile_report_record_move_cause_if_nonzero(
      report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_SLICE,
      slice_packet_count, slice_unit_count);
  loom_target_compile_report_record_move_cause_if_nonzero(
      report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_CONCAT,
      concat_packet_count, concat_unit_count);
}

static void
loom_target_compile_report_record_operand_bank_materialization_moves(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame) {
  uint64_t packet_count = 0;
  uint64_t unit_count = 0;
  const loom_module_t* module = frame->schedule.module;
  const loom_liveness_analysis_t* liveness = &frame->allocation.liveness;
  const loom_low_descriptor_set_t* descriptor_set =
      frame->schedule.target.descriptor_set;
  for (iree_host_size_t i = 0; i < frame->schedule.node_count; ++i) {
    const loom_low_schedule_node_t* node = &frame->schedule.nodes[i];
    if (node->kind != LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR ||
        !loom_target_compile_report_descriptor_semantic_tag_is(
            descriptor_set, node->descriptor, IREE_SV("register.copy.b32"))) {
      continue;
    }
    ++packet_count;
    unit_count += loom_target_compile_report_result_register_unit_count(
        module, liveness, node);
  }
  loom_target_compile_report_record_move_cause_if_nonzero(
      report,
      LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_OPERAND_BANK_MATERIALIZATION,
      packet_count, unit_count);
}

static void loom_target_compile_report_record_structural_packet_moves(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame) {
  uint64_t constant_packet_count = 0;
  uint64_t constant_unit_count = 0;
  const loom_module_t* module = frame->schedule.module;
  for (iree_host_size_t i = 0; i < frame->schedule.node_count; ++i) {
    const loom_op_t* op = frame->schedule.nodes[i].op;
    if (op == NULL) {
      continue;
    }
    if (loom_low_const_isa(op)) {
      ++constant_packet_count;
      constant_unit_count +=
          loom_target_compile_report_value_register_unit_count(
              module, loom_low_const_result(op));
    }
  }
  loom_target_compile_report_record_move_cause_if_nonzero(
      report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_CONSTANT_MATERIALIZATION,
      constant_packet_count, constant_unit_count);
}

static void loom_target_compile_report_record_move_causes(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame) {
  loom_target_compile_report_record_edge_copy_moves(report, &frame->allocation);
  loom_target_compile_report_record_packet_moves(report, &frame->allocation);
  loom_target_compile_report_record_operand_bank_materialization_moves(report,
                                                                       frame);
  loom_target_compile_report_record_structural_packet_moves(report, frame);
}

static loom_target_compile_report_source_low_selection_kind_t
loom_target_compile_report_source_low_selection_kind(
    loom_low_lower_report_selection_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_LOWER_REPORT_SELECTION_RULE:
      return LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_RULE;
    case LOOM_LOW_LOWER_REPORT_SELECTION_PLAN:
      return LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_PLAN;
    case LOOM_LOW_LOWER_REPORT_SELECTION_NONE:
    default:
      return LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_NONE;
  }
}

static bool loom_target_compile_report_mul3_u32(uint32_t x, uint32_t y,
                                                uint32_t z,
                                                uint64_t* out_result) {
  uint64_t xy = 0;
  return iree_checked_mul_u64(x, y, &xy) &&
         iree_checked_mul_u64(xy, z, out_result);
}

static void loom_target_compile_report_record_static_workload(
    loom_target_compile_report_t* report,
    const loom_target_workgroup_size_t* workgroup_size,
    const loom_target_dispatch_workgroup_count_t* workgroup_count,
    const loom_target_workgroup_cluster_size_t* workgroup_cluster_size) {
  loom_target_compile_report_workload_t workload = {0};
  if (workgroup_size != NULL) {
    workload.flags |= LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_SIZE;
    workload.workgroup_size = *workgroup_size;
    if (loom_target_compile_report_mul3_u32(
            workload.workgroup_size.x, workload.workgroup_size.y,
            workload.workgroup_size.z, &workload.flat_workgroup_size)) {
      workload.flags |= LOOM_TARGET_COMPILE_REPORT_WORKLOAD_FLAT_WORKGROUP_SIZE;
    }
  }
  if (workgroup_count != NULL) {
    workload.flags |= LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_COUNT;
    workload.workgroup_count = *workgroup_count;
    if (loom_target_compile_report_mul3_u32(
            workload.workgroup_count.x, workload.workgroup_count.y,
            workload.workgroup_count.z, &workload.dispatch_workgroup_count)) {
      workload.flags |=
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKGROUP_COUNT;
    }
  }
  if (workgroup_cluster_size != NULL) {
    workload.flags |=
        LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_CLUSTER_SIZE;
    workload.workgroup_cluster_size = *workgroup_cluster_size;
    if (loom_target_compile_report_mul3_u32(
            workload.workgroup_cluster_size.x,
            workload.workgroup_cluster_size.y,
            workload.workgroup_cluster_size.z,
            &workload.flat_workgroup_cluster_size)) {
      workload.flags |=
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_FLAT_WORKGROUP_CLUSTER_SIZE;
    }
  }
  if (iree_all_bits_set(
          workload.flags,
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_FLAT_WORKGROUP_SIZE |
              LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKGROUP_COUNT) &&
      iree_checked_mul_u64(workload.flat_workgroup_size,
                           workload.dispatch_workgroup_count,
                           &workload.dispatch_workitem_count)) {
    workload.flags |=
        LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT;
  }
  loom_target_compile_report_record_workload(report, &workload);
}

static void loom_target_compile_report_record_low_workload(
    loom_target_compile_report_t* report,
    const loom_low_lower_result_t* lower_result) {
  const loom_target_workgroup_size_t* workgroup_size = NULL;
  if (iree_any_bit_set(lower_result->static_launch_config_flags,
                       LOOM_LOW_LOWER_STATIC_LAUNCH_CONFIG_WORKGROUP_SIZE)) {
    workgroup_size = &lower_result->static_workgroup_size;
  }
  const loom_target_dispatch_workgroup_count_t* workgroup_count = NULL;
  if (iree_any_bit_set(lower_result->static_launch_config_flags,
                       LOOM_LOW_LOWER_STATIC_LAUNCH_CONFIG_WORKGROUP_COUNT)) {
    workgroup_count = &lower_result->static_workgroup_count;
  }
  const loom_target_workgroup_cluster_size_t* workgroup_cluster_size = NULL;
  if (iree_any_bit_set(
          lower_result->static_launch_config_flags,
          LOOM_LOW_LOWER_STATIC_LAUNCH_CONFIG_WORKGROUP_CLUSTER_SIZE)) {
    workgroup_cluster_size = &lower_result->static_workgroup_cluster_size;
  }
  loom_target_compile_report_record_static_workload(
      report, workgroup_size, workgroup_count, workgroup_cluster_size);
}

void loom_target_compile_report_record_low_kernel_workload(
    loom_target_compile_report_t* report, const loom_op_t* low_function_op) {
  loom_target_workgroup_size_t workgroup_size = {0};
  const bool has_workgroup_size = loom_low_kernel_def_static_workgroup_size(
      low_function_op, &workgroup_size);
  loom_target_dispatch_workgroup_count_t workgroup_count = {0};
  const bool has_workgroup_count = loom_low_kernel_def_static_workgroup_count(
      low_function_op, &workgroup_count);
  loom_target_workgroup_cluster_size_t workgroup_cluster_size = {0};
  const bool has_workgroup_cluster_size =
      loom_low_kernel_def_static_workgroup_cluster_size(
          low_function_op, &workgroup_cluster_size);
  loom_target_compile_report_record_static_workload(
      report, has_workgroup_size ? &workgroup_size : NULL,
      has_workgroup_count ? &workgroup_count : NULL,
      has_workgroup_cluster_size ? &workgroup_cluster_size : NULL);
}

iree_status_t loom_target_compile_report_record_low_lowering(
    loom_target_compile_report_t* report,
    const loom_low_lower_result_t* lower_result) {
  loom_target_compile_report_record_low_workload(report, lower_result);
  const bool has_source_low_summary =
      lower_result->selected_source_op_count != 0 ||
      lower_result->emitted_low_op_count != 0 ||
      lower_result->report_rows.head != NULL ||
      lower_result->memory_report_rows.count != 0;
  if (!has_source_low_summary) {
    return iree_ok_status();
  }
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS;
  report->source_low_selected_op_count +=
      lower_result->selected_source_op_count;
  report->source_low_emitted_op_count += lower_result->emitted_low_op_count;
  for (const loom_low_lower_report_row_vec_t* vec =
           lower_result->report_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_low_lower_report_row_t* source_rows =
        loom_low_lower_report_row_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      const loom_low_lower_report_row_t* source_row = &source_rows[i];
      const loom_target_compile_report_source_low_row_t row = {
          .function_name = source_row->function_name,
          .source_op_name = source_row->source_op_name,
          .source_op_kind = source_row->source_op_kind,
          .selection_kind =
              loom_target_compile_report_source_low_selection_kind(
                  source_row->selection_kind),
          .plan_key = source_row->plan_key,
          .descriptor_key = source_row->descriptor_key,
          .descriptor_semantic_tag = source_row->descriptor_semantic_tag,
          .emitted_low_op_count = source_row->emitted_low_op_count,
          .execution_count_plus_one = source_row->execution_count_plus_one,
      };
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_record_source_low_row(report, &row));
    }
  }
  for (iree_host_size_t i = 0; i < lower_result->memory_report_rows.count;
       ++i) {
    const loom_low_lower_memory_report_row_t* source_row =
        &lower_result->memory_report_rows.rows[i];
    const loom_target_compile_report_memory_interval_t source_interval =
        loom_target_compile_report_source_interval(
            &source_row->source_interval);
    const loom_target_compile_report_source_low_memory_row_t row = {
        .function_name = source_row->function_name,
        .source_op_name = source_row->source_op_name,
        .source_op_kind = source_row->source_op_kind,
        .source_root_name = source_row->source_root_name,
        .source_root_argument_index = source_row->source_root_argument_index,
        .memory_space = source_row->memory_space,
        .operation_kind = source_row->operation_kind,
        .packet_key = source_row->packet_key,
        .strategy_key = source_row->strategy_key,
        .address_form = source_row->address_form,
        .dynamic_term_kind = source_row->dynamic_term_kind,
        .fallback_reason = source_row->fallback_reason,
        .static_offset_bytes = source_row->static_offset_bytes,
        .element_byte_count = source_row->element_byte_count,
        .vector_lane_count = source_row->vector_lane_count,
        .issued_read_byte_count = source_row->issued_read_byte_count,
        .issued_write_byte_count = source_row->issued_write_byte_count,
        .issued_read_unknown_width_count =
            source_row->issued_read_unknown_width_count,
        .issued_write_unknown_width_count =
            source_row->issued_write_unknown_width_count,
        .dynamic_stride_bytes = source_row->dynamic_stride_bytes,
        .vector_lane_stride_bytes = source_row->vector_lane_stride_bytes,
        .storage_element_format = source_row->storage_element_format,
        .storage_scale_format = source_row->storage_scale_format,
        .storage_secondary_scale_format =
            source_row->storage_secondary_scale_format,
        .storage_payload_packing = source_row->storage_payload_packing,
        .storage_scale_topology = source_row->storage_scale_topology,
        .storage_affine_policy = source_row->storage_affine_policy,
        .storage_rounding_policy = source_row->storage_rounding_policy,
        .storage_codebook_policy = source_row->storage_codebook_policy,
        .storage_sparsity_policy = source_row->storage_sparsity_policy,
        .bank_service =
            loom_target_compile_report_bank_service(&source_row->bank_service),
        .subgroup_access = loom_target_compile_report_subgroup_access(
            &source_row->subgroup_access),
        .source_interval = source_interval,
        .execution_count_plus_one = source_row->execution_count_plus_one,
    };
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_record_source_low_memory_row(report, &row));
  }
  return iree_ok_status();
}

void loom_target_compile_report_record_low_planning(
    loom_target_compile_report_t* report,
    const loom_low_planning_statistics_t* statistics) {
  if (statistics->frame_build_count == 0) return;
  const bool first_record = !iree_any_bit_set(
      report->detail_flags, LOOM_TARGET_COMPILE_REPORT_DETAIL_LOW_PLANNING);
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_LOW_PLANNING;
  if (first_record) {
    report->low_planning = *statistics;
  } else {
    loom_low_planning_statistics_accumulate(&report->low_planning, statistics);
  }
}

iree_status_t loom_target_compile_report_record_low_emission_frame(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame) {
  loom_target_compile_report_record_low_frame_identity(report, frame);
  loom_low_allocation_value_scratch_t value_scratch = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocation_acquire_value_scratch(
      &frame->allocation, &value_scratch));
  const loom_liveness_analysis_t* liveness = &frame->allocation.liveness;
  uint64_t peak_live_units = 0;
  for (iree_host_size_t i = 0; i < liveness->pressure_summary_count; ++i) {
    const uint64_t live_units = liveness->pressure_summaries[i].peak_live_units;
    peak_live_units = iree_max(peak_live_units, live_units);
  }
  loom_target_compile_report_record_schedule(
      report, frame->schedule.node_count, frame->schedule.scheduled_node_count,
      frame->schedule.dependencies.count, frame->schedule.resource_use_count,
      frame->schedule.hazard_gap_count, frame->schedule.model_summary_count,
      liveness->pressure_summary_count, peak_live_units);
  loom_target_compile_report_record_low_static_instruction_mix(report, frame);
  loom_target_compile_report_low_dynamic_context_t dynamic_context = {0};
  iree_status_t status =
      loom_target_compile_report_low_dynamic_context_initialize(
          frame, &dynamic_context);
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_record_low_dynamic_mix(
        report, frame, &dynamic_context);
  }
  if (iree_status_is_ok(status)) {
    loom_target_compile_report_record_move_causes(report, frame);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_record_low_allocation_contents(
        report, &frame->allocation, &frame->schedule, &dynamic_context);
  }
  if (iree_status_is_ok(status)) {
    loom_target_compile_report_record_allocation_materialization(
        report, frame->materialized_spill_storage_count,
        frame->materialized_spill_storage_bytes,
        frame->materialized_spill_store_count,
        frame->materialized_spill_store_bytes, frame->materialized_reload_count,
        frame->materialized_reload_bytes);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_record_materialized_spill_rows(report,
                                                                       frame);
  }
  loom_low_allocation_release_value_scratch(&value_scratch);
  loom_target_compile_report_low_dynamic_context_deinitialize(&dynamic_context);
  return status;
}
