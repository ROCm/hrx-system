// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/kernel_emission.h"

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/target/arch/amdgpu/planning/packet_plan.h"
#include "loom/target/arch/amdgpu/planning/wait_counters.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/emit/native/amdgpu/kernel_assembly.h"
#include "loom/target/reporting/low_mix.h"
#include "loom/target/reporting/low_names.h"

static void loom_amdgpu_kernel_emission_accumulate_wait_action(
    loom_target_compile_report_wait_plan_t* summary,
    const loom_amdgpu_wait_plan_action_t* action) {
  ++summary->action_count;
  switch (action->kind) {
    case LOOM_AMDGPU_WAIT_PLAN_ACTION_EXPLICIT:
      ++summary->explicit_action_count;
      break;
    case LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED:
      ++summary->planned_action_count;
      break;
    case LOOM_AMDGPU_WAIT_PLAN_ACTION_UNKNOWN:
    default:
      IREE_ASSERT(false, "wait plan action kind must be known");
      break;
  }
  if (action->target_count == 0) {
    ++summary->full_drain_count;
    summary->max_full_drain_outstanding_before =
        iree_max(summary->max_full_drain_outstanding_before,
                 (uint64_t)action->outstanding_before);
  } else {
    ++summary->partial_wait_count;
  }
  const uint64_t drained_count =
      action->outstanding_before > action->target_count
          ? (uint64_t)action->outstanding_before - action->target_count
          : 0;
  summary->drained_count += drained_count;
  summary->max_drained_count =
      iree_max(summary->max_drained_count, drained_count);
  summary->max_outstanding_before = iree_max(
      summary->max_outstanding_before, (uint64_t)action->outstanding_before);
}

static iree_string_view_t loom_amdgpu_kernel_emission_wait_action_name(
    loom_amdgpu_wait_plan_action_kind_t kind) {
  switch (kind) {
    case LOOM_AMDGPU_WAIT_PLAN_ACTION_EXPLICIT:
      return IREE_SV("explicit");
    case LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED:
      return IREE_SV("planned");
    case LOOM_AMDGPU_WAIT_PLAN_ACTION_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

typedef struct loom_amdgpu_kernel_emission_wait_endpoint_t {
  // Node index in the schedule table, or UINT32_MAX.
  uint32_t node_index;
  // Scheduled ordinal for |node_index|, or UINT32_MAX.
  uint32_t scheduled_ordinal;
  // Operation mnemonic for |node_index|, or empty.
  iree_string_view_t operation_name;
  // Descriptor key for |node_index|, or empty.
  iree_string_view_t descriptor_key;
  // Descriptor semantic tag for |node_index|, or empty.
  iree_string_view_t semantic_tag;
} loom_amdgpu_kernel_emission_wait_endpoint_t;

static loom_amdgpu_kernel_emission_wait_endpoint_t
loom_amdgpu_kernel_emission_wait_endpoint(
    const loom_amdgpu_wait_plan_t* wait_plan, uint32_t node_index) {
  loom_amdgpu_kernel_emission_wait_endpoint_t endpoint = {
      .node_index = UINT32_MAX,
      .scheduled_ordinal = UINT32_MAX,
      .operation_name = iree_string_view_empty(),
      .descriptor_key = iree_string_view_empty(),
      .semantic_tag = iree_string_view_empty(),
  };
  const loom_low_schedule_table_t* schedule = wait_plan->schedule;
  if (node_index == LOOM_LOW_SCHEDULE_NODE_NONE ||
      node_index >= schedule->node_count) {
    return endpoint;
  }

  const loom_low_schedule_node_t* node = &schedule->nodes[node_index];
  endpoint.node_index = node_index;
  endpoint.scheduled_ordinal = node->scheduled_ordinal;
  if (node->op != NULL) {
    endpoint.operation_name = loom_op_name(schedule->module, node->op);
  }
  if (node->descriptor != NULL) {
    endpoint.descriptor_key = loom_low_descriptor_set_string(
        schedule->target.descriptor_set, node->descriptor->key_string_offset);
    if (node->descriptor->semantic_tag_string_offset !=
        LOOM_LOW_STRING_OFFSET_NONE) {
      endpoint.semantic_tag = loom_low_descriptor_set_string(
          schedule->target.descriptor_set,
          node->descriptor->semantic_tag_string_offset);
    }
  }
  return endpoint;
}

static iree_status_t loom_amdgpu_kernel_emission_record_wait_plan(
    loom_target_compile_report_t* report,
    const loom_amdgpu_packet_plan_t* packet_plan) {
  if (report == NULL || packet_plan->wait_plan.action_count == 0) {
    return iree_ok_status();
  }
  const loom_amdgpu_wait_plan_t* wait_plan = &packet_plan->wait_plan;
  loom_target_compile_report_wait_plan_t summary = {0};
  loom_target_compile_report_wait_plan_t
      counter_summaries[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT] = {0};
  loom_target_compile_report_wait_plan_t
      reason_summaries[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT]
                      [LOOM_AMDGPU_WAIT_PLAN_REASON_COUNT] = {0};
  for (iree_host_size_t i = 0; i < wait_plan->action_count; ++i) {
    const loom_amdgpu_wait_plan_action_t* action = &wait_plan->actions[i];
    IREE_ASSERT(loom_amdgpu_wait_counter_id_is_valid(action->counter_id),
                "wait plan action must name a concrete counter");
    IREE_ASSERT(action->reason < LOOM_AMDGPU_WAIT_PLAN_REASON_COUNT,
                "wait plan action must name a concrete reason");
    loom_amdgpu_kernel_emission_accumulate_wait_action(&summary, action);
    const uint32_t counter_index =
        loom_amdgpu_wait_counter_slot_from_id(action->counter_id);
    loom_amdgpu_kernel_emission_accumulate_wait_action(
        &counter_summaries[counter_index], action);
    loom_amdgpu_kernel_emission_accumulate_wait_action(
        &reason_summaries[counter_index][action->reason], action);
  }
  loom_target_compile_report_record_wait_plan(report, &summary);
  for (uint32_t i = 0; i < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++i) {
    if (counter_summaries[i].action_count == 0) {
      continue;
    }
    const uint32_t counter_id = i + 1;
    const loom_target_compile_report_wait_counter_row_t row = {
        .function_name = report->function_name,
        .counter_name = loom_amdgpu_wait_counter_name(counter_id),
        .counter_id = counter_id,
        .summary = counter_summaries[i],
    };
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_record_wait_counter_row(report, &row));
  }
  for (uint32_t i = 0; i < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++i) {
    const uint32_t counter_id = i + 1;
    const iree_string_view_t counter_name =
        loom_amdgpu_wait_counter_name(counter_id);
    for (uint32_t reason_id = 0; reason_id < LOOM_AMDGPU_WAIT_PLAN_REASON_COUNT;
         ++reason_id) {
      if (reason_summaries[i][reason_id].action_count == 0) {
        continue;
      }
      const loom_target_compile_report_wait_reason_summary_row_t row = {
          .function_name = report->function_name,
          .counter_name = counter_name,
          .reason_name = loom_amdgpu_wait_plan_reason_name(
              (loom_amdgpu_wait_plan_reason_t)reason_id),
          .counter_id = counter_id,
          .reason_id = reason_id,
          .summary = reason_summaries[i][reason_id],
      };
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_record_wait_reason_summary_row(report,
                                                                    &row));
    }
  }
  if (!loom_target_compile_report_wants_details(
          report, LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN)) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < wait_plan->action_count; ++i) {
    const loom_amdgpu_wait_plan_action_t* action = &wait_plan->actions[i];
    const uint32_t outstanding_after = action->target_count;
    const uint32_t drained_count =
        action->outstanding_before > outstanding_after
            ? action->outstanding_before - outstanding_after
            : 0;
    const loom_amdgpu_kernel_emission_wait_endpoint_t producer =
        loom_amdgpu_kernel_emission_wait_endpoint(wait_plan,
                                                  action->producer_node);
    const loom_amdgpu_kernel_emission_wait_endpoint_t consumer =
        loom_amdgpu_kernel_emission_wait_endpoint(wait_plan,
                                                  action->consumer_node);
    const loom_target_compile_report_wait_action_row_t row = {
        .function_name = report->function_name,
        .counter_name = loom_amdgpu_wait_counter_name(action->counter_id),
        .action_name =
            loom_amdgpu_kernel_emission_wait_action_name(action->kind),
        .reason_name = loom_amdgpu_wait_plan_reason_name(action->reason),
        .counter_id = action->counter_id,
        .action_id = (uint32_t)action->kind,
        .reason_id = (uint32_t)action->reason,
        .block_index = action->block_index,
        .node_index = action->node_index,
        .scheduled_ordinal = action->scheduled_ordinal,
        .producer_node = producer.node_index,
        .producer_scheduled_ordinal = producer.scheduled_ordinal,
        .producer_operation_name = producer.operation_name,
        .producer_descriptor_key = producer.descriptor_key,
        .producer_semantic_tag = producer.semantic_tag,
        .consumer_node = consumer.node_index,
        .consumer_scheduled_ordinal = consumer.scheduled_ordinal,
        .consumer_operation_name = consumer.operation_name,
        .consumer_descriptor_key = consumer.descriptor_key,
        .consumer_semantic_tag = consumer.semantic_tag,
        .target_count = action->target_count,
        .outstanding_before = action->outstanding_before,
        .outstanding_after = outstanding_after,
        .drained_count = drained_count,
    };
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_record_wait_action_row(report, &row));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_emission_record_native_insertions(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame,
    const loom_amdgpu_kernel_hsaco_contribution_t* contribution) {
  if (report == NULL || contribution->native_insertion_count == 0) {
    return iree_ok_status();
  }

  const loom_low_descriptor_set_t* descriptor_set =
      frame->schedule.target.descriptor_set;
  loom_target_compile_report_low_dynamic_context_t dynamic_context = {0};
  iree_status_t status =
      loom_target_compile_report_low_dynamic_context_initialize(
          frame, &dynamic_context);
  for (iree_host_size_t i = 0;
       i < contribution->native_insertion_count && iree_status_is_ok(status);
       ++i) {
    const loom_amdgpu_native_insertion_t* insertion =
        &contribution->native_insertions[i];
    const loom_low_schedule_node_t* node =
        &frame->schedule.nodes[insertion->node_index];
    const loom_low_schedule_block_t* block =
        &frame->schedule.blocks[insertion->block_index];
    loom_target_compile_report_target_insertion_kind_t insertion_kind =
        LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_NONE;
    iree_string_view_t packet_key = iree_string_view_empty();
    switch (insertion->kind) {
      case LOOM_AMDGPU_NATIVE_INSERTION_ADDRESS_STATE:
        insertion_kind = LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_STATE;
        break;
      case LOOM_AMDGPU_NATIVE_INSERTION_WAIT:
        insertion_kind = LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_WAIT;
        break;
      case LOOM_AMDGPU_NATIVE_INSERTION_S_NOP:
        insertion_kind = LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_DELAY;
        packet_key = IREE_SV("amdgpu.s_nop");
        break;
      case LOOM_AMDGPU_NATIVE_INSERTION_S_DELAY_ALU:
        insertion_kind = LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_DELAY;
        break;
      case LOOM_AMDGPU_NATIVE_INSERTION_V_NOP:
        insertion_kind = LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_DELAY;
        packet_key = IREE_SV("amdgpu.v_nop");
        break;
      case LOOM_AMDGPU_NATIVE_INSERTION_BRANCH_ISLAND_SKIP:
      case LOOM_AMDGPU_NATIVE_INSERTION_BRANCH_ISLAND_HOP:
        insertion_kind = LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_OTHER;
        packet_key = IREE_SV("amdgpu.s_branch");
        break;
      case LOOM_AMDGPU_NATIVE_INSERTION_NONE:
      default:
        IREE_ASSERT_UNREACHABLE(
            "native encoder produced an invalid insertion kind");
        IREE_BUILTIN_UNREACHABLE();
    }
    if (insertion->descriptor_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
      const loom_low_descriptor_t* packet_descriptor =
          loom_amdgpu_descriptor_ref_descriptor(descriptor_set,
                                                insertion->descriptor_ref);
      IREE_ASSERT(packet_descriptor != NULL);
      packet_key = loom_low_descriptor_set_string(
          descriptor_set, packet_descriptor->key_string_offset);
    }
    iree_string_view_t boundary_descriptor_key = iree_string_view_empty();
    if (node->descriptor != NULL) {
      boundary_descriptor_key = loom_low_descriptor_set_string(
          descriptor_set, node->descriptor->key_string_offset);
    }
    loom_target_compile_report_target_insertion_row_t row = {
        .function_name = report->function_name,
        .insertion_kind = insertion_kind,
        .packet_key = packet_key,
        .block_name =
            loom_target_compile_report_block_name(frame->module, block->block),
        .block_index = insertion->block_index,
        .node_index = insertion->node_index,
        .scheduled_ordinal = insertion->scheduled_ordinal,
        .boundary_operation_name = node->op != NULL
                                       ? loom_op_name(frame->module, node->op)
                                       : iree_string_view_empty(),
        .boundary_descriptor_key = boundary_descriptor_key,
        .static_packet_count = 1,
    };
    uint64_t execution_multiplier = 0;
    if (insertion->kind != LOOM_AMDGPU_NATIVE_INSERTION_BRANCH_ISLAND_HOP &&
        dynamic_context.exact &&
        loom_target_compile_report_low_node_execution_multiplier(
            frame->module, &dynamic_context.fact_table,
            dynamic_context.block_multipliers, node, &execution_multiplier)) {
      row.flags =
          LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_FLAG_DYNAMIC_PACKET_COUNT;
      row.dynamic_packet_count = execution_multiplier;
    }
    status =
        loom_target_compile_report_record_target_insertion_row(report, &row);
  }
  loom_target_compile_report_low_dynamic_context_deinitialize(&dynamic_context);
  return status;
}

static loom_target_compile_report_target_resources_t
loom_amdgpu_kernel_emission_target_resources(
    const loom_amdgpu_kernel_hsaco_summary_t* summary) {
  const loom_amdgpu_kernel_hsaco_target_resources_t* target_resources =
      &summary->target_resources;
  return (loom_target_compile_report_target_resources_t){
      .scalar_register_class = target_resources->scalar_register_class,
      .scalar_register_count = target_resources->scalar_register_count,
      .vector_register_class = target_resources->vector_register_class,
      .vector_register_count = target_resources->vector_register_count,
      .subgroup_size = target_resources->wave_size,
      .max_subgroups_per_simd = target_resources->max_waves_per_simd,
      .resident_subgroups_per_simd = target_resources->resident_waves_per_simd,
      .occupancy_percent = target_resources->occupancy_percent,
      .limiting_resource = target_resources->limiting_resource,
      .residency_summary = target_resources->residency_summary,
  };
}

static void loom_amdgpu_kernel_emission_record_summary(
    loom_target_compile_report_t* report,
    const loom_amdgpu_kernel_hsaco_summary_t* summary) {
  if (report == NULL) return;

  loom_target_compile_report_record_emission(report, summary->instruction_count,
                                             summary->text_byte_count,
                                             summary->text_storage_byte_count);
  const loom_target_compile_report_emission_breakdown_t emission_breakdown = {
      .body_instruction_count = summary->body_instruction_count,
      .entry_instruction_count = summary->entry_instruction_count,
      .coissued_instruction_count = summary->coissued_instruction_count,
      .coissued_component_count = summary->coissued_component_count,
  };
  loom_target_compile_report_record_emission_breakdown(report,
                                                       &emission_breakdown);
  loom_target_compile_report_record_memory(report,
                                           summary->private_segment_fixed_size,
                                           summary->group_segment_fixed_size);
  const loom_target_compile_report_target_resources_t target_resources =
      loom_amdgpu_kernel_emission_target_resources(summary);
  loom_target_compile_report_record_target_resources(report, &target_resources);
}

iree_status_t loom_amdgpu_kernel_emission_build(
    const loom_low_emission_frame_t* frame,
    const loom_amdgpu_hal_kernel_abi_layout_t* abi_layout,
    const loom_amdgpu_hal_kernel_abi_verify_result_t* abi_verify,
    const loom_amdgpu_native_preflight_t* preflight,
    iree_string_builder_t* target_listing, loom_target_compile_report_t* report,
    loom_amdgpu_kernel_hsaco_contribution_t* out_contribution,
    iree_arena_allocator_t* table_arena) {
  loom_amdgpu_packet_plan_t packet_plan = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_packet_plan_build(
      &frame->schedule, &frame->allocation, table_arena, &packet_plan));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_kernel_emission_record_wait_plan(report, &packet_plan));

  const loom_amdgpu_kernel_hsaco_options_t hsaco_options = {
      .abi_layout = abi_layout,
      .abi_verify = abi_verify,
      .preflight = preflight,
      .packet_plan = &packet_plan,
      .encoding_flags =
          report != NULL
              ? LOOM_AMDGPU_ENCODE_INSTRUCTION_STREAM_FLAG_CAPTURE_NATIVE_INSERTIONS
              : LOOM_AMDGPU_ENCODE_INSTRUCTION_STREAM_FLAG_NONE,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_kernel_hsaco_contribution(
      &frame->schedule, &frame->allocation, &hsaco_options, out_contribution,
      table_arena));

  if (target_listing != NULL) {
    if (iree_string_builder_size(target_listing) != 0) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(target_listing, "\n\n"));
    }
    const loom_amdgpu_kernel_assembly_options_t assembly_options = {
        .abi_layout = abi_layout,
        .abi_verify = abi_verify,
        .preflight = preflight,
        .packet_plan = &packet_plan,
        .branch_layout = &out_contribution->branch_layout,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_kernel_assembly(
        &frame->schedule, &frame->allocation, &assembly_options, target_listing,
        table_arena));
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_emission_record_native_insertions(
      report, frame, out_contribution));
  loom_amdgpu_kernel_emission_record_summary(report,
                                             &out_contribution->summary);
  return iree_ok_status();
}
