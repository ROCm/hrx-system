// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/compile_report_low.h"

#include "loom/codegen/low/diagnostics.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"

static bool loom_target_compile_report_descriptor_semantic_tag_is(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, iree_string_view_t tag) {
  if (descriptor_set == NULL || descriptor == NULL) {
    return false;
  }
  if (descriptor->semantic_tag_string_offset == LOOM_LOW_STRING_OFFSET_NONE) {
    return false;
  }
  const iree_string_view_t semantic_tag = loom_low_descriptor_set_string(
      descriptor_set, descriptor->semantic_tag_string_offset);
  return iree_string_view_equal(semantic_tag, tag);
}

static iree_string_view_t loom_target_compile_report_descriptor_semantic_tag(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  if (descriptor_set == NULL || descriptor == NULL ||
      descriptor->semantic_tag_string_offset == LOOM_LOW_STRING_OFFSET_NONE) {
    return iree_string_view_empty();
  }
  return loom_low_descriptor_set_string(descriptor_set,
                                        descriptor->semantic_tag_string_offset);
}

static bool loom_target_compile_report_string_contains(
    iree_string_view_t value, iree_string_view_t needle) {
  return iree_string_view_find(value, needle, /*pos=*/0) !=
         IREE_STRING_VIEW_NPOS;
}

static const loom_low_schedule_class_t*
loom_target_compile_report_descriptor_schedule_class(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  if (descriptor_set == NULL || descriptor == NULL ||
      descriptor->schedule_class_id >= descriptor_set->schedule_class_count) {
    return NULL;
  }
  return &descriptor_set->schedule_classes[descriptor->schedule_class_id];
}

static iree_string_view_t loom_target_compile_report_schedule_class_name(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_schedule_node_t* node) {
  if (!iree_string_view_is_empty(node->schedule_class_name)) {
    return node->schedule_class_name;
  }
  const loom_low_schedule_class_t* schedule_class =
      loom_target_compile_report_descriptor_schedule_class(descriptor_set,
                                                           node->descriptor);
  if (schedule_class == NULL ||
      schedule_class->name_string_offset == LOOM_LOW_STRING_OFFSET_NONE) {
    return iree_string_view_empty();
  }
  return loom_low_descriptor_set_string(descriptor_set,
                                        schedule_class->name_string_offset);
}

static bool loom_target_compile_report_schedule_class_uses_resource_kind(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_schedule_class_t* schedule_class,
    loom_low_resource_kind_t kind) {
  if (descriptor_set == NULL || schedule_class == NULL) {
    return false;
  }
  const uint32_t end = (uint32_t)schedule_class->issue_use_start +
                       schedule_class->issue_use_count;
  if (end > descriptor_set->issue_use_count) {
    return false;
  }
  for (uint16_t i = 0; i < schedule_class->issue_use_count; ++i) {
    const loom_low_issue_use_t* issue_use =
        &descriptor_set->issue_uses[schedule_class->issue_use_start + i];
    if (issue_use->resource_id >= descriptor_set->resource_count) {
      continue;
    }
    if (descriptor_set->resources[issue_use->resource_id].kind == kind) {
      return true;
    }
  }
  return false;
}

static void loom_target_compile_report_record_low_static_instruction_mix(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame) {
  const loom_low_descriptor_set_t* descriptor_set =
      frame->schedule.target.descriptor_set;
  loom_target_compile_report_static_instruction_mix_t mix = {0};
  for (iree_host_size_t i = 0; i < frame->schedule.node_count; ++i) {
    const loom_low_schedule_node_t* node = &frame->schedule.nodes[i];
    if (node->kind != LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR ||
        node->descriptor == NULL) {
      continue;
    }
    ++mix.descriptor_count;
    const loom_low_descriptor_t* descriptor = node->descriptor;
    const loom_low_schedule_class_t* schedule_class =
        loom_target_compile_report_descriptor_schedule_class(descriptor_set,
                                                             descriptor);
    const iree_string_view_t schedule_class_name =
        loom_target_compile_report_schedule_class_name(descriptor_set, node);
    const iree_string_view_t semantic_tag =
        loom_target_compile_report_descriptor_semantic_tag(descriptor_set,
                                                           descriptor);

    bool scalar_alu =
        iree_string_view_starts_with(schedule_class_name,
                                     IREE_SV("amdgpu.salu")) ||
        loom_target_compile_report_schedule_class_uses_resource_kind(
            descriptor_set, schedule_class, LOOM_LOW_RESOURCE_KIND_SCALAR_ALU);
    bool vector_alu =
        iree_string_view_starts_with(schedule_class_name,
                                     IREE_SV("amdgpu.valu")) ||
        loom_target_compile_report_schedule_class_uses_resource_kind(
            descriptor_set, schedule_class, LOOM_LOW_RESOURCE_KIND_VECTOR_ALU);
    bool matrix =
        iree_string_view_starts_with(semantic_tag, IREE_SV("matrix.")) ||
        iree_string_view_starts_with(schedule_class_name,
                                     IREE_SV("amdgpu.mfma")) ||
        iree_string_view_starts_with(schedule_class_name,
                                     IREE_SV("amdgpu.wmma")) ||
        loom_target_compile_report_schedule_class_uses_resource_kind(
            descriptor_set, schedule_class, LOOM_LOW_RESOURCE_KIND_MATRIX);
    bool mfma =
        iree_string_view_starts_with(semantic_tag, IREE_SV("matrix.mfma.")) ||
        iree_string_view_starts_with(semantic_tag, IREE_SV("matrix.smfmac.")) ||
        iree_string_view_starts_with(schedule_class_name,
                                     IREE_SV("amdgpu.mfma"));
    bool wmma =
        iree_string_view_starts_with(semantic_tag, IREE_SV("matrix.wmma.")) ||
        iree_string_view_starts_with(semantic_tag, IREE_SV("matrix.swmmac.")) ||
        iree_string_view_starts_with(schedule_class_name,
                                     IREE_SV("amdgpu.wmma"));
    const bool dot =
        iree_string_view_starts_with(semantic_tag, IREE_SV("dot."));
    const bool cache =
        iree_string_view_starts_with(semantic_tag, IREE_SV("memory.cache."));
    bool global_memory =
        iree_string_view_starts_with(schedule_class_name,
                                     IREE_SV("amdgpu.vmem")) ||
        iree_string_view_starts_with(semantic_tag, IREE_SV("memory.global."));
    bool local_memory =
        iree_string_view_starts_with(schedule_class_name,
                                     IREE_SV("amdgpu.lds")) ||
        iree_string_view_equal(schedule_class_name,
                               IREE_SV("amdgpu.vmem.load.lds")) ||
        iree_string_view_starts_with(semantic_tag,
                                     IREE_SV("memory.workgroup."));
    bool scalar_memory = iree_string_view_starts_with(schedule_class_name,
                                                      IREE_SV("amdgpu.smem"));
    bool generic_memory =
        iree_string_view_starts_with(semantic_tag,
                                     IREE_SV("memory.generic.")) ||
        iree_string_view_starts_with(semantic_tag, IREE_SV("memory.load.")) ||
        iree_string_view_starts_with(semantic_tag, IREE_SV("memory.store.")) ||
        iree_string_view_starts_with(semantic_tag, IREE_SV("memory.hal."));
    const bool memory_resource =
        loom_target_compile_report_schedule_class_uses_resource_kind(
            descriptor_set, schedule_class, LOOM_LOW_RESOURCE_KIND_LOAD) ||
        loom_target_compile_report_schedule_class_uses_resource_kind(
            descriptor_set, schedule_class, LOOM_LOW_RESOURCE_KIND_STORE);
    if (memory_resource && !global_memory && !local_memory && !scalar_memory &&
        !generic_memory) {
      generic_memory = true;
    }
    const bool atomic = loom_target_compile_report_string_contains(
        semantic_tag, IREE_SV(".atomic."));
    const bool branch =
        iree_string_view_starts_with(semantic_tag, IREE_SV("control.branch")) ||
        iree_string_view_starts_with(semantic_tag,
                                     IREE_SV("control.cond_branch")) ||
        iree_string_view_starts_with(semantic_tag, IREE_SV("control.return")) ||
        iree_string_view_starts_with(semantic_tag, IREE_SV("control.call"));
    const bool barrier = iree_string_view_starts_with(
                             semantic_tag, IREE_SV("control.barrier")) ||
                         iree_string_view_starts_with(
                             schedule_class_name, IREE_SV("amdgpu.barrier"));
    const bool control =
        iree_string_view_starts_with(semantic_tag, IREE_SV("control.")) ||
        (schedule_class != NULL &&
         iree_all_bits_set(schedule_class->flags,
                           LOOM_LOW_SCHEDULE_CLASS_FLAG_CONTROL)) ||
        loom_target_compile_report_schedule_class_uses_resource_kind(
            descriptor_set, schedule_class, LOOM_LOW_RESOURCE_KIND_CONTROL);
    const bool conversion =
        iree_string_view_starts_with(semantic_tag, IREE_SV("convert."));
    const bool register_move =
        iree_string_view_starts_with(semantic_tag, IREE_SV("register.copy.")) ||
        iree_string_view_starts_with(semantic_tag, IREE_SV("integer.move."));

    mix.scalar_alu_count += scalar_alu ? 1 : 0;
    mix.vector_alu_count += vector_alu ? 1 : 0;
    mix.matrix_count += matrix ? 1 : 0;
    mix.mfma_count += mfma ? 1 : 0;
    mix.wmma_count += wmma ? 1 : 0;
    mix.dot_count += dot ? 1 : 0;
    mix.global_memory_count += global_memory ? 1 : 0;
    mix.local_memory_count += local_memory ? 1 : 0;
    mix.scalar_memory_count += scalar_memory ? 1 : 0;
    mix.generic_memory_count += generic_memory ? 1 : 0;
    mix.atomic_count += atomic ? 1 : 0;
    mix.branch_count += branch ? 1 : 0;
    mix.barrier_count += barrier ? 1 : 0;
    mix.control_count += control ? 1 : 0;
    mix.conversion_count += conversion ? 1 : 0;
    mix.cache_count += cache ? 1 : 0;
    mix.register_move_count += register_move ? 1 : 0;

    const bool recognized = scalar_alu || vector_alu || matrix || mfma ||
                            wmma || dot || global_memory || local_memory ||
                            scalar_memory || generic_memory || atomic ||
                            branch || barrier || control || conversion ||
                            cache || register_move;
    mix.unknown_count += recognized ? 0 : 1;
  }
  loom_target_compile_report_record_static_instruction_mix(report, &mix);
}

static iree_string_view_t
loom_target_compile_report_low_frame_emitted_function_name(
    const loom_low_emission_frame_t* frame) {
  const iree_string_view_t export_symbol =
      frame->target.bundle_storage.export_plan.export_symbol;
  if (!iree_string_view_is_empty(export_symbol)) {
    return export_symbol;
  }
  return loom_low_diagnostic_function_name(frame->module, frame->function_op);
}

static void loom_target_compile_report_record_low_frame_identity(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame) {
  report->function_name =
      loom_target_compile_report_low_frame_emitted_function_name(frame);
  report->lowered_symbol =
      loom_low_diagnostic_function_name(frame->module, frame->function_op);
  report->target_bundle_name = frame->target.bundle_storage.bundle.name;
  report->target_snapshot_name = frame->target.bundle_storage.snapshot.name;
  report->target_export_name = frame->target.bundle_storage.export_plan.name;
  report->target_export_symbol =
      frame->target.bundle_storage.export_plan.export_symbol;
  report->target_config_name = frame->target.bundle_storage.config.name;
}

static iree_string_view_t loom_target_compile_report_module_string(
    const loom_module_t* module, loom_string_id_t string_id,
    iree_string_view_t fallback) {
  if (module == NULL || string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return fallback;
  }
  return module->strings.entries[string_id];
}

static iree_string_view_t loom_target_compile_report_value_name(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (module == NULL || value_id >= module->values.count) {
    return IREE_SV("<unknown>");
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  return loom_target_compile_report_module_string(module, value->name_id,
                                                  IREE_SV("<unnamed>"));
}

static iree_string_view_t loom_target_compile_report_block_name(
    const loom_module_t* module, const loom_block_t* block) {
  if (block == NULL) {
    return IREE_SV("<anonymous>");
  }
  return loom_target_compile_report_module_string(module, block->label_id,
                                                  IREE_SV("<anonymous>"));
}

static iree_string_view_t loom_target_compile_report_value_class_name(
    const loom_module_t* module, loom_liveness_value_class_t value_class) {
  (void)module;
  (void)value_class;
  return iree_string_view_empty();
}

static iree_string_view_t loom_target_compile_report_op_name(
    const loom_module_t* module, const loom_op_t* op) {
  if (module == NULL || op == NULL) {
    return IREE_SV("<unknown>");
  }
  return loom_op_name(module, op);
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

static const loom_low_allocation_assignment_t*
loom_target_compile_report_map_assignment(
    const loom_low_allocation_table_t* allocation, loom_value_id_t value_id) {
  return loom_low_allocation_map_active_value_assignment(allocation, value_id,
                                                         NULL);
}

static uint64_t loom_target_compile_report_slice_move_unit_count(
    const loom_low_allocation_table_t* allocation, const loom_op_t* op) {
  uint64_t unit_count = 0;
  const int64_t offset = loom_low_slice_offset(op);
  IREE_ASSERT(offset >= 0 && offset <= UINT32_MAX,
              "verified low.slice offset must fit in uint32_t");
  const loom_low_allocation_assignment_t* source_assignment =
      loom_target_compile_report_map_assignment(allocation,
                                                loom_low_slice_source(op));
  const loom_low_allocation_assignment_t* result_assignment =
      loom_target_compile_report_map_assignment(allocation,
                                                loom_low_slice_result(op));
  const uint32_t source_offset = (uint32_t)offset;
  IREE_ASSERT(source_offset <= source_assignment->location_count &&
                  result_assignment->location_count <=
                      source_assignment->location_count - source_offset,
              "verified low.slice range must fit source assignment");
  IREE_ASSERT(loom_low_allocation_storage_assignment_classes_share(
                  allocation->target.descriptor_set, source_assignment,
                  result_assignment),
              "allocated low.slice values must share one target storage class");
  for (uint32_t unit_index = 0; unit_index < result_assignment->location_count;
       ++unit_index) {
    if (!loom_low_allocation_storage_assignment_subranges_equal(
            allocation->target.descriptor_set, result_assignment, unit_index,
            source_assignment, source_offset + unit_index, /*unit_count=*/1)) {
      ++unit_count;
    }
  }
  return unit_count;
}

static uint64_t loom_target_compile_report_concat_move_unit_count(
    const loom_low_allocation_table_t* allocation, const loom_op_t* op) {
  if (!loom_low_allocation_move_topology_concat_requires_packet_materialization(
          allocation, op)) {
    return 0;
  }
  uint64_t unit_count = 0;
  const loom_low_allocation_assignment_t* result_assignment =
      loom_target_compile_report_map_assignment(allocation,
                                                loom_low_concat_result(op));
  uint32_t result_offset = 0;
  loom_value_slice_t sources = loom_low_concat_sources(op);
  for (uint16_t i = 0; i < sources.count; ++i) {
    const loom_low_allocation_assignment_t* source_assignment =
        loom_target_compile_report_map_assignment(allocation,
                                                  sources.values[i]);
    IREE_ASSERT(loom_low_allocation_storage_assignment_classes_share(
                    allocation->target.descriptor_set, result_assignment,
                    source_assignment),
                "allocated low.concat values must share one target storage "
                "class");
    IREE_ASSERT(result_offset <= result_assignment->location_count &&
                    source_assignment->location_count <=
                        result_assignment->location_count - result_offset,
                "verified low.concat range must fit result");
    for (uint32_t source_unit = 0;
         source_unit < source_assignment->location_count; ++source_unit) {
      if (!loom_low_allocation_storage_assignment_subranges_equal(
              allocation->target.descriptor_set, result_assignment,
              result_offset + source_unit, source_assignment, source_unit,
              /*unit_count=*/1)) {
        ++unit_count;
      }
    }
    result_offset += source_assignment->location_count;
  }
  IREE_ASSERT_EQ(result_offset, result_assignment->location_count);
  return unit_count;
}

static void loom_target_compile_report_record_low_copy_moves(
    loom_target_compile_report_t* report,
    const loom_low_allocation_table_t* allocation) {
  uint64_t packet_count = 0;
  uint64_t unit_count = 0;
  for (iree_host_size_t i = 0; i < allocation->copy_decision_count; ++i) {
    const loom_low_allocation_copy_decision_t* decision =
        &allocation->copy_decisions[i];
    if (decision->kind != LOOM_LOW_ALLOCATION_COPY_MATERIALIZED) {
      continue;
    }
    ++packet_count;
    IREE_ASSERT(
        decision->result_assignment_index < allocation->assignment_count,
        "verified copy decision result assignment must fit allocation "
        "table");
    unit_count += allocation->assignments[decision->result_assignment_index]
                      .location_count;
  }
  loom_target_compile_report_record_move_cause_if_nonzero(
      report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_COPY, packet_count,
      unit_count);
}

static void loom_target_compile_report_record_edge_copy_moves(
    loom_target_compile_report_t* report,
    const loom_low_allocation_table_t* allocation) {
  uint64_t packet_count = 0;
  uint64_t unit_count = 0;
  for (iree_host_size_t i = 0; i < allocation->edge_copy_group_count; ++i) {
    const loom_low_allocation_edge_copy_group_t* group =
        &allocation->edge_copy_groups[i];
    IREE_ASSERT(group->copy_start <= allocation->edge_copy_count &&
                    group->copy_count <=
                        allocation->edge_copy_count - group->copy_start,
                "verified edge-copy group range must fit allocation table");
    uint64_t group_unit_count = 0;
    for (uint32_t j = 0; j < group->copy_count; ++j) {
      const loom_low_allocation_edge_copy_t* edge_copy =
          &allocation->edge_copies[group->copy_start + j];
      IREE_ASSERT(
          edge_copy->source_assignment_index < allocation->assignment_count &&
              edge_copy->destination_assignment_index <
                  allocation->assignment_count,
          "verified edge-copy assignments must fit allocation table");
      const loom_low_allocation_assignment_t* source_assignment =
          &allocation->assignments[edge_copy->source_assignment_index];
      const loom_low_allocation_assignment_t* destination_assignment =
          &allocation->assignments[edge_copy->destination_assignment_index];
      for (uint32_t unit_index = 0; unit_index < edge_copy->unit_count;
           ++unit_index) {
        if (!loom_low_allocation_storage_assignment_subranges_equal(
                allocation->target.descriptor_set, destination_assignment,
                edge_copy->destination_unit_offset + unit_index,
                source_assignment, edge_copy->source_unit_offset + unit_index,
                /*unit_count=*/1)) {
          ++group_unit_count;
        }
      }
    }
    if (group_unit_count != 0) {
      ++packet_count;
      unit_count += group_unit_count;
    }
  }
  loom_target_compile_report_record_move_cause_if_nonzero(
      report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_BRANCH_EDGE, packet_count,
      unit_count);
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
  uint64_t slice_packet_count = 0;
  uint64_t slice_unit_count = 0;
  uint64_t concat_packet_count = 0;
  uint64_t concat_unit_count = 0;
  const loom_module_t* module = frame->schedule.module;
  const loom_low_allocation_table_t* allocation = &frame->allocation;
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
    } else if (loom_low_slice_isa(op)) {
      const uint64_t move_count =
          loom_target_compile_report_slice_move_unit_count(allocation, op);
      if (move_count != 0) {
        ++slice_packet_count;
        slice_unit_count += move_count;
      }
    } else if (loom_low_concat_isa(op)) {
      const uint64_t move_count =
          loom_target_compile_report_concat_move_unit_count(allocation, op);
      if (move_count != 0) {
        ++concat_packet_count;
        concat_unit_count += move_count;
      }
    }
  }
  loom_target_compile_report_record_move_cause_if_nonzero(
      report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_CONSTANT_MATERIALIZATION,
      constant_packet_count, constant_unit_count);
  loom_target_compile_report_record_move_cause_if_nonzero(
      report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_SLICE,
      slice_packet_count, slice_unit_count);
  loom_target_compile_report_record_move_cause_if_nonzero(
      report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_CONCAT,
      concat_packet_count, concat_unit_count);
}

static void loom_target_compile_report_record_move_causes(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame) {
  loom_target_compile_report_record_low_copy_moves(report, &frame->allocation);
  loom_target_compile_report_record_edge_copy_moves(report, &frame->allocation);
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

iree_status_t loom_target_compile_report_record_low_lowering(
    loom_target_compile_report_t* report,
    const loom_low_lower_result_t* lower_result) {
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
          .rule_set_index = source_row->rule_set_index,
          .rule_index = source_row->rule_index,
          .plan_id = source_row->plan_id,
          .plan_detail = source_row->plan_detail,
          .descriptor_id = source_row->descriptor_id,
          .emitted_low_op_count = source_row->emitted_low_op_count,
      };
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_record_source_low_row(report, &row));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_record_pressure_rows(
    loom_target_compile_report_t* report,
    const loom_liveness_analysis_t* liveness) {
  if (!loom_target_compile_report_wants_details(
          report, LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS)) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < liveness->pressure_summary_count; ++i) {
    const loom_liveness_pressure_summary_t* summary =
        &liveness->pressure_summaries[i];
    const loom_target_compile_report_pressure_row_t row = {
        .function_name = report->function_name,
        .register_class = loom_target_compile_report_value_class_name(
            liveness->module, summary->value_class),
        .type_kind = summary->value_class.type_kind,
        .element_type = summary->value_class.element_type,
        .peak_live_units = summary->peak_live_units,
        .peak_live_values = summary->peak_live_values,
        .peak_point = summary->peak_point,
        .peak_block_name = loom_target_compile_report_block_name(
            liveness->module, summary->peak_block),
        .peak_operation_name = summary->peak_op
                                   ? loom_target_compile_report_op_name(
                                         liveness->module, summary->peak_op)
                                   : IREE_SV("<block-boundary>"),
    };
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_record_pressure_row(report, &row));
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_record_spill_rows(
    loom_target_compile_report_t* report,
    const loom_low_allocation_table_t* allocation) {
  if (!loom_target_compile_report_wants_details(
          report, LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS)) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < allocation->spill_plan_count; ++i) {
    const loom_low_allocation_spill_plan_t* spill_plan =
        &allocation->spill_plans[i];
    const loom_low_allocation_assignment_t* assignment = NULL;
    if (spill_plan->assignment_index < allocation->assignment_count) {
      assignment = &allocation->assignments[spill_plan->assignment_index];
    }
    const loom_liveness_value_class_t value_class =
        assignment != NULL ? assignment->value_class
                           : (loom_liveness_value_class_t){0};
    const loom_target_compile_report_spill_row_t row = {
        .function_name = report->function_name,
        .value_name = loom_target_compile_report_value_name(
            allocation->module, spill_plan->value_id),
        .register_class = loom_target_compile_report_value_class_name(
            allocation->module, value_class),
        .type_kind = value_class.type_kind,
        .element_type = value_class.element_type,
        .assignment_index = spill_plan->assignment_index,
        .slot_index = spill_plan->slot_index,
        .slot_space = loom_low_spill_slot_space_name(spill_plan->slot_space),
        .byte_size = spill_plan->byte_size,
        .byte_alignment = spill_plan->byte_alignment,
        .store_count = spill_plan->store_count,
        .reload_count = spill_plan->reload_count,
    };
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_record_spill_row(report, &row));
  }
  return iree_ok_status();
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
      frame->schedule.dependency_count, frame->schedule.resource_use_count,
      frame->schedule.hazard_gap_count, frame->schedule.model_summary_count,
      liveness->pressure_summary_count, peak_live_units);
  loom_target_compile_report_record_allocation(
      report, frame->allocation.assignment_count, frame->allocation.spill_count,
      frame->allocation.spill_plan_count,
      frame->allocation.coalesced_copy_count,
      frame->allocation.materialized_copy_count);
  loom_target_compile_report_record_low_static_instruction_mix(report, frame);
  loom_target_compile_report_record_move_causes(report, frame);
  iree_status_t status =
      loom_target_compile_report_record_pressure_rows(report, liveness);
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_record_spill_rows(report,
                                                          &frame->allocation);
  }
  loom_low_allocation_release_value_scratch(&value_scratch);
  return status;
}
