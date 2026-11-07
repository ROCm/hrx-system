// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_plan_metadata.h"

#include <string.h>

#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/read_only_data.h"
#include "loom/codegen/low/storage_layout.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

static iree_status_t loom_aie2p_leaf_plan_copy_function_name(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    iree_string_view_t* out_function_name) {
  *out_function_name = iree_string_view_empty();
  const iree_string_view_t function_name =
      loom_low_diagnostic_function_name(frame->module, frame->function_op);
  if (iree_string_view_is_empty(function_name)) {
    return iree_ok_status();
  }
  char* function_name_data = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, function_name.size,
                                           (void**)&function_name_data));
  memcpy(function_name_data, function_name.data, function_name.size);
  *out_function_name =
      iree_make_string_view(function_name_data, function_name.size);
  return iree_ok_status();
}

static void loom_aie2p_leaf_plan_retain_storage_requirements(
    const loom_low_emission_frame_t* frame,
    loom_aie2p_leaf_program_plan_t* plan) {
  for (loom_storage_space_t space = 0; space < LOOM_STORAGE_SPACE_COUNT_;
       ++space) {
    const loom_low_storage_layout_requirement_t requirement =
        loom_low_storage_layout_requirement(
            &frame->schedule.requirements.storage_layout, space);
    plan->storage_requirements[space] = (loom_aie2p_leaf_storage_requirement_t){
        .byte_length = requirement.byte_length,
        .minimum_alignment = requirement.minimum_alignment,
    };
  }
  plan->spill = (loom_aie2p_leaf_storage_requirement_t){
      .byte_length = frame->materialized_spill_storage_bytes,
      .minimum_alignment = frame->materialized_spill_storage_minimum_alignment,
  };
}

static iree_status_t loom_aie2p_leaf_plan_retain_read_only_data(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_aie2p_leaf_program_plan_t* plan) {
  const loom_low_function_requirements_t* requirements =
      &frame->schedule.requirements;
  if (requirements->read_only_data.count == 0) {
    return iree_ok_status();
  }
  loom_aie2p_leaf_read_only_data_t* read_only_data = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, requirements->read_only_data.count, sizeof(*read_only_data),
      (void**)&read_only_data));
  for (iree_host_size_t i = 0; i < requirements->read_only_data.count; ++i) {
    const loom_low_read_only_data_requirement_t* requirement =
        &requirements->read_only_data.values[i];
    IREE_ASSERT_EQ(requirement->symbol.module_id, 0u);
    IREE_ASSERT_LT(requirement->symbol.symbol_id, frame->module->symbols.count);
    const loom_symbol_t* symbol =
        &frame->module->symbols.entries[requirement->symbol.symbol_id];
    const iree_string_view_t source_name =
        loom_string_table_get(&frame->module->strings, symbol->name_id);
    char* name_data = NULL;
    if (!iree_string_view_is_empty(source_name)) {
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate(arena, source_name.size, (void**)&name_data));
      memcpy(name_data, source_name.data, source_name.size);
    }
    uint8_t* contents_data = NULL;
    if (requirement->contents.data_length != 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate(
          arena, requirement->contents.data_length, (void**)&contents_data));
      memcpy(contents_data, requirement->contents.data,
             requirement->contents.data_length);
    }
    uint32_t* bank_conflicts = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, requirement->bank_conflicts.count, sizeof(*bank_conflicts),
        (void**)&bank_conflicts));
    iree_host_size_t bank_conflict_count = 0;
    for (iree_host_size_t j = 0; j < requirement->bank_conflicts.count; ++j) {
      const uint32_t ordinal = loom_low_read_only_data_ordinal(
          &requirements->read_only_data, requirement->bank_conflicts.values[j]);
      if (ordinal != UINT32_MAX) {
        bank_conflicts[bank_conflict_count++] = ordinal;
      }
    }
    read_only_data[i] = (loom_aie2p_leaf_read_only_data_t){
        .name = iree_make_string_view(name_data, source_name.size),
        .contents = iree_make_const_byte_span(
            contents_data, requirement->contents.data_length),
        .minimum_alignment = requirement->minimum_alignment,
        .bank_conflicts = bank_conflicts,
        .bank_conflict_count = bank_conflict_count,
    };
  }
  plan->read_only_data = read_only_data;
  plan->read_only_data_count = requirements->read_only_data.count;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_leaf_plan_retain_resource_imports(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_aie2p_leaf_program_plan_t* plan) {
  const loom_low_function_requirements_t* requirements =
      &frame->schedule.requirements;
  const iree_host_size_t resource_count = requirements->resource_count;
  if (resource_count == 0) {
    return iree_ok_status();
  }

  loom_aie2p_leaf_resource_import_t* resources = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, resource_count, sizeof(*resources), (void**)&resources));
  for (iree_host_size_t resource_index = 0; resource_index < resource_count;
       ++resource_index) {
    const loom_op_t* op = requirements->resources[resource_index];
    const loom_low_schedule_node_t* node =
        loom_low_schedule_node_for_op(&frame->schedule, op);
    IREE_ASSERT(node != NULL && node->result_count == 1);
    const loom_low_packet_view_t packet = loom_low_packet_at_node(
        &frame->schedule, (uint32_t)(node - frame->schedule.nodes));
    const loom_low_allocation_assignment_t* result_assignment =
        loom_low_packet_result_assignment(&frame->allocation, &packet, 0);
    IREE_ASSERT(result_assignment != NULL);
    IREE_ASSERT_EQ(result_assignment->location_kind,
                   LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);

    loom_aie2p_leaf_resource_flags_t flags = 0;
    uint64_t extent = 0;
    if (loom_low_resource_has_extent(op)) {
      flags |= LOOM_AIE2P_LEAF_RESOURCE_FLAG_STATIC_EXTENT;
      extent = (uint64_t)loom_low_resource_extent(op);
    }
    uint32_t cache_swizzle_stride = 0;
    if (loom_low_resource_has_cache_swizzle_stride(op)) {
      flags |= LOOM_AIE2P_LEAF_RESOURCE_FLAG_CACHE_SWIZZLE_STRIDE;
      cache_swizzle_stride =
          (uint32_t)loom_low_resource_cache_swizzle_stride(op);
    }

    uint32_t extent_physical_register = UINT32_MAX;
    uint16_t extent_descriptor_register_class_id = 0;
    uint32_t extent_physical_register_count = 0;
    if (loom_low_resource_extent_value_is_present(op)) {
      flags |= LOOM_AIE2P_LEAF_RESOURCE_FLAG_DYNAMIC_EXTENT;
      const loom_low_allocation_assignment_t* extent_assignment =
          loom_low_packet_operand_assignment(&frame->allocation, &packet, 0);
      IREE_ASSERT(extent_assignment != NULL);
      IREE_ASSERT_EQ(extent_assignment->location_kind,
                     LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
      extent_physical_register = extent_assignment->location_base;
      extent_descriptor_register_class_id =
          extent_assignment->descriptor_reg_class_id;
      extent_physical_register_count = extent_assignment->location_count;
    }

    const loom_type_id_t source_type_id = loom_low_resource_source_type(op);
    IREE_ASSERT_LT(source_type_id, frame->module->types.count);
    resources[resource_index] = (loom_aie2p_leaf_resource_import_t){
        .index = (uint64_t)loom_low_resource_index(op),
        .extent = extent,
        .cache_swizzle_stride = cache_swizzle_stride,
        .physical_register = result_assignment->location_base,
        .physical_register_count = result_assignment->location_count,
        .extent_physical_register = extent_physical_register,
        .descriptor_register_class_id =
            result_assignment->descriptor_reg_class_id,
        .extent_descriptor_register_class_id =
            extent_descriptor_register_class_id,
        .extent_physical_register_count = extent_physical_register_count,
        .flags = flags,
        .import_kind = loom_low_resource_import_kind(op),
        .source_type_kind = loom_type_kind(
            loom_type_table_get(&frame->module->types, source_type_id)),
    };
  }
  plan->resource_imports = resources;
  plan->resource_import_count = resource_count;
  return iree_ok_status();
}

iree_status_t loom_aie2p_leaf_plan_retain_metadata(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_aie2p_leaf_program_plan_t* plan) {
  IREE_RETURN_IF_ERROR(loom_aie2p_leaf_plan_copy_function_name(
      frame, arena, &plan->function_name));
  loom_aie2p_leaf_plan_retain_storage_requirements(frame, plan);
  IREE_RETURN_IF_ERROR(
      loom_aie2p_leaf_plan_retain_read_only_data(frame, arena, plan));
  return loom_aie2p_leaf_plan_retain_resource_imports(frame, arena, plan);
}
