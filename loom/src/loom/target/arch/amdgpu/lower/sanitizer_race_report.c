// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/sanitizer_race_report.h"

#include "loom/codegen/low/builder.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/abi/feedback.h"
#include "loom/target/arch/amdgpu/lower/feedback.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"

static void loom_amdgpu_sanitizer_race_require_data_register(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t value, uint32_t unit_count) {
  IREE_ASSERT(value < builder->module->values.count,
              "AMDGPU sanitizer race report received an invalid low value");
  const loom_type_t type = loom_module_value_type(builder->module, value);
  IREE_ASSERT(loom_low_type_is_register(type) &&
                  loom_low_register_type_descriptor_set_stable_id(type) ==
                      descriptor_set->stable_id &&
                  loom_low_register_type_unit_count(type) == unit_count,
              "AMDGPU sanitizer race report received a low value with an "
              "unsupported register shape");
  const uint16_t register_class = loom_low_register_type_class_id(type);
  IREE_ASSERT(register_class == LOOM_AMDGPU_REG_CLASS_ID_SGPR ||
                  register_class == LOOM_AMDGPU_REG_CLASS_ID_VGPR,
              "AMDGPU sanitizer race report value must be an SGPR or VGPR");
}

static void loom_amdgpu_sanitizer_race_require_register_class(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t value, uint32_t unit_count, uint16_t register_class) {
  IREE_ASSERT(value < builder->module->values.count,
              "AMDGPU sanitizer race report received an invalid low value");
  const loom_type_t type = loom_module_value_type(builder->module, value);
  IREE_ASSERT(loom_low_type_is_register(type) &&
                  loom_low_register_type_descriptor_set_stable_id(type) ==
                      descriptor_set->stable_id &&
                  loom_low_register_type_unit_count(type) == unit_count,
              "AMDGPU sanitizer race report received a low value with an "
              "unsupported register shape");
  IREE_ASSERT(loom_low_register_type_class_id(type) == register_class,
              "AMDGPU sanitizer race report received a low value with an "
              "unsupported register class");
}

static void loom_amdgpu_sanitizer_validate_race_report_values(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_race_report_t* report) {
  loom_amdgpu_sanitizer_race_require_data_register(builder, descriptor_set,
                                                   report->check_kind, 1);
  loom_amdgpu_sanitizer_race_require_data_register(builder, descriptor_set,
                                                   report->flags, 1);
  loom_amdgpu_sanitizer_race_require_data_register(builder, descriptor_set,
                                                   report->memory_space, 1);
  loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->current_access_kind, 1);
  loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->prior_access_kind, 1);
  loom_amdgpu_sanitizer_race_require_data_register(builder, descriptor_set,
                                                   report->access_size, 1);
  loom_amdgpu_sanitizer_race_require_data_register(builder, descriptor_set,
                                                   report->current_site_id, 2);
  loom_amdgpu_sanitizer_race_require_data_register(builder, descriptor_set,
                                                   report->prior_site_id, 2);
  loom_amdgpu_sanitizer_race_require_data_register(builder, descriptor_set,
                                                   report->memory_address, 2);
  loom_amdgpu_sanitizer_race_require_data_register(builder, descriptor_set,
                                                   report->shadow_address, 2);
  loom_amdgpu_sanitizer_race_require_data_register(builder, descriptor_set,
                                                   report->shadow_value, 2);
  loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->current_workgroup_id_x, 1);
  loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->current_workgroup_id_y, 1);
  loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->current_workgroup_id_z, 1);
  loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->current_workitem_id_x, 1);
  loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->current_workitem_id_y, 1);
  loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->current_workitem_id_z, 1);
  loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->prior_workgroup_id_x, 1);
  loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->prior_workgroup_id_y, 1);
  loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->prior_workgroup_id_z, 1);
  loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->prior_workitem_id_x, 1);
  loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->prior_workitem_id_y, 1);
  loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->prior_workitem_id_z, 1);
}

iree_status_t loom_amdgpu_build_sanitizer_race_report_payload(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    const loom_amdgpu_sanitizer_race_report_t* report,
    loom_location_id_t location) {
  loom_amdgpu_sanitizer_validate_race_report_values(builder, descriptor_set,
                                                    report);

  const uint32_t payload_base = LOOM_AMDGPU_FEEDBACK_PACKET_BYTE_LENGTH;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_u32_constant(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_RECORD_LENGTH_OFFSET,
      LOOM_AMDGPU_TSAN_REPORT_BYTE_LENGTH, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_u32_constant(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_ABI_VERSION_OFFSET,
      LOOM_AMDGPU_TSAN_REPORT_ABI_VERSION, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CHECK_KIND_OFFSET,
      report->check_kind, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_FLAGS_OFFSET, report->flags,
      location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_MEMORY_SPACE_OFFSET,
      report->memory_space, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_ACCESS_KIND_OFFSET,
      report->current_access_kind, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_ACCESS_KIND_OFFSET,
      report->prior_access_kind, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_ACCESS_SIZE_OFFSET,
      report->access_size, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b64(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_SITE_ID_OFFSET,
      report->current_site_id, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b64(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_SITE_ID_OFFSET,
      report->prior_site_id, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b64(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_MEMORY_ADDRESS_OFFSET,
      report->memory_address, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b64(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_SHADOW_ADDRESS_OFFSET,
      report->shadow_address, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b64(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_SHADOW_VALUE_OFFSET,
      report->shadow_value, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKGROUP_ID_X_OFFSET,
      report->current_workgroup_id_x, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKGROUP_ID_Y_OFFSET,
      report->current_workgroup_id_y, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKGROUP_ID_Z_OFFSET,
      report->current_workgroup_id_z, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKITEM_ID_X_OFFSET,
      report->current_workitem_id_x, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKITEM_ID_Y_OFFSET,
      report->current_workitem_id_y, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKITEM_ID_Z_OFFSET,
      report->current_workitem_id_z, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKGROUP_ID_X_OFFSET,
      report->prior_workgroup_id_x, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKGROUP_ID_Y_OFFSET,
      report->prior_workgroup_id_y, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKGROUP_ID_Z_OFFSET,
      report->prior_workgroup_id_z, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKITEM_ID_X_OFFSET,
      report->prior_workitem_id_x, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKITEM_ID_Y_OFFSET,
      report->prior_workitem_id_y, location));
  return loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKITEM_ID_Z_OFFSET,
      report->prior_workitem_id_z, location);
}

static iree_status_t loom_amdgpu_sanitizer_build_race_report_payload_callback(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    const void* payload_context, loom_location_id_t location) {
  return loom_amdgpu_build_sanitizer_race_report_payload(
      builder, descriptor_set, packet_address,
      (const loom_amdgpu_sanitizer_race_report_t*)payload_context, location);
}

static iree_status_t
loom_amdgpu_sanitizer_build_race_report_terminate_from_current_block(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t feedback_config_symbol,
    const loom_amdgpu_feedback_packet_source_t* source,
    const loom_amdgpu_sanitizer_race_report_t* report,
    loom_location_id_t location, loom_block_t** out_terminal_block) {
  loom_amdgpu_sanitizer_validate_race_report_values(builder, descriptor_set,
                                                    report);
  const loom_amdgpu_feedback_packet_producer_t producer = {
      .payload_byte_length = LOOM_AMDGPU_TSAN_REPORT_BYTE_LENGTH,
      .packet_kind = LOOM_AMDGPU_FEEDBACK_PACKET_KIND_TSAN,
      .packet_flags = LOOM_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC,
      .source = source,
      .build_payload = loom_amdgpu_sanitizer_build_race_report_payload_callback,
      .payload_context = report,
  };
  return loom_amdgpu_build_feedback_packet_producer_terminate(
      builder, descriptor_set, feedback_config_symbol, &producer, location,
      out_terminal_block);
}

iree_status_t loom_amdgpu_build_sanitizer_race_report_terminate(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t feedback_config_symbol,
    const loom_amdgpu_feedback_packet_source_t* source,
    const loom_amdgpu_sanitizer_race_report_t* report,
    loom_location_id_t location) {
  return loom_amdgpu_sanitizer_build_race_report_terminate_from_current_block(
      builder, descriptor_set, feedback_config_symbol, source, report, location,
      /*out_terminal_block=*/NULL);
}

static iree_status_t loom_amdgpu_sanitizer_race_define_register_block_arg(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_block_t* block, uint16_t register_class, uint32_t unit_count,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, register_class, unit_count, &type));
  return loom_builder_define_block_arg(builder, block, type, out_value);
}

static iree_status_t loom_amdgpu_sanitizer_define_race_report_island_args(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_block_t* entry_block,
    loom_amdgpu_sanitizer_race_report_island_t* island) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2,
      &island->source_args.dispatch_ptr));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1,
      &island->source_args.workgroup_id_x));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->source_args.workitem_id_x));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.check_kind));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.flags));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.memory_space));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.current_access_kind));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.prior_access_kind));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.access_size));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2,
      &island->report_args.current_site_id));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2,
      &island->report_args.prior_site_id));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2,
      &island->report_args.memory_address));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2,
      &island->report_args.shadow_address));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2,
      &island->report_args.shadow_value));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.current_workgroup_id_x));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.current_workgroup_id_y));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.current_workgroup_id_z));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.current_workitem_id_x));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.current_workitem_id_y));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.current_workitem_id_z));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.prior_workgroup_id_x));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.prior_workgroup_id_y));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.prior_workgroup_id_z));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.prior_workitem_id_x));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.prior_workitem_id_y));
  return loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.prior_workitem_id_z);
}

iree_status_t loom_amdgpu_build_sanitizer_race_report_island(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_block_t* after_block, loom_symbol_ref_t feedback_config_symbol,
    loom_location_id_t location,
    loom_amdgpu_sanitizer_race_report_island_t* out_island) {
  IREE_ASSERT_ARGUMENT(out_island);
  *out_island = (loom_amdgpu_sanitizer_race_report_island_t){0};
  IREE_ASSERT(after_block->parent_region != NULL,
              "AMDGPU sanitizer race report island requires a low region "
              "block");

  loom_amdgpu_sanitizer_race_report_island_t island = {0};
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, after_block->parent_region,
      (uint16_t)(after_block->region_index + 1), &island.entry_block));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_define_race_report_island_args(
      builder, descriptor_set, island.entry_block, &island));
  loom_builder_set_block(builder, island.entry_block);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_build_race_report_terminate_from_current_block(
          builder, descriptor_set, feedback_config_symbol, &island.source_args,
          &island.report_args, location, &island.terminal_block));

  *out_island = island;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_sanitizer_race_report_branch(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_race_report_island_t* island,
    const loom_amdgpu_feedback_packet_source_t* source,
    const loom_amdgpu_sanitizer_race_report_t* report,
    loom_location_id_t location) {
  IREE_ASSERT(builder->ip.before_op == NULL,
              "AMDGPU sanitizer race report branch must be built at the end "
              "of a low block");
  loom_amdgpu_sanitizer_race_require_register_class(
      builder, descriptor_set, source->dispatch_ptr, 2,
      LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  loom_amdgpu_sanitizer_race_require_register_class(
      builder, descriptor_set, source->workgroup_id_x, 1,
      LOOM_AMDGPU_REG_CLASS_ID_SGPR);

  loom_value_id_t args[26] = {0};
  args[0] = source->dispatch_ptr;
  args[1] = source->workgroup_id_x;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, source->workitem_id_x, 1, location, &args[2]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->check_kind, 1, location, &args[3]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->flags, 1, location, &args[4]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->memory_space, 1, location, &args[5]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->current_access_kind, 1, location,
      &args[6]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->prior_access_kind, 1, location,
      &args[7]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->access_size, 1, location, &args[8]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->current_site_id, 2, location, &args[9]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->prior_site_id, 2, location, &args[10]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->memory_address, 2, location, &args[11]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->shadow_address, 2, location, &args[12]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->shadow_value, 2, location, &args[13]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->current_workgroup_id_x, 1, location,
      &args[14]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->current_workgroup_id_y, 1, location,
      &args[15]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->current_workgroup_id_z, 1, location,
      &args[16]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->current_workitem_id_x, 1, location,
      &args[17]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->current_workitem_id_y, 1, location,
      &args[18]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->current_workitem_id_z, 1, location,
      &args[19]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->prior_workgroup_id_x, 1, location,
      &args[20]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->prior_workgroup_id_y, 1, location,
      &args[21]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->prior_workgroup_id_z, 1, location,
      &args[22]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->prior_workitem_id_x, 1, location,
      &args[23]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->prior_workitem_id_y, 1, location,
      &args[24]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->prior_workitem_id_z, 1, location,
      &args[25]));

  loom_op_t* branch_op = NULL;
  return loom_low_br_build(builder, island->entry_block, args,
                           IREE_ARRAYSIZE(args), location, &branch_op);
}

iree_status_t loom_amdgpu_build_sanitizer_race_report_failure_mask_branch(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_race_report_island_t* island,
    loom_value_id_t failure_mask,
    const loom_amdgpu_feedback_packet_source_t* source,
    const loom_amdgpu_sanitizer_race_report_t* report,
    loom_location_id_t location,
    loom_amdgpu_sanitizer_race_report_failure_branch_t* out_branch) {
  IREE_ASSERT_ARGUMENT(out_branch);
  *out_branch = (loom_amdgpu_sanitizer_race_report_failure_branch_t){0};
  loom_amdgpu_sanitizer_race_report_failure_branch_t branch = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_build_sanitizer_race_report_failure_mask_split(
          builder, descriptor_set, failure_mask, location, &branch));

  IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_race_report_branch(
      builder, descriptor_set, island, source, report, location));

  loom_builder_set_block(builder, branch.continuation_block);
  *out_branch = branch;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_sanitizer_race_report_failure_mask_split(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t failure_mask, loom_location_id_t location,
    loom_amdgpu_sanitizer_race_report_failure_branch_t* out_branch) {
  IREE_ASSERT_ARGUMENT(out_branch);
  *out_branch = (loom_amdgpu_sanitizer_race_report_failure_branch_t){0};
  loom_amdgpu_feedback_failure_branch_t feedback_branch = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_failure_mask_split(
      builder, descriptor_set, failure_mask, location, &feedback_branch));
  *out_branch = (loom_amdgpu_sanitizer_race_report_failure_branch_t){
      .failure_block = feedback_branch.failure_block,
      .continuation_block = feedback_branch.continuation_block,
  };
  return iree_ok_status();
}
