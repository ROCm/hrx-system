// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/provider.h"

#include <inttypes.h>
#include <stdint.h>

#include "loom/ir/module.h"
#include "loom/pass/builder.h"
#include "loom/target/arch/spirv/descriptors/low_registry.h"
#include "loom/target/arch/spirv/low_verify.h"
#include "loom/target/arch/spirv/lower/lower.h"
#include "loom/target/arch/spirv/math_policy.h"
#include "loom/target/arch/spirv/ops/ops.h"
#include "loom/target/arch/spirv/ops/registry.h"
#include "loom/target/arch/spirv/records/target_records.h"

static const loom_low_verify_provider_t* const kLoomSpirvLowVerifyProviders[] =
    {
        &loom_spirv_low_verify_provider,
};

static bool loom_spirv_provider_matches_selection_bundle(
    const loom_target_bundle_t* bundle) {
  return bundle != NULL && bundle->snapshot != NULL &&
         bundle->export_plan != NULL && bundle->config != NULL &&
         bundle->snapshot->codegen_format == LOOM_TARGET_CODEGEN_FORMAT_SPIRV &&
         bundle->snapshot->artifact_format ==
             LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY;
}

static iree_status_t loom_spirv_provider_prepare_u64_override(
    uint64_t selected_value, uint64_t default_value,
    loom_spirv_target_build_flags_t flag,
    loom_spirv_target_build_flags_t* build_flags, int64_t* out_value,
    iree_string_view_t field_name) {
  *out_value = 0;
  if (selected_value == default_value) {
    return iree_ok_status();
  }
  if (selected_value > (uint64_t)INT64_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "SPIR-V target materialization field '%.*s' value %" PRIu64
        " exceeds the target-record i64 range",
        (int)field_name.size, field_name.data, selected_value);
  }
  *build_flags |= flag;
  *out_value = (int64_t)selected_value;
  return iree_ok_status();
}

static void loom_spirv_provider_prepare_u32_override(
    uint32_t selected_value, uint32_t default_value,
    loom_spirv_target_build_flags_t flag,
    loom_spirv_target_build_flags_t* build_flags, int64_t* out_value) {
  *out_value = 0;
  if (selected_value == default_value) {
    return;
  }
  *build_flags |= flag;
  *out_value = selected_value;
}

static void loom_spirv_provider_prepare_enum_override(
    uint8_t selected_value, uint8_t default_value,
    loom_spirv_target_build_flags_t flag,
    loom_spirv_target_build_flags_t* build_flags, uint8_t* out_value) {
  *out_value = 0;
  if (selected_value == default_value) {
    return;
  }
  *build_flags |= flag;
  *out_value = selected_value;
}

static iree_status_t loom_spirv_provider_prepare_string_override(
    loom_module_t* module, iree_string_view_t selected_value,
    iree_string_view_t default_value, loom_spirv_target_build_flags_t flag,
    loom_spirv_target_build_flags_t* build_flags, loom_string_id_t* out_value) {
  *out_value = LOOM_STRING_ID_INVALID;
  if (iree_string_view_equal(selected_value, default_value)) {
    return iree_ok_status();
  }
  *build_flags |= flag;
  return loom_module_intern_string(module, selected_value, out_value);
}

static bool loom_spirv_provider_match_enum_override(const loom_op_t* target_op,
                                                    uint8_t attr_index,
                                                    uint8_t selected_value,
                                                    uint8_t default_value) {
  const loom_attribute_t attr = loom_op_attrs(target_op)[attr_index];
  if (loom_attr_is_absent(attr)) {
    return selected_value == default_value;
  }
  return loom_attr_as_enum(attr) == selected_value;
}

static bool loom_spirv_provider_match_u64_override(const loom_op_t* target_op,
                                                   uint8_t attr_index,
                                                   uint64_t selected_value,
                                                   uint64_t default_value) {
  const loom_attribute_t attr = loom_op_attrs(target_op)[attr_index];
  if (loom_attr_is_absent(attr)) {
    return selected_value == default_value;
  }
  const int64_t attr_value = loom_attr_as_i64(attr);
  return attr_value >= 0 && selected_value == (uint64_t)attr_value;
}

static bool loom_spirv_provider_match_string_override(
    const loom_module_t* module, const loom_op_t* target_op, uint8_t attr_index,
    iree_string_view_t selected_value, iree_string_view_t default_value) {
  const loom_attribute_t attr = loom_op_attrs(target_op)[attr_index];
  if (loom_attr_is_absent(attr)) {
    return iree_string_view_equal(selected_value, default_value);
  }
  const loom_string_id_t string_id = loom_attr_as_string_id(attr);
  if (string_id >= module->strings.count) {
    return false;
  }
  return iree_string_view_equal(module->strings.entries[string_id],
                                selected_value);
}

static bool loom_spirv_provider_target_record_matches_bundle(
    const loom_module_t* module, const loom_op_t* target_op,
    const loom_target_bundle_t* selected_bundle) {
  if (loom_spirv_target_kind(target_op) != LOOM_SPIRV_TARGET_KIND_VULKAN1_3) {
    return false;
  }

  const loom_target_bundle_t* default_bundle =
      &loom_spirv_low_target_bundle_vulkan1_3;
  const loom_target_snapshot_t* selected_snapshot = selected_bundle->snapshot;
  const loom_target_snapshot_t* default_snapshot = default_bundle->snapshot;
  const loom_target_export_plan_t* selected_export_plan =
      selected_bundle->export_plan;
  const loom_target_export_plan_t* default_export_plan =
      default_bundle->export_plan;
  const loom_target_config_t* selected_config = selected_bundle->config;
  const loom_target_config_t* default_config = default_bundle->config;

  return loom_spirv_provider_match_enum_override(
             target_op, loom_spirv_target_codegen_format_ATTR_INDEX,
             selected_snapshot->codegen_format,
             default_snapshot->codegen_format) &&
         loom_spirv_provider_match_enum_override(
             target_op, loom_spirv_target_artifact_format_ATTR_INDEX,
             selected_snapshot->artifact_format,
             default_snapshot->artifact_format) &&
         loom_spirv_provider_match_u64_override(
             target_op, loom_spirv_target_default_pointer_bitwidth_ATTR_INDEX,
             selected_snapshot->default_pointer_bitwidth,
             default_snapshot->default_pointer_bitwidth) &&
         loom_spirv_provider_match_u64_override(
             target_op, loom_spirv_target_index_bitwidth_ATTR_INDEX,
             selected_snapshot->index_bitwidth,
             default_snapshot->index_bitwidth) &&
         loom_spirv_provider_match_u64_override(
             target_op, loom_spirv_target_offset_bitwidth_ATTR_INDEX,
             selected_snapshot->offset_bitwidth,
             default_snapshot->offset_bitwidth) &&
         loom_spirv_provider_match_u64_override(
             target_op, loom_spirv_target_max_workgroup_size_x_ATTR_INDEX,
             selected_snapshot->max_workgroup_size.x,
             default_snapshot->max_workgroup_size.x) &&
         loom_spirv_provider_match_u64_override(
             target_op, loom_spirv_target_max_workgroup_size_y_ATTR_INDEX,
             selected_snapshot->max_workgroup_size.y,
             default_snapshot->max_workgroup_size.y) &&
         loom_spirv_provider_match_u64_override(
             target_op, loom_spirv_target_max_workgroup_size_z_ATTR_INDEX,
             selected_snapshot->max_workgroup_size.z,
             default_snapshot->max_workgroup_size.z) &&
         loom_spirv_provider_match_u64_override(
             target_op, loom_spirv_target_max_flat_workgroup_size_ATTR_INDEX,
             selected_snapshot->max_flat_workgroup_size,
             default_snapshot->max_flat_workgroup_size) &&
         loom_spirv_provider_match_u64_override(
             target_op, loom_spirv_target_subgroup_size_ATTR_INDEX,
             selected_snapshot->subgroup_size,
             default_snapshot->subgroup_size) &&
         loom_spirv_provider_match_u64_override(
             target_op, loom_spirv_target_max_grid_size_x_ATTR_INDEX,
             selected_snapshot->max_grid_size.x,
             default_snapshot->max_grid_size.x) &&
         loom_spirv_provider_match_u64_override(
             target_op, loom_spirv_target_max_grid_size_y_ATTR_INDEX,
             selected_snapshot->max_grid_size.y,
             default_snapshot->max_grid_size.y) &&
         loom_spirv_provider_match_u64_override(
             target_op, loom_spirv_target_max_grid_size_z_ATTR_INDEX,
             selected_snapshot->max_grid_size.z,
             default_snapshot->max_grid_size.z) &&
         loom_spirv_provider_match_u64_override(
             target_op, loom_spirv_target_max_flat_grid_size_ATTR_INDEX,
             selected_snapshot->max_flat_grid_size,
             default_snapshot->max_flat_grid_size) &&
         loom_spirv_provider_match_u64_override(
             target_op, loom_spirv_target_max_workgroup_count_x_ATTR_INDEX,
             selected_snapshot->max_workgroup_count.x,
             default_snapshot->max_workgroup_count.x) &&
         loom_spirv_provider_match_u64_override(
             target_op, loom_spirv_target_max_workgroup_count_y_ATTR_INDEX,
             selected_snapshot->max_workgroup_count.y,
             default_snapshot->max_workgroup_count.y) &&
         loom_spirv_provider_match_u64_override(
             target_op, loom_spirv_target_max_workgroup_count_z_ATTR_INDEX,
             selected_snapshot->max_workgroup_count.z,
             default_snapshot->max_workgroup_count.z) &&
         loom_spirv_provider_match_u64_override(
             target_op, loom_spirv_target_memory_space_generic_ATTR_INDEX,
             selected_snapshot->memory_spaces.generic,
             default_snapshot->memory_spaces.generic) &&
         loom_spirv_provider_match_u64_override(
             target_op, loom_spirv_target_memory_space_global_ATTR_INDEX,
             selected_snapshot->memory_spaces.global,
             default_snapshot->memory_spaces.global) &&
         loom_spirv_provider_match_u64_override(
             target_op, loom_spirv_target_memory_space_workgroup_ATTR_INDEX,
             selected_snapshot->memory_spaces.workgroup,
             default_snapshot->memory_spaces.workgroup) &&
         loom_spirv_provider_match_u64_override(
             target_op, loom_spirv_target_memory_space_constant_ATTR_INDEX,
             selected_snapshot->memory_spaces.constant,
             default_snapshot->memory_spaces.constant) &&
         loom_spirv_provider_match_u64_override(
             target_op, loom_spirv_target_memory_space_private_ATTR_INDEX,
             selected_snapshot->memory_spaces.private_memory,
             default_snapshot->memory_spaces.private_memory) &&
         loom_spirv_provider_match_u64_override(
             target_op, loom_spirv_target_memory_space_host_ATTR_INDEX,
             selected_snapshot->memory_spaces.host,
             default_snapshot->memory_spaces.host) &&
         loom_spirv_provider_match_u64_override(
             target_op, loom_spirv_target_memory_space_descriptor_ATTR_INDEX,
             selected_snapshot->memory_spaces.descriptor,
             default_snapshot->memory_spaces.descriptor) &&
         loom_spirv_provider_match_enum_override(
             target_op, loom_spirv_target_abi_ATTR_INDEX,
             selected_export_plan->abi_kind, default_export_plan->abi_kind) &&
         loom_spirv_provider_match_string_override(
             module, target_op, loom_spirv_target_export_symbol_ATTR_INDEX,
             selected_export_plan->export_symbol,
             default_export_plan->export_symbol) &&
         loom_spirv_provider_match_enum_override(
             target_op, loom_spirv_target_linkage_ATTR_INDEX,
             selected_export_plan->linkage, default_export_plan->linkage) &&
         loom_spirv_provider_match_u64_override(
             target_op, loom_spirv_target_hal_buffer_resource_flags_ATTR_INDEX,
             selected_export_plan->hal_kernel.buffer_resource_flags,
             default_export_plan->hal_kernel.buffer_resource_flags) &&
         loom_spirv_provider_match_string_override(
             module, target_op, loom_spirv_target_contract_set_key_ATTR_INDEX,
             selected_config->contract_set_key,
             default_config->contract_set_key) &&
         loom_spirv_provider_match_u64_override(
             target_op, loom_spirv_target_contract_feature_bits_ATTR_INDEX,
             selected_config->contract_feature_bits,
             default_config->contract_feature_bits);
}

static iree_status_t loom_spirv_provider_validate_materialized_target_symbol(
    const loom_module_t* module, iree_string_view_t symbol_name,
    const loom_target_bundle_t* selected_bundle, loom_symbol_ref_t target_ref,
    bool* out_reusable) {
  *out_reusable = false;

  const loom_symbol_t* symbol = &module->symbols.entries[target_ref.symbol_id];
  if (symbol->defining_op == NULL) {
    return iree_ok_status();
  }
  if (!loom_spirv_target_isa(symbol->defining_op)) {
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "SPIR-V target materialization symbol '@%.*s' already names a "
        "non-SPIR-V target op",
        (int)symbol_name.size, symbol_name.data);
  }
  if (!loom_spirv_provider_target_record_matches_bundle(
          module, symbol->defining_op, selected_bundle)) {
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "SPIR-V target materialization symbol '@%.*s' already names a "
        "target record that does not match selected bundle '%.*s'",
        (int)symbol_name.size, symbol_name.data,
        (int)selected_bundle->name.size, selected_bundle->name.data);
  }

  *out_reusable = true;
  return iree_ok_status();
}

static iree_status_t loom_spirv_provider_build_target_record_for_bundle(
    loom_builder_t* builder, loom_symbol_ref_t symbol,
    const loom_target_bundle_t* selected_bundle, loom_location_id_t location,
    loom_op_t** out_target_op) {
  const loom_target_bundle_t* default_bundle =
      &loom_spirv_low_target_bundle_vulkan1_3;
  const loom_target_snapshot_t* selected_snapshot = selected_bundle->snapshot;
  const loom_target_snapshot_t* default_snapshot = default_bundle->snapshot;
  const loom_target_export_plan_t* selected_export_plan =
      selected_bundle->export_plan;
  const loom_target_export_plan_t* default_export_plan =
      default_bundle->export_plan;
  const loom_target_config_t* selected_config = selected_bundle->config;
  const loom_target_config_t* default_config = default_bundle->config;

  loom_spirv_target_build_flags_t build_flags = 0;
  uint8_t codegen_format = 0;
  uint8_t artifact_format = 0;
  int64_t default_pointer_bitwidth = 0;
  int64_t index_bitwidth = 0;
  int64_t offset_bitwidth = 0;
  int64_t max_workgroup_size_x = 0;
  int64_t max_workgroup_size_y = 0;
  int64_t max_workgroup_size_z = 0;
  int64_t max_flat_workgroup_size = 0;
  int64_t subgroup_size = 0;
  int64_t max_grid_size_x = 0;
  int64_t max_grid_size_y = 0;
  int64_t max_grid_size_z = 0;
  int64_t max_flat_grid_size = 0;
  int64_t max_workgroup_count_x = 0;
  int64_t max_workgroup_count_y = 0;
  int64_t max_workgroup_count_z = 0;
  int64_t memory_space_generic = 0;
  int64_t memory_space_global = 0;
  int64_t memory_space_workgroup = 0;
  int64_t memory_space_constant = 0;
  int64_t memory_space_private = 0;
  int64_t memory_space_host = 0;
  int64_t memory_space_descriptor = 0;
  uint8_t abi = 0;
  loom_string_id_t export_symbol = LOOM_STRING_ID_INVALID;
  uint8_t linkage = 0;
  int64_t hal_buffer_resource_flags = 0;
  loom_string_id_t contract_set_key = LOOM_STRING_ID_INVALID;
  int64_t contract_feature_bits = 0;

  loom_spirv_provider_prepare_enum_override(
      selected_snapshot->codegen_format, default_snapshot->codegen_format,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_CODEGEN_FORMAT, &build_flags,
      &codegen_format);
  loom_spirv_provider_prepare_enum_override(
      selected_snapshot->artifact_format, default_snapshot->artifact_format,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_ARTIFACT_FORMAT, &build_flags,
      &artifact_format);
  loom_spirv_provider_prepare_u32_override(
      selected_snapshot->default_pointer_bitwidth,
      default_snapshot->default_pointer_bitwidth,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_DEFAULT_POINTER_BITWIDTH, &build_flags,
      &default_pointer_bitwidth);
  loom_spirv_provider_prepare_u32_override(
      selected_snapshot->index_bitwidth, default_snapshot->index_bitwidth,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_INDEX_BITWIDTH, &build_flags,
      &index_bitwidth);
  loom_spirv_provider_prepare_u32_override(
      selected_snapshot->offset_bitwidth, default_snapshot->offset_bitwidth,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_OFFSET_BITWIDTH, &build_flags,
      &offset_bitwidth);
  loom_spirv_provider_prepare_u32_override(
      selected_snapshot->max_workgroup_size.x,
      default_snapshot->max_workgroup_size.x,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_X, &build_flags,
      &max_workgroup_size_x);
  loom_spirv_provider_prepare_u32_override(
      selected_snapshot->max_workgroup_size.y,
      default_snapshot->max_workgroup_size.y,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_Y, &build_flags,
      &max_workgroup_size_y);
  loom_spirv_provider_prepare_u32_override(
      selected_snapshot->max_workgroup_size.z,
      default_snapshot->max_workgroup_size.z,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_Z, &build_flags,
      &max_workgroup_size_z);
  loom_spirv_provider_prepare_u32_override(
      selected_snapshot->max_flat_workgroup_size,
      default_snapshot->max_flat_workgroup_size,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_MAX_FLAT_WORKGROUP_SIZE, &build_flags,
      &max_flat_workgroup_size);
  loom_spirv_provider_prepare_u32_override(
      selected_snapshot->subgroup_size, default_snapshot->subgroup_size,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_SUBGROUP_SIZE, &build_flags,
      &subgroup_size);
  loom_spirv_provider_prepare_u32_override(
      selected_snapshot->max_grid_size.x, default_snapshot->max_grid_size.x,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_MAX_GRID_SIZE_X, &build_flags,
      &max_grid_size_x);
  loom_spirv_provider_prepare_u32_override(
      selected_snapshot->max_grid_size.y, default_snapshot->max_grid_size.y,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_MAX_GRID_SIZE_Y, &build_flags,
      &max_grid_size_y);
  loom_spirv_provider_prepare_u32_override(
      selected_snapshot->max_grid_size.z, default_snapshot->max_grid_size.z,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_MAX_GRID_SIZE_Z, &build_flags,
      &max_grid_size_z);
  IREE_RETURN_IF_ERROR(loom_spirv_provider_prepare_u64_override(
      selected_snapshot->max_flat_grid_size,
      default_snapshot->max_flat_grid_size,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_MAX_FLAT_GRID_SIZE, &build_flags,
      &max_flat_grid_size, IREE_SV("max_flat_grid_size")));
  loom_spirv_provider_prepare_u32_override(
      selected_snapshot->max_workgroup_count.x,
      default_snapshot->max_workgroup_count.x,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_X, &build_flags,
      &max_workgroup_count_x);
  loom_spirv_provider_prepare_u32_override(
      selected_snapshot->max_workgroup_count.y,
      default_snapshot->max_workgroup_count.y,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_Y, &build_flags,
      &max_workgroup_count_y);
  loom_spirv_provider_prepare_u32_override(
      selected_snapshot->max_workgroup_count.z,
      default_snapshot->max_workgroup_count.z,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_Z, &build_flags,
      &max_workgroup_count_z);
  loom_spirv_provider_prepare_u32_override(
      selected_snapshot->memory_spaces.generic,
      default_snapshot->memory_spaces.generic,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_GENERIC, &build_flags,
      &memory_space_generic);
  loom_spirv_provider_prepare_u32_override(
      selected_snapshot->memory_spaces.global,
      default_snapshot->memory_spaces.global,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_GLOBAL, &build_flags,
      &memory_space_global);
  loom_spirv_provider_prepare_u32_override(
      selected_snapshot->memory_spaces.workgroup,
      default_snapshot->memory_spaces.workgroup,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_WORKGROUP, &build_flags,
      &memory_space_workgroup);
  loom_spirv_provider_prepare_u32_override(
      selected_snapshot->memory_spaces.constant,
      default_snapshot->memory_spaces.constant,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_CONSTANT, &build_flags,
      &memory_space_constant);
  loom_spirv_provider_prepare_u32_override(
      selected_snapshot->memory_spaces.private_memory,
      default_snapshot->memory_spaces.private_memory,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_PRIVATE, &build_flags,
      &memory_space_private);
  loom_spirv_provider_prepare_u32_override(
      selected_snapshot->memory_spaces.host,
      default_snapshot->memory_spaces.host,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_HOST, &build_flags,
      &memory_space_host);
  loom_spirv_provider_prepare_u32_override(
      selected_snapshot->memory_spaces.descriptor,
      default_snapshot->memory_spaces.descriptor,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_DESCRIPTOR, &build_flags,
      &memory_space_descriptor);
  loom_spirv_provider_prepare_enum_override(
      selected_export_plan->abi_kind, default_export_plan->abi_kind,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_ABI, &build_flags, &abi);
  IREE_RETURN_IF_ERROR(loom_spirv_provider_prepare_string_override(
      builder->module, selected_export_plan->export_symbol,
      default_export_plan->export_symbol,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_EXPORT_SYMBOL, &build_flags,
      &export_symbol));
  loom_spirv_provider_prepare_enum_override(
      selected_export_plan->linkage, default_export_plan->linkage,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_LINKAGE, &build_flags, &linkage);
  loom_spirv_provider_prepare_u32_override(
      selected_export_plan->hal_kernel.buffer_resource_flags,
      default_export_plan->hal_kernel.buffer_resource_flags,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_HAL_BUFFER_RESOURCE_FLAGS, &build_flags,
      &hal_buffer_resource_flags);
  IREE_RETURN_IF_ERROR(loom_spirv_provider_prepare_string_override(
      builder->module, selected_config->contract_set_key,
      default_config->contract_set_key,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_CONTRACT_SET_KEY, &build_flags,
      &contract_set_key));
  IREE_RETURN_IF_ERROR(loom_spirv_provider_prepare_u64_override(
      selected_config->contract_feature_bits,
      default_config->contract_feature_bits,
      LOOM_SPIRV_TARGET_BUILD_FLAG_HAS_CONTRACT_FEATURE_BITS, &build_flags,
      &contract_feature_bits, IREE_SV("contract_feature_bits")));

  return loom_spirv_target_build(
      builder, build_flags, LOOM_SPIRV_TARGET_KIND_VULKAN1_3, symbol,
      codegen_format, artifact_format, default_pointer_bitwidth, index_bitwidth,
      offset_bitwidth, max_workgroup_size_x, max_workgroup_size_y,
      max_workgroup_size_z, max_flat_workgroup_size, subgroup_size,
      max_grid_size_x, max_grid_size_y, max_grid_size_z, max_flat_grid_size,
      max_workgroup_count_x, max_workgroup_count_y, max_workgroup_count_z,
      memory_space_generic, memory_space_global, memory_space_workgroup,
      memory_space_constant, memory_space_private, memory_space_host,
      memory_space_descriptor, abi, export_symbol, linkage,
      hal_buffer_resource_flags, contract_set_key, contract_feature_bits,
      location, out_target_op);
}

static iree_status_t loom_spirv_provider_resolve_profile_target_ref(
    loom_module_t* module, const loom_target_bundle_t* selected_bundle,
    loom_symbol_ref_t* out_target_ref) {
  *out_target_ref = loom_symbol_ref_null();
  if (module == NULL || module->body == NULL ||
      module->body->block_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V target materialization requires a module "
                            "with a body block");
  }
  if (!loom_spirv_provider_matches_selection_bundle(selected_bundle)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "SPIR-V target materialization requires a selected SPIR-V bundle");
  }
  if (iree_string_view_is_empty(selected_bundle->name)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "SPIR-V target materialization selected a bundle with no name");
  }

  loom_string_id_t symbol_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, selected_bundle->name,
                                                 &symbol_name_id));
  uint16_t symbol_id = loom_module_find_symbol(module, symbol_name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_module_add_symbol(module, symbol_name_id, &symbol_id));
  }
  *out_target_ref = (loom_symbol_ref_t){.module_id = 0, .symbol_id = symbol_id};

  bool reusable = false;
  IREE_RETURN_IF_ERROR(loom_spirv_provider_validate_materialized_target_symbol(
      module, selected_bundle->name, selected_bundle, *out_target_ref,
      &reusable));
  if (reusable) {
    return iree_ok_status();
  }

  loom_block_t* module_block = loom_module_block(module);
  loom_builder_t builder = {0};
  loom_builder_initialize(module, &module->arena, module_block, &builder);
  if (module_block->first_op != NULL) {
    loom_builder_set_before(&builder, module_block->first_op);
  }

  loom_op_t* target_op = NULL;
  return loom_spirv_provider_build_target_record_for_bundle(
      &builder, *out_target_ref, selected_bundle, LOOM_LOCATION_UNKNOWN,
      &target_op);
}

static iree_status_t loom_spirv_provider_materialize_selection(
    const loom_target_provider_t* provider,
    const loom_target_selection_materialization_request_t* request,
    bool* out_materialized, loom_symbol_ref_t* out_target_ref) {
  (void)provider;
  *out_materialized = false;
  *out_target_ref = loom_symbol_ref_null();
  if (!loom_spirv_provider_matches_selection_bundle(
          request->target_selection.bundle)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_spirv_provider_resolve_profile_target_ref(
      request->module, request->target_selection.bundle, out_target_ref));
  *out_materialized = true;
  return iree_ok_status();
}

const loom_target_provider_t loom_spirv_target_provider = {
    .register_context = loom_spirv_ops_register_dialect,
    .initialize_low_descriptor_registry =
        loom_spirv_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_spirv_low_lower_policy_registry_initialize,
    .initialize_math_policy_registry =
        loom_spirv_math_policy_registry_initialize,
    .low_verify_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomSpirvLowVerifyProviders),
            .values = kLoomSpirvLowVerifyProviders,
        },
    .materialize_selection = loom_spirv_provider_materialize_selection,
};

static const loom_target_provider_t* const kLoomSpirvTargetProviders[] = {
    &loom_spirv_target_provider,
};

const loom_target_provider_set_t loom_spirv_target_provider_set = {
    .providers = kLoomSpirvTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomSpirvTargetProviders),
};
