// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/kernel_record.h"

#include <string.h>

#include "loom/codegen/low/function.h"
#include "loom/codegen/low/packet.h"
#include "loom/ops/low/kernel.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/amdhsa_target_id.h"
#include "loom/target/arch/amdgpu/artifact_key.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/emit/native/amdgpu/kernel_entry.h"
#include "loom/target/emit/native/amdgpu/preflight.h"
#include "loom/target/emit/native/amdgpu/storage_layout.h"
#include "loom/target/launch.h"

static bool loom_amdgpu_kernel_record_has_abi_source(
    const loom_amdgpu_hal_kernel_abi_verify_result_t* abi_verify,
    loom_amdgpu_hal_kernel_abi_source_kind_t source_kind) {
  return ((abi_verify->live_in_source_bits >> source_kind) & 1u) != 0;
}

static iree_status_t loom_amdgpu_kernel_record_concat3(
    iree_string_view_t a, iree_string_view_t b, iree_string_view_t c,
    iree_string_view_t* out_value, iree_arena_allocator_t* arena) {
  *out_value = iree_string_view_empty();
  iree_host_size_t prefix_length = 0;
  iree_host_size_t length = 0;
  if (!iree_host_size_checked_add(a.size, b.size, &prefix_length) ||
      !iree_host_size_checked_add(prefix_length, c.size, &length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU kernel record string length overflows");
  }
  char* data = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, length, (void**)&data));
  char* cursor = data;
  if (!iree_string_view_is_empty(a)) {
    memcpy(cursor, a.data, a.size);
    cursor += a.size;
  }
  if (!iree_string_view_is_empty(b)) {
    memcpy(cursor, b.data, b.size);
    cursor += b.size;
  }
  if (!iree_string_view_is_empty(c)) {
    memcpy(cursor, c.data, c.size);
  }
  *out_value = iree_make_string_view(data, length);
  return iree_ok_status();
}

static bool loom_amdgpu_kernel_record_symbol_start_char(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_' ||
         c == '$';
}

static bool loom_amdgpu_kernel_record_symbol_continue_char(char c) {
  return loom_amdgpu_kernel_record_symbol_start_char(c) ||
         (c >= '0' && c <= '9') || c == '.';
}

static iree_status_t loom_amdgpu_kernel_record_validate_symbol(
    iree_string_view_t symbol) {
  if (iree_string_view_is_empty(symbol)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel emission symbol is required");
  }
  if (!loom_amdgpu_kernel_record_symbol_start_char(symbol.data[0])) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel emission symbol '%.*s' has an invalid first character",
        (int)symbol.size, symbol.data);
  }
  for (iree_host_size_t i = 1; i < symbol.size; ++i) {
    if (!loom_amdgpu_kernel_record_symbol_continue_char(symbol.data[i])) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU kernel emission symbol '%.*s' contains an invalid character",
          (int)symbol.size, symbol.data);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_record_symbol_name(
    const loom_module_t* module, const loom_op_t* function_op,
    iree_string_view_t* out_symbol) {
  *out_symbol = iree_string_view_empty();
  loom_symbol_ref_t symbol_ref = loom_low_function_callee(function_op);
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel emission function symbol is invalid");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id == LOOM_STRING_ID_INVALID ||
      symbol->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel emission function symbol has no "
                            "module string");
  }
  *out_symbol = module->strings.entries[symbol->name_id];
  return loom_amdgpu_kernel_record_validate_symbol(*out_symbol);
}

static iree_status_t loom_amdgpu_kernel_record_export_symbol(
    const loom_low_resolved_target_t* target, const loom_module_t* module,
    const loom_op_t* function_op, iree_string_view_t* out_symbol) {
  const loom_target_bundle_t* bundle = loom_low_resolved_target_bundle(target);
  if (!iree_string_view_is_empty(bundle->export_plan->export_symbol)) {
    *out_symbol = bundle->export_plan->export_symbol;
    return loom_amdgpu_kernel_record_validate_symbol(*out_symbol);
  }
  return loom_amdgpu_kernel_record_symbol_name(module, function_op, out_symbol);
}

static iree_status_t loom_amdgpu_kernel_record_validate_target(
    const loom_low_resolved_target_t* target) {
  const loom_target_bundle_t* bundle = loom_low_resolved_target_bundle(target);
  const loom_target_snapshot_t* snapshot = bundle->snapshot;
  const loom_target_export_plan_t* export_plan = bundle->export_plan;
  if (target->descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel emission target bundle is required");
  }
  if (snapshot->codegen_format != LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU kernel emission requires low_native codegen snapshots");
  }
  if (snapshot->artifact_format != LOOM_TARGET_ARTIFACT_FORMAT_ELF) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU kernel emission requires ELF artifacts");
  }
  if (export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU kernel emission requires a HAL kernel ABI");
  }
  if (export_plan->linkage != LOOM_TARGET_LINKAGE_DEFAULT) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU kernel emission requires default linkage");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_record_validate_function_shape(
    const loom_op_t* function_op) {
  if (!loom_low_function_def_isa(function_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel emission requires low.func.def or low.kernel.def");
  }
  const loom_region_t* body = loom_low_function_const_body(function_op);
  if (body == NULL || body->block_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel emission function body is required");
  }
  const loom_block_t* entry_block = loom_region_const_entry_block(body);
  if (entry_block->arg_count != 0 || function_op->result_count != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU kernel emission requires ABI-lowered kernels with no low "
        "function arguments or results");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_record_build_storage_layout(
    const loom_low_storage_layout_t* source_layout,
    iree_arena_allocator_t* arena, loom_amdgpu_storage_layout_t* out_layout) {
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_storage_layout_build(source_layout, arena, out_layout));
  if (out_layout->segment_sizes.group_segment_fixed_size > UINT32_MAX ||
      out_layout->segment_sizes.private_segment_fixed_size > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU kernel emission fixed segment sizes exceed metadata limits");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_record_collect_descriptor_flags(
    const loom_low_schedule_table_t* schedule,
    const loom_amdgpu_hal_kernel_abi_verify_result_t* abi_verify,
    bool target_has_packed_workitem_id, bool target_has_cluster_launch_state,
    loom_amdgpu_kernel_descriptor_flags_t* out_flags) {
  loom_amdgpu_kernel_descriptor_flags_t flags = 0;
  if (loom_amdgpu_kernel_record_has_abi_source(
          abi_verify, LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_DISPATCH_PTR)) {
    flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_DISPATCH_PTR;
  }
  if (loom_amdgpu_kernel_record_has_abi_source(
          abi_verify, LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_KERNARG_SEGMENT_PTR)) {
    flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_KERNARG_SEGMENT_PTR;
  }
  if (loom_amdgpu_kernel_record_has_abi_source(
          abi_verify, LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_DISPATCH_ID)) {
    flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_DISPATCH_ID;
  }

  const loom_amdgpu_hal_kernel_abi_launch_workgroup_id_flags_t launch_flags =
      abi_verify->launch_workgroup_id_flags;
  if (loom_amdgpu_kernel_record_has_abi_source(
          abi_verify, LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_X) ||
      iree_any_bit_set(launch_flags,
                       LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_X)) {
    flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X;
  }
  if (loom_amdgpu_kernel_record_has_abi_source(
          abi_verify, LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_Y) ||
      iree_any_bit_set(launch_flags,
                       LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_Y)) {
    flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Y;
  }
  if (loom_amdgpu_kernel_record_has_abi_source(
          abi_verify, LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_Z) ||
      iree_any_bit_set(launch_flags,
                       LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_Z)) {
    flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Z;
  }

  if (target_has_cluster_launch_state) {
    loom_target_workgroup_cluster_size_t cluster_size = {0};
    if (loom_low_kernel_def_static_workgroup_cluster_size(schedule->function_op,
                                                          &cluster_size)) {
      // A clustered dispatch requires the x enable bit even when the kernel
      // does not consume a coordinate. This is the launch-state enable for
      // TTMP6/TTMP7/TTMP9 on gfx1250.
      flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X;
    }
  }

  const bool has_packed_xy = loom_amdgpu_kernel_record_has_abi_source(
      abi_verify, LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_PACKED_XY);
  const bool has_packed_xyz = loom_amdgpu_kernel_record_has_abi_source(
      abi_verify, LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_PACKED_XYZ);
  if (has_packed_xy || has_packed_xyz) {
    if (!target_has_packed_workitem_id) {
      const loom_amdgpu_hal_kernel_abi_source_kind_t source_kind =
          has_packed_xyz
              ? LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_PACKED_XYZ
              : LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_PACKED_XY;
      const iree_string_view_t source_name =
          loom_amdgpu_hal_kernel_abi_source_name(source_kind);
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU kernel emission target does not support %.*s live-ins",
          (int)source_name.size, source_name.data);
    }
    flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X |
             LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y;
    if (has_packed_xyz) {
      flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Z;
    }
  } else {
    const bool has_workitem_x = loom_amdgpu_kernel_record_has_abi_source(
        abi_verify, LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_X);
    const bool has_workitem_y = loom_amdgpu_kernel_record_has_abi_source(
        abi_verify, LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_Y);
    const bool has_workitem_z = loom_amdgpu_kernel_record_has_abi_source(
        abi_verify, LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_Z);
    if (target_has_packed_workitem_id && (has_workitem_y || has_workitem_z)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU kernel emission requires packed workitem-id live-ins when "
          "workitem_id.y/z are used on this target");
    }
    if (has_workitem_x) {
      flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X;
    }
    if (has_workitem_y) {
      flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y;
    }
    if (has_workitem_z) {
      flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Z;
    }
  }
  if (iree_any_bit_set(
          flags, LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Z)) {
    flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X |
             LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y;
  } else if (iree_any_bit_set(
                 flags,
                 LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y)) {
    flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X;
  }
  *out_flags = flags;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_record_build_metadata_arguments(
    const loom_amdgpu_hal_kernel_abi_layout_t* abi_layout,
    const loom_amdgpu_metadata_argument_t** out_arguments,
    iree_arena_allocator_t* arena) {
  *out_arguments = NULL;
  const iree_host_size_t argument_count = abi_layout->parameter_count;
  if (argument_count == 0) {
    return iree_ok_status();
  }
  loom_amdgpu_metadata_argument_t* arguments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, argument_count, sizeof(*arguments), (void**)&arguments));
  for (iree_host_size_t i = 0; i < abi_layout->resource_count; ++i) {
    const loom_amdgpu_hal_kernarg_resource_t* resource =
        &abi_layout->resources[i];
    IREE_ASSERT_LT(resource->parameter_index, argument_count);
    arguments[resource->parameter_index] = (loom_amdgpu_metadata_argument_t){
        .name = resource->name,
        .offset = resource->kernarg_offset,
        .size = resource->kernarg_size,
        .alignment = resource->kernarg_alignment,
        .kind = LOOM_AMDGPU_METADATA_ARGUMENT_GLOBAL_BUFFER,
        .address_space = IREE_SV("global"),
    };
  }
  for (iree_host_size_t i = 0; i < abi_layout->direct_arg_count; ++i) {
    const loom_amdgpu_hal_kernarg_direct_arg_t* direct_arg =
        &abi_layout->direct_args[i];
    IREE_ASSERT_LT(direct_arg->parameter_index, argument_count);
    arguments[direct_arg->parameter_index] = (loom_amdgpu_metadata_argument_t){
        .name = direct_arg->name,
        .offset = direct_arg->kernarg_offset,
        .size = direct_arg->kernarg_size,
        .alignment = direct_arg->kernarg_alignment,
        .kind = LOOM_AMDGPU_METADATA_ARGUMENT_BY_VALUE,
    };
  }
  *out_arguments = arguments;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_record_max_flat_workgroup_size(
    const loom_target_snapshot_t* snapshot,
    const loom_target_hal_kernel_abi_t* hal_kernel,
    uint32_t* out_max_flat_workgroup_size) {
  const loom_target_workgroup_size_t* required =
      &hal_kernel->required_workgroup_size;
  if (loom_target_workgroup_size_is_concrete(required)) {
    const bool has_flat_workgroup_size =
        loom_target_workgroup_size_flat_product_u32(
            required, out_max_flat_workgroup_size);
    IREE_ASSERT(has_flat_workgroup_size,
                "validated HAL kernel workgroup size fits uint32_t");
    (void)has_flat_workgroup_size;
    return iree_ok_status();
  }
  *out_max_flat_workgroup_size = hal_kernel->flat_workgroup_size_max != 0
                                     ? hal_kernel->flat_workgroup_size_max
                                     : snapshot->max_flat_workgroup_size;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_kernel_record_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_kernel_record_options_t* options,
    loom_amdgpu_kernel_record_t* out_record,
    iree_arena_allocator_t* scratch_arena) {
  *out_record = (loom_amdgpu_kernel_record_t){0};
  loom_amdgpu_native_preflight_t derived_preflight = {0};
  const loom_amdgpu_native_preflight_t* preflight = options->preflight;
  if (preflight != NULL) {
    if (preflight->schedule != schedule ||
        preflight->allocation != allocation) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU kernel record preflight does not match the scheduled "
          "function");
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_native_preflight_analyze(
        schedule, allocation, /*options=*/NULL, &derived_preflight));
    preflight = &derived_preflight;
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_kernel_record_validate_target(&schedule->target));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_kernel_record_validate_function_shape(schedule->function_op));

  const loom_amdgpu_hal_kernel_abi_verify_result_t* abi_verify =
      options->abi_verify;

  iree_string_view_t symbol = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_export_symbol(
      &schedule->target, schedule->module, schedule->function_op, &symbol));

  loom_amdgpu_hal_kernel_abi_layout_t derived_abi_layout = {0};
  const loom_amdgpu_hal_kernel_abi_layout_t* abi_layout = options->abi_layout;
  if (abi_layout != NULL) {
    if (abi_layout->function_op != schedule->function_op) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU kernel record ABI layout does not belong to the scheduled "
          "function");
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_layout_from_low(
        schedule->module, schedule->function_op, &derived_abi_layout,
        scratch_arena));
    abi_layout = &derived_abi_layout;
  }

  loom_amdgpu_storage_layout_t storage_layout = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_build_storage_layout(
      &schedule->storage_layout, scratch_arena, &storage_layout));

  const loom_target_bundle_t* bundle =
      loom_low_resolved_target_bundle(&schedule->target);
  const loom_target_hal_kernel_abi_t* hal_kernel =
      &bundle->export_plan->hal_kernel;
  const loom_amdgpu_target_facts_t* target_facts =
      loom_amdgpu_target_facts_cast(schedule->target.target_facts);
  if (target_facts == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU kernel emission requires an AMDGPU "
                            "processor target record");
  }
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_target_processor(target_facts->identity.target);
  IREE_ASSERT(processor != NULL);
  const uint32_t wavefront_size = bundle->snapshot->subgroup_size;
  IREE_ASSERT(loom_amdgpu_processor_properties_support_wavefront_size(
      &processor->properties, wavefront_size));

  const uint32_t user_sgpr_count = abi_verify->user_sgpr_count;
  loom_amdgpu_kernel_descriptor_flags_t descriptor_flags = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_collect_descriptor_flags(
      schedule, abi_verify,
      loom_amdgpu_processor_properties_kernel_descriptor_has_flags(
          &processor->properties,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_PACKED_WORKITEM_ID),
      loom_amdgpu_processor_properties_have_flags(
          &processor->properties,
          LOOM_AMDGPU_PROCESSOR_INFO_FLAG_CLUSTER_LAUNCH_STATE),
      &descriptor_flags));
  uint32_t system_vgpr_workitem_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_kernel_descriptor_workitem_id_mode_from_flags(
          descriptor_flags, &system_vgpr_workitem_id));

  const loom_amdgpu_kernel_entry_envelope_t* entry_envelope =
      loom_amdgpu_kernel_entry_envelope_for_properties(&processor->properties);
  uint32_t next_free_sgpr = preflight->next_free_sgpr > user_sgpr_count
                                ? preflight->next_free_sgpr
                                : user_sgpr_count;
  if (entry_envelope->minimum_sgpr_count > next_free_sgpr) {
    next_free_sgpr = entry_envelope->minimum_sgpr_count;
  }
  uint32_t next_free_vgpr = preflight->next_free_vgpr;
  if (entry_envelope->minimum_vgpr_count > next_free_vgpr) {
    next_free_vgpr = entry_envelope->minimum_vgpr_count;
  }

  iree_string_view_t artifact_target_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_artifact_key_format_arena(
      &target_facts->identity, scratch_arena, &artifact_target_key));
  iree_string_view_t code_object_target_id = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_amdhsa_target_id_format(
      &target_facts->identity, scratch_arena, &code_object_target_id));
  iree_string_view_t descriptor_symbol = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_concat3(
      symbol, IREE_SV(".kd"), iree_string_view_empty(), &descriptor_symbol,
      scratch_arena));

  const loom_amdgpu_metadata_argument_t* arguments = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_build_metadata_arguments(
      abi_layout, &arguments, scratch_arena));
  const bool has_required_workgroup_size =
      loom_target_workgroup_size_is_concrete(
          &hal_kernel->required_workgroup_size);
  loom_target_workgroup_cluster_size_t workgroup_cluster_size = {0};
  const bool has_workgroup_cluster_size =
      loom_low_kernel_def_static_workgroup_cluster_size(
          schedule->function_op, &workgroup_cluster_size);
  uint32_t max_flat_workgroup_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_max_flat_workgroup_size(
      bundle->snapshot, hal_kernel, &max_flat_workgroup_size));

  *out_record = (loom_amdgpu_kernel_record_t){
      .symbol = symbol,
      .descriptor_symbol = descriptor_symbol,
      .artifact_target_key = artifact_target_key,
      .code_object_target_id = code_object_target_id,
      .processor = processor,
      .abi_layout = *abi_layout,
      .storage_layout = storage_layout,
      .metadata =
          {
              .name = symbol,
              .descriptor_symbol = descriptor_symbol,
              .kernarg_segment_size = abi_layout->kernarg_segment_size,
              .kernarg_segment_alignment =
                  abi_layout->kernarg_segment_alignment,
              .wavefront_size = wavefront_size,
              .group_segment_fixed_size = (uint32_t)storage_layout.segment_sizes
                                              .group_segment_fixed_size,
              .private_segment_fixed_size =
                  (uint32_t)
                      storage_layout.segment_sizes.private_segment_fixed_size,
              .sgpr_count = next_free_sgpr,
              .vgpr_count = next_free_vgpr,
              .max_flat_workgroup_size = max_flat_workgroup_size,
              .required_workgroup_size = hal_kernel->required_workgroup_size,
              .has_required_workgroup_size = has_required_workgroup_size,
              .workgroup_cluster_size = workgroup_cluster_size,
              .has_workgroup_cluster_size = has_workgroup_cluster_size,
              .target_extensions =
                  target_facts->properties.kernel_metadata_extensions,
              .arguments = arguments,
              .argument_count = abi_layout->parameter_count,
          },
      .descriptor_flags = descriptor_flags,
      .system_vgpr_workitem_id = system_vgpr_workitem_id,
      .user_sgpr_count = user_sgpr_count,
  };
  return iree_ok_status();
}
