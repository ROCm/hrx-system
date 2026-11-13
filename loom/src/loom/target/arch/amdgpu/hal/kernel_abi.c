// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/hal/kernel_abi.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/function.h"
#include "loom/codegen/low/representation_binding.h"
#include "loom/error/parameter.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/registers.h"

#define LOOM_AMDGPU_HAL_KERNEL_ABI_COORDINATE_DIMENSION_COUNT 3u
#define LOOM_AMDGPU_HAL_KERNEL_ABI_NO_DIMENSION UINT8_MAX

typedef enum loom_amdgpu_hal_kernel_abi_reg_class_e {
  LOOM_AMDGPU_HAL_KERNEL_ABI_REG_CLASS_NONE = 0,
  LOOM_AMDGPU_HAL_KERNEL_ABI_REG_CLASS_SGPR = 1,
  LOOM_AMDGPU_HAL_KERNEL_ABI_REG_CLASS_VGPR = 2,
} loom_amdgpu_hal_kernel_abi_reg_class_t;

typedef enum loom_amdgpu_hal_kernel_abi_source_role_e {
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_ROLE_UNKNOWN = 0,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_ROLE_USER_SGPR = 1,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_ROLE_WORKGROUP_ID = 2,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_ROLE_WORKITEM_ID = 3,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_ROLE_PACKED_WORKITEM_ID = 4,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_ROLE_M0 = 5,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_ROLE_FIXED_SGPR = 6,
} loom_amdgpu_hal_kernel_abi_source_role_t;

typedef struct loom_amdgpu_hal_kernel_abi_source_info_t {
  // Stable low.live_in source ID stored in IR.
  uint64_t stable_id;
  // Stable low.live_in source spelling stored in IR.
  iree_string_view_t name;
  // ABI placement and validation role for this live-in source.
  loom_amdgpu_hal_kernel_abi_source_role_t role;
  // Expected ordinary low-register class, or NONE for special registers.
  loom_amdgpu_hal_kernel_abi_reg_class_t reg_class;
  // Required low location count for the live-in value.
  uint8_t unit_count;
  // Coordinate dimension for workgroup/workitem roles, or NO_DIMENSION.
  uint8_t dimension;
  // Workgroup-coordinate launch state required when importing this source.
  loom_amdgpu_hal_kernel_abi_launch_workgroup_id_flags_t
      launch_workgroup_id_flags;
  // Fixed SGPR location for FIXED_SGPR sources, or UINT16_MAX.
  uint16_t fixed_location;
} loom_amdgpu_hal_kernel_abi_source_info_t;

#define LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_INFO(                               \
    kind, source_role, source_reg_class, source_unit_count, source_dimension) \
  [LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_##kind] = {                              \
      .stable_id = LOOM_AMDGPU_HAL_KERNEL_ABI_##kind##_SOURCE_ID,             \
      .name = IREE_SVL(LOOM_AMDGPU_HAL_KERNEL_ABI_##kind##_SOURCE),           \
      .role = LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_ROLE_##source_role,           \
      .reg_class = LOOM_AMDGPU_HAL_KERNEL_ABI_REG_CLASS_##source_reg_class,   \
      .unit_count = source_unit_count,                                        \
      .dimension = source_dimension,                                          \
      .launch_workgroup_id_flags = 0,                                         \
      .fixed_location = UINT16_MAX,                                           \
  }

#define LOOM_AMDGPU_HAL_KERNEL_ABI_FIXED_SGPR_SOURCE_INFO(kind, location, \
                                                          launch_flags)   \
  [LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_##kind] = {                          \
      .stable_id = LOOM_AMDGPU_HAL_KERNEL_ABI_##kind##_SOURCE_ID,         \
      .name = IREE_SVL(LOOM_AMDGPU_HAL_KERNEL_ABI_##kind##_SOURCE),       \
      .role = LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_ROLE_FIXED_SGPR,          \
      .reg_class = LOOM_AMDGPU_HAL_KERNEL_ABI_REG_CLASS_SGPR,             \
      .unit_count = 1,                                                    \
      .dimension = LOOM_AMDGPU_HAL_KERNEL_ABI_NO_DIMENSION,               \
      .launch_workgroup_id_flags = launch_flags,                          \
      .fixed_location = location,                                         \
  }

static const loom_amdgpu_hal_kernel_abi_source_info_t
    kLoomAmdgpuHalKernelAbiSourceInfos[] = {
        LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_INFO(
            KERNARG_SEGMENT_PTR, USER_SGPR, SGPR, 2,
            LOOM_AMDGPU_HAL_KERNEL_ABI_NO_DIMENSION),
        LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_INFO(
            DISPATCH_PTR, USER_SGPR, SGPR, 2,
            LOOM_AMDGPU_HAL_KERNEL_ABI_NO_DIMENSION),
        LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_INFO(
            DISPATCH_ID, USER_SGPR, SGPR, 2,
            LOOM_AMDGPU_HAL_KERNEL_ABI_NO_DIMENSION),
        LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_INFO(WORKGROUP_ID_X, WORKGROUP_ID,
                                               SGPR, 1, 0),
        LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_INFO(WORKGROUP_ID_Y, WORKGROUP_ID,
                                               SGPR, 1, 1),
        LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_INFO(WORKGROUP_ID_Z, WORKGROUP_ID,
                                               SGPR, 1, 2),
        LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_INFO(WORKITEM_ID_X, WORKITEM_ID, VGPR,
                                               1, 0),
        LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_INFO(WORKITEM_ID_Y, WORKITEM_ID, VGPR,
                                               1, 1),
        LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_INFO(WORKITEM_ID_Z, WORKITEM_ID, VGPR,
                                               1, 2),
        LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_INFO(
            WORKITEM_ID_PACKED_XY, PACKED_WORKITEM_ID, VGPR, 1,
            LOOM_AMDGPU_HAL_KERNEL_ABI_NO_DIMENSION),
        LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_INFO(
            WORKITEM_ID_PACKED_XYZ, PACKED_WORKITEM_ID, VGPR, 1,
            LOOM_AMDGPU_HAL_KERNEL_ABI_NO_DIMENSION),
        LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_INFO(
            M0, M0, NONE, 1, LOOM_AMDGPU_HAL_KERNEL_ABI_NO_DIMENSION),
        LOOM_AMDGPU_HAL_KERNEL_ABI_FIXED_SGPR_SOURCE_INFO(
            CLUSTER_WORKGROUP_INFO, 114,
            LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_KNOWN_FLAGS),
        LOOM_AMDGPU_HAL_KERNEL_ABI_FIXED_SGPR_SOURCE_INFO(
            CLUSTER_ID_YZ, 115,
            LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_KNOWN_FLAGS),
        LOOM_AMDGPU_HAL_KERNEL_ABI_FIXED_SGPR_SOURCE_INFO(
            CLUSTER_ID_X, 117,
            LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_X),
        LOOM_AMDGPU_HAL_KERNEL_ABI_FIXED_SGPR_SOURCE_INFO(
            CLUSTER_ID_Y, 115,
            LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_X |
                LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_Y),
        LOOM_AMDGPU_HAL_KERNEL_ABI_FIXED_SGPR_SOURCE_INFO(
            CLUSTER_ID_Z, 115,
            LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_X |
                LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_Z),
        LOOM_AMDGPU_HAL_KERNEL_ABI_FIXED_SGPR_SOURCE_INFO(
            CLUSTER_WORKGROUP_INFO_X, 114,
            LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_X),
        LOOM_AMDGPU_HAL_KERNEL_ABI_FIXED_SGPR_SOURCE_INFO(
            CLUSTER_WORKGROUP_INFO_XY, 114,
            LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_X |
                LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_Y),
        LOOM_AMDGPU_HAL_KERNEL_ABI_FIXED_SGPR_SOURCE_INFO(
            CLUSTER_WORKGROUP_INFO_XZ, 114,
            LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_X |
                LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_Z),
};
static_assert(IREE_ARRAYSIZE(kLoomAmdgpuHalKernelAbiSourceInfos) <= 64,
              "AMDGPU HAL ABI source presence fits its retained bitset");

#undef LOOM_AMDGPU_HAL_KERNEL_ABI_FIXED_SGPR_SOURCE_INFO
#undef LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_INFO

static const loom_amdgpu_hal_kernel_abi_source_info_t*
loom_amdgpu_hal_kernel_abi_source_info_from_kind(
    loom_amdgpu_hal_kernel_abi_source_kind_t source_kind) {
  if (source_kind <= LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_UNKNOWN) {
    return NULL;
  }
  const iree_host_size_t source_index = (iree_host_size_t)source_kind;
  if (source_index >= IREE_ARRAYSIZE(kLoomAmdgpuHalKernelAbiSourceInfos)) {
    return NULL;
  }
  const loom_amdgpu_hal_kernel_abi_source_info_t* source_info =
      &kLoomAmdgpuHalKernelAbiSourceInfos[source_index];
  if (source_info->role == LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_ROLE_UNKNOWN) {
    return NULL;
  }
  return source_info;
}

iree_string_view_t loom_amdgpu_hal_kernel_abi_source_name(
    loom_amdgpu_hal_kernel_abi_source_kind_t source_kind) {
  const loom_amdgpu_hal_kernel_abi_source_info_t* source_info =
      loom_amdgpu_hal_kernel_abi_source_info_from_kind(source_kind);
  if (source_info == NULL) {
    return iree_string_view_empty();
  }
  return source_info->name;
}

static loom_amdgpu_hal_kernel_abi_source_kind_t
loom_amdgpu_hal_kernel_abi_source_kind_from_stable_id(uint64_t source_id) {
  for (iree_host_size_t i = 1;
       i < IREE_ARRAYSIZE(kLoomAmdgpuHalKernelAbiSourceInfos); ++i) {
    const loom_amdgpu_hal_kernel_abi_source_info_t* source_info =
        &kLoomAmdgpuHalKernelAbiSourceInfos[i];
    if (source_info->stable_id != 0 && source_info->stable_id == source_id) {
      return (loom_amdgpu_hal_kernel_abi_source_kind_t)i;
    }
  }
  return LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_UNKNOWN;
}

static const loom_low_descriptor_t*
loom_amdgpu_hal_kernel_abi_low_op_descriptor(
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* op) {
  return &descriptor_set->descriptors[loom_low_op_descriptor(op)];
}

loom_amdgpu_hal_kernel_abi_source_kind_t
loom_amdgpu_hal_kernel_abi_live_in_source_kind(const loom_module_t* module,
                                               loom_value_id_t value_id) {
  if (module == NULL || value_id >= module->values.count) {
    return LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_UNKNOWN;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_UNKNOWN;
  }
  const loom_op_t* def_op = loom_value_def_op(value);
  if (def_op == NULL || !loom_low_live_in_isa(def_op)) {
    return LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_UNKNOWN;
  }
  return loom_amdgpu_hal_kernel_abi_source_kind_from_stable_id(
      (uint64_t)loom_low_live_in_source_id(def_op));
}

static uint16_t loom_amdgpu_hal_kernel_abi_reg_class_id(
    loom_amdgpu_hal_kernel_abi_reg_class_t reg_class) {
  return reg_class == LOOM_AMDGPU_HAL_KERNEL_ABI_REG_CLASS_SGPR
             ? LOOM_AMDGPU_REG_CLASS_ID_SGPR
             : LOOM_AMDGPU_REG_CLASS_ID_VGPR;
}

static bool loom_amdgpu_hal_kernel_abi_register_value_matches(
    const loom_module_t* module,
    const loom_low_register_type_resolver_t* register_type_resolver,
    loom_value_id_t value_id, loom_amdgpu_hal_kernel_abi_reg_class_t reg_class,
    uint32_t unit_count) {
  const uint16_t expected_class_id =
      loom_amdgpu_hal_kernel_abi_reg_class_id(reg_class);
  loom_type_t type = loom_module_value_type(module, value_id);
  uint16_t actual_class_id = LOOM_LOW_REG_CLASS_NONE;
  bool found_actual_class = loom_low_register_type_resolver_try_resolve(
      register_type_resolver, type, &actual_class_id, NULL);
  return found_actual_class && actual_class_id == expected_class_id &&
         loom_low_register_type_unit_count(type) == unit_count;
}

static bool loom_amdgpu_hal_kernel_abi_direct_arg_unit_count_supported(
    uint32_t unit_count) {
  return unit_count >= 1 &&
         unit_count <= LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_ARG_MAX_UNIT_COUNT;
}

static bool loom_amdgpu_hal_kernel_abi_direct_arg_type_matches(
    const loom_module_t* module,
    const loom_low_register_type_resolver_t* register_type_resolver,
    loom_value_id_t value_id) {
  const uint16_t expected_class_id = LOOM_AMDGPU_REG_CLASS_ID_SGPR;
  loom_type_t type = loom_module_value_type(module, value_id);
  uint16_t actual_class_id = LOOM_LOW_REG_CLASS_NONE;
  bool found_actual_class = loom_low_register_type_resolver_try_resolve(
      register_type_resolver, type, &actual_class_id, NULL);
  const uint32_t unit_count = loom_low_register_type_unit_count(type);
  return found_actual_class && actual_class_id == expected_class_id &&
         loom_amdgpu_hal_kernel_abi_direct_arg_unit_count_supported(unit_count);
}

static bool loom_amdgpu_hal_kernel_abi_single_physical_register_matches(
    const loom_module_t* module,
    const loom_low_register_type_resolver_t* register_type_resolver,
    loom_value_id_t value_id) {
  loom_type_t type = loom_module_value_type(module, value_id);
  uint16_t actual_class_id = LOOM_LOW_REG_CLASS_NONE;
  const loom_low_reg_class_t* actual_class = NULL;
  bool found_actual_class = loom_low_register_type_resolver_try_resolve(
      register_type_resolver, type, &actual_class_id, &actual_class);
  return found_actual_class && actual_class != NULL &&
         loom_low_register_type_unit_count(type) == 1 &&
         actual_class->allocatable_count == 1 &&
         iree_any_bit_set(actual_class->flags,
                          LOOM_LOW_REG_CLASS_FLAG_PHYSICAL) &&
         iree_any_bit_set(actual_class->flags,
                          LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE);
}

static void loom_amdgpu_hal_kernel_abi_initialize_live_in_values(
    loom_value_id_t* live_in_values, iree_host_size_t live_in_value_count) {
  for (iree_host_size_t i = 0; i < live_in_value_count; ++i) {
    live_in_values[i] = LOOM_VALUE_ID_INVALID;
  }
}

static bool loom_amdgpu_hal_kernel_abi_has_source_role_live_in(
    const loom_value_id_t* live_in_values,
    loom_amdgpu_hal_kernel_abi_source_role_t role) {
  for (iree_host_size_t i = 1;
       i < IREE_ARRAYSIZE(kLoomAmdgpuHalKernelAbiSourceInfos); ++i) {
    const loom_amdgpu_hal_kernel_abi_source_info_t* source_info =
        &kLoomAmdgpuHalKernelAbiSourceInfos[i];
    if (source_info->role == role &&
        live_in_values[i] != LOOM_VALUE_ID_INVALID) {
      return true;
    }
  }
  return false;
}

static void loom_amdgpu_hal_kernel_abi_append_fixed_value(
    loom_low_allocation_fixed_value_t* fixed_values,
    iree_host_size_t* fixed_value_count, loom_value_id_t value_id,
    uint32_t location_base, uint32_t location_count) {
  IREE_ASSERT_LT(*fixed_value_count,
                 LOOM_AMDGPU_HAL_KERNEL_ABI_MAX_FIXED_VALUE_COUNT);
  fixed_values[(*fixed_value_count)++] = (loom_low_allocation_fixed_value_t){
      .value_id = value_id,
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = location_base,
      .location_count = location_count,
  };
}

static void loom_amdgpu_hal_kernel_abi_build_fixed_values(
    const loom_value_id_t* live_in_values,
    const loom_amdgpu_processor_properties_t* properties,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  static const loom_amdgpu_hal_kernel_abi_source_kind_t kUserSgprSourceKinds[] =
      {
          LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_DISPATCH_PTR,
          LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_KERNARG_SEGMENT_PTR,
          LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_DISPATCH_ID,
      };
  uint32_t user_sgpr_base = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kUserSgprSourceKinds); ++i) {
    const loom_amdgpu_hal_kernel_abi_source_kind_t source_kind =
        kUserSgprSourceKinds[i];
    const loom_value_id_t value_id = live_in_values[source_kind];
    if (value_id == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    const loom_amdgpu_hal_kernel_abi_source_info_t* source_info =
        loom_amdgpu_hal_kernel_abi_source_info_from_kind(source_kind);
    IREE_ASSERT(source_info != NULL,
                "verified AMDGPU HAL ABI user SGPR source info");
    loom_amdgpu_hal_kernel_abi_append_fixed_value(
        result->fixed_values, &result->fixed_value_count, value_id,
        user_sgpr_base, source_info->unit_count);
    user_sgpr_base += source_info->unit_count;
  }
  if (loom_amdgpu_processor_properties_kernel_descriptor_has_flags(
          properties,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_USER_FLAT_SCRATCH_INIT)) {
    result->flat_scratch_sgprs.init = user_sgpr_base;
    // USER_SGPR_COUNT may exceed the enabled input count. Keep workgroup IDs
    // fixed even when spill repair is the first producer of private storage.
    user_sgpr_base += 2;
  }
  result->user_sgpr_count = user_sgpr_base;

  uint32_t workgroup_id_sgpr = user_sgpr_base;
  for (iree_host_size_t i = 1;
       i < IREE_ARRAYSIZE(kLoomAmdgpuHalKernelAbiSourceInfos); ++i) {
    const loom_amdgpu_hal_kernel_abi_source_info_t* source_info =
        &kLoomAmdgpuHalKernelAbiSourceInfos[i];
    const loom_value_id_t value_id = live_in_values[i];
    if (value_id == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    uint32_t location_base = 0;
    switch (source_info->role) {
      case LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_ROLE_USER_SGPR:
        continue;
      case LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_ROLE_WORKGROUP_ID:
        location_base = workgroup_id_sgpr++;
        break;
      case LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_ROLE_WORKITEM_ID:
        location_base = source_info->dimension;
        break;
      case LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_ROLE_PACKED_WORKITEM_ID:
      case LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_ROLE_M0:
        break;
      case LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_ROLE_FIXED_SGPR:
        IREE_ASSERT_NE(source_info->fixed_location, UINT16_MAX,
                       "verified AMDGPU HAL ABI fixed SGPR location");
        location_base = source_info->fixed_location;
        break;
      default:
        IREE_ASSERT_UNREACHABLE("verified AMDGPU HAL ABI source role");
        IREE_BUILTIN_UNREACHABLE();
    }
    loom_amdgpu_hal_kernel_abi_append_fixed_value(
        result->fixed_values, &result->fixed_value_count, value_id,
        location_base, source_info->unit_count);
  }
  result->flat_scratch_sgprs.wavefront_offset = workgroup_id_sgpr;
}

static bool loom_amdgpu_hal_kernel_abi_can_emit(
    uint32_t max_errors,
    const loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  return max_errors == 0 || result->error_count < max_errors;
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count,
    const loom_diagnostic_related_op_t* related_ops,
    iree_host_size_t related_op_count, uint32_t max_errors,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  if (!loom_amdgpu_hal_kernel_abi_can_emit(max_errors, result)) {
    return iree_ok_status();
  }
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
      .related_ops = related_ops,
      .related_op_count = related_op_count,
  };
  IREE_RETURN_IF_ERROR(iree_diagnostic_emit(emitter, &emission));
  ++result->error_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_resource_count_overflow(
    const loom_op_t* function_op, iree_host_size_t resource_count,
    uint32_t max_errors, iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_u64(resource_count),
      loom_param_u64(LOOM_AMDGPU_HAL_KERNEL_ABI_MAX_RESOURCE_COUNT),
  };
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, function_op, LOOM_ERR_AMDGPU_006, params, IREE_ARRAYSIZE(params),
      NULL, 0, max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_resource_import_kind_error(
    const loom_op_t* resource_op, uint8_t actual_import_kind,
    uint32_t max_errors, iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_u32(LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BINDING),
      loom_param_u32(actual_import_kind),
  };
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, resource_op, LOOM_ERR_AMDGPU_007, params, IREE_ARRAYSIZE(params),
      NULL, 0, max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_resource_result_type_error(
    const loom_module_t* module, const loom_op_t* resource_op,
    uint32_t max_errors, iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_type(loom_module_value_type(
          module, loom_low_resource_result(resource_op))),
      loom_param_u32(LOOM_AMDGPU_REG_CLASS_ID_SGPR),
      loom_param_u32(2),
  };
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, resource_op, LOOM_ERR_AMDGPU_008, params, IREE_ARRAYSIZE(params),
      NULL, 0, max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_binding_index_negative(
    const loom_op_t* resource_op, int64_t binding_index, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_i64(binding_index),
  };
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, resource_op, LOOM_ERR_AMDGPU_009, params, IREE_ARRAYSIZE(params),
      NULL, 0, max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_binding_index_sparse(
    const loom_op_t* resource_op, uint64_t binding_index,
    iree_host_size_t resource_count, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_u64(binding_index),
      loom_param_u64(resource_count),
  };
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, resource_op, LOOM_ERR_AMDGPU_010, params, IREE_ARRAYSIZE(params),
      NULL, 0, max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_binding_index_duplicate(
    const loom_op_t* resource_op, const loom_op_t* previous_op,
    uint64_t binding_index, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_u64(binding_index),
  };
  const loom_diagnostic_related_op_t related[] = {{
      .label = IREE_SV("previous binding"),
      .op = previous_op,
      .field_ref = loom_low_resource_index_diagnostic_ref(),
  }};
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, resource_op, LOOM_ERR_AMDGPU_011, params, IREE_ARRAYSIZE(params),
      related, IREE_ARRAYSIZE(related), max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_missing_binding(
    const loom_op_t* function_op, uint64_t binding_index, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_u64(binding_index),
  };
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, function_op, LOOM_ERR_AMDGPU_012, params, IREE_ARRAYSIZE(params),
      NULL, 0, max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_descriptor_attr_error(
    const loom_op_t* op, iree_string_view_t attr_name,
    loom_attr_kind_t actual_kind, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(attr_name),
      loom_param_u32(LOOM_ATTR_I64),
      loom_param_u32((uint32_t)actual_kind),
  };
  return loom_amdgpu_hal_kernel_abi_emit(emitter, op, LOOM_ERR_AMDGPU_013,
                                         params, IREE_ARRAYSIZE(params), NULL,
                                         0, max_errors, result);
}

static iree_status_t
loom_amdgpu_hal_kernel_abi_emit_descriptor_cache_swizzle_error(
    const loom_op_t* op, iree_string_view_t descriptor_set_key,
    uint64_t cache_swizzle_stride, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(descriptor_set_key),
      loom_param_u64(cache_swizzle_stride),
  };
  return loom_amdgpu_hal_kernel_abi_emit(emitter, op, LOOM_ERR_AMDGPU_014,
                                         params, IREE_ARRAYSIZE(params), NULL,
                                         0, max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_live_in_duplicate(
    const loom_op_t* live_in_op, const loom_op_t* previous_op,
    iree_string_view_t source_name, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(source_name),
  };
  const loom_diagnostic_related_op_t related[] = {{
      .label = IREE_SV("previous live-in"),
      .op = previous_op,
      .field_ref = loom_low_live_in_source_diagnostic_ref(),
  }};
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, live_in_op, LOOM_ERR_AMDGPU_015, params, IREE_ARRAYSIZE(params),
      related, IREE_ARRAYSIZE(related), max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_fixed_live_in_overlap(
    const loom_op_t* live_in_op, const loom_op_t* conflicting_op,
    iree_string_view_t source_name, iree_string_view_t conflicting_source_name,
    uint16_t fixed_location, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(source_name),
      loom_param_string(conflicting_source_name),
      loom_param_u32(fixed_location),
  };
  const loom_diagnostic_related_op_t related[] = {{
      .label = IREE_SV("conflicting live-in"),
      .op = conflicting_op,
      .field_ref = loom_low_live_in_source_diagnostic_ref(),
  }};
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, live_in_op, LOOM_ERR_AMDGPU_045, params, IREE_ARRAYSIZE(params),
      related, IREE_ARRAYSIZE(related), max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_workitem_live_in_mix(
    const loom_op_t* live_in_op, const loom_op_t* conflicting_op,
    iree_string_view_t source_name, iree_string_view_t conflicting_source_name,
    uint32_t max_errors, iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(source_name),
      loom_param_string(conflicting_source_name),
  };
  const loom_diagnostic_related_op_t related[] = {{
      .label = IREE_SV("conflicting live-in"),
      .op = conflicting_op,
      .field_ref = loom_low_live_in_source_diagnostic_ref(),
  }};
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, live_in_op, LOOM_ERR_AMDGPU_016, params, IREE_ARRAYSIZE(params),
      related, IREE_ARRAYSIZE(related), max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_live_in_type_error(
    const loom_module_t* module, const loom_op_t* live_in_op,
    iree_string_view_t source_name, uint16_t expected_reg_class_id,
    uint32_t expected_unit_count, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(source_name),
      loom_param_type(
          loom_module_value_type(module, loom_low_live_in_result(live_in_op))),
      loom_param_u32(expected_reg_class_id),
      loom_param_u32(expected_unit_count),
  };
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, live_in_op, LOOM_ERR_AMDGPU_017, params, IREE_ARRAYSIZE(params),
      NULL, 0, max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_m0_type_error(
    const loom_module_t* module, const loom_op_t* live_in_op,
    iree_string_view_t source_name, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(source_name),
      loom_param_type(
          loom_module_value_type(module, loom_low_live_in_result(live_in_op))),
  };
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, live_in_op, LOOM_ERR_AMDGPU_018, params, IREE_ARRAYSIZE(params),
      NULL, 0, max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_direct_arg_type_error(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_value_id_t arg_id, uint16_t argument_index,
    uint16_t expected_reg_class_id, uint32_t expected_unit_count,
    uint32_t max_errors, iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_u32(argument_index),
      loom_param_type(loom_module_value_type(module, arg_id)),
      loom_param_u32(expected_reg_class_id),
      loom_param_u32(expected_unit_count),
  };
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, function_op, LOOM_ERR_AMDGPU_025, params, IREE_ARRAYSIZE(params),
      NULL, 0, max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_verify_direct_arguments(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_low_register_type_resolver_t* register_type_resolver,
    uint32_t max_errors, iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_region_t* body = loom_low_function_const_body(function_op);
  if (body == NULL || body->block_count == 0) {
    return iree_ok_status();
  }

  const loom_block_t* entry_block = loom_region_const_entry_block(body);
  const uint16_t expected_reg_class_id =
      loom_amdgpu_hal_kernel_abi_reg_class_id(
          LOOM_AMDGPU_HAL_KERNEL_ABI_REG_CLASS_SGPR);
  for (uint16_t i = 0; i < entry_block->arg_count; ++i) {
    const loom_value_id_t arg_id = loom_block_arg_id(entry_block, i);
    const bool type_matches =
        loom_amdgpu_hal_kernel_abi_direct_arg_type_matches(
            module, register_type_resolver, arg_id);
    if (!type_matches) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_hal_kernel_abi_emit_direct_arg_type_error(
              module, function_op, arg_id, i, expected_reg_class_id,
              LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_ARG_MAX_UNIT_COUNT, max_errors,
              emitter, result));
    } else if (result->layout.direct_args != NULL) {
      loom_amdgpu_hal_kernarg_direct_arg_t* argument = &(
          (loom_amdgpu_hal_kernarg_direct_arg_t*)result->layout.direct_args)[i];
      const loom_string_id_t name_id =
          loom_module_value(module, arg_id)->name_id;
      argument->arg_id = arg_id;
      argument->name = name_id != LOOM_STRING_ID_INVALID
                           ? loom_string_table_get(&module->strings, name_id)
                           : iree_string_view_empty();
      argument->abi_type = loom_module_value_type(module, arg_id);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_verify_resource_type(
    const loom_module_t* module,
    const loom_low_register_type_resolver_t* register_type_resolver,
    const loom_op_t* resource_op, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const bool type_matches = loom_amdgpu_hal_kernel_abi_register_value_matches(
      module, register_type_resolver, loom_low_resource_result(resource_op),
      LOOM_AMDGPU_HAL_KERNEL_ABI_REG_CLASS_SGPR, 2);
  if (!type_matches) {
    return loom_amdgpu_hal_kernel_abi_emit_resource_result_type_error(
        module, resource_op, max_errors, emitter, result);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_verify_resources(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_low_register_type_resolver_t* register_type_resolver,
    uint32_t max_errors, iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result,
    iree_arena_allocator_t* arena) {
  const loom_region_t* body = loom_low_function_const_body(function_op);
  const bool declared_resource_count_present =
      loom_amdgpu_hal_kernel_abi_has_layout_attr(function_op);
  iree_host_size_t resource_count = result->layout.resource_count;
  if (body != NULL && !declared_resource_count_present) {
    for (uint16_t block_index = 0; block_index < body->block_count;
         ++block_index) {
      const loom_block_t* block = loom_region_const_block(body, block_index);
      const loom_op_t* resource_op = NULL;
      loom_block_for_each_op(block, resource_op) {
        if (loom_low_resource_isa(resource_op)) {
          ++resource_count;
        }
      }
    }
  }
  if (resource_count == 0 && !declared_resource_count_present) {
    return iree_ok_status();
  }
  if (resource_count > LOOM_AMDGPU_HAL_KERNEL_ABI_MAX_RESOURCE_COUNT) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_hal_kernel_abi_emit_resource_count_overflow(
            function_op, resource_count, max_errors, emitter, result));
    return iree_ok_status();
  }

  const loom_op_t** resource_slots = NULL;
  if (resource_count != 0 && !declared_resource_count_present) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, resource_count,
                                                   sizeof(*resource_slots),
                                                   (void**)&resource_slots));
    memset(resource_slots, 0, resource_count * sizeof(*resource_slots));
  }

  if (body != NULL) {
    for (uint16_t block_index = 0; block_index < body->block_count;
         ++block_index) {
      const loom_block_t* block = loom_region_const_block(body, block_index);
      const loom_op_t* resource_op = NULL;
      loom_block_for_each_op(block, resource_op) {
        if (!loom_low_resource_isa(resource_op)) {
          continue;
        }

        if (loom_low_resource_import_kind(resource_op) !=
            LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BINDING) {
          IREE_RETURN_IF_ERROR(
              loom_amdgpu_hal_kernel_abi_emit_resource_import_kind_error(
                  resource_op, loom_low_resource_import_kind(resource_op),
                  max_errors, emitter, result));
        }
        IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_verify_resource_type(
            module, register_type_resolver, resource_op, max_errors, emitter,
            result));

        const int64_t binding_index = loom_low_resource_index(resource_op);
        if (binding_index < 0) {
          IREE_RETURN_IF_ERROR(
              loom_amdgpu_hal_kernel_abi_emit_binding_index_negative(
                  resource_op, binding_index, max_errors, emitter, result));
          continue;
        }
        if ((uint64_t)binding_index >= resource_count) {
          IREE_RETURN_IF_ERROR(
              loom_amdgpu_hal_kernel_abi_emit_binding_index_sparse(
                  resource_op, (uint64_t)binding_index, resource_count,
                  max_errors, emitter, result));
          continue;
        }

        const iree_host_size_t slot_index = (iree_host_size_t)binding_index;
        loom_amdgpu_hal_kernarg_resource_t* declared_resource =
            declared_resource_count_present
                ? &((loom_amdgpu_hal_kernarg_resource_t*)
                        result->layout.resources)[slot_index]
                : NULL;
        const loom_op_t* previous_op = declared_resource != NULL
                                           ? declared_resource->resource_op
                                           : resource_slots[slot_index];
        if (previous_op != NULL) {
          IREE_RETURN_IF_ERROR(
              loom_amdgpu_hal_kernel_abi_emit_binding_index_duplicate(
                  resource_op, previous_op, (uint64_t)binding_index, max_errors,
                  emitter, result));
          continue;
        }
        if (declared_resource != NULL) {
          declared_resource->resource_op = resource_op;
          const loom_type_id_t source_type =
              loom_low_resource_source_type(resource_op);
          declared_resource->source_type =
              source_type != LOOM_TYPE_ID_INVALID
                  ? loom_type_table_get(&module->types, source_type)
                  : loom_type_none();
          declared_resource->abi_type = loom_module_value_type(
              module, loom_low_resource_result(resource_op));
        } else {
          resource_slots[slot_index] = resource_op;
        }
      }
    }
  }

  if (declared_resource_count_present) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < resource_count; ++i) {
    if (resource_slots[i] != NULL) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_emit_missing_binding(
        function_op, i, max_errors, emitter, result));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_verify_low_ops(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_low_descriptor_set_t* descriptor_set, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  loom_amdgpu_buffer_resource_cache_swizzle_t cache_swizzle_kind =
      LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE;
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
      loom_amdgpu_target_info_descriptor_set_at(
          descriptor_set->descriptor_set_ordinal);
  IREE_ASSERT(descriptor_set_info != NULL);
  cache_swizzle_kind = descriptor_set_info->buffer_resource.cache_swizzle;
  const bool supports_cache_swizzle =
      cache_swizzle_kind ==
      LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT;
  const loom_low_descriptor_t* static_buffer_descriptor =
      loom_amdgpu_descriptor_ref_descriptor(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_HAL_BUFFER_DESCRIPTOR);
  const loom_low_descriptor_t* dynamic_buffer_descriptor =
      loom_amdgpu_descriptor_ref_descriptor(
          descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_HAL_BUFFER_DESCRIPTOR_EXTENT);
  const loom_low_descriptor_t* cluster_workgroup_flat_id_descriptor =
      loom_amdgpu_descriptor_ref_descriptor(
          descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_S_GETREG_B32_CLUSTER_WORKGROUP_FLAT_ID);

  const loom_region_t* body = loom_low_function_const_body(function_op);
  if (body == NULL) {
    return iree_ok_status();
  }
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(body, block_index);
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (!loom_low_op_isa(op)) {
        continue;
      }
      const loom_low_descriptor_t* descriptor =
          loom_amdgpu_hal_kernel_abi_low_op_descriptor(descriptor_set, op);
      if (descriptor == cluster_workgroup_flat_id_descriptor) {
        result->launch_workgroup_id_flags |=
            LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_KNOWN_FLAGS;
        continue;
      }
      if (descriptor != static_buffer_descriptor &&
          descriptor != dynamic_buffer_descriptor) {
        continue;
      }

      const loom_named_attr_slice_t attrs = loom_low_op_attrs(op);
      loom_attribute_t cache_swizzle_stride_attr = loom_attr_absent();
      if (attrs.count == loom_low_descriptor_layout(descriptor_set, descriptor)
                             ->immediate_count) {
        cache_swizzle_stride_attr =
            descriptor == dynamic_buffer_descriptor
                ? loom_amdgpu_hal_buffer_descriptor_extent_cache_swizzle_stride(
                      attrs)
                : loom_amdgpu_hal_buffer_descriptor_cache_swizzle_stride(attrs);
      }
      if (cache_swizzle_stride_attr.kind != LOOM_ATTR_I64) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_hal_kernel_abi_emit_descriptor_attr_error(
                op, IREE_SV("cache_swizzle_stride"),
                (loom_attr_kind_t)cache_swizzle_stride_attr.kind, max_errors,
                emitter, result));
        continue;
      }

      const int64_t cache_swizzle_stride = cache_swizzle_stride_attr.i64;
      if (cache_swizzle_stride == 0 || supports_cache_swizzle) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_hal_kernel_abi_emit_descriptor_cache_swizzle_error(
              op, descriptor_set_info->key, (uint64_t)cache_swizzle_stride,
              max_errors, emitter, result));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_verify_live_in_type(
    const loom_module_t* module,
    const loom_low_register_type_resolver_t* register_type_resolver,
    const loom_op_t* live_in_op,
    loom_amdgpu_hal_kernel_abi_reg_class_t reg_class, uint32_t unit_count,
    iree_string_view_t source_name, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const uint16_t expected_reg_class_id =
      loom_amdgpu_hal_kernel_abi_reg_class_id(reg_class);
  const bool type_matches = loom_amdgpu_hal_kernel_abi_register_value_matches(
      module, register_type_resolver, loom_low_live_in_result(live_in_op),
      reg_class, unit_count);
  if (!type_matches) {
    return loom_amdgpu_hal_kernel_abi_emit_live_in_type_error(
        module, live_in_op, source_name, expected_reg_class_id, unit_count,
        max_errors, emitter, result);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_verify_m0_live_in_type(
    const loom_module_t* module,
    const loom_low_register_type_resolver_t* register_type_resolver,
    const loom_op_t* live_in_op, iree_string_view_t source_name,
    uint32_t max_errors, iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const bool type_matches =
      loom_amdgpu_hal_kernel_abi_single_physical_register_matches(
          module, register_type_resolver, loom_low_live_in_result(live_in_op));
  if (!type_matches) {
    return loom_amdgpu_hal_kernel_abi_emit_m0_type_error(
        module, live_in_op, source_name, max_errors, emitter, result);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_verify_source_live_in_type(
    const loom_module_t* module,
    const loom_low_register_type_resolver_t* register_type_resolver,
    const loom_op_t* live_in_op,
    const loom_amdgpu_hal_kernel_abi_source_info_t* source_info,
    uint32_t max_errors, iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  if (source_info->role == LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_ROLE_M0) {
    return loom_amdgpu_hal_kernel_abi_verify_m0_live_in_type(
        module, register_type_resolver, live_in_op, source_info->name,
        max_errors, emitter, result);
  }
  IREE_ASSERT_NE(source_info->reg_class,
                 LOOM_AMDGPU_HAL_KERNEL_ABI_REG_CLASS_NONE);
  return loom_amdgpu_hal_kernel_abi_verify_live_in_type(
      module, register_type_resolver, live_in_op, source_info->reg_class,
      source_info->unit_count, source_info->name, max_errors, emitter, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_verify_live_ins(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_low_register_type_resolver_t* register_type_resolver,
    const loom_amdgpu_processor_properties_t* properties, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_region_t* body = loom_low_function_const_body(function_op);
  if (body == NULL || body->block_count == 0) {
    return iree_ok_status();
  }

  const loom_op_t*
      live_in_ops[IREE_ARRAYSIZE(kLoomAmdgpuHalKernelAbiSourceInfos)] = {0};
  const loom_op_t*
      workitem_id_ops[LOOM_AMDGPU_HAL_KERNEL_ABI_COORDINATE_DIMENSION_COUNT] = {
          NULL,
          NULL,
          NULL,
      };
  const loom_op_t* packed_workitem_op = NULL;
  iree_string_view_t packed_workitem_source_name = iree_string_view_empty();
  loom_value_id_t
      live_in_values[IREE_ARRAYSIZE(kLoomAmdgpuHalKernelAbiSourceInfos)];
  loom_amdgpu_hal_kernel_abi_initialize_live_in_values(
      live_in_values, IREE_ARRAYSIZE(live_in_values));

  const loom_block_t* entry_block = loom_region_const_entry_block(body);
  const loom_op_t* live_in_op = NULL;
  loom_block_for_each_op(entry_block, live_in_op) {
    if (!loom_low_live_in_isa(live_in_op)) {
      continue;
    }
    const loom_amdgpu_hal_kernel_abi_source_kind_t source_kind =
        loom_amdgpu_hal_kernel_abi_source_kind_from_stable_id(
            (uint64_t)loom_low_live_in_source_id(live_in_op));
    const loom_amdgpu_hal_kernel_abi_source_info_t* source_info =
        loom_amdgpu_hal_kernel_abi_source_info_from_kind(source_kind);
    if (source_info == NULL) {
      continue;
    }

    const iree_host_size_t source_index = (iree_host_size_t)source_kind;
    if (source_info->role ==
        LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_ROLE_FIXED_SGPR) {
      for (iree_host_size_t i = 1;
           i < IREE_ARRAYSIZE(kLoomAmdgpuHalKernelAbiSourceInfos); ++i) {
        const loom_amdgpu_hal_kernel_abi_source_info_t* conflicting_info =
            &kLoomAmdgpuHalKernelAbiSourceInfos[i];
        if (i == source_index || live_in_ops[i] == NULL ||
            conflicting_info->role !=
                LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_ROLE_FIXED_SGPR ||
            conflicting_info->fixed_location != source_info->fixed_location) {
          continue;
        }
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_hal_kernel_abi_emit_fixed_live_in_overlap(
                live_in_op, live_in_ops[i], source_info->name,
                conflicting_info->name, source_info->fixed_location, max_errors,
                emitter, result));
        break;
      }
    }
    if (source_info->role !=
            LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_ROLE_PACKED_WORKITEM_ID &&
        live_in_ops[source_index] != NULL) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_emit_live_in_duplicate(
          live_in_op, live_in_ops[source_index], source_info->name, max_errors,
          emitter, result));
    }
    live_in_ops[source_index] = live_in_op;
    live_in_values[source_index] = loom_low_live_in_result(live_in_op);
    if (source_kind == LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_KERNARG_SEGMENT_PTR) {
      result->kernarg_segment_ptr = live_in_values[source_index];
    }
    result->live_in_source_bits |= UINT64_C(1) << source_index;
    result->launch_workgroup_id_flags |= source_info->launch_workgroup_id_flags;

    switch (source_info->role) {
      case LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_ROLE_WORKITEM_ID: {
        if (packed_workitem_op != NULL) {
          IREE_RETURN_IF_ERROR(
              loom_amdgpu_hal_kernel_abi_emit_workitem_live_in_mix(
                  live_in_op, packed_workitem_op, source_info->name,
                  packed_workitem_source_name, max_errors, emitter, result));
        }
        IREE_ASSERT_LT(source_info->dimension,
                       LOOM_AMDGPU_HAL_KERNEL_ABI_COORDINATE_DIMENSION_COUNT);
        workitem_id_ops[source_info->dimension] = live_in_op;
        break;
      }
      case LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_ROLE_PACKED_WORKITEM_ID: {
        if (packed_workitem_op != NULL) {
          IREE_RETURN_IF_ERROR(
              loom_amdgpu_hal_kernel_abi_emit_live_in_duplicate(
                  live_in_op, packed_workitem_op, source_info->name, max_errors,
                  emitter, result));
        }
        for (uint32_t i = 0;
             i < LOOM_AMDGPU_HAL_KERNEL_ABI_COORDINATE_DIMENSION_COUNT; ++i) {
          if (workitem_id_ops[i] == NULL) {
            continue;
          }
          const loom_amdgpu_hal_kernel_abi_source_kind_t unpacked_source_kind =
              (loom_amdgpu_hal_kernel_abi_source_kind_t)(LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_X +
                                                         i);
          const loom_amdgpu_hal_kernel_abi_source_info_t* unpacked_source_info =
              loom_amdgpu_hal_kernel_abi_source_info_from_kind(
                  unpacked_source_kind);
          IREE_ASSERT(unpacked_source_info != NULL,
                      "verified AMDGPU HAL ABI workitem source info");
          IREE_RETURN_IF_ERROR(
              loom_amdgpu_hal_kernel_abi_emit_workitem_live_in_mix(
                  live_in_op, workitem_id_ops[i], source_info->name,
                  unpacked_source_info->name, max_errors, emitter, result));
        }
        packed_workitem_op = live_in_op;
        packed_workitem_source_name = source_info->name;
        break;
      }
      default:
        break;
    }

    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_verify_source_live_in_type(
        module, register_type_resolver, live_in_op, source_info, max_errors,
        emitter, result));
  }
  if (result->error_count == 0) {
    loom_amdgpu_hal_kernel_abi_build_fixed_values(live_in_values, properties,
                                                  result);
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_hal_kernel_abi_verify_low(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_processor_properties_t* properties, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    loom_amdgpu_hal_kernel_abi_verify_result_t* out_result) {
  *out_result = (loom_amdgpu_hal_kernel_abi_verify_result_t){
      .kernarg_segment_ptr = LOOM_VALUE_ID_INVALID,
  };
  if (loom_amdgpu_hal_kernel_abi_has_layout_attr(function_op)) {
    bool accepted = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_layout_from_attr(
        module, function_op, emitter, arena, &accepted, &out_result->layout));
    if (!accepted) {
      out_result->error_count = 1;
      return iree_ok_status();
    }
  }

  loom_low_register_type_resolver_t register_type_resolver =
      loom_low_register_type_resolver_for_descriptor_set(descriptor_set);
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_verify_direct_arguments(
      module, function_op, &register_type_resolver, max_errors, emitter,
      out_result));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_verify_resources(
      module, function_op, &register_type_resolver, max_errors, emitter,
      out_result, arena));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_verify_low_ops(
      module, function_op, descriptor_set, max_errors, emitter, out_result));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_verify_live_ins(
      module, function_op, &register_type_resolver, properties, max_errors,
      emitter, out_result));
  if (out_result->error_count != 0 ||
      loom_amdgpu_hal_kernel_abi_has_layout_attr(function_op)) {
    return iree_ok_status();
  }
  return loom_amdgpu_hal_kernel_abi_layout_build(module, function_op, arena,
                                                 &out_result->layout);
}
