// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/hal/kernel_abi.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/function.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/hal/binding_descriptor.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/registers.h"

#define LOOM_AMDGPU_HAL_KERNEL_ABI_COORDINATE_DIMENSION_COUNT 3u
// ABI layout snapshots store resource arrays as uint16_t-counted attributes.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_MAX_RESOURCE_COUNT UINT16_MAX
#define LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_ARG_MAX_UNIT_COUNT 2u
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

static iree_string_view_t loom_amdgpu_hal_kernel_abi_module_string(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static const loom_low_descriptor_t*
loom_amdgpu_hal_kernel_abi_low_op_descriptor(
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* op) {
  return &descriptor_set->descriptors[loom_low_op_descriptor(op)];
}

static loom_type_t loom_amdgpu_hal_kernel_abi_type_attr(
    const loom_module_t* module, loom_type_id_t type_id) {
  if (type_id == LOOM_TYPE_ID_INVALID || type_id >= module->types.count) {
    return loom_type_none();
  }
  return module->types.entries[type_id];
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

static iree_status_t loom_amdgpu_hal_kernel_abi_reg_class_id(
    loom_amdgpu_hal_kernel_abi_reg_class_t reg_class,
    uint16_t* out_descriptor_reg_class_id) {
  switch (reg_class) {
    case LOOM_AMDGPU_HAL_KERNEL_ABI_REG_CLASS_SGPR:
      *out_descriptor_reg_class_id = LOOM_AMDGPU_REG_CLASS_ID_SGPR;
      return iree_ok_status();
    case LOOM_AMDGPU_HAL_KERNEL_ABI_REG_CLASS_VGPR:
      *out_descriptor_reg_class_id = LOOM_AMDGPU_REG_CLASS_ID_VGPR;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown AMDGPU HAL kernel ABI register class %u",
                              (unsigned)reg_class);
  }
}

static iree_status_t loom_amdgpu_hal_kernel_abi_descriptor_reg_class(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_hal_kernel_abi_reg_class_t reg_class,
    uint16_t* out_descriptor_reg_class_id,
    const loom_low_reg_class_t** out_descriptor_reg_class) {
  *out_descriptor_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  if (out_descriptor_reg_class) {
    *out_descriptor_reg_class = NULL;
  }
  uint16_t descriptor_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_reg_class_id(
      reg_class, &descriptor_reg_class_id));
  if (descriptor_reg_class_id >= descriptor_set->reg_class_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU HAL kernel ABI descriptor register-class ID %" PRIu16
        " is out of range",
        descriptor_reg_class_id);
  }
  *out_descriptor_reg_class_id = descriptor_reg_class_id;
  if (out_descriptor_reg_class) {
    *out_descriptor_reg_class =
        &descriptor_set->reg_classes[descriptor_reg_class_id];
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_register_value_matches(
    const loom_module_t* module,
    const loom_low_register_type_resolver_t* register_type_resolver,
    loom_value_id_t value_id, loom_amdgpu_hal_kernel_abi_reg_class_t reg_class,
    uint32_t unit_count, bool* out_matches) {
  *out_matches = false;
  uint16_t expected_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_descriptor_reg_class(
      register_type_resolver->descriptor_set, reg_class, &expected_class_id,
      NULL));
  loom_type_t type = loom_module_value_type(module, value_id);
  uint16_t actual_class_id = LOOM_LOW_REG_CLASS_NONE;
  bool found_actual_class = loom_low_register_type_resolver_try_resolve(
      register_type_resolver, type, &actual_class_id, NULL);
  *out_matches = found_actual_class && actual_class_id == expected_class_id &&
                 loom_low_register_type_unit_count(type) == unit_count;
  return iree_ok_status();
}

static bool loom_amdgpu_hal_kernel_abi_direct_arg_unit_count_supported(
    uint32_t unit_count) {
  return unit_count >= 1 &&
         unit_count <= LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_ARG_MAX_UNIT_COUNT;
}

static iree_status_t loom_amdgpu_hal_kernel_abi_direct_arg_type_matches(
    const loom_module_t* module,
    const loom_low_register_type_resolver_t* register_type_resolver,
    loom_value_id_t value_id, bool* out_matches) {
  *out_matches = false;
  uint16_t expected_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_descriptor_reg_class(
      register_type_resolver->descriptor_set,
      LOOM_AMDGPU_HAL_KERNEL_ABI_REG_CLASS_SGPR, &expected_class_id, NULL));
  loom_type_t type = loom_module_value_type(module, value_id);
  uint16_t actual_class_id = LOOM_LOW_REG_CLASS_NONE;
  bool found_actual_class = loom_low_register_type_resolver_try_resolve(
      register_type_resolver, type, &actual_class_id, NULL);
  const uint32_t unit_count = loom_low_register_type_unit_count(type);
  *out_matches =
      found_actual_class && actual_class_id == expected_class_id &&
      loom_amdgpu_hal_kernel_abi_direct_arg_unit_count_supported(unit_count);
  return iree_ok_status();
}

static uint32_t loom_amdgpu_hal_kernel_abi_direct_arg_size_from_type(
    loom_type_t type) {
  const uint32_t unit_count = loom_low_register_type_unit_count(type);
  IREE_ASSERT(
      loom_amdgpu_hal_kernel_abi_direct_arg_unit_count_supported(unit_count),
      "verified AMDGPU HAL ABI direct arguments have supported widths");
  return unit_count * LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_SCALAR_KERNARG_SIZE;
}

static iree_status_t
loom_amdgpu_hal_kernel_abi_single_physical_register_matches(
    const loom_module_t* module,
    const loom_low_register_type_resolver_t* register_type_resolver,
    loom_value_id_t value_id, bool* out_matches) {
  *out_matches = false;
  loom_type_t type = loom_module_value_type(module, value_id);
  uint16_t actual_class_id = LOOM_LOW_REG_CLASS_NONE;
  const loom_low_reg_class_t* actual_class = NULL;
  bool found_actual_class = loom_low_register_type_resolver_try_resolve(
      register_type_resolver, type, &actual_class_id, &actual_class);
  *out_matches =
      found_actual_class && actual_class != NULL &&
      loom_low_register_type_unit_count(type) == 1 &&
      actual_class->allocatable_count == 1 &&
      iree_any_bit_set(actual_class->flags, LOOM_LOW_REG_CLASS_FLAG_PHYSICAL) &&
      iree_any_bit_set(actual_class->flags,
                       LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE);
  return iree_ok_status();
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
      .field_ref = loom_diagnostic_field_ref(
          LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, loom_low_resource_index_ATTR_INDEX),
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
      .field_ref = loom_diagnostic_field_ref(
          LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, loom_low_live_in_source_ATTR_INDEX),
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
      .field_ref = loom_diagnostic_field_ref(
          LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, loom_low_live_in_source_ATTR_INDEX),
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
      .field_ref = loom_diagnostic_field_ref(
          LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, loom_low_live_in_source_ATTR_INDEX),
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

static iree_status_t loom_amdgpu_hal_kernel_abi_format_resource_name(
    uint32_t binding_index, iree_string_view_t* out_name,
    iree_arena_allocator_t* arena) {
  char scratch[32];
  int length =
      iree_snprintf(scratch, sizeof(scratch), "binding%" PRIu32, binding_index);
  IREE_ASSERT(length >= 0 && (iree_host_size_t)length < sizeof(scratch));
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, (iree_host_size_t)length, (void**)&storage));
  memcpy(storage, scratch, (iree_host_size_t)length);
  *out_name = iree_make_string_view(storage, (iree_host_size_t)length);
  return iree_ok_status();
}

static iree_string_view_t loom_amdgpu_hal_kernel_abi_value_name(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return iree_string_view_empty();
  }
  const loom_string_id_t name_id = loom_module_value(module, value_id)->name_id;
  return loom_amdgpu_hal_kernel_abi_module_string(module, name_id);
}

static const loom_attribute_t* loom_amdgpu_hal_kernel_abi_find_layout_attr(
    loom_named_attr_slice_t attrs, loom_string_id_t key_id) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    if (attrs.entries[i].name_id == key_id) {
      return &attrs.entries[i].value;
    }
  }
  return NULL;
}

static iree_status_t loom_amdgpu_hal_kernel_abi_direct_arg_from_block_arg(
    const loom_module_t* module, const loom_block_t* entry_block,
    uint16_t argument_index, uint32_t parameter_index, uint32_t kernarg_offset,
    loom_amdgpu_hal_kernarg_direct_arg_t* out_direct_arg) {
  const loom_value_id_t arg_id = loom_block_arg_id(entry_block, argument_index);
  const loom_type_t abi_type = loom_module_value_type(module, arg_id);
  const uint32_t kernarg_size =
      loom_amdgpu_hal_kernel_abi_direct_arg_size_from_type(abi_type);
  *out_direct_arg = (loom_amdgpu_hal_kernarg_direct_arg_t){
      .arg_id = arg_id,
      .name = loom_amdgpu_hal_kernel_abi_value_name(module, arg_id),
      .parameter_index = parameter_index,
      .argument_index = argument_index,
      .kernarg_offset = kernarg_offset,
      .kernarg_size = kernarg_size,
      .kernarg_alignment =
          LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_SCALAR_KERNARG_ALIGNMENT,
      .abi_type = abi_type,
  };
  return iree_ok_status();
}

typedef struct loom_amdgpu_hal_kernel_abi_layout_attr_keys_t {
  loom_string_id_t constant_count;
  loom_string_id_t direct_arg_count;
  loom_string_id_t direct_arg_names;
  loom_string_id_t direct_arg_offsets;
  loom_string_id_t direct_arg_parameter_indices;
  loom_string_id_t direct_arg_sizes;
  loom_string_id_t parameter_count;
  loom_string_id_t resource_count;
  loom_string_id_t resource_offsets;
  loom_string_id_t resource_parameter_indices;
  loom_string_id_t uses_kernarg_segment_ptr;
} loom_amdgpu_hal_kernel_abi_layout_attr_keys_t;

static iree_status_t loom_amdgpu_hal_kernel_abi_intern_layout_attr_keys(
    loom_module_t* module,
    loom_amdgpu_hal_kernel_abi_layout_attr_keys_t* out_keys) {
  *out_keys = (loom_amdgpu_hal_kernel_abi_layout_attr_keys_t){0};
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("constant_count"), &out_keys->constant_count));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("direct_arg_count"), &out_keys->direct_arg_count));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("direct_arg_names"), &out_keys->direct_arg_names));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("direct_arg_offsets"), &out_keys->direct_arg_offsets));
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, IREE_SV("direct_arg_parameter_indices"),
                                &out_keys->direct_arg_parameter_indices));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("direct_arg_sizes"), &out_keys->direct_arg_sizes));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("parameter_count"), &out_keys->parameter_count));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("resource_count"), &out_keys->resource_count));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("resource_offsets"), &out_keys->resource_offsets));
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, IREE_SV("resource_parameter_indices"),
                                &out_keys->resource_parameter_indices));
  return loom_module_intern_string(module, IREE_SV("uses_kernarg_segment_ptr"),
                                   &out_keys->uses_kernarg_segment_ptr);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_format_layout_entry_key(
    loom_module_t* module, iree_string_view_t prefix, iree_host_size_t ordinal,
    loom_string_id_t* out_key_id) {
  char scratch[64];
  int length = iree_snprintf(scratch, sizeof(scratch), "%.*s%" PRIhsz,
                             (int)prefix.size, prefix.data, ordinal);
  IREE_ASSERT(length >= 0 && (iree_host_size_t)length < sizeof(scratch));
  return loom_module_intern_string(
      module, iree_make_string_view(scratch, (iree_host_size_t)length),
      out_key_id);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_make_string_attr(
    loom_module_t* module, iree_string_view_t value,
    loom_attribute_t* out_attr) {
  loom_string_id_t string_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, value, &string_id));
  *out_attr = loom_attr_string(string_id);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_make_direct_arg_names_attr(
    loom_module_t* module, const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    iree_arena_allocator_t* scratch_arena, loom_attribute_t* out_attr) {
  loom_named_attr_t* entries = NULL;
  if (layout->direct_arg_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(scratch_arena, layout->direct_arg_count,
                                  sizeof(*entries), (void**)&entries));
    memset(entries, 0, layout->direct_arg_count * sizeof(*entries));
  }
  for (iree_host_size_t i = 0; i < layout->direct_arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_format_layout_entry_key(
        module, IREE_SV("arg"), i, &entries[i].name_id));
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_make_string_attr(
        module, layout->direct_args[i].name, &entries[i].value));
  }
  return loom_module_make_canonical_attr_dict(
      module, loom_make_named_attr_slice(entries, layout->direct_arg_count),
      out_attr);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_make_direct_arg_sizes_attr(
    const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    iree_arena_allocator_t* scratch_arena, loom_attribute_t* out_attr) {
  IREE_ASSERT_LE(layout->direct_arg_count, (iree_host_size_t)UINT16_MAX);
  int64_t* entries = NULL;
  if (layout->direct_arg_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(scratch_arena, layout->direct_arg_count,
                                  sizeof(*entries), (void**)&entries));
  }
  for (iree_host_size_t i = 0; i < layout->direct_arg_count; ++i) {
    entries[i] = layout->direct_args[i].kernarg_size;
  }
  *out_attr = loom_attr_i64_array(entries, (uint16_t)layout->direct_arg_count);
  return iree_ok_status();
}

typedef enum loom_amdgpu_hal_kernel_abi_layout_array_kind_e {
  LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_RESOURCE_PARAMETER_INDICES = 0,
  LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_RESOURCE_OFFSETS = 1,
  LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_DIRECT_ARG_PARAMETER_INDICES = 2,
  LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_DIRECT_ARG_OFFSETS = 3,
} loom_amdgpu_hal_kernel_abi_layout_array_kind_t;

static iree_host_size_t loom_amdgpu_hal_kernel_abi_layout_array_count(
    const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    loom_amdgpu_hal_kernel_abi_layout_array_kind_t kind) {
  switch (kind) {
    case LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_RESOURCE_PARAMETER_INDICES:
    case LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_RESOURCE_OFFSETS:
      return layout->resource_count;
    case LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_DIRECT_ARG_PARAMETER_INDICES:
    case LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_DIRECT_ARG_OFFSETS:
      return layout->direct_arg_count;
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU HAL ABI layout array kind");
  IREE_BUILTIN_UNREACHABLE();
}

static uint32_t loom_amdgpu_hal_kernel_abi_layout_array_value(
    const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    loom_amdgpu_hal_kernel_abi_layout_array_kind_t kind,
    iree_host_size_t index) {
  switch (kind) {
    case LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_RESOURCE_PARAMETER_INDICES:
      return layout->resources[index].parameter_index;
    case LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_RESOURCE_OFFSETS:
      return layout->resources[index].kernarg_offset;
    case LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_DIRECT_ARG_PARAMETER_INDICES:
      return layout->direct_args[index].parameter_index;
    case LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_DIRECT_ARG_OFFSETS:
      return layout->direct_args[index].kernarg_offset;
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU HAL ABI layout array kind");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_make_layout_u32_array_attr(
    const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    loom_amdgpu_hal_kernel_abi_layout_array_kind_t kind,
    iree_arena_allocator_t* scratch_arena, loom_attribute_t* out_attr) {
  const iree_host_size_t count =
      loom_amdgpu_hal_kernel_abi_layout_array_count(layout, kind);
  IREE_ASSERT_LE(count, (iree_host_size_t)UINT16_MAX);
  int64_t* entries = NULL;
  if (count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, count, sizeof(*entries), (void**)&entries));
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    entries[i] = loom_amdgpu_hal_kernel_abi_layout_array_value(layout, kind, i);
  }
  *out_attr = loom_attr_i64_array(entries, (uint16_t)count);
  return iree_ok_status();
}

bool loom_amdgpu_hal_kernel_abi_has_layout_attr(const loom_op_t* function_op) {
  return loom_low_kernel_def_isa(function_op) &&
         !loom_attr_is_absent(loom_op_attrs(
             function_op)[loom_low_kernel_def_abi_layout_ATTR_INDEX]);
}

iree_status_t loom_amdgpu_hal_kernel_abi_make_layout_attr(
    loom_module_t* module, const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    iree_arena_allocator_t* scratch_arena, loom_attribute_t* out_attr) {
  if (module == NULL || layout == NULL || scratch_arena == NULL ||
      out_attr == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL ABI layout attr construction requires a module, layout, "
        "scratch arena, and output");
  }

  loom_amdgpu_hal_kernel_abi_layout_attr_keys_t keys = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_abi_intern_layout_attr_keys(module, &keys));

  loom_attribute_t direct_arg_names_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_make_direct_arg_names_attr(
      module, layout, scratch_arena, &direct_arg_names_attr));
  loom_attribute_t direct_arg_offsets_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_make_layout_u32_array_attr(
      layout, LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_DIRECT_ARG_OFFSETS,
      scratch_arena, &direct_arg_offsets_attr));
  loom_attribute_t direct_arg_parameter_indices_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_make_layout_u32_array_attr(
      layout,
      LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_DIRECT_ARG_PARAMETER_INDICES,
      scratch_arena, &direct_arg_parameter_indices_attr));
  loom_attribute_t direct_arg_sizes_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_make_direct_arg_sizes_attr(
      layout, scratch_arena, &direct_arg_sizes_attr));
  loom_attribute_t resource_offsets_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_make_layout_u32_array_attr(
      layout, LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_RESOURCE_OFFSETS,
      scratch_arena, &resource_offsets_attr));
  loom_attribute_t resource_parameter_indices_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_make_layout_u32_array_attr(
      layout,
      LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_RESOURCE_PARAMETER_INDICES,
      scratch_arena, &resource_parameter_indices_attr));

  loom_named_attr_t entries[] = {
      {.name_id = keys.constant_count,
       .value = loom_attr_i64(layout->constant_count)},
      {.name_id = keys.direct_arg_count,
       .value = loom_attr_i64(layout->direct_arg_count)},
      {.name_id = keys.direct_arg_names, .value = direct_arg_names_attr},
      {.name_id = keys.direct_arg_offsets, .value = direct_arg_offsets_attr},
      {.name_id = keys.direct_arg_parameter_indices,
       .value = direct_arg_parameter_indices_attr},
      {.name_id = keys.direct_arg_sizes, .value = direct_arg_sizes_attr},
      {.name_id = keys.parameter_count,
       .value = loom_attr_i64(layout->parameter_count)},
      {.name_id = keys.resource_count,
       .value = loom_attr_i64(layout->resource_count)},
      {.name_id = keys.resource_offsets, .value = resource_offsets_attr},
      {.name_id = keys.resource_parameter_indices,
       .value = resource_parameter_indices_attr},
      {.name_id = keys.uses_kernarg_segment_ptr,
       .value = loom_attr_bool(layout->uses_kernarg_segment_ptr)},
  };
  return loom_module_make_canonical_attr_dict(
      module, loom_make_named_attr_slice(entries, IREE_ARRAYSIZE(entries)),
      out_attr);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_resource_from_import(
    const loom_module_t* module, const loom_op_t* resource_op,
    loom_amdgpu_hal_kernarg_resource_t* out_resource,
    iree_arena_allocator_t* arena) {
  *out_resource = (loom_amdgpu_hal_kernarg_resource_t){0};

  iree_string_view_t resource_name = iree_string_view_empty();
  int64_t binding_index = loom_low_resource_index(resource_op);
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_format_resource_name(
      (uint32_t)binding_index, &resource_name, arena));

  uint64_t kernarg_offset =
      (uint64_t)binding_index *
      LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_SIZE;

  *out_resource = (loom_amdgpu_hal_kernarg_resource_t){
      .resource_op = resource_op,
      .name = resource_name,
      .binding_index = (uint32_t)binding_index,
      .parameter_index = (uint32_t)binding_index,
      .kernarg_offset = (uint32_t)kernarg_offset,
      .kernarg_size = LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_SIZE,
      .kernarg_alignment =
          LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT,
      .source_type = loom_amdgpu_hal_kernel_abi_type_attr(
          module, loom_low_resource_source_type(resource_op)),
      .abi_type =
          loom_module_value_type(module, loom_low_resource_result(resource_op)),
  };
  return iree_ok_status();
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
  uint16_t expected_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_reg_class_id(
      LOOM_AMDGPU_HAL_KERNEL_ABI_REG_CLASS_SGPR, &expected_reg_class_id));
  for (uint16_t i = 0; i < entry_block->arg_count; ++i) {
    const loom_value_id_t arg_id = loom_block_arg_id(entry_block, i);
    bool type_matches = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_direct_arg_type_matches(
        module, register_type_resolver, arg_id, &type_matches));
    if (!type_matches) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_hal_kernel_abi_emit_direct_arg_type_error(
              module, function_op, arg_id, i, expected_reg_class_id,
              LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_ARG_MAX_UNIT_COUNT, max_errors,
              emitter, result));
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
  bool type_matches = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_register_value_matches(
      module, register_type_resolver, loom_low_resource_result(resource_op),
      LOOM_AMDGPU_HAL_KERNEL_ABI_REG_CLASS_SGPR, 2, &type_matches));
  if (!type_matches) {
    return loom_amdgpu_hal_kernel_abi_emit_resource_result_type_error(
        module, resource_op, max_errors, emitter, result);
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_hal_kernel_abi_declared_resource_count_from_attr(
    const loom_module_t* module, const loom_op_t* function_op,
    bool* out_present, iree_host_size_t* out_resource_count) {
  *out_present = false;
  *out_resource_count = 0;
  if (!loom_amdgpu_hal_kernel_abi_has_layout_attr(function_op)) {
    return iree_ok_status();
  }

  const loom_string_id_t resource_count_key =
      loom_module_lookup_string(module, IREE_SV("resource_count"));
  if (resource_count_key == LOOM_STRING_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL ABI layout attr is missing resource_count");
  }
  const loom_attribute_t* attr = loom_amdgpu_hal_kernel_abi_find_layout_attr(
      loom_low_kernel_def_abi_layout(function_op), resource_count_key);
  if (attr == NULL || attr->kind != LOOM_ATTR_I64) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL ABI layout attr has a malformed resource_count");
  }
  const int64_t resource_count = loom_attr_as_i64(*attr);
  if (resource_count < 0 || (uint64_t)resource_count >
                                LOOM_AMDGPU_HAL_KERNEL_ABI_MAX_RESOURCE_COUNT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL ABI layout attr resource_count exceeds ABI capacity");
  }
  *out_present = true;
  *out_resource_count = (iree_host_size_t)resource_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_verify_resources(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_low_register_type_resolver_t* register_type_resolver,
    uint32_t max_errors, iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result,
    iree_arena_allocator_t* arena) {
  const loom_region_t* body = loom_low_function_const_body(function_op);
  iree_host_size_t resource_count = 0;
  if (body != NULL) {
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
  bool declared_resource_count_present = false;
  iree_host_size_t declared_resource_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_abi_declared_resource_count_from_attr(
          module, function_op, &declared_resource_count_present,
          &declared_resource_count));
  if (declared_resource_count_present) {
    resource_count = declared_resource_count;
  }
  if (resource_count == 0) {
    return iree_ok_status();
  }
  if (resource_count > LOOM_AMDGPU_HAL_KERNEL_ABI_MAX_RESOURCE_COUNT) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_hal_kernel_abi_emit_resource_count_overflow(
            function_op, resource_count, max_errors, emitter, result));
    return iree_ok_status();
  }

  const loom_op_t** resource_slots = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, resource_count, sizeof(*resource_slots), (void**)&resource_slots));
  memset(resource_slots, 0, resource_count * sizeof(*resource_slots));

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
        if (resource_slots[slot_index] != NULL) {
          IREE_RETURN_IF_ERROR(
              loom_amdgpu_hal_kernel_abi_emit_binding_index_duplicate(
                  resource_op, resource_slots[slot_index],
                  (uint64_t)binding_index, max_errors, emitter, result));
          continue;
        }
        resource_slots[slot_index] = resource_op;
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

      loom_named_attr_slice_t attrs = loom_low_op_attrs(op);
      if (attrs.count <=
              LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_ATTR_CACHE_SWIZZLE_STRIDE ||
          attrs.entries
                  [LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_ATTR_CACHE_SWIZZLE_STRIDE]
                      .value.kind != LOOM_ATTR_I64) {
        loom_attr_kind_t actual_kind = LOOM_ATTR_ABSENT;
        if (attrs.count >
            LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_ATTR_CACHE_SWIZZLE_STRIDE) {
          actual_kind =
              (loom_attr_kind_t)attrs
                  .entries
                      [LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_ATTR_CACHE_SWIZZLE_STRIDE]
                  .value.kind;
        }
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_hal_kernel_abi_emit_descriptor_attr_error(
                op, IREE_SV("cache_swizzle_stride"), actual_kind, max_errors,
                emitter, result));
        continue;
      }

      const int64_t cache_swizzle_stride =
          attrs
              .entries
                  [LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_ATTR_CACHE_SWIZZLE_STRIDE]
              .value.i64;
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
  uint16_t expected_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_reg_class_id(
      reg_class, &expected_reg_class_id));
  bool type_matches = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_register_value_matches(
      module, register_type_resolver, loom_low_live_in_result(live_in_op),
      reg_class, unit_count, &type_matches));
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
  bool type_matches = false;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_abi_single_physical_register_matches(
          module, register_type_resolver, loom_low_live_in_result(live_in_op),
          &type_matches));
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
    uint32_t max_errors, iree_diagnostic_emitter_t emitter,
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
    loom_amdgpu_hal_kernel_abi_build_fixed_values(live_in_values, result);
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_hal_kernel_abi_verify_low(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_low_descriptor_set_t* descriptor_set, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* out_result,
    iree_arena_allocator_t* arena) {
  *out_result = (loom_amdgpu_hal_kernel_abi_verify_result_t){0};
  if (module == NULL || function_op == NULL || descriptor_set == NULL ||
      arena == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL kernel ABI verification requires a module, function, "
        "descriptor set, and arena");
  }
  if (!loom_low_function_def_isa(function_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL kernel ABI verification requires low.func.def or "
        "low.kernel.def");
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
  return loom_amdgpu_hal_kernel_abi_verify_live_ins(
      module, function_op, &register_type_resolver, max_errors, emitter,
      out_result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_attach_direct_args_from_low(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_amdgpu_hal_kernel_abi_layout_t* layout) {
  if (layout->direct_arg_count == 0) {
    return iree_ok_status();
  }
  const loom_region_t* body = loom_low_function_const_body(function_op);
  const loom_block_t* entry_block = body != NULL && body->block_count != 0
                                        ? loom_region_const_entry_block(body)
                                        : NULL;
  if (entry_block == NULL || entry_block->arg_count == 0) {
    return iree_ok_status();
  }
  if (entry_block->arg_count != layout->direct_arg_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL kernel ABI layout direct argument count does not match the "
        "low entry block");
  }
  loom_amdgpu_hal_kernarg_direct_arg_t* direct_args =
      (loom_amdgpu_hal_kernarg_direct_arg_t*)layout->direct_args;
  for (uint16_t i = 0; i < entry_block->arg_count; ++i) {
    loom_amdgpu_hal_kernarg_direct_arg_t* direct_arg = &direct_args[i];
    const loom_value_id_t arg_id = loom_block_arg_id(entry_block, i);
    direct_arg->arg_id = arg_id;
    direct_arg->name = loom_amdgpu_hal_kernel_abi_value_name(module, arg_id);
    direct_arg->argument_index = i;
    direct_arg->abi_type = loom_module_value_type(module, arg_id);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_attach_resource_from_low(
    const loom_module_t* module, const loom_op_t* resource_op,
    loom_amdgpu_hal_kernel_abi_layout_t* layout) {
  const int64_t binding_index = loom_low_resource_index(resource_op);
  if (binding_index < 0 || (uint64_t)binding_index >= layout->resource_count) {
    IREE_ASSERT_UNREACHABLE("verified AMDGPU HAL ABI resource binding index");
    IREE_BUILTIN_UNREACHABLE();
  }
  loom_amdgpu_hal_kernarg_resource_t* resources =
      (loom_amdgpu_hal_kernarg_resource_t*)layout->resources;
  loom_amdgpu_hal_kernarg_resource_t* resource =
      &resources[(iree_host_size_t)binding_index];
  if (resource->resource_op != NULL) {
    IREE_ASSERT_UNREACHABLE("verified AMDGPU HAL ABI resource binding index");
    IREE_BUILTIN_UNREACHABLE();
  }
  resource->resource_op = resource_op;
  resource->source_type = loom_amdgpu_hal_kernel_abi_type_attr(
      module, loom_low_resource_source_type(resource_op));
  resource->abi_type =
      loom_module_value_type(module, loom_low_resource_result(resource_op));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_attach_resources_from_low(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_amdgpu_hal_kernel_abi_layout_t* layout) {
  if (layout->resource_count == 0) {
    return iree_ok_status();
  }
  const loom_region_t* body = loom_low_function_const_body(function_op);
  if (body == NULL) {
    return iree_ok_status();
  }
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(body, block_index);
    const loom_op_t* resource_op = NULL;
    loom_block_for_each_op(block, resource_op) {
      if (!loom_low_resource_isa(resource_op)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_attach_resource_from_low(
          module, resource_op, layout));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_attach_low_artifacts_to_layout(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_amdgpu_hal_kernel_abi_layout_t* layout) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_attach_direct_args_from_low(
      module, function_op, layout));
  return loom_amdgpu_hal_kernel_abi_attach_resources_from_low(
      module, function_op, layout);
}

iree_status_t loom_amdgpu_hal_kernel_abi_layout_from_low(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_amdgpu_hal_kernel_abi_layout_t* out_layout,
    iree_arena_allocator_t* arena) {
  *out_layout = (loom_amdgpu_hal_kernel_abi_layout_t){0};
  if (module == NULL || function_op == NULL || arena == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL kernel ABI layout requires a module, function, and arena");
  }
  if (!loom_low_function_def_isa(function_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL kernel ABI layout requires low.func.def or "
        "low.kernel.def");
  }
  if (loom_amdgpu_hal_kernel_abi_has_layout_attr(function_op)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_layout_from_attr(
        module, function_op, out_layout, arena));
    return loom_amdgpu_hal_kernel_abi_attach_low_artifacts_to_layout(
        module, function_op, out_layout);
  }

  const loom_region_t* body = loom_low_function_const_body(function_op);
  const loom_block_t* entry_block = body != NULL && body->block_count != 0
                                        ? loom_region_const_entry_block(body)
                                        : NULL;
  const iree_host_size_t direct_arg_count =
      entry_block != NULL ? entry_block->arg_count : 0;
  bool direct_arg_used = false;
  if (entry_block != NULL) {
    for (uint16_t i = 0; i < entry_block->arg_count; ++i) {
      const loom_value_id_t arg_id = loom_block_arg_id(entry_block, i);
      const loom_value_t* arg_value = loom_module_value(module, arg_id);
      if (arg_value->use_count != 0 ||
          loom_module_value_has_type_uses(module, arg_id)) {
        direct_arg_used = true;
        break;
      }
    }
  }
  iree_host_size_t resource_count = 0;
  bool resource_used = false;
  if (body != NULL) {
    for (uint16_t block_index = 0; block_index < body->block_count;
         ++block_index) {
      const loom_block_t* block = loom_region_const_block(body, block_index);
      const loom_op_t* resource_op = NULL;
      loom_block_for_each_op(block, resource_op) {
        if (!loom_low_resource_isa(resource_op)) {
          continue;
        }
        ++resource_count;
        const loom_value_id_t result = loom_low_resource_result(resource_op);
        const loom_value_t* result_value = loom_module_value(module, result);
        resource_used |= result_value->use_count != 0 ||
                         loom_module_value_has_type_uses(module, result);
      }
    }
  }
  if (resource_count > LOOM_AMDGPU_HAL_KERNEL_ABI_MAX_RESOURCE_COUNT) {
    IREE_ASSERT_UNREACHABLE("verified AMDGPU HAL ABI resource count");
    IREE_BUILTIN_UNREACHABLE();
  }
  const uint64_t kernarg_resource_bytes =
      (uint64_t)resource_count *
      LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_SIZE;
  if (direct_arg_count > UINT32_MAX) {
    IREE_ASSERT_UNREACHABLE("verified AMDGPU HAL ABI direct argument count");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_amdgpu_hal_kernarg_resource_t* resources = NULL;
  if (resource_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, resource_count, sizeof(*resources), (void**)&resources));
    memset(resources, 0, resource_count * sizeof(*resources));
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

        const int64_t binding_index = loom_low_resource_index(resource_op);
        if (binding_index < 0 || (uint64_t)binding_index >= resource_count) {
          IREE_ASSERT_UNREACHABLE(
              "verified AMDGPU HAL ABI resource binding index");
          IREE_BUILTIN_UNREACHABLE();
        }
        const iree_host_size_t slot_index = (iree_host_size_t)binding_index;
        if (resources[slot_index].resource_op != NULL) {
          IREE_ASSERT_UNREACHABLE(
              "verified AMDGPU HAL ABI resource binding index");
          IREE_BUILTIN_UNREACHABLE();
        }

        loom_amdgpu_hal_kernarg_resource_t resource = {0};
        IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_resource_from_import(
            module, resource_op, &resource, arena));
        resources[slot_index] = resource;
      }
    }
  }

  for (iree_host_size_t i = 0; i < resource_count; ++i) {
    if (resources[i].resource_op == NULL) {
      IREE_ASSERT_UNREACHABLE("verified AMDGPU HAL ABI resource binding set");
      IREE_BUILTIN_UNREACHABLE();
    }
  }

  loom_amdgpu_hal_kernarg_direct_arg_t* direct_args = NULL;
  uint64_t kernarg_direct_args_end = kernarg_resource_bytes;
  uint64_t constant_count = 0;
  if (direct_arg_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, direct_arg_count, sizeof(*direct_args), (void**)&direct_args));
    memset(direct_args, 0, direct_arg_count * sizeof(*direct_args));
    for (uint16_t i = 0; i < entry_block->arg_count; ++i) {
      if (kernarg_direct_args_end > UINT32_MAX) {
        IREE_ASSERT_UNREACHABLE(
            "verified AMDGPU HAL ABI direct argument extent");
        IREE_BUILTIN_UNREACHABLE();
      }
      const uint32_t parameter_index = (uint32_t)(resource_count + i);
      IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_direct_arg_from_block_arg(
          module, entry_block, i, parameter_index,
          (uint32_t)kernarg_direct_args_end, &direct_args[i]));
      kernarg_direct_args_end += direct_args[i].kernarg_size;
      constant_count += direct_args[i].kernarg_size /
                        LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_SCALAR_KERNARG_SIZE;
    }
  }
  const uint64_t kernarg_segment_size = iree_align_uint64(
      kernarg_direct_args_end,
      LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT);
  if (kernarg_segment_size > UINT32_MAX || constant_count > UINT32_MAX) {
    IREE_ASSERT_UNREACHABLE("verified AMDGPU HAL ABI direct argument extent");
    IREE_BUILTIN_UNREACHABLE();
  }

  *out_layout = (loom_amdgpu_hal_kernel_abi_layout_t){
      .function_op = function_op,
      .parameter_count = (uint32_t)(resource_count + direct_arg_count),
      .kernarg_segment_size = (uint32_t)kernarg_segment_size,
      .kernarg_segment_alignment =
          LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT,
      .uses_kernarg_segment_ptr = resource_used || direct_arg_used,
      .constant_count = (uint32_t)constant_count,
      .resources = resources,
      .resource_count = resource_count,
      .direct_args = direct_args,
      .direct_arg_count = direct_arg_count,
  };

  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_lookup_layout_attr_keys(
    const loom_module_t* module,
    loom_amdgpu_hal_kernel_abi_layout_attr_keys_t* out_keys) {
  *out_keys = (loom_amdgpu_hal_kernel_abi_layout_attr_keys_t){
      .constant_count =
          loom_module_lookup_string(module, IREE_SV("constant_count")),
      .direct_arg_count =
          loom_module_lookup_string(module, IREE_SV("direct_arg_count")),
      .direct_arg_names =
          loom_module_lookup_string(module, IREE_SV("direct_arg_names")),
      .direct_arg_offsets =
          loom_module_lookup_string(module, IREE_SV("direct_arg_offsets")),
      .direct_arg_parameter_indices = loom_module_lookup_string(
          module, IREE_SV("direct_arg_parameter_indices")),
      .direct_arg_sizes =
          loom_module_lookup_string(module, IREE_SV("direct_arg_sizes")),
      .parameter_count =
          loom_module_lookup_string(module, IREE_SV("parameter_count")),
      .resource_count =
          loom_module_lookup_string(module, IREE_SV("resource_count")),
      .resource_offsets =
          loom_module_lookup_string(module, IREE_SV("resource_offsets")),
      .resource_parameter_indices = loom_module_lookup_string(
          module, IREE_SV("resource_parameter_indices")),
      .uses_kernarg_segment_ptr = loom_module_lookup_string(
          module, IREE_SV("uses_kernarg_segment_ptr")),
  };
  if (out_keys->constant_count == LOOM_STRING_ID_INVALID ||
      out_keys->direct_arg_count == LOOM_STRING_ID_INVALID ||
      out_keys->direct_arg_names == LOOM_STRING_ID_INVALID ||
      out_keys->direct_arg_offsets == LOOM_STRING_ID_INVALID ||
      out_keys->direct_arg_parameter_indices == LOOM_STRING_ID_INVALID ||
      out_keys->direct_arg_sizes == LOOM_STRING_ID_INVALID ||
      out_keys->parameter_count == LOOM_STRING_ID_INVALID ||
      out_keys->resource_count == LOOM_STRING_ID_INVALID ||
      out_keys->resource_offsets == LOOM_STRING_ID_INVALID ||
      out_keys->resource_parameter_indices == LOOM_STRING_ID_INVALID ||
      out_keys->uses_kernarg_segment_ptr == LOOM_STRING_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL ABI layout attr is missing required keys");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_require_layout_attr(
    loom_named_attr_slice_t attrs, loom_string_id_t key_id,
    loom_attr_kind_t expected_kind, const loom_attribute_t** out_attr) {
  const loom_attribute_t* attr =
      loom_amdgpu_hal_kernel_abi_find_layout_attr(attrs, key_id);
  if (attr == NULL || attr->kind != expected_kind) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL ABI layout attr has a missing or malformed field");
  }
  *out_attr = attr;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_decode_u32_layout_attr(
    loom_named_attr_slice_t attrs, loom_string_id_t key_id,
    uint32_t* out_value) {
  const loom_attribute_t* attr = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_require_layout_attr(
      attrs, key_id, LOOM_ATTR_I64, &attr));
  const int64_t value = loom_attr_as_i64(*attr);
  if (value < 0 || value > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL ABI layout attr integer field is outside u32 range");
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_decode_u32_array_element(
    const loom_attribute_t* attr, uint32_t expected_count,
    uint32_t element_index, iree_string_view_t field_name,
    uint32_t* out_value) {
  if (attr->kind != LOOM_ATTR_I64_ARRAY || attr->count != expected_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL ABI layout attr field '%.*s' has an inconsistent count",
        (int)field_name.size, field_name.data);
  }
  if (attr->count != 0 && attr->i64_array == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL ABI layout attr field '%.*s' has malformed storage",
        (int)field_name.size, field_name.data);
  }
  const int64_t value = attr->i64_array[element_index];
  if (value < 0 || value > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL ABI layout attr field '%.*s' has an element outside u32 "
        "range",
        (int)field_name.size, field_name.data);
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_mark_layout_parameter_index(
    bool* parameter_slots, uint32_t parameter_count, uint32_t parameter_index) {
  if (parameter_index >= parameter_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL ABI layout attr parameter index exceeds parameter_count");
  }
  if (parameter_slots[parameter_index]) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL ABI layout attr parameter index is duplicated");
  }
  parameter_slots[parameter_index] = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_decode_direct_arg_size(
    const loom_attribute_t* direct_arg_sizes_attr, uint32_t direct_arg_count,
    uint32_t argument_index, uint32_t* out_size) {
  if (direct_arg_sizes_attr->kind != LOOM_ATTR_I64_ARRAY ||
      direct_arg_sizes_attr->count != direct_arg_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL ABI layout attr direct argument sizes are inconsistent");
  }
  if (direct_arg_sizes_attr->count != 0 &&
      direct_arg_sizes_attr->i64_array == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL ABI layout attr direct argument sizes are malformed");
  }
  const int64_t size = direct_arg_sizes_attr->i64_array[argument_index];
  if (size <= 0 ||
      (size % LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_SCALAR_KERNARG_SIZE) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL ABI layout attr direct argument size is not a whole "
        "number of constant words");
  }
  const int64_t max_size =
      (int64_t)LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_ARG_MAX_UNIT_COUNT *
      LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_SCALAR_KERNARG_SIZE;
  if (size > max_size) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL ABI layout attr direct argument size exceeds supported "
        "direct argument width");
  }
  *out_size = (uint32_t)size;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_lookup_direct_arg_name_key(
    const loom_module_t* module, iree_host_size_t ordinal,
    loom_string_id_t* out_key_id) {
  char scratch[64];
  int length = iree_snprintf(scratch, sizeof(scratch), "arg%" PRIhsz, ordinal);
  IREE_ASSERT(length >= 0 && (iree_host_size_t)length < sizeof(scratch));
  *out_key_id = loom_module_lookup_string(
      module, iree_make_string_view(scratch, (iree_host_size_t)length));
  if (*out_key_id == LOOM_STRING_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL ABI layout attr is missing an indexed entry");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_decode_resource_from_layout(
    uint32_t binding_index, uint32_t parameter_index, uint32_t kernarg_offset,
    iree_arena_allocator_t* arena,
    loom_amdgpu_hal_kernarg_resource_t* out_resource) {
  iree_string_view_t name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_format_resource_name(
      binding_index, &name, arena));
  if ((kernarg_offset %
       LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL ABI layout attr resource offset is not aligned");
  }
  *out_resource = (loom_amdgpu_hal_kernarg_resource_t){
      .resource_op = NULL,
      .name = name,
      .binding_index = binding_index,
      .parameter_index = parameter_index,
      .kernarg_offset = kernarg_offset,
      .kernarg_size = LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_SIZE,
      .kernarg_alignment =
          LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT,
      .source_type = loom_type_none(),
      .abi_type = loom_type_none(),
  };
  return iree_ok_status();
}

iree_status_t loom_amdgpu_hal_kernel_abi_layout_from_attr(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_amdgpu_hal_kernel_abi_layout_t* out_layout,
    iree_arena_allocator_t* arena) {
  *out_layout = (loom_amdgpu_hal_kernel_abi_layout_t){0};
  if (module == NULL || function_op == NULL || arena == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL ABI layout attr load requires a module, function, and "
        "arena");
  }
  if (!loom_low_kernel_def_isa(function_op) ||
      !loom_amdgpu_hal_kernel_abi_has_layout_attr(function_op)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU HAL ABI layout attr load requires low.kernel.def with "
        "abi_layout");
  }

  const loom_named_attr_slice_t top_attrs =
      loom_low_kernel_def_abi_layout(function_op);
  loom_amdgpu_hal_kernel_abi_layout_attr_keys_t keys = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_abi_lookup_layout_attr_keys(module, &keys));

  const loom_attribute_t* uses_kernarg_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_require_layout_attr(
      top_attrs, keys.uses_kernarg_segment_ptr, LOOM_ATTR_BOOL,
      &uses_kernarg_attr));
  const loom_attribute_t* direct_arg_names_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_require_layout_attr(
      top_attrs, keys.direct_arg_names, LOOM_ATTR_DICT,
      &direct_arg_names_attr));
  const loom_attribute_t* direct_arg_offsets_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_require_layout_attr(
      top_attrs, keys.direct_arg_offsets, LOOM_ATTR_I64_ARRAY,
      &direct_arg_offsets_attr));
  const loom_attribute_t* direct_arg_parameter_indices_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_require_layout_attr(
      top_attrs, keys.direct_arg_parameter_indices, LOOM_ATTR_I64_ARRAY,
      &direct_arg_parameter_indices_attr));
  const loom_attribute_t* direct_arg_sizes_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_require_layout_attr(
      top_attrs, keys.direct_arg_sizes, LOOM_ATTR_I64_ARRAY,
      &direct_arg_sizes_attr));
  const loom_attribute_t* resource_offsets_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_require_layout_attr(
      top_attrs, keys.resource_offsets, LOOM_ATTR_I64_ARRAY,
      &resource_offsets_attr));
  const loom_attribute_t* resource_parameter_indices_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_require_layout_attr(
      top_attrs, keys.resource_parameter_indices, LOOM_ATTR_I64_ARRAY,
      &resource_parameter_indices_attr));

  uint32_t parameter_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_decode_u32_layout_attr(
      top_attrs, keys.parameter_count, &parameter_count));
  uint32_t resource_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_decode_u32_layout_attr(
      top_attrs, keys.resource_count, &resource_count));
  uint32_t direct_arg_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_decode_u32_layout_attr(
      top_attrs, keys.direct_arg_count, &direct_arg_count));
  if (direct_arg_count > UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL ABI layout attr direct argument count exceeds u16 range");
  }
  uint32_t constant_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_decode_u32_layout_attr(
      top_attrs, keys.constant_count, &constant_count));
  if (resource_count > LOOM_AMDGPU_HAL_KERNEL_ABI_MAX_RESOURCE_COUNT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL ABI layout attr resource count exceeds ABI capacity");
  }
  if (parameter_count != resource_count + direct_arg_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL ABI layout attr parameter_count is inconsistent");
  }

  bool* parameter_slots = NULL;
  if (parameter_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, parameter_count,
                                                   sizeof(*parameter_slots),
                                                   (void**)&parameter_slots));
    memset(parameter_slots, 0, parameter_count * sizeof(*parameter_slots));
  }

  loom_amdgpu_hal_kernarg_resource_t* resources = NULL;
  if (resource_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, resource_count, sizeof(*resources), (void**)&resources));
    memset(resources, 0, resource_count * sizeof(*resources));
  }
  for (uint32_t i = 0; i < resource_count; ++i) {
    uint32_t parameter_index = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_decode_u32_array_element(
        resource_parameter_indices_attr, resource_count, i,
        IREE_SV("resource_parameter_indices"), &parameter_index));
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_mark_layout_parameter_index(
        parameter_slots, parameter_count, parameter_index));
    uint32_t kernarg_offset = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_decode_u32_array_element(
        resource_offsets_attr, resource_count, i, IREE_SV("resource_offsets"),
        &kernarg_offset));
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_decode_resource_from_layout(
        i, parameter_index, kernarg_offset, arena, &resources[i]));
  }

  const loom_named_attr_slice_t direct_arg_names =
      loom_attr_as_dict(*direct_arg_names_attr);
  if (direct_arg_names.count != direct_arg_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HAL ABI layout attr direct argument name "
                            "count is inconsistent");
  }
  loom_amdgpu_hal_kernarg_direct_arg_t* direct_args = NULL;
  if (direct_arg_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, direct_arg_count, sizeof(*direct_args), (void**)&direct_args));
    memset(direct_args, 0, direct_arg_count * sizeof(*direct_args));
  }
  uint64_t kernarg_segment_end = 0;
  for (uint32_t i = 0; i < resource_count; ++i) {
    const uint64_t resource_end =
        (uint64_t)resources[i].kernarg_offset + resources[i].kernarg_size;
    kernarg_segment_end = iree_max(kernarg_segment_end, resource_end);
  }
  uint64_t decoded_constant_count = 0;
  for (uint32_t i = 0; i < direct_arg_count; ++i) {
    loom_string_id_t key_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_lookup_direct_arg_name_key(
        module, i, &key_id));
    const loom_attribute_t* name_attr =
        loom_amdgpu_hal_kernel_abi_find_layout_attr(direct_arg_names, key_id);
    if (name_attr == NULL || name_attr->kind != LOOM_ATTR_STRING) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU HAL ABI layout attr has a malformed direct argument name");
    }
    uint32_t kernarg_size = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_decode_direct_arg_size(
        direct_arg_sizes_attr, direct_arg_count, i, &kernarg_size));
    uint32_t parameter_index = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_decode_u32_array_element(
        direct_arg_parameter_indices_attr, direct_arg_count, i,
        IREE_SV("direct_arg_parameter_indices"), &parameter_index));
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_mark_layout_parameter_index(
        parameter_slots, parameter_count, parameter_index));
    uint32_t kernarg_offset = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_decode_u32_array_element(
        direct_arg_offsets_attr, direct_arg_count, i,
        IREE_SV("direct_arg_offsets"), &kernarg_offset));
    if ((kernarg_offset %
         LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_SCALAR_KERNARG_ALIGNMENT) != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU HAL ABI layout attr direct argument offset is not aligned");
    }
    const uint64_t direct_arg_end = (uint64_t)kernarg_offset + kernarg_size;
    if (direct_arg_end > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU HAL ABI layout attr kernarg segment exceeds u32 range");
    }
    direct_args[i] = (loom_amdgpu_hal_kernarg_direct_arg_t){
        .arg_id = LOOM_VALUE_ID_INVALID,
        .name = loom_amdgpu_hal_kernel_abi_module_string(
            module, loom_attr_as_string_id(*name_attr)),
        .parameter_index = parameter_index,
        .argument_index = (uint16_t)i,
        .kernarg_offset = kernarg_offset,
        .kernarg_size = kernarg_size,
        .kernarg_alignment =
            LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_SCALAR_KERNARG_ALIGNMENT,
        .abi_type = loom_type_none(),
    };
    kernarg_segment_end = iree_max(kernarg_segment_end, direct_arg_end);
    decoded_constant_count +=
        kernarg_size / LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_SCALAR_KERNARG_SIZE;
  }
  if (decoded_constant_count != constant_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL ABI layout attr constant count is inconsistent");
  }
  const uint64_t kernarg_segment_size = iree_align_uint64(
      kernarg_segment_end,
      LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT);
  if (kernarg_segment_size > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL ABI layout attr kernarg segment exceeds u32 range");
  }

  *out_layout = (loom_amdgpu_hal_kernel_abi_layout_t){
      .function_op = function_op,
      .parameter_count = parameter_count,
      .kernarg_segment_size = (uint32_t)kernarg_segment_size,
      .kernarg_segment_alignment =
          LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT,
      .uses_kernarg_segment_ptr = loom_attr_as_bool(*uses_kernarg_attr),
      .constant_count = constant_count,
      .resources = resources,
      .resource_count = resource_count,
      .direct_args = direct_args,
      .direct_arg_count = direct_arg_count,
  };
  return iree_ok_status();
}
