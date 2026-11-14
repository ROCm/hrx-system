// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "profile_rows.h"

#include "diagnostic.h"
#include "loom/target/arch/spirv/error_catalog.h"
#include "loomc/iree.h"
#include "result.h"

static loomc_status_t loomc_spirv_rows_validate_string_view(
    loomc_string_view_t value) {
  if (value.data == NULL && value.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  return loomc_ok_status();
}

loom_spirv_feature_atom_t loomc_spirv_feature_atom_from_public(
    loomc_spirv_feature_t feature) {
  switch (feature) {
    case LOOMC_SPIRV_FEATURE_VULKAN_SHADER:
      return LOOM_SPIRV_FEATURE_ATOM_VULKAN_SHADER;
    case LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER:
      return LOOM_SPIRV_FEATURE_ATOM_PHYSICAL_STORAGE_BUFFER;
    case LOOMC_SPIRV_FEATURE_FLOAT16:
      return LOOM_SPIRV_FEATURE_ATOM_FLOAT16;
    case LOOMC_SPIRV_FEATURE_FLOAT64:
      return LOOM_SPIRV_FEATURE_ATOM_FLOAT64;
    case LOOMC_SPIRV_FEATURE_INT8:
      return LOOM_SPIRV_FEATURE_ATOM_INT8;
    case LOOMC_SPIRV_FEATURE_INT16:
      return LOOM_SPIRV_FEATURE_ATOM_INT16;
    case LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS:
      return LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_8BIT_ACCESS;
    case LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_16BIT_ACCESS:
      return LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_16BIT_ACCESS;
    case LOOMC_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV:
      return LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_VECTOR_NV;
    case LOOMC_SPIRV_FEATURE_COOPERATIVE_VECTOR_TRAINING_NV:
      return LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_VECTOR_TRAINING_NV;
    case LOOMC_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR:
      return LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_MATRIX_KHR;
    case LOOMC_SPIRV_FEATURE_BFLOAT16_TYPE_KHR:
      return LOOM_SPIRV_FEATURE_ATOM_BFLOAT16_TYPE_KHR;
    case LOOMC_SPIRV_FEATURE_BFLOAT16_DOT_PRODUCT_KHR:
      return LOOM_SPIRV_FEATURE_ATOM_BFLOAT16_DOT_PRODUCT_KHR;
    case LOOMC_SPIRV_FEATURE_BFLOAT16_COOPERATIVE_MATRIX_KHR:
      return LOOM_SPIRV_FEATURE_ATOM_BFLOAT16_COOPERATIVE_MATRIX_KHR;
    case LOOMC_SPIRV_FEATURE_INT64:
      return LOOM_SPIRV_FEATURE_ATOM_INT64;
    case LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM:
      return LOOM_SPIRV_FEATURE_ATOM_GROUP_NON_UNIFORM;
    case LOOMC_SPIRV_FEATURE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE:
      return LOOM_SPIRV_FEATURE_ATOM_VULKAN_MEMORY_MODEL_DEVICE_SCOPE;
    case LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_INT64_ATOMICS:
      return LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_INT64_ATOMICS;
    case LOOMC_SPIRV_FEATURE_WORKGROUP_INT64_ATOMICS:
      return LOOM_SPIRV_FEATURE_ATOM_WORKGROUP_INT64_ATOMICS;
    case LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT16_ATOMICS:
      return LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_FLOAT16_ATOMICS;
    case LOOMC_SPIRV_FEATURE_WORKGROUP_FLOAT16_ATOMICS:
      return LOOM_SPIRV_FEATURE_ATOM_WORKGROUP_FLOAT16_ATOMICS;
    case LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT16_ATOMIC_ADD:
      return LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_FLOAT16_ATOMIC_ADD;
    case LOOMC_SPIRV_FEATURE_WORKGROUP_FLOAT16_ATOMIC_ADD:
      return LOOM_SPIRV_FEATURE_ATOM_WORKGROUP_FLOAT16_ATOMIC_ADD;
    case LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT32_ATOMICS:
      return LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_FLOAT32_ATOMICS;
    case LOOMC_SPIRV_FEATURE_WORKGROUP_FLOAT32_ATOMICS:
      return LOOM_SPIRV_FEATURE_ATOM_WORKGROUP_FLOAT32_ATOMICS;
    case LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT32_ATOMIC_ADD:
      return LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_FLOAT32_ATOMIC_ADD;
    case LOOMC_SPIRV_FEATURE_WORKGROUP_FLOAT32_ATOMIC_ADD:
      return LOOM_SPIRV_FEATURE_ATOM_WORKGROUP_FLOAT32_ATOMIC_ADD;
    case LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT64_ATOMICS:
      return LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_FLOAT64_ATOMICS;
    case LOOMC_SPIRV_FEATURE_WORKGROUP_FLOAT64_ATOMICS:
      return LOOM_SPIRV_FEATURE_ATOM_WORKGROUP_FLOAT64_ATOMICS;
    case LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT64_ATOMIC_ADD:
      return LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_FLOAT64_ATOMIC_ADD;
    case LOOMC_SPIRV_FEATURE_WORKGROUP_FLOAT64_ATOMIC_ADD:
      return LOOM_SPIRV_FEATURE_ATOM_WORKGROUP_FLOAT64_ATOMIC_ADD;
    case LOOMC_SPIRV_FEATURE_FLOAT32_DENORM_PRESERVE:
      return LOOM_SPIRV_FEATURE_ATOM_FLOAT32_DENORM_PRESERVE;
    case LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_BALLOT:
      return LOOM_SPIRV_FEATURE_ATOM_GROUP_NON_UNIFORM_BALLOT;
    case LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_VOTE:
      return LOOM_SPIRV_FEATURE_ATOM_GROUP_NON_UNIFORM_VOTE;
    case LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_ARITHMETIC:
      return LOOM_SPIRV_FEATURE_ATOM_GROUP_NON_UNIFORM_ARITHMETIC;
    case LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_CLUSTERED:
      return LOOM_SPIRV_FEATURE_ATOM_GROUP_NON_UNIFORM_CLUSTERED;
    case LOOMC_SPIRV_FEATURE_SUBGROUP_EXTENDED_TYPES:
      return LOOM_SPIRV_FEATURE_ATOM_SUBGROUP_EXTENDED_TYPES;
    case LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_SHUFFLE:
      return LOOM_SPIRV_FEATURE_ATOM_GROUP_NON_UNIFORM_SHUFFLE;
    case LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_SHUFFLE_RELATIVE:
      return LOOM_SPIRV_FEATURE_ATOM_GROUP_NON_UNIFORM_SHUFFLE_RELATIVE;
    case LOOMC_SPIRV_FEATURE_INTEGER_DOT_PRODUCT:
      return LOOM_SPIRV_FEATURE_ATOM_INTEGER_DOT_PRODUCT;
    case LOOMC_SPIRV_FEATURE_INTEGER_DOT_PRODUCT_INPUT_ALL:
      return LOOM_SPIRV_FEATURE_ATOM_INTEGER_DOT_PRODUCT_INPUT_ALL;
    case LOOMC_SPIRV_FEATURE_INTEGER_DOT_PRODUCT_INPUT4X8_BIT:
      return LOOM_SPIRV_FEATURE_ATOM_INTEGER_DOT_PRODUCT_INPUT4X8_BIT;
    case LOOMC_SPIRV_FEATURE_INTEGER_DOT_PRODUCT_INPUT4X8_BIT_PACKED:
      return LOOM_SPIRV_FEATURE_ATOM_INTEGER_DOT_PRODUCT_INPUT4X8_BIT_PACKED;
    case LOOMC_SPIRV_FEATURE_FLOAT16_64_DENORM_PRESERVE:
      return LOOM_SPIRV_FEATURE_ATOM_FLOAT16_64_DENORM_PRESERVE;
    case LOOMC_SPIRV_FEATURE_FLOAT16_64_ROUNDING_RTE:
      return LOOM_SPIRV_FEATURE_ATOM_FLOAT16_64_ROUNDING_RTE;
    case LOOMC_SPIRV_FEATURE_FLOAT32_ROUNDING_RTE:
      return LOOM_SPIRV_FEATURE_ATOM_FLOAT32_ROUNDING_RTE;
    case LOOMC_SPIRV_FEATURE_FLOAT16_SIGNED_ZERO_INF_NAN_PRESERVE:
      return LOOM_SPIRV_FEATURE_ATOM_FLOAT16_SIGNED_ZERO_INF_NAN_PRESERVE;
    case LOOMC_SPIRV_FEATURE_FLOAT32_SIGNED_ZERO_INF_NAN_PRESERVE:
      return LOOM_SPIRV_FEATURE_ATOM_FLOAT32_SIGNED_ZERO_INF_NAN_PRESERVE;
    case LOOMC_SPIRV_FEATURE_FLOAT64_SIGNED_ZERO_INF_NAN_PRESERVE:
      return LOOM_SPIRV_FEATURE_ATOM_FLOAT64_SIGNED_ZERO_INF_NAN_PRESERVE;
    case LOOMC_SPIRV_FEATURE_FLOAT64_TO_FLOAT16_CONVERSION:
      return LOOM_SPIRV_FEATURE_ATOM_FLOAT64_TO_FLOAT16_CONVERSION;
    case LOOMC_SPIRV_FEATURE_FMA_FLOAT16:
      return LOOM_SPIRV_FEATURE_ATOM_FMA_FLOAT16;
    case LOOMC_SPIRV_FEATURE_FMA_FLOAT32:
      return LOOM_SPIRV_FEATURE_ATOM_FMA_FLOAT32;
    case LOOMC_SPIRV_FEATURE_FMA_FLOAT64:
      return LOOM_SPIRV_FEATURE_ATOM_FMA_FLOAT64;
    case LOOMC_SPIRV_FEATURE_FLOAT_CONTROLS2:
      return LOOM_SPIRV_FEATURE_ATOM_FLOAT_CONTROLS2;
    case LOOMC_SPIRV_FEATURE_UNKNOWN:
    case LOOMC_SPIRV_FEATURE_COUNT:
      return LOOM_SPIRV_FEATURE_ATOM_UNKNOWN;
  }
  return LOOM_SPIRV_FEATURE_ATOM_UNKNOWN;
}

static loomc_spirv_feature_bits_t loomc_spirv_rows_feature_bits_from_loom(
    loom_spirv_feature_bits_t feature_bits) {
  loomc_spirv_feature_bits_t public_feature_bits = 0;
  for (uint32_t i = LOOMC_SPIRV_FEATURE_UNKNOWN + 1;
       i < LOOMC_SPIRV_FEATURE_COUNT; ++i) {
    const loomc_spirv_feature_t feature = (loomc_spirv_feature_t)i;
    const loom_spirv_feature_atom_t atom =
        loomc_spirv_feature_atom_from_public(feature);
    if ((feature_bits & loom_spirv_feature_atom_bit(atom)) != 0) {
      public_feature_bits |= loomc_spirv_feature_bit(feature);
    }
  }
  return public_feature_bits;
}

static loomc_spirv_feature_bits_t loomc_spirv_rows_public_feature_mask(void) {
  loomc_spirv_feature_bits_t mask = 0;
  for (uint32_t i = LOOMC_SPIRV_FEATURE_UNKNOWN + 1;
       i < LOOMC_SPIRV_FEATURE_COUNT; ++i) {
    mask |= loomc_spirv_feature_bit((loomc_spirv_feature_t)i);
  }
  return mask;
}

static loomc_status_t loomc_spirv_rows_feature_bits_to_loom(
    loomc_spirv_feature_bits_t feature_bits,
    loom_spirv_feature_bits_t* out_feature_bits) {
  *out_feature_bits = 0;
  if ((feature_bits & ~loomc_spirv_rows_public_feature_mask()) != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "SPIR-V feature bitset contains unknown bits");
  }
  for (uint32_t i = LOOMC_SPIRV_FEATURE_UNKNOWN + 1;
       i < LOOMC_SPIRV_FEATURE_COUNT; ++i) {
    const loomc_spirv_feature_t feature = (loomc_spirv_feature_t)i;
    if ((feature_bits & loomc_spirv_feature_bit(feature)) == 0) {
      continue;
    }
    *out_feature_bits |= loom_spirv_feature_atom_bit(
        loomc_spirv_feature_atom_from_public(feature));
  }
  return loomc_ok_status();
}

static loomc_spirv_scalar_type_t loomc_spirv_rows_scalar_type_from_loom(
    loom_spirv_scalar_type_t type) {
  switch (type) {
    case LOOM_SPIRV_SCALAR_TYPE_UNKNOWN:
      return LOOMC_SPIRV_SCALAR_TYPE_UNKNOWN;
    case LOOM_SPIRV_SCALAR_TYPE_F16:
      return LOOMC_SPIRV_SCALAR_TYPE_F16;
    case LOOM_SPIRV_SCALAR_TYPE_F32:
      return LOOMC_SPIRV_SCALAR_TYPE_F32;
    case LOOM_SPIRV_SCALAR_TYPE_F64:
      return LOOMC_SPIRV_SCALAR_TYPE_F64;
    case LOOM_SPIRV_SCALAR_TYPE_BF16:
      return LOOMC_SPIRV_SCALAR_TYPE_BF16;
    case LOOM_SPIRV_SCALAR_TYPE_S8:
      return LOOMC_SPIRV_SCALAR_TYPE_S8;
    case LOOM_SPIRV_SCALAR_TYPE_S16:
      return LOOMC_SPIRV_SCALAR_TYPE_S16;
    case LOOM_SPIRV_SCALAR_TYPE_S32:
      return LOOMC_SPIRV_SCALAR_TYPE_S32;
    case LOOM_SPIRV_SCALAR_TYPE_S64:
      return LOOMC_SPIRV_SCALAR_TYPE_S64;
    case LOOM_SPIRV_SCALAR_TYPE_U8:
      return LOOMC_SPIRV_SCALAR_TYPE_U8;
    case LOOM_SPIRV_SCALAR_TYPE_U16:
      return LOOMC_SPIRV_SCALAR_TYPE_U16;
    case LOOM_SPIRV_SCALAR_TYPE_U32:
      return LOOMC_SPIRV_SCALAR_TYPE_U32;
    case LOOM_SPIRV_SCALAR_TYPE_U64:
      return LOOMC_SPIRV_SCALAR_TYPE_U64;
  }
  IREE_ASSERT_UNREACHABLE("unknown internal SPIR-V scalar type");
  IREE_BUILTIN_UNREACHABLE();
}

static loomc_status_t loomc_spirv_rows_scalar_type_to_loom(
    loomc_spirv_scalar_type_t type, loom_spirv_scalar_type_t* out_type) {
  *out_type = LOOM_SPIRV_SCALAR_TYPE_UNKNOWN;
  switch (type) {
    case LOOMC_SPIRV_SCALAR_TYPE_F16:
      *out_type = LOOM_SPIRV_SCALAR_TYPE_F16;
      return loomc_ok_status();
    case LOOMC_SPIRV_SCALAR_TYPE_F32:
      *out_type = LOOM_SPIRV_SCALAR_TYPE_F32;
      return loomc_ok_status();
    case LOOMC_SPIRV_SCALAR_TYPE_F64:
      *out_type = LOOM_SPIRV_SCALAR_TYPE_F64;
      return loomc_ok_status();
    case LOOMC_SPIRV_SCALAR_TYPE_BF16:
      *out_type = LOOM_SPIRV_SCALAR_TYPE_BF16;
      return loomc_ok_status();
    case LOOMC_SPIRV_SCALAR_TYPE_S8:
      *out_type = LOOM_SPIRV_SCALAR_TYPE_S8;
      return loomc_ok_status();
    case LOOMC_SPIRV_SCALAR_TYPE_S16:
      *out_type = LOOM_SPIRV_SCALAR_TYPE_S16;
      return loomc_ok_status();
    case LOOMC_SPIRV_SCALAR_TYPE_S32:
      *out_type = LOOM_SPIRV_SCALAR_TYPE_S32;
      return loomc_ok_status();
    case LOOMC_SPIRV_SCALAR_TYPE_S64:
      *out_type = LOOM_SPIRV_SCALAR_TYPE_S64;
      return loomc_ok_status();
    case LOOMC_SPIRV_SCALAR_TYPE_U8:
      *out_type = LOOM_SPIRV_SCALAR_TYPE_U8;
      return loomc_ok_status();
    case LOOMC_SPIRV_SCALAR_TYPE_U16:
      *out_type = LOOM_SPIRV_SCALAR_TYPE_U16;
      return loomc_ok_status();
    case LOOMC_SPIRV_SCALAR_TYPE_U32:
      *out_type = LOOM_SPIRV_SCALAR_TYPE_U32;
      return loomc_ok_status();
    case LOOMC_SPIRV_SCALAR_TYPE_U64:
      *out_type = LOOM_SPIRV_SCALAR_TYPE_U64;
      return loomc_ok_status();
    case LOOMC_SPIRV_SCALAR_TYPE_UNKNOWN:
      break;
  }
  return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                           "SPIR-V scalar type is invalid");
}

static loomc_spirv_scope_t loomc_spirv_rows_scope_from_loom(
    loom_spirv_scope_t scope) {
  switch (scope) {
    case LOOM_SPIRV_SCOPE_CROSS_DEVICE:
      return LOOMC_SPIRV_SCOPE_CROSS_DEVICE;
    case LOOM_SPIRV_SCOPE_DEVICE:
      return LOOMC_SPIRV_SCOPE_DEVICE;
    case LOOM_SPIRV_SCOPE_WORKGROUP:
      return LOOMC_SPIRV_SCOPE_WORKGROUP;
    case LOOM_SPIRV_SCOPE_SUBGROUP:
      return LOOMC_SPIRV_SCOPE_SUBGROUP;
    case LOOM_SPIRV_SCOPE_INVOCATION:
      return LOOMC_SPIRV_SCOPE_INVOCATION;
    case LOOM_SPIRV_SCOPE_QUEUE_FAMILY:
      return LOOMC_SPIRV_SCOPE_QUEUE_FAMILY;
    case LOOM_SPIRV_SCOPE_SHADER_CALL_KHR:
      return LOOMC_SPIRV_SCOPE_SHADER_CALL_KHR;
    case LOOM_SPIRV_SCOPE_MAX:
      break;
  }
  IREE_ASSERT_UNREACHABLE("unknown internal SPIR-V scope");
  IREE_BUILTIN_UNREACHABLE();
}

static loomc_status_t loomc_spirv_rows_scope_to_loom(
    loomc_spirv_scope_t scope, loom_spirv_scope_t* out_scope) {
  *out_scope = LOOM_SPIRV_SCOPE_MAX;
  switch (scope) {
    case LOOMC_SPIRV_SCOPE_CROSS_DEVICE:
      *out_scope = LOOM_SPIRV_SCOPE_CROSS_DEVICE;
      return loomc_ok_status();
    case LOOMC_SPIRV_SCOPE_DEVICE:
      *out_scope = LOOM_SPIRV_SCOPE_DEVICE;
      return loomc_ok_status();
    case LOOMC_SPIRV_SCOPE_WORKGROUP:
      *out_scope = LOOM_SPIRV_SCOPE_WORKGROUP;
      return loomc_ok_status();
    case LOOMC_SPIRV_SCOPE_SUBGROUP:
      *out_scope = LOOM_SPIRV_SCOPE_SUBGROUP;
      return loomc_ok_status();
    case LOOMC_SPIRV_SCOPE_INVOCATION:
      *out_scope = LOOM_SPIRV_SCOPE_INVOCATION;
      return loomc_ok_status();
    case LOOMC_SPIRV_SCOPE_QUEUE_FAMILY:
      *out_scope = LOOM_SPIRV_SCOPE_QUEUE_FAMILY;
      return loomc_ok_status();
    case LOOMC_SPIRV_SCOPE_SHADER_CALL_KHR:
      *out_scope = LOOM_SPIRV_SCOPE_SHADER_CALL_KHR;
      return loomc_ok_status();
  }
  return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                           "SPIR-V scope is invalid");
}

static loomc_spirv_component_type_t loomc_spirv_rows_component_type_from_loom(
    loom_spirv_component_type_t component_type) {
  switch (component_type) {
    case LOOM_SPIRV_COMPONENT_TYPE_FLOAT16_NV:
      return LOOMC_SPIRV_COMPONENT_TYPE_FLOAT16_NV;
    case LOOM_SPIRV_COMPONENT_TYPE_FLOAT32_NV:
      return LOOMC_SPIRV_COMPONENT_TYPE_FLOAT32_NV;
    case LOOM_SPIRV_COMPONENT_TYPE_FLOAT64_NV:
      return LOOMC_SPIRV_COMPONENT_TYPE_FLOAT64_NV;
    case LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT8_NV:
      return LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT8_NV;
    case LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT16_NV:
      return LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT16_NV;
    case LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT32_NV:
      return LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT32_NV;
    case LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT64_NV:
      return LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT64_NV;
    case LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT8_NV:
      return LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT8_NV;
    case LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT16_NV:
      return LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT16_NV;
    case LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT32_NV:
      return LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT32_NV;
    case LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT64_NV:
      return LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT64_NV;
    case LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT8_PACKED_NV:
      return LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT8_PACKED_NV;
    case LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT8_PACKED_NV:
      return LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT8_PACKED_NV;
    case LOOM_SPIRV_COMPONENT_TYPE_FLOAT_E4_M3_NV:
      return LOOMC_SPIRV_COMPONENT_TYPE_FLOAT_E4_M3_NV;
    case LOOM_SPIRV_COMPONENT_TYPE_FLOAT_E5_M2_NV:
      return LOOMC_SPIRV_COMPONENT_TYPE_FLOAT_E5_M2_NV;
    case LOOM_SPIRV_COMPONENT_TYPE_MAX:
      break;
  }
  IREE_ASSERT_UNREACHABLE("unknown internal SPIR-V component type");
  IREE_BUILTIN_UNREACHABLE();
}

static loomc_status_t loomc_spirv_rows_component_type_to_loom(
    loomc_spirv_component_type_t component_type,
    loom_spirv_component_type_t* out_component_type) {
  *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_MAX;
  switch (component_type) {
    case LOOMC_SPIRV_COMPONENT_TYPE_FLOAT16_NV:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_FLOAT16_NV;
      return loomc_ok_status();
    case LOOMC_SPIRV_COMPONENT_TYPE_FLOAT32_NV:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_FLOAT32_NV;
      return loomc_ok_status();
    case LOOMC_SPIRV_COMPONENT_TYPE_FLOAT64_NV:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_FLOAT64_NV;
      return loomc_ok_status();
    case LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT8_NV:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT8_NV;
      return loomc_ok_status();
    case LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT16_NV:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT16_NV;
      return loomc_ok_status();
    case LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT32_NV:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT32_NV;
      return loomc_ok_status();
    case LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT64_NV:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT64_NV;
      return loomc_ok_status();
    case LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT8_NV:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT8_NV;
      return loomc_ok_status();
    case LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT16_NV:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT16_NV;
      return loomc_ok_status();
    case LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT32_NV:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT32_NV;
      return loomc_ok_status();
    case LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT64_NV:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT64_NV;
      return loomc_ok_status();
    case LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT8_PACKED_NV:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT8_PACKED_NV;
      return loomc_ok_status();
    case LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT8_PACKED_NV:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT8_PACKED_NV;
      return loomc_ok_status();
    case LOOMC_SPIRV_COMPONENT_TYPE_FLOAT_E4_M3_NV:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_FLOAT_E4_M3_NV;
      return loomc_ok_status();
    case LOOMC_SPIRV_COMPONENT_TYPE_FLOAT_E5_M2_NV:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_FLOAT_E5_M2_NV;
      return loomc_ok_status();
  }
  return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                           "SPIR-V component type is invalid");
}

void loomc_spirv_cooperative_matrix_row_from_loom(
    const loom_spirv_cooperative_matrix_property_t* property,
    loomc_spirv_cooperative_matrix_row_t* out_row) {
  *out_row = (loomc_spirv_cooperative_matrix_row_t){
      .name = loomc_string_view_from_iree(property->name),
      .state = LOOMC_TARGET_FACT_STATE_TRUE,
      .provenance = loomc_string_view_empty(),
      .required_features = loomc_spirv_rows_feature_bits_from_loom(
          property->required_feature_bits),
      .m_size = property->m_size,
      .n_size = property->n_size,
      .k_size = property->k_size,
      .lhs_type = loomc_spirv_rows_scalar_type_from_loom(property->lhs_type),
      .rhs_type = loomc_spirv_rows_scalar_type_from_loom(property->rhs_type),
      .accumulator_type =
          loomc_spirv_rows_scalar_type_from_loom(property->accumulator_type),
      .result_type =
          loomc_spirv_rows_scalar_type_from_loom(property->result_type),
      .scope = loomc_spirv_rows_scope_from_loom(property->scope),
      .layout_flags =
          (loomc_spirv_cooperative_matrix_layout_flags_t)property->layout_flags,
      .storage_class_flags =
          (loomc_spirv_storage_class_flags_t)property->storage_class_flags,
      .operand_flags = (loomc_spirv_cooperative_matrix_operand_flags_t)
                           property->operand_flags,
  };
}

static void loomc_spirv_rows_vector_row_from_loom(
    const loom_spirv_cooperative_vector_property_t* property,
    loomc_spirv_cooperative_vector_row_t* out_row) {
  *out_row = (loomc_spirv_cooperative_vector_row_t){
      .name = loomc_string_view_from_iree(property->name),
      .state = LOOMC_TARGET_FACT_STATE_TRUE,
      .provenance = loomc_string_view_empty(),
      .required_features = loomc_spirv_rows_feature_bits_from_loom(
          property->required_feature_bits),
      .m_size = property->m_size,
      .k_size = property->k_size,
      .input_type =
          loomc_spirv_rows_component_type_from_loom(property->input_type),
      .input_interpretation = loomc_spirv_rows_component_type_from_loom(
          property->input_interpretation),
      .matrix_interpretation = loomc_spirv_rows_component_type_from_loom(
          property->matrix_interpretation),
      .bias_interpretation = loomc_spirv_rows_component_type_from_loom(
          property->bias_interpretation),
      .result_type =
          loomc_spirv_rows_component_type_from_loom(property->result_type),
      .matrix_layout_flags =
          (loomc_spirv_cooperative_vector_matrix_layout_flags_t)
              property->matrix_layout_flags,
      .storage_class_flags =
          (loomc_spirv_storage_class_flags_t)property->storage_class_flags,
      .flags = (loomc_spirv_cooperative_vector_flags_t)property->flags,
  };
}

static loomc_status_t loomc_spirv_rows_matrix_row_to_loom(
    const loomc_spirv_cooperative_matrix_row_t* row,
    loom_spirv_cooperative_matrix_property_t* out_property) {
  *out_property = (loom_spirv_cooperative_matrix_property_t){
      .name = iree_string_view_from_loomc(row->name),
      .m_size = row->m_size,
      .n_size = row->n_size,
      .k_size = row->k_size,
      .layout_flags =
          (loom_spirv_cooperative_matrix_layout_flags_t)row->layout_flags,
      .storage_class_flags =
          (loom_spirv_storage_class_flags_t)row->storage_class_flags,
      .operand_flags =
          (loom_spirv_cooperative_matrix_operand_flags_t)row->operand_flags,
  };
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_feature_bits_to_loom(
      row->required_features, &out_property->required_feature_bits));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_scalar_type_to_loom(
      row->lhs_type, &out_property->lhs_type));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_scalar_type_to_loom(
      row->rhs_type, &out_property->rhs_type));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_scalar_type_to_loom(
      row->accumulator_type, &out_property->accumulator_type));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_scalar_type_to_loom(
      row->result_type, &out_property->result_type));
  LOOMC_RETURN_IF_ERROR(
      loomc_spirv_rows_scope_to_loom(row->scope, &out_property->scope));
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_rows_vector_row_to_loom(
    const loomc_spirv_cooperative_vector_row_t* row,
    loom_spirv_cooperative_vector_property_t* out_property) {
  *out_property = (loom_spirv_cooperative_vector_property_t){
      .name = iree_string_view_from_loomc(row->name),
      .m_size = row->m_size,
      .k_size = row->k_size,
      .matrix_layout_flags =
          (loom_spirv_cooperative_vector_matrix_layout_flags_t)
              row->matrix_layout_flags,
      .storage_class_flags =
          (loom_spirv_storage_class_flags_t)row->storage_class_flags,
      .flags = (loom_spirv_cooperative_vector_flags_t)row->flags,
  };
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_feature_bits_to_loom(
      row->required_features, &out_property->required_feature_bits));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_component_type_to_loom(
      row->input_type, &out_property->input_type));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_component_type_to_loom(
      row->input_interpretation, &out_property->input_interpretation));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_component_type_to_loom(
      row->matrix_interpretation, &out_property->matrix_interpretation));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_component_type_to_loom(
      row->bias_interpretation, &out_property->bias_interpretation));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_component_type_to_loom(
      row->result_type, &out_property->result_type));
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_rows_validate_fact_state(
    loomc_target_fact_state_t state) {
  if (state != LOOMC_TARGET_FACT_STATE_UNKNOWN &&
      state != LOOMC_TARGET_FACT_STATE_FALSE &&
      state != LOOMC_TARGET_FACT_STATE_TRUE) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "target fact state is invalid");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_rows_validate_row_name(
    loomc_string_view_t name) {
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_validate_string_view(name));
  if (loomc_string_view_is_empty(name)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "SPIR-V cooperative row name must not be empty");
  }
  return loomc_ok_status();
}

static bool loomc_spirv_rows_scalar_type_is_signed(
    loomc_spirv_scalar_type_t type) {
  return type == LOOMC_SPIRV_SCALAR_TYPE_S8 ||
         type == LOOMC_SPIRV_SCALAR_TYPE_S16 ||
         type == LOOMC_SPIRV_SCALAR_TYPE_S32 ||
         type == LOOMC_SPIRV_SCALAR_TYPE_S64;
}

static loomc_status_t loomc_spirv_rows_validate_matrix_row(
    const loomc_spirv_cooperative_matrix_row_t* row) {
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_validate_fact_state(row->state));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_validate_string_view(row->provenance));
  if (row->state == LOOMC_TARGET_FACT_STATE_UNKNOWN) {
    return loomc_spirv_rows_validate_string_view(row->name);
  }
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_validate_row_name(row->name));
  loom_spirv_feature_bits_t required_feature_bits = 0;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_feature_bits_to_loom(
      row->required_features, &required_feature_bits));
  if (row->m_size == 0 || row->n_size == 0 || row->k_size == 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "SPIR-V cooperative matrix row dimensions must be "
                             "non-zero");
  }
  loom_spirv_scalar_type_t scalar_type = LOOM_SPIRV_SCALAR_TYPE_UNKNOWN;
  LOOMC_RETURN_IF_ERROR(
      loomc_spirv_rows_scalar_type_to_loom(row->lhs_type, &scalar_type));
  LOOMC_RETURN_IF_ERROR(
      loomc_spirv_rows_scalar_type_to_loom(row->rhs_type, &scalar_type));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_scalar_type_to_loom(
      row->accumulator_type, &scalar_type));
  LOOMC_RETURN_IF_ERROR(
      loomc_spirv_rows_scalar_type_to_loom(row->result_type, &scalar_type));
  loom_spirv_scope_t scope = LOOM_SPIRV_SCOPE_MAX;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_scope_to_loom(row->scope, &scope));
  const loomc_spirv_cooperative_matrix_layout_flags_t layout_mask =
      LOOMC_SPIRV_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_BIT |
      LOOMC_SPIRV_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR_BIT;
  if (row->layout_flags == 0 || (row->layout_flags & ~layout_mask) != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "SPIR-V cooperative matrix row layout flags are "
                             "invalid");
  }
  const loomc_spirv_storage_class_flags_t storage_class_mask =
      LOOMC_SPIRV_STORAGE_CLASS_BIT_WORKGROUP |
      LOOMC_SPIRV_STORAGE_CLASS_BIT_STORAGE_BUFFER |
      LOOMC_SPIRV_STORAGE_CLASS_BIT_PHYSICAL_STORAGE_BUFFER;
  if (row->storage_class_flags == 0 ||
      (row->storage_class_flags & ~storage_class_mask) != 0) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "SPIR-V cooperative matrix row storage-class flags are invalid");
  }
  const loomc_spirv_cooperative_matrix_operand_flags_t operand_mask =
      LOOMC_SPIRV_COOPERATIVE_MATRIX_OPERAND_A_SIGNED_COMPONENTS |
      LOOMC_SPIRV_COOPERATIVE_MATRIX_OPERAND_B_SIGNED_COMPONENTS |
      LOOMC_SPIRV_COOPERATIVE_MATRIX_OPERAND_C_SIGNED_COMPONENTS |
      LOOMC_SPIRV_COOPERATIVE_MATRIX_OPERAND_RESULT_SIGNED_COMPONENTS |
      LOOMC_SPIRV_COOPERATIVE_MATRIX_OPERAND_SATURATING_ACCUMULATION;
  if ((row->operand_flags & ~operand_mask) != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "SPIR-V cooperative matrix row operand flags are "
                             "invalid");
  }
  const loomc_spirv_scalar_type_t component_types[] = {
      row->lhs_type, row->rhs_type, row->accumulator_type, row->result_type};
  const loomc_spirv_cooperative_matrix_operand_flags_t signed_flags[] = {
      LOOMC_SPIRV_COOPERATIVE_MATRIX_OPERAND_A_SIGNED_COMPONENTS,
      LOOMC_SPIRV_COOPERATIVE_MATRIX_OPERAND_B_SIGNED_COMPONENTS,
      LOOMC_SPIRV_COOPERATIVE_MATRIX_OPERAND_C_SIGNED_COMPONENTS,
      LOOMC_SPIRV_COOPERATIVE_MATRIX_OPERAND_RESULT_SIGNED_COMPONENTS};
  for (loomc_host_size_t i = 0; i < IREE_ARRAYSIZE(component_types); ++i) {
    if (loomc_spirv_rows_scalar_type_is_signed(component_types[i]) !=
        ((row->operand_flags & signed_flags[i]) != 0)) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "SPIR-V cooperative matrix signed operand flags must match their "
          "component types");
    }
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_rows_validate_vector_row(
    const loomc_spirv_cooperative_vector_row_t* row) {
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_validate_fact_state(row->state));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_validate_string_view(row->provenance));
  if (row->state == LOOMC_TARGET_FACT_STATE_UNKNOWN) {
    return loomc_spirv_rows_validate_string_view(row->name);
  }
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_validate_row_name(row->name));
  loom_spirv_feature_bits_t required_feature_bits = 0;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_feature_bits_to_loom(
      row->required_features, &required_feature_bits));
  if (row->m_size == 0 || row->k_size == 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "SPIR-V cooperative vector row dimensions must be "
                             "non-zero");
  }
  loom_spirv_component_type_t component_type = LOOM_SPIRV_COMPONENT_TYPE_MAX;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_component_type_to_loom(
      row->input_type, &component_type));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_component_type_to_loom(
      row->input_interpretation, &component_type));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_component_type_to_loom(
      row->matrix_interpretation, &component_type));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_component_type_to_loom(
      row->bias_interpretation, &component_type));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_component_type_to_loom(
      row->result_type, &component_type));
  const loomc_spirv_cooperative_vector_matrix_layout_flags_t layout_mask =
      LOOMC_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_BIT |
      LOOMC_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_COLUMN_MAJOR_BIT |
      LOOMC_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_BIT |
      LOOMC_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_TRAINING_OPTIMAL_BIT;
  if (row->matrix_layout_flags == 0 ||
      (row->matrix_layout_flags & ~layout_mask) != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "SPIR-V cooperative vector row matrix-layout "
                             "flags are invalid");
  }
  const loomc_spirv_storage_class_flags_t storage_class_mask =
      LOOMC_SPIRV_STORAGE_CLASS_BIT_WORKGROUP |
      LOOMC_SPIRV_STORAGE_CLASS_BIT_STORAGE_BUFFER |
      LOOMC_SPIRV_STORAGE_CLASS_BIT_PHYSICAL_STORAGE_BUFFER;
  if (row->storage_class_flags == 0 ||
      (row->storage_class_flags & ~storage_class_mask) != 0) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "SPIR-V cooperative vector row storage-class flags are invalid");
  }
  const loomc_spirv_cooperative_vector_flags_t flags_mask =
      LOOMC_SPIRV_COOPERATIVE_VECTOR_FLAG_TRANSPOSE |
      LOOMC_SPIRV_COOPERATIVE_VECTOR_FLAG_TRAINING;
  if ((row->flags & ~flags_mask) != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "SPIR-V cooperative vector row flags are invalid");
  }
  return loomc_ok_status();
}

loomc_status_t loomc_spirv_profile_validate_cooperative_row_options(
    const loomc_spirv_profile_options_t* options) {
  if (options == NULL) {
    return loomc_ok_status();
  }
  if (options->cooperative_matrix_row_count != 0 &&
      options->cooperative_matrix_rows == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "SPIR-V cooperative_matrix_row_count is non-zero but "
        "cooperative_matrix_rows is NULL");
  }
  for (loomc_host_size_t i = 0; i < options->cooperative_matrix_row_count;
       ++i) {
    LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_validate_matrix_row(
        &options->cooperative_matrix_rows[i]));
  }
  if (options->cooperative_vector_row_count != 0 &&
      options->cooperative_vector_rows == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "SPIR-V cooperative_vector_row_count is non-zero but "
        "cooperative_vector_rows is NULL");
  }
  for (loomc_host_size_t i = 0; i < options->cooperative_vector_row_count;
       ++i) {
    LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_validate_vector_row(
        &options->cooperative_vector_rows[i]));
  }
  return loomc_ok_status();
}

static bool loomc_spirv_rows_matrix_row_keys_equal(
    const loomc_spirv_cooperative_matrix_row_t* lhs,
    const loomc_spirv_cooperative_matrix_row_t* rhs) {
  return lhs->required_features == rhs->required_features &&
         lhs->m_size == rhs->m_size && lhs->n_size == rhs->n_size &&
         lhs->k_size == rhs->k_size && lhs->lhs_type == rhs->lhs_type &&
         lhs->rhs_type == rhs->rhs_type &&
         lhs->accumulator_type == rhs->accumulator_type &&
         lhs->result_type == rhs->result_type && lhs->scope == rhs->scope &&
         lhs->layout_flags == rhs->layout_flags &&
         lhs->storage_class_flags == rhs->storage_class_flags &&
         lhs->operand_flags == rhs->operand_flags;
}

static bool loomc_spirv_rows_vector_row_keys_equal(
    const loomc_spirv_cooperative_vector_row_t* lhs,
    const loomc_spirv_cooperative_vector_row_t* rhs) {
  return lhs->required_features == rhs->required_features &&
         lhs->m_size == rhs->m_size && lhs->k_size == rhs->k_size &&
         lhs->input_type == rhs->input_type &&
         lhs->input_interpretation == rhs->input_interpretation &&
         lhs->matrix_interpretation == rhs->matrix_interpretation &&
         lhs->bias_interpretation == rhs->bias_interpretation &&
         lhs->result_type == rhs->result_type &&
         lhs->matrix_layout_flags == rhs->matrix_layout_flags &&
         lhs->storage_class_flags == rhs->storage_class_flags &&
         lhs->flags == rhs->flags;
}

static loomc_status_t loomc_spirv_rows_emit_diagnostic(
    loomc_result_t* result, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count) {
  const loom_diagnostic_emission_t emission = {
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  LOOMC_RETURN_IF_ERROR(loomc_result_add_loom_diagnostic_emission(
      result, NULL, (loom_source_resolver_t){0}, LOOM_EMITTER_VERIFIER,
      &emission, NULL));
  loomc_result_set_status_code(result, LOOMC_STATUS_FAILED_PRECONDITION);
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_rows_fail_row_contradiction(
    loomc_result_t* result, const char* row_kind, loomc_string_view_t lhs_name,
    loomc_string_view_t lhs_provenance, loomc_target_fact_state_t lhs_state,
    loomc_string_view_t rhs_name, loomc_string_view_t rhs_provenance,
    loomc_target_fact_state_t rhs_state) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(iree_make_cstring_view(row_kind)),
      loom_param_string(iree_string_view_from_loomc(lhs_name)),
      loom_param_string(iree_string_view_from_loomc(lhs_provenance)),
      loom_param_bool(lhs_state == LOOMC_TARGET_FACT_STATE_TRUE),
      loom_param_string(iree_string_view_from_loomc(rhs_name)),
      loom_param_string(iree_string_view_from_loomc(rhs_provenance)),
      loom_param_bool(rhs_state == LOOMC_TARGET_FACT_STATE_TRUE),
  };
  return loomc_spirv_rows_emit_diagnostic(result, LOOM_ERR_SPIRV_041, params,
                                          IREE_ARRAYSIZE(params));
}

static loomc_status_t loomc_spirv_rows_fail_row_name_contradiction(
    loomc_result_t* result, const char* row_kind, loomc_string_view_t name,
    loomc_string_view_t lhs_provenance, loomc_string_view_t rhs_provenance) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(iree_make_cstring_view(row_kind)),
      loom_param_string(iree_string_view_from_loomc(name)),
      loom_param_string(iree_string_view_from_loomc(lhs_provenance)),
      loom_param_string(iree_string_view_from_loomc(rhs_provenance)),
  };
  return loomc_spirv_rows_emit_diagnostic(result, LOOM_ERR_SPIRV_042, params,
                                          IREE_ARRAYSIZE(params));
}

static void loomc_spirv_rows_deinitialize_matrix_row(
    loomc_allocator_t allocator, loomc_spirv_cooperative_matrix_row_t* row) {
  loomc_allocator_free(allocator, (void*)row->name.data);
  loomc_allocator_free(allocator, (void*)row->provenance.data);
  *row = (loomc_spirv_cooperative_matrix_row_t){0};
}

static void loomc_spirv_rows_deinitialize_vector_row(
    loomc_allocator_t allocator, loomc_spirv_cooperative_vector_row_t* row) {
  loomc_allocator_free(allocator, (void*)row->name.data);
  loomc_allocator_free(allocator, (void*)row->provenance.data);
  *row = (loomc_spirv_cooperative_vector_row_t){0};
}

static loomc_status_t loomc_spirv_rows_clone_matrix_row(
    const loomc_spirv_cooperative_matrix_row_t* source,
    loomc_allocator_t allocator, loomc_spirv_cooperative_matrix_row_t* target) {
  *target = *source;
  target->name = loomc_string_view_empty();
  target->provenance = loomc_string_view_empty();
  loomc_status_t status =
      loomc_string_view_clone(source->name, allocator, &target->name);
  if (loomc_status_is_ok(status)) {
    status = loomc_string_view_clone(source->provenance, allocator,
                                     &target->provenance);
  }
  if (!loomc_status_is_ok(status)) {
    loomc_spirv_rows_deinitialize_matrix_row(allocator, target);
  }
  return status;
}

static loomc_status_t loomc_spirv_rows_clone_vector_row(
    const loomc_spirv_cooperative_vector_row_t* source,
    loomc_allocator_t allocator, loomc_spirv_cooperative_vector_row_t* target) {
  *target = *source;
  target->name = loomc_string_view_empty();
  target->provenance = loomc_string_view_empty();
  loomc_status_t status =
      loomc_string_view_clone(source->name, allocator, &target->name);
  if (loomc_status_is_ok(status)) {
    status = loomc_string_view_clone(source->provenance, allocator,
                                     &target->provenance);
  }
  if (!loomc_status_is_ok(status)) {
    loomc_spirv_rows_deinitialize_vector_row(allocator, target);
  }
  return status;
}

static loomc_status_t loomc_spirv_rows_apply_matrix_row_fact(
    const loomc_spirv_cooperative_matrix_row_t* row,
    loomc_spirv_cooperative_row_fact_set_t* row_facts, loomc_result_t* result,
    loomc_allocator_t allocator) {
  if (row->state == LOOMC_TARGET_FACT_STATE_UNKNOWN) {
    return loomc_ok_status();
  }
  for (loomc_host_size_t i = 0; i < row_facts->matrix_row_count; ++i) {
    const loomc_spirv_cooperative_matrix_row_t* current =
        &row_facts->matrix_rows[i];
    const bool same_key = loomc_spirv_rows_matrix_row_keys_equal(current, row);
    if (same_key && current->state == row->state) {
      return loomc_ok_status();
    }
    if (same_key) {
      return loomc_spirv_rows_fail_row_contradiction(
          result, "matrix", current->name, current->provenance, current->state,
          row->name, row->provenance, row->state);
    }
    if (loomc_string_view_equal(current->name, row->name)) {
      return loomc_spirv_rows_fail_row_name_contradiction(
          result, "matrix", row->name, current->provenance, row->provenance);
    }
  }
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_clone_matrix_row(
      row, allocator, &row_facts->matrix_rows[row_facts->matrix_row_count]));
  ++row_facts->matrix_row_count;
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_rows_apply_vector_row_fact(
    const loomc_spirv_cooperative_vector_row_t* row,
    loomc_spirv_cooperative_row_fact_set_t* row_facts, loomc_result_t* result,
    loomc_allocator_t allocator) {
  if (row->state == LOOMC_TARGET_FACT_STATE_UNKNOWN) {
    return loomc_ok_status();
  }
  for (loomc_host_size_t i = 0; i < row_facts->vector_row_count; ++i) {
    const loomc_spirv_cooperative_vector_row_t* current =
        &row_facts->vector_rows[i];
    const bool same_key = loomc_spirv_rows_vector_row_keys_equal(current, row);
    if (same_key && current->state == row->state) {
      return loomc_ok_status();
    }
    if (same_key) {
      return loomc_spirv_rows_fail_row_contradiction(
          result, "vector", current->name, current->provenance, current->state,
          row->name, row->provenance, row->state);
    }
    if (loomc_string_view_equal(current->name, row->name)) {
      return loomc_spirv_rows_fail_row_name_contradiction(
          result, "vector", row->name, current->provenance, row->provenance);
    }
  }
  LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_clone_vector_row(
      row, allocator, &row_facts->vector_rows[row_facts->vector_row_count]));
  ++row_facts->vector_row_count;
  return loomc_ok_status();
}

static loomc_host_size_t loomc_spirv_rows_known_matrix_row_count(
    const loomc_spirv_cooperative_matrix_row_t* rows,
    loomc_host_size_t row_count) {
  loomc_host_size_t known_count = 0;
  for (loomc_host_size_t i = 0; i < row_count; ++i) {
    if (rows[i].state != LOOMC_TARGET_FACT_STATE_UNKNOWN) {
      ++known_count;
    }
  }
  return known_count;
}

static loomc_host_size_t loomc_spirv_rows_known_vector_row_count(
    const loomc_spirv_cooperative_vector_row_t* rows,
    loomc_host_size_t row_count) {
  loomc_host_size_t known_count = 0;
  for (loomc_host_size_t i = 0; i < row_count; ++i) {
    if (rows[i].state != LOOMC_TARGET_FACT_STATE_UNKNOWN) {
      ++known_count;
    }
  }
  return known_count;
}

static loomc_status_t loomc_spirv_rows_initialize_matrix_rows(
    const loomc_spirv_cooperative_row_fact_set_t* base_row_facts,
    const loomc_spirv_profile_options_t* options,
    loomc_spirv_cooperative_row_fact_set_t* row_facts, loomc_result_t* result,
    loomc_allocator_t allocator) {
  const loomc_spirv_cooperative_matrix_row_t* base_rows =
      base_row_facts ? base_row_facts->matrix_rows : NULL;
  const loomc_host_size_t base_row_count =
      base_row_facts ? base_row_facts->matrix_row_count : 0;
  const loomc_spirv_cooperative_matrix_row_t* option_rows =
      options ? options->cooperative_matrix_rows : NULL;
  const loomc_host_size_t option_row_count =
      options ? options->cooperative_matrix_row_count : 0;
  const loomc_host_size_t capacity =
      loomc_spirv_rows_known_matrix_row_count(base_rows, base_row_count) +
      loomc_spirv_rows_known_matrix_row_count(option_rows, option_row_count);
  if (capacity == 0) {
    return loomc_ok_status();
  }
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc(
      allocator, capacity * sizeof(*row_facts->matrix_rows),
      (void**)&row_facts->matrix_rows));
  for (loomc_host_size_t i = 0;
       i < base_row_count && loomc_result_succeeded(result); ++i) {
    LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_apply_matrix_row_fact(
        &base_rows[i], row_facts, result, allocator));
  }
  for (loomc_host_size_t i = 0;
       i < option_row_count && loomc_result_succeeded(result); ++i) {
    LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_apply_matrix_row_fact(
        &option_rows[i], row_facts, result, allocator));
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_rows_initialize_vector_rows(
    const loomc_spirv_cooperative_row_fact_set_t* base_row_facts,
    const loomc_spirv_profile_options_t* options,
    loomc_spirv_cooperative_row_fact_set_t* row_facts, loomc_result_t* result,
    loomc_allocator_t allocator) {
  const loomc_spirv_cooperative_vector_row_t* base_rows =
      base_row_facts ? base_row_facts->vector_rows : NULL;
  const loomc_host_size_t base_row_count =
      base_row_facts ? base_row_facts->vector_row_count : 0;
  const loomc_spirv_cooperative_vector_row_t* option_rows =
      options ? options->cooperative_vector_rows : NULL;
  const loomc_host_size_t option_row_count =
      options ? options->cooperative_vector_row_count : 0;
  const loomc_host_size_t capacity =
      loomc_spirv_rows_known_vector_row_count(base_rows, base_row_count) +
      loomc_spirv_rows_known_vector_row_count(option_rows, option_row_count);
  if (capacity == 0) {
    return loomc_ok_status();
  }
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc(
      allocator, capacity * sizeof(*row_facts->vector_rows),
      (void**)&row_facts->vector_rows));
  for (loomc_host_size_t i = 0;
       i < base_row_count && loomc_result_succeeded(result); ++i) {
    LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_apply_vector_row_fact(
        &base_rows[i], row_facts, result, allocator));
  }
  for (loomc_host_size_t i = 0;
       i < option_row_count && loomc_result_succeeded(result); ++i) {
    LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_apply_vector_row_fact(
        &option_rows[i], row_facts, result, allocator));
  }
  return loomc_ok_status();
}

loomc_status_t loomc_spirv_cooperative_row_fact_set_initialize(
    const loomc_spirv_cooperative_row_fact_set_t* base_row_facts,
    const loomc_spirv_profile_options_t* options, loomc_result_t* result,
    loomc_allocator_t allocator,
    loomc_spirv_cooperative_row_fact_set_t* out_row_facts) {
  *out_row_facts = (loomc_spirv_cooperative_row_fact_set_t){0};
  loomc_status_t status = loomc_spirv_rows_initialize_matrix_rows(
      base_row_facts, options, out_row_facts, result, allocator);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_spirv_rows_initialize_vector_rows(
        base_row_facts, options, out_row_facts, result, allocator);
  }
  if (!loomc_status_is_ok(status) || !loomc_result_succeeded(result)) {
    loomc_spirv_cooperative_row_fact_set_deinitialize(out_row_facts, allocator);
  }
  return status;
}

void loomc_spirv_cooperative_row_fact_set_deinitialize(
    loomc_spirv_cooperative_row_fact_set_t* row_facts,
    loomc_allocator_t allocator) {
  if (row_facts == NULL) {
    return;
  }
  loom_spirv_cooperative_property_storage_deinitialize(
      &row_facts->cooperative_property_storage,
      iree_allocator_from_loomc(allocator));
  for (loomc_host_size_t i = 0; i < row_facts->matrix_row_count; ++i) {
    loomc_spirv_rows_deinitialize_matrix_row(allocator,
                                             &row_facts->matrix_rows[i]);
  }
  loomc_allocator_free(allocator, row_facts->matrix_rows);
  for (loomc_host_size_t i = 0; i < row_facts->vector_row_count; ++i) {
    loomc_spirv_rows_deinitialize_vector_row(allocator,
                                             &row_facts->vector_rows[i]);
  }
  loomc_allocator_free(allocator, row_facts->vector_rows);
  *row_facts = (loomc_spirv_cooperative_row_fact_set_t){0};
}

static loomc_status_t loomc_spirv_rows_matrix_property_matches_row(
    const loom_spirv_cooperative_matrix_property_t* property,
    const loomc_spirv_cooperative_matrix_row_t* row, bool* out_matches) {
  loomc_spirv_cooperative_matrix_row_t property_row = {0};
  loomc_spirv_cooperative_matrix_row_from_loom(property, &property_row);
  *out_matches = loomc_spirv_rows_matrix_row_keys_equal(&property_row, row);
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_rows_vector_property_matches_row(
    const loom_spirv_cooperative_vector_property_t* property,
    const loomc_spirv_cooperative_vector_row_t* row, bool* out_matches) {
  loomc_spirv_cooperative_vector_row_t property_row = {0};
  loomc_spirv_rows_vector_row_from_loom(property, &property_row);
  *out_matches = loomc_spirv_rows_vector_row_keys_equal(&property_row, row);
  return loomc_ok_status();
}

static bool loomc_spirv_rows_matrix_properties_match(
    const loom_spirv_cooperative_matrix_property_t* lhs,
    const loom_spirv_cooperative_matrix_property_t* rhs) {
  return lhs->required_feature_bits == rhs->required_feature_bits &&
         lhs->m_size == rhs->m_size && lhs->n_size == rhs->n_size &&
         lhs->k_size == rhs->k_size && lhs->lhs_type == rhs->lhs_type &&
         lhs->rhs_type == rhs->rhs_type &&
         lhs->accumulator_type == rhs->accumulator_type &&
         lhs->result_type == rhs->result_type && lhs->scope == rhs->scope &&
         lhs->layout_flags == rhs->layout_flags &&
         lhs->storage_class_flags == rhs->storage_class_flags &&
         lhs->operand_flags == rhs->operand_flags;
}

static bool loomc_spirv_rows_vector_properties_match(
    const loom_spirv_cooperative_vector_property_t* lhs,
    const loom_spirv_cooperative_vector_property_t* rhs) {
  return lhs->required_feature_bits == rhs->required_feature_bits &&
         lhs->m_size == rhs->m_size && lhs->k_size == rhs->k_size &&
         lhs->input_type == rhs->input_type &&
         lhs->input_interpretation == rhs->input_interpretation &&
         lhs->matrix_interpretation == rhs->matrix_interpretation &&
         lhs->bias_interpretation == rhs->bias_interpretation &&
         lhs->result_type == rhs->result_type &&
         lhs->matrix_layout_flags == rhs->matrix_layout_flags &&
         lhs->storage_class_flags == rhs->storage_class_flags &&
         lhs->flags == rhs->flags;
}

static bool loomc_spirv_rows_matrix_property_already_present(
    const loom_spirv_cooperative_matrix_property_t* rows,
    loomc_host_size_t row_count,
    const loom_spirv_cooperative_matrix_property_t* row) {
  for (loomc_host_size_t i = 0; i < row_count; ++i) {
    if (loomc_spirv_rows_matrix_properties_match(&rows[i], row)) {
      return true;
    }
  }
  return false;
}

static bool loomc_spirv_rows_vector_property_already_present(
    const loom_spirv_cooperative_vector_property_t* rows,
    loomc_host_size_t row_count,
    const loom_spirv_cooperative_vector_property_t* row) {
  for (loomc_host_size_t i = 0; i < row_count; ++i) {
    if (loomc_spirv_rows_vector_properties_match(&rows[i], row)) {
      return true;
    }
  }
  return false;
}

static loomc_status_t loomc_spirv_rows_matrix_property_is_unavailable(
    const loomc_spirv_cooperative_row_fact_set_t* row_facts,
    const loom_spirv_cooperative_matrix_property_t* property,
    bool* out_unavailable) {
  *out_unavailable = false;
  for (loomc_host_size_t i = 0; i < row_facts->matrix_row_count; ++i) {
    const loomc_spirv_cooperative_matrix_row_t* row =
        &row_facts->matrix_rows[i];
    bool matches = false;
    if (row->state == LOOMC_TARGET_FACT_STATE_FALSE) {
      LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_matrix_property_matches_row(
          property, row, &matches));
    }
    if (matches) {
      *out_unavailable = true;
      return loomc_ok_status();
    }
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_rows_vector_property_is_unavailable(
    const loomc_spirv_cooperative_row_fact_set_t* row_facts,
    const loom_spirv_cooperative_vector_property_t* property,
    bool* out_unavailable) {
  *out_unavailable = false;
  for (loomc_host_size_t i = 0; i < row_facts->vector_row_count; ++i) {
    const loomc_spirv_cooperative_vector_row_t* row =
        &row_facts->vector_rows[i];
    bool matches = false;
    if (row->state == LOOMC_TARGET_FACT_STATE_FALSE) {
      LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_vector_property_matches_row(
          property, row, &matches));
    }
    if (matches) {
      *out_unavailable = true;
      return loomc_ok_status();
    }
  }
  return loomc_ok_status();
}

static loomc_host_size_t loomc_spirv_rows_matrix_row_state_count(
    const loomc_spirv_cooperative_row_fact_set_t* row_facts,
    loomc_target_fact_state_t state) {
  loomc_host_size_t count = 0;
  for (loomc_host_size_t i = 0; i < row_facts->matrix_row_count; ++i) {
    if (row_facts->matrix_rows[i].state == state) {
      ++count;
    }
  }
  return count;
}

static loomc_host_size_t loomc_spirv_rows_vector_row_state_count(
    const loomc_spirv_cooperative_row_fact_set_t* row_facts,
    loomc_target_fact_state_t state) {
  loomc_host_size_t count = 0;
  for (loomc_host_size_t i = 0; i < row_facts->vector_row_count; ++i) {
    if (row_facts->vector_rows[i].state == state) {
      ++count;
    }
  }
  return count;
}

loomc_status_t loomc_spirv_cooperative_row_fact_set_prepare_properties(
    loomc_spirv_cooperative_row_fact_set_t* row_facts,
    const loom_spirv_feature_set_t* feature_set, loomc_allocator_t allocator,
    const loom_spirv_cooperative_property_set_t** out_property_set) {
  *out_property_set = NULL;
  if (row_facts->matrix_row_count == 0 && row_facts->vector_row_count == 0) {
    loom_spirv_cooperative_property_set_prepare(
        feature_set, &row_facts->cooperative_properties);
    row_facts->prepared_properties = &row_facts->cooperative_properties;
    *out_property_set = row_facts->prepared_properties;
    return loomc_ok_status();
  }

  iree_host_size_t model_matrix_count = 0;
  const loom_spirv_cooperative_matrix_property_t* model_matrix_rows =
      loom_spirv_cooperative_matrix_model_properties(&model_matrix_count);
  iree_host_size_t model_vector_count = 0;
  const loom_spirv_cooperative_vector_property_t* model_vector_rows =
      loom_spirv_cooperative_vector_model_properties(&model_vector_count);
  const loomc_host_size_t matrix_capacity =
      model_matrix_count + loomc_spirv_rows_matrix_row_state_count(
                               row_facts, LOOMC_TARGET_FACT_STATE_TRUE);
  const loomc_host_size_t vector_capacity =
      model_vector_count + loomc_spirv_rows_vector_row_state_count(
                               row_facts, LOOMC_TARGET_FACT_STATE_TRUE);
  loom_spirv_cooperative_matrix_property_t* matrix_rows = NULL;
  loom_spirv_cooperative_vector_property_t* vector_rows = NULL;
  loomc_host_size_t matrix_count = 0;
  loomc_host_size_t vector_count = 0;
  loomc_status_t status = loomc_ok_status();
  if (matrix_capacity != 0) {
    status = loomc_allocator_malloc(allocator,
                                    matrix_capacity * sizeof(*matrix_rows),
                                    (void**)&matrix_rows);
  }
  if (loomc_status_is_ok(status) && vector_capacity != 0) {
    status = loomc_allocator_malloc(allocator,
                                    vector_capacity * sizeof(*vector_rows),
                                    (void**)&vector_rows);
  }
  for (iree_host_size_t i = 0;
       i < model_matrix_count && loomc_status_is_ok(status); ++i) {
    bool unavailable = false;
    status = loomc_spirv_rows_matrix_property_is_unavailable(
        row_facts, &model_matrix_rows[i], &unavailable);
    if (!loomc_status_is_ok(status)) {
      break;
    }
    if (unavailable) {
      continue;
    }
    matrix_rows[matrix_count++] = model_matrix_rows[i];
  }
  for (loomc_host_size_t i = 0;
       i < row_facts->matrix_row_count && loomc_status_is_ok(status); ++i) {
    const loomc_spirv_cooperative_matrix_row_t* row =
        &row_facts->matrix_rows[i];
    if (row->state != LOOMC_TARGET_FACT_STATE_TRUE) {
      continue;
    }
    loom_spirv_cooperative_matrix_property_t property = {0};
    status = loomc_spirv_rows_matrix_row_to_loom(row, &property);
    if (!loomc_status_is_ok(status)) {
      break;
    }
    if (loomc_spirv_rows_matrix_property_already_present(
            matrix_rows, matrix_count, &property)) {
      continue;
    }
    matrix_rows[matrix_count++] = property;
  }
  for (iree_host_size_t i = 0;
       i < model_vector_count && loomc_status_is_ok(status); ++i) {
    bool unavailable = false;
    status = loomc_spirv_rows_vector_property_is_unavailable(
        row_facts, &model_vector_rows[i], &unavailable);
    if (!loomc_status_is_ok(status)) {
      break;
    }
    if (unavailable) {
      continue;
    }
    vector_rows[vector_count++] = model_vector_rows[i];
  }
  for (loomc_host_size_t i = 0;
       i < row_facts->vector_row_count && loomc_status_is_ok(status); ++i) {
    const loomc_spirv_cooperative_vector_row_t* row =
        &row_facts->vector_rows[i];
    if (row->state != LOOMC_TARGET_FACT_STATE_TRUE) {
      continue;
    }
    loom_spirv_cooperative_vector_property_t property = {0};
    status = loomc_spirv_rows_vector_row_to_loom(row, &property);
    if (!loomc_status_is_ok(status)) {
      break;
    }
    if (loomc_spirv_rows_vector_property_already_present(
            vector_rows, vector_count, &property)) {
      continue;
    }
    vector_rows[vector_count++] = property;
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_status_from_iree(
        loom_spirv_cooperative_property_storage_initialize(
            feature_set->atom_bits, matrix_rows, matrix_count, vector_rows,
            vector_count, iree_allocator_from_loomc(allocator),
            &row_facts->cooperative_property_storage));
  }
  if (loomc_status_is_ok(status)) {
    row_facts->prepared_properties =
        &row_facts->cooperative_property_storage.set;
    *out_property_set = row_facts->prepared_properties;
  }
  loomc_allocator_free(allocator, vector_rows);
  loomc_allocator_free(allocator, matrix_rows);
  return status;
}

loomc_host_size_t loomc_spirv_cooperative_row_fact_set_matrix_row_count(
    const loomc_spirv_cooperative_row_fact_set_t* row_facts) {
  const loom_spirv_cooperative_property_set_t* cooperative_properties =
      row_facts->prepared_properties;
  return cooperative_properties->matrix_property_count +
         loomc_spirv_rows_matrix_row_state_count(row_facts,
                                                 LOOMC_TARGET_FACT_STATE_FALSE);
}

loomc_host_size_t loomc_spirv_cooperative_row_fact_set_vector_row_count(
    const loomc_spirv_cooperative_row_fact_set_t* row_facts) {
  const loom_spirv_cooperative_property_set_t* cooperative_properties =
      row_facts->prepared_properties;
  return cooperative_properties->vector_property_count +
         loomc_spirv_rows_vector_row_state_count(row_facts,
                                                 LOOMC_TARGET_FACT_STATE_FALSE);
}

static loomc_status_t loomc_spirv_rows_explicit_matrix_row_for_property(
    const loomc_spirv_cooperative_row_fact_set_t* row_facts,
    const loom_spirv_cooperative_matrix_property_t* property,
    loomc_target_fact_state_t state,
    const loomc_spirv_cooperative_matrix_row_t** out_row) {
  *out_row = NULL;
  for (loomc_host_size_t i = 0; i < row_facts->matrix_row_count; ++i) {
    const loomc_spirv_cooperative_matrix_row_t* row =
        &row_facts->matrix_rows[i];
    bool matches = false;
    if (row->state == state) {
      LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_matrix_property_matches_row(
          property, row, &matches));
    }
    if (matches) {
      *out_row = row;
      return loomc_ok_status();
    }
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_rows_explicit_vector_row_for_property(
    const loomc_spirv_cooperative_row_fact_set_t* row_facts,
    const loom_spirv_cooperative_vector_property_t* property,
    loomc_target_fact_state_t state,
    const loomc_spirv_cooperative_vector_row_t** out_row) {
  *out_row = NULL;
  for (loomc_host_size_t i = 0; i < row_facts->vector_row_count; ++i) {
    const loomc_spirv_cooperative_vector_row_t* row =
        &row_facts->vector_rows[i];
    bool matches = false;
    if (row->state == state) {
      LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_vector_property_matches_row(
          property, row, &matches));
    }
    if (matches) {
      *out_row = row;
      return loomc_ok_status();
    }
  }
  return loomc_ok_status();
}

static const loom_spirv_cooperative_property_set_t*
loomc_spirv_rows_prepared_property_set(
    const loomc_spirv_cooperative_row_fact_set_t* row_facts) {
  return row_facts->prepared_properties;
}

loomc_status_t loomc_spirv_cooperative_row_fact_set_matrix_row_at(
    const loomc_spirv_cooperative_row_fact_set_t* row_facts,
    loomc_host_size_t index, loomc_spirv_cooperative_matrix_row_t* out_row) {
  const loom_spirv_cooperative_property_set_t* cooperative_properties =
      loomc_spirv_rows_prepared_property_set(row_facts);
  if (index < cooperative_properties->matrix_property_count) {
    const loom_spirv_cooperative_matrix_property_t* property =
        &cooperative_properties->matrix_properties[index];
    loomc_spirv_cooperative_matrix_row_from_loom(property, out_row);
    const loomc_spirv_cooperative_matrix_row_t* explicit_row = NULL;
    LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_explicit_matrix_row_for_property(
        row_facts, property, LOOMC_TARGET_FACT_STATE_TRUE, &explicit_row));
    if (explicit_row != NULL) {
      out_row->name = explicit_row->name;
      out_row->provenance = explicit_row->provenance;
    }
    return loomc_ok_status();
  }
  index -= cooperative_properties->matrix_property_count;
  for (loomc_host_size_t i = 0; i < row_facts->matrix_row_count; ++i) {
    const loomc_spirv_cooperative_matrix_row_t* row =
        &row_facts->matrix_rows[i];
    if (row->state != LOOMC_TARGET_FACT_STATE_FALSE) {
      continue;
    }
    if (index == 0) {
      *out_row = *row;
      return loomc_ok_status();
    }
    --index;
  }
  return loomc_make_status(LOOMC_STATUS_OUT_OF_RANGE,
                           "SPIR-V cooperative matrix row index is out of "
                           "range");
}

loomc_status_t loomc_spirv_cooperative_row_fact_set_vector_row_at(
    const loomc_spirv_cooperative_row_fact_set_t* row_facts,
    loomc_host_size_t index, loomc_spirv_cooperative_vector_row_t* out_row) {
  const loom_spirv_cooperative_property_set_t* cooperative_properties =
      loomc_spirv_rows_prepared_property_set(row_facts);
  if (index < cooperative_properties->vector_property_count) {
    const loom_spirv_cooperative_vector_property_t* property =
        &cooperative_properties->vector_properties[index];
    loomc_spirv_rows_vector_row_from_loom(property, out_row);
    const loomc_spirv_cooperative_vector_row_t* explicit_row = NULL;
    LOOMC_RETURN_IF_ERROR(loomc_spirv_rows_explicit_vector_row_for_property(
        row_facts, property, LOOMC_TARGET_FACT_STATE_TRUE, &explicit_row));
    if (explicit_row != NULL) {
      out_row->name = explicit_row->name;
      out_row->provenance = explicit_row->provenance;
    }
    return loomc_ok_status();
  }
  index -= cooperative_properties->vector_property_count;
  for (loomc_host_size_t i = 0; i < row_facts->vector_row_count; ++i) {
    const loomc_spirv_cooperative_vector_row_t* row =
        &row_facts->vector_rows[i];
    if (row->state != LOOMC_TARGET_FACT_STATE_FALSE) {
      continue;
    }
    if (index == 0) {
      *out_row = *row;
      return loomc_ok_status();
    }
    --index;
  }
  return loomc_make_status(LOOMC_STATUS_OUT_OF_RANGE,
                           "SPIR-V cooperative vector row index is out of "
                           "range");
}
