// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/spirv/iree_hal.h"

#include <inttypes.h>
#include <stdint.h>

#include "diagnostic.h"
#include "iree/hal/drivers/vulkan/device_spec.h"
#include "loom/target/arch/spirv/cooperative_properties.h"
#include "loom/target/arch/spirv/features.h"
#include "loom/target/arch/spirv/profile.h"
#include "loomc/iree.h"
#include "profile_rows.h"
#include "result.h"
#include "target.h"

#define LOOMC_SPIRV_IREE_HAL_VULKAN_API_VERSION(major, minor, patch) \
  ((((uint32_t)(major)) << 22) | (((uint32_t)(minor)) << 12) |       \
   ((uint32_t)(patch)))

enum {
  LOOMC_SPIRV_IREE_HAL_VULKAN_COMPONENT_TYPE_BFLOAT16_KHR = 1000141000,
  LOOMC_SPIRV_IREE_HAL_VULKAN_SUBGROUP_FEATURE_BALLOT_BIT = 0x00000008,
  LOOMC_SPIRV_IREE_HAL_VULKAN_SUBGROUP_FEATURE_VOTE_BIT = 0x00000002,
  LOOMC_SPIRV_IREE_HAL_VULKAN_SUBGROUP_FEATURE_ARITHMETIC_BIT = 0x00000004,
  LOOMC_SPIRV_IREE_HAL_VULKAN_SUBGROUP_FEATURE_SHUFFLE_BIT = 0x00000010,
  LOOMC_SPIRV_IREE_HAL_VULKAN_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT =
      0x00000020,
  LOOMC_SPIRV_IREE_HAL_VULKAN_SUBGROUP_FEATURE_CLUSTERED_BIT = 0x00000040,
  // PCI-SIG vendor identifier reported by AMD Vulkan physical devices.
  LOOMC_SPIRV_IREE_HAL_VULKAN_AMD_VENDOR_ID = 0x1002,
  LOOMC_SPIRV_IREE_HAL_VULKAN_API_VERSION_1_3 =
      LOOMC_SPIRV_IREE_HAL_VULKAN_API_VERSION(1, 3, 0),
};

typedef struct loomc_spirv_iree_hal_feature_row_t {
  // General Vulkan HAL feature bits required for this fact.
  iree_hal_vulkan_general_features_t required_general_features;
  // Vulkan shader-atomic feature bits required for this fact.
  iree_hal_vulkan_shader_atomic_features_t required_atomic_features;
  // Vulkan device-spec flags required for this fact.
  iree_hal_vulkan_device_spec_flags_t required_device_flags;
  // Vulkan subgroup-operation bits required for this fact.
  uint32_t required_subgroup_operations;
  // Public SPIR-V feature represented by the row.
  loomc_spirv_feature_t feature;
  // Stable provenance for the device observation.
  const char* provenance;
} loomc_spirv_iree_hal_feature_row_t;

#define LOOMC_SPIRV_IREE_HAL_GENERAL_FEATURE_ROW(required_features,      \
                                                 public_feature, source) \
  {                                                                      \
      .required_general_features = (required_features),                  \
      .feature = LOOMC_SPIRV_FEATURE_##public_feature,                   \
      .provenance = "iree-hal:vulkan.feature." source,                   \
  }

#define LOOMC_SPIRV_IREE_HAL_ATOMIC_FEATURE_ROW(                 \
    required_features, required_atomics, public_feature, source) \
  {                                                              \
      .required_general_features = (required_features),          \
      .required_atomic_features = (required_atomics),            \
      .feature = LOOMC_SPIRV_FEATURE_##public_feature,           \
      .provenance = "iree-hal:vulkan.atomic." source,            \
  }

// clang-format off
static const loomc_spirv_iree_hal_feature_row_t
    kLoomcSpirvIreeHalFeatureRows[] = {
        LOOMC_SPIRV_IREE_HAL_GENERAL_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_ENABLE_COOPERATIVE_MATRIX,
            COOPERATIVE_MATRIX_KHR, "cooperative_matrix_khr"),
        LOOMC_SPIRV_IREE_HAL_GENERAL_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_8BIT_ACCESS,
            STORAGE_BUFFER_8BIT_ACCESS, "storage_buffer_8bit_access"),
        LOOMC_SPIRV_IREE_HAL_GENERAL_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_16BIT_ACCESS,
            STORAGE_BUFFER_16BIT_ACCESS, "storage_buffer_16bit_access"),
        LOOMC_SPIRV_IREE_HAL_GENERAL_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16, FLOAT16,
            "shader_float16"),
        LOOMC_SPIRV_IREE_HAL_GENERAL_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT64, FLOAT64,
            "shader_float64"),
        LOOMC_SPIRV_IREE_HAL_GENERAL_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT8, INT8, "shader_int8"),
        LOOMC_SPIRV_IREE_HAL_GENERAL_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT16, INT16, "shader_int16"),
        LOOMC_SPIRV_IREE_HAL_GENERAL_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_SUBGROUP_EXTENDED_TYPES,
            SUBGROUP_EXTENDED_TYPES, "shader_subgroup_extended_types"),
        LOOMC_SPIRV_IREE_HAL_GENERAL_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_TYPE,
            BFLOAT16_TYPE_KHR, "shader_bfloat16_type"),
        LOOMC_SPIRV_IREE_HAL_GENERAL_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_TYPE |
                IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_DOT_PRODUCT,
            BFLOAT16_DOT_PRODUCT_KHR, "shader_bfloat16_dot_product"),
        LOOMC_SPIRV_IREE_HAL_GENERAL_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_ENABLE_COOPERATIVE_MATRIX |
                IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_TYPE |
                IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_COOPERATIVE_MATRIX,
            BFLOAT16_COOPERATIVE_MATRIX_KHR,
            "shader_bfloat16_cooperative_matrix"),
        LOOMC_SPIRV_IREE_HAL_GENERAL_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL |
                IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE,
            VULKAN_MEMORY_MODEL_DEVICE_SCOPE,
            "vulkan_memory_model_device_scope"),
        LOOMC_SPIRV_IREE_HAL_ATOMIC_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT64,
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_INT64,
            STORAGE_BUFFER_INT64_ATOMICS, "shader_buffer_int64_atomics"),
        LOOMC_SPIRV_IREE_HAL_ATOMIC_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT64,
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_INT64,
            WORKGROUP_INT64_ATOMICS, "shader_shared_int64_atomics"),
        LOOMC_SPIRV_IREE_HAL_ATOMIC_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16,
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT16,
            STORAGE_BUFFER_FLOAT16_ATOMICS, "shader_buffer_float16_atomics"),
        LOOMC_SPIRV_IREE_HAL_ATOMIC_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16,
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT16,
            WORKGROUP_FLOAT16_ATOMICS, "shader_shared_float16_atomics"),
        LOOMC_SPIRV_IREE_HAL_ATOMIC_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16,
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT16_ADD,
            STORAGE_BUFFER_FLOAT16_ATOMIC_ADD,
            "shader_buffer_float16_atomic_add"),
        LOOMC_SPIRV_IREE_HAL_ATOMIC_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16,
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT16_ADD,
            WORKGROUP_FLOAT16_ATOMIC_ADD, "shader_shared_float16_atomic_add"),
        LOOMC_SPIRV_IREE_HAL_ATOMIC_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_NONE,
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT32,
            STORAGE_BUFFER_FLOAT32_ATOMICS, "shader_buffer_float32_atomics"),
        LOOMC_SPIRV_IREE_HAL_ATOMIC_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_NONE,
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT32,
            WORKGROUP_FLOAT32_ATOMICS, "shader_shared_float32_atomics"),
        LOOMC_SPIRV_IREE_HAL_ATOMIC_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_NONE,
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT32_ADD,
            STORAGE_BUFFER_FLOAT32_ATOMIC_ADD,
            "shader_buffer_float32_atomic_add"),
        LOOMC_SPIRV_IREE_HAL_ATOMIC_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_NONE,
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT32_ADD,
            WORKGROUP_FLOAT32_ATOMIC_ADD, "shader_shared_float32_atomic_add"),
        LOOMC_SPIRV_IREE_HAL_ATOMIC_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT64,
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT64,
            STORAGE_BUFFER_FLOAT64_ATOMICS, "shader_buffer_float64_atomics"),
        LOOMC_SPIRV_IREE_HAL_ATOMIC_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT64,
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT64,
            WORKGROUP_FLOAT64_ATOMICS, "shader_shared_float64_atomics"),
        LOOMC_SPIRV_IREE_HAL_ATOMIC_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT64,
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT64_ADD,
            STORAGE_BUFFER_FLOAT64_ATOMIC_ADD,
            "shader_buffer_float64_atomic_add"),
        LOOMC_SPIRV_IREE_HAL_ATOMIC_FEATURE_ROW(
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT64,
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT64_ADD,
            WORKGROUP_FLOAT64_ATOMIC_ADD, "shader_shared_float64_atomic_add"),
        {
            .required_device_flags =
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT32_DENORM_PRESERVE,
            .feature = LOOMC_SPIRV_FEATURE_FLOAT32_DENORM_PRESERVE,
            .provenance = "iree-hal:vulkan.device.float32_denorm_preserve",
        },
        {
            .required_device_flags =
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT16_64_DENORM_PRESERVE,
            .feature = LOOMC_SPIRV_FEATURE_FLOAT16_64_DENORM_PRESERVE,
            .provenance = "iree-hal:vulkan.device.float16_64_denorm_preserve",
        },
        {
            .required_device_flags =
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT16_64_ROUNDING_RTE,
            .feature = LOOMC_SPIRV_FEATURE_FLOAT16_64_ROUNDING_RTE,
            .provenance = "iree-hal:vulkan.device.float16_64_rounding_rte",
        },
        {
            .required_device_flags =
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT32_ROUNDING_RTE,
            .feature = LOOMC_SPIRV_FEATURE_FLOAT32_ROUNDING_RTE,
            .provenance = "iree-hal:vulkan.device.float32_rounding_rte",
        },
        {
            .required_device_flags =
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT16_SIGNED_ZERO_INF_NAN_PRESERVE,
            .feature =
                LOOMC_SPIRV_FEATURE_FLOAT16_SIGNED_ZERO_INF_NAN_PRESERVE,
            .provenance =
                "iree-hal:vulkan.device.float16_signed_zero_inf_nan_preserve",
        },
        {
            .required_device_flags =
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT32_SIGNED_ZERO_INF_NAN_PRESERVE,
            .feature =
                LOOMC_SPIRV_FEATURE_FLOAT32_SIGNED_ZERO_INF_NAN_PRESERVE,
            .provenance =
                "iree-hal:vulkan.device.float32_signed_zero_inf_nan_preserve",
        },
        {
            .required_device_flags =
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT64_SIGNED_ZERO_INF_NAN_PRESERVE,
            .feature =
                LOOMC_SPIRV_FEATURE_FLOAT64_SIGNED_ZERO_INF_NAN_PRESERVE,
            .provenance =
                "iree-hal:vulkan.device.float64_signed_zero_inf_nan_preserve",
        },
        {
            .required_subgroup_operations =
                LOOMC_SPIRV_IREE_HAL_VULKAN_SUBGROUP_FEATURE_BALLOT_BIT,
            .feature = LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_BALLOT,
            .provenance =
                "iree-hal:vulkan.device.subgroup_supported_operations",
        },
        {
            .required_subgroup_operations =
                LOOMC_SPIRV_IREE_HAL_VULKAN_SUBGROUP_FEATURE_VOTE_BIT,
            .feature = LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_VOTE,
            .provenance =
                "iree-hal:vulkan.device.subgroup_supported_operations",
        },
        {
            .required_subgroup_operations =
                LOOMC_SPIRV_IREE_HAL_VULKAN_SUBGROUP_FEATURE_ARITHMETIC_BIT,
            .feature = LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_ARITHMETIC,
            .provenance =
                "iree-hal:vulkan.device.subgroup_supported_operations",
        },
        {
            .required_subgroup_operations =
                LOOMC_SPIRV_IREE_HAL_VULKAN_SUBGROUP_FEATURE_SHUFFLE_BIT,
            .feature = LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_SHUFFLE,
            .provenance =
                "iree-hal:vulkan.device.subgroup_supported_operations",
        },
        {
            .required_subgroup_operations =
                LOOMC_SPIRV_IREE_HAL_VULKAN_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT,
            .feature = LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_SHUFFLE_RELATIVE,
            .provenance =
                "iree-hal:vulkan.device.subgroup_supported_operations",
        },
        {
            .required_subgroup_operations =
                LOOMC_SPIRV_IREE_HAL_VULKAN_SUBGROUP_FEATURE_CLUSTERED_BIT,
            .feature = LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_CLUSTERED,
            .provenance =
                "iree-hal:vulkan.device.subgroup_supported_operations",
        },
};
// clang-format on

#undef LOOMC_SPIRV_IREE_HAL_ATOMIC_FEATURE_ROW
#undef LOOMC_SPIRV_IREE_HAL_GENERAL_FEATURE_ROW

typedef struct loomc_spirv_iree_hal_profile_facts_t {
  // Feature facts indexed by public feature ID; unobserved slots stay unknown.
  loomc_spirv_feature_fact_t feature_facts[LOOMC_SPIRV_FEATURE_COUNT];

  // Numeric limits indexed by public limit ID.
  loomc_spirv_limit_fact_t limit_facts[LOOMC_SPIRV_LIMIT_COUNT];

  // Environment facts indexed by public environment ID.
  loomc_spirv_environment_fact_t
      environment_facts[LOOMC_SPIRV_ENVIRONMENT_COUNT];
} loomc_spirv_iree_hal_profile_facts_t;

static loomc_status_t loomc_spirv_iree_hal_validate_string_view(
    loomc_string_view_t value) {
  if (value.data == NULL && value.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_iree_hal_validate_options(
    const loomc_spirv_iree_hal_target_options_t* options) {
  if (options == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "SPIR-V IREE HAL options must not be NULL");
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_SPIRV_IREE_HAL_TARGET_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "SPIR-V IREE HAL options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "SPIR-V IREE HAL options structure_size is too small");
  }
  if (options->next != NULL) {
    return loomc_make_status(LOOMC_STATUS_UNIMPLEMENTED,
                             "SPIR-V IREE HAL option extensions are not "
                             "supported");
  }
  LOOMC_RETURN_IF_ERROR(
      loomc_spirv_iree_hal_validate_string_view(options->identifier));
  if (options->device == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "SPIR-V IREE HAL options require a device");
  }
  return loomc_ok_status();
}

static void loomc_spirv_iree_hal_set_bool_feature(
    loomc_spirv_iree_hal_profile_facts_t* facts, bool value,
    loomc_spirv_feature_t feature, loomc_string_view_t provenance) {
  facts->feature_facts[feature].state =
      value ? LOOMC_TARGET_FACT_STATE_TRUE : LOOMC_TARGET_FACT_STATE_FALSE;
  facts->feature_facts[feature].provenance = provenance;
}

static void loomc_spirv_iree_hal_set_limit_fact(
    loomc_spirv_iree_hal_profile_facts_t* facts, loomc_spirv_limit_t limit,
    uint64_t value, loomc_string_view_t provenance) {
  facts->limit_facts[limit] = (loomc_spirv_limit_fact_t){
      .limit = limit,
      .state = LOOMC_TARGET_FACT_STATE_TRUE,
      .value = value,
      .provenance = provenance,
  };
}

static void loomc_spirv_iree_hal_set_environment_fact(
    loomc_spirv_iree_hal_profile_facts_t* facts,
    loomc_spirv_environment_t environment, uint64_t value,
    loomc_string_view_t provenance) {
  facts->environment_facts[environment] = (loomc_spirv_environment_fact_t){
      .environment = environment,
      .state = LOOMC_TARGET_FACT_STATE_TRUE,
      .value = value,
      .provenance = provenance,
  };
}

static loomc_status_t loomc_spirv_iree_hal_fail_status(loomc_result_t* result,
                                                       loomc_status_t status) {
  return loomc_result_fail_status_diagnostic_consume(
      result, NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
      loomc_make_cstring_view("SPIRV/IREE_HAL"), status);
}

static loomc_status_t loomc_spirv_iree_hal_fail_cstring(loomc_result_t* result,
                                                        const char* message) {
  return loomc_spirv_iree_hal_fail_status(
      result, loomc_make_status(LOOMC_STATUS_UNAVAILABLE, message));
}

static uint32_t loomc_spirv_iree_hal_vulkan_api_version_major(
    uint32_t api_version) {
  return api_version >> 22;
}

static uint32_t loomc_spirv_iree_hal_vulkan_api_version_minor(
    uint32_t api_version) {
  return (api_version >> 12) & 0x3FFu;
}

static bool loomc_spirv_iree_hal_vulkan_feature_enabled(
    const iree_hal_vulkan_device_spec_t* spec,
    iree_hal_vulkan_general_features_t feature) {
  return iree_all_bits_set(spec->enabled_features.general, feature);
}

static uint32_t loomc_spirv_iree_hal_scalar_component_type(
    loom_spirv_scalar_type_t scalar_type) {
  switch (scalar_type) {
    case LOOM_SPIRV_SCALAR_TYPE_F16:
      return LOOMC_SPIRV_COMPONENT_TYPE_FLOAT16_NV;
    case LOOM_SPIRV_SCALAR_TYPE_F32:
      return LOOMC_SPIRV_COMPONENT_TYPE_FLOAT32_NV;
    case LOOM_SPIRV_SCALAR_TYPE_F64:
      return LOOMC_SPIRV_COMPONENT_TYPE_FLOAT64_NV;
    case LOOM_SPIRV_SCALAR_TYPE_S8:
      return LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT8_NV;
    case LOOM_SPIRV_SCALAR_TYPE_S16:
      return LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT16_NV;
    case LOOM_SPIRV_SCALAR_TYPE_S32:
      return LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT32_NV;
    case LOOM_SPIRV_SCALAR_TYPE_S64:
      return LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT64_NV;
    case LOOM_SPIRV_SCALAR_TYPE_U8:
      return LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT8_NV;
    case LOOM_SPIRV_SCALAR_TYPE_U16:
      return LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT16_NV;
    case LOOM_SPIRV_SCALAR_TYPE_U32:
      return LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT32_NV;
    case LOOM_SPIRV_SCALAR_TYPE_U64:
      return LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT64_NV;
    case LOOM_SPIRV_SCALAR_TYPE_BF16:
      return LOOMC_SPIRV_IREE_HAL_VULKAN_COMPONENT_TYPE_BFLOAT16_KHR;
    case LOOM_SPIRV_SCALAR_TYPE_UNKNOWN:
      break;
  }
  IREE_CHECK_UNREACHABLE("validated cooperative matrix scalar type");
}

static bool loomc_spirv_iree_hal_matrix_row_is_supported(
    const loom_spirv_cooperative_matrix_property_t* row,
    const iree_hal_vulkan_device_spec_t* vulkan_spec) {
  const iree_hal_vulkan_cooperative_matrix_property_t requirement = {
      .m_size = row->m_size,
      .n_size = row->n_size,
      .k_size = row->k_size,
      .a_type = loomc_spirv_iree_hal_scalar_component_type(row->lhs_type),
      .b_type = loomc_spirv_iree_hal_scalar_component_type(row->rhs_type),
      .c_type =
          loomc_spirv_iree_hal_scalar_component_type(row->accumulator_type),
      .result_type =
          loomc_spirv_iree_hal_scalar_component_type(row->result_type),
      .saturating_accumulation = iree_any_bit_set(
          row->operand_flags,
          LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_SATURATING_ACCUMULATION),
      .scope = (uint32_t)row->scope,
  };
  return iree_hal_vulkan_device_spec_supports_cooperative_matrix(vulkan_spec,
                                                                 &requirement);
}

static loomc_status_t loomc_spirv_iree_hal_collect_unavailable_matrix_rows(
    const iree_hal_vulkan_device_spec_t* vulkan_spec,
    loomc_allocator_t allocator,
    loomc_spirv_cooperative_matrix_row_t** out_rows,
    loomc_host_size_t* out_row_count) {
  *out_rows = NULL;
  *out_row_count = 0;
  iree_host_size_t model_row_count = 0;
  const loom_spirv_cooperative_matrix_property_t* model_rows =
      loom_spirv_cooperative_matrix_model_properties(&model_row_count);
  loomc_spirv_cooperative_matrix_row_t* rows = NULL;
  loomc_host_size_t row_count = 0;
  loomc_status_t status = loomc_ok_status();
  for (loomc_host_size_t i = 0;
       i < model_row_count && loomc_status_is_ok(status); ++i) {
    if (loomc_spirv_iree_hal_matrix_row_is_supported(&model_rows[i],
                                                     vulkan_spec)) {
      continue;
    }
    loomc_spirv_cooperative_matrix_row_t row;
    loomc_spirv_cooperative_matrix_row_from_loom(&model_rows[i], &row);
    row.state = LOOMC_TARGET_FACT_STATE_FALSE;
    row.provenance = loomc_make_cstring_view(
        "iree-hal:vulkan.device.cooperative_matrix_properties");
    if (rows == NULL) {
      // The first unsupported row bounds the remaining output capacity. Keep
      // each query result instead of repeating discovery to size the array.
      status = loomc_allocator_malloc(
          allocator, (model_row_count - i) * sizeof(*rows), (void**)&rows);
    }
    if (loomc_status_is_ok(status)) {
      rows[row_count++] = row;
    }
  }
  if (loomc_status_is_ok(status)) {
    *out_rows = rows;
    *out_row_count = row_count;
  } else {
    loomc_allocator_free(allocator, rows);
  }
  return status;
}

static loomc_status_t loomc_spirv_iree_hal_decode_vulkan_spec(
    iree_hal_device_t* device, iree_hal_vulkan_device_spec_t* out_spec) {
  const iree_hal_device_spec_t* device_spec = iree_hal_device_spec(device);
  if (device_spec == NULL) {
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "IREE HAL device does not expose immutable device facts"));
  }
  const iree_hal_device_spec_facet_t* vulkan_facet =
      iree_hal_vulkan_device_spec_find_facet(device_spec);
  if (vulkan_facet == NULL) {
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "IREE HAL device spec does not expose Vulkan device facts"));
  }
  return loomc_status_from_iree(
      iree_hal_vulkan_device_spec_decode_facet(vulkan_facet, out_spec));
}

static loomc_status_t loomc_spirv_iree_hal_load_device_specs(
    iree_hal_device_t* device,
    const iree_hal_device_dispatch_spec_t** out_dispatch,
    iree_hal_vulkan_device_spec_t* out_vulkan_spec) {
  const iree_hal_device_spec_t* device_spec = iree_hal_device_spec(device);
  if (device_spec == NULL) {
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "IREE HAL device does not expose immutable device facts"));
  }
  const iree_hal_device_dispatch_spec_t* dispatch =
      iree_hal_device_spec_dispatch(device_spec);
  if (dispatch == NULL) {
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "IREE HAL device spec does not expose dispatch capability facts"));
  }
  const iree_hal_device_spec_facet_t* vulkan_facet =
      iree_hal_vulkan_device_spec_find_facet(device_spec);
  if (vulkan_facet == NULL) {
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "IREE HAL device spec does not expose Vulkan device facts"));
  }
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(
      iree_hal_vulkan_device_spec_decode_facet(vulkan_facet, out_vulkan_spec)));
  *out_dispatch = dispatch;
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_iree_hal_query_facts(
    const loomc_spirv_iree_hal_target_options_t* options,
    loomc_spirv_iree_hal_profile_facts_t* out_facts,
    iree_hal_vulkan_device_spec_t* out_vulkan_spec, loomc_result_t* result,
    const iree_hal_executable_target_t** out_executable_target) {
  *out_facts = (loomc_spirv_iree_hal_profile_facts_t){0};
  for (uint32_t i = 1; i < LOOMC_SPIRV_FEATURE_COUNT; ++i) {
    out_facts->feature_facts[i].feature = (loomc_spirv_feature_t)i;
  }
  for (uint32_t i = 1; i < LOOMC_SPIRV_LIMIT_COUNT; ++i) {
    out_facts->limit_facts[i].limit = (loomc_spirv_limit_t)i;
  }
  for (uint32_t i = 1; i < LOOMC_SPIRV_ENVIRONMENT_COUNT; ++i) {
    out_facts->environment_facts[i].environment = (loomc_spirv_environment_t)i;
  }
  *out_vulkan_spec = (iree_hal_vulkan_device_spec_t){0};
  *out_executable_target = NULL;
  const iree_hal_device_spec_t* device_spec =
      iree_hal_device_spec(options->device);
  if (device_spec == NULL) {
    return loomc_spirv_iree_hal_fail_cstring(
        result, "IREE HAL device does not expose immutable device facts");
  }
  const iree_hal_device_identity_spec_t* identity =
      iree_hal_device_spec_identity(device_spec);
  const iree_hal_executable_target_selection_t target_selection = {
      .family = IREE_SV("spirv"),
      .target_key = IREE_SV("vulkan1.3+bda"),
      .kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_GENERIC,
      .physical_device_affinity = options->physical_device_affinity,
  };
  const iree_hal_executable_target_selection_result_t target_result =
      iree_hal_device_spec_select_executable_target(device_spec,
                                                    &target_selection);
  if (target_result.outcome ==
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH) {
    return loomc_spirv_iree_hal_fail_cstring(
        result,
        "IREE HAL device does not support the vulkan1.3+bda SPIR-V target");
  } else if (target_result.outcome ==
             IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS) {
    return loomc_spirv_iree_hal_fail_cstring(
        result,
        "IREE HAL device reports ambiguous vulkan1.3+bda SPIR-V targets");
  }
  *out_executable_target = target_result.target;

  const iree_hal_device_dispatch_spec_t* dispatch = NULL;
  iree_hal_vulkan_device_spec_t vulkan_spec = {0};
  loomc_status_t status = loomc_spirv_iree_hal_load_device_specs(
      options->device, &dispatch, &vulkan_spec);
  if (!loomc_status_is_ok(status)) {
    return loomc_spirv_iree_hal_fail_status(result, status);
  }
  if (dispatch->subgroup.default_size == 0 ||
      dispatch->subgroup.minimum_size == 0 ||
      dispatch->subgroup.maximum_size == 0 ||
      (dispatch->subgroup.minimum_size != dispatch->subgroup.maximum_size &&
       vulkan_spec.maximum_workgroup_subgroup_count == 0) ||
      dispatch->launch.maximum_workgroup_invocations == 0 ||
      dispatch->launch.maximum_workgroup_size[0] == 0 ||
      dispatch->launch.maximum_workgroup_size[1] == 0 ||
      dispatch->launch.maximum_workgroup_size[2] == 0 ||
      dispatch->launch.maximum_workgroup_count[0] == 0 ||
      dispatch->launch.maximum_workgroup_count[1] == 0 ||
      dispatch->launch.maximum_workgroup_count[2] == 0 ||
      dispatch->execution.maximum_workgroup_local_memory_size == 0) {
    return loomc_spirv_iree_hal_fail_cstring(
        result,
        "IREE HAL device spec does not expose complete dispatch capability "
        "facts");
  }
  if (vulkan_spec.api_version < LOOMC_SPIRV_IREE_HAL_VULKAN_API_VERSION_1_3) {
    return loomc_spirv_iree_hal_fail_cstring(
        result, "IREE HAL SPIR-V profile requires Vulkan API version 1.3");
  }

  const bool buffer_device_address =
      loomc_spirv_iree_hal_vulkan_feature_enabled(
          &vulkan_spec, IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES);
  if (!buffer_device_address) {
    return loomc_spirv_iree_hal_fail_cstring(
        result, "IREE HAL SPIR-V profile requires buffer_device_address");
  }

  const bool shader_int64 = loomc_spirv_iree_hal_vulkan_feature_enabled(
      &vulkan_spec, IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT64);
  if (!shader_int64) {
    return loomc_spirv_iree_hal_fail_cstring(
        result, "IREE HAL SPIR-V profile requires shader_int64");
  }

  const loomc_string_view_t api_provenance =
      loomc_make_cstring_view("iree-hal:vulkan.device.api_version");
  loomc_spirv_iree_hal_set_bool_feature(
      out_facts, true, LOOMC_SPIRV_FEATURE_VULKAN_SHADER, api_provenance);
  loomc_spirv_iree_hal_set_bool_feature(
      out_facts, true, LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM, api_provenance);
  loomc_spirv_iree_hal_set_bool_feature(
      out_facts,
      identity->vendor_id != LOOMC_SPIRV_IREE_HAL_VULKAN_AMD_VENDOR_ID,
      LOOMC_SPIRV_FEATURE_FLOAT64_TO_FLOAT16_CONVERSION,
      loomc_make_cstring_view("iree-hal:identity.vendor_id"));
  loomc_spirv_iree_hal_set_environment_fact(
      out_facts, LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION,
      loomc_spirv_max_version_from_vulkan_api_version(
          loomc_spirv_iree_hal_vulkan_api_version_major(
              vulkan_spec.api_version),
          loomc_spirv_iree_hal_vulkan_api_version_minor(
              vulkan_spec.api_version)),
      api_provenance);

  loomc_spirv_iree_hal_set_bool_feature(
      out_facts, buffer_device_address,
      LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER,
      loomc_make_cstring_view("iree-hal:vulkan.feature.buffer_device_address"));
  loomc_spirv_iree_hal_set_bool_feature(
      out_facts, shader_int64, LOOMC_SPIRV_FEATURE_INT64,
      loomc_make_cstring_view("iree-hal:vulkan.feature.shader_int64"));

  loomc_spirv_iree_hal_set_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_DEFAULT_SUBGROUP_SIZE,
      dispatch->subgroup.default_size,
      loomc_make_cstring_view("iree-hal:vulkan.device.subgroup_size"));
  loomc_spirv_iree_hal_set_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MIN_SUBGROUP_SIZE,
      dispatch->subgroup.minimum_size,
      loomc_make_cstring_view("iree-hal:vulkan.device.min_subgroup_size"));
  loomc_spirv_iree_hal_set_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MAX_SUBGROUP_SIZE,
      dispatch->subgroup.maximum_size,
      loomc_make_cstring_view("iree-hal:vulkan.device.max_subgroup_size"));
  if (vulkan_spec.maximum_workgroup_subgroup_count != 0) {
    loomc_spirv_iree_hal_set_limit_fact(
        out_facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SUBGROUP_COUNT,
        vulkan_spec.maximum_workgroup_subgroup_count,
        loomc_make_cstring_view(
            "iree-hal:vulkan.device.max_compute_workgroup_subgroups"));
  }
  loomc_spirv_iree_hal_set_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MAX_FLAT_WORKGROUP_SIZE,
      dispatch->launch.maximum_workgroup_invocations,
      loomc_make_cstring_view(
          "iree-hal:vulkan.device.max_compute_workgroup_invocations"));
  loomc_spirv_iree_hal_set_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_STORAGE_BYTES,
      iree_min(dispatch->execution.maximum_workgroup_local_memory_size,
               INT64_MAX),
      loomc_make_cstring_view(
          "iree-hal:vulkan.device.max_compute_shared_memory_size"));
  loomc_spirv_iree_hal_set_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X,
      dispatch->launch.maximum_workgroup_size[0],
      loomc_make_cstring_view(
          "iree-hal:vulkan.device.max_compute_workgroup_size_x"));
  loomc_spirv_iree_hal_set_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Y,
      dispatch->launch.maximum_workgroup_size[1],
      loomc_make_cstring_view(
          "iree-hal:vulkan.device.max_compute_workgroup_size_y"));
  loomc_spirv_iree_hal_set_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Z,
      dispatch->launch.maximum_workgroup_size[2],
      loomc_make_cstring_view(
          "iree-hal:vulkan.device.max_compute_workgroup_size_z"));
  loomc_spirv_iree_hal_set_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_X,
      dispatch->launch.maximum_workgroup_count[0],
      loomc_make_cstring_view(
          "iree-hal:vulkan.device.max_compute_workgroup_count_x"));
  loomc_spirv_iree_hal_set_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_Y,
      dispatch->launch.maximum_workgroup_count[1],
      loomc_make_cstring_view(
          "iree-hal:vulkan.device.max_compute_workgroup_count_y"));
  loomc_spirv_iree_hal_set_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_Z,
      dispatch->launch.maximum_workgroup_count[2],
      loomc_make_cstring_view(
          "iree-hal:vulkan.device.max_compute_workgroup_count_z"));

  const loomc_spirv_limit_t grid_limits[] = {LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_X,
                                             LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_Y,
                                             LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_Z};
  const char* grid_provenance[] = {"iree-hal:vulkan.device.max_grid_size_x",
                                   "iree-hal:vulkan.device.max_grid_size_y",
                                   "iree-hal:vulkan.device.max_grid_size_z"};
  uint64_t flat_grid_size = dispatch->launch.maximum_workgroup_invocations;
  for (uint32_t i = 0; i < 3; ++i) {
    const uint32_t workgroup_size =
        iree_min(dispatch->launch.maximum_workgroup_size[i],
                 dispatch->launch.maximum_workgroup_invocations);
    const uint32_t workgroup_count =
        dispatch->launch.maximum_workgroup_count[i];
    const uint64_t grid_size = (uint64_t)workgroup_size * workgroup_count;
    loomc_spirv_iree_hal_set_limit_fact(
        out_facts, grid_limits[i], iree_min(grid_size, UINT32_MAX),
        loomc_make_cstring_view(grid_provenance[i]));
    // Grid limits are upper bounds. Saturate at the compiler representation
    // limit when the device admits more workitems than it can represent.
    flat_grid_size = flat_grid_size > INT64_MAX / workgroup_count
                         ? INT64_MAX
                         : flat_grid_size * workgroup_count;
  }
  loomc_spirv_iree_hal_set_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MAX_FLAT_GRID_SIZE, flat_grid_size,
      loomc_make_cstring_view("iree-hal:vulkan.device.max_flat_grid_size"));

  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomcSpirvIreeHalFeatureRows); ++i) {
    const loomc_spirv_iree_hal_feature_row_t* row =
        &kLoomcSpirvIreeHalFeatureRows[i];
    const bool supported =
        iree_all_bits_set(vulkan_spec.enabled_features.general,
                          row->required_general_features) &&
        iree_all_bits_set(vulkan_spec.enabled_features.atomics,
                          row->required_atomic_features) &&
        iree_all_bits_set(vulkan_spec.flags, row->required_device_flags) &&
        iree_all_bits_set(vulkan_spec.subgroup_supported_operations,
                          row->required_subgroup_operations);
    loomc_spirv_iree_hal_set_bool_feature(
        out_facts, supported, row->feature,
        loomc_make_cstring_view(row->provenance));
  }
  *out_vulkan_spec = vulkan_spec;
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_iree_hal_validate_profile_requirements(
    const loomc_target_profile_t* target_profile,
    const loomc_spirv_iree_hal_profile_facts_t* facts,
    const iree_hal_vulkan_device_spec_t* vulkan_spec) {
  const loom_spirv_target_profile_t* profile = loom_spirv_target_profile_cast(
      loomc_target_profile_loom_target_profile(target_profile));
  if (profile == NULL) {
    return loomc_make_status(LOOMC_STATUS_UNAVAILABLE,
                             "IREE HAL Vulkan requires a SPIR-V profile");
  }

  const loom_spirv_feature_bits_t feature_bits =
      profile->base.target_bundle->config->contract_feature_bits;
  uint32_t minimum_version = 0;
  for (uint32_t i = 1; i < LOOMC_SPIRV_FEATURE_COUNT; ++i) {
    const loom_spirv_feature_atom_t atom =
        loomc_spirv_feature_atom_from_public((loomc_spirv_feature_t)i);
    if (!iree_any_bit_set(feature_bits, loom_spirv_feature_atom_bit(atom))) {
      continue;
    }
    const loom_spirv_feature_atom_descriptor_t* descriptor =
        loom_spirv_feature_atom_descriptor(atom);
    if (facts->feature_facts[i].state != LOOMC_TARGET_FACT_STATE_TRUE) {
      return loomc_status_from_iree(iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "forced SPIR-V profile requires device feature '%.*s'",
          (int)descriptor->name.size, descriptor->name.data));
    }
    minimum_version =
        iree_max(minimum_version, descriptor->minimum_spirv_version);
  }
  const uint64_t maximum_version =
      facts->environment_facts[LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION].value;
  if (minimum_version > maximum_version) {
    return loomc_status_from_iree(
        iree_make_status(IREE_STATUS_UNAVAILABLE,
                         "forced SPIR-V profile requires version 0x%08" PRIx32
                         ", but the device supports only 0x%08" PRIx64,
                         minimum_version, maximum_version));
  }

  const loom_target_snapshot_t* snapshot =
      profile->base.target_bundle->snapshot;
  const uint64_t requirements[LOOMC_SPIRV_LIMIT_COUNT] = {
      [LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X] = snapshot->max_workgroup_size.x,
      [LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Y] = snapshot->max_workgroup_size.y,
      [LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Z] = snapshot->max_workgroup_size.z,
      [LOOMC_SPIRV_LIMIT_MAX_FLAT_WORKGROUP_SIZE] =
          snapshot->max_flat_workgroup_size,
      [LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_STORAGE_BYTES] =
          snapshot->max_workgroup_storage_bytes,
      [LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE] = snapshot->subgroup_size,
      [LOOMC_SPIRV_LIMIT_DEFAULT_SUBGROUP_SIZE] =
          profile->subgroup.default_size,
      [LOOMC_SPIRV_LIMIT_MIN_SUBGROUP_SIZE] = profile->subgroup.minimum_size,
      [LOOMC_SPIRV_LIMIT_MAX_SUBGROUP_SIZE] = profile->subgroup.maximum_size,
      [LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SUBGROUP_COUNT] =
          profile->subgroup.maximum_workgroup_subgroup_count,
      [LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_X] =
          snapshot->max_workgroup_count.x,
      [LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_Y] =
          snapshot->max_workgroup_count.y,
      [LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_Z] =
          snapshot->max_workgroup_count.z,
      [LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_X] = snapshot->max_grid_size.x,
      [LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_Y] = snapshot->max_grid_size.y,
      [LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_Z] = snapshot->max_grid_size.z,
      [LOOMC_SPIRV_LIMIT_MAX_FLAT_GRID_SIZE] = snapshot->max_flat_grid_size,
  };
  for (uint32_t i = 1; i < LOOMC_SPIRV_LIMIT_COUNT; ++i) {
    const loomc_spirv_limit_fact_t* available = &facts->limit_facts[i];
    bool incompatible = requirements[i] > available->value;
    if (i == LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE ||
        i == LOOMC_SPIRV_LIMIT_MIN_SUBGROUP_SIZE ||
        i == LOOMC_SPIRV_LIMIT_MAX_SUBGROUP_SIZE) {
      // A fixed width or either endpoint must be selectable on this device.
      const uint64_t minimum =
          facts->limit_facts[LOOMC_SPIRV_LIMIT_MIN_SUBGROUP_SIZE].value;
      const uint64_t maximum =
          facts->limit_facts[LOOMC_SPIRV_LIMIT_MAX_SUBGROUP_SIZE].value;
      if (requirements[i] != 0 &&
          (requirements[i] < minimum || requirements[i] > maximum)) {
        return loomc_status_from_iree(iree_make_status(
            IREE_STATUS_UNAVAILABLE,
            "forced SPIR-V profile subgroup width %" PRIu64
            " is outside the device's selectable range [%" PRIu64 ", %" PRIu64
            "]",
            requirements[i], minimum, maximum));
      }
      continue;
    } else if (i == LOOMC_SPIRV_LIMIT_DEFAULT_SUBGROUP_SIZE) {
      incompatible =
          requirements[i] != 0 && requirements[i] != available->value;
    }
    if (incompatible) {
      return loomc_status_from_iree(
          iree_make_status(IREE_STATUS_UNAVAILABLE,
                           "forced SPIR-V profile limit %" PRIu64
                           " is incompatible with '%.*s' = %" PRIu64,
                           requirements[i], (int)available->provenance.size,
                           available->provenance.data, available->value));
    }
  }

  if (iree_any_bit_set(feature_bits,
                       LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR)) {
    const loom_spirv_cooperative_property_set_t* properties =
        profile->cooperative_properties;
    loom_spirv_cooperative_property_set_t modeled_properties;
    if (properties == NULL) {
      const loom_spirv_feature_set_t features = {.atom_bits = feature_bits};
      loom_spirv_cooperative_property_set_prepare(&features,
                                                  &modeled_properties);
      properties = &modeled_properties;
    }
    for (uint16_t i = 0; i < properties->matrix_property_count; ++i) {
      const loom_spirv_cooperative_matrix_property_t* property =
          &properties->matrix_properties[i];
      if (!iree_all_bits_set(feature_bits, property->required_feature_bits)) {
        continue;
      }
      if (!loomc_spirv_iree_hal_matrix_row_is_supported(property,
                                                        vulkan_spec)) {
        return loomc_status_from_iree(iree_make_status(
            IREE_STATUS_UNAVAILABLE,
            "forced SPIR-V profile requires cooperative matrix row '%.*s'",
            (int)property->name.size, property->name.data));
      }
    }
  }
  return loomc_ok_status();
}

loomc_status_t loomc_target_select_spirv_iree_hal(
    loomc_target_environment_t* target_environment,
    const loomc_spirv_iree_hal_target_options_t* options,
    loomc_allocator_t allocator,
    loomc_iree_hal_target_selection_t* out_selection,
    loomc_result_t** out_result) {
  if (out_selection == NULL || out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_selection and out_result must not be NULL");
  }
  *out_selection = (loomc_iree_hal_target_selection_t){0};
  *out_result = NULL;
  if (target_environment == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "target_environment must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_validate_options(options));
  if (options->target_profile != NULL) {
    LOOMC_RETURN_IF_ERROR(loomc_target_profile_validate_environment(
        options->target_profile, target_environment));
  }

  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_result_create(
      LOOMC_STATUS_OK, LOOMC_SOURCE_RETENTION_EXACT, allocator, &result));
  loomc_spirv_iree_hal_profile_facts_t facts = {0};
  iree_hal_vulkan_device_spec_t vulkan_spec = {0};
  const iree_hal_executable_target_t* executable_target = NULL;
  loomc_status_t status = loomc_spirv_iree_hal_query_facts(
      options, &facts, &vulkan_spec, result, &executable_target);
  loomc_target_profile_t* target_profile = NULL;
  loomc_spirv_cooperative_matrix_row_t* matrix_rows = NULL;
  loomc_host_size_t matrix_row_count = 0;
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    if (options->target_profile != NULL) {
      status = loomc_spirv_iree_hal_validate_profile_requirements(
          options->target_profile, &facts, &vulkan_spec);
      if (!loomc_status_is_ok(status)) {
        status = loomc_spirv_iree_hal_fail_status(result, status);
      } else {
        loomc_target_profile_retain(options->target_profile);
        target_profile = options->target_profile;
      }
    } else {
      if (loomc_spirv_iree_hal_vulkan_feature_enabled(
              &vulkan_spec,
              IREE_HAL_VULKAN_FEATURE_ENABLE_COOPERATIVE_MATRIX)) {
        status = loomc_spirv_iree_hal_collect_unavailable_matrix_rows(
            &vulkan_spec, allocator, &matrix_rows, &matrix_row_count);
      }
    }
    if (loomc_status_is_ok(status) && options->target_profile == NULL) {
      loomc_result_release(result);
      result = NULL;
      loomc_spirv_profile_options_t profile_options = {
          /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
          /*.structure_size=*/sizeof(profile_options),
          /*.next=*/NULL,
          /*.identifier=*/options->identifier,
          /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_VULKAN_1_3_BDA,
          /*.feature_facts=*/facts.feature_facts + 1,
          /*.feature_fact_count=*/LOOMC_SPIRV_FEATURE_COUNT - 1,
          /*.limit_facts=*/facts.limit_facts + 1,
          /*.limit_fact_count=*/LOOMC_SPIRV_LIMIT_COUNT - 1,
          /*.environment_facts=*/facts.environment_facts + 1,
          /*.environment_fact_count=*/LOOMC_SPIRV_ENVIRONMENT_COUNT - 1,
          /*.cooperative_matrix_rows=*/matrix_rows,
          /*.cooperative_matrix_row_count=*/matrix_row_count,
      };
      status = loomc_target_profile_create_spirv(target_environment,
                                                 &profile_options, allocator,
                                                 &target_profile, &result);
    }
  }
  if (loomc_status_is_ok(status)) {
    if (loomc_result_succeeded(result)) {
      *out_selection = (loomc_iree_hal_target_selection_t){
          .target_profile = target_profile,
          .executable_target = executable_target,
      };
      target_profile = NULL;
    }
    *out_result = result;
    result = NULL;
  }
  loomc_allocator_free(allocator, matrix_rows);
  loomc_target_profile_release(target_profile);
  loomc_result_release(result);
  return status;
}

static loomc_status_t loomc_spirv_iree_hal_device_is_supported(
    iree_hal_device_t* device, bool* out_supported) {
  *out_supported = false;
  iree_hal_vulkan_device_spec_t vulkan_spec = {0};
  loomc_status_t status =
      loomc_spirv_iree_hal_decode_vulkan_spec(device, &vulkan_spec);
  if (!loomc_status_is_ok(status)) {
    const iree_status_code_t code =
        iree_status_code(iree_status_from_loomc(status));
    if (code == IREE_STATUS_UNAVAILABLE) {
      loomc_status_free(status);
      return loomc_ok_status();
    }
    return status;
  }
  if (vulkan_spec.api_version < LOOMC_SPIRV_IREE_HAL_VULKAN_API_VERSION_1_3) {
    return loomc_ok_status();
  }
  *out_supported = true;
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_iree_hal_provider_select_target(
    void* user_data, loomc_target_environment_t* target_environment,
    const loomc_iree_hal_target_options_t* options, loomc_allocator_t allocator,
    bool* out_supported, loomc_iree_hal_target_selection_t* out_selection,
    loomc_result_t** out_result) {
  (void)user_data;
  *out_supported = false;
  *out_selection = (loomc_iree_hal_target_selection_t){0};
  *out_result = NULL;
  if (options->target_profile != NULL) {
    *out_supported =
        loom_spirv_target_profile_cast(loomc_target_profile_loom_target_profile(
            options->target_profile)) != NULL;
  } else {
    LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_device_is_supported(
        options->device, out_supported));
  }
  if (!*out_supported) {
    return loomc_ok_status();
  }

  loomc_spirv_iree_hal_target_options_t spirv_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_IREE_HAL_TARGET_OPTIONS,
      /*.structure_size=*/sizeof(spirv_options),
      /*.next=*/options->next,
      /*.identifier=*/options->identifier,
      /*.device=*/options->device,
      /*.physical_device_affinity=*/options->physical_device_affinity,
      /*.target_profile=*/options->target_profile,
  };
  return loomc_target_select_spirv_iree_hal(
      target_environment, &spirv_options, allocator, out_selection, out_result);
}

const loomc_iree_hal_target_provider_t* loomc_spirv_iree_hal_target_provider(
    void) {
  static const loomc_iree_hal_target_provider_t provider = {
      /*.name=*/{"spirv.iree_hal.vulkan", 21},
      /*.user_data=*/NULL,
      /*.select_target=*/loomc_spirv_iree_hal_provider_select_target,
  };
  return &provider;
}
