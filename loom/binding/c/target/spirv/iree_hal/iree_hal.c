// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/spirv/iree_hal.h"

#include <stdint.h>

#include "diagnostic.h"
#include "iree/hal/drivers/vulkan/device_spec.h"
#include "loomc/iree.h"
#include "result.h"

#define LOOMC_SPIRV_IREE_HAL_VULKAN_API_VERSION(major, minor, patch) \
  ((((uint32_t)(major)) << 22) | (((uint32_t)(minor)) << 12) |       \
   ((uint32_t)(patch)))

enum {
  LOOMC_SPIRV_IREE_HAL_FEATURE_FACT_CAPACITY = 24,
  LOOMC_SPIRV_IREE_HAL_LIMIT_FACT_CAPACITY = 16,
  LOOMC_SPIRV_IREE_HAL_ENVIRONMENT_FACT_CAPACITY = 4,
  LOOMC_SPIRV_IREE_HAL_VULKAN_API_VERSION_1_3 =
      LOOMC_SPIRV_IREE_HAL_VULKAN_API_VERSION(1, 3, 0),
};

typedef struct loomc_spirv_iree_hal_profile_facts_t {
  // Feature facts normalized from HAL device queries.
  loomc_spirv_feature_fact_t
      feature_facts[LOOMC_SPIRV_IREE_HAL_FEATURE_FACT_CAPACITY];

  // Number of entries in feature_facts.
  loomc_host_size_t feature_fact_count;

  // Numeric limit facts normalized from HAL device queries.
  loomc_spirv_limit_fact_t
      limit_facts[LOOMC_SPIRV_IREE_HAL_LIMIT_FACT_CAPACITY];

  // Number of entries in limit_facts.
  loomc_host_size_t limit_fact_count;

  // Environment facts normalized from HAL device queries.
  loomc_spirv_environment_fact_t
      environment_facts[LOOMC_SPIRV_IREE_HAL_ENVIRONMENT_FACT_CAPACITY];

  // Number of entries in environment_facts.
  loomc_host_size_t environment_fact_count;
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
    const loomc_spirv_iree_hal_profile_options_t* options) {
  if (options == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "SPIR-V IREE HAL options must not be NULL");
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_SPIRV_IREE_HAL_PROFILE_OPTIONS) {
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

static loomc_status_t loomc_spirv_iree_hal_add_feature_fact(
    loomc_spirv_iree_hal_profile_facts_t* facts, loomc_spirv_feature_t feature,
    loomc_target_fact_state_t state, loomc_string_view_t provenance) {
  if (state == LOOMC_TARGET_FACT_STATE_UNKNOWN) {
    return loomc_ok_status();
  }
  if (facts->feature_fact_count >= LOOMC_SPIRV_IREE_HAL_FEATURE_FACT_CAPACITY) {
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "too many SPIR-V IREE HAL feature facts");
  }
  facts->feature_facts[facts->feature_fact_count++] =
      (loomc_spirv_feature_fact_t){
          .feature = feature,
          .state = state,
          .provenance = provenance,
      };
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_iree_hal_add_bool_feature(
    loomc_spirv_iree_hal_profile_facts_t* facts, bool value,
    loomc_spirv_feature_t feature, loomc_string_view_t provenance) {
  return loomc_spirv_iree_hal_add_feature_fact(
      facts, feature,
      value ? LOOMC_TARGET_FACT_STATE_TRUE : LOOMC_TARGET_FACT_STATE_FALSE,
      provenance);
}

static loomc_status_t loomc_spirv_iree_hal_add_optional_feature(
    loomc_spirv_iree_hal_profile_facts_t* facts, bool known, bool value,
    loomc_spirv_feature_t feature, loomc_string_view_t provenance) {
  return loomc_spirv_iree_hal_add_feature_fact(
      facts, feature,
      known ? (value ? LOOMC_TARGET_FACT_STATE_TRUE
                     : LOOMC_TARGET_FACT_STATE_FALSE)
            : LOOMC_TARGET_FACT_STATE_UNKNOWN,
      provenance);
}

static loomc_status_t loomc_spirv_iree_hal_add_limit_fact(
    loomc_spirv_iree_hal_profile_facts_t* facts, loomc_spirv_limit_t limit,
    uint64_t value, loomc_string_view_t provenance) {
  if (facts->limit_fact_count >= LOOMC_SPIRV_IREE_HAL_LIMIT_FACT_CAPACITY) {
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "too many SPIR-V IREE HAL limit facts");
  }
  facts->limit_facts[facts->limit_fact_count++] = (loomc_spirv_limit_fact_t){
      .limit = limit,
      .state = LOOMC_TARGET_FACT_STATE_TRUE,
      .value = value,
      .provenance = provenance,
  };
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_iree_hal_add_environment_fact(
    loomc_spirv_iree_hal_profile_facts_t* facts,
    loomc_spirv_environment_t environment, uint64_t value,
    loomc_string_view_t provenance) {
  if (facts->environment_fact_count >=
      LOOMC_SPIRV_IREE_HAL_ENVIRONMENT_FACT_CAPACITY) {
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "too many SPIR-V IREE HAL environment facts");
  }
  facts->environment_facts[facts->environment_fact_count++] =
      (loomc_spirv_environment_fact_t){
          .environment = environment,
          .state = LOOMC_TARGET_FACT_STATE_TRUE,
          .value = value,
          .provenance = provenance,
      };
  return loomc_ok_status();
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
    const loomc_spirv_iree_hal_profile_options_t* options,
    loomc_spirv_iree_hal_profile_facts_t* out_facts, loomc_result_t* result) {
  *out_facts = (loomc_spirv_iree_hal_profile_facts_t){0};
  const iree_hal_device_spec_t* device_spec =
      iree_hal_device_spec(options->device);
  if (device_spec == NULL) {
    return loomc_spirv_iree_hal_fail_cstring(
        result, "IREE HAL device does not expose immutable device facts");
  }
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

  const iree_hal_device_dispatch_spec_t* dispatch = NULL;
  iree_hal_vulkan_device_spec_t vulkan_spec = {0};
  loomc_status_t status = loomc_spirv_iree_hal_load_device_specs(
      options->device, &dispatch, &vulkan_spec);
  if (!loomc_status_is_ok(status)) {
    return loomc_spirv_iree_hal_fail_status(result, status);
  }
  if (dispatch->subgroup.default_size == 0 ||
      dispatch->launch.maximum_workgroup_invocations == 0 ||
      dispatch->launch.maximum_workgroup_size[0] == 0 ||
      dispatch->launch.maximum_workgroup_size[1] == 0 ||
      dispatch->launch.maximum_workgroup_size[2] == 0 ||
      dispatch->launch.maximum_workgroup_count[0] == 0 ||
      dispatch->launch.maximum_workgroup_count[1] == 0 ||
      dispatch->launch.maximum_workgroup_count[2] == 0) {
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
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_bool_feature(
      out_facts, true, LOOMC_SPIRV_FEATURE_VULKAN_SHADER, api_provenance));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_environment_fact(
      out_facts, LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION,
      loomc_spirv_max_version_from_vulkan_api_version(
          loomc_spirv_iree_hal_vulkan_api_version_major(
              vulkan_spec.api_version),
          loomc_spirv_iree_hal_vulkan_api_version_minor(
              vulkan_spec.api_version)),
      api_provenance));

  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_bool_feature(
      out_facts, buffer_device_address,
      LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER,
      loomc_make_cstring_view(
          "iree-hal:vulkan.feature.buffer_device_address")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_bool_feature(
      out_facts, shader_int64, LOOMC_SPIRV_FEATURE_INT64,
      loomc_make_cstring_view("iree-hal:vulkan.feature.shader_int64")));

  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE,
      dispatch->subgroup.default_size,
      loomc_make_cstring_view("iree-hal:vulkan.device.subgroup_size")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MAX_FLAT_WORKGROUP_SIZE,
      dispatch->launch.maximum_workgroup_invocations,
      loomc_make_cstring_view(
          "iree-hal:vulkan.device.max_compute_workgroup_invocations")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X,
      dispatch->launch.maximum_workgroup_size[0],
      loomc_make_cstring_view(
          "iree-hal:vulkan.device.max_compute_workgroup_size_x")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Y,
      dispatch->launch.maximum_workgroup_size[1],
      loomc_make_cstring_view(
          "iree-hal:vulkan.device.max_compute_workgroup_size_y")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Z,
      dispatch->launch.maximum_workgroup_size[2],
      loomc_make_cstring_view(
          "iree-hal:vulkan.device.max_compute_workgroup_size_z")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_X,
      dispatch->launch.maximum_workgroup_count[0],
      loomc_make_cstring_view(
          "iree-hal:vulkan.device.max_compute_workgroup_count_x")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_Y,
      dispatch->launch.maximum_workgroup_count[1],
      loomc_make_cstring_view(
          "iree-hal:vulkan.device.max_compute_workgroup_count_y")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_Z,
      dispatch->launch.maximum_workgroup_count[2],
      loomc_make_cstring_view(
          "iree-hal:vulkan.device.max_compute_workgroup_count_z")));

  const bool cooperative_matrix = loomc_spirv_iree_hal_vulkan_feature_enabled(
      &vulkan_spec, IREE_HAL_VULKAN_FEATURE_ENABLE_COOPERATIVE_MATRIX);
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_optional_feature(
      out_facts, /*known=*/true, cooperative_matrix,
      LOOMC_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR,
      loomc_make_cstring_view(
          "iree-hal:vulkan.feature.cooperative_matrix_khr")));

  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_optional_feature(
      out_facts, /*known=*/true,
      loomc_spirv_iree_hal_vulkan_feature_enabled(
          &vulkan_spec,
          IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_8BIT_ACCESS),
      LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS,
      loomc_make_cstring_view(
          "iree-hal:vulkan.feature.storage_buffer_8bit_access")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_optional_feature(
      out_facts, /*known=*/true,
      loomc_spirv_iree_hal_vulkan_feature_enabled(
          &vulkan_spec,
          IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_16BIT_ACCESS),
      LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_16BIT_ACCESS,
      loomc_make_cstring_view(
          "iree-hal:vulkan.feature.storage_buffer_16bit_access")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_optional_feature(
      out_facts, /*known=*/true,
      loomc_spirv_iree_hal_vulkan_feature_enabled(
          &vulkan_spec, IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16),
      LOOMC_SPIRV_FEATURE_FLOAT16,
      loomc_make_cstring_view("iree-hal:vulkan.feature.shader_float16")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_optional_feature(
      out_facts, /*known=*/true,
      loomc_spirv_iree_hal_vulkan_feature_enabled(
          &vulkan_spec, IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT64),
      LOOMC_SPIRV_FEATURE_FLOAT64,
      loomc_make_cstring_view("iree-hal:vulkan.feature.shader_float64")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_optional_feature(
      out_facts, /*known=*/true,
      loomc_spirv_iree_hal_vulkan_feature_enabled(
          &vulkan_spec, IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT8),
      LOOMC_SPIRV_FEATURE_INT8,
      loomc_make_cstring_view("iree-hal:vulkan.feature.shader_int8")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_optional_feature(
      out_facts, /*known=*/true,
      loomc_spirv_iree_hal_vulkan_feature_enabled(
          &vulkan_spec, IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT16),
      LOOMC_SPIRV_FEATURE_INT16,
      loomc_make_cstring_view("iree-hal:vulkan.feature.shader_int16")));
  const bool bfloat16_type = loomc_spirv_iree_hal_vulkan_feature_enabled(
      &vulkan_spec, IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_TYPE);
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_optional_feature(
      out_facts, /*known=*/true, bfloat16_type,
      LOOMC_SPIRV_FEATURE_BFLOAT16_TYPE_KHR,
      loomc_make_cstring_view("iree-hal:vulkan.feature.shader_bfloat16_type")));
  const bool bfloat16_dot_product = loomc_spirv_iree_hal_vulkan_feature_enabled(
      &vulkan_spec, IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_DOT_PRODUCT);
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_optional_feature(
      out_facts, /*known=*/true, bfloat16_dot_product && bfloat16_type,
      LOOMC_SPIRV_FEATURE_BFLOAT16_DOT_PRODUCT_KHR,
      loomc_make_cstring_view(
          "iree-hal:vulkan.feature.shader_bfloat16_dot_product")));
  const bool bfloat16_cooperative_matrix =
      loomc_spirv_iree_hal_vulkan_feature_enabled(
          &vulkan_spec,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_COOPERATIVE_MATRIX);
  return loomc_spirv_iree_hal_add_optional_feature(
      out_facts, /*known=*/true,
      bfloat16_cooperative_matrix && bfloat16_type && cooperative_matrix,
      LOOMC_SPIRV_FEATURE_BFLOAT16_COOPERATIVE_MATRIX_KHR,
      loomc_make_cstring_view(
          "iree-hal:vulkan.feature.shader_bfloat16_cooperative_matrix"));
}

loomc_status_t loomc_target_profile_create_spirv_iree_hal(
    loomc_target_environment_t* target_environment,
    const loomc_spirv_iree_hal_profile_options_t* options,
    loomc_allocator_t allocator, loomc_target_profile_t** out_profile,
    loomc_result_t** out_result) {
  if (out_profile == NULL || out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_profile and out_result must not be NULL");
  }
  *out_profile = NULL;
  *out_result = NULL;
  if (target_environment == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "target_environment must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_validate_options(options));

  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator, &result));
  loomc_spirv_iree_hal_profile_facts_t facts = {0};
  loomc_status_t status =
      loomc_spirv_iree_hal_query_facts(options, &facts, result);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    loomc_result_release(result);
    result = NULL;
    loomc_spirv_profile_options_t profile_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
        /*.structure_size=*/sizeof(profile_options),
        /*.next=*/NULL,
        /*.identifier=*/options->identifier,
        /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_VULKAN_1_3_BDA,
        /*.feature_facts=*/facts.feature_facts,
        /*.feature_fact_count=*/facts.feature_fact_count,
        /*.limit_facts=*/facts.limit_facts,
        /*.limit_fact_count=*/facts.limit_fact_count,
        /*.environment_facts=*/facts.environment_facts,
        /*.environment_fact_count=*/facts.environment_fact_count,
    };
    return loomc_target_profile_create_spirv(target_environment,
                                             &profile_options, allocator,
                                             out_profile, out_result);
  }
  if (loomc_status_is_ok(status)) {
    *out_result = result;
    result = NULL;
  } else {
    loomc_result_release(result);
  }
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

static loomc_status_t loomc_spirv_iree_hal_provider_create_profile(
    void* user_data, loomc_target_environment_t* target_environment,
    const loomc_iree_hal_profile_options_t* options,
    loomc_allocator_t allocator, bool* out_supported,
    loomc_target_profile_t** out_profile, loomc_result_t** out_result) {
  (void)user_data;
  *out_supported = false;
  *out_profile = NULL;
  *out_result = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_spirv_iree_hal_device_is_supported(options->device, out_supported));
  if (!*out_supported) {
    return loomc_ok_status();
  }

  loomc_spirv_iree_hal_profile_options_t spirv_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_IREE_HAL_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(spirv_options),
      /*.next=*/options->next,
      /*.identifier=*/options->identifier,
      /*.device=*/options->device,
      /*.physical_device_affinity=*/options->physical_device_affinity,
  };
  return loomc_target_profile_create_spirv_iree_hal(
      target_environment, &spirv_options, allocator, out_profile, out_result);
}

const loomc_iree_hal_profile_provider_t* loomc_spirv_iree_hal_profile_provider(
    void) {
  static const loomc_iree_hal_profile_provider_t provider = {
      /*.name=*/{"spirv.iree_hal.vulkan", 21},
      /*.user_data=*/NULL,
      /*.create_profile=*/loomc_spirv_iree_hal_provider_create_profile,
  };
  return &provider;
}
