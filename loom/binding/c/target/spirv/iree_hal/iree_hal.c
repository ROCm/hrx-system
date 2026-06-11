// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/spirv/iree_hal.h"

#include <stdint.h>

#include "diagnostic.h"
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
  if (options->executable_cache == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "SPIR-V IREE HAL options require an executable cache");
  }
  return loomc_ok_status();
}

static const char* loomc_spirv_iree_hal_string_data(iree_string_view_t value) {
  return value.data != NULL ? value.data : "";
}

static loomc_status_t loomc_spirv_iree_hal_query_u32(
    iree_hal_device_t* device, iree_string_view_t category,
    iree_string_view_t key, uint32_t* out_value) {
  int64_t value = 0;
  iree_status_t query_status =
      iree_hal_device_query_i64(device, category, key, &value);
  if (!iree_status_is_ok(query_status)) {
    return loomc_status_from_iree(query_status);
  }
  if (value < 0 || value > UINT32_MAX) {
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "HAL query '%.*s :: %.*s' returned value out of uint32_t range",
        (int)category.size, loomc_spirv_iree_hal_string_data(category),
        (int)key.size, loomc_spirv_iree_hal_string_data(key)));
  }
  *out_value = (uint32_t)value;
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_iree_hal_query_optional_bool(
    iree_hal_device_t* device, iree_string_view_t key, bool* out_known,
    bool* out_value) {
  *out_known = false;
  *out_value = false;
  int64_t value = 0;
  iree_status_t query_status =
      iree_hal_device_query_i64(device, IREE_SV("vulkan.feature"), key, &value);
  if (!iree_status_is_ok(query_status)) {
    iree_status_code_t code = iree_status_code(query_status);
    if (code == IREE_STATUS_NOT_FOUND || code == IREE_STATUS_UNIMPLEMENTED) {
      iree_status_free(query_status);
      return loomc_ok_status();
    }
    return loomc_status_from_iree(query_status);
  }
  *out_known = true;
  *out_value = value != 0;
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

static loomc_status_t loomc_spirv_iree_hal_query_required_feature(
    iree_hal_device_t* device, const char* key, bool* out_value) {
  uint32_t value = 0;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_query_u32(
      device, IREE_SV("vulkan.feature"), iree_make_cstring_view(key), &value));
  *out_value = value != 0;
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_iree_hal_query_optional_feature(
    iree_hal_device_t* device, const char* key, bool* out_known,
    bool* out_value) {
  return loomc_spirv_iree_hal_query_optional_bool(
      device, iree_make_cstring_view(key), out_known, out_value);
}

#define LOOMC_SPIRV_IREE_HAL_RETURN_IF_QUERY_ERROR(result, expr)       \
  do {                                                                 \
    loomc_status_t query_status = (expr);                              \
    if (!loomc_status_is_ok(query_status)) {                           \
      return loomc_spirv_iree_hal_fail_status((result), query_status); \
    }                                                                  \
  } while (0)

static loomc_status_t loomc_spirv_iree_hal_query_facts(
    const loomc_spirv_iree_hal_profile_options_t* options,
    loomc_spirv_iree_hal_profile_facts_t* out_facts, loomc_result_t* result) {
  *out_facts = (loomc_spirv_iree_hal_profile_facts_t){0};
  if (!iree_hal_executable_cache_can_prepare_format(
          options->executable_cache,
          IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION,
          IREE_SV("vulkan-spirv-bda"))) {
    return loomc_spirv_iree_hal_fail_cstring(
        result, "IREE HAL executable cache cannot prepare vulkan-spirv-bda");
  }

  uint32_t api_version = 0;
  loomc_status_t status =
      loomc_spirv_iree_hal_query_u32(options->device, IREE_SV("vulkan.device"),
                                     IREE_SV("api_version"), &api_version);
  if (!loomc_status_is_ok(status)) {
    return loomc_spirv_iree_hal_fail_status(result, status);
  }
  if (api_version < LOOMC_SPIRV_IREE_HAL_VULKAN_API_VERSION_1_3) {
    return loomc_spirv_iree_hal_fail_cstring(
        result, "IREE HAL SPIR-V profile requires Vulkan API version 1.3");
  }

  bool buffer_device_address = false;
  status = loomc_spirv_iree_hal_query_required_feature(
      options->device, "buffer_device_address", &buffer_device_address);
  if (!loomc_status_is_ok(status)) {
    return loomc_spirv_iree_hal_fail_status(result, status);
  }
  if (!buffer_device_address) {
    return loomc_spirv_iree_hal_fail_cstring(
        result, "IREE HAL SPIR-V profile requires buffer_device_address");
  }

  bool shader_int64 = false;
  status = loomc_spirv_iree_hal_query_required_feature(
      options->device, "shader_int64", &shader_int64);
  if (!loomc_status_is_ok(status)) {
    return loomc_spirv_iree_hal_fail_status(result, status);
  }
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
          loomc_spirv_iree_hal_vulkan_api_version_major(api_version),
          loomc_spirv_iree_hal_vulkan_api_version_minor(api_version)),
      api_provenance));

  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_bool_feature(
      out_facts, buffer_device_address,
      LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER,
      loomc_make_cstring_view(
          "iree-hal:vulkan.feature.buffer_device_address")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_bool_feature(
      out_facts, shader_int64, LOOMC_SPIRV_FEATURE_INT64,
      loomc_make_cstring_view("iree-hal:vulkan.feature.shader_int64")));

  uint32_t limit_value = 0;
  LOOMC_SPIRV_IREE_HAL_RETURN_IF_QUERY_ERROR(
      result,
      loomc_spirv_iree_hal_query_u32(options->device, IREE_SV("vulkan.device"),
                                     IREE_SV("subgroup_size"), &limit_value));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE, limit_value,
      loomc_make_cstring_view("iree-hal:vulkan.device.subgroup_size")));
  LOOMC_SPIRV_IREE_HAL_RETURN_IF_QUERY_ERROR(
      result, loomc_spirv_iree_hal_query_u32(
                  options->device, IREE_SV("vulkan.device"),
                  IREE_SV("max_compute_workgroup_invocations"), &limit_value));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MAX_FLAT_WORKGROUP_SIZE, limit_value,
      loomc_make_cstring_view(
          "iree-hal:vulkan.device.max_compute_workgroup_invocations")));
  LOOMC_SPIRV_IREE_HAL_RETURN_IF_QUERY_ERROR(
      result, loomc_spirv_iree_hal_query_u32(
                  options->device, IREE_SV("vulkan.device"),
                  IREE_SV("max_compute_workgroup_size_x"), &limit_value));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X, limit_value,
      loomc_make_cstring_view(
          "iree-hal:vulkan.device.max_compute_workgroup_size_x")));
  LOOMC_SPIRV_IREE_HAL_RETURN_IF_QUERY_ERROR(
      result, loomc_spirv_iree_hal_query_u32(
                  options->device, IREE_SV("vulkan.device"),
                  IREE_SV("max_compute_workgroup_size_y"), &limit_value));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Y, limit_value,
      loomc_make_cstring_view(
          "iree-hal:vulkan.device.max_compute_workgroup_size_y")));
  LOOMC_SPIRV_IREE_HAL_RETURN_IF_QUERY_ERROR(
      result, loomc_spirv_iree_hal_query_u32(
                  options->device, IREE_SV("vulkan.device"),
                  IREE_SV("max_compute_workgroup_size_z"), &limit_value));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Z, limit_value,
      loomc_make_cstring_view(
          "iree-hal:vulkan.device.max_compute_workgroup_size_z")));
  LOOMC_SPIRV_IREE_HAL_RETURN_IF_QUERY_ERROR(
      result, loomc_spirv_iree_hal_query_u32(
                  options->device, IREE_SV("vulkan.device"),
                  IREE_SV("max_compute_workgroup_count_x"), &limit_value));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_X, limit_value,
      loomc_make_cstring_view(
          "iree-hal:vulkan.device.max_compute_workgroup_count_x")));
  LOOMC_SPIRV_IREE_HAL_RETURN_IF_QUERY_ERROR(
      result, loomc_spirv_iree_hal_query_u32(
                  options->device, IREE_SV("vulkan.device"),
                  IREE_SV("max_compute_workgroup_count_y"), &limit_value));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_Y, limit_value,
      loomc_make_cstring_view(
          "iree-hal:vulkan.device.max_compute_workgroup_count_y")));
  LOOMC_SPIRV_IREE_HAL_RETURN_IF_QUERY_ERROR(
      result, loomc_spirv_iree_hal_query_u32(
                  options->device, IREE_SV("vulkan.device"),
                  IREE_SV("max_compute_workgroup_count_z"), &limit_value));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_limit_fact(
      out_facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_Z, limit_value,
      loomc_make_cstring_view(
          "iree-hal:vulkan.device.max_compute_workgroup_count_z")));

  bool known = false;
  bool value = false;
  LOOMC_SPIRV_IREE_HAL_RETURN_IF_QUERY_ERROR(
      result, loomc_spirv_iree_hal_query_optional_feature(
                  options->device, "cooperative_matrix_khr", &known, &value));
  const bool cooperative_matrix = known && value;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_optional_feature(
      out_facts, known, value, LOOMC_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR,
      loomc_make_cstring_view(
          "iree-hal:vulkan.feature.cooperative_matrix_khr")));

  LOOMC_SPIRV_IREE_HAL_RETURN_IF_QUERY_ERROR(
      result,
      loomc_spirv_iree_hal_query_optional_feature(
          options->device, "storage_buffer_8bit_access", &known, &value));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_optional_feature(
      out_facts, known, value, LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS,
      loomc_make_cstring_view(
          "iree-hal:vulkan.feature.storage_buffer_8bit_access")));
  LOOMC_SPIRV_IREE_HAL_RETURN_IF_QUERY_ERROR(
      result,
      loomc_spirv_iree_hal_query_optional_feature(
          options->device, "storage_buffer_16bit_access", &known, &value));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_optional_feature(
      out_facts, known, value, LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_16BIT_ACCESS,
      loomc_make_cstring_view(
          "iree-hal:vulkan.feature.storage_buffer_16bit_access")));
  LOOMC_SPIRV_IREE_HAL_RETURN_IF_QUERY_ERROR(
      result, loomc_spirv_iree_hal_query_optional_feature(
                  options->device, "shader_float16", &known, &value));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_optional_feature(
      out_facts, known, value, LOOMC_SPIRV_FEATURE_FLOAT16,
      loomc_make_cstring_view("iree-hal:vulkan.feature.shader_float16")));
  LOOMC_SPIRV_IREE_HAL_RETURN_IF_QUERY_ERROR(
      result, loomc_spirv_iree_hal_query_optional_feature(
                  options->device, "shader_float64", &known, &value));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_optional_feature(
      out_facts, known, value, LOOMC_SPIRV_FEATURE_FLOAT64,
      loomc_make_cstring_view("iree-hal:vulkan.feature.shader_float64")));
  LOOMC_SPIRV_IREE_HAL_RETURN_IF_QUERY_ERROR(
      result, loomc_spirv_iree_hal_query_optional_feature(
                  options->device, "shader_int8", &known, &value));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_optional_feature(
      out_facts, known, value, LOOMC_SPIRV_FEATURE_INT8,
      loomc_make_cstring_view("iree-hal:vulkan.feature.shader_int8")));
  LOOMC_SPIRV_IREE_HAL_RETURN_IF_QUERY_ERROR(
      result, loomc_spirv_iree_hal_query_optional_feature(
                  options->device, "shader_int16", &known, &value));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_optional_feature(
      out_facts, known, value, LOOMC_SPIRV_FEATURE_INT16,
      loomc_make_cstring_view("iree-hal:vulkan.feature.shader_int16")));
  LOOMC_SPIRV_IREE_HAL_RETURN_IF_QUERY_ERROR(
      result, loomc_spirv_iree_hal_query_optional_feature(
                  options->device, "shader_bfloat16_type", &known, &value));
  const bool bfloat16_type = known && value;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_optional_feature(
      out_facts, known, value, LOOMC_SPIRV_FEATURE_BFLOAT16_TYPE_KHR,
      loomc_make_cstring_view("iree-hal:vulkan.feature.shader_bfloat16_type")));
  LOOMC_SPIRV_IREE_HAL_RETURN_IF_QUERY_ERROR(
      result,
      loomc_spirv_iree_hal_query_optional_feature(
          options->device, "shader_bfloat16_dot_product", &known, &value));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_iree_hal_add_optional_feature(
      out_facts, known, value && bfloat16_type,
      LOOMC_SPIRV_FEATURE_BFLOAT16_DOT_PRODUCT_KHR,
      loomc_make_cstring_view(
          "iree-hal:vulkan.feature.shader_bfloat16_dot_product")));
  LOOMC_SPIRV_IREE_HAL_RETURN_IF_QUERY_ERROR(
      result, loomc_spirv_iree_hal_query_optional_feature(
                  options->device, "shader_bfloat16_cooperative_matrix", &known,
                  &value));
  return loomc_spirv_iree_hal_add_optional_feature(
      out_facts, known, value && bfloat16_type && cooperative_matrix,
      LOOMC_SPIRV_FEATURE_BFLOAT16_COOPERATIVE_MATRIX_KHR,
      loomc_make_cstring_view(
          "iree-hal:vulkan.feature.shader_bfloat16_cooperative_matrix"));
}

#undef LOOMC_SPIRV_IREE_HAL_RETURN_IF_QUERY_ERROR

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
  int64_t value = 0;
  iree_status_t status = iree_hal_device_query_i64(
      device, IREE_SV("vulkan.device"), IREE_SV("api_version"), &value);
  if (iree_status_is_ok(status)) {
    *out_supported = true;
    return loomc_ok_status();
  }
  const iree_status_code_t code = iree_status_code(status);
  if (code == IREE_STATUS_NOT_FOUND || code == IREE_STATUS_UNIMPLEMENTED) {
    iree_status_free(status);
    return loomc_ok_status();
  }
  return loomc_status_from_iree(status);
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
      /*.executable_cache=*/options->executable_cache,
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
