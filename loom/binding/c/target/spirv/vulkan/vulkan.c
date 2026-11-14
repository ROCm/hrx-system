// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/spirv/vulkan.h"

#include <string.h>

#include "iree/base/internal/debugging.h"

enum {
  LOOMC_SPIRV_VULKAN_LIMIT_FACT_CAPACITY = 16,
  LOOMC_SPIRV_VULKAN_ENVIRONMENT_FACT_CAPACITY = 4,
  // PCI-SIG vendor identifier reported by AMD Vulkan physical devices.
  LOOMC_SPIRV_VULKAN_AMD_VENDOR_ID = 0x1002,
};

typedef struct loomc_spirv_vulkan_profile_facts_t {
  // Feature facts normalized from Vulkan feature structs.
  loomc_spirv_feature_fact_t feature_facts[LOOMC_SPIRV_FEATURE_COUNT];

  // Number of entries in feature_facts.
  loomc_host_size_t feature_fact_count;

  // Numeric limit facts normalized from Vulkan properties.
  loomc_spirv_limit_fact_t limit_facts[LOOMC_SPIRV_VULKAN_LIMIT_FACT_CAPACITY];

  // Number of entries in limit_facts.
  loomc_host_size_t limit_fact_count;

  // Environment facts normalized from Vulkan properties.
  loomc_spirv_environment_fact_t
      environment_facts[LOOMC_SPIRV_VULKAN_ENVIRONMENT_FACT_CAPACITY];

  // Number of entries in environment_facts.
  loomc_host_size_t environment_fact_count;
} loomc_spirv_vulkan_profile_facts_t;

typedef struct loomc_spirv_vulkan_float_width_controls_t {
  // Whether the control is supported for F16.
  VkBool32 f16;
  // Whether the control is supported for F32.
  VkBool32 f32;
  // Whether the control is supported for F64.
  VkBool32 f64;
} loomc_spirv_vulkan_float_width_controls_t;

typedef struct loomc_spirv_vulkan_float_controls_t {
  // Widths whose denormal modes can be controlled independently.
  VkShaderFloatControlsIndependence denorm_independence;
  // Per-width denormal-preservation support.
  loomc_spirv_vulkan_float_width_controls_t denorm_preserve;
  // Per-width signed-zero, infinity, and NaN preservation support.
  loomc_spirv_vulkan_float_width_controls_t signed_zero_inf_nan_preserve;
  // Widths whose rounding modes can be controlled independently.
  VkShaderFloatControlsIndependence rounding_independence;
  // Per-width round-to-nearest-even support.
  loomc_spirv_vulkan_float_width_controls_t rounding_rte;
} loomc_spirv_vulkan_float_controls_t;

typedef struct loomc_spirv_vulkan_extensions_t {
  // VK_KHR_shader_float16_int8 is supported.
  bool shader_float16_int8_khr;

  // VK_KHR_shader_subgroup_extended_types is supported.
  bool shader_subgroup_extended_types_khr;

  // VK_KHR_8bit_storage is supported.
  bool storage_8bit_khr;

  // VK_KHR_16bit_storage is supported.
  bool storage_16bit_khr;

  // VK_KHR_buffer_device_address is supported.
  bool buffer_device_address_khr;

  // VK_KHR_shader_atomic_int64 is supported.
  bool shader_atomic_int64_khr;

  // VK_KHR_shader_fma is supported.
  bool shader_fma_khr;

  // VK_KHR_shader_float_controls2 is supported.
  bool shader_float_controls2_khr;

  // VK_KHR_shader_float_controls is supported.
  bool shader_float_controls_khr;

  // VK_KHR_vulkan_memory_model is supported.
  bool vulkan_memory_model_khr;

  // VK_EXT_shader_atomic_float is supported.
  bool shader_atomic_float_ext;

  // VK_EXT_shader_atomic_float2 is supported.
  bool shader_atomic_float2_ext;

  // VK_EXT_subgroup_size_control is supported.
  bool subgroup_size_control_ext;

  // VK_KHR_cooperative_matrix is supported.
  bool cooperative_matrix_khr;
} loomc_spirv_vulkan_extensions_t;

static loomc_status_t loomc_spirv_vulkan_validate_string_view(
    loomc_string_view_t value) {
  if (value.data == NULL && value.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_vulkan_validate_function_table(
    const loomc_spirv_vulkan_function_table_t* functions) {
  if (functions == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "Vulkan function table must not be NULL");
  }
  if (functions->type != LOOMC_STRUCTURE_TYPE_NONE &&
      functions->type != LOOMC_STRUCTURE_TYPE_SPIRV_VULKAN_FUNCTION_TABLE) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "SPIR-V Vulkan function table has an unknown structure type");
  }
  if (functions->structure_size != 0 &&
      functions->structure_size < sizeof(*functions)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "SPIR-V Vulkan function table structure_size is too small");
  }
  if (functions->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "SPIR-V Vulkan function table extensions are not supported");
  }
  if (functions->get_physical_device_properties2 == NULL ||
      functions->get_physical_device_features2 == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "SPIR-V Vulkan function table must provide properties2 and features2 "
        "queries");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_vulkan_validate_options(
    const loomc_spirv_vulkan_profile_options_t* options) {
  if (options == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "SPIR-V Vulkan options must not be NULL");
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_SPIRV_VULKAN_PROFILE_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "SPIR-V Vulkan options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "SPIR-V Vulkan options structure_size is too small");
  }
  if (options->next != NULL) {
    return loomc_make_status(LOOMC_STATUS_UNIMPLEMENTED,
                             "SPIR-V Vulkan option extensions are not "
                             "supported");
  }
  if (options->physical_device == VK_NULL_HANDLE) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "Vulkan physical_device must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(
      loomc_spirv_vulkan_validate_string_view(options->identifier));
  return loomc_spirv_vulkan_validate_function_table(options->functions);
}

static loomc_status_t loomc_spirv_vulkan_add_feature_fact(
    loomc_spirv_vulkan_profile_facts_t* facts, loomc_spirv_feature_t feature,
    loomc_target_fact_state_t state, loomc_string_view_t provenance) {
  if (state == LOOMC_TARGET_FACT_STATE_UNKNOWN) {
    return loomc_ok_status();
  }
  if (facts->feature_fact_count >= LOOMC_SPIRV_FEATURE_COUNT) {
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "too many Vulkan SPIR-V feature facts");
  }
  facts->feature_facts[facts->feature_fact_count++] =
      (loomc_spirv_feature_fact_t){
          .feature = feature,
          .state = state,
          .provenance = provenance,
      };
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_vulkan_add_bool_feature(
    loomc_spirv_vulkan_profile_facts_t* facts, VkBool32 value,
    loomc_spirv_feature_t feature, loomc_string_view_t provenance) {
  return loomc_spirv_vulkan_add_feature_fact(
      facts, feature,
      value ? LOOMC_TARGET_FACT_STATE_TRUE : LOOMC_TARGET_FACT_STATE_FALSE,
      provenance);
}

static loomc_status_t loomc_spirv_vulkan_add_limit_fact(
    loomc_spirv_vulkan_profile_facts_t* facts, loomc_spirv_limit_t limit,
    uint64_t value, loomc_string_view_t provenance) {
  if (facts->limit_fact_count >= LOOMC_SPIRV_VULKAN_LIMIT_FACT_CAPACITY) {
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "too many Vulkan SPIR-V limit facts");
  }
  facts->limit_facts[facts->limit_fact_count++] = (loomc_spirv_limit_fact_t){
      .limit = limit,
      .state = LOOMC_TARGET_FACT_STATE_TRUE,
      .value = value,
      .provenance = provenance,
  };
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_vulkan_add_float_control_features(
    loomc_spirv_vulkan_profile_facts_t* facts,
    const loomc_spirv_vulkan_float_controls_t* controls) {
  const bool denorm_all = controls->denorm_independence ==
                              VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE &&
                          controls->denorm_preserve.f16 &&
                          controls->denorm_preserve.f32 &&
                          controls->denorm_preserve.f64;
  const bool denorm_pair = controls->denorm_preserve.f16 &&
                           controls->denorm_preserve.f64 &&
                           (controls->denorm_independence !=
                                VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE ||
                            denorm_all);
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
      facts, denorm_all, LOOMC_SPIRV_FEATURE_FLOAT_ALL_DENORM_PRESERVE,
      loomc_make_cstring_view("vulkan:float_all_denorm_preserve")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
      facts, denorm_pair, LOOMC_SPIRV_FEATURE_FLOAT16_64_DENORM_PRESERVE,
      loomc_make_cstring_view("vulkan:float16_64_denorm_preserve")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
      facts,
      controls->denorm_preserve.f32 &&
          (controls->denorm_independence !=
               VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE ||
           denorm_all),
      LOOMC_SPIRV_FEATURE_FLOAT32_DENORM_PRESERVE,
      loomc_make_cstring_view("vulkan:float32_denorm_preserve")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
      facts,
      denorm_pair || (controls->denorm_preserve.f16 &&
                      controls->denorm_independence ==
                          VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL),
      LOOMC_SPIRV_FEATURE_FLOAT16_DENORM_PRESERVE,
      loomc_make_cstring_view("vulkan:float16_denorm_preserve")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
      facts,
      denorm_pair || (controls->denorm_preserve.f64 &&
                      controls->denorm_independence ==
                          VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL),
      LOOMC_SPIRV_FEATURE_FLOAT64_DENORM_PRESERVE,
      loomc_make_cstring_view("vulkan:float64_denorm_preserve")));
  const bool rounding_all = controls->rounding_independence ==
                                VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE &&
                            controls->rounding_rte.f16 &&
                            controls->rounding_rte.f32 &&
                            controls->rounding_rte.f64;
  const bool rounding_pair = controls->rounding_rte.f16 &&
                             controls->rounding_rte.f64 &&
                             (controls->rounding_independence !=
                                  VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE ||
                              rounding_all);
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
      facts, rounding_all, LOOMC_SPIRV_FEATURE_FLOAT_ALL_ROUNDING_RTE,
      loomc_make_cstring_view("vulkan:float_all_rounding_rte")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
      facts, rounding_pair, LOOMC_SPIRV_FEATURE_FLOAT16_64_ROUNDING_RTE,
      loomc_make_cstring_view("vulkan:float16_64_rounding_rte")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
      facts,
      controls->rounding_rte.f32 &&
          (controls->rounding_independence !=
               VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE ||
           rounding_all),
      LOOMC_SPIRV_FEATURE_FLOAT32_ROUNDING_RTE,
      loomc_make_cstring_view("vulkan:float32_rounding_rte")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
      facts,
      rounding_pair || (controls->rounding_rte.f16 &&
                        controls->rounding_independence ==
                            VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL),
      LOOMC_SPIRV_FEATURE_FLOAT16_ROUNDING_RTE,
      loomc_make_cstring_view("vulkan:float16_rounding_rte")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
      facts,
      rounding_pair || (controls->rounding_rte.f64 &&
                        controls->rounding_independence ==
                            VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL),
      LOOMC_SPIRV_FEATURE_FLOAT64_ROUNDING_RTE,
      loomc_make_cstring_view("vulkan:float64_rounding_rte")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
      facts, controls->signed_zero_inf_nan_preserve.f16,
      LOOMC_SPIRV_FEATURE_FLOAT16_SIGNED_ZERO_INF_NAN_PRESERVE,
      loomc_make_cstring_view("vulkan:shaderSignedZeroInfNanPreserveFloat16")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
      facts, controls->signed_zero_inf_nan_preserve.f32,
      LOOMC_SPIRV_FEATURE_FLOAT32_SIGNED_ZERO_INF_NAN_PRESERVE,
      loomc_make_cstring_view("vulkan:shaderSignedZeroInfNanPreserveFloat32")));
  return loomc_spirv_vulkan_add_bool_feature(
      facts, controls->signed_zero_inf_nan_preserve.f64,
      LOOMC_SPIRV_FEATURE_FLOAT64_SIGNED_ZERO_INF_NAN_PRESERVE,
      loomc_make_cstring_view("vulkan:shaderSignedZeroInfNanPreserveFloat64"));
}

static loomc_status_t loomc_spirv_vulkan_add_environment_fact(
    loomc_spirv_vulkan_profile_facts_t* facts,
    loomc_spirv_environment_t environment, uint64_t value,
    loomc_string_view_t provenance) {
  if (facts->environment_fact_count >=
      LOOMC_SPIRV_VULKAN_ENVIRONMENT_FACT_CAPACITY) {
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "too many Vulkan SPIR-V environment facts");
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

static loomc_status_t loomc_spirv_vulkan_status_from_result(VkResult result) {
  switch (result) {
    case VK_SUCCESS:
      return loomc_ok_status();
    case VK_ERROR_OUT_OF_HOST_MEMORY:
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
      return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                               "Vulkan query exhausted memory");
    case VK_ERROR_INITIALIZATION_FAILED:
    case VK_ERROR_DEVICE_LOST:
      return loomc_make_status(LOOMC_STATUS_UNAVAILABLE,
                               "Vulkan query is unavailable");
    default:
      return loomc_make_status(LOOMC_STATUS_UNKNOWN, "Vulkan query failed");
  }
}

static bool loomc_spirv_vulkan_extension_name_equal(
    const VkExtensionProperties* extension, const char* name) {
  return strcmp(extension->extensionName, name) == 0;
}

static void loomc_spirv_vulkan_record_extension(
    const VkExtensionProperties* extension,
    loomc_spirv_vulkan_extensions_t* out_extensions) {
  if (loomc_spirv_vulkan_extension_name_equal(
          extension, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)) {
    out_extensions->shader_float16_int8_khr = true;
  } else if (loomc_spirv_vulkan_extension_name_equal(
                 extension,
                 VK_KHR_SHADER_SUBGROUP_EXTENDED_TYPES_EXTENSION_NAME)) {
    out_extensions->shader_subgroup_extended_types_khr = true;
  } else if (loomc_spirv_vulkan_extension_name_equal(
                 extension, VK_KHR_8BIT_STORAGE_EXTENSION_NAME)) {
    out_extensions->storage_8bit_khr = true;
  } else if (loomc_spirv_vulkan_extension_name_equal(
                 extension, VK_KHR_16BIT_STORAGE_EXTENSION_NAME)) {
    out_extensions->storage_16bit_khr = true;
  } else if (loomc_spirv_vulkan_extension_name_equal(
                 extension, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)) {
    out_extensions->buffer_device_address_khr = true;
  } else if (loomc_spirv_vulkan_extension_name_equal(
                 extension, VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME)) {
    out_extensions->shader_atomic_int64_khr = true;
  } else if (loomc_spirv_vulkan_extension_name_equal(
                 extension, VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME)) {
    out_extensions->shader_float_controls_khr = true;
  } else if (loomc_spirv_vulkan_extension_name_equal(
                 extension, VK_KHR_SHADER_FMA_EXTENSION_NAME)) {
    out_extensions->shader_fma_khr = true;
  } else if (loomc_spirv_vulkan_extension_name_equal(
                 extension, VK_KHR_SHADER_FLOAT_CONTROLS_2_EXTENSION_NAME)) {
    out_extensions->shader_float_controls2_khr = true;
  } else if (loomc_spirv_vulkan_extension_name_equal(
                 extension, VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME)) {
    out_extensions->vulkan_memory_model_khr = true;
  } else if (loomc_spirv_vulkan_extension_name_equal(
                 extension, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)) {
    out_extensions->shader_atomic_float_ext = true;
  } else if (loomc_spirv_vulkan_extension_name_equal(
                 extension, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)) {
    out_extensions->subgroup_size_control_ext = true;
  } else if (loomc_spirv_vulkan_extension_name_equal(
                 extension, VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME)) {
    out_extensions->shader_atomic_float2_ext = true;
#if defined(VK_KHR_cooperative_matrix)
  } else if (loomc_spirv_vulkan_extension_name_equal(
                 extension, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
    out_extensions->cooperative_matrix_khr = true;
#endif  // defined(VK_KHR_cooperative_matrix)
  }
}

static loomc_status_t loomc_spirv_vulkan_query_extensions(
    const loomc_spirv_vulkan_profile_options_t* options,
    loomc_allocator_t allocator,
    loomc_spirv_vulkan_extensions_t* out_extensions) {
  *out_extensions = (loomc_spirv_vulkan_extensions_t){0};
  if (options->functions->enumerate_device_extension_properties == NULL) {
    return loomc_ok_status();
  }

  uint32_t extension_count = 0;
  IREE_LEAK_CHECK_DISABLE_PUSH();
  VkResult result = options->functions->enumerate_device_extension_properties(
      options->physical_device, NULL, &extension_count, NULL);
  IREE_LEAK_CHECK_DISABLE_POP();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_status_from_result(result));
  if (extension_count == 0) {
    return loomc_ok_status();
  }

  VkExtensionProperties* extensions = NULL;
  loomc_status_t status = loomc_ok_status();
  bool completed = false;
  for (uint32_t attempt = 0; attempt < 4 && loomc_status_is_ok(status);
       ++attempt) {
    loomc_allocator_free(allocator, extensions);
    extensions = NULL;
    if ((loomc_host_size_t)extension_count >
        LOOMC_HOST_SIZE_MAX / sizeof(*extensions)) {
      status = loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                                 "Vulkan device extension count is too large");
      break;
    }
    status = loomc_allocator_malloc(
        allocator, sizeof(*extensions) * extension_count, (void**)&extensions);
    if (!loomc_status_is_ok(status)) {
      break;
    }
    uint32_t written_extension_count = extension_count;
    IREE_LEAK_CHECK_DISABLE_PUSH();
    result = options->functions->enumerate_device_extension_properties(
        options->physical_device, NULL, &written_extension_count, extensions);
    IREE_LEAK_CHECK_DISABLE_POP();
    if (result == VK_INCOMPLETE) {
      extension_count = written_extension_count > extension_count
                            ? written_extension_count
                            : extension_count + 16;
      continue;
    }
    status = loomc_spirv_vulkan_status_from_result(result);
    if (!loomc_status_is_ok(status)) {
      break;
    }
    for (uint32_t i = 0; i < written_extension_count; ++i) {
      loomc_spirv_vulkan_record_extension(&extensions[i], out_extensions);
    }
    completed = true;
    break;
  }
  loomc_allocator_free(allocator, extensions);
  if (loomc_status_is_ok(status) && !completed) {
    return loomc_make_status(LOOMC_STATUS_UNAVAILABLE,
                             "Vulkan device extensions changed too often "
                             "during enumeration");
  }
  return status;
}

static void loomc_spirv_vulkan_link_out_struct(void** inout_chain,
                                               void* next_struct) {
  VkBaseOutStructure* next = (VkBaseOutStructure*)next_struct;
  next->pNext = (VkBaseOutStructure*)*inout_chain;
  *inout_chain = next_struct;
}

static loomc_status_t loomc_spirv_vulkan_query_properties(
    const loomc_spirv_vulkan_profile_options_t* options,
    const loomc_spirv_vulkan_extensions_t* extensions,
    loomc_spirv_vulkan_profile_facts_t* facts, uint32_t* out_api_version,
    uint32_t* out_subgroup_size,
    VkPhysicalDeviceSubgroupSizeControlProperties* out_subgroup_control) {
  VkPhysicalDeviceProperties2 properties2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
  };
  IREE_LEAK_CHECK_DISABLE_PUSH();
  options->functions->get_physical_device_properties2(options->physical_device,
                                                      &properties2);
  IREE_LEAK_CHECK_DISABLE_POP();
  *out_api_version = properties2.properties.apiVersion;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
      facts,
      properties2.properties.vendorID != LOOMC_SPIRV_VULKAN_AMD_VENDOR_ID,
      LOOMC_SPIRV_FEATURE_FLOAT64_TO_FLOAT16_CONVERSION,
      loomc_make_cstring_view("vulkan:vendorID")));

  const uint32_t max_spirv_version =
      loomc_spirv_max_version_from_vulkan_api_version(
          VK_VERSION_MAJOR(*out_api_version),
          VK_VERSION_MINOR(*out_api_version));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_environment_fact(
      facts, LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION, max_spirv_version,
      loomc_make_cstring_view("vulkan:apiVersion")));

  const VkPhysicalDeviceLimits* limits = &properties2.properties.limits;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_limit_fact(
      facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X,
      limits->maxComputeWorkGroupSize[0],
      loomc_make_cstring_view("vulkan:maxComputeWorkGroupSize[0]")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_limit_fact(
      facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Y,
      limits->maxComputeWorkGroupSize[1],
      loomc_make_cstring_view("vulkan:maxComputeWorkGroupSize[1]")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_limit_fact(
      facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Z,
      limits->maxComputeWorkGroupSize[2],
      loomc_make_cstring_view("vulkan:maxComputeWorkGroupSize[2]")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_limit_fact(
      facts, LOOMC_SPIRV_LIMIT_MAX_FLAT_WORKGROUP_SIZE,
      limits->maxComputeWorkGroupInvocations,
      loomc_make_cstring_view("vulkan:maxComputeWorkGroupInvocations")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_limit_fact(
      facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_STORAGE_BYTES,
      limits->maxComputeSharedMemorySize,
      loomc_make_cstring_view("vulkan:maxComputeSharedMemorySize")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_limit_fact(
      facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_X,
      limits->maxComputeWorkGroupCount[0],
      loomc_make_cstring_view("vulkan:maxComputeWorkGroupCount[0]")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_limit_fact(
      facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_Y,
      limits->maxComputeWorkGroupCount[1],
      loomc_make_cstring_view("vulkan:maxComputeWorkGroupCount[1]")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_limit_fact(
      facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_Z,
      limits->maxComputeWorkGroupCount[2],
      loomc_make_cstring_view("vulkan:maxComputeWorkGroupCount[2]")));

  const bool has_vulkan_11 = *out_api_version >= VK_API_VERSION_1_1;
  const bool has_vulkan_12 = *out_api_version >= VK_API_VERSION_1_2;
  const bool query_float_controls_extension =
      !has_vulkan_12 && extensions->shader_float_controls_khr;
  if (!has_vulkan_11 && !query_float_controls_extension) {
    return loomc_ok_status();
  }

  VkPhysicalDeviceSubgroupProperties subgroup_properties = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
  };
  VkPhysicalDeviceVulkan12Properties vulkan12_properties = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES,
  };
  VkPhysicalDeviceFloatControlsProperties float_controls_properties = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES,
  };

  void* property_chain = NULL;
  *out_subgroup_control = (VkPhysicalDeviceSubgroupSizeControlProperties){
      .sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES,
  };
  if (*out_api_version >= VK_API_VERSION_1_3 ||
      extensions->subgroup_size_control_ext) {
    loomc_spirv_vulkan_link_out_struct(&property_chain, out_subgroup_control);
  }
  if (query_float_controls_extension) {
    loomc_spirv_vulkan_link_out_struct(&property_chain,
                                       &float_controls_properties);
  }
  if (has_vulkan_12) {
    loomc_spirv_vulkan_link_out_struct(&property_chain, &vulkan12_properties);
  }
  if (has_vulkan_11) {
    loomc_spirv_vulkan_link_out_struct(&property_chain, &subgroup_properties);
  }
  properties2 = (VkPhysicalDeviceProperties2){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = property_chain,
  };
  IREE_LEAK_CHECK_DISABLE_PUSH();
  options->functions->get_physical_device_properties2(options->physical_device,
                                                      &properties2);
  IREE_LEAK_CHECK_DISABLE_POP();
  if (has_vulkan_11) {
    *out_subgroup_size = subgroup_properties.subgroupSize;
    const bool has_group_non_uniform =
        (subgroup_properties.supportedOperations &
         VK_SUBGROUP_FEATURE_BASIC_BIT) != 0;
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts, has_group_non_uniform, LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM,
        loomc_make_cstring_view("vulkan:subgroupSupportedOperations")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts,
        has_group_non_uniform && (subgroup_properties.supportedOperations &
                                  VK_SUBGROUP_FEATURE_BALLOT_BIT) != 0,
        LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_BALLOT,
        loomc_make_cstring_view("vulkan:subgroupSupportedOperations")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts,
        has_group_non_uniform && (subgroup_properties.supportedOperations &
                                  VK_SUBGROUP_FEATURE_VOTE_BIT) != 0,
        LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_VOTE,
        loomc_make_cstring_view("vulkan:subgroupSupportedOperations")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts,
        has_group_non_uniform && (subgroup_properties.supportedOperations &
                                  VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0,
        LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_ARITHMETIC,
        loomc_make_cstring_view("vulkan:subgroupSupportedOperations")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts,
        has_group_non_uniform && (subgroup_properties.supportedOperations &
                                  VK_SUBGROUP_FEATURE_SHUFFLE_BIT) != 0,
        LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_SHUFFLE,
        loomc_make_cstring_view("vulkan:subgroupSupportedOperations")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts,
        has_group_non_uniform &&
            (subgroup_properties.supportedOperations &
             VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) != 0,
        LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_SHUFFLE_RELATIVE,
        loomc_make_cstring_view("vulkan:subgroupSupportedOperations")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts,
        has_group_non_uniform && (subgroup_properties.supportedOperations &
                                  VK_SUBGROUP_FEATURE_CLUSTERED_BIT) != 0,
        LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_CLUSTERED,
        loomc_make_cstring_view("vulkan:subgroupSupportedOperations")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_limit_fact(
        facts, LOOMC_SPIRV_LIMIT_DEFAULT_SUBGROUP_SIZE,
        subgroup_properties.subgroupSize,
        loomc_make_cstring_view("vulkan:subgroupSize")));
  }

  if (has_vulkan_12) {
    const loomc_spirv_vulkan_float_controls_t controls = {
        .denorm_independence = vulkan12_properties.denormBehaviorIndependence,
        .denorm_preserve =
            {
                .f16 = vulkan12_properties.shaderDenormPreserveFloat16,
                .f32 = vulkan12_properties.shaderDenormPreserveFloat32,
                .f64 = vulkan12_properties.shaderDenormPreserveFloat64,
            },
        .signed_zero_inf_nan_preserve =
            {
                .f16 =
                    vulkan12_properties.shaderSignedZeroInfNanPreserveFloat16,
                .f32 =
                    vulkan12_properties.shaderSignedZeroInfNanPreserveFloat32,
                .f64 =
                    vulkan12_properties.shaderSignedZeroInfNanPreserveFloat64,
            },
        .rounding_independence = vulkan12_properties.roundingModeIndependence,
        .rounding_rte =
            {
                .f16 = vulkan12_properties.shaderRoundingModeRTEFloat16,
                .f32 = vulkan12_properties.shaderRoundingModeRTEFloat32,
                .f64 = vulkan12_properties.shaderRoundingModeRTEFloat64,
            },
    };
    return loomc_spirv_vulkan_add_float_control_features(facts, &controls);
  }
  if (query_float_controls_extension) {
    const loomc_spirv_vulkan_float_controls_t controls = {
        .denorm_independence =
            float_controls_properties.denormBehaviorIndependence,
        .denorm_preserve =
            {
                .f16 = float_controls_properties.shaderDenormPreserveFloat16,
                .f32 = float_controls_properties.shaderDenormPreserveFloat32,
                .f64 = float_controls_properties.shaderDenormPreserveFloat64,
            },
        .signed_zero_inf_nan_preserve =
            {
                .f16 = float_controls_properties
                           .shaderSignedZeroInfNanPreserveFloat16,
                .f32 = float_controls_properties
                           .shaderSignedZeroInfNanPreserveFloat32,
                .f64 = float_controls_properties
                           .shaderSignedZeroInfNanPreserveFloat64,
            },
        .rounding_independence =
            float_controls_properties.roundingModeIndependence,
        .rounding_rte =
            {
                .f16 = float_controls_properties.shaderRoundingModeRTEFloat16,
                .f32 = float_controls_properties.shaderRoundingModeRTEFloat32,
                .f64 = float_controls_properties.shaderRoundingModeRTEFloat64,
            },
    };
    return loomc_spirv_vulkan_add_float_control_features(facts, &controls);
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_vulkan_query_features(
    const loomc_spirv_vulkan_profile_options_t* options, uint32_t api_version,
    const loomc_spirv_vulkan_extensions_t* extensions, uint32_t subgroup_size,
    const VkPhysicalDeviceSubgroupSizeControlProperties* subgroup_control,
    loomc_spirv_vulkan_profile_facts_t* facts) {
  VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_control_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
  };
  VkPhysicalDeviceVulkan12Features vulkan12_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
  };
  VkPhysicalDeviceShaderFloat16Int8Features float16_int8_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
  };
  VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures subgroup_type_features = {
      .sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES,
  };
  VkPhysicalDevice8BitStorageFeatures storage8_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
  };
  VkPhysicalDevice16BitStorageFeatures storage16_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
  };
  VkPhysicalDeviceBufferDeviceAddressFeatures bda_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
  };
  VkPhysicalDeviceShaderAtomicInt64Features atomic_int64_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES,
  };
  VkPhysicalDeviceVulkanMemoryModelFeatures memory_model_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES,
  };
  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_features = {
      .sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
  };
  VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT atomic_float2_features = {
      .sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT,
  };

  VkPhysicalDeviceShaderFmaFeaturesKHR fma_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR,
  };
  VkPhysicalDeviceShaderFloatControls2Features float_controls2_features = {
      .sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES,
  };

  void* feature_chain = NULL;
  if (api_version >= VK_API_VERSION_1_4 ||
      extensions->shader_float_controls2_khr) {
    loomc_spirv_vulkan_link_out_struct(&feature_chain,
                                       &float_controls2_features);
  }
  if (extensions->shader_fma_khr) {
    loomc_spirv_vulkan_link_out_struct(&feature_chain, &fma_features);
  }
  if (api_version >= VK_API_VERSION_1_3 ||
      extensions->subgroup_size_control_ext) {
    loomc_spirv_vulkan_link_out_struct(&feature_chain,
                                       &subgroup_control_features);
  }
#if defined(VK_KHR_cooperative_matrix)
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative_matrix_features = {
      .sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
  };
  if (extensions->cooperative_matrix_khr) {
    loomc_spirv_vulkan_link_out_struct(&feature_chain,
                                       &cooperative_matrix_features);
  }
#endif  // defined(VK_KHR_cooperative_matrix)
  const bool has_vulkan_11 = api_version >= VK_API_VERSION_1_1;
  const bool has_vulkan_12 = api_version >= VK_API_VERSION_1_2;
  const bool query_shader_float16_int8_extension =
      !has_vulkan_12 && extensions->shader_float16_int8_khr;
  const bool query_subgroup_type_extension =
      !has_vulkan_12 && extensions->shader_subgroup_extended_types_khr;
  const bool query_storage_8bit_extension =
      !has_vulkan_12 && extensions->storage_8bit_khr;
  const bool query_storage_16bit_features =
      has_vulkan_11 || extensions->storage_16bit_khr;
  const bool query_buffer_device_address_extension =
      !has_vulkan_12 && extensions->buffer_device_address_khr;
  const bool query_atomic_int64_extension =
      !has_vulkan_12 && extensions->shader_atomic_int64_khr;
  const bool query_vulkan_memory_model_extension =
      !has_vulkan_12 && extensions->vulkan_memory_model_khr;
  if (extensions->shader_atomic_float2_ext) {
    loomc_spirv_vulkan_link_out_struct(&feature_chain, &atomic_float2_features);
  }
  if (extensions->shader_atomic_float_ext) {
    loomc_spirv_vulkan_link_out_struct(&feature_chain, &atomic_float_features);
  }
  if (query_vulkan_memory_model_extension) {
    loomc_spirv_vulkan_link_out_struct(&feature_chain, &memory_model_features);
  }
  if (query_atomic_int64_extension) {
    loomc_spirv_vulkan_link_out_struct(&feature_chain, &atomic_int64_features);
  }
  if (query_buffer_device_address_extension) {
    loomc_spirv_vulkan_link_out_struct(&feature_chain, &bda_features);
  }
  if (query_storage_16bit_features) {
    loomc_spirv_vulkan_link_out_struct(&feature_chain, &storage16_features);
  }
  if (query_storage_8bit_extension) {
    loomc_spirv_vulkan_link_out_struct(&feature_chain, &storage8_features);
  }
  if (query_shader_float16_int8_extension) {
    loomc_spirv_vulkan_link_out_struct(&feature_chain, &float16_int8_features);
  }
  if (query_subgroup_type_extension) {
    loomc_spirv_vulkan_link_out_struct(&feature_chain, &subgroup_type_features);
  }
  if (has_vulkan_12) {
    loomc_spirv_vulkan_link_out_struct(&feature_chain, &vulkan12_features);
  }

  VkPhysicalDeviceFeatures2 features2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = feature_chain,
  };
  IREE_LEAK_CHECK_DISABLE_PUSH();
  options->functions->get_physical_device_features2(options->physical_device,
                                                    &features2);
  IREE_LEAK_CHECK_DISABLE_POP();

  if (has_vulkan_11) {
    const bool supports_compute_control =
        subgroup_control_features.subgroupSizeControl &&
        (subgroup_control->requiredSubgroupSizeStages &
         VK_SHADER_STAGE_COMPUTE_BIT) != 0;
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_limit_fact(
        facts, LOOMC_SPIRV_LIMIT_MIN_SUBGROUP_SIZE,
        supports_compute_control ? subgroup_control->minSubgroupSize
                                 : subgroup_size,
        loomc_make_cstring_view("vulkan:minSubgroupSize")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_limit_fact(
        facts, LOOMC_SPIRV_LIMIT_MAX_SUBGROUP_SIZE,
        supports_compute_control ? subgroup_control->maxSubgroupSize
                                 : subgroup_size,
        loomc_make_cstring_view("vulkan:maxSubgroupSize")));
    if (supports_compute_control) {
      LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_limit_fact(
          facts, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SUBGROUP_COUNT,
          subgroup_control->maxComputeWorkgroupSubgroups,
          loomc_make_cstring_view("vulkan:maxComputeWorkgroupSubgroups")));
    }
  }

  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_feature_fact(
      facts, LOOMC_SPIRV_FEATURE_VULKAN_SHADER, LOOMC_TARGET_FACT_STATE_TRUE,
      loomc_make_cstring_view("vulkan:physicalDevice")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
      facts, float_controls2_features.shaderFloatControls2,
      LOOMC_SPIRV_FEATURE_FLOAT_CONTROLS2,
      loomc_make_cstring_view("vulkan:shaderFloatControls2")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
      facts, extensions->shader_fma_khr && fma_features.shaderFmaFloat16,
      LOOMC_SPIRV_FEATURE_FMA_FLOAT16,
      loomc_make_cstring_view("vulkan:shaderFmaFloat16")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
      facts, extensions->shader_fma_khr && fma_features.shaderFmaFloat32,
      LOOMC_SPIRV_FEATURE_FMA_FLOAT32,
      loomc_make_cstring_view("vulkan:shaderFmaFloat32")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
      facts, extensions->shader_fma_khr && fma_features.shaderFmaFloat64,
      LOOMC_SPIRV_FEATURE_FMA_FLOAT64,
      loomc_make_cstring_view("vulkan:shaderFmaFloat64")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
      facts, features2.features.shaderFloat64, LOOMC_SPIRV_FEATURE_FLOAT64,
      loomc_make_cstring_view("vulkan:shaderFloat64")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
      facts, features2.features.shaderInt16, LOOMC_SPIRV_FEATURE_INT16,
      loomc_make_cstring_view("vulkan:shaderInt16")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
      facts, features2.features.shaderInt64, LOOMC_SPIRV_FEATURE_INT64,
      loomc_make_cstring_view("vulkan:shaderInt64")));
  if (has_vulkan_12) {
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts, vulkan12_features.shaderSubgroupExtendedTypes,
        LOOMC_SPIRV_FEATURE_SUBGROUP_EXTENDED_TYPES,
        loomc_make_cstring_view("vulkan:shaderSubgroupExtendedTypes")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts, vulkan12_features.shaderFloat16, LOOMC_SPIRV_FEATURE_FLOAT16,
        loomc_make_cstring_view("vulkan:shaderFloat16")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts, vulkan12_features.shaderInt8, LOOMC_SPIRV_FEATURE_INT8,
        loomc_make_cstring_view("vulkan:shaderInt8")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts, vulkan12_features.storageBuffer8BitAccess,
        LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS,
        loomc_make_cstring_view("vulkan:storageBuffer8BitAccess")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts, vulkan12_features.bufferDeviceAddress,
        LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER,
        loomc_make_cstring_view("vulkan:bufferDeviceAddress")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts,
        vulkan12_features.vulkanMemoryModel &&
            vulkan12_features.vulkanMemoryModelDeviceScope,
        LOOMC_SPIRV_FEATURE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE,
        loomc_make_cstring_view(
            "vulkan:vulkanMemoryModel+vulkanMemoryModelDeviceScope")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts,
        features2.features.shaderInt64 &&
            vulkan12_features.shaderBufferInt64Atomics,
        LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_INT64_ATOMICS,
        loomc_make_cstring_view("vulkan:shaderBufferInt64Atomics")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts,
        features2.features.shaderInt64 &&
            vulkan12_features.shaderSharedInt64Atomics,
        LOOMC_SPIRV_FEATURE_WORKGROUP_INT64_ATOMICS,
        loomc_make_cstring_view("vulkan:shaderSharedInt64Atomics")));
  }
  if (query_storage_16bit_features) {
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts, storage16_features.storageBuffer16BitAccess,
        LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_16BIT_ACCESS,
        loomc_make_cstring_view("vulkan:storageBuffer16BitAccess")));
  }
  if (query_shader_float16_int8_extension) {
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts, float16_int8_features.shaderFloat16, LOOMC_SPIRV_FEATURE_FLOAT16,
        loomc_make_cstring_view("vulkan:shaderFloat16")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts, float16_int8_features.shaderInt8, LOOMC_SPIRV_FEATURE_INT8,
        loomc_make_cstring_view("vulkan:shaderInt8")));
  }
  if (query_storage_8bit_extension) {
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts, storage8_features.storageBuffer8BitAccess,
        LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS,
        loomc_make_cstring_view("vulkan:storageBuffer8BitAccess")));
  }
  if (query_subgroup_type_extension) {
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts, subgroup_type_features.shaderSubgroupExtendedTypes,
        LOOMC_SPIRV_FEATURE_SUBGROUP_EXTENDED_TYPES,
        loomc_make_cstring_view("vulkan:shaderSubgroupExtendedTypes")));
  }
  if (query_buffer_device_address_extension) {
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts, bda_features.bufferDeviceAddress,
        LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER,
        loomc_make_cstring_view("vulkan:bufferDeviceAddress")));
  }
  if (query_atomic_int64_extension) {
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts,
        features2.features.shaderInt64 &&
            atomic_int64_features.shaderBufferInt64Atomics,
        LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_INT64_ATOMICS,
        loomc_make_cstring_view("vulkan:shaderBufferInt64Atomics")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts,
        features2.features.shaderInt64 &&
            atomic_int64_features.shaderSharedInt64Atomics,
        LOOMC_SPIRV_FEATURE_WORKGROUP_INT64_ATOMICS,
        loomc_make_cstring_view("vulkan:shaderSharedInt64Atomics")));
  }
  if (query_vulkan_memory_model_extension) {
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts,
        memory_model_features.vulkanMemoryModel &&
            memory_model_features.vulkanMemoryModelDeviceScope,
        LOOMC_SPIRV_FEATURE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE,
        loomc_make_cstring_view(
            "vulkan:vulkanMemoryModel+vulkanMemoryModelDeviceScope")));
  }
  if (extensions->shader_atomic_float_ext) {
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts, atomic_float_features.shaderBufferFloat32Atomics,
        LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT32_ATOMICS,
        loomc_make_cstring_view("vulkan:shaderBufferFloat32Atomics")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts, atomic_float_features.shaderSharedFloat32Atomics,
        LOOMC_SPIRV_FEATURE_WORKGROUP_FLOAT32_ATOMICS,
        loomc_make_cstring_view("vulkan:shaderSharedFloat32Atomics")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts, atomic_float_features.shaderBufferFloat32AtomicAdd,
        LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT32_ATOMIC_ADD,
        loomc_make_cstring_view("vulkan:shaderBufferFloat32AtomicAdd")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts, atomic_float_features.shaderSharedFloat32AtomicAdd,
        LOOMC_SPIRV_FEATURE_WORKGROUP_FLOAT32_ATOMIC_ADD,
        loomc_make_cstring_view("vulkan:shaderSharedFloat32AtomicAdd")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts,
        features2.features.shaderFloat64 &&
            atomic_float_features.shaderBufferFloat64Atomics,
        LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT64_ATOMICS,
        loomc_make_cstring_view("vulkan:shaderBufferFloat64Atomics")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts,
        features2.features.shaderFloat64 &&
            atomic_float_features.shaderSharedFloat64Atomics,
        LOOMC_SPIRV_FEATURE_WORKGROUP_FLOAT64_ATOMICS,
        loomc_make_cstring_view("vulkan:shaderSharedFloat64Atomics")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts,
        features2.features.shaderFloat64 &&
            atomic_float_features.shaderBufferFloat64AtomicAdd,
        LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT64_ATOMIC_ADD,
        loomc_make_cstring_view("vulkan:shaderBufferFloat64AtomicAdd")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts,
        features2.features.shaderFloat64 &&
            atomic_float_features.shaderSharedFloat64AtomicAdd,
        LOOMC_SPIRV_FEATURE_WORKGROUP_FLOAT64_ATOMIC_ADD,
        loomc_make_cstring_view("vulkan:shaderSharedFloat64AtomicAdd")));
  }
  if (extensions->shader_atomic_float2_ext) {
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts,
        (has_vulkan_12 ? vulkan12_features.shaderFloat16
                       : float16_int8_features.shaderFloat16) &&
            atomic_float2_features.shaderBufferFloat16Atomics,
        LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT16_ATOMICS,
        loomc_make_cstring_view("vulkan:shaderBufferFloat16Atomics")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts,
        (has_vulkan_12 ? vulkan12_features.shaderFloat16
                       : float16_int8_features.shaderFloat16) &&
            atomic_float2_features.shaderSharedFloat16Atomics,
        LOOMC_SPIRV_FEATURE_WORKGROUP_FLOAT16_ATOMICS,
        loomc_make_cstring_view("vulkan:shaderSharedFloat16Atomics")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts,
        (has_vulkan_12 ? vulkan12_features.shaderFloat16
                       : float16_int8_features.shaderFloat16) &&
            atomic_float2_features.shaderBufferFloat16AtomicAdd,
        LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT16_ATOMIC_ADD,
        loomc_make_cstring_view("vulkan:shaderBufferFloat16AtomicAdd")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts,
        (has_vulkan_12 ? vulkan12_features.shaderFloat16
                       : float16_int8_features.shaderFloat16) &&
            atomic_float2_features.shaderSharedFloat16AtomicAdd,
        LOOMC_SPIRV_FEATURE_WORKGROUP_FLOAT16_ATOMIC_ADD,
        loomc_make_cstring_view("vulkan:shaderSharedFloat16AtomicAdd")));
  }
#if defined(VK_KHR_cooperative_matrix)
  if (extensions->cooperative_matrix_khr) {
    return loomc_spirv_vulkan_add_bool_feature(
        facts, cooperative_matrix_features.cooperativeMatrix,
        LOOMC_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR,
        loomc_make_cstring_view("vulkan:cooperativeMatrix"));
  }
#endif  // defined(VK_KHR_cooperative_matrix)
  return loomc_ok_status();
}

loomc_status_t loomc_target_profile_create_spirv_vulkan(
    loomc_target_environment_t* target_environment,
    const loomc_spirv_vulkan_profile_options_t* options,
    loomc_allocator_t allocator, loomc_target_profile_t** out_profile,
    loomc_result_t** out_result) {
  if (out_profile != NULL) {
    *out_profile = NULL;
  }
  if (out_result != NULL) {
    *out_result = NULL;
  }
  if (target_environment == NULL || out_profile == NULL || out_result == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "target_environment, out_profile, and out_result must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_validate_options(options));

  loomc_spirv_vulkan_profile_facts_t facts = {0};
  uint32_t api_version = 0;
  uint32_t subgroup_size = 0;
  VkPhysicalDeviceSubgroupSizeControlProperties subgroup_control = {0};
  loomc_spirv_vulkan_extensions_t extensions = {0};
  IREE_LEAK_CHECK_DISABLE_PUSH();
  loomc_status_t status =
      loomc_spirv_vulkan_query_extensions(options, allocator, &extensions);
  if (loomc_status_is_ok(status)) {
    status = loomc_spirv_vulkan_query_properties(options, &extensions, &facts,
                                                 &api_version, &subgroup_size,
                                                 &subgroup_control);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_spirv_vulkan_query_features(options, api_version,
                                               &extensions, subgroup_size,
                                               &subgroup_control, &facts);
  }
  IREE_LEAK_CHECK_DISABLE_POP();
  LOOMC_RETURN_IF_ERROR(status);

  loomc_spirv_profile_options_t profile_options = {
      .type = LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      .structure_size = sizeof(profile_options),
      .identifier = !loomc_string_view_is_empty(options->identifier)
                        ? options->identifier
                        : loomc_make_cstring_view("<vulkan-profile>"),
      .preset = LOOMC_SPIRV_PROFILE_PRESET_NONE,
      .feature_facts = facts.feature_facts,
      .feature_fact_count = facts.feature_fact_count,
      .limit_facts = facts.limit_facts,
      .limit_fact_count = facts.limit_fact_count,
      .environment_facts = facts.environment_facts,
      .environment_fact_count = facts.environment_fact_count,
  };
  return loomc_target_profile_create_spirv(target_environment, &profile_options,
                                           allocator, out_profile, out_result);
}
