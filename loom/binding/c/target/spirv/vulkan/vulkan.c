// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/spirv/vulkan.h"

#include <string.h>

#include "iree/base/internal/debugging.h"

enum {
  LOOMC_SPIRV_VULKAN_FEATURE_FACT_CAPACITY = 24,
  LOOMC_SPIRV_VULKAN_LIMIT_FACT_CAPACITY = 16,
  LOOMC_SPIRV_VULKAN_ENVIRONMENT_FACT_CAPACITY = 4,
};

typedef struct loomc_spirv_vulkan_profile_facts_t {
  // Feature facts normalized from Vulkan feature structs.
  loomc_spirv_feature_fact_t
      feature_facts[LOOMC_SPIRV_VULKAN_FEATURE_FACT_CAPACITY];

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

typedef struct loomc_spirv_vulkan_extensions_t {
  // VK_KHR_shader_float16_int8 is supported.
  bool shader_float16_int8_khr;

  // VK_KHR_8bit_storage is supported.
  bool storage_8bit_khr;

  // VK_KHR_16bit_storage is supported.
  bool storage_16bit_khr;

  // VK_KHR_buffer_device_address is supported.
  bool buffer_device_address_khr;

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
  if (facts->feature_fact_count >= LOOMC_SPIRV_VULKAN_FEATURE_FACT_CAPACITY) {
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
                 extension, VK_KHR_8BIT_STORAGE_EXTENSION_NAME)) {
    out_extensions->storage_8bit_khr = true;
  } else if (loomc_spirv_vulkan_extension_name_equal(
                 extension, VK_KHR_16BIT_STORAGE_EXTENSION_NAME)) {
    out_extensions->storage_16bit_khr = true;
  } else if (loomc_spirv_vulkan_extension_name_equal(
                 extension, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)) {
    out_extensions->buffer_device_address_khr = true;
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
    loomc_spirv_vulkan_profile_facts_t* facts, uint32_t* out_api_version) {
  VkPhysicalDeviceProperties2 properties2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
  };
  IREE_LEAK_CHECK_DISABLE_PUSH();
  options->functions->get_physical_device_properties2(options->physical_device,
                                                      &properties2);
  IREE_LEAK_CHECK_DISABLE_POP();
  *out_api_version = properties2.properties.apiVersion;

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

  if (*out_api_version < VK_API_VERSION_1_1) {
    return loomc_ok_status();
  }
  VkPhysicalDeviceSubgroupProperties subgroup_properties = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
  };
  properties2 = (VkPhysicalDeviceProperties2){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &subgroup_properties,
  };
  IREE_LEAK_CHECK_DISABLE_PUSH();
  options->functions->get_physical_device_properties2(options->physical_device,
                                                      &properties2);
  IREE_LEAK_CHECK_DISABLE_POP();
  return loomc_spirv_vulkan_add_limit_fact(
      facts, LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE, subgroup_properties.subgroupSize,
      loomc_make_cstring_view("vulkan:subgroupSize"));
}

static loomc_status_t loomc_spirv_vulkan_query_features(
    const loomc_spirv_vulkan_profile_options_t* options, uint32_t api_version,
    const loomc_spirv_vulkan_extensions_t* extensions,
    loomc_spirv_vulkan_profile_facts_t* facts) {
  VkPhysicalDeviceVulkan12Features vulkan12_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
  };
  VkPhysicalDeviceShaderFloat16Int8Features float16_int8_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
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

  void* feature_chain = NULL;
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
  const bool query_storage_8bit_extension =
      !has_vulkan_12 && extensions->storage_8bit_khr;
  const bool query_storage_16bit_features =
      has_vulkan_11 || extensions->storage_16bit_khr;
  const bool query_buffer_device_address_extension =
      !has_vulkan_12 && extensions->buffer_device_address_khr;
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

  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_feature_fact(
      facts, LOOMC_SPIRV_FEATURE_VULKAN_SHADER, LOOMC_TARGET_FACT_STATE_TRUE,
      loomc_make_cstring_view("vulkan:physicalDevice")));
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
  if (query_buffer_device_address_extension) {
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkan_add_bool_feature(
        facts, bda_features.bufferDeviceAddress,
        LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER,
        loomc_make_cstring_view("vulkan:bufferDeviceAddress")));
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
  loomc_spirv_vulkan_extensions_t extensions = {0};
  IREE_LEAK_CHECK_DISABLE_PUSH();
  loomc_status_t status =
      loomc_spirv_vulkan_query_properties(options, &facts, &api_version);
  if (loomc_status_is_ok(status)) {
    status =
        loomc_spirv_vulkan_query_extensions(options, allocator, &extensions);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_spirv_vulkan_query_features(options, api_version,
                                               &extensions, &facts);
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
