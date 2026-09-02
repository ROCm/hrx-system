// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/spirv/testbench_requirements.h"

#include "iree/hal/drivers/vulkan/device_spec.h"
#include "loom/tooling/execution/hal/testbench_actual.h"
#include "loom/tooling/target/spirv/vulkan_profile.h"

typedef struct loom_spirv_vulkan_feature_requirement_t {
  // Stable requirement feature name accepted by hal.vulkan.feature.
  iree_string_view_t name;
  // Vulkan HAL features that must be enabled on the logical device.
  iree_hal_vulkan_features_t required_features;
} loom_spirv_vulkan_feature_requirement_t;

static const loom_spirv_vulkan_feature_requirement_t
    kLoomSpirvVulkanFeatureRequirements[] = {
        {
            .name = IREE_SVL("buffer_device_address"),
            .required_features.general =
                IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES,
        },
        {
            .name = IREE_SVL("cooperative_matrix_khr"),
            .required_features.general =
                IREE_HAL_VULKAN_FEATURE_ENABLE_COOPERATIVE_MATRIX,
        },
        {
            .name = IREE_SVL("shader_bfloat16_cooperative_matrix"),
            .required_features.general =
                IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_COOPERATIVE_MATRIX,
        },
        {
            .name = IREE_SVL("shader_bfloat16_dot_product"),
            .required_features.general =
                IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_DOT_PRODUCT,
        },
        {
            .name = IREE_SVL("shader_bfloat16_type"),
            .required_features.general =
                IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_TYPE,
        },
        {
            .name = IREE_SVL("shader_buffer_float16_atomic_add"),
            .required_features.atomics =
                IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT16_ADD,
        },
        {
            .name = IREE_SVL("shader_buffer_float16_atomics"),
            .required_features.atomics =
                IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT16,
        },
        {
            .name = IREE_SVL("shader_buffer_float32_atomic_add"),
            .required_features.atomics =
                IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT32_ADD,
        },
        {
            .name = IREE_SVL("shader_buffer_float32_atomics"),
            .required_features.atomics =
                IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT32,
        },
        {
            .name = IREE_SVL("shader_buffer_float64_atomic_add"),
            .required_features.atomics =
                IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT64_ADD,
        },
        {
            .name = IREE_SVL("shader_buffer_float64_atomics"),
            .required_features.atomics =
                IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT64,
        },
        {
            .name = IREE_SVL("shader_buffer_int64_atomics"),
            .required_features.atomics =
                IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_INT64,
        },
        {
            .name = IREE_SVL("shader_shared_float16_atomic_add"),
            .required_features.atomics =
                IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT16_ADD,
        },
        {
            .name = IREE_SVL("shader_shared_float16_atomics"),
            .required_features.atomics =
                IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT16,
        },
        {
            .name = IREE_SVL("shader_shared_float32_atomic_add"),
            .required_features.atomics =
                IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT32_ADD,
        },
        {
            .name = IREE_SVL("shader_shared_float32_atomics"),
            .required_features.atomics =
                IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT32,
        },
        {
            .name = IREE_SVL("shader_shared_float64_atomic_add"),
            .required_features.atomics =
                IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT64_ADD,
        },
        {
            .name = IREE_SVL("shader_shared_float64_atomics"),
            .required_features.atomics =
                IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT64,
        },
        {
            .name = IREE_SVL("shader_shared_int64_atomics"),
            .required_features.atomics =
                IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_INT64,
        },
        {
            .name = IREE_SVL("shader_float16"),
            .required_features.general =
                IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16,
        },
        {
            .name = IREE_SVL("shader_float64"),
            .required_features.general =
                IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT64,
        },
        {
            .name = IREE_SVL("shader_int8"),
            .required_features.general =
                IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT8,
        },
        {
            .name = IREE_SVL("shader_int16"),
            .required_features.general =
                IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT16,
        },
        {
            .name = IREE_SVL("shader_int64"),
            .required_features.general =
                IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT64,
        },
        {
            .name = IREE_SVL("shader_integer_dot_product"),
            .required_features.general =
                IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INTEGER_DOT_PRODUCT,
        },
        {
            .name = IREE_SVL("storage_buffer_8bit_access"),
            .required_features.general =
                IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_8BIT_ACCESS,
        },
        {
            .name = IREE_SVL("storage_buffer_16bit_access"),
            .required_features.general =
                IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_16BIT_ACCESS,
        },
        {
            .name = IREE_SVL("subgroup_size_control"),
            .required_features.general =
                IREE_HAL_VULKAN_FEATURE_ENABLE_SUBGROUP_SIZE_CONTROL,
        },
        {
            .name = IREE_SVL("vulkan_memory_model"),
            .required_features.general =
                IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL,
        },
        {
            .name = IREE_SVL("vulkan_memory_model_device_scope"),
            .required_features.general =
                IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE,
        },
};

static const loom_spirv_vulkan_feature_requirement_t*
loom_spirv_vulkan_feature_requirement_lookup(iree_string_view_t name) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomSpirvVulkanFeatureRequirements); ++i) {
    const loom_spirv_vulkan_feature_requirement_t* requirement =
        &kLoomSpirvVulkanFeatureRequirements[i];
    if (iree_string_view_equal(requirement->name, name)) {
      return requirement;
    }
  }
  return NULL;
}

static iree_status_t loom_spirv_vulkan_testbench_decode_device_spec(
    loom_run_hal_testbench_context_t* context,
    iree_hal_vulkan_device_spec_t* out_vulkan_spec) {
  const iree_hal_device_spec_t* device_spec =
      iree_hal_device_spec(context->runtime.device);
  if (device_spec == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan HAL device does not expose an immutable device spec");
  }
  const iree_hal_device_spec_facet_t* facet =
      iree_hal_vulkan_device_spec_find_facet(device_spec);
  if (facet == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan HAL device spec does not expose a Vulkan facet");
  }
  return iree_hal_vulkan_device_spec_decode_facet(facet, out_vulkan_spec);
}

static iree_status_t loom_spirv_vulkan_testbench_require_vulkan_device(
    loom_run_hal_testbench_context_t* context,
    loom_testbench_requirement_provider_result_t* out_result) {
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_context_ensure_runtime(context));
  if (context->device_provider != NULL &&
      iree_string_view_equal(context->device_provider->driver_name,
                             IREE_SV("vulkan"))) {
    return iree_ok_status();
  }
  *out_result = (loom_testbench_requirement_provider_result_t){
      .state = LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_UNAVAILABLE,
      .provider_code = IREE_SV("hal_driver_mismatch"),
      .display_message =
          IREE_SV("Vulkan requirement requires a Vulkan HAL device"),
  };
  return iree_ok_status();
}

static iree_status_t loom_spirv_vulkan_hal_testbench_query_feature_requirement(
    void* user_data, const loom_module_t* module, loom_named_attr_slice_t attrs,
    loom_testbench_requirement_provider_result_t* out_result) {
  loom_run_hal_testbench_context_t* context =
      (loom_run_hal_testbench_context_t*)user_data;
  *out_result = (loom_testbench_requirement_provider_result_t){
      .state = LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_UNSATISFIED,
      .provider_code = IREE_SV("feature_disabled"),
      .display_message =
          IREE_SV("Vulkan device does not expose requested feature"),
  };

  iree_string_view_t feature_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_testbench_requirement_read_string_attr(
      module, attrs, IREE_SV("feature"), &feature_name));
  const loom_spirv_vulkan_feature_requirement_t* requirement =
      loom_spirv_vulkan_feature_requirement_lookup(feature_name);
  if (requirement == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown Vulkan feature requirement `%.*s`",
                            (int)feature_name.size, feature_name.data);
  }
  IREE_RETURN_IF_ERROR(
      loom_spirv_vulkan_testbench_require_vulkan_device(context, out_result));
  if (out_result->state ==
      LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_UNAVAILABLE) {
    return iree_ok_status();
  }

  iree_hal_vulkan_device_spec_t vulkan_spec = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_vulkan_testbench_decode_device_spec(context, &vulkan_spec));
  const bool satisfied =
      iree_all_bits_set(vulkan_spec.enabled_features.general,
                        requirement->required_features.general) &&
      iree_all_bits_set(vulkan_spec.enabled_features.atomics,
                        requirement->required_features.atomics);
  out_result->state =
      satisfied ? LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_SATISFIED
                : LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_UNSATISFIED;
  if (satisfied) {
    out_result->provider_code = iree_string_view_empty();
    out_result->display_message = iree_string_view_empty();
  }
  return iree_ok_status();
}

static const loom_spirv_cooperative_matrix_property_t*
loom_spirv_vulkan_hal_testbench_find_matrix_row(
    const loom_spirv_target_profile_t* profile, iree_string_view_t row_name) {
  if (profile == NULL || profile->cooperative_properties == NULL) {
    return NULL;
  }
  const loom_spirv_cooperative_property_set_t* properties =
      profile->cooperative_properties;
  for (uint16_t i = 0; i < properties->matrix_property_count; ++i) {
    const loom_spirv_cooperative_matrix_property_t* row =
        &properties->matrix_properties[i];
    if (iree_string_view_equal(row->name, row_name)) {
      return row;
    }
  }
  return NULL;
}

static iree_status_t
loom_spirv_vulkan_hal_testbench_query_cooperative_matrix_requirement(
    void* user_data, const loom_module_t* module, loom_named_attr_slice_t attrs,
    loom_testbench_requirement_provider_result_t* out_result) {
  loom_run_hal_testbench_context_t* context =
      (loom_run_hal_testbench_context_t*)user_data;
  *out_result = (loom_testbench_requirement_provider_result_t){
      .state = LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_UNSATISFIED,
      .provider_code = IREE_SV("cooperative_matrix_row_unavailable"),
      .display_message =
          IREE_SV("Vulkan device does not expose requested cooperative matrix "
                  "row"),
  };

  iree_string_view_t row_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_testbench_requirement_read_string_attr(
      module, attrs, IREE_SV("row"), &row_name));
  IREE_RETURN_IF_ERROR(
      loom_spirv_vulkan_testbench_require_vulkan_device(context, out_result));
  if (out_result->state ==
      LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_UNAVAILABLE) {
    return iree_ok_status();
  }

  loom_spirv_vulkan_hal_profile_facts_t facts = {0};
  iree_hal_vulkan_cooperative_matrix_property_t* device_rows = NULL;
  iree_host_size_t device_row_count = 0;
  loom_spirv_vulkan_hal_target_profile_storage_t profile_storage = {0};
  bool profile_storage_initialized = false;

  iree_status_t status =
      loom_spirv_vulkan_hal_profile_query(context->runtime.device, &facts);
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(
          facts.flags,
          LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_COOPERATIVE_MATRIX_KHR)) {
    status = loom_spirv_vulkan_hal_query_cooperative_matrix_properties(
        context->runtime.device, context->host_allocator, &device_rows,
        &device_row_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_spirv_vulkan_hal_target_profile_storage_initialize(
        &facts, device_rows, device_row_count, context->host_allocator,
        &profile_storage);
    profile_storage_initialized = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    const bool satisfied = loom_spirv_vulkan_hal_testbench_find_matrix_row(
                               &profile_storage.profile, row_name) != NULL;
    out_result->state =
        satisfied ? LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_SATISFIED
                  : LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_UNSATISFIED;
    if (satisfied) {
      out_result->provider_code = iree_string_view_empty();
      out_result->display_message = iree_string_view_empty();
    }
  }

  if (profile_storage_initialized) {
    loom_spirv_vulkan_hal_target_profile_storage_deinitialize(
        &profile_storage, context->host_allocator);
  }
  iree_allocator_free(context->host_allocator, device_rows);
  return status;
}

void loom_spirv_vulkan_feature_testbench_requirement_provider_initialize(
    loom_run_hal_testbench_context_t* context,
    loom_testbench_requirement_provider_t* out_provider) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = (loom_testbench_requirement_provider_t){
      .name = IREE_SVL("hal.vulkan.feature"),
      .user_data = context,
      .query = loom_spirv_vulkan_hal_testbench_query_feature_requirement,
  };
}

void loom_spirv_vulkan_cooperative_matrix_testbench_requirement_provider_initialize(
    loom_run_hal_testbench_context_t* context,
    loom_testbench_requirement_provider_t* out_provider) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = (loom_testbench_requirement_provider_t){
      .name = IREE_SVL("hal.spirv.cooperative_matrix"),
      .user_data = context,
      .query =
          loom_spirv_vulkan_hal_testbench_query_cooperative_matrix_requirement,
  };
}
