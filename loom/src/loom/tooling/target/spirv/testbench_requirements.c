// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/spirv/testbench_requirements.h"

#include "loom/tooling/execution/hal/testbench_actual.h"
#include "loom/tooling/target/spirv/vulkan_profile.h"

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
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_context_ensure_runtime(context));
  if (context->artifact_provider == NULL ||
      !iree_string_view_equal(context->artifact_provider->hal_driver_name,
                              IREE_SV("vulkan"))) {
    *out_result = (loom_testbench_requirement_provider_result_t){
        .state = LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_UNAVAILABLE,
        .provider_code = IREE_SV("hal_driver_mismatch"),
        .display_message =
            IREE_SV("SPIR-V Vulkan requirement requires a Vulkan HAL device"),
    };
    return iree_ok_status();
  }

  loom_spirv_vulkan_hal_profile_facts_t facts = {0};
  iree_hal_vulkan_cooperative_matrix_property_t* device_rows = NULL;
  iree_host_size_t device_row_count = 0;
  loom_spirv_vulkan_hal_target_profile_storage_t profile_storage = {0};
  bool profile_storage_initialized = false;

  iree_status_t status = loom_spirv_vulkan_hal_profile_query(
      context->runtime.device, context->runtime.executable_cache, &facts);
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

void loom_spirv_vulkan_hal_testbench_requirement_provider_initialize(
    loom_run_hal_testbench_context_t* context,
    loom_testbench_requirement_provider_t* out_provider) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = (loom_testbench_requirement_provider_t){
      .name = IREE_SV("hal.spirv.cooperative_matrix"),
      .user_data = context,
      .query =
          loom_spirv_vulkan_hal_testbench_query_cooperative_matrix_requirement,
  };
}
