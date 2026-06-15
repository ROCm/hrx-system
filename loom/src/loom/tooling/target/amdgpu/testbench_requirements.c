// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/amdgpu/testbench_requirements.h"

#include "loom/target/arch/amdgpu/target_info_defs.h"
#include "loom/tooling/execution/hal/testbench_actual.h"

static iree_status_t loom_amdgpu_hal_testbench_query_descriptor_set_requirement(
    void* user_data, const loom_module_t* module, loom_named_attr_slice_t attrs,
    loom_testbench_requirement_provider_result_t* out_result) {
  loom_run_hal_testbench_context_t* context =
      (loom_run_hal_testbench_context_t*)user_data;
  *out_result = (loom_testbench_requirement_provider_result_t){
      .state = LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_UNSATISFIED,
      .provider_code = IREE_SV("descriptor_set_mismatch"),
      .display_message =
          IREE_SV("AMDGPU device does not use requested descriptor set"),
  };

  iree_string_view_t required_descriptor_set = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_testbench_requirement_read_string_attr(
      module, attrs, IREE_SV("descriptor_set"), &required_descriptor_set));
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_context_ensure_runtime(context));
  if (context->artifact_provider == NULL ||
      !iree_string_view_equal(context->artifact_provider->hal_driver_name,
                              IREE_SV("amdgpu"))) {
    *out_result = (loom_testbench_requirement_provider_result_t){
        .state = LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_UNAVAILABLE,
        .provider_code = IREE_SV("hal_driver_mismatch"),
        .display_message =
            IREE_SV("AMDGPU requirement requires an AMDGPU HAL device"),
    };
    return iree_ok_status();
  }
  if (context->artifact_provider->select_device_target == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU requirement provider is missing a device "
                            "target selection hook");
  }

  loom_run_hal_device_target_t target = {0};
  IREE_RETURN_IF_ERROR(context->artifact_provider->select_device_target(
      context->artifact_provider, &context->runtime, context->host_allocator,
      &target));
  const loom_amdgpu_processor_info_t* processor =
      (const loom_amdgpu_processor_info_t*)target.data;
  const bool satisfied =
      processor != NULL && iree_string_view_equal(processor->descriptor_set.key,
                                                  required_descriptor_set);
  out_result->state =
      satisfied ? LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_SATISFIED
                : LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_UNSATISFIED;
  if (satisfied) {
    out_result->provider_code = iree_string_view_empty();
    out_result->display_message = iree_string_view_empty();
  }
  if (context->artifact_provider->deinitialize_device_target != NULL) {
    context->artifact_provider->deinitialize_device_target(
        context->artifact_provider, &target, context->host_allocator);
  }
  return iree_ok_status();
}

void loom_amdgpu_hal_testbench_requirement_provider_initialize(
    loom_run_hal_testbench_context_t* context,
    loom_testbench_requirement_provider_t* out_provider) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = (loom_testbench_requirement_provider_t){
      .name = IREE_SV("hal.amdgpu.descriptor_set"),
      .user_data = context,
      .query = loom_amdgpu_hal_testbench_query_descriptor_set_requirement,
  };
}
