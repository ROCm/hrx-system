// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/amdgpu/testbench_requirements.h"

#include "loom/target/arch/amdgpu/profile.h"
#include "loom/target/arch/amdgpu/target_identity.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"
#include "loom/tooling/execution/hal/testbench_actual.h"
#include "loomc/interop.h"

// Copies the canonical identity while the selected profile is still owned.
// The identity's target points into the process-lifetime target catalog.
static iree_status_t loom_amdgpu_hal_testbench_select_device_identity(
    loom_run_hal_testbench_context_t* context,
    loom_amdgpu_target_identity_t* out_identity) {
  loomc_iree_hal_target_selection_t selection = {0};
  loomc_result_t* result = NULL;
  iree_status_t status = loom_run_hal_testbench_context_select_target(
      context, loomc_make_cstring_view("AMDGPU testbench requirement"),
      /*target_profile=*/NULL, &selection, &result);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_require_successful_result(
        result, IREE_SV("AMDGPU HAL target selection failed"));
  }
  if (iree_status_is_ok(status)) {
    const loom_amdgpu_target_profile_t* profile =
        loom_amdgpu_target_profile_cast(
            loomc_target_profile_get_interop_view(selection.target_profile));
    *out_identity = profile->identity;
  }
  loomc_result_release(result);
  loomc_target_profile_release(selection.target_profile);
  return status;
}

static bool loom_amdgpu_hal_testbench_has_device(
    const loom_run_hal_testbench_context_t* context,
    loom_testbench_requirement_provider_result_t* out_result) {
  if (iree_string_view_equal(context->driver_name, IREE_SV("amdgpu"))) {
    return true;
  }
  *out_result = (loom_testbench_requirement_provider_result_t){
      .state = LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_UNAVAILABLE,
      .provider_code = IREE_SV("hal_driver_mismatch"),
      .display_message =
          IREE_SV("AMDGPU requirement requires an AMDGPU HAL device"),
  };
  return false;
}

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
  if (!loom_amdgpu_hal_testbench_has_device(context, out_result)) {
    return iree_ok_status();
  }
  loom_amdgpu_target_identity_t identity;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_testbench_select_device_identity(context, &identity));
  if (iree_string_view_equal(identity.target->descriptor_set_key,
                             required_descriptor_set)) {
    *out_result = (loom_testbench_requirement_provider_result_t){
        .state = LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_SATISFIED,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_testbench_query_target_requirement(
    void* user_data, const loom_module_t* module, loom_named_attr_slice_t attrs,
    loom_testbench_requirement_provider_result_t* out_result) {
  loom_run_hal_testbench_context_t* context =
      (loom_run_hal_testbench_context_t*)user_data;
  *out_result = (loom_testbench_requirement_provider_result_t){
      .state = LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_UNSATISFIED,
      .provider_code = IREE_SV("target_mismatch"),
      .display_message =
          IREE_SV("AMDGPU device does not satisfy requested target"),
  };

  iree_string_view_t required_target = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_testbench_requirement_read_string_attr(
      module, attrs, IREE_SV("target"), &required_target));
  const loom_amdgpu_target_profile_t* required_profile = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_profile_select(required_target, &required_profile));
  if (!loom_amdgpu_hal_testbench_has_device(context, out_result)) {
    return iree_ok_status();
  }
  loom_amdgpu_target_identity_t identity;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_testbench_select_device_identity(context, &identity));
  if (loom_amdgpu_target_identity_satisfies_requirement(
          &identity, &required_profile->identity)) {
    *out_result = (loom_testbench_requirement_provider_result_t){
        .state = LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_SATISFIED,
    };
  }
  return iree_ok_status();
}

void loom_amdgpu_hal_descriptor_set_testbench_requirement_provider_initialize(
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

void loom_amdgpu_hal_target_testbench_requirement_provider_initialize(
    loom_run_hal_testbench_context_t* context,
    loom_testbench_requirement_provider_t* out_provider) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = (loom_testbench_requirement_provider_t){
      .name = IREE_SV("hal.amdgpu.target"),
      .user_data = context,
      .query = loom_amdgpu_hal_testbench_query_target_requirement,
  };
}
