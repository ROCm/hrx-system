// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/amdgpu/device_provider.h"

#include "loom/target/arch/amdgpu/artifact_key.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/profile.h"
#include "loom/target/arch/amdgpu/provider.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/tooling/execution/hal/runtime.h"
#include "loom/tooling/target/amdgpu/artifact_provider.h"

static bool loom_amdgpu_device_provider_supports_artifact_key(
    iree_string_view_t artifact_key) {
  iree_string_view_t target_name = artifact_key;
  iree_string_view_t feature_suffix = iree_string_view_empty();
  iree_string_view_split(artifact_key, ':', &target_name, &feature_suffix);
  return loom_amdgpu_target_info_find_target(target_name) != NULL;
}

static iree_status_t loom_amdgpu_device_provider_select_compatible_candidate(
    const loom_run_hal_runtime_t* runtime,
    const loom_amdgpu_target_identity_t* authored_requirement,
    iree_hal_executable_target_kind_flags_t kind_flags,
    iree_hal_executable_target_selection_result_t* out_result) {
  *out_result = (iree_hal_executable_target_selection_result_t){
      .outcome = IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH,
      .target = NULL,
      .target_ordinal = IREE_HOST_SIZE_MAX,
  };
  const iree_hal_device_spec_t* device_spec =
      iree_hal_device_spec(runtime->device);
  if (device_spec == NULL) {
    return iree_ok_status();
  }

  const iree_hal_device_executable_spec_t* executable_spec =
      iree_hal_device_spec_executables(device_spec);
  const iree_hal_executable_target_t* selected_target = NULL;
  iree_host_size_t selected_ordinal = IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = 0; i < executable_spec->target_count; ++i) {
    const iree_hal_executable_target_t* candidate =
        &executable_spec->targets[i];
    const iree_hal_executable_target_kind_flags_t candidate_kind_flag =
        1u << candidate->kind;
    if (!iree_string_view_equal(candidate->family, IREE_SV("amdgpu")) ||
        !iree_any_bit_set(kind_flags, candidate_kind_flag)) {
      continue;
    }
    if (!loom_amdgpu_device_provider_supports_artifact_key(
            candidate->target_key)) {
      continue;
    }

    loom_amdgpu_target_identity_t candidate_identity = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_artifact_key_parse(candidate->target_key,
                                                        &candidate_identity));
    loom_amdgpu_target_profile_t candidate_profile = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_target_profile_initialize(
        &candidate_identity, &candidate_profile));
    const iree_hal_executable_target_kind_t parsed_kind =
        loom_amdgpu_target_info_is_generic(candidate_identity.target)
            ? IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC
            : IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT;
    if (parsed_kind != candidate->kind) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU device target '%.*s' kind does not match its target key",
          (int)candidate->target_key.size, candidate->target_key.data);
    }
    if (!loom_amdgpu_target_properties_support_hsaco(
            &candidate_profile.properties) ||
        (authored_requirement != NULL &&
         !loom_amdgpu_target_identity_satisfies_requirement(
             &candidate_profile.identity, authored_requirement))) {
      continue;
    }

    if (selected_target == NULL ||
        candidate->priority > selected_target->priority) {
      out_result->outcome =
          IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED;
      selected_target = candidate;
      selected_ordinal = i;
    } else if (candidate->priority == selected_target->priority) {
      out_result->outcome =
          IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS;
    }
  }

  if (out_result->outcome ==
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED) {
    out_result->target = selected_target;
    out_result->target_ordinal = selected_ordinal;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_device_provider_try_select_target(
    const loom_run_hal_runtime_t* runtime,
    const loom_amdgpu_target_identity_t* authored_requirement,
    iree_hal_executable_target_kind_flags_t kind_flags,
    iree_allocator_t allocator, bool* out_selected,
    loom_device_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(out_selected);
  IREE_ASSERT_ARGUMENT(out_target);
  *out_selected = false;

  iree_hal_executable_target_selection_result_t result = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_device_provider_select_compatible_candidate(
      runtime, authored_requirement, kind_flags, &result));
  if (result.outcome == IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH) {
    return iree_ok_status();
  }
  if (result.outcome ==
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU HAL device spec reports ambiguous executable targets");
  }

  IREE_RETURN_IF_ERROR(loom_target_provider_select_profile(
      &loom_amdgpu_target_provider, result.target->target_key, allocator,
      &out_target->profile_selection));
  out_target->executable_target = result.target;
  out_target->artifact_target = (loom_artifact_target_t){
      .target_profile = out_target->profile_selection.profile,
      .target_key = out_target->profile_selection.selector,
  };
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_device_provider_select_compatible_target(
    const loom_device_provider_t* provider,
    const loom_run_hal_runtime_t* runtime,
    const loom_amdgpu_target_identity_t* authored_requirement,
    iree_allocator_t allocator, loom_device_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(out_target);

  *out_target = (loom_device_target_t){0};

  iree_status_t status = iree_ok_status();
  bool selected = false;
  status = loom_amdgpu_device_provider_try_select_target(
      runtime, authored_requirement, IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
      allocator, &selected, out_target);
  if (iree_status_is_ok(status) && !selected) {
    status = loom_amdgpu_device_provider_try_select_target(
        runtime, authored_requirement,
        IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_GENERIC, allocator, &selected,
        out_target);
  }

  if (iree_status_is_ok(status) &&
      out_target->artifact_target.target_profile == NULL) {
    if (authored_requirement != NULL) {
      status = iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "selected %.*s HAL device has no Loom-supported native target "
          "satisfying authored target '%.*s'",
          (int)provider->artifact_provider->target_family_name.size,
          provider->artifact_provider->target_family_name.data,
          (int)authored_requirement->target->name.size,
          authored_requirement->target->name.data);
    } else {
      status = iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "selected %.*s HAL device has no Loom-supported native target",
          (int)provider->artifact_provider->target_family_name.size,
          provider->artifact_provider->target_family_name.data);
    }
  }
  return status;
}

static iree_status_t loom_amdgpu_device_provider_select_target(
    const loom_device_provider_t* provider,
    const loom_run_hal_runtime_t* runtime, iree_allocator_t allocator,
    loom_device_target_t* out_target) {
  return loom_amdgpu_device_provider_select_compatible_target(
      provider, runtime, /*authored_requirement=*/NULL, allocator, out_target);
}

static iree_status_t
loom_amdgpu_device_provider_select_compatible_target_from_facts(
    const loom_device_provider_t* provider,
    const loom_run_hal_runtime_t* runtime,
    const loom_target_facts_t* target_requirement, iree_allocator_t allocator,
    loom_device_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(out_target);

  *out_target = (loom_device_target_t){0};

  const loom_amdgpu_target_facts_t* amdgpu_requirement =
      loom_amdgpu_target_facts_cast(target_requirement);
  if (target_requirement != NULL && amdgpu_requirement == NULL) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "AMDGPU device provider cannot satisfy target family '%.*s'",
        (int)target_requirement->fact_type->name.size,
        target_requirement->fact_type->name.data);
  }
  const loom_amdgpu_target_identity_t* authored_requirement =
      amdgpu_requirement != NULL ? &amdgpu_requirement->identity : NULL;
  return loom_amdgpu_device_provider_select_compatible_target(
      provider, runtime, authored_requirement, allocator, out_target);
}

static void loom_amdgpu_device_provider_deinitialize_target(
    const loom_device_provider_t* provider, loom_device_target_t* target,
    iree_allocator_t allocator) {
  (void)provider;
  (void)allocator;
  if (target == NULL) {
    return;
  }
  loom_target_profile_selection_deinitialize(&target->profile_selection);
  *target = (loom_device_target_t){0};
}

const loom_device_provider_t loom_amdgpu_device_provider = {
    .artifact_provider = &loom_amdgpu_artifact_provider,
    .driver_name = IREE_SVL("amdgpu"),
    .select_target = loom_amdgpu_device_provider_select_target,
    .select_compatible_target =
        loom_amdgpu_device_provider_select_compatible_target_from_facts,
    .deinitialize_target = loom_amdgpu_device_provider_deinitialize_target,
};
