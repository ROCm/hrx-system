// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/amdgpu/artifact_provider.h"

#include "iree/hal/executable/amdgpu/executable_target.h"
#include "iree/hal/executable/amdgpu/target_id.h"
#include "loom/target/arch/amdgpu/ops/target.h"
#include "loom/target/arch/amdgpu/profile.h"
#include "loom/target/arch/amdgpu/records/target_records.h"
#include "loom/target/arch/amdgpu/runtime_requirements.h"
#include "loom/target/arch/amdgpu/target_id/target_id.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/hal_kernel_library.h"
#include "loom/target/emit/native/amdgpu/runtime_globals.h"
#include "loom/tooling/execution/hal/runtime.h"

typedef struct loom_amdgpu_hal_artifact_storage_t {
  // Target-native HSACO artifact storage.
  loom_amdgpu_hal_kernel_library_t kernel_library;
} loom_amdgpu_hal_artifact_storage_t;

typedef struct loom_amdgpu_hal_device_target_storage_t {
  // Structured AMDGPU target profile. This remains first so the owning storage
  // can be recovered from its target-neutral base pointer.
  loom_amdgpu_target_profile_t profile;

  // NUL-terminated storage for the canonical provider-facing target key.
  char target_key_storage[128];

  // Canonical provider-facing target key borrowing |target_key_storage|.
  iree_string_view_t target_key;
} loom_amdgpu_hal_device_target_storage_t;

static void loom_amdgpu_hal_device_target_storage_deinitialize(
    loom_amdgpu_hal_device_target_storage_t* storage,
    iree_allocator_t allocator) {
  if (storage == NULL) {
    return;
  }
  iree_allocator_free(allocator, storage);
}

static loom_amdgpu_target_feature_state_t
loom_amdgpu_hal_artifact_provider_map_feature_state(
    iree_hal_amdgpu_target_feature_state_t state) {
  switch (state) {
    case IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_UNSUPPORTED:
      return LOOM_AMDGPU_TARGET_FEATURE_UNSUPPORTED;
    case IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_OFF:
      return LOOM_AMDGPU_TARGET_FEATURE_OFF;
    case IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ON:
      return LOOM_AMDGPU_TARGET_FEATURE_ON;
    default:
      return LOOM_AMDGPU_TARGET_FEATURE_ANY;
  }
}

static iree_status_t loom_amdgpu_hal_artifact_provider_parse_target_profile(
    iree_string_view_t target_key, loom_amdgpu_target_profile_t* out_profile) {
  *out_profile = (loom_amdgpu_target_profile_t){0};

  iree_hal_amdgpu_target_identity_t hal_identity = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_identity_parse_artifact_key(
      target_key, &hal_identity));
  const loom_amdgpu_target_info_t* target = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_info_lookup_target(hal_identity.target, &target));

  loom_amdgpu_target_identity_t identity = {
      .target = target,
      .amdhsa_features =
          {
              .sramecc = loom_amdgpu_hal_artifact_provider_map_feature_state(
                  hal_identity.amdhsa_features.sramecc),
              .xnack = loom_amdgpu_hal_artifact_provider_map_feature_state(
                  hal_identity.amdhsa_features.xnack),
          },
  };
  return loom_amdgpu_target_profile_initialize(&identity, out_profile);
}

static iree_status_t
loom_amdgpu_hal_artifact_provider_format_profile_target_key(
    const loom_amdgpu_target_profile_t* profile,
    iree_host_size_t buffer_capacity, char* buffer,
    iree_string_view_t* out_target_key) {
  return loom_amdgpu_artifact_target_key_format(
      &profile->identity, buffer_capacity, buffer, out_target_key);
}

static iree_status_t loom_amdgpu_hal_artifact_provider_try_select_profile(
    const loom_amdgpu_target_profile_t* profile,
    const iree_hal_executable_target_t* hal_target, iree_allocator_t allocator,
    bool* out_selected, loom_run_hal_device_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(out_selected);
  IREE_ASSERT_ARGUMENT(out_target);
  *out_selected = false;

  if (!loom_amdgpu_target_properties_support_hsaco(&profile->properties)) {
    return iree_ok_status();
  }

  loom_amdgpu_hal_device_target_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*storage), (void**)&storage));
  *storage = (loom_amdgpu_hal_device_target_storage_t){
      .profile = *profile,
  };
  iree_status_t status =
      loom_amdgpu_hal_artifact_provider_format_profile_target_key(
          &storage->profile, sizeof(storage->target_key_storage),
          storage->target_key_storage, &storage->target_key);
  if (!iree_status_is_ok(status)) {
    loom_amdgpu_hal_device_target_storage_deinitialize(storage, allocator);
    return status;
  }

  *out_target = (loom_run_hal_device_target_t){
      .hal_target = hal_target,
      .target_profile = &storage->profile.base,
      .target_key = storage->target_key,
  };
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_artifact_provider_try_select_target_key(
    iree_string_view_t target_key,
    const iree_hal_executable_target_t* hal_target, iree_allocator_t allocator,
    bool* out_selected, loom_run_hal_device_target_t* out_target) {
  loom_amdgpu_target_profile_t profile = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_artifact_provider_parse_target_profile(
      target_key, &profile));
  return loom_amdgpu_hal_artifact_provider_try_select_profile(
      &profile, hal_target, allocator, out_selected, out_target);
}

static iree_status_t loom_amdgpu_hal_artifact_provider_try_select_device_target(
    const loom_run_hal_runtime_t* runtime,
    iree_hal_executable_target_kind_flags_t kind_flags,
    iree_allocator_t allocator, bool* out_selected,
    loom_run_hal_device_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(out_selected);
  IREE_ASSERT_ARGUMENT(out_target);
  *out_selected = false;

  const iree_hal_device_spec_t* device_spec =
      iree_hal_device_spec(runtime->device);
  if (device_spec == NULL) {
    return iree_ok_status();
  }

  const iree_hal_executable_target_selection_t selection = {
      .family = IREE_SV("amdgpu"),
      .kind_flags = kind_flags,
  };
  const iree_hal_executable_target_selection_result_t result =
      iree_hal_device_spec_select_executable_target(device_spec, &selection);
  if (result.outcome == IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH) {
    return iree_ok_status();
  }
  if (result.outcome ==
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU HAL device spec reports ambiguous executable targets");
  }

  return loom_amdgpu_hal_artifact_provider_try_select_target_key(
      result.target->target_key, result.target, allocator, out_selected,
      out_target);
}

static iree_status_t loom_amdgpu_hal_artifact_provider_select_device_target(
    const loom_run_hal_artifact_provider_t* provider,
    const loom_run_hal_runtime_t* runtime, iree_allocator_t allocator,
    loom_run_hal_device_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(out_target);

  *out_target = (loom_run_hal_device_target_t){0};

  iree_status_t status = iree_ok_status();
  bool selected = false;
  status = loom_amdgpu_hal_artifact_provider_try_select_device_target(
      runtime, IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT, allocator, &selected,
      out_target);
  if (iree_status_is_ok(status) && !selected) {
    status = loom_amdgpu_hal_artifact_provider_try_select_device_target(
        runtime, IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_GENERIC, allocator,
        &selected, out_target);
  }

  if (iree_status_is_ok(status) && out_target->target_profile == NULL) {
    status = iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "selected %.*s HAL device has no Loom-supported native target",
        (int)provider->target_family_name.size,
        provider->target_family_name.data);
  }
  return status;
}

static iree_status_t
loom_amdgpu_hal_artifact_provider_select_function_device_target(
    const loom_run_hal_artifact_provider_t* provider,
    const loom_run_hal_runtime_t* runtime, const loom_module_t* module,
    loom_func_like_t function, iree_allocator_t allocator,
    loom_run_hal_device_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(out_target);

  *out_target = (loom_run_hal_device_target_t){0};

  const loom_symbol_ref_t target_ref = loom_func_like_target(function);
  loom_amdgpu_target_identity_t identity = {0};
  if (!loom_amdgpu_target_identity_from_ref(module, target_ref, &identity)) {
    return loom_amdgpu_hal_artifact_provider_select_device_target(
        provider, runtime, allocator, out_target);
  }
  loom_amdgpu_target_profile_t profile = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_profile_initialize(&identity, &profile));

  const iree_hal_device_spec_t* device_spec =
      iree_hal_device_spec(runtime->device);
  if (device_spec == NULL) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "selected AMDGPU HAL device does not expose immutable device facts");
  }
  char target_key_storage[128] = {0};
  iree_string_view_t target_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_artifact_provider_format_profile_target_key(
          &profile, sizeof(target_key_storage), target_key_storage,
          &target_key));
  iree_hal_executable_target_selection_result_t result = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_device_spec_select_executable_target(
      device_spec, target_key, /*physical_device_affinity=*/0, &result));
  if (result.outcome == IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "selected AMDGPU HAL device does not support authored target '%.*s'",
        (int)target_key.size, target_key.data);
  }
  if (result.outcome ==
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "selected AMDGPU HAL device reports ambiguous matches for authored "
        "target '%.*s'",
        (int)target_key.size, target_key.data);
  }

  bool selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_artifact_provider_try_select_profile(
      &profile, result.target, allocator, &selected, out_target));
  if (!selected) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "authored AMDGPU target '%.*s' cannot be emitted as HSACO by Loom",
        (int)target_key.size, target_key.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_artifact_provider_select_target_key(
    const loom_run_hal_artifact_provider_t* provider,
    iree_string_view_t target_key, iree_allocator_t allocator,
    loom_run_hal_device_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(out_target);
  *out_target = (loom_run_hal_device_target_t){0};

  bool selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_artifact_provider_try_select_target_key(
      target_key, /*hal_target=*/NULL, allocator, &selected, out_target));
  const loom_amdgpu_target_profile_t* target_profile =
      loom_amdgpu_target_profile_cast(out_target->target_profile);
  if (!selected || target_profile == NULL ||
      target_profile->identity.target == NULL) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "AMDGPU target '%.*s' cannot be emitted as HSACO "
                            "by Loom",
                            (int)target_key.size, target_key.data);
  }

  return iree_ok_status();
}

static void loom_amdgpu_hal_artifact_provider_deinitialize_device_target(
    const loom_run_hal_artifact_provider_t* provider,
    loom_run_hal_device_target_t* target, iree_allocator_t allocator) {
  (void)provider;
  if (target == NULL) {
    return;
  }
  loom_amdgpu_hal_device_target_storage_deinitialize(
      (loom_amdgpu_hal_device_target_storage_t*)target->target_profile,
      allocator);
  *target = (loom_run_hal_device_target_t){0};
}

static loom_amdgpu_runtime_global_flags_t
loom_amdgpu_hal_artifact_provider_runtime_globals(
    const loom_target_pipeline_options_t* target_pipeline_options) {
  const loom_amdgpu_runtime_requirements_t requirements =
      loom_amdgpu_runtime_requirements_from_target_pipeline_options(
          target_pipeline_options);
  loom_amdgpu_runtime_global_flags_t runtime_globals =
      LOOM_AMDGPU_RUNTIME_GLOBAL_NONE;
  if (iree_any_bit_set(requirements,
                       LOOM_AMDGPU_RUNTIME_REQUIREMENT_FEEDBACK)) {
    runtime_globals |= LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG;
  }
  if (iree_any_bit_set(requirements,
                       LOOM_AMDGPU_RUNTIME_REQUIREMENT_ASAN_SHADOW)) {
    runtime_globals |= LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG;
  }
  if (iree_any_bit_set(requirements,
                       LOOM_AMDGPU_RUNTIME_REQUIREMENT_TSAN_SHADOW)) {
    runtime_globals |= LOOM_AMDGPU_RUNTIME_GLOBAL_TSAN_CONFIG;
  }
  return runtime_globals;
}

static iree_status_t loom_amdgpu_hal_artifact_provider_emit_artifact(
    const loom_run_hal_artifact_provider_t* provider, loom_module_t* module,
    const loom_run_hal_device_target_t* target,
    loom_diagnostic_sink_t diagnostic_sink,
    loom_source_resolver_t source_resolver, uint32_t max_errors,
    const loom_target_pipeline_options_t* target_pipeline_options,
    loom_run_candidate_artifact_flags_t artifact_flags,
    const loom_run_candidate_artifact_manifest_options_t* artifact_manifest,
    loom_target_compile_report_t* report, iree_allocator_t allocator,
    bool* out_emitted, loom_run_hal_artifact_t* out_artifact) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(out_emitted);
  IREE_ASSERT_ARGUMENT(out_artifact);

  *out_emitted = false;
  *out_artifact = (loom_run_hal_artifact_t){0};

  loom_amdgpu_hal_artifact_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*storage), (void**)&storage));
  *storage = (loom_amdgpu_hal_artifact_storage_t){0};

  const loom_amdgpu_hal_kernel_library_options_t library_options = {
      .target_selection =
          {
              .profile = target->target_profile,
          },
      .runtime_globals = loom_amdgpu_hal_artifact_provider_runtime_globals(
          target_pipeline_options),
      .diagnostic_sink = diagnostic_sink,
      .source_resolver = source_resolver,
      .max_errors = max_errors,
      .report = report,
      .capture_target_listing = iree_all_bits_set(
          artifact_flags, LOOM_RUN_CANDIDATE_ARTIFACT_FLAG_TARGET_LISTING),
      .artifact_name = artifact_manifest ? artifact_manifest->artifact_name
                                         : iree_string_view_empty(),
      .artifact_manifest_identifier = artifact_manifest
                                          ? artifact_manifest->identifier
                                          : iree_string_view_empty(),
      .artifact_manifest =
          {
              .mode = artifact_manifest
                          ? artifact_manifest->mode
                          : LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE,
          },
  };
  bool library_emitted = false;
  iree_status_t status = loom_amdgpu_emit_hal_kernel_library(
      module, &library_options, allocator, &library_emitted,
      &storage->kernel_library);
  if (iree_status_is_ok(status) && library_emitted) {
    iree_const_byte_span_t hsaco_data =
        iree_make_const_byte_span(storage->kernel_library.hsaco_data,
                                  storage->kernel_library.hsaco_data_length);
    *out_artifact = (loom_run_hal_artifact_t){
        .hal_target = target->hal_target,
        .target_key = storage->kernel_library.target_key,
        .target_bundle = loom_run_hal_device_target_bundle(target),
        .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
        .target_artifact_data = hsaco_data,
        .target_listing_format = storage->kernel_library.target_listing_format,
        .target_listing_data = iree_make_const_byte_span(
            (const uint8_t*)storage->kernel_library.target_listing_data,
            storage->kernel_library.target_listing_data_length),
        .sidecars =
            storage->kernel_library.artifact_manifest.contents.data != NULL
                ? &storage->kernel_library.artifact_manifest
                : NULL,
        .sidecar_count =
            storage->kernel_library.artifact_manifest.contents.data != NULL ? 1
                                                                            : 0,
        .executable_data = hsaco_data,
        .storage = storage,
    };
    *out_emitted = true;
  } else {
    loom_amdgpu_hal_kernel_library_deinitialize(&storage->kernel_library,
                                                allocator);
    iree_allocator_free(allocator, storage);
  }
  return status;
}

static void loom_amdgpu_hal_artifact_provider_deinitialize_artifact(
    const loom_run_hal_artifact_provider_t* provider,
    loom_run_hal_artifact_t* artifact, iree_allocator_t allocator) {
  (void)provider;
  if (artifact == NULL || artifact->storage == NULL) {
    return;
  }
  loom_amdgpu_hal_artifact_storage_t* storage =
      (loom_amdgpu_hal_artifact_storage_t*)artifact->storage;
  loom_amdgpu_hal_kernel_library_deinitialize(&storage->kernel_library,
                                              allocator);
  iree_allocator_free(allocator, storage);
  *artifact = (loom_run_hal_artifact_t){0};
}

const loom_run_hal_artifact_provider_t loom_amdgpu_hal_artifact_provider = {
    .name = IREE_SVL("amdgpu-hal"),
    .hal_driver_name = IREE_SVL("amdgpu"),
    .target_family_name = IREE_SVL("AMDGPU"),
    .select_device_target =
        loom_amdgpu_hal_artifact_provider_select_device_target,
    .select_function_device_target =
        loom_amdgpu_hal_artifact_provider_select_function_device_target,
    .select_target_key = loom_amdgpu_hal_artifact_provider_select_target_key,
    .deinitialize_device_target =
        loom_amdgpu_hal_artifact_provider_deinitialize_device_target,
    .emit_artifact = loom_amdgpu_hal_artifact_provider_emit_artifact,
    .deinitialize_artifact =
        loom_amdgpu_hal_artifact_provider_deinitialize_artifact,
};
