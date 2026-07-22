// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/amdgpu/artifact_provider.h"

#include <inttypes.h>

#include "iree/hal/executable/amdgpu/target_id.h"
#include "loom/target/arch/amdgpu/records/target_records.h"
#include "loom/target/arch/amdgpu/runtime_requirements.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/hal_kernel_library.h"
#include "loom/target/emit/native/amdgpu/runtime_globals.h"
#include "loom/tooling/execution/hal/runtime.h"

typedef struct loom_amdgpu_hal_artifact_storage_t {
  // Target-native HSACO artifact storage.
  loom_amdgpu_hal_kernel_library_t kernel_library;
} loom_amdgpu_hal_artifact_storage_t;

static iree_status_t loom_amdgpu_hal_artifact_provider_parse_target_profile(
    iree_string_view_t target_key, loom_amdgpu_target_profile_t* out_profile) {
  *out_profile = (loom_amdgpu_target_profile_t){0};

  iree_hal_amdgpu_target_id_t target_id = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_id_parse(
      target_key,
      IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_ARCH_ONLY |
          IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_FEATURE_SUFFIXES,
      &target_id));
  const loom_amdgpu_processor_info_t* processor = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_info_lookup_processor(
      target_id.processor, &processor));

  loom_amdgpu_gfx1250_revision_t gfx1250_revision =
      LOOM_AMDGPU_GFX1250_REVISION_UNSPECIFIED;
  if (iree_string_view_equal(processor->name, IREE_SV("gfx1250"))) {
    switch (target_id.gfx1250_b0_specific) {
      case IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_OFF:
        gfx1250_revision = LOOM_AMDGPU_GFX1250_REVISION_A0;
        break;
      case IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY:
      case IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ON:
        gfx1250_revision = LOOM_AMDGPU_GFX1250_REVISION_B0;
        break;
      case IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_UNSUPPORTED:
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "gfx1250 target '%.*s' reports unsupported silicon revision "
            "selection",
            (int)target_key.size, target_key.data);
    }
  }
  *out_profile = (loom_amdgpu_target_profile_t){
      .processor = processor,
      .gfx1250_revision = gfx1250_revision,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_artifact_provider_try_select_profile(
    const loom_amdgpu_target_profile_t* profile,
    const iree_hal_executable_target_t* hal_target, bool* out_selected,
    iree_string_view_t target_key, iree_allocator_t allocator,
    loom_run_hal_device_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(out_selected);
  IREE_ASSERT_ARGUMENT(out_target);
  *out_selected = false;
  if (profile == NULL || profile->processor == NULL) {
    return iree_ok_status();
  }
  const loom_amdgpu_processor_info_t* processor = profile->processor;

  if (!loom_amdgpu_processor_supports_hsaco(processor)) {
    return iree_ok_status();
  }

  const loom_target_bundle_t* target_bundle =
      loom_amdgpu_target_bundle_for_descriptor_set(
          processor->descriptor_set.ordinal);
  if (target_bundle == NULL) {
    return iree_ok_status();
  }

  loom_amdgpu_target_profile_t* owned_profile = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, sizeof(*owned_profile),
                                             (void**)&owned_profile));
  *owned_profile = *profile;
  *out_target = (loom_run_hal_device_target_t){
      .hal_target = hal_target,
      .data = owned_profile,
      .target_bundle = target_bundle,
      .target_key = target_key,
  };
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_artifact_provider_try_select_device_target(
    const loom_run_hal_runtime_t* runtime,
    iree_hal_executable_target_kind_flags_t kind_flags, bool* out_selected,
    iree_allocator_t allocator, loom_run_hal_device_target_t* out_target) {
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

  loom_amdgpu_target_profile_t profile = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_artifact_provider_parse_target_profile(
      result.target->target_key, &profile));
  return loom_amdgpu_hal_artifact_provider_try_select_profile(
      &profile, result.target, out_selected, result.target->target_key,
      allocator, out_target);
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
      runtime, IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT, &selected, allocator,
      out_target);
  if (iree_status_is_ok(status) && !selected) {
    status = loom_amdgpu_hal_artifact_provider_try_select_device_target(
        runtime, IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_GENERIC, &selected,
        allocator, out_target);
  }

  if (iree_status_is_ok(status) && out_target->data == NULL) {
    status = iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "selected %.*s HAL device has no Loom-supported native target",
        (int)provider->target_family_name.size,
        provider->target_family_name.data);
  }
  return status;
}

static iree_status_t loom_amdgpu_hal_artifact_provider_select_target_key(
    const loom_run_hal_artifact_provider_t* provider,
    iree_string_view_t target_key, iree_allocator_t allocator,
    loom_run_hal_device_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(out_target);

  *out_target = (loom_run_hal_device_target_t){0};

  loom_amdgpu_target_profile_t profile = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_artifact_provider_parse_target_profile(
      target_key, &profile));
  const loom_amdgpu_processor_info_t* processor = profile.processor;
  if (!loom_amdgpu_processor_supports_hsaco(processor)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "AMDGPU processor '%.*s' cannot be emitted as "
                            "HSACO by Loom",
                            (int)target_key.size, target_key.data);
  }

  const loom_target_bundle_t* target_bundle =
      loom_amdgpu_target_bundle_for_descriptor_set(
          processor->descriptor_set.ordinal);
  if (target_bundle == NULL) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "AMDGPU processor '%.*s' has no Loom target "
                            "bundle for descriptor set '%.*s'",
                            (int)target_key.size, target_key.data,
                            (int)processor->descriptor_set.key.size,
                            processor->descriptor_set.key.data);
  }

  bool selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_artifact_provider_try_select_profile(
      &profile, /*hal_target=*/NULL, &selected, target_key, allocator,
      out_target));
  if (!selected) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "AMDGPU target '%.*s' could not be selected",
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
  iree_allocator_free(allocator, (void*)target->data);
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

  const loom_amdgpu_target_profile_t* target_profile =
      (const loom_amdgpu_target_profile_t*)target->data;

  loom_amdgpu_hal_artifact_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*storage), (void**)&storage));
  *storage = (loom_amdgpu_hal_artifact_storage_t){0};

  const loom_amdgpu_hal_kernel_library_options_t library_options = {
      .target_profile = target_profile,
      .target_selection =
          {
              .bundle = target->target_bundle,
              .data = target->data,
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
        .target_bundle = target->target_bundle,
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
    .select_target_key = loom_amdgpu_hal_artifact_provider_select_target_key,
    .deinitialize_device_target =
        loom_amdgpu_hal_artifact_provider_deinitialize_device_target,
    .emit_artifact = loom_amdgpu_hal_artifact_provider_emit_artifact,
    .deinitialize_artifact =
        loom_amdgpu_hal_artifact_provider_deinitialize_artifact,
};
