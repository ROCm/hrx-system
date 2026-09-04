// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/amdgpu/artifact_provider.h"

#include "loom/target/arch/amdgpu/profile.h"
#include "loom/target/arch/amdgpu/runtime_requirements.h"
#include "loom/target/emit/native/amdgpu/hal_kernel_library.h"
#include "loom/target/emit/native/amdgpu/runtime_globals.h"

typedef struct loom_amdgpu_compile_artifact_storage_t {
  // Target-native HSACO artifact storage.
  loom_amdgpu_hal_kernel_library_t kernel_library;
} loom_amdgpu_compile_artifact_storage_t;

static loom_amdgpu_runtime_global_flags_t
loom_amdgpu_artifact_provider_runtime_globals(
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

static iree_status_t loom_amdgpu_artifact_provider_emit_artifact(
    const loom_artifact_provider_t* provider, loom_module_t* module,
    const loom_artifact_target_t* target, const loom_compile_options_t* options,
    iree_allocator_t allocator, bool* out_emitted,
    loom_artifact_t* out_artifact) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(out_emitted);
  IREE_ASSERT_ARGUMENT(out_artifact);

  *out_emitted = false;
  *out_artifact = (loom_artifact_t){0};

  loom_amdgpu_compile_artifact_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*storage), (void**)&storage));
  *storage = (loom_amdgpu_compile_artifact_storage_t){0};

  const loom_amdgpu_hal_kernel_library_options_t library_options = {
      .function_versions = options->function_versions,
      .runtime_globals = loom_amdgpu_artifact_provider_runtime_globals(
          &options->target_pipeline_options),
      .diagnostic_sink = options->diagnostic_sink,
      .source_resolver = options->source_resolver,
      .max_errors = options->max_errors,
      .report = options->report,
      .capture_target_listing = iree_all_bits_set(
          options->artifact_flags, LOOM_COMPILE_ARTIFACT_FLAG_TARGET_LISTING),
      .artifact_name = options->artifact_manifest.artifact_name,
      .artifact_manifest_identifier = options->artifact_manifest.identifier,
      .artifact_manifest =
          {
              .mode = options->artifact_manifest.mode,
          },
  };
  bool library_emitted = false;
  iree_status_t status = loom_amdgpu_emit_hal_kernel_library(
      module, &library_options, allocator, &library_emitted,
      &storage->kernel_library);
  if (iree_status_is_ok(status) && library_emitted) {
    *out_artifact = (loom_artifact_t){
        .target_key = storage->kernel_library.target_key,
        .target_bundle = &storage->kernel_library.target_bundle_storage.bundle,
        .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
        .target_artifact_data = storage->kernel_library.hsaco_data,
        .target_listing_format = storage->kernel_library.target_listing_format,
        .target_listing_data = storage->kernel_library.target_listing_data,
        .sidecars = storage->kernel_library.artifact_manifest.contents != NULL
                        ? &storage->kernel_library.artifact_manifest
                        : NULL,
        .sidecar_count =
            storage->kernel_library.artifact_manifest.contents != NULL ? 1 : 0,
        .executable_data = storage->kernel_library.hsaco_data,
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

static void loom_amdgpu_artifact_provider_deinitialize_artifact(
    const loom_artifact_provider_t* provider, loom_artifact_t* artifact,
    iree_allocator_t allocator) {
  (void)provider;
  if (artifact == NULL) {
    return;
  }
  if (artifact->storage != NULL) {
    loom_amdgpu_compile_artifact_storage_t* storage =
        (loom_amdgpu_compile_artifact_storage_t*)artifact->storage;
    loom_amdgpu_hal_kernel_library_deinitialize(&storage->kernel_library,
                                                allocator);
    iree_allocator_free(allocator, storage);
  }
  *artifact = (loom_artifact_t){0};
}

const loom_artifact_provider_t loom_amdgpu_artifact_provider = {
    .name = IREE_SVL("amdgpu-hal"),
    .public_artifact_format = IREE_SVL("amdgpu-hsaco"),
    .flags = LOOM_ARTIFACT_PROVIDER_FLAG_CANONICAL,
    .target_profile_type = &loom_amdgpu_target_profile_type,
    .artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE,
    .emit_artifact = loom_amdgpu_artifact_provider_emit_artifact,
    .deinitialize_artifact =
        loom_amdgpu_artifact_provider_deinitialize_artifact,
};
