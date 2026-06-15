// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/amdgpu/artifact_provider.h"

#include "loom/target/arch/amdgpu/records/target_records.h"
#include "loom/target/arch/amdgpu/target_id/target_id.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/hal_kernel_library.h"
#include "loom/tooling/execution/hal/runtime.h"

typedef struct loom_amdgpu_hal_artifact_storage_t {
  // Target-native HSACO artifact storage.
  loom_amdgpu_hal_kernel_library_t kernel_library;
} loom_amdgpu_hal_artifact_storage_t;

static iree_status_t loom_amdgpu_hal_artifact_provider_format_target_id(
    const loom_amdgpu_processor_info_t* processor,
    iree_string_builder_t* builder) {
  iree_string_builder_reset(builder);
  return loom_amdgpu_amdhsa_target_id_append(processor,
                                             iree_string_view_empty(), builder);
}

static iree_status_t loom_amdgpu_hal_artifact_provider_select_device_target(
    const loom_run_hal_artifact_provider_t* provider,
    const loom_run_hal_runtime_t* runtime, iree_allocator_t allocator,
    loom_run_hal_device_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(out_target);

  *out_target = (loom_run_hal_device_target_t){0};

  iree_string_builder_t target_id_builder;
  iree_string_builder_initialize(allocator, &target_id_builder);

  iree_status_t status = iree_ok_status();
  const iree_host_size_t processor_count =
      loom_amdgpu_target_info_processor_count();
  for (iree_host_size_t i = 0;
       i < processor_count && out_target->data == NULL &&
       iree_status_is_ok(status);
       ++i) {
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_processor_at(i);
    bool emit_supported = false;
    status = loom_amdgpu_target_info_processor_supports_hsaco(processor,
                                                              &emit_supported);
    if (!iree_status_is_ok(status) || !emit_supported) {
      continue;
    }
    const loom_target_bundle_t* target_bundle =
        loom_amdgpu_target_bundle_for_descriptor_set(
            processor->descriptor_set.ordinal);
    if (target_bundle == NULL) {
      continue;
    }
    status = loom_amdgpu_hal_artifact_provider_format_target_id(
        processor, &target_id_builder);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (iree_hal_executable_cache_can_prepare_format(
            runtime->executable_cache,
            IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION,
            iree_string_builder_view(&target_id_builder))) {
      *out_target = (loom_run_hal_device_target_t){
          .data = processor,
          .target_bundle = target_bundle,
          .target_key = processor->name,
      };
    }
  }

  iree_string_builder_deinitialize(&target_id_builder);
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
  (void)allocator;

  *out_target = (loom_run_hal_device_target_t){0};

  const loom_amdgpu_processor_info_t* processor = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_info_lookup_processor(target_key, &processor));
  bool emit_supported = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_info_processor_supports_hsaco(
      processor, &emit_supported));
  if (!emit_supported) {
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

  *out_target = (loom_run_hal_device_target_t){
      .data = processor,
      .target_bundle = target_bundle,
      .target_key = processor->name,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_artifact_provider_emit_artifact(
    const loom_run_hal_artifact_provider_t* provider, loom_module_t* module,
    const loom_run_hal_device_target_t* target,
    loom_diagnostic_sink_t diagnostic_sink,
    loom_source_resolver_t source_resolver, uint32_t max_errors,
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

  const loom_amdgpu_processor_info_t* processor =
      (const loom_amdgpu_processor_info_t*)target->data;

  loom_amdgpu_hal_artifact_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*storage), (void**)&storage));
  *storage = (loom_amdgpu_hal_artifact_storage_t){0};

  const loom_amdgpu_hal_kernel_library_options_t library_options = {
      .processor = processor ? processor->name : iree_string_view_empty(),
      .target_selection =
          {
              .bundle = target->target_bundle,
              .data = target->data,
          },
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
        .executable_format = storage->kernel_library.executable_format,
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
    .emit_artifact = loom_amdgpu_hal_artifact_provider_emit_artifact,
    .deinitialize_artifact =
        loom_amdgpu_hal_artifact_provider_deinitialize_artifact,
};
