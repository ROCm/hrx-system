// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/spirv/artifact_provider.h"

#include "loom/target/arch/spirv/descriptors/low_registry.h"
#include "loom/target/arch/spirv/low_verify.h"
#include "loom/target/emit/spirv/module_builder.h"
#include "loom/target/emit/spirv/module_emitter.h"
#include "loom/target/entry_selection.h"
#include "loom/tooling/execution/hal/runtime.h"
#include "loom/tooling/target/spirv/vulkan_profile.h"

typedef struct loom_spirv_hal_artifact_storage_t {
  // SPIR-V binary module bytes emitted for the selected artifact entries.
  loom_spirv_module_binary_t module;
} loom_spirv_hal_artifact_storage_t;

static bool loom_spirv_hal_artifact_provider_bundle_is_compatible(
    void* user_data, const loom_target_entry_t* entry) {
  (void)user_data;
  const loom_target_bundle_t* bundle = &entry->bundle_storage.bundle;
  const loom_target_snapshot_t* snapshot = bundle->snapshot;
  const loom_target_export_plan_t* export_plan = bundle->export_plan;
  return snapshot != NULL && export_plan != NULL &&
         snapshot->codegen_format == LOOM_TARGET_CODEGEN_FORMAT_SPIRV &&
         snapshot->artifact_format ==
             LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY &&
         export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL;
}

static iree_status_t loom_spirv_hal_artifact_provider_select_device_target(
    const loom_run_hal_artifact_provider_t* provider,
    const loom_run_hal_runtime_t* runtime, iree_allocator_t allocator,
    loom_run_hal_device_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(out_target);

  *out_target = (loom_run_hal_device_target_t){0};

  loom_spirv_vulkan_hal_profile_facts_t facts = {0};
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query(
      runtime->device, runtime->executable_cache, &facts));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_initialize_target_bundle(
      &facts, &out_target->target_storage));

  iree_hal_vulkan_cooperative_matrix_property_t* matrix_rows = NULL;
  iree_host_size_t matrix_row_count = 0;
  iree_status_t status = iree_ok_status();
  if (iree_any_bit_set(
          facts.flags,
          LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_COOPERATIVE_MATRIX_KHR)) {
    status = loom_spirv_vulkan_hal_query_cooperative_matrix_properties(
        runtime->device, allocator, &matrix_rows, &matrix_row_count);
  }
  loom_spirv_vulkan_hal_target_profile_storage_t* profile_storage = NULL;
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(
          facts.flags,
          LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_COOPERATIVE_MATRIX_KHR)) {
    status = iree_allocator_malloc(allocator, sizeof(*profile_storage),
                                   (void**)&profile_storage);
  }
  if (iree_status_is_ok(status) && profile_storage != NULL) {
    status = loom_spirv_vulkan_hal_target_profile_storage_initialize(
        &facts, matrix_rows, matrix_row_count, allocator, profile_storage);
  }
  iree_allocator_free(allocator, matrix_rows);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, profile_storage);
    return status;
  }

  out_target->data = profile_storage != NULL ? &profile_storage->profile : NULL;
  out_target->target_bundle = &out_target->target_storage.bundle;
  out_target->target_key = out_target->target_storage.bundle.name;
  return iree_ok_status();
}

static void loom_spirv_hal_artifact_provider_deinitialize_device_target(
    const loom_run_hal_artifact_provider_t* provider,
    loom_run_hal_device_target_t* target, iree_allocator_t allocator) {
  (void)provider;
  if (target == NULL) {
    return;
  }
  if (target->data != NULL) {
    loom_spirv_vulkan_hal_target_profile_storage_t* storage =
        (loom_spirv_vulkan_hal_target_profile_storage_t*)target->data;
    loom_spirv_vulkan_hal_target_profile_storage_deinitialize(storage,
                                                              allocator);
    iree_allocator_free(allocator, storage);
  }
  *target = (loom_run_hal_device_target_t){0};
}

static iree_status_t loom_spirv_hal_artifact_provider_emit_entries(
    loom_module_t* module, const loom_target_entry_options_t* target_options,
    loom_target_entry_list_t entries,
    const loom_run_hal_device_target_t* target,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    const loom_target_low_descriptor_registry_t* low_registry,
    iree_arena_allocator_t* arena, iree_allocator_t allocator,
    bool* out_emitted, loom_run_hal_artifact_t* out_artifact) {
  *out_emitted = false;

  loom_low_verify_result_t low_verify_result = {0};
  loom_low_verify_scratch_t low_verify_scratch =
      loom_low_verify_scratch_for_module(module);
  const loom_target_selection_t target_selection = {
      .bundle = target->target_bundle,
      .data = target->data,
  };
  IREE_RETURN_IF_ERROR(loom_target_entry_verify_low_module(
      module, low_registry, diagnostic_emitter, target_selection,
      loom_target_entry_max_errors(target_options, /*default_max_errors=*/20),
      loom_spirv_low_verify_provider_list(), &low_verify_scratch,
      &low_verify_result));
  if (low_verify_result.error_count != 0) {
    return iree_ok_status();
  }

  loom_op_t** entry_ops = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, entries.count, sizeof(*entry_ops), (void**)&entry_ops));
  for (uint16_t i = 0; i < entries.count; ++i) {
    entry_ops[i] = entries.values[i].func.op;
  }
  loom_spirv_emit_low_module_options_t emit_options = {0};
  loom_spirv_emit_low_module_options_initialize(&emit_options);
  emit_options.entry_ops = entry_ops;
  emit_options.entry_count = entries.count;

  loom_spirv_hal_artifact_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*storage), (void**)&storage));
  *storage = (loom_spirv_hal_artifact_storage_t){0};

  iree_status_t status = loom_spirv_emit_low_module(
      module, &low_registry->registry, target_selection,
      loom_target_entry_emitter(diagnostic_emitter), arena, &emit_options,
      &storage->module, allocator);
  if (iree_status_is_ok(status) && diagnostic_emitter->error_count == 0) {
    const iree_const_byte_span_t module_bytes =
        loom_spirv_module_binary_byte_span(&storage->module);
    *out_artifact = (loom_run_hal_artifact_t){
        .executable_format = IREE_SV("vulkan-spirv-bda"),
        .target_bundle = target->target_bundle,
        .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY,
        .target_artifact_data = module_bytes,
        .executable_data = module_bytes,
        .storage = storage,
    };
    *out_emitted = true;
  } else {
    loom_spirv_module_binary_deinitialize(&storage->module, allocator);
    iree_allocator_free(allocator, storage);
  }
  return status;
}

static iree_status_t loom_spirv_hal_artifact_provider_emit_artifact(
    const loom_run_hal_artifact_provider_t* provider, loom_module_t* module,
    const loom_run_hal_device_target_t* target,
    loom_diagnostic_sink_t diagnostic_sink,
    loom_source_resolver_t source_resolver, uint32_t max_errors,
    loom_run_candidate_artifact_flags_t artifact_flags,
    loom_target_compile_report_t* report, iree_allocator_t allocator,
    bool* out_emitted, loom_run_hal_artifact_t* out_artifact) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(out_emitted);
  IREE_ASSERT_ARGUMENT(out_artifact);
  (void)artifact_flags;

  *out_emitted = false;
  *out_artifact = (loom_run_hal_artifact_t){0};

  const loom_target_entry_options_t target_options = {
      .diagnostic_sink = diagnostic_sink,
      .source_resolver = source_resolver,
      .max_errors = max_errors,
      .effective_target_bundle = target->target_bundle,
  };
  loom_target_entry_diagnostic_emitter_t diagnostic_emitter = {0};
  loom_target_entry_diagnostic_emitter_initialize(
      module, &target_options, LOOM_EMITTER_VERIFIER, &diagnostic_emitter);

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(32 * 1024, allocator, &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_target_low_descriptor_registry_t low_registry = {0};
  loom_spirv_low_descriptor_registry_initialize(&low_registry);

  iree_status_t status = iree_ok_status();
  loom_verify_result_t verify_result = {0};
  if (iree_status_is_ok(status)) {
    status = loom_target_entry_verify_module(
        module, &target_options, /*default_max_errors=*/20, &verify_result);
  }

  const loom_target_entry_predicate_t entry_predicate = {
      .fn = loom_spirv_hal_artifact_provider_bundle_is_compatible,
      .user_data = NULL,
  };
  loom_target_entry_list_t entries = {0};
  bool selected = false;
  if (iree_status_is_ok(status) && verify_result.error_count == 0) {
    status = loom_target_entry_select_all_entries(
        module, &target_options, entry_predicate, &diagnostic_emitter,
        IREE_SV("SPIR-V Vulkan HAL"), &arena, &selected, &entries);
  }
  if (iree_status_is_ok(status) && verify_result.error_count == 0 && selected &&
      diagnostic_emitter.error_count == 0) {
    if (report != NULL) {
      loom_target_compile_report_record_target_bundle(
          report, &entries.values[0].bundle_storage.bundle);
    }
    status = loom_spirv_hal_artifact_provider_emit_entries(
        module, &target_options, entries, target, &diagnostic_emitter,
        &low_registry, &arena, allocator, out_emitted, out_artifact);
  }
  if (report != NULL) {
    loom_target_compile_report_record_status(report, iree_status_code(status));
  }

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
  return status;
}

static void loom_spirv_hal_artifact_provider_deinitialize_artifact(
    const loom_run_hal_artifact_provider_t* provider,
    loom_run_hal_artifact_t* artifact, iree_allocator_t allocator) {
  (void)provider;
  if (artifact == NULL || artifact->storage == NULL) {
    return;
  }
  loom_spirv_hal_artifact_storage_t* storage =
      (loom_spirv_hal_artifact_storage_t*)artifact->storage;
  loom_spirv_module_binary_deinitialize(&storage->module, allocator);
  iree_allocator_free(allocator, storage);
  *artifact = (loom_run_hal_artifact_t){0};
}

const loom_run_hal_artifact_provider_t loom_spirv_vulkan_hal_artifact_provider =
    {
        .name = IREE_SVL("spirv-vulkan-hal"),
        .hal_driver_name = IREE_SVL("vulkan"),
        .target_family_name = IREE_SVL("SPIR-V/Vulkan"),
        .default_pipeline_options =
            {
                .control_flow_lowering =
                    LOOM_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW,
            },
        .select_device_target =
            loom_spirv_hal_artifact_provider_select_device_target,
        .deinitialize_device_target =
            loom_spirv_hal_artifact_provider_deinitialize_device_target,
        .emit_artifact = loom_spirv_hal_artifact_provider_emit_artifact,
        .deinitialize_artifact =
            loom_spirv_hal_artifact_provider_deinitialize_artifact,
};
