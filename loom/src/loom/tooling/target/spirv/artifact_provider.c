// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/spirv/artifact_provider.h"

#include "loom/target/arch/spirv/descriptors/low_registry.h"
#include "loom/target/arch/spirv/low_verify.h"
#include "loom/target/arch/spirv/profile.h"
#include "loom/target/emit/spirv/module_builder.h"
#include "loom/target/emit/spirv/module_emitter.h"
#include "loom/target/emit/spirv/product_contract.h"
#include "loom/target/entry_selection.h"
#include "loom/target/reporting/artifact_manifest_collect.h"

typedef struct loom_spirv_compile_artifact_storage_t {
  // Immutable SPIR-V binary module contents.
  iree_byte_sequence_t* module_contents;
  // Durable target bundle resolved from the emitted entry.
  loom_target_bundle_storage_t target_bundle_storage;
  // Artifact manifest sidecar emitted for module.
  loom_target_emit_sidecar_artifact_t artifact_manifest;
} loom_spirv_compile_artifact_storage_t;

static void loom_spirv_compile_artifact_storage_free(
    loom_spirv_compile_artifact_storage_t* storage,
    iree_allocator_t allocator) {
  if (storage == NULL) {
    return;
  }
  iree_byte_sequence_release(storage->module_contents);
  iree_byte_sequence_release(storage->artifact_manifest.contents);
  iree_allocator_free(allocator, storage);
}

static bool loom_spirv_artifact_provider_bundle_is_compatible(
    void* user_data, const loom_target_entry_t* entry) {
  const loom_target_bundle_t* selected_bundle =
      (const loom_target_bundle_t*)user_data;
  const loom_target_bundle_t* bundle = loom_target_entry_bundle(entry);
  const loom_target_snapshot_t* snapshot = bundle->snapshot;
  const loom_target_export_plan_t* export_plan = bundle->export_plan;
  return snapshot != NULL && export_plan != NULL &&
         snapshot->codegen_format == LOOM_TARGET_CODEGEN_FORMAT_SPIRV &&
         snapshot->artifact_format ==
             LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY &&
         export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL &&
         (selected_bundle == NULL ||
          iree_string_view_equal(bundle->config->contract_set_key,
                                 selected_bundle->config->contract_set_key));
}

static iree_status_t loom_spirv_artifact_provider_emit_entries(
    loom_module_t* module, const loom_target_entry_options_t* target_options,
    loom_target_entry_list_t entries, const loom_artifact_target_t* target,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    const loom_target_low_descriptor_registry_t* low_registry,
    const loom_compile_artifact_manifest_options_t* artifact_manifest,
    iree_arena_allocator_t* arena, iree_allocator_t allocator,
    bool* out_emitted, loom_artifact_t* out_artifact) {
  *out_emitted = false;

  loom_low_verify_result_t low_verify_result = {0};
  loom_low_verify_scratch_t low_verify_scratch =
      loom_low_verify_scratch_for_module(module);
  IREE_RETURN_IF_ERROR(loom_target_entry_verify_low_module(
      module, low_registry, target_options, diagnostic_emitter,
      /*default_max_errors=*/20, loom_spirv_low_verify_provider_list(),
      &low_verify_scratch, &low_verify_result));
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
  emit_options.function_versions = target_options->function_versions;
  emit_options.entry_ops = entry_ops;
  emit_options.entry_count = entries.count;

  loom_spirv_compile_artifact_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*storage), (void**)&storage));
  *storage = (loom_spirv_compile_artifact_storage_t){0};

  loom_spirv_module_binary_t module_binary = {0};
  iree_status_t status = loom_spirv_emit_low_module(
      module, &low_registry->registry,
      loom_target_entry_emitter(diagnostic_emitter), arena, &emit_options,
      &module_binary, allocator);
  if (iree_status_is_ok(status) && diagnostic_emitter->error_count == 0) {
    iree_byte_span_t module_contents = iree_make_byte_span(
        module_binary.words, module_binary.word_count * sizeof(uint32_t));
    status = iree_byte_sequence_create_from_span_move(
        &module_contents, allocator, &storage->module_contents);
    if (iree_status_is_ok(status)) {
      module_binary = (loom_spirv_module_binary_t){0};
    }
  }
  if (iree_status_is_ok(status) && diagnostic_emitter->error_count == 0) {
    storage->target_bundle_storage = entries.values[0].target_facts->storage;
    loom_target_bundle_storage_rebind(&storage->target_bundle_storage);
  }
  if (iree_status_is_ok(status) && diagnostic_emitter->error_count == 0 &&
      artifact_manifest != NULL &&
      artifact_manifest->mode != LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    loom_target_artifact_manifest_collect_options_t manifest_options;
    loom_target_artifact_manifest_collect_options_initialize(&manifest_options);
    manifest_options.mode = artifact_manifest->mode;
    manifest_options.artifact_name = artifact_manifest->artifact_name;
    manifest_options.artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY;
    manifest_options.flags =
        LOOM_TARGET_ARTIFACT_MANIFEST_COLLECT_FLAG_ARTIFACT_BYTE_LENGTH;
    manifest_options.artifact_byte_length =
        iree_byte_sequence_length(storage->module_contents);
    loom_target_artifact_manifest_json_t artifact_manifest_json = {0};
    status = loom_target_artifact_manifest_collect_json_from_entries(
        module, entries, &manifest_options, arena, allocator,
        &artifact_manifest_json);
    if (iree_status_is_ok(status) &&
        artifact_manifest_json.contents.data != NULL) {
      iree_byte_span_t manifest_contents =
          iree_make_byte_span((uint8_t*)artifact_manifest_json.contents.data,
                              artifact_manifest_json.contents.data_length);
      iree_byte_sequence_t* manifest_sequence = NULL;
      status = iree_byte_sequence_create_from_span_move(
          &manifest_contents, allocator, &manifest_sequence);
      if (iree_status_is_ok(status)) {
        artifact_manifest_json.contents = iree_const_byte_span_empty();
      }
      storage->artifact_manifest = (loom_target_emit_sidecar_artifact_t){
          .kind = LOOM_TARGET_EMIT_SIDECAR_ARTIFACT_KIND_ARTIFACT_MANIFEST,
          .identifier = artifact_manifest->identifier,
          .contents = manifest_sequence,
      };
    }
    loom_target_artifact_manifest_json_release(&artifact_manifest_json,
                                               allocator);
  }
  if (iree_status_is_ok(status) && diagnostic_emitter->error_count == 0) {
    *out_artifact = (loom_artifact_t){
        .target_key = target->target_key,
        .target_bundle = &storage->target_bundle_storage.bundle,
        .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY,
        .target_artifact_data = storage->module_contents,
        .sidecars = storage->artifact_manifest.contents != NULL
                        ? &storage->artifact_manifest
                        : NULL,
        .sidecar_count = storage->artifact_manifest.contents != NULL ? 1 : 0,
        .executable_data = storage->module_contents,
        .storage = storage,
    };
    *out_emitted = true;
  } else {
    loom_spirv_compile_artifact_storage_free(storage, allocator);
  }
  loom_spirv_module_binary_deinitialize(&module_binary, allocator);
  return status;
}

static iree_status_t loom_spirv_artifact_provider_emit_artifact(
    const loom_artifact_provider_t* provider, loom_module_t* module,
    const loom_artifact_target_t* target, const loom_compile_options_t* options,
    iree_allocator_t allocator, bool* out_emitted,
    loom_artifact_t* out_artifact) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(out_emitted);
  IREE_ASSERT_ARGUMENT(out_artifact);

  *out_emitted = false;
  *out_artifact = (loom_artifact_t){0};

  const loom_target_entry_options_t target_options = {
      .function_versions = options->function_versions,
      .diagnostic_sink = options->diagnostic_sink,
      .source_resolver = options->source_resolver,
      .max_errors = options->max_errors,
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
      .fn = loom_spirv_artifact_provider_bundle_is_compatible,
      .user_data = (void*)loom_artifact_target_bundle(target),
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
    if (options->report != NULL) {
      loom_target_compile_report_record_target_bundle(
          options->report, loom_target_entry_bundle(&entries.values[0]));
    }
    status = loom_spirv_artifact_provider_emit_entries(
        module, &target_options, entries, target, &diagnostic_emitter,
        &low_registry, &options->artifact_manifest, &arena, allocator,
        out_emitted, out_artifact);
  }
  if (options->report != NULL) {
    loom_target_compile_report_record_status(options->report,
                                             iree_status_code(status));
  }

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
  return status;
}

static void loom_spirv_artifact_provider_deinitialize_artifact(
    const loom_artifact_provider_t* provider, loom_artifact_t* artifact,
    iree_allocator_t allocator) {
  (void)provider;
  if (artifact == NULL) {
    return;
  }
  if (artifact->storage != NULL) {
    loom_spirv_compile_artifact_storage_t* storage =
        (loom_spirv_compile_artifact_storage_t*)artifact->storage;
    loom_spirv_compile_artifact_storage_free(storage, allocator);
  }
  *artifact = (loom_artifact_t){0};
}

const loom_artifact_provider_t loom_spirv_vulkan_artifact_provider = {
    .name = IREE_SVL("spirv-vulkan-hal"),
    .target_family_name = IREE_SVL("SPIR-V/Vulkan"),
    .artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE,
    .default_pipeline_options =
        {
            .control_flow_lowering =
                LOOM_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW,
        },
    .emit_artifact = loom_spirv_artifact_provider_emit_artifact,
    .deinitialize_artifact = loom_spirv_artifact_provider_deinitialize_artifact,
    .target_profile_type = &loom_spirv_target_profile_type,
    .product_contract = &loom_spirv_binary_kernel_product_contract,
};
