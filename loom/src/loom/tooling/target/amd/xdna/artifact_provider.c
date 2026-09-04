// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/amd/xdna/artifact_provider.h"

#include "loom/target/arch/amd/xdna/aie2p/descriptors/low_registry.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/artifact.h"
#include "loom/target/arch/amd/xdna/aie2p/profile.h"
#include "loom/target/arch/amd/xdna/aie2p/records/target_records.h"
#include "loom/target/entry_selection.h"

static iree_status_t loom_xdna_artifact_provider_emit_artifact(
    const loom_artifact_provider_t* provider, loom_module_t* module,
    const loom_artifact_target_t* target, const loom_compile_options_t* options,
    iree_allocator_t allocator, bool* out_emitted,
    loom_artifact_t* out_artifact) {
  (void)provider;
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_emitted);
  IREE_ASSERT_ARGUMENT(out_artifact);
  *out_emitted = false;
  *out_artifact = (loom_artifact_t){0};

  const loom_aie2p_target_profile_t* profile =
      loom_aie2p_target_profile_cast(target->target_profile);
  if (target->target_profile != NULL && profile == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "XDNA artifact emission requires an AIE2P target profile");
  }
  if (options->artifact_flags != LOOM_COMPILE_ARTIFACT_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "XDNA artifact emission has no debug artifacts");
  }
  if (options->artifact_manifest.mode !=
      LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "XDNA artifact emission has no sidecar manifest");
  }

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
  loom_target_low_descriptor_registry_t low_descriptor_registry = {0};
  loom_aie2p_low_descriptor_registry_initialize(&low_descriptor_registry);

  iree_byte_sequence_t* contents = NULL;
  iree_status_t status = loom_aie2p_xdna_artifact_emit(
      &(loom_aie2p_xdna_artifact_request_t){
          .module = module,
          .function_versions = options->function_versions,
          .low_descriptor_registry = &low_descriptor_registry.registry,
          .device_profile = profile != NULL ? profile->device_profile : NULL,
          .compile_report = options->report,
          .diagnostic_emitter = loom_target_entry_emitter(&diagnostic_emitter),
          .scratch_arena = &arena,
          .allocator = allocator,
      },
      &contents);
  if (iree_status_is_ok(status) && diagnostic_emitter.error_count == 0 &&
      contents != NULL) {
    *out_artifact = (loom_artifact_t){
        .target_key = target->target_key,
        .target_bundle = &loom_aie2p_array_target_bundle,
        .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
        .target_artifact_data = contents,
        .executable_data = contents,
        .storage = contents,
    };
    *out_emitted = true;
    contents = NULL;
  }
  iree_byte_sequence_release(contents);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
  return status;
}

static void loom_xdna_artifact_provider_deinitialize_artifact(
    const loom_artifact_provider_t* provider, loom_artifact_t* artifact,
    iree_allocator_t allocator) {
  (void)provider;
  (void)allocator;
  if (artifact == NULL) return;
  iree_byte_sequence_release((iree_byte_sequence_t*)artifact->storage);
  *artifact = (loom_artifact_t){0};
}

const loom_artifact_provider_t loom_xdna_artifact_provider = {
    .name = IREE_SVL("xdna"),
    .public_artifact_format = IREE_SVL(LOOM_XDNA_ARTIFACT_FORMAT),
    .flags = LOOM_ARTIFACT_PROVIDER_FLAG_CANONICAL,
    .target_profile_type = &loom_aie2p_target_profile_type,
    .artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE,
    .emit_artifact = loom_xdna_artifact_provider_emit_artifact,
    .deinitialize_artifact = loom_xdna_artifact_provider_deinitialize_artifact,
};
