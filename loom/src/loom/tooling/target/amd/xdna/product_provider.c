// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/amd/xdna/product_provider.h"

#include "loom/product/kernel.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/low_registry.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/artifact.h"
#include "loom/target/arch/amd/xdna/aie2p/profile.h"
#include "loom/target/arch/amd/xdna/product_contract.h"
#include "loom/target/entry_selection.h"
#include "loom/tooling/compile/artifact_product.h"

static const loom_product_artifact_schema_t kLoomXdnaProductArtifactSchemas[] =
    {
        {
            .role = IREE_SVL(LOOM_PRODUCT_ARTIFACT_ROLE_KERNEL),
            .format = IREE_SVL(LOOM_XDNA_PRODUCT_FORMAT),
            .minimum_count = 1,
            .maximum_count = 1,
        },
};

const loom_product_format_t loom_xdna_product_format = {
    .operation = &loom_kernel_product_operation,
    .name = IREE_SVL(LOOM_XDNA_PRODUCT_FORMAT),
    .persistence = LOOM_PRODUCT_PERSISTENCE_SINGLE_FILE,
    .single_file =
        {
            .role = IREE_SVL(LOOM_PRODUCT_ARTIFACT_ROLE_KERNEL),
            .extension = IREE_SVL(".xdna"),
        },
    .artifact_schemas = kLoomXdnaProductArtifactSchemas,
    .artifact_schema_count = IREE_ARRAYSIZE(kLoomXdnaProductArtifactSchemas),
};

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
  if (profile == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "XDNA product emission requires an explicit AIE2P target profile");
  }
  if (options->artifact_flags != LOOM_COMPILE_ARTIFACT_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "XDNA product emission has no debug artifacts");
  }
  if (options->artifact_manifest.mode !=
      LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "XDNA product emission has no sidecar manifest");
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
          .device_profile = profile->device_profile,
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
        .target_bundle = profile->base.target_bundle,
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

static const loom_artifact_provider_t kLoomXdnaArtifactProvider = {
    .name = IREE_SVL("xdna"),
    .target_family_name = IREE_SVL("AMD XDNA AIE2P"),
    .artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE,
    .emit_artifact = loom_xdna_artifact_provider_emit_artifact,
    .deinitialize_artifact = loom_xdna_artifact_provider_deinitialize_artifact,
    .target_profile_type = &loom_aie2p_target_profile_type,
    .product_contract = &loom_xdna_kernel_product_contract,
};

static iree_status_t loom_xdna_product_provider_build(
    const loom_product_format_provider_t* provider,
    const loom_product_build_request_t* request, loom_product_t** out_product) {
  return loom_artifact_provider_build_kernel_product(
      provider, &kLoomXdnaArtifactProvider, request, out_product);
}

const loom_product_format_provider_t loom_xdna_product_provider = {
    .name = IREE_SVL("xdna"),
    .operation = &loom_kernel_product_operation,
    .format = &loom_xdna_product_format,
    .target_profile_type = &loom_aie2p_target_profile_type,
    .flags = LOOM_PRODUCT_FORMAT_PROVIDER_CANONICAL,
    .target_product_contract = &loom_xdna_kernel_product_contract,
    .build = loom_xdna_product_provider_build,
};
