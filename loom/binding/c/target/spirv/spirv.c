// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/provider.h"
#include "loom/target/emit/spirv/module_emitter.h"
#include "loomc/target/spirv/base.h"
#include "target.h"

static void loomc_spirv_emit_artifact_release(void* storage,
                                              iree_allocator_t allocator) {
  iree_allocator_free(allocator, storage);
}

static iree_status_t loomc_spirv_emit_module_artifact(
    const loom_target_emit_request_t* request,
    loom_target_emit_artifact_t* out_artifact) {
  *out_artifact = (loom_target_emit_artifact_t){0};

  loom_spirv_module_binary_t binary = {0};
  iree_status_t status = loom_spirv_emit_low_module(
      request->module, request->low_descriptor_registry,
      request->target_selection, request->diagnostic_emitter,
      request->scratch_arena, /*options=*/NULL, &binary, request->allocator);
  if (iree_status_is_ok(status)) {
    out_artifact->target_artifact_format =
        LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY;
    out_artifact->contents = iree_make_const_byte_span(
        binary.words, binary.word_count * sizeof(uint32_t));
    out_artifact->storage = binary.words;
    out_artifact->release = loomc_spirv_emit_artifact_release;
    binary.words = NULL;
    binary.word_count = 0;
  }

  loom_spirv_module_binary_deinitialize(&binary, request->allocator);
  return status;
}

static const loom_target_emitter_t loomc_spirv_emitter = {
    .name = {"spirv", 5},
    .public_artifact_format = {LOOMC_ARTIFACT_FORMAT_SPIRV,
                               sizeof(LOOMC_ARTIFACT_FORMAT_SPIRV) - 1},
    .default_identifier = {"module.spv", 10},
    .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY,
    .emit = loomc_spirv_emit_module_artifact,
};

static const loom_target_emitter_t* const kLoomcSpirvEmitters[] = {
    &loomc_spirv_emitter,
};

static const loom_target_provider_t loomc_spirv_emit_target_provider = {
    .emitter_list =
        {
            .values = kLoomcSpirvEmitters,
            .count = IREE_ARRAYSIZE(kLoomcSpirvEmitters),
        },
};

static const loom_target_provider_t* const kLoomcSpirvTargetProviders[] = {
    &loom_spirv_target_provider,
    &loomc_spirv_emit_target_provider,
};

static const loom_target_provider_set_t loomc_spirv_target_provider_set = {
    .providers = kLoomcSpirvTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomcSpirvTargetProviders),
};

loomc_status_t loomc_target_environment_create_spirv(
    loomc_allocator_t allocator,
    loomc_target_environment_t** out_target_environment) {
  return loomc_target_environment_create_from_provider_set(
      &loomc_spirv_target_provider_set, allocator, out_target_environment);
}
