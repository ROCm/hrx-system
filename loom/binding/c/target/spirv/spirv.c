// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_HAVE_VM_LAUNCH_CONFIG
#define LOOMC_TARGET_HAVE_VM_LAUNCH_CONFIG 0
#endif  // LOOMC_TARGET_HAVE_VM_LAUNCH_CONFIG

#include "loom/target/arch/spirv/provider.h"
#if LOOMC_TARGET_HAVE_VM_LAUNCH_CONFIG
#include "loom/target/arch/vm/provider.h"
#endif  // LOOMC_TARGET_HAVE_VM_LAUNCH_CONFIG
#include "loom/target/emit/spirv/module_emitter.h"
#if LOOMC_TARGET_HAVE_VM_LAUNCH_CONFIG
#include "loom/target/emit/vm/launch_config_compiler.h"
#endif  // LOOMC_TARGET_HAVE_VM_LAUNCH_CONFIG
#include "loomc/target/spirv/base.h"
#include "target.h"

static iree_status_t loomc_spirv_emit_module_artifact(
    const loom_target_emit_request_t* request,
    loom_target_emit_artifact_t* out_artifact) {
  *out_artifact = (loom_target_emit_artifact_t){0};

  loom_spirv_emit_low_module_options_t options = {0};
  loom_spirv_emit_low_module_options_initialize(&options);
  options.function_versions = request->function_versions;
  loom_spirv_module_binary_t binary = {0};
  iree_status_t status = loom_spirv_emit_low_module(
      request->module, request->low_descriptor_registry,
      request->diagnostic_emitter, request->scratch_arena, &options, &binary,
      request->allocator);
  if (iree_status_is_ok(status)) {
    iree_byte_span_t contents =
        iree_make_byte_span(binary.words, binary.word_count * sizeof(uint32_t));
    iree_byte_sequence_t* sequence = NULL;
    status = iree_byte_sequence_create_from_span_move(
        &contents, request->allocator, &sequence);
    if (iree_status_is_ok(status)) {
      binary.words = NULL;
      binary.word_count = 0;
      out_artifact->target_artifact_format =
          LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY;
      out_artifact->contents = sequence;
    }
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
#if LOOMC_TARGET_HAVE_VM_LAUNCH_CONFIG
    &loom_vm_target_provider,
    &loom_vm_launch_config_compiler_provider,
#endif  // LOOMC_TARGET_HAVE_VM_LAUNCH_CONFIG
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
