// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/compile/configured/emitter.h"

#ifndef LOOM_CONFIG_EMITTER_HAVE_LLVMIR
#define LOOM_CONFIG_EMITTER_HAVE_LLVMIR 0
#endif  // LOOM_CONFIG_EMITTER_HAVE_LLVMIR
#ifndef LOOM_CONFIG_EMITTER_HAVE_LLVMIR_AMDGPU
#define LOOM_CONFIG_EMITTER_HAVE_LLVMIR_AMDGPU 0
#endif  // LOOM_CONFIG_EMITTER_HAVE_LLVMIR_AMDGPU
#ifndef LOOM_CONFIG_EMITTER_HAVE_LLVMIR_X86
#define LOOM_CONFIG_EMITTER_HAVE_LLVMIR_X86 0
#endif  // LOOM_CONFIG_EMITTER_HAVE_LLVMIR_X86

#if LOOM_CONFIG_EMITTER_HAVE_LLVMIR
#include "loom/target/emit/llvmir/artifact_emitter.h"
#if LOOM_CONFIG_EMITTER_HAVE_LLVMIR_AMDGPU
#include "loom/target/emit/llvmir/amdgpu/target_env.h"
#endif  // LOOM_CONFIG_EMITTER_HAVE_LLVMIR_AMDGPU
#if LOOM_CONFIG_EMITTER_HAVE_LLVMIR_X86
#include "loom/target/emit/llvmir/x86/target_env.h"
#endif  // LOOM_CONFIG_EMITTER_HAVE_LLVMIR_X86
#endif  // LOOM_CONFIG_EMITTER_HAVE_LLVMIR

iree_status_t loom_configured_target_emitter_emit(
    const loom_target_emitter_t* emitter,
    const loom_target_emit_request_t* request,
    loom_target_emit_artifact_t* out_artifact) {
  if (emitter == NULL || emitter->emit == NULL || request == NULL ||
      out_artifact == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "configured target emitter inputs are incomplete");
  }
#if LOOM_CONFIG_EMITTER_HAVE_LLVMIR
  switch (emitter->target_artifact_format) {
    case LOOM_TARGET_ARTIFACT_FORMAT_LLVMIR_TEXT:
    case LOOM_TARGET_ARTIFACT_FORMAT_LLVMIR_BITCODE: {
      const loom_llvmir_target_profile_provider_t* target_profile_providers[] =
          {
#if LOOM_CONFIG_EMITTER_HAVE_LLVMIR_X86
              loom_llvmir_x86_target_profile_provider(),
#endif  // LOOM_CONFIG_EMITTER_HAVE_LLVMIR_X86
#if LOOM_CONFIG_EMITTER_HAVE_LLVMIR_AMDGPU
              loom_llvmir_amdgpu_target_profile_provider(),
#endif  // LOOM_CONFIG_EMITTER_HAVE_LLVMIR_AMDGPU
              NULL,
          };
      const loom_llvmir_target_profile_registry_t target_profile_registry = {
          .providers = target_profile_providers,
          .provider_count = IREE_ARRAYSIZE(target_profile_providers) - 1,
      };
      loom_llvmir_artifact_emitter_options_t options;
      loom_llvmir_artifact_emitter_options_initialize(&options);
      options.next = request->option_chain;
      options.target_profile_registry = &target_profile_registry;
      loom_target_emit_request_t configured_request = *request;
      configured_request.option_chain = &options;
      return emitter->emit(&configured_request, out_artifact);
    }
    default:
      break;
  }
#endif  // LOOM_CONFIG_EMITTER_HAVE_LLVMIR
  return emitter->emit(request, out_artifact);
}
