// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// LLVMIR target artifact emitters.
//
// This package contributes prepared-target-low emitters for LLVMIR text and
// bitcode artifacts. It does not register the LLVMIR dialect or lowering
// descriptors by itself; compose it with loom_llvmir_target_provider when a
// tool wants both LLVMIR target IR support and artifact emission.

#ifndef LOOM_TARGET_EMIT_LLVMIR_ARTIFACT_EMITTER_H_
#define LOOM_TARGET_EMIT_LLVMIR_ARTIFACT_EMITTER_H_

#include "loom/target/emit/llvmir/target_presets.h"
#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_llvmir_artifact_emitter_option_type_e {
  LOOM_LLVMIR_ARTIFACT_EMITTER_OPTION_TYPE_OPTIONS = 0x4C4C564D,
} loom_llvmir_artifact_emitter_option_type_t;

typedef struct loom_llvmir_artifact_emitter_options_t {
  // Option descriptor type.
  loom_llvmir_artifact_emitter_option_type_t type;
  // Size of this structure in bytes.
  iree_host_size_t structure_size;
  // Next emitter option descriptor.
  const void* next;
  // Optional registry of linked target profiles for kernel projection.
  const loom_llvmir_target_profile_registry_t* target_profile_registry;
} loom_llvmir_artifact_emitter_options_t;

void loom_llvmir_artifact_emitter_options_initialize(
    loom_llvmir_artifact_emitter_options_t* out_options);

// Provider contribution containing LLVMIR text and bitcode artifact emitters.
extern const loom_target_provider_t loom_llvmir_artifact_emitter_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_LLVMIR_ARTIFACT_EMITTER_H_
