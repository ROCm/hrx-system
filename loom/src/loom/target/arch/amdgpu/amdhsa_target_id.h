// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDHSA code-object target-ID parsing, formatting, and ELF projection.

#ifndef LOOM_TARGET_ARCH_AMDGPU_AMDHSA_TARGET_ID_H_
#define LOOM_TARGET_ARCH_AMDGPU_AMDHSA_TARGET_ID_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/target/arch/amdgpu/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

// Parsed AMDHSA code-object target ID.
typedef struct loom_amdgpu_amdhsa_target_id_t {
  // Processor row selected by the target-ID processor component.
  const loom_amdgpu_processor_info_t* processor;

  // Target-ID feature suffix after ':', or empty when no suffix is present.
  iree_string_view_t feature_suffix;

  // Structured feature states parsed from the target-ID suffix.
  loom_amdgpu_amdhsa_feature_states_t features;
} loom_amdgpu_amdhsa_target_id_t;

// Canonical AMDHSA target-ID prefix.
extern const iree_string_view_t loom_amdgpu_amdhsa_target_id_prefix;

// Parses a colon-separated AMDHSA feature suffix for |processor|.
//
// Empty suffixes select unconstrained states for supported features and
// explicit UNSUPPORTED states for the remaining known features.
iree_status_t loom_amdgpu_amdhsa_feature_suffix_parse(
    const loom_amdgpu_processor_info_t* processor,
    iree_string_view_t feature_suffix,
    loom_amdgpu_amdhsa_feature_states_t* out_features);

// Appends explicit AMDHSA feature states in canonical order.
iree_status_t loom_amdgpu_amdhsa_feature_suffix_append(
    const loom_amdgpu_amdhsa_feature_states_t* features,
    iree_string_builder_t* builder);

// Parses an AMDHSA target ID such as
// `amdgcn-amd-amdhsa--gfx11-generic`.
iree_status_t loom_amdgpu_amdhsa_target_id_parse(
    iree_string_view_t value, loom_amdgpu_amdhsa_target_id_t* out_target_id);

// Formats the AMDHSA code-object target ID projected from |identity| into
// |arena|.
//
// The canonical compiler target is projected to its backend processor. Target
// overlay semantics remain in the enclosing artifact identity.
iree_status_t loom_amdgpu_amdhsa_target_id_format(
    const loom_amdgpu_target_identity_t* identity,
    iree_arena_allocator_t* arena, iree_string_view_t* out_target_id);

// Resolves the AMDGPU ELF e_flags implied by |target_id|.
iree_status_t loom_amdgpu_amdhsa_target_id_elf_flags(
    const loom_amdgpu_amdhsa_target_id_t* target_id, uint32_t* out_elf_flags);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_AMDHSA_TARGET_ID_H_
