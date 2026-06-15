// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target-low aliases that should not be directly authored.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOW_ALIASES_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOW_ALIASES_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_low_blocked_alias_t {
  // Low descriptor key or direct mnemonic spelling that should not be authored.
  iree_string_view_t alias_name;
  // ISA mnemonic that carries the compatibility semantics.
  iree_string_view_t alias_mnemonic;
  // Semantic class that makes the alias incompatible with ordinary authoring.
  iree_string_view_t alias_semantics;
  // Stable descriptor key authors should use for ordinary FMA semantics.
  iree_string_view_t replacement_descriptor_name;
  // ISA mnemonic authors should use for ordinary FMA semantics.
  iree_string_view_t replacement_mnemonic;
  // Diagnostic decision applied to the alias.
  iree_string_view_t decision;
  // Diagnostic reason for the decision.
  iree_string_view_t reason;
} loom_amdgpu_low_blocked_alias_t;

// Returns the blocked alias matching |name|, or NULL when |name| is not an
// AMDGPU target-low alias with compatibility-only semantics.
const loom_amdgpu_low_blocked_alias_t* loom_amdgpu_low_blocked_alias_lookup(
    iree_string_view_t name);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOW_ALIASES_H_
