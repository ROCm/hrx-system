// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-owned diagnostics for descriptor-backed text low assembly.

#ifndef LOOM_TARGET_LOW_ASM_DIAGNOSTICS_H_
#define LOOM_TARGET_LOW_ASM_DIAGNOSTICS_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/format/text/low_asm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_low_asm_diagnostic_provider_t
    loom_target_low_asm_diagnostic_provider_t;

typedef iree_status_t (
    *loom_target_low_asm_diagnostic_try_unknown_mnemonic_fn_t)(
    const loom_target_low_asm_diagnostic_provider_t* provider,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t mnemonic,
    loom_text_low_asm_diagnostic_t* out_diagnostic);

struct loom_target_low_asm_diagnostic_provider_t {
  // Stable provider name available to callbacks.
  iree_string_view_t name;
  // Attempts to diagnose an unknown descriptor-set mnemonic. Sets
  // |out_diagnostic->error| NULL when the mnemonic does not belong to this
  // provider.
  loom_target_low_asm_diagnostic_try_unknown_mnemonic_fn_t try_unknown_mnemonic;
};

typedef struct loom_target_low_asm_diagnostic_provider_list_t {
  // Total number of values in the list.
  iree_host_size_t count;
  // Value list or NULL if no values.
  const loom_target_low_asm_diagnostic_provider_t* const* values;
} loom_target_low_asm_diagnostic_provider_list_t;

// Creates a low asm diagnostic provider list from borrowed storage.
static inline loom_target_low_asm_diagnostic_provider_list_t
loom_target_low_asm_diagnostic_provider_list_make(
    const loom_target_low_asm_diagnostic_provider_t* const* values,
    iree_host_size_t count) {
  return (loom_target_low_asm_diagnostic_provider_list_t){
      /*.count=*/count,
      /*.values=*/values,
  };
}

// Returns an empty low asm diagnostic provider list.
static inline loom_target_low_asm_diagnostic_provider_list_t
loom_target_low_asm_diagnostic_provider_list_empty(void) {
  return (loom_target_low_asm_diagnostic_provider_list_t){0};
}

// Returns true if |list| has no low asm diagnostic providers.
static inline bool loom_target_low_asm_diagnostic_provider_list_is_empty(
    loom_target_low_asm_diagnostic_provider_list_t list) {
  return list.count == 0;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_LOW_ASM_DIAGNOSTICS_H_
