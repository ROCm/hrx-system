// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/low_asm_diagnostics.h"

#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/low_aliases.h"

static bool loom_amdgpu_low_asm_descriptor_set_is_amdgpu(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t* out_descriptor_set_name) {
  *out_descriptor_set_name = iree_string_view_empty();
  if (descriptor_set == NULL) {
    return false;
  }
  *out_descriptor_set_name = loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset);
  return iree_string_view_starts_with(*out_descriptor_set_name,
                                      IREE_SV("amdgpu."));
}

static iree_status_t loom_amdgpu_low_asm_try_unknown_mnemonic(
    const loom_target_low_asm_diagnostic_provider_t* provider,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t mnemonic,
    loom_text_low_asm_diagnostic_t* out_diagnostic) {
  (void)provider;
  *out_diagnostic = (loom_text_low_asm_diagnostic_t){0};
  iree_string_view_t descriptor_set_name = iree_string_view_empty();
  if (!loom_amdgpu_low_asm_descriptor_set_is_amdgpu(descriptor_set,
                                                    &descriptor_set_name)) {
    return iree_ok_status();
  }

  const loom_amdgpu_low_blocked_alias_t* alias =
      loom_amdgpu_low_blocked_alias_lookup(mnemonic);
  if (alias == NULL) {
    return iree_ok_status();
  }

  out_diagnostic->error = LOOM_ERR_AMDGPU_032;
  out_diagnostic->params[0] = loom_param_string(descriptor_set_name);
  out_diagnostic->params[1] = loom_param_string(alias->alias_mnemonic);
  out_diagnostic->params[2] = loom_param_string(alias->alias_semantics);
  out_diagnostic->params[3] =
      loom_param_string(alias->replacement_descriptor_name);
  out_diagnostic->params[4] = loom_param_string(alias->replacement_mnemonic);
  out_diagnostic->params[5] = loom_param_string(alias->decision);
  out_diagnostic->params[6] = loom_param_string(alias->reason);
  out_diagnostic->param_count = 7;
  return iree_ok_status();
}

const loom_target_low_asm_diagnostic_provider_t
    loom_amdgpu_low_asm_diagnostic_provider = {
        .name = IREE_SVL("amdgpu"),
        .try_unknown_mnemonic = loom_amdgpu_low_asm_try_unknown_mnemonic,
};
