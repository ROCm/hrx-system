// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU per-function native text contracts.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_FUNCTION_TEXT_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_FUNCTION_TEXT_H_

#include "iree/base/api.h"
#include "loom/ir/attribute.h"

#ifdef __cplusplus
extern "C" {
#endif

// Native control action selected for a source-level low.return.
typedef enum loom_amdgpu_function_return_mode_e {
  // Terminates the current wave with s_endpgm.
  LOOM_AMDGPU_FUNCTION_RETURN_MODE_END_PROGRAM = 0,
  // Resumes the caller through the fixed direct-call return-address pair.
  LOOM_AMDGPU_FUNCTION_RETURN_MODE_CALLER = 1,
} loom_amdgpu_function_return_mode_t;

// One unresolved direct call within an encoded function text fragment.
//
// The encoder owns exact instruction-local offsets. A later program layout
// binds |callee| to its final text position and patches both literal words.
typedef struct loom_amdgpu_internal_call_fixup_t {
  // Verified module-local helper symbol targeted by the call.
  loom_symbol_ref_t callee;
  // Function-local byte offset of the PC captured by s_getpc_b64.
  uint64_t base_pc_byte_offset;
  // Function-local byte offset of the s_add_u32 literal word.
  uint64_t low_literal_byte_offset;
  // Function-local byte offset of the s_addc_u32 literal word.
  uint64_t high_literal_byte_offset;
} loom_amdgpu_internal_call_fixup_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_FUNCTION_TEXT_H_
