// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Function-local data-symbol fixups awaiting final artifact binding.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_DATA_SYMBOL_FIXUP_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_DATA_SYMBOL_FIXUP_H_

#include "iree/base/api.h"
#include "loom/ir/types.h"
#include "loom/target/emit/native/amdgpu/text_fixup.h"

#ifdef __cplusplus
extern "C" {
#endif

// One function-local text literal patch retaining its module symbol identity.
typedef struct loom_amdgpu_data_symbol_fixup_t {
  // Kind of literal value written by this fixup.
  loom_amdgpu_hsaco_text_fixup_kind_t kind;
  // Module-local data symbol used as the target address.
  loom_symbol_id_t target_symbol_id;
  // Source node in the producing function's retained schedule.
  uint32_t node_index;
  // Byte offset of the 32-bit literal word within the function text bytes.
  uint64_t literal_byte_offset;
  // Byte offset within the function text of the PC-relative base address.
  uint64_t base_pc_byte_offset;
  // Byte offset from the start of the target data symbol.
  uint64_t target_symbol_byte_offset;
} loom_amdgpu_data_symbol_fixup_t;
static_assert(sizeof(loom_amdgpu_data_symbol_fixup_t) == 32,
              "AMDGPU module data-symbol fixups stay compact");

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_DATA_SYMBOL_FIXUP_H_
