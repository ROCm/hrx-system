// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU native text fixups resolved after final code-object layout.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_TEXT_FIXUP_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_TEXT_FIXUP_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Kind of text literal patch applied after HSACO section layout is fixed.
typedef enum loom_amdgpu_hsaco_text_fixup_kind_e {
  LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_NONE = 0,
  // Patches the low 32 bits of a data-symbol address relative to a text PC.
  LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_DATA_SYMBOL_REL32_LO = 1,
  // Patches the high 32 bits of a data-symbol address relative to a text PC.
  LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_DATA_SYMBOL_REL32_HI = 2,
} loom_amdgpu_hsaco_text_fixup_kind_t;

// One text literal patch against an AMDGPU code-object data symbol.
typedef struct loom_amdgpu_hsaco_text_fixup_t {
  // Kind of literal value written by this fixup.
  loom_amdgpu_hsaco_text_fixup_kind_t kind;
  // Byte offset of the 32-bit literal word within the kernel text bytes.
  uint64_t literal_byte_offset;
  // Byte offset within the kernel text of the PC-relative base address.
  uint64_t base_pc_byte_offset;
  // Data symbol used as the target address.
  iree_string_view_t target_symbol;
  // Byte offset from the start of |target_symbol|.
  uint64_t target_symbol_byte_offset;
} loom_amdgpu_hsaco_text_fixup_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_TEXT_FIXUP_H_
