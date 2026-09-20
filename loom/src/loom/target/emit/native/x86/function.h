// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Native x86-64 SysV function encoding from completed Low emission frames.

#ifndef LOOM_TARGET_EMIT_NATIVE_X86_FUNCTION_H_
#define LOOM_TARGET_EMIT_NATIVE_X86_FUNCTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/frame.h"
#include "loom/ir/attribute.h"

#ifdef __cplusplus
extern "C" {
#endif

// One direct-call relocation retained from an encoded function.
typedef struct loom_x86_function_call_fixup_t {
  // Byte offset of the signed rel32 field within |text|.
  uint32_t text_offset;
  // Module-local function symbol referenced by the call.
  loom_symbol_ref_t target;
} loom_x86_function_call_fixup_t;

static_assert(sizeof(loom_x86_function_call_fixup_t) == 8,
              "x86 call fixups must remain compact");

// Native bytes and unresolved direct calls for one x86-64 function.
typedef struct loom_x86_function_encoding_t {
  // Complete SysV function body, including prologue and epilogues.
  iree_const_byte_span_t text;
  // Direct-call fixups in instruction order.
  const loom_x86_function_call_fixup_t* call_fixups;
  // Number of records in |call_fixups|.
  iree_host_size_t call_fixup_count;
} loom_x86_function_encoding_t;

// Encodes one completed spill-free scalar x86 Low frame as a callable SysV
// function. Final bytes and call fixups are owned by |arena|. Temporary frame
// planning, indexing, and sizing storage is released before this call returns.
iree_status_t loom_x86_encode_sysv_function(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_x86_function_encoding_t* out_encoding);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_X86_FUNCTION_H_
