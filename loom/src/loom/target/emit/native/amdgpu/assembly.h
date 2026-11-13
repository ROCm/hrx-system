// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU native assembly-fragment emission from target-low packet tables.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_ASSEMBLY_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_ASSEMBLY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/string_builder.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/error/emitter.h"
#include "loom/target/emit/native/amdgpu/function_text.h"
#include "loom/target/emit/native/amdgpu/storage_layout.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_assembly_fragment_options_t {
  // Optional target-owned packet plan applied during assembly emission. When
  // present, |instruction_layout| must come from encoding this same plan.
  const struct loom_amdgpu_packet_plan_t* packet_plan;
  // Optional function-local storage layout shared with kernel metadata.
  const loom_amdgpu_storage_layout_t* storage_layout;
  // Exact native placement decisions shared with binary encoding.
  const struct loom_amdgpu_instruction_layout_t* instruction_layout;
  // Native action selected for source-level low.return packets.
  loom_amdgpu_function_return_mode_t return_mode;
} loom_amdgpu_assembly_fragment_options_t;

// Emits an AMDGPU assembly fragment from an addressability-accepted frame and
// the target-owned insertion plans built from that frame. Values are physically
// allocated and unspilled. NULL options select no inserted packets or envelope.
// Unsupported syntax emits a source-backed diagnostic and sets |out_accepted|
// to false, including when |emitter| has no callback. Statuses transport only
// infrastructure failures. The fragment contains no kernel metadata or object
// envelope.
iree_status_t loom_amdgpu_emit_assembly_fragment(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_assembly_fragment_options_t* options,
    iree_string_builder_t* builder, iree_arena_allocator_t* scratch_arena,
    iree_diagnostic_emitter_t emitter, bool* out_accepted);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_ASSEMBLY_H_
