// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU native machine-code emission from target-low packet tables.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_ENCODING_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_ENCODING_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/target/emit/native/amdgpu/branch_layout.h"
#include "loom/target/emit/native/amdgpu/storage_layout.h"
#include "loom/target/emit/native/amdgpu/text_fixup.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_native_insertion_kind_e {
  // No native insertion kind was recorded.
  LOOM_AMDGPU_NATIVE_INSERTION_NONE = 0,
  // An S_SET_VGPR_MSB address-state transition.
  LOOM_AMDGPU_NATIVE_INSERTION_ADDRESS_STATE = 1,
  // A target wait-counter packet.
  LOOM_AMDGPU_NATIVE_INSERTION_WAIT = 2,
  // An S_NOP delay packet.
  LOOM_AMDGPU_NATIVE_INSERTION_S_NOP = 3,
  // An S_DELAY_ALU delay packet.
  LOOM_AMDGPU_NATIVE_INSERTION_S_DELAY_ALU = 4,
  // A V_NOP vector issue packet.
  LOOM_AMDGPU_NATIVE_INSERTION_V_NOP = 5,
  // An S_BRANCH skipping a co-located branch-island group.
  LOOM_AMDGPU_NATIVE_INSERTION_BRANCH_ISLAND_SKIP = 6,
  // An S_BRANCH implementing one branch-island hop.
  LOOM_AMDGPU_NATIVE_INSERTION_BRANCH_ISLAND_HOP = 7,
} loom_amdgpu_native_insertion_kind_t;

// One target-owned instruction inserted while encoding a scheduled packet.
typedef struct loom_amdgpu_native_insertion_t {
  // Stable target-owned insertion kind.
  loom_amdgpu_native_insertion_kind_t kind;
  // Region block ordinal containing the insertion point.
  uint32_t block_index;
  // Schedule node whose native expansion contains the insertion.
  uint32_t node_index;
  // Scheduled ordinal of |node_index| within |block_index|.
  uint32_t scheduled_ordinal;
  // Kind-specific immediate encoded by the inserted instruction.
  uint16_t immediate;
  // Stable descriptor reference, or LOOM_AMDGPU_DESCRIPTOR_REF_NONE when the
  // inserted packet has no descriptor row.
  uint16_t descriptor_ref;
} loom_amdgpu_native_insertion_t;

typedef enum loom_amdgpu_encode_instruction_stream_flag_bits_e {
  // No optional encoding products are requested.
  LOOM_AMDGPU_ENCODE_INSTRUCTION_STREAM_FLAG_NONE = 0u,
  // Capture target-owned native insertion rows alongside encoded text.
  LOOM_AMDGPU_ENCODE_INSTRUCTION_STREAM_FLAG_CAPTURE_NATIVE_INSERTIONS = 1u
                                                                         << 0,
} loom_amdgpu_encode_instruction_stream_flag_bits_t;
typedef uint32_t loom_amdgpu_encode_instruction_stream_flags_t;

typedef struct loom_amdgpu_encode_instruction_stream_options_t {
  // Optional target-owned packet plan applied during native encoding.
  const struct loom_amdgpu_packet_plan_t* packet_plan;
  // Optional function-local storage layout shared with kernel metadata.
  const loom_amdgpu_storage_layout_t* storage_layout;
  // Optional encoding products requested by the caller.
  loom_amdgpu_encode_instruction_stream_flags_t flags;
} loom_amdgpu_encode_instruction_stream_options_t;

typedef struct loom_amdgpu_encoded_instruction_stream_t {
  // Encoded executable text bytes.
  iree_const_byte_span_t text;
  // Number of native instructions encoded in |text|.
  uint64_t instruction_count;
  // Exact branch-island layout applied to |text|, or empty when every branch
  // was directly encodable.
  loom_amdgpu_branch_layout_t branch_layout;
  // Text literal fixups resolved after final HSACO section layout.
  const loom_amdgpu_hsaco_text_fixup_t* text_fixups;
  // Number of entries in |text_fixups|.
  iree_host_size_t text_fixup_count;
  // Target-owned instructions inserted during native encoding when capture was
  // requested.
  const loom_amdgpu_native_insertion_t* native_insertions;
  // Number of entries in |native_insertions|.
  iree_host_size_t native_insertion_count;
} loom_amdgpu_encoded_instruction_stream_t;

// Encodes one AMDGPU target-low function from an addressability-accepted
// emission frame into an arena-owned instruction byte stream. The returned
// bytes are only the executable text payload; kernel descriptors, metadata,
// ELF sections, and relocations are emitted by later native code-object layers.
// Values must be physically allocated and unspilled. This entry emits no
// packet-plan insertions; frames that require them use the options form below.
iree_status_t loom_amdgpu_encode_instruction_stream(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_const_byte_span_t* out_text, iree_arena_allocator_t* arena);

// Encodes one instruction stream with target-owned insertion plans built from
// the same emission frame.
iree_status_t loom_amdgpu_encode_instruction_stream_with_options(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_encode_instruction_stream_options_t* options,
    iree_const_byte_span_t* out_text, iree_arena_allocator_t* arena);

// Encodes one instruction stream and returns text plus relocation fixups.
iree_status_t loom_amdgpu_encode_instruction_stream_result(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    loom_amdgpu_encoded_instruction_stream_t* out_stream,
    iree_arena_allocator_t* arena);

// Encodes one instruction stream with target-owned insertion plans built from
// the same emission frame and returns text plus relocation fixups.
iree_status_t loom_amdgpu_encode_instruction_stream_result_with_options(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_encode_instruction_stream_options_t* options,
    loom_amdgpu_encoded_instruction_stream_t* out_stream,
    iree_arena_allocator_t* arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_ENCODING_H_
