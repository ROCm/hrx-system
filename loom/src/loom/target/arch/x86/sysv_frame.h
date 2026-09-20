// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86-64 SysV frame plans over completed target-low emission frames.

#ifndef LOOM_TARGET_ARCH_X86_SYSV_FRAME_H_
#define LOOM_TARGET_ARCH_X86_SYSV_FRAME_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation/move.h"
#include "loom/codegen/low/frame.h"
#include "loom/target/arch/x86/sysv_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

enum loom_x86_sysv_frame_slot_kind_e {
  // Argument bytes above the entry return address and allocated frame.
  LOOM_X86_SYSV_FRAME_SLOT_INCOMING_ARGUMENT = 0,
  // Argument bytes at the current stack pointer before a call.
  LOOM_X86_SYSV_FRAME_SLOT_OUTGOING_ARGUMENT = 1,
  // Storage preserving a caller-clobbered register across a call.
  LOOM_X86_SYSV_FRAME_SLOT_CALLER_SAVE = 2,
  // Storage breaking a parallel register-move cycle.
  LOOM_X86_SYSV_FRAME_SLOT_MOVE_SCRATCH = 3,
  // Storage preserving a register owned by the caller.
  LOOM_X86_SYSV_FRAME_SLOT_CALLEE_SAVE = 4,
};
typedef uint8_t loom_x86_sysv_frame_slot_kind_t;

// One addressable stack location referenced by planned physical moves.
typedef struct loom_x86_sysv_frame_slot_t {
  // Byte offset interpreted according to |kind|.
  uint64_t byte_offset;
  // Bytes read or written through the slot.
  uint32_t byte_size;
  // Required byte alignment within the owned frame region.
  uint16_t byte_alignment;
  // Semantic use of this slot.
  loom_x86_sysv_frame_slot_kind_t kind;
  // Reserved for future frame-slot flags.
  uint8_t reserved;
} loom_x86_sysv_frame_slot_t;

static_assert(sizeof(loom_x86_sysv_frame_slot_t) == 16,
              "x86 frame slots must remain compact");

// Compact contiguous range in the frame plan's move table.
typedef struct loom_x86_sysv_frame_move_range_t {
  // First move row.
  uint32_t start;
  // Number of move rows.
  uint32_t count;
} loom_x86_sysv_frame_move_range_t;

// ABI work surrounding one scheduled low.func.call.
typedef struct loom_x86_sysv_call_plan_t {
  // Schedule node containing the call operation.
  uint32_t node_index;
  // First row across the three consecutive move groups below.
  uint32_t move_start;
  // Moves preserving values live through the call.
  uint32_t save_move_count;
  // Parallel argument-placement moves after preservation.
  uint32_t argument_move_count;
  // Combined result-placement and preserved-value restore moves.
  uint32_t result_and_restore_move_count;
} loom_x86_sysv_call_plan_t;

static_assert(sizeof(loom_x86_sysv_call_plan_t) == 20,
              "x86 call plans must remain compact");

// ABI work preceding one scheduled low.return.
typedef struct loom_x86_sysv_return_plan_t {
  // Schedule node containing the return operation.
  uint32_t node_index;
  // Result-placement moves that must precede callee-register restoration.
  loom_x86_sysv_frame_move_range_t result_moves;
} loom_x86_sysv_return_plan_t;

// Complete target-owned SysV frame plan for one allocated low function.
typedef struct loom_x86_sysv_frame_plan_t {
  // Decoded ABI layout for the current function.
  loom_x86_sysv_abi_layout_t abi_layout;
  // Bytes subtracted from rsp on entry and restored on every return.
  uint64_t frame_size;
  // Base byte offset of generic low stack storage within the frame.
  uint64_t stack_storage_offset;
  // Bytes occupied by generic low stack storage.
  uint64_t stack_storage_size;
  // Addressable ABI and preservation slots.
  const loom_x86_sysv_frame_slot_t* slots;
  // Number of entries in |slots|.
  uint32_t slot_count;
  // Final sequential physical moves shared by all ranges below.
  const loom_low_move_t* moves;
  // Number of entries in |moves|.
  uint32_t move_count;
  // Register preservation performed immediately after frame allocation.
  loom_x86_sysv_frame_move_range_t callee_save_moves;
  // ABI argument placement performed after callee-register preservation.
  loom_x86_sysv_frame_move_range_t entry_moves;
  // Register restoration performed after each return's result moves.
  loom_x86_sysv_frame_move_range_t callee_restore_moves;
  // Per-call ABI plans in final scheduled liveness order.
  const loom_x86_sysv_call_plan_t* calls;
  // Number of entries in |calls|.
  uint32_t call_count;
  // Per-return ABI plans in source node order.
  const loom_x86_sysv_return_plan_t* returns;
  // Number of entries in |returns|.
  uint32_t return_count;
} loom_x86_sysv_frame_plan_t;

// Returns the whole-function allocation reservation for rsp. The reservation
// must be passed to emission-frame construction before building a SysV plan.
loom_low_allocation_reserved_range_t
loom_x86_sysv_frame_stack_pointer_reservation(void);

// Builds the target-owned ABI frame plan for a completed spill-free x86 frame.
// The immutable plan is owned by |arena|; temporary joins and move sequencing
// are owned by |scratch_arena|. Authored ABI layouts are validated here before
// their placement data crosses into native emission.
iree_status_t loom_x86_sysv_frame_plan_build(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* scratch_arena,
    loom_x86_sysv_frame_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_SYSV_FRAME_H_
