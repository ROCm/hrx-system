// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Retained AMDGPU natural-loop retirement plans shared by planning and
// emission. Planning owns every pointer and planned block identity below;
// emission only fills the explicitly marked Low value identities.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_CONTROL_LOOP_PLAN_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_CONTROL_LOOP_PLAN_H_

#include "loom/target/arch/amdgpu/lower/control_mask.h"

typedef struct loom_amdgpu_loop_plan_t loom_amdgpu_loop_plan_t;
typedef struct loom_amdgpu_loop_exit_plan_t loom_amdgpu_loop_exit_plan_t;

typedef struct loom_amdgpu_loop_exit_region_adapter_t {
  // Low-only block that retains one clean-arm continuation tuple.
  loom_low_lower_block_ref_t block;
} loom_amdgpu_loop_exit_region_adapter_t;

// One divergent selection whose second scheduled arm may retire lanes from an
// enclosing natural loop. The clean arm is always scheduled first, independent
// of its source successor position. Its values and surviving mask remain live
// while the exit-bearing arm executes, then both arms reconverge before the
// loop's shared retirement machinery observes the completed iteration state.
typedef struct loom_amdgpu_loop_exit_region_plan_t {
  // Shared natural-loop retirement plan owning the nested exit.
  loom_amdgpu_loop_plan_t* loop;
  // Direct loop exit at the entry of the second scheduled arm.
  loom_amdgpu_loop_exit_plan_t* exit;
  // Source selection whose arms are scheduled by this plan.
  const loom_op_t* source_op;
  // Source successor index of the first, nonretiring arm.
  uint8_t clean_successor_index;
  // Shared source continuation after both arms complete.
  uint16_t continuation_index;
  // Planned Low identity of the shared continuation.
  loom_low_lower_block_ref_t continuation;
  // Number of values carried to the shared continuation.
  uint16_t merge_arg_count;
  // Number of authored and realization values requiring lane-masked merge.
  // Supplemental control carriers follow these values and are immutable
  // across both arms.
  uint16_t source_merge_arg_count;
  // Low-only adapters retaining clean-arm continuation tuples.
  loom_amdgpu_loop_exit_region_adapter_t* clean_adapters;
  // Number of populated clean-arm adapters.
  uint16_t clean_adapter_count;
  // Shared clean-arm tuple plus its surviving EXEC mask.
  loom_low_lower_block_ref_t arm_dispatch_block;
  // Source path used when the clean arm has no active lanes.
  loom_low_lower_block_ref_t no_clean_block;
  // Original entry of the exit-bearing arm.
  loom_low_lower_block_ref_t arm_body_block;
  // Direct continuation when the exit-bearing arm has no active lanes.
  loom_low_lower_block_ref_t clean_only_block;
  // Shared merge for every ordinary exit from the second arm.
  loom_low_lower_block_ref_t merge_block;
  // Transfers captured return values to the natural-loop capture block.
  loom_low_lower_block_ref_t capture_forward_block;
  // Continues directly when no lane returned from the second arm.
  loom_low_lower_block_ref_t direct_forward_block;
  // Transfers an all-retired second arm with only clean-arm survivors.
  loom_low_lower_block_ref_t no_repeat_block;
  // Per-argument logical constants at the arm merge.
  const loom_amdgpu_mask_merge_constants_t* merge_constants;
  // Entry EXEC saved while the source selection executes.
  loom_value_id_t saved_exec;
  // Original source predicate used to merge the authored result tuple.
  loom_value_id_t source_condition;
} loom_amdgpu_loop_exit_region_plan_t;

typedef struct loom_amdgpu_loop_exit_adapter_t {
  // Low-only block that captures one source exit edge's result tuple.
  loom_low_lower_block_ref_t block;
} loom_amdgpu_loop_exit_adapter_t;

typedef enum loom_amdgpu_loop_exit_kind_e {
  // Lanes leave the loop for its final semantic continuation.
  LOOM_AMDGPU_LOOP_EXIT_TERMINAL = 0,
  // Lanes finish a nested loop but remain active in its parent loop.
  LOOM_AMDGPU_LOOP_EXIT_LOCAL,
} loom_amdgpu_loop_exit_kind_t;

typedef enum loom_amdgpu_loop_validation_e {
  LOOM_AMDGPU_LOOP_VALIDATION_UNCHECKED = 0,
  LOOM_AMDGPU_LOOP_VALIDATION_SUPPORTED,
  LOOM_AMDGPU_LOOP_VALIDATION_REJECTED,
} loom_amdgpu_loop_validation_t;

struct loom_amdgpu_loop_exit_plan_t {
  // Shared saved-mask lifetime for the natural loop.
  loom_amdgpu_loop_plan_t* loop;
  // Source conditional terminator owning the exit successors.
  const loom_op_t* source_op;
  // Bit i is set when successor i exits the loop.
  uint8_t exit_successor_mask;
  // Semantic lifetime crossed by the exit successors.
  loom_amdgpu_loop_exit_kind_t kind;
  // True after every exit successor has a retained ownership class.
  bool classified;
  // True when the exit selector is lane-varying.
  bool is_divergent;
  // True when a completed child loop supplies this parent resume site.
  bool is_synthetic;
  // True when the resume tuple contains every destination argument.
  bool resume_full_tuple;
  // Dense selector among divergent exits that may resume the loop.
  uint16_t selector;
  // Low-only block restoring the active mask when no lane takes this exit.
  loom_low_lower_block_ref_t no_exit_block;
  // Original in-loop destination reached when lanes remain active.
  loom_low_lower_block_ref_t repeat_dest;
  // Source-local repeat destination before nested reconvergence.
  loom_low_lower_block_ref_t source_repeat_dest;
  // Selector dispatch block for the remaining active lanes.
  loom_low_lower_block_ref_t dispatch_block;
  // Argument-bearing continuation reached after selector dispatch.
  loom_low_lower_block_ref_t resume_block;
  // Local result capture used before a nested arm reconverges.
  loom_low_lower_block_ref_t local_capture_block;
  // Nested branch scheduler owning this exit, or NULL for a direct exit.
  loom_amdgpu_loop_exit_region_plan_t* region;
  // First capture-block argument carrying this exit's resume tuple.
  uint16_t resume_arg_offset;
  // Number of resume arguments consumed by this exit's dispatch block.
  uint16_t resume_arg_count;
  // Exit-edge adapters carrying result values into the shared capture block.
  loom_amdgpu_loop_exit_adapter_t* adapters;
  // Number of populated exit-edge adapters.
  uint16_t adapter_count;
  // Completed child supplying a synthetic parent exit.
  loom_amdgpu_loop_plan_t* child_loop;
};

struct loom_amdgpu_loop_plan_t {
  // Dense retained natural-loop identity.
  uint16_t loop_index;
  // Number of records allocated in exit_plans.
  uint32_t exit_plan_capacity;
  // Low-only block that saves parent EXEC before the loop header.
  loom_low_lower_block_ref_t entry_block;
  // Original low loop header destination reached after entry_block.
  loom_low_lower_block_ref_t header_dest;
  // Shared block that accumulates return payloads under each exiting mask.
  loom_low_lower_block_ref_t capture_block;
  // Low-only block that restores parent EXEC after all lanes leave the loop.
  loom_low_lower_block_ref_t restore_block;
  // Shared source continuation reached after restore_block.
  loom_low_lower_block_ref_t restore_dest;
  // Shared semantic continuation reached after all loop-exit regions.
  uint16_t continuation_index;
  // Parent-local continuation reached by normally completed child lanes.
  uint16_t local_continuation_index;
  // Low value produced by entry_block, or INVALID before emission.
  loom_value_id_t saved_exec;
  // Native EXEC-mask carrier appended to the shared capture signature.
  loom_type_t exec_mask_type;
  // SGPR selector carrier appended when more than one resume site exists.
  loom_type_t selector_type;
  // True when exiting lanes must execute an exit region or retain values.
  bool capture_exits;
  // True when terminal lanes forward through an enclosing loop capture.
  bool forwards_to_parent;
  // True when the shared capture is also the loop's initial state header.
  bool entry_seeded;
  // True after semantic capture state and every resume closure are planned.
  bool prepared;
  // True after every synthetic block and edge interposition is planned.
  bool finalized;
  // Number of authored continuation arguments preceding private capture state.
  uint16_t capture_value_count;
  // Number of divergent exits assigned selector dispatch blocks.
  uint16_t divergent_exit_count;
  // First shared-capture argument reserved for nested resume tuples.
  uint16_t resume_arg_offset;
  // Total nested resume arguments appended to the shared capture tuple.
  uint16_t resume_arg_count;
  // First capture argument carrying the initial loop-header tuple.
  uint16_t entry_arg_offset;
  // Number of initial loop-header arguments retained by the state header.
  uint16_t entry_arg_count;
  // First capture argument carrying the parent-local completion tuple.
  uint16_t local_arg_offset;
  // Number of parent-local continuation arguments retained by the child.
  uint16_t local_arg_count;
  // Capture argument carrying the accumulated parent-local lane mask.
  uint16_t local_mask_offset;
  // Number of direct exit edges collected during branch preparation.
  uint32_t exit_edge_count;
  // Number of accounted exits reaching the final semantic continuation.
  uint32_t terminal_exit_edge_count;
  // Number of accounted exits completing only this nested loop.
  uint32_t local_exit_edge_count;
  // Number of populated records in exit_plans.
  uint32_t exit_plan_count;
  // Boundary validation state for shared retirement.
  loom_amdgpu_loop_validation_t validation;
  // Dispatch block resuming the first loop-header invocation.
  loom_low_lower_block_ref_t entry_dispatch_block;
  // Argument-bearing adapter resuming the first loop-header invocation.
  loom_low_lower_block_ref_t entry_resume_block;
  // Parent synthetic exit receiving this child's completed lane classes.
  loom_amdgpu_loop_exit_plan_t* parent_exit;
  // Direct exit records, allocated inline to the loop's direct exit count.
  loom_amdgpu_loop_exit_plan_t exit_plans[];
};

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_CONTROL_LOOP_PLAN_H_
