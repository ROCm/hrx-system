// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Author-requested ordinary read-ahead schedules.

#ifndef LOOM_TRANSFORMS_SCF_SCF_PIPELINE_PLAN_H_
#define LOOM_TRANSFORMS_SCF_SCF_PIPELINE_PLAN_H_

#include "loom/transforms/scf/scf_body.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t loom_scf_pipeline_stage_flags_t;
enum loom_scf_pipeline_stage_bits_e {
  LOOM_SCF_PIPELINE_STAGE_PRODUCER = 1u << 0,
  LOOM_SCF_PIPELINE_STAGE_CONSUMER = 1u << 1,
};

enum loom_scf_pipeline_branch_e {
  LOOM_SCF_PIPELINE_BRANCH_THEN = 0,
  LOOM_SCF_PIPELINE_BRANCH_ELSE = 1,
  LOOM_SCF_PIPELINE_BRANCH_COUNT = 2,
};

// Retained interior cut for one top-level conditional scheduling unit. The
// producer branch stages identify the ordinary-read closure within that source
// branch. Everything else is reconstructed in the consumer conditional.
typedef struct loom_scf_pipeline_guarded_partition_t {
  // Source branch bodies in then/else order, including clone correspondence.
  loom_scf_body_t branches[LOOM_SCF_PIPELINE_BRANCH_COUNT];
  // Producer/consumer assignment for operations in producer_branch.
  loom_scf_pipeline_stage_flags_t* producer_branch_stages;
  // Stage-specific conditional dependencies used by the outer cut.
  loom_scf_body_operation_t producer;
  // Stage-specific conditional dependencies used by the outer cut.
  loom_scf_body_operation_t consumer;
  // Packed references owned by producer and consumer.
  loom_scf_body_reference_t* references;
  // Branch containing the retained ordinary-read closure.
  uint8_t producer_branch;
  // Branch-local producer values yielded by the synthetic producer condition.
  loom_value_id_t* queue_values;
  // Number of values in queue_values.
  uint32_t queue_value_count;
} loom_scf_pipeline_guarded_partition_t;

typedef struct loom_scf_pipeline_plan_t {
  // Complete source operations and local payload dependencies.
  loom_scf_body_t body;
  // Stages executing each body operation, in authored order. Reconstructed
  // address arithmetic belongs to both stages with distinct iteration inputs.
  loom_scf_pipeline_stage_flags_t* stages;
  // Optional guarded partition indexed by authored body operation.
  loom_scf_pipeline_guarded_partition_t** guarded_partitions;
  // Number of selected guarded partitions.
  uint32_t guarded_partition_count;
  // Number of operations reconstructed in the consumer instead of queued.
  uint32_t rematerialized_count;
  // Source values carried from a producer iteration to its consumer.
  loom_value_id_t* queue_values;
  // Number of values in each queued iteration record.
  uint32_t queue_value_count;
  // Number of static ordinary load operations, including nested regions.
  uint32_t read_count;
} loom_scf_pipeline_plan_t;

typedef struct loom_scf_pipeline_rejection_t {
  // Source operation preventing the requested schedule, or NULL on success.
  const loom_op_t* op;
  // Structural requirement violated by that operation.
  iree_string_view_t constraint;
} loom_scf_pipeline_rejection_t;

// Builds the two-stage read-ahead schedule of a verified loop body.
// Ordinary reads and their transitive payload prerequisites form the producer.
// When the body has ordered memory effects, only proven read-ahead loads from
// concrete global-like storage move; workgroup accesses, including guarded
// reads, stay intact in the ordered consumer with workgroup barriers and the
// carried recurrence. |spaces| retains
// the owning analysis's classifications across nested reconstruction without
// borrowing its invalidated fact table. Nested scf.if/scf.for operations remain
// intact within their assigned stage unless a guarded read is split from its
// carried consumer. Convergent units and workgroup memory effects stay in the
// consumer when |has_static_bounds| proves exact lower and upper bounds.
// Along with the caller's exact positive step, this preserves participants
// across the serial/main/drain split. The plan owns the complete cut, including
// values referenced only by types or attributes. Materializers and reports
// consume this cut directly.
// Queued index add/sub results are reconstructed in the consumer when their
// complete prerequisites are already available there. This removes redundant
// queue entries and keeps their arithmetic relationship to other queued values.
//
// The admission contract excludes other nested control, global-like/unknown
// writes, global fences, source-order constraints, mixed producer/consumer
// units, producers depending on carried state, and cross-stage values whose
// types vary with the iteration. Convergence in a read unit or prerequisite
// cannot move into the producer. Rejections identify the source requirement;
// status failures identify allocation or representation limits. Plan storage
// belongs to |arena| and borrows the source IR until reconstruction completes.
iree_status_t loom_scf_pipeline_plan_build(
    loom_module_t* module, const loom_block_t* block, bool has_static_bounds,
    const loom_scf_memory_t* spaces, iree_arena_allocator_t* arena,
    loom_scf_pipeline_plan_t* out_plan,
    loom_scf_pipeline_rejection_t* out_rejection);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_SCF_SCF_PIPELINE_PLAN_H_
