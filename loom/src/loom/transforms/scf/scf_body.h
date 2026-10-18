// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Retained dependencies of structured loop scheduling units.

#ifndef LOOM_TRANSFORMS_SCF_SCF_BODY_H_
#define LOOM_TRANSFORMS_SCF_SCF_BODY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/transforms/scf/scf_memory.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t loom_scf_body_effect_flags_t;
enum loom_scf_body_effect_flag_bits_e {
  LOOM_SCF_BODY_EFFECT_READ = 1u << 0,
  LOOM_SCF_BODY_EFFECT_WRITE = 1u << 1,
  LOOM_SCF_BODY_EFFECT_ORDERED = 1u << 2,
  // A read effect that is not an ordinary memory load.
  LOOM_SCF_BODY_EFFECT_NON_LOAD_READ = 1u << 3,
  // A compiler source-order constraint, independent of runtime effects.
  LOOM_SCF_BODY_EFFECT_SOURCE_ORDER = 1u << 4,
  // Execution must preserve its original dynamic participant set. Independent
  // memory and ordering effects remain represented by their own bits.
  LOOM_SCF_BODY_EFFECT_CONVERGENT = 1u << 5,
};

// Classifies one non-structured operation's declared effects. Region owners
// aggregate nested effects through their owning traversal.
loom_scf_body_effect_flags_t loom_scf_body_operation_effects(
    const loom_module_t* module, const loom_op_t* op);

typedef struct loom_scf_body_reference_t {
  // Body-local value required before materializing the referencing operation.
  loom_value_id_t value_id;
  // A detached definition only requires a mapping to exist. A body-local
  // definition requires a distinct value for the materialized iteration.
  bool allow_identity_mapping;
} loom_scf_body_reference_t;

typedef enum loom_scf_body_mode_e {
  // Capture scheduling units, complete payload dependencies and source effects.
  LOOM_SCF_BODY_MODE_SCHEDULE,
  // Capture only memory accesses for cloning without scheduling admission.
  LOOM_SCF_BODY_MODE_PROJECT_MEMORY,
} loom_scf_body_mode_t;

// Space-qualified effects used by the read-ahead cut. Unknown or other spaces
// remain distinct from the proven read-ahead/workgroup separation.
enum loom_scf_body_memory_effect_bits_e {
  LOOM_SCF_BODY_MEMORY_READ_AHEAD_LOAD = 1u << 0,
  LOOM_SCF_BODY_MEMORY_WORKGROUP = 1u << 1,
  LOOM_SCF_BODY_MEMORY_OTHER_LOAD = 1u << 2,
  LOOM_SCF_BODY_MEMORY_UNSUPPORTED = 1u << 3,
};

typedef struct loom_scf_body_operation_t {
  // Borrowed source operation, including its complete verified payload.
  const loom_op_t* op;
  // First dependency in the body's packed reference array.
  iree_host_size_t reference_begin;
  // Number of outer-body dependencies, including nested region captures.
  iree_host_size_t reference_count;
  // Combined effects governing whether different iterations may commute.
  loom_scf_body_effect_flags_t effects;
  // Number of ordinary load operations, including loads inside nested regions.
  uint32_t load_count;
} loom_scf_body_operation_t;

// Optional memory projection of one scheduling unit. Ordinary unroll planning
// allocates neither this array nor the operation correspondence.
typedef struct loom_scf_body_access_unit_t {
  // First selected access in the body's packed correspondence.
  uint32_t begin;
  // Number of accesses in this complete nested scheduling unit.
  uint32_t count;
  // Combined space-qualified memory effects.
  uint8_t effects;
} loom_scf_body_access_unit_t;

typedef struct loom_scf_body_t {
  // Optional memory classifications and clone correspondence.
  struct {
    // Selected memory operations in clone visitation order.
    loom_ir_remap_op_projection_t* operations;
    // Access spans and effects indexed by scheduling unit.
    loom_scf_body_access_unit_t* units;
    // Number of selected accesses.
    uint32_t count;
  } accesses;
  // Source operations in authored order, excluding the terminator.
  loom_scf_body_operation_t* operations;
  // Number of source operations.
  uint32_t count;
  // Indices of source-order boundaries in this block, in authored order.
  // Fences inside nested regions retain their scope within the cloned unit.
  uint32_t* source_order_boundaries;
  // Number of top-level source-order boundaries.
  uint32_t source_order_boundary_count;
  // Complete local payload dependencies, grouped by operation.
  loom_scf_body_reference_t* references;
  // Number of entries in references.
  iree_host_size_t reference_count;
  // Terminator payload dependencies, used when forwarding carried state.
  loom_scf_body_operation_t terminator;
} loom_scf_body_t;

// With |mode| PROJECT_MEMORY, captures only the accesses needed for cloning.
// Arbitrary verified nested control remains legal in that mode. With SCHEDULE,
// a non-NULL |spaces| adds space-qualified effects and clone correspondence;
// consumers needing only scheduling dependencies pass NULL without allocating
// the optional access arrays.
//
// Captures the verified |block|'s live operations, source effects and complete
// local SSA dependencies in one traversal. When |capture_block| is non-NULL,
// values defined directly in that enclosing block are retained as dependencies
// alongside values defined in |block|. This lets an enclosing schedule plan a
// structured branch without turning definitions from deeper nested regions
// into independent scheduling edges. A structured if/for and its regions form
// one scheduling unit; their outer-body captures and effects are retained
// together. Result-type dependencies come from the IR's
// maintained type-use table. Attributes, including predicates and encoding
// parameters, are traversed once during construction. External captures and
// self references in an operation's result types need no scheduling edge.
//
// Nested control other than scf.if/scf.for, or an operation with successors, is
// returned through |out_unstructured_op| for a source-policy diagnostic. The
// body is ready for scheduling when no unsupported operation is returned.
// Status failures identify allocation or size limits. All arrays belong to
// |arena|, borrow the source IR, and remain valid while that IR is unchanged;
// emitting clones does not invalidate them.
iree_status_t loom_scf_body_build(
    const loom_module_t* module, const loom_block_t* block,
    const loom_block_t* capture_block, const loom_scf_memory_t* spaces,
    loom_scf_body_mode_t mode, iree_arena_allocator_t* arena,
    loom_scf_body_t* out_body, const loom_op_t** out_unstructured_op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_SCF_SCF_BODY_H_
