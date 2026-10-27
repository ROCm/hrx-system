// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocation-wide promotion of private typed cells to ordinary SSA.
//
// Shared byte-region facts partition accesses by allocation and exact typed
// byte interval. Every use of an allocation participates in admission: unknown
// observers, overlapping representations, and observable accesses keep storage.
// Reaching values and control-flow transports are planned before any mutation.

#ifndef LOOM_TRANSFORMS_KERNEL_PRIVATE_STORAGE_H_
#define LOOM_TRANSFORMS_KERNEL_PRIVATE_STORAGE_H_

#include "loom/ir/local_value_domain.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_private_storage_allocation_t
    loom_private_storage_allocation_t;
typedef struct loom_private_storage_cell_t loom_private_storage_cell_t;
typedef struct loom_private_storage_value_t loom_private_storage_value_t;
typedef struct loom_private_storage_transport_t
    loom_private_storage_transport_t;

// One allocation is the admission unit. Losing an observer keeps all its cells
// in memory, including cells that are otherwise individually scalarizable.
struct loom_private_storage_allocation_t {
  // Original private buffer.alloca.
  loom_op_t* op;
  // Next allocation in reverse discovery order.
  loom_private_storage_allocation_t* next;
  // Exact allocation size, or zero when no static size is known.
  int64_t byte_count;
  // First cell in the allocation's contiguous partition.
  iree_host_size_t cell_begin;
  // Number of cells reset when this allocation executes.
  iree_host_size_t cell_count;
  // Whether every observer and live reaching value supports promotion.
  bool selected;
};

// Exact typed byte interval shared by all equivalent accesses. Cells are sparse
// in the bytes actually used; allocation size does not size the plan.
struct loom_private_storage_cell_t {
  // Owning allocation and its whole-object admission decision.
  loom_private_storage_allocation_t* allocation;
  // Root-relative inclusive byte offset.
  int64_t begin;
  // Root-relative exclusive byte offset.
  int64_t end;
  // Scalar value type, identical for every access to this interval.
  loom_type_t type;
  // Dense index in the plan's cell array.
  iree_host_size_t index;
};

// A planned reaching definition. Original SSA definitions are leaves; merges
// retain their incoming definitions and the control signature that owns them.
// NULL represents an uninitialized cell, never an IR poison or invented value.
struct loom_private_storage_value_t {
  // Original SSA definition for a leaf, INVALID for a planned transport.
  loom_value_id_t source;
  // SSA definition assigned by the rewrite, initially INVALID.
  loom_value_id_t replacement;
  // Canonical definition after trivial merge folding.
  loom_private_storage_value_t* representative;
  // Incoming reaching definitions for a transport.
  loom_private_storage_value_t** inputs;
  // Number of incoming definitions.
  iree_host_size_t input_count;
  // Transport owning this definition, or NULL for an original SSA value.
  loom_private_storage_transport_t* transport;
  // Dense ordinal used by the shared SCC solver.
  iree_host_size_t ordinal;
  // Whether every incoming path has an initialized value.
  bool initialized;
};

// One memory access and its retained cell/replacement decision.
typedef struct loom_private_storage_access_t {
  // Original view.load or view.store.
  loom_op_t* op;
  // Exact admitted byte cell.
  loom_private_storage_cell_t* cell;
  // Reaching definition for a load, stored SSA definition for a store.
  loom_private_storage_value_t* value;
  // Next access in discovery order.
  struct loom_private_storage_access_t* next;
} loom_private_storage_access_t;

// One appended control-flow tuple slot. A branch has a result only; a counted
// loop also has an entry argument; a condition loop additionally forwards the
// condition region's state into a body argument. A CFG join has an entry only.
struct loom_private_storage_transport_t {
  // Cell whose stored value crosses this boundary.
  loom_private_storage_cell_t* cell;
  // New block argument at CFG entry or the loop's recurring entry.
  loom_private_storage_value_t* entry;
  // Condition-controlled loop body argument, or NULL.
  loom_private_storage_value_t* body;
  // New structured-operation result, or NULL for a CFG entry.
  loom_private_storage_value_t* result;
  // Incoming initial value for a loop.
  loom_private_storage_value_t* initial;
  // Values forwarded by each structured region terminator or CFG predecessor.
  loom_private_storage_value_t** outgoing;
  // Number of entries in outgoing.
  iree_host_size_t outgoing_count;
  // Next slot in the same signature.
  loom_private_storage_transport_t* next;
  // Whether a promoted load needs this slot and its incoming values.
  bool live;
};

// Retained CFG predecessor, including the edge ordinal needed for critical
// edge splitting when the terminator cannot carry a block-argument payload.
typedef struct loom_private_storage_predecessor_t {
  // Original predecessor terminator.
  loom_op_t* terminator;
  // Successor ordinal targeting the joined block.
  uint16_t successor_index;
} loom_private_storage_predecessor_t;

// Control signature to extend. Existing arguments, results, attributes, and
// region payloads remain the prefix; promotion appends the live cell slots.
typedef struct loom_private_storage_signature_t {
  // Structured operation, or NULL for a CFG block signature.
  loom_op_t* op;
  // CFG join block, or NULL for a structured operation.
  loom_block_t* block;
  // Resulting structured region count, including an explicit fallthrough.
  uint8_t region_count;
  // Ordered appended slots.
  loom_private_storage_transport_t* transports;
  // Indexed predecessor edges for a CFG block.
  loom_private_storage_predecessor_t* predecessors;
  // Number of predecessor edges.
  iree_host_size_t predecessor_count;
  // Next signature in child-before-parent rewrite order.
  struct loom_private_storage_signature_t* next;
} loom_private_storage_signature_t;

// Retained allocation/access/value-flow plan for one function. All storage
// belongs to the pass arena. The local domain remains acquired until release.
// Admission and incoming edges are fixed after construction; rewriting updates
// only the SSA correspondence and representative links. Consumers replay these
// decisions after mutation invalidates the borrowed analysis facts.
typedef struct loom_private_storage_plan_t {
  // Module whose IR identities the plan borrows.
  loom_module_t* module;
  // Arena owning plan arrays and records.
  iree_arena_allocator_t* arena;
  // Compact value domain used by the plan and rewrite correspondence.
  loom_local_value_domain_t domain;
  // Allocation admission records.
  loom_private_storage_allocation_t* allocations;
  // Sparse byte cells in allocation/offset order.
  loom_private_storage_cell_t* cells;
  // Number of cells.
  iree_host_size_t cell_count;
  // Memory accesses with retained cell identity.
  loom_private_storage_access_t* accesses;
  // Signature edits in child-before-parent order.
  loom_private_storage_signature_t* signatures;
  // Original SSA leaves indexed by local ordinal, or NULL when unreferenced.
  loom_private_storage_value_t** values;
  // Promotable load accesses indexed by result ordinal, or NULL.
  loom_private_storage_access_t** loads;
  // Private storage projections erased after their memory users are removed.
  loom_op_t** projections;
  // Number of entries in projections.
  iree_host_size_t projection_count;
} loom_private_storage_plan_t;

// Builds an allocation-wide promotion plan. The no-private-allocation path
// leaves an empty plan without acquiring facts, regions, or a local domain.
// Release the plan even when construction fails after acquiring its domain.
iree_status_t loom_private_storage_plan_build(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function,
    loom_private_storage_plan_t* out_plan);

// Releases the module's local ordinal scratch. Arena storage stays
// caller-owned.
void loom_private_storage_plan_release(loom_private_storage_plan_t* plan);

// Returns the retained representative of a reaching definition, or NULL for an
// uninitialized cell. This follows plan identities, never IR use-def chains.
loom_private_storage_value_t* loom_private_storage_value_resolve(
    loom_private_storage_value_t* value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_KERNEL_PRIVATE_STORAGE_H_
