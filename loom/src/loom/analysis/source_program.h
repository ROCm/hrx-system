// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Retained source-program structural index.
//
// Analyses and lowering stages that inspect one source region tree share this
// immutable index instead of recursively walking the same IR from independent
// queries. The index is a compact pre-order event stream. Block events retain
// argument-boundary context, operation events retain child-subtree limits, and
// all semantic payload remains in the borrowed source IR.

#ifndef LOOM_ANALYSIS_SOURCE_PROGRAM_H_
#define LOOM_ANALYSIS_SOURCE_PROGRAM_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/local_value_domain.h"

#ifdef __cplusplus
extern "C" {
#endif

// Dense ordinal of one source-program event.
typedef uint32_t loom_source_program_node_ordinal_t;

// Sentinel used when no source-program event ordinal exists.
#define LOOM_SOURCE_PROGRAM_NODE_ORDINAL_INVALID UINT32_MAX

enum loom_source_program_node_kind_e {
  // Marks entry into one source block before any of its operations.
  LOOM_SOURCE_PROGRAM_NODE_BLOCK = 0,
  // Visits one source operation before any child regions.
  LOOM_SOURCE_PROGRAM_NODE_OPERATION = 1,
};
typedef uint8_t loom_source_program_node_kind_t;

enum loom_source_program_node_flag_bits_e {
  // The block is the entry block of the indexed root region.
  LOOM_SOURCE_PROGRAM_NODE_ROOT_ENTRY_BLOCK = (uint8_t)1u << 0,
};
typedef uint8_t loom_source_program_node_flags_t;

// One event in source-program pre-order.
typedef struct loom_source_program_node_t {
  // Borrowed loom_block_t or loom_op_t selected by kind.
  const void* object;
  // Operation owning a block's region, or NULL for operation events when the
  // source operation itself already carries its parent.
  const loom_op_t* context_op;
  // One-past event ordinal of this block or operation's complete nested
  // subtree. Leaf operations use their own ordinal plus one.
  loom_source_program_node_ordinal_t subtree_limit;
  // Region nesting depth, with the indexed root region at depth zero.
  uint16_t region_depth;
  // Event payload kind.
  loom_source_program_node_kind_t kind;
  // Event classification flags.
  loom_source_program_node_flags_t flags;
} loom_source_program_node_t;

static_assert(sizeof(loom_source_program_node_t) == 24,
              "loom_source_program_node_t must be 24 bytes");

// Immutable structural index for one source region tree.
typedef struct loom_source_program_t {
  // Borrowed module containing the indexed source program.
  const loom_module_t* module;
  // Borrowed root region covered by the index.
  const loom_region_t* root_region;
  // Borrowed dense value domain covering root_region and its child regions.
  const loom_local_value_domain_t* value_domain;
  // Pre-order block and operation event array.
  loom_source_program_node_t* nodes;
  // Number of populated events in nodes.
  loom_source_program_node_ordinal_t node_count;
  // Allocated event capacity in nodes.
  iree_host_size_t node_capacity;
  // Number of indexed operations.
  uint32_t operation_count;
  // Number of indexed blocks.
  uint32_t block_count;
  // Number of indexed regions.
  uint32_t region_count;
} loom_source_program_t;

// Builds a retained structural index for |root_region|.
//
// |root_context_op| supplies diagnostic/type-mapping context for root block
// arguments. Nested block events retain the operation that owns their region.
// |value_domain| must cover the complete region tree. All index storage is
// arena-owned and remains valid until |arena| is reset or deinitialized.
iree_status_t loom_source_program_build(
    const loom_module_t* module, const loom_op_t* root_context_op,
    const loom_region_t* root_region,
    const loom_local_value_domain_t* value_domain,
    iree_arena_allocator_t* arena, loom_source_program_t* out_program);

// Returns the block carried by |node|.
static inline const loom_block_t* loom_source_program_node_block(
    const loom_source_program_node_t* node) {
  IREE_ASSERT_EQ(node->kind, LOOM_SOURCE_PROGRAM_NODE_BLOCK);
  return (const loom_block_t*)node->object;
}

// Returns the operation carried by |node|.
static inline const loom_op_t* loom_source_program_node_operation(
    const loom_source_program_node_t* node) {
  IREE_ASSERT_EQ(node->kind, LOOM_SOURCE_PROGRAM_NODE_OPERATION);
  return (const loom_op_t*)node->object;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_SOURCE_PROGRAM_H_
