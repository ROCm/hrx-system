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

// Dense ordinal of one indexed source use.
typedef uint32_t loom_source_program_use_ordinal_t;

// Sentinel used when no source use or span exists.
#define LOOM_SOURCE_PROGRAM_USE_ORDINAL_INVALID UINT32_MAX

enum loom_source_program_value_flow_kind_bits_e {
  // A CFG successor payload flows into a destination block argument.
  LOOM_SOURCE_PROGRAM_VALUE_FLOW_CFG_PAYLOAD = (uint16_t)1u << 0,
  // An operand transfers storage ownership to a tied result.
  LOOM_SOURCE_PROGRAM_VALUE_FLOW_TIED_RESULT = (uint16_t)1u << 1,
  // An operand/result pair is covered by the fact-identity trait.
  LOOM_SOURCE_PROGRAM_VALUE_FLOW_FACT_IDENTITY = (uint16_t)1u << 2,
  // An operand/result pair is covered by the value-alias trait.
  LOOM_SOURCE_PROGRAM_VALUE_FLOW_VALUE_ALIAS = (uint16_t)1u << 3,
  // Values occupy the same logical LoopLike carried-state position.
  LOOM_SOURCE_PROGRAM_VALUE_FLOW_LOOP_CARRY = (uint16_t)1u << 4,
  // A RegionBranch yield flows to its corresponding operation result.
  LOOM_SOURCE_PROGRAM_VALUE_FLOW_REGION_YIELD = (uint16_t)1u << 5,
  // A condition guards values converging at one CFG merge position.
  LOOM_SOURCE_PROGRAM_VALUE_FLOW_CONTROL_MERGE = (uint16_t)1u << 6,
  // A later condition must be available while an earlier condition is live.
  LOOM_SOURCE_PROGRAM_VALUE_FLOW_CONDITION_ORDER = (uint16_t)1u << 7,
};
typedef uint16_t loom_source_program_value_flow_kinds_t;

#define LOOM_SOURCE_PROGRAM_VALUE_FLOW_KIND_MASK                                           \
  ((loom_source_program_value_flow_kinds_t)(LOOM_SOURCE_PROGRAM_VALUE_FLOW_CFG_PAYLOAD |   \
                                            LOOM_SOURCE_PROGRAM_VALUE_FLOW_TIED_RESULT |   \
                                            LOOM_SOURCE_PROGRAM_VALUE_FLOW_FACT_IDENTITY | \
                                            LOOM_SOURCE_PROGRAM_VALUE_FLOW_VALUE_ALIAS |   \
                                            LOOM_SOURCE_PROGRAM_VALUE_FLOW_LOOP_CARRY |    \
                                            LOOM_SOURCE_PROGRAM_VALUE_FLOW_REGION_YIELD |  \
                                            LOOM_SOURCE_PROGRAM_VALUE_FLOW_CONTROL_MERGE | \
                                            LOOM_SOURCE_PROGRAM_VALUE_FLOW_CONDITION_ORDER))

// One target-independent directed flow between source values.
//
// Flow kinds retain why the edge exists. Analyses select the kinds and
// directions relevant to their evidence domain. Representation planning may
// treat selected preserving kinds as undirected component-union edges without
// discarding direction for all other consumers. Explicit representation
// transitions are intentionally absent.
typedef struct loom_source_program_value_flow_t {
  // Function-local value ordinal at the natural source of the flow.
  loom_value_ordinal_t source;
  // Function-local value ordinal at the natural target of the flow.
  loom_value_ordinal_t target;
  // Reasons this directed flow exists.
  loom_source_program_value_flow_kinds_t kinds;
  // Reserved for common source-program flow flags; must be zero.
  uint16_t reserved;
} loom_source_program_value_flow_t;

static_assert(sizeof(loom_source_program_value_flow_t) == 12,
              "loom_source_program_value_flow_t must be 12 bytes");

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

enum loom_source_program_value_flag_bits_e {
  // The defining operation or block is inside the indexed region tree.
  LOOM_SOURCE_PROGRAM_VALUE_DEFINED = (uint16_t)1u << 0,
  // The value is an indexed block argument rather than an operation result.
  LOOM_SOURCE_PROGRAM_VALUE_BLOCK_ARGUMENT = (uint16_t)1u << 1,
  // The value is an argument of the indexed root region's entry block.
  LOOM_SOURCE_PROGRAM_VALUE_ROOT_ENTRY_ARGUMENT = (uint16_t)1u << 2,
  // At least one indexed CFG payload flows into this block argument.
  LOOM_SOURCE_PROGRAM_VALUE_HAS_CFG_PREDECESSOR = (uint16_t)1u << 3,
  // An operation result is used by an operation in another source block.
  LOOM_SOURCE_PROGRAM_VALUE_HAS_CROSS_BLOCK_USE = (uint16_t)1u << 4,
  // At least one use has the generic control-condition operand role.
  LOOM_SOURCE_PROGRAM_VALUE_HAS_CONTROL_CONDITION_USE = (uint16_t)1u << 5,
  // At least one use has the generic select-condition operand role.
  LOOM_SOURCE_PROGRAM_VALUE_HAS_SELECT_CONDITION_USE = (uint16_t)1u << 6,
  // At least one use has the generic select-payload operand role.
  LOOM_SOURCE_PROGRAM_VALUE_HAS_SELECT_PAYLOAD_USE = (uint16_t)1u << 7,
  // At least one use has the generic broadcast-source operand role.
  LOOM_SOURCE_PROGRAM_VALUE_HAS_BROADCAST_SOURCE_USE = (uint16_t)1u << 8,
  // At least one use has the generic composite-element operand role.
  LOOM_SOURCE_PROGRAM_VALUE_HAS_COMPOSITE_ELEMENT_USE = (uint16_t)1u << 9,
};
typedef uint16_t loom_source_program_value_flags_t;

#define LOOM_SOURCE_PROGRAM_VALUE_FLAG_MASK                                                  \
  ((loom_source_program_value_flags_t)(LOOM_SOURCE_PROGRAM_VALUE_DEFINED |                   \
                                       LOOM_SOURCE_PROGRAM_VALUE_BLOCK_ARGUMENT |            \
                                       LOOM_SOURCE_PROGRAM_VALUE_ROOT_ENTRY_ARGUMENT |       \
                                       LOOM_SOURCE_PROGRAM_VALUE_HAS_CFG_PREDECESSOR |       \
                                       LOOM_SOURCE_PROGRAM_VALUE_HAS_CROSS_BLOCK_USE |       \
                                       LOOM_SOURCE_PROGRAM_VALUE_HAS_CONTROL_CONDITION_USE | \
                                       LOOM_SOURCE_PROGRAM_VALUE_HAS_SELECT_CONDITION_USE |  \
                                       LOOM_SOURCE_PROGRAM_VALUE_HAS_SELECT_PAYLOAD_USE |    \
                                       LOOM_SOURCE_PROGRAM_VALUE_HAS_BROADCAST_SOURCE_USE |  \
                                       LOOM_SOURCE_PROGRAM_VALUE_HAS_COMPOSITE_ELEMENT_USE))

// Dense structural metadata for one function-local source value.
typedef struct loom_source_program_value_t {
  // Node defining the value, or INVALID when defined outside the program.
  loom_source_program_node_ordinal_t definition_node;
  // Block node containing the definition, or INVALID when external.
  loom_source_program_node_ordinal_t definition_block_node;
  // First indexed use in the program use pool.
  loom_source_program_use_ordinal_t use_start;
  // Number of indexed uses in the program use pool.
  uint32_t use_count;
  // Definition and aggregate-use classification flags.
  loom_source_program_value_flags_t flags;
  // Reserved for common value classifications; must be zero.
  uint16_t reserved;
} loom_source_program_value_t;

static_assert(sizeof(loom_source_program_value_t) == 20,
              "loom_source_program_value_t must be 20 bytes");

// One source operand use retained in deterministic program order.
typedef struct loom_source_program_use_t {
  // Function-local ordinal of the used value.
  loom_value_ordinal_t value;
  // Operation node containing the use.
  loom_source_program_node_ordinal_t user_node;
  // Physical operand index in the user operation.
  uint16_t operand_index;
  // Generic loom_operand_role_t value from the user operation descriptor.
  uint8_t operand_role;
  // Reserved for common use classifications; must be zero.
  uint8_t reserved;
} loom_source_program_use_t;

static_assert(sizeof(loom_source_program_use_t) == 12,
              "loom_source_program_use_t must be 12 bytes");

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
  // Dense structural metadata indexed by local value ordinal.
  loom_source_program_value_t* values;
  // Source operand uses grouped by value and ordered by user node.
  loom_source_program_use_t* uses;
  // Number of populated entries in uses.
  loom_source_program_use_ordinal_t use_count;
  // Allocated use capacity in uses.
  iree_host_size_t use_capacity;
  // Typed directed source-value flows.
  loom_source_program_value_flow_t* value_flows;
  // Number of populated entries in value_flows.
  uint32_t value_flow_count;
  // Allocated flow capacity in value_flows.
  iree_host_size_t value_flow_capacity;
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

// Returns dense metadata for |value_id|, or NULL when it is outside the
// indexed value domain.
static inline const loom_source_program_value_t* loom_source_program_try_value(
    const loom_source_program_t* program, loom_value_id_t value_id) {
  const loom_value_ordinal_t ordinal =
      loom_local_value_domain_try_ordinal(program->value_domain, value_id);
  return ordinal == LOOM_VALUE_ORDINAL_INVALID ? NULL
                                               : &program->values[ordinal];
}

// Returns the retained use span for |value|.
static inline const loom_source_program_use_t* loom_source_program_value_uses(
    const loom_source_program_t* program,
    const loom_source_program_value_t* value) {
  return value->use_count == 0 ? NULL : &program->uses[value->use_start];
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_SOURCE_PROGRAM_H_
