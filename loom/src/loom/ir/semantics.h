// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Semantic metadata vocabulary shared by generated dialect tables and runtime
// compiler policy.

#ifndef LOOM_IR_SEMANTICS_H_
#define LOOM_IR_SEMANTICS_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Source-level placement phase for an op kind. This names where the op belongs
// in Loom's semantic pipeline; target legality is derived from this metadata
// and target-contract families instead of being authored per target.
typedef uint8_t loom_op_phase_t;

typedef enum loom_op_phase_e {
  // No semantic phase has been declared for this op kind.
  LOOM_OP_PHASE_UNSPECIFIED = 0,
  // Ordinary executable program operation.
  LOOM_OP_PHASE_EXECUTABLE = 1,
  // Source-structure operation that should be lowered away before target-low
  // IR.
  LOOM_OP_PHASE_SOURCE_STRUCTURE = 2,
  // Module metadata consumed by compilation/configuration rather than emitted.
  LOOM_OP_PHASE_MODULE_METADATA = 3,
  // Compile-time query that must resolve or become dead before target-low IR.
  LOOM_OP_PHASE_COMPILE_TIME_QUERY = 4,
} loom_op_phase_e;

// Fact-dependent condition under which an operation may execute on additional
// control paths. Zero means that no conditional execution-safety proof is
// declared; unconditional safety remains represented by
// LOOM_TRAIT_SAFE_TO_SPECULATE.
typedef uint8_t loom_execution_condition_t;

typedef enum loom_execution_condition_e {
  LOOM_EXECUTION_CONDITION_NONE = 0,
  // Integer left shift with a count in [0, element width). Fixed-width
  // payloads wrap unless canonical no-wrap instance flags require a stronger
  // operand proof. Index results must fit the selected signed target carrier.
  LOOM_EXECUTION_CONDITION_INTEGER_SHIFT_LEFT = 1,
  // Integer arithmetic or logical right shift with a count in
  // [0, element width).
  LOOM_EXECUTION_CONDITION_INTEGER_SHIFT_RIGHT = 2,
  // Integer addition whose declared no-wrap promises hold, or mathematical
  // index addition whose result fits the selected signed target carrier.
  // Offset addition is total modulo its unsigned target carrier.
  LOOM_EXECUTION_CONDITION_INTEGER_ADD = 3,
  // Integer subtraction with the same fixed-width, index, and offset domains
  // as INTEGER_ADD.
  LOOM_EXECUTION_CONDITION_INTEGER_SUBTRACT = 4,
  // Integer multiplication whose declared no-wrap promises hold, or
  // mathematical index multiplication whose result fits the selected signed
  // target carrier.
  LOOM_EXECUTION_CONDITION_INTEGER_MULTIPLY = 5,
  // Integer multiply-add whose declared no-wrap promises hold, or
  // mathematical index multiply-add whose result fits the selected signed
  // target carrier.
  LOOM_EXECUTION_CONDITION_INTEGER_MULTIPLY_ADD = 6,
  // Unsigned integer quotient or signed/unsigned remainder with a nonzero
  // divisor. Signed remainder remains defined for MIN modulo -1.
  LOOM_EXECUTION_CONDITION_INTEGER_NONZERO_DIVISOR = 7,
  // Signed integer quotient with a nonzero divisor and a representable
  // result: the dividend and divisor cannot be MIN and -1 together.
  LOOM_EXECUTION_CONDITION_INTEGER_SIGNED_QUOTIENT = 8,
  // Logical index quotient/remainder with a nonnegative dividend and a
  // positive divisor. The result fits independently of target index width.
  LOOM_EXECUTION_CONDITION_INDEX_DIVISION = 9,
  LOOM_EXECUTION_CONDITION_COUNT_ = 10,
} loom_execution_condition_e;

static_assert(LOOM_EXECUTION_CONDITION_COUNT_ <= UINT8_MAX + 1u,
              "execution conditions must fit in uint8_t");

// Bitset of target-contract families represented by an op or type.
typedef uint32_t loom_contract_family_set_t;

enum loom_contract_family_bits_e {
  // Vector coordinate materialization contract.
  LOOM_CONTRACT_VECTOR_COORDINATE = 1u << 0,
  // Register permutation contract.
  LOOM_CONTRACT_REGISTER_PERMUTATION = 1u << 1,
  // Vector table lookup contract.
  LOOM_CONTRACT_VECTOR_TABLE_LOOKUP = 1u << 2,
  // Vector contraction contract.
  LOOM_CONTRACT_VECTOR_CONTRACTION = 1u << 3,
  // Atomic memory access contract.
  LOOM_CONTRACT_MEMORY_ATOMIC = 1u << 4,
  // Kernel asynchronous pipeline contract.
  LOOM_CONTRACT_KERNEL_ASYNC = 1u << 5,
  // Kernel synchronization contract.
  LOOM_CONTRACT_KERNEL_SYNCHRONIZATION = 1u << 6,
  // Tensor memory transfer contract.
  LOOM_CONTRACT_TENSOR_MEMORY = 1u << 7,
  // Sanitizer race observation contract.
  LOOM_CONTRACT_SANITIZER_RACE = 1u << 8,
};

// Returns true when |set| contains any contract family from |mask|.
static inline bool loom_contract_family_set_has_any(
    loom_contract_family_set_t set, loom_contract_family_set_t mask) {
  return (set & mask) != 0;
}

// Returns true when |set| contains every contract family from |mask|.
static inline bool loom_contract_family_set_has_all(
    loom_contract_family_set_t set, loom_contract_family_set_t mask) {
  return (set & mask) == mask;
}

// Generated semantic metadata for one op kind.
typedef struct loom_op_semantics_t {
  // Source-level placement phase.
  loom_op_phase_t phase;
  // One-based index into the dialect's sparse condition-refinement table, or
  // zero when the op does not refine an operand on a control-flow edge.
  uint8_t condition_refinement_index;
  // Fact-dependent execution-safety condition, or NONE when no shared proof
  // can establish that the op is safe to execute on an additional path.
  loom_execution_condition_t execution_condition;
  // Relative cost for transforms that eagerly execute the op on additional
  // control paths, or zero when generic eager execution is not profitable.
  uint8_t eager_execution_cost;
  // All execution effects manage the declared operand/result lifetimes.
  // A closed unused lifetime may disappear together; this does not permit
  // dropping cleanup of a live value or moving its lifetime boundaries.
  bool lifetime_effects_only;
  // One-based payload effect-contract attribute index for an application,
  // or zero when no such contract is declared. Independent of direct-callee
  // identity so unresolved template applications share the same semantics.
  uint8_t callable_effect_attr_index_plus_one;
  // Target-contract families required to preserve this op's semantics.
  loom_contract_family_set_t contract_families;
} loom_op_semantics_t;

static_assert(sizeof(loom_op_semantics_t) == 12,
              "op semantic rows must remain twelve bytes");

// Returns the default empty semantic metadata row.
static inline loom_op_semantics_t loom_op_semantics_empty(void) {
  return (loom_op_semantics_t){0};
}

// Semantic role for registered non-scalar type declarations.
typedef uint8_t loom_type_semantic_t;

typedef enum loom_type_semantic_e {
  // Ordinary value type with no special control or target-contract role.
  LOOM_TYPE_SEMANTIC_ORDINARY = 0,
  // Opaque control token used to sequence semantic effects.
  LOOM_TYPE_SEMANTIC_CONTROL_TOKEN = 1,
  // Opaque value whose payload represents a target contract.
  LOOM_TYPE_SEMANTIC_TARGET_CONTRACT_VALUE = 2,
  // Reference-counted resource requiring ownership-aware lowering.
  LOOM_TYPE_SEMANTIC_MANAGED_REFERENCE = 3,
} loom_type_semantic_e;

// Generated semantic metadata for one registered type declaration.
typedef struct loom_type_semantics_t {
  // Semantic role for values of this type.
  loom_type_semantic_t semantic;
  // Target-contract families embodied by this type.
  loom_contract_family_set_t contract_families;
} loom_type_semantics_t;

// Returns the default ordinary type semantic metadata row.
static inline loom_type_semantics_t loom_type_semantics_empty(void) {
  return (loom_type_semantics_t){0};
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_IR_SEMANTICS_H_
