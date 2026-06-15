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
} loom_op_phase_e;

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
  // Target-contract families required to preserve this op's semantics.
  loom_contract_family_set_t contract_families;
} loom_op_semantics_t;

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
