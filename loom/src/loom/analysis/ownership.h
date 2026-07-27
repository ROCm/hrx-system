// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent ownership metadata queries.
//
// Ownership facts are static op descriptor metadata interpreted through a
// resource-family policy. The metadata says how operands and results transfer
// ownership; the policy decides which concrete value types participate.

#ifndef LOOM_ANALYSIS_OWNERSHIP_H_
#define LOOM_ANALYSIS_OWNERSHIP_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef bool (*loom_ownership_type_matches_fn_t)(loom_type_t type,
                                                 void* user_data);

// Managed-resource family recognized by an ownership analysis consumer.
typedef struct loom_ownership_resource_family_t {
  // Human-readable resource family name used in diagnostics.
  iree_string_view_t name;
  // Type predicate for values belonging to this resource family.
  loom_ownership_type_matches_fn_t type_matches;
  // Opaque payload passed to |type_matches|.
  void* user_data;
} loom_ownership_resource_family_t;

// Concrete ownership effect for one operation operand value.
typedef struct loom_ownership_operand_effect_t {
  // Operand value index on the operation.
  uint16_t operand_index;
  // Descriptor field index that owns this operand value.
  uint8_t field_index;
  // Value ID carried by the operand.
  loom_value_id_t value_id;
  // Ownership action declared by the operand descriptor.
  loom_operand_ownership_effect_t effect;
  // Carrier mode declared by the operand descriptor.
  loom_ownership_carrier_t carrier;
  // Borrowed descriptor that supplied the ownership metadata.
  const loom_operand_descriptor_t* descriptor;
} loom_ownership_operand_effect_t;

// Concrete ownership effect for one operation result value.
typedef struct loom_ownership_result_effect_t {
  // Result value index on the operation.
  uint16_t result_index;
  // Descriptor field index that owns this result value.
  uint8_t field_index;
  // Operand index used by aliasing/tied effects, or NONE.
  uint16_t source_operand_index;
  // Value ID produced by the result.
  loom_value_id_t value_id;
  // Ownership action declared by the result descriptor.
  loom_result_ownership_effect_t effect;
  // Borrowed descriptor that supplied the ownership metadata.
  const loom_result_descriptor_t* descriptor;
} loom_ownership_result_effect_t;

// Returns true if |type| belongs to |family|.
bool loom_ownership_type_matches(const loom_ownership_resource_family_t* family,
                                 loom_type_t type);

// Returns true if |value_id| belongs to |family|.
bool loom_ownership_value_matches(
    const loom_module_t* module, const loom_ownership_resource_family_t* family,
    loom_value_id_t value_id);

// Resolves ownership metadata for one operand value. Returns false when the
// operand has no ownership effect or the index is outside the op's operands.
bool loom_ownership_operand_effect_at(
    const loom_module_t* module, const loom_op_t* op, uint16_t operand_index,
    loom_ownership_operand_effect_t* out_effect);

// Resolves ownership metadata for one result value. Returns false when the
// result has no ownership effect or the index is outside the op's results.
bool loom_ownership_result_effect_at(
    const loom_module_t* module, const loom_op_t* op, uint16_t result_index,
    loom_ownership_result_effect_t* out_effect);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_OWNERSHIP_H_
