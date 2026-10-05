// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU physical representations for narrow scalar integer carriers.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_SOURCE_INTEGER_REPRESENTATION_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_SOURCE_INTEGER_REPRESENTATION_H_

#include "loom/codegen/low/lower/representation_observer.h"

#ifdef __cplusplus
extern "C" {
#endif

// The declared i8/i16 payload is preserved in the low carrier bits. Bits above
// the declared width have no usable value.
#define LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_LOW_BITS \
  ((loom_low_representation_id_t)UINT16_C(259))

// Bits above the declared i8/i16 width replicate its sign bit.
#define LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_SIGN_EXTENDED \
  ((loom_low_representation_id_t)UINT16_C(260))

// Bits above the declared i8/i16 width are zero.
#define LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_ZERO_EXTENDED \
  ((loom_low_representation_id_t)UINT16_C(261))

typedef uint8_t loom_amdgpu_source_integer_representation_action_t;

enum loom_amdgpu_source_integer_representation_action_e {
  // A narrow result can preserve low bits or normalize its carrier.
  LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_FLEXIBLE_RESULT = 5,
  // A narrow result is emitted as a sign-extended carrier.
  LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_SIGN_EXTENDED_RESULT = 6,
  // A narrow integer constant has value-dependent carrier guarantees.
  LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_CONSTANT_RESULT = 7,
  // A signed conversion consumes a narrow operand and may produce one.
  LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_SIGNED_CONVERSION = 8,
  // An unsigned conversion consumes a narrow operand and may produce one.
  LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_UNSIGNED_CONVERSION = 9,
  // First payload operand/result preserve the same native carrier bits.
  LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_TRANSPORT_PAYLOAD = 10,
};

// Returns whether |type| uses a target narrow integer carrier.
static inline bool loom_amdgpu_source_integer_representation_type_is_narrow(
    loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return scalar_type == LOOM_SCALAR_TYPE_I8 ||
         scalar_type == LOOM_SCALAR_TYPE_I16;
}

// Records target alternatives at one narrow integer operation boundary.
void loom_amdgpu_source_integer_representation_observe_boundary(
    loom_amdgpu_source_integer_representation_action_t action,
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder);

// Records the carrier guarantee visible at narrow callable inputs.
void loom_amdgpu_source_integer_representation_observe_callable_boundary(
    loom_low_lower_representation_callable_boundary_kind_t kind,
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder);

// Returns the solved narrow integer representation for |source_value_id|.
loom_low_representation_id_t loom_amdgpu_source_integer_representation_lookup(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id);

// Normalizes the declared i8/i16 payload in |low_source| to
// |required_representation|. The source and result are one-unit VGPR values;
// the input is reused when it already carries the required representation.
iree_status_t loom_amdgpu_normalize_narrow_integer(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t source_bit_count,
    loom_low_representation_id_t source_representation,
    loom_low_representation_id_t required_representation,
    loom_value_id_t* out_low_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SOURCE_INTEGER_REPRESENTATION_H_
