// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Sparse declared input requirements in signature-local coordinates.

#ifndef LOOM_IR_INPUT_CONTRACT_H_
#define LOOM_IR_INPUT_CONTRACT_H_

#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// One requirement on an actual input. Static encoding identities belong to
// the module owning the contract; zero denotes native dense representation.
typedef struct loom_input_requirement_t {
  // Argument whose dimension or complete encoding is constrained.
  uint16_t argument;
  // Dimension axis, or UINT8_MAX for the complete encoding.
  uint8_t axis;
  // True when expected identifies another input argument instead of a literal.
  bool expected_is_argument;
  // Zeroed padding for canonical immutable rows.
  uint32_t reserved;
  // Literal extent, static encoding identity, or zero-based input ordinal.
  uint64_t expected;
} loom_input_requirement_t;

// Family obligations or a provider's additional input obligations.
typedef struct loom_input_contract_t {
  // Number of immutable requirements in values.
  uint32_t count;
  // Zeroed padding aligning the trailing records.
  uint32_t reserved;
  // Requirements in argument/axis order, followed by each argument's encoding.
  loom_input_requirement_t values[];
} loom_input_contract_t;

typedef uint8_t loom_input_encoding_kind_t;
enum loom_input_encoding_kind_e {
  LOOM_INPUT_ENCODING_NONE = 0,
  LOOM_INPUT_ENCODING_STATIC = 1,
  LOOM_INPUT_ENCODING_ARGUMENT = 2,
};

// Construction-time projection of one shaped signature input. Composite and
// nominal roots remain exact and do not have a shape projection.
typedef struct loom_input_shape_t {
  // Native group, tensor, tile, vector, or view type kind.
  uint8_t kind;
  // Scalar element type, or NONE for a group.
  uint8_t element_type;
  // Number of trailing dimensions.
  uint8_t rank;
  // View alignment override in bytes, or zero for natural alignment.
  uint8_t access_alignment;
  // Bit i selects a one-based input ordinal in dimensions[i].
  uint16_t dynamic_axes;
  // Representation of the encoding expectation.
  loom_input_encoding_kind_t attachment;
  // Zeroed padding for canonical construction-time identity.
  uint8_t reserved;
  // Static encoding identity or zero-based input ordinal.
  uint64_t encoding_ordinal;
  // Literal extents or one-based input ordinals; dynamic zero is unbound.
  uint64_t dimensions[];
} loom_input_shape_t;

// Emits requirements in source argument/axis order, omitting requirements
// identical to |inherited|. A NULL output counts the exact required storage.
uint32_t loom_input_shape_requirements(const loom_input_shape_t* shape,
                                       const loom_input_shape_t* inherited,
                                       uint16_t argument,
                                       loom_input_requirement_t* output);

// Projects verified native signature inputs into the same source coordinates
// used by indexed libraries. |inherited_arguments| is the family's argument
// tuple, or NULL when constructing the family's own obligations. Both tuples
// contain |argument_count| values. Only top-level shaped inputs participate;
// structural admission owns kind, element, rank, and alignment compatibility.
// A NULL output counts the exact required storage. No IR is constructed.
uint32_t loom_input_contract_project(const loom_module_t* module,
                                     const loom_value_id_t* arguments,
                                     uint16_t argument_count,
                                     const loom_value_id_t* inherited_arguments,
                                     loom_input_requirement_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_IR_INPUT_CONTRACT_H_
