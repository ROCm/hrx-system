// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Descriptor-backed generic type family metadata.

#ifndef LOOM_IR_PARAMETERIZED_TYPE_H_
#define LOOM_IR_PARAMETERIZED_TYPE_H_

#include "loom/ir/attribute_schema.h"

#ifdef __cplusplus
extern "C" {
#endif

enum loom_parameterized_type_flag_bits_e {
  // A type whose optional parameters are all absent omits the entire <...>
  // parameter list in canonical text.
  LOOM_PARAMETERIZED_TYPE_OMIT_EMPTY_PARAMETER_LIST = 1u << 0,
};
typedef uint8_t loom_parameterized_type_flags_t;

// Generated metadata for one descriptor-backed type family.
typedef struct loom_parameterized_type_descriptor_t {
  // Immutable owning type descriptor, including its name and text format. The
  // owner and parameter schema share a lifetime and refer to each other.
  const struct loom_type_descriptor_t* type_descriptor;

  // Parameter descriptors in stable declaration order.
  const loom_attr_descriptor_t* parameter_descriptors;

  // Type-owned facts used by joins and transport of this parameterized family.
  const struct loom_value_fact_domain_t* fact_domain;

  // Runtime type kind carrying the parameters. LOOM_TYPE_PARAMETERIZED uses
  // indirect immutable slots; compact kinds use their own representation.
  loom_type_kind_t ir_kind;

  // loom_type_flags_t bits used by a compact inline-enum representation.
  uint8_t type_flags;

  // Number of descriptor-indexed parameter slots.
  uint8_t parameter_count;

  // Text and representation behavior bits.
  loom_parameterized_type_flags_t flags;

  // Exact-build generated indirect family identity, or NONE for compact
  // representations and handwritten descriptors without a generated family.
  loom_parameterized_type_kind_t kind;
  // Logical handle mode for LOOM_TYPE_REFERENCE, otherwise zero.
  loom_reference_ownership_t reference_ownership;
} loom_parameterized_type_descriptor_t;

static_assert(sizeof(loom_parameterized_type_descriptor_t) ==
                  (IREE_PTR_SIZE == 8 ? 32 : 20),
              "parameterized type descriptor must remain compact");

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_IR_PARAMETERIZED_TYPE_H_
