// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Descriptor-backed parameterized attribute family metadata.

#ifndef LOOM_IR_PARAMETERIZED_ATTR_H_
#define LOOM_IR_PARAMETERIZED_ATTR_H_

#include "loom/ir/attribute_schema.h"
#include "loom/util/bstring.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  // Sentinel indicating that a family has no compact positional parameter.
  LOOM_PARAMETERIZED_ATTR_NO_PRIMARY_PARAMETER = UINT8_MAX,
};

typedef struct loom_target_condition_descriptor_t
    loom_target_condition_descriptor_t;

// Generated metadata for one descriptor-backed parameterized attribute family.
typedef struct loom_parameterized_attr_descriptor_t {
  // Stable dotted public family name.
  loom_bstring_t name;
  // Dense context-local family identity.
  loom_parameterized_attr_kind_t kind;
  // Number of descriptor-indexed parameter slots.
  uint8_t parameter_count;
  // Required parameter printed first without its name in text assembly, or
  // LOOM_PARAMETERIZED_ATTR_NO_PRIMARY_PARAMETER when all parameters are named.
  uint8_t primary_parameter_index;
  // Parameter descriptors in stable declaration order.
  const loom_attr_descriptor_t* parameter_descriptors;
  // Optional typed target-applicability semantics for this family.
  const loom_target_condition_descriptor_t* target_condition;
} loom_parameterized_attr_descriptor_t;

static_assert(sizeof(loom_parameterized_attr_descriptor_t) == 32,
              "parameterized attribute descriptor must remain 32 bytes");

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_IR_PARAMETERIZED_ATTR_H_
