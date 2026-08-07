// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Descriptor-backed generic type family metadata.

#ifndef LOOM_IR_PARAMETERIZED_TYPE_H_
#define LOOM_IR_PARAMETERIZED_TYPE_H_

#include "loom/ir/attribute_schema.h"
#include "loom/util/bstring.h"

#ifdef __cplusplus
extern "C" {
#endif

// Generated metadata for one descriptor-backed generic type family.
typedef struct loom_parameterized_type_descriptor_t {
  // Stable dotted public family name.
  loom_bstring_t name;
  // Number of descriptor-indexed parameter slots.
  uint8_t parameter_count;
  // Parameter descriptors in stable declaration order.
  const loom_attr_descriptor_t* parameter_descriptors;
} loom_parameterized_type_descriptor_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_IR_PARAMETERIZED_TYPE_H_
