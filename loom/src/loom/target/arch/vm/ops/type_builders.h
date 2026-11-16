// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// clang-format off
#ifndef LOOM_TARGET_ARCH_VM_OPS_TYPE_BUILDERS_H_
#define LOOM_TARGET_ARCH_VM_OPS_TYPE_BUILDERS_H_

#include "iree/base/api.h"
#include "loom/ir/attribute.h"
#include "loom/ir/ir.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_module_t loom_module_t;

// Exact-build family keys; serialized types use their public names.
enum {
  LOOM_PARAMETERIZED_TYPE_VM_SPAN = LOOM_PARAMETERIZED_TYPE_KIND(LOOM_DIALECT_VM, 0),
};

static inline bool loom_vm_span_type_isa(loom_type_t type) {
  return loom_type_is_parameterized(type) && loom_type_parameterized_kind(type) == LOOM_PARAMETERIZED_TYPE_VM_SPAN;
}
enum { LOOM_VM_SPAN_TYPE_ELEMENT_TYPE_PARAMETER_INDEX = 0 };
static inline loom_type_id_t loom_vm_span_type_element_type(loom_type_t type) {
  return loom_attr_as_type_id(loom_type_parameterized_parameters(type)[LOOM_VM_SPAN_TYPE_ELEMENT_TYPE_PARAMETER_INDEX]);
}
iree_status_t loom_vm_span_type_make(
    loom_module_t* module,
    loom_type_id_t element_type,
    loom_type_t* out_type);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_ARCH_VM_OPS_TYPE_BUILDERS_H_
