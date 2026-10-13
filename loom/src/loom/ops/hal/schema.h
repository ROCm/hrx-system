// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_OPS_HAL_SCHEMA_H_
#define LOOM_OPS_HAL_SCHEMA_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable semantic method declarations. These describe source values, not
// target register classes or a native C ABI. Python emits their data; source
// verification and target projection consume the same indexed tree.
enum loom_hal_schema_kind_e {
  LOOM_HAL_SCHEMA_VALUE = 0,
  LOOM_HAL_SCHEMA_RECORD = 1,
  LOOM_HAL_SCHEMA_RANGE = 2,
  LOOM_HAL_SCHEMA_OPTIONAL = 3,
};

typedef struct loom_hal_schema_node_t {
  // Value, positional record, homogeneous range, or optional value.
  uint8_t kind;
  // Exact scalar type for scalar leaves, or NONE for all other nodes.
  loom_scalar_type_t scalar_type;
  // Immediate child count; ranges and optionals each have one schema child.
  uint16_t child_count;
  // First child ordinal in the declaration's edge table.
  uint16_t child_base;
  // Exact source type name for managed-reference leaves, otherwise empty.
  iree_string_view_t reference_type;
} loom_hal_schema_node_t;

typedef struct loom_hal_method_t {
  // Exact entry name within the HAL module, without the dialect prefix.
  iree_string_view_t name;
  // Exact semantics version; no compatible-version matching is performed.
  uint32_t version;
  // Root record schema describing all source arguments in field order.
  uint16_t arguments;
} loom_hal_method_t;

// Generated immutable nodes and edges, shared by all declared HAL methods.
extern const loom_hal_schema_node_t loom_hal_schema_nodes[];
extern const uint16_t loom_hal_schema_children[];
// Indexed by the HAL dialect operation ordinal.
extern const loom_hal_method_t loom_hal_methods[];

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_HAL_SCHEMA_H_
