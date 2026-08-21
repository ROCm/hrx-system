// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable type-table facts established by bytecode validation.

#ifndef LOOM_FORMAT_BYTECODE_READER_TYPE_PLAN_H_
#define LOOM_FORMAT_BYTECODE_READER_TYPE_PLAN_H_

#include "iree/base/api.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// One dense validated type-table entry.
typedef struct loom_bytecode_type_plan_entry_t {
  // Complete by-value type when the entry has no sparse fact.
  loom_type_t direct_type;
  // Absolute bytecode offset of the entry kind.
  uint64_t bytecode_offset;
} loom_bytecode_type_plan_entry_t;

// Common header for one sparse fact in topological type-table order.
typedef struct loom_bytecode_type_fact_t {
  // Next sparse fact in increasing type ID order.
  struct loom_bytecode_type_fact_t* next;
  // Dense type-table entry described by this fact.
  loom_type_id_t type_id;
  // Non-direct loom_type_kind_t discriminator.
  loom_type_kind_t kind;
} loom_bytecode_type_fact_t;

// Type-reference facts for one function type.
typedef struct loom_bytecode_function_type_fact_t {
  // Common sparse type-fact header.
  loom_bytecode_type_fact_t base;
  // Number of leading argument type IDs.
  uint16_t argument_count;
  // Number of trailing result type IDs.
  uint16_t result_count;
  // Argument then result type IDs in wire order.
  loom_type_id_t type_ids[];
} loom_bytecode_function_type_fact_t;

// Type-reference facts for one dialect type.
typedef struct loom_bytecode_dialect_type_fact_t {
  // Common sparse type-fact header.
  loom_bytecode_type_fact_t base;
  // Validated STRINGS family name ID.
  loom_string_id_t name_id;
  // Number of parameter type IDs.
  uint16_t parameter_count;
  // Parameter type IDs in wire order.
  loom_type_id_t type_ids[];
} loom_bytecode_dialect_type_fact_t;

// Materialization facts for one descriptor-backed type.
typedef struct loom_bytecode_parameterized_type_fact_t {
  // Common sparse type-fact header.
  loom_bytecode_type_fact_t base;
  // Resolved static family descriptor.
  const loom_parameterized_type_descriptor_t* descriptor;
  // Absolute offset of the first present parameter name.
  uint64_t parameters_offset;
  // Exact byte length of all present parameter names, kinds, and values.
  iree_host_size_t parameters_length;
  // Number of present parameters in the exact payload.
  uint8_t present_count;
  // Validated descriptor indices for present parameters in wire order.
  uint8_t parameter_indices[];
} loom_bytecode_parameterized_type_fact_t;

// Materialization facts for one typed target-register payload.
typedef struct loom_bytecode_typed_register_fact_t {
  // Common sparse type-fact header.
  loom_bytecode_type_fact_t base;
  // First target-owned carrier payload word.
  uint64_t carrier_payload0;
  // Second target-owned carrier payload word.
  uint64_t carrier_payload1;
  // Prior semantic value type ID.
  loom_type_id_t value_type_id;
} loom_bytecode_typed_register_fact_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_TYPE_PLAN_H_
