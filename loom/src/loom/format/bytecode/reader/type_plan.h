// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Invocation-owned type-table facts established by bytecode validation.

#ifndef LOOM_FORMAT_BYTECODE_READER_TYPE_PLAN_H_
#define LOOM_FORMAT_BYTECODE_READER_TYPE_PLAN_H_

#include "iree/base/api.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Parent metadata and child arity for a function, dialect, or typed register.
// Canonical construction consumes retained child IDs without a scratch payload.
typedef struct loom_bytecode_structural_type_plan_t {
  // Dialect parameter count stored in loom_type_t::encoding_flags, else zero.
  uint16_t parameter_count;
  // Validated STRINGS family name ID for dialect types, else zero.
  loom_string_id_t name_id;
  // Number of trailing children, up to 2 * UINT16_MAX for function types.
  uint32_t dependency_count;
} loom_bytecode_structural_type_plan_t;

// One dense validated type-table entry. The sparse fact selects the union arm.
typedef struct loom_bytecode_type_plan_entry_t {
  union {
    // Complete by-value type when the entry has no sparse fact.
    loom_type_t direct_type;
    // Native layout for a structural fact; unused for parameterized facts.
    loom_bytecode_structural_type_plan_t structural;
    // Canonical output identity after ordered full construction consumes the
    // entry. Later entries and body references use this completion directly.
    loom_type_id_t completed_type;
  };
  // Absolute bytecode offset of the entry kind.
  uint64_t bytecode_offset;
} loom_bytecode_type_plan_entry_t;

static_assert(sizeof(loom_bytecode_type_plan_entry_t) == 32,
              "type plans retain one 32-byte entry per wire type");

// Common header for one sparse fact in topological type-table order.
typedef struct loom_bytecode_type_fact_t {
  // Next sparse fact in increasing type ID order.
  struct loom_bytecode_type_fact_t* next;
  // Dense type-table entry described by this fact.
  loom_type_id_t type_id;
  // Non-direct loom_type_kind_t discriminator.
  loom_type_kind_t kind;
} loom_bytecode_type_fact_t;

// Resolved payload identity with a source-local declaration name. SYMBOLS
// validation proves agreement before either reader interns the type.
typedef struct loom_bytecode_reference_type_fact_t {
  // Common sparse type-fact header.
  loom_bytecode_type_fact_t base;
  // STRINGS identity of the external provider namespace.
  loom_string_id_t namespace_id;
  // STRINGS identity of the exact payload name within the provider namespace.
  loom_string_id_t name_id;
  // STRINGS identity of the module-local declaration name.
  loom_string_id_t declaration_name_id;
  // Declared owning or borrowed handle mode.
  loom_reference_ownership_t ownership;
} loom_bytecode_reference_type_fact_t;

// Native prefix and type-reference facts for a structural payload. The prefix
// contains the function signature header, the two register carrier words, or
// zeros for dialect types. Canonical construction copies the parent metadata
// and fills children from the retained IDs.
typedef struct loom_bytecode_structural_type_fact_t {
  // Common sparse type-fact header.
  loom_bytecode_type_fact_t base;
  // Native payload prefix, zero-filled beyond the kind's fixed header.
  uint64_t payload_prefix[2];
  // Prior child type IDs in native payload order. Full materialization consumes
  // these slots, replacing source ordinals with completed native IDs.
  loom_type_id_t type_ids[];
} loom_bytecode_structural_type_fact_t;

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

// Invocation-owned validated static type construction plan.
typedef struct loom_bytecode_type_plan_t {
  // Dense parent metadata, or NULL when the consumer retains only entry ranges.
  loom_bytecode_type_plan_entry_t* entries;
  // Number of validated source types.
  iree_host_size_t count;
  // Sparse non-direct facts in increasing source type order.
  loom_bytecode_type_fact_t* facts;
} loom_bytecode_type_plan_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_TYPE_PLAN_H_
