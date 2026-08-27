// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Physical storage composition helpers for encoding values.
//
// `#encoding.storage` composes an address layout with an encoded storage
// schema in the existing shaped-type encoding attachment slot:
//
//   %storage = encoding.define #encoding.storage {
//     layout = %layout : encoding<layout>,
//     schema = %schema : encoding<schema>
//   } : encoding<storage>
//
// Memory operations still see physical storage. These helpers recover the
// address-layout part for address arithmetic without making loads/stores decode
// schema bytes into logical elements.

#ifndef LOOM_OPS_ENCODING_STORAGE_H_
#define LOOM_OPS_ENCODING_STORAGE_H_

#include "loom/ir/encoding.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_fact_context_t loom_fact_context_t;
typedef struct loom_value_facts_t loom_value_facts_t;
typedef struct loom_value_fact_address_layout_t
    loom_value_fact_address_layout_t;
typedef struct loom_value_fact_storage_schema_t
    loom_value_fact_storage_schema_t;

// Authored operands that materialize one explicit strided address layout.
// The slices borrow storage from the defining encoding.layout.strided op and
// remain valid for the lifetime of the module.
typedef struct loom_encoding_address_layout_operands_t {
  // Full-rank element strides with INT64_MIN sentinels for dynamic operands.
  loom_attribute_t static_strides;
  // Dynamic element-stride SSA values in sentinel order. Borrowed from the
  // defining operation.
  const loom_value_id_t* dynamic_stride_values;
  // Number of entries in dynamic_stride_values.
  uint16_t dynamic_stride_count;
} loom_encoding_address_layout_operands_t;

// Registers the storage-composition family with |context|. Built-in
// context setup calls this through the encoding family registry.
iree_status_t loom_encoding_register_storage_family(loom_context_t* context);

// Maximum static layout rank decoded into caller-provided stride storage.
// Shaped type ranks are packed in four header bits, so no well-formed consumer
// can use more than 15 layout strides.
#define LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK 15

// Decodes a static address-layout encoding into a summary. Strided layouts
// write per-axis facts into caller-owned |stride_storage| and point
// |out_layout->strides| at that storage. Returns false when |encoding_id| is
// not a known address-layout encoding, recursion exceeds the safety bound, or
// the caller did not provide enough stride storage.
bool loom_encoding_query_static_address_layout(
    const loom_module_t* module, uint16_t encoding_id,
    loom_value_facts_t* stride_storage, iree_host_size_t stride_capacity,
    loom_value_fact_address_layout_t* out_layout);

// Decodes a static storage-schema encoding into a summary. Generic operand
// schemas fill in packed fragment and scale facts; other schema
// families still return true with only static_spec_encoding_id populated.
// Returns false when |encoding_id| is not a known storage-schema encoding or
// recursion exceeds the safety bound.
bool loom_encoding_query_static_storage_schema(
    const loom_module_t* module, uint16_t encoding_id,
    loom_value_fact_storage_schema_t* out_schema);

// Queries exact family-wide physical record geometry. Returns false when the
// encoding is invalid, unregistered, or parameterized per instance.
bool loom_encoding_query_static_record_geometry(
    const loom_module_t* module, uint16_t encoding_id,
    loom_encoding_record_geometry_t* out_geometry);

// Queries a shaped type's address-layout summary. An absent attachment on a
// tile, tensor, or view is the native dense layout. Explicit static encodings
// resolve through the module, while SSA encodings use context-owned facts. This
// does not walk call graphs or inspect callers; block-argument encodings only
// resolve when a previous analysis has seeded facts for them in |context|.
// Strided static layouts use caller-owned |stride_storage| with the same
// lifetime rules as loom_encoding_query_static_address_layout().
bool loom_encoding_query_type_address_layout(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_type_t type, loom_value_facts_t* stride_storage,
    iree_host_size_t stride_capacity,
    loom_value_fact_address_layout_t* out_layout);

// Queries the authored operands that materialize a shaped type's explicit
// strided address layout. This follows verified encoding refinement and
// storage-composition values but does not recover provenance from facts or
// inspect callers. Returns false for dense/static layouts and for non-exact
// fact-only layouts whose stride values do not cross the type boundary.
bool loom_encoding_query_type_address_layout_operands(
    const loom_module_t* module, loom_type_t type,
    loom_encoding_address_layout_operands_t* out_operands);

// Queries a shaped type's storage-schema summary from static encodings or
// context-owned SSA encoding facts. This mirrors
// loom_encoding_query_type_address_layout for consumers that need physical
// payload semantics instead of address arithmetic.
bool loom_encoding_query_type_storage_schema(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_type_t type, loom_value_fact_storage_schema_t* out_schema);

// Queries scalar content facts implied by an encoded storage schema. This is
// the storage-schema half of loom_encoding_query_type_storage_content_facts()
// for operations such as vector.decode that carry an explicit schema witness
// instead of a shaped memory type encoding.
bool loom_encoding_query_storage_schema_content_facts(
    const loom_value_fact_storage_schema_t* storage_schema,
    loom_scalar_type_t element_type, loom_value_facts_t* out_facts);

// Queries scalar content facts implied by a shaped type's storage schema. This
// only returns true when the schema carries a target-independent content
// contract, such as finite-only payloads, subnormal flushing, or finite-number
// FP8 formats that exclude infinities.
bool loom_encoding_query_type_storage_content_facts(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_type_t type, loom_value_facts_t* out_facts);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_ENCODING_STORAGE_H_
