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
#include "loom/util/fact_layout.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_fact_context_t loom_fact_context_t;
typedef struct loom_builder_t loom_builder_t;
typedef struct loom_value_facts_t loom_value_facts_t;
typedef struct loom_value_fact_address_layout_t
    loom_value_fact_address_layout_t;
typedef struct loom_value_fact_encoding_summary_t
    loom_value_fact_encoding_summary_t;
typedef struct loom_value_fact_storage_schema_t
    loom_value_fact_storage_schema_t;

// Registers the storage-composition family with |context|. Built-in
// context setup calls this through the encoding family registry.
iree_status_t loom_encoding_register_storage_family(loom_context_t* context);

// Interns a canonical static encoding equivalent to |summary| when its
// semantics can be represented without losing information. Exact named specs
// reuse their existing module ID, including nested layout specifications.
// Known built-in dense/strided families and physical-storage compositions are
// reconstructed from complete facts. Address geometry alone does not establish
// a family identity. Other summaries leave |out_encoding_id| zero.
//
// This mutates the module encoding table and is intended for specialization
// transforms that need a durable static type attachment after the defining SSA
// encoding values disappear.
iree_status_t loom_encoding_intern_exact_summary(
    loom_module_t* module, const loom_value_fact_encoding_summary_t* summary,
    uint16_t* out_encoding_id);

// Builds physical storage from a dynamic address layout and an exact static
// storage schema. The storage-family spec and parameter names are interned in
// the builder's module; only |layout| remains an SSA parameter.
iree_status_t loom_encoding_storage_build(loom_builder_t* builder,
                                          loom_value_id_t layout,
                                          uint16_t schema_encoding_id,
                                          loom_type_t result_type,
                                          loom_location_id_t location,
                                          loom_op_t** out_op);

// Maximum static layout rank decoded into caller-provided stride storage.
// Shaped type ranks are packed in four header bits, so no well-formed consumer
// can use more than 15 layout strides.
#define LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK LOOM_TYPE_MAX_RANK

// Decodes a static address-layout encoding into a summary. Strided layouts
// write per-axis facts into caller-owned |stride_storage| and point
// |out_layout->strides| at that storage. Returns false when |encoding_id| is
// not a known address-layout encoding, recursion exceeds the safety bound, or
// the caller did not provide enough stride storage.
bool loom_encoding_query_static_address_layout(
    const loom_module_t* module, uint16_t encoding_id,
    loom_value_facts_t* stride_storage, iree_host_size_t stride_capacity,
    loom_value_fact_address_layout_t* out_layout);

// Queries an SSA encoding value's address-layout summary from context-owned
// facts. The returned stride facts borrow fact-table storage. This does not
// inspect the value's producer or recover scope-local SSA stride bindings.
bool loom_encoding_query_value_address_layout(
    const loom_fact_context_t* context, loom_value_id_t value_id,
    loom_value_fact_address_layout_t* out_layout);

// Queries the retained exact materialization recipe for an encoding value.
// Storage is owned by |context|'s fact table and survives until its scope is
// cleared. An absent recipe does not trigger producer inspection.
loom_value_fact_layout_origin_t loom_encoding_query_value_layout_origin(
    const loom_fact_context_t* context, loom_value_id_t value_id);

// Decodes a static storage-schema encoding into a summary. Generic operand
// schemas fill in packed fragment and scale facts; other schema
// families still return true with only static_spec_encoding_id populated.
// Returns false when |encoding_id| is not a known storage-schema encoding or
// recursion exceeds the safety bound.
bool loom_encoding_query_static_storage_schema(
    const loom_module_t* module, uint16_t encoding_id,
    loom_value_fact_storage_schema_t* out_schema);

// Queries the exact family-wide physical record layout. The returned generated
// tables have process lifetime. Returns false when the encoding is invalid,
// unregistered, or parameterized per instance.
bool loom_encoding_query_static_record_layout(
    const loom_module_t* module, uint16_t encoding_id,
    const loom_encoding_record_layout_t** out_layout);

// Queries exact family-wide physical record geometry. Returns false under the
// same conditions as loom_encoding_query_static_record_layout().
bool loom_encoding_query_static_record_geometry(
    const loom_module_t* module, uint16_t encoding_id,
    loom_encoding_record_geometry_t* out_geometry);

// Returns the auxiliary operand keys physically embedded in |layout|. Scale
// field hierarchy levels map to the corresponding scale key; affine minimum
// and sum-correction fields map to their shared keys. The layout must come from
// loom_encoding_query_static_record_layout().
loom_encoding_auxiliary_key_flags_t
loom_encoding_record_embedded_auxiliary_keys(
    const loom_encoding_record_layout_t* layout);

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

// Queries the retained exact materialization recipe for a shaped type's SSA
// encoding. Storage is owned by |context|'s fact table and survives until its
// scope is cleared. This does not inspect IR producers.
loom_value_fact_layout_origin_t loom_encoding_query_type_layout_origin(
    const loom_fact_context_t* context, loom_type_t type);

// Queries a shaped type's storage-schema summary from static encodings or
// context-owned SSA encoding facts. This mirrors
// loom_encoding_query_type_address_layout for consumers that need physical
// payload semantics instead of address arithmetic.
bool loom_encoding_query_type_storage_schema(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_type_t type, loom_value_fact_storage_schema_t* out_schema);

// Queries the exact fixed-record layout carried by a shaped type's resolved
// storage schema. The returned generated table has process lifetime. Returns
// false when the type has no resolved static schema or that schema is not a
// fixed-record family.
bool loom_encoding_query_type_record_layout(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_type_t type, const loom_encoding_record_layout_t** out_layout);

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
