// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Scope-owned SSA materialization proofs.

#ifndef LOOM_UTIL_FACT_ORIGIN_H_
#define LOOM_UTIL_FACT_ORIGIN_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/util/fact_layout.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_value_fact_table_t loom_value_fact_table_t;

// Independently inferred materialization proof families.
enum loom_value_fact_origin_bits_t {
  LOOM_VALUE_FACT_ORIGIN_UNIFORM_ELEMENT = 1u << 0,
  LOOM_VALUE_FACT_ORIGIN_STATIC_LANE = 1u << 1,
  LOOM_VALUE_FACT_ORIGIN_EXACT_LANE = 1u << 2,
  LOOM_VALUE_FACT_ORIGIN_UNIFORM_SCALE = 1u << 3,
  LOOM_VALUE_FACT_ORIGIN_CONTEXTUAL_QUERY = 1u << 4,
  LOOM_VALUE_FACT_ORIGIN_LAYOUT = 1u << 5,
  LOOM_VALUE_FACT_ORIGIN_EXACT_SCALAR = 1u << 6,
};

// One ordinary operation's complete origin publication. Definitions replace
// prior proofs in place; completion withdraws families not inferred or
// forwarded. Only the inference owner installs this stack-owned scope.
// Callbacks publish each result family at most once through the table's
// normal definition APIs; they select a final proof before publishing it.
typedef struct loom_value_fact_origin_publication_t {
  // Module owning the result definition ordinals.
  const loom_module_t* module;
  // Operation whose results are being inferred.
  const loom_op_t* op;
  // Numeric equivalents of the operation's operands at this definition.
  // Borrowed from IR or scoped scratch for the duration of inference. These
  // may have different storage types and cannot replace IR operands.
  const loom_value_id_t* numeric_operands;
  // Reusable family-bit scratch indexed by result ordinal.
  uint8_t* published;
  // Optional shared accumulator for numeric and auxiliary changes.
  bool* changed;
} loom_value_fact_origin_publication_t;

// Begins a complete ordinary-result publication, growing scratch only when the
// operation exceeds the previous result count. Inference callbacks do not
// recursively compute other operations in the same table.
iree_status_t loom_value_fact_table_begin_origins(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op, bool* changed,
    loom_value_fact_origin_publication_t* publication);

// Ends the active publication. Successful inference withdraws unreported proof
// families; failure leaves the partially computed table to its failing owner.
void loom_value_fact_table_end_origins(loom_value_fact_table_t* table,
                                       bool complete);

// Records a result-family definition and its semantic change in an active
// publication. Direct table definitions outside inference need no publication.
void loom_value_fact_table_record_origin(loom_value_fact_table_t* table,
                                         loom_value_id_t value_id,
                                         uint8_t family, bool changed);

// Withdraws a value's proofs without losing scope membership or stable slots.
void loom_value_fact_table_clear_origins(loom_value_fact_table_t* table,
                                         loom_value_id_t value_id);

// Scalar values that materialize every element of one aggregate value.
typedef struct loom_value_fact_uniform_element_origin_t {
  // Scalar value with the aggregate element type.
  loom_value_id_t scalar_value_id;
  // Scalar value preserving the elements before an exact float extension.
  loom_value_id_t exact_scalar_value_id;
} loom_value_fact_uniform_element_origin_t;

// Static strided logical-lane origin for one aggregate value. Result lane N is
// derived from source lane source_lane_offset + N * source_lane_stride. The
// source and result may have different element types when an operation
// preserves lane provenance while changing representation.
typedef struct loom_value_fact_static_lane_origin_t {
  // Aggregate source value containing the materialized lanes.
  loom_value_id_t source_value_id;
  // First logical source lane used by result lane zero.
  uint32_t source_lane_offset;
  // Logical source lane stride between adjacent result lanes.
  uint32_t source_lane_stride;
} loom_value_fact_static_lane_origin_t;

// Static strided logical-lane origin preserving each lane's scalar value
// exactly. Result lane N has the same value as source lane source_lane_offset +
// N * source_lane_stride after conversion to the result element type. For
// floating-point values the relation preserves signed zero and infinity but
// does not promise NaN payload identity.
typedef struct loom_value_fact_exact_lane_origin_t {
  // Aggregate source value containing the exact lane values.
  loom_value_id_t source_value_id;
  // First logical source lane used by result lane zero.
  uint32_t source_lane_offset;
  // Logical source lane stride between adjacent result lanes.
  uint32_t source_lane_stride;
} loom_value_fact_exact_lane_origin_t;

// Scalar source preserving the represented numeric value exactly. Source and
// result types may differ when a conversion changes representation without
// changing any value admitted by the source facts.
typedef struct loom_value_fact_exact_scalar_origin_t {
  // Scalar value whose represented numeric value is preserved.
  loom_value_id_t source_value_id;
} loom_value_fact_exact_scalar_origin_t;

// Uniform scalar scale applied lanewise to one aggregate value. Result lane N
// is source lane N multiplied by scale_value_id.
typedef struct loom_value_fact_uniform_scale_origin_t {
  // Aggregate source value containing the unscaled lanes.
  loom_value_id_t source_value_id;
  // Scalar value uniformly multiplied into each result lane.
  loom_value_id_t scale_value_id;
} loom_value_fact_uniform_scale_origin_t;

// Structured contextual query materialized by an SSA value. The family is
// shared with the paired static applicability attribute. The optional key
// distinguishes projections within a family, such as one SPIR-V extension;
// queries with no additional identity use an absent key.
typedef struct loom_value_fact_contextual_query_origin_t {
  // Context-local parameterized attribute family identifying query semantics.
  loom_parameterized_attr_kind_t family_kind;

  // Reserved for alignment and future query-origin flags. Always zero.
  uint16_t reserved;

  // Borrowed structured projection key owned by the current module.
  loom_attribute_t key;
} loom_value_fact_contextual_query_origin_t;

static_assert(sizeof(loom_value_fact_contextual_query_origin_t) == 24,
              "contextual query origin must remain 24 bytes");

// Complete materialization publication retained across a fact-only solve.
// Inline origins are copied; layout bindings borrow immutable scope storage.
typedef struct loom_value_fact_origin_snapshot_t {
  // Scalar and exact scalar materializations of a uniform aggregate.
  loom_value_fact_uniform_element_origin_t uniform_element;
  // Source-lane provenance, independent of element representation.
  loom_value_fact_static_lane_origin_t static_lane;
  // Exact value provenance. Aggregate values use the strided lane mapping;
  // scalar values use only source_value_id. A value's stable type makes the
  // two representations mutually exclusive across one solve.
  loom_value_fact_exact_lane_origin_t exact_value;
  // Aggregate source and its uniform scalar multiplier.
  loom_value_fact_uniform_scale_origin_t uniform_scale;
  // Contextual query descriptor with its module-owned projection key.
  loom_value_fact_contextual_query_origin_t contextual_query;
  // Immutable exact layout recipe, valid until scope clear.
  loom_value_fact_layout_origin_t layout_origin;
} loom_value_fact_origin_snapshot_t;

// Captures every origin family, including absent proofs. The snapshot remains
// valid until the table scope or module is cleared; it does not retain IR.
void loom_value_fact_table_snapshot_origins(
    const loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_origin_snapshot_t* out_snapshot);

// Compares materialization semantics, including layout binding contents rather
// than their allocation addresses. Both snapshots belong to the same scope.
bool loom_value_fact_origin_snapshot_equal(
    const loom_value_fact_origin_snapshot_t* lhs,
    const loom_value_fact_origin_snapshot_t* rhs);

// Clears the populated origin scope while retaining reusable lookup storage.
void loom_value_fact_table_clear_origin_scope(loom_value_fact_table_t* table);

// Replaces all materialization proofs for one value from |source|, including
// absence. Borrowed descriptors are copied into the target scope.
iree_status_t loom_value_fact_table_clone_origins(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    loom_value_id_t value_id);

// Forwards source proofs for families not already published by inference.
// Called only within the alias result's active publication.
iree_status_t loom_value_fact_table_forward_origins(
    loom_value_fact_table_t* table, loom_value_id_t source_value_id,
    loom_value_id_t result_value_id);

// Defines the scalar SSA values that materialize every element of aggregate
// |value_id|. |scalar_value_id| has the aggregate element type, while
// |exact_scalar_value_id| may have a narrower floating-point type whose value
// is preserved exactly by extension to the aggregate element type. INVALID
// scalar IDs withdraw the proof.
iree_status_t loom_value_fact_table_define_uniform_element_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_id_t scalar_value_id, loom_value_id_t exact_scalar_value_id);

// Returns true when |value_id| has a known scalar SSA origin that materializes
// every element. The query validates that |value_id| is shaped, the origin is
// scalar, and both have matching element types.
bool loom_value_fact_table_query_uniform_element_origin(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id, loom_value_id_t* out_scalar_value_id);

// Returns the exact scalar SSA origin for every element of |value_id|. The
// origin may have a narrower floating-point type than the aggregate.
bool loom_value_fact_table_query_exact_uniform_element_origin(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id, loom_value_id_t* out_scalar_value_id);

// Defines a static strided source-lane view for aggregate |value_id|. The
// relation is a materialization proof and is validated by the query API against
// the current module value types and static lane counts. An INVALID source
// withdraws the proof.
iree_status_t loom_value_fact_table_define_static_lane_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_static_lane_origin_t origin);

// Returns true when |value_id| has a known static source-lane view. The query
// validates that both values are vectors with static lane counts and an
// in-bounds strided source lane mapping. Element types may differ when the
// defining operation preserves lane provenance while changing representation.
bool loom_value_fact_table_query_static_lane_origin(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id, loom_value_fact_static_lane_origin_t* out_origin);

// Defines an exact strided source-lane value origin for aggregate |value_id|.
// The relation is validated by the query API against the current module value
// types and static lane counts. An INVALID source withdraws the proof.
iree_status_t loom_value_fact_table_define_exact_lane_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_exact_lane_origin_t origin);

// Returns true when |value_id| has a known exact source-lane value origin. The
// query validates that both values are vectors with static lane counts and an
// in-bounds strided source lane mapping. Element types may differ because an
// exact floating-point extension preserves the represented value.
bool loom_value_fact_table_query_exact_lane_origin(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id, loom_value_fact_exact_lane_origin_t* out_origin);

// Defines the exact scalar value source for |value_id|. The producer must have
// proved that every value admitted by its current source facts survives the
// complete conversion unchanged. An INVALID source withdraws the proof.
iree_status_t loom_value_fact_table_define_exact_scalar_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_exact_scalar_origin_t origin);

// Publishes a proved numeric identity across scalar representations. Retains
// the canonical exact origin and inherits source-local integer relations.
// INVALID source withdraws both products when a conversion can change value.
iree_status_t loom_value_fact_table_define_numeric_identity(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_id_t source_value_id);

// Returns the exact scalar value source for |value_id|. The query validates
// that both values are scalar; their representation types may differ.
bool loom_value_fact_table_query_exact_scalar_origin(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id,
    loom_value_fact_exact_scalar_origin_t* out_origin);

// Defines a value as the lanewise multiplication of |origin.source_value_id|
// and scalar |origin.scale_value_id|. The relation is a materialization proof
// and is validated by the query API against the current module value types.
// INVALID source and scale IDs withdraw the proof.
iree_status_t loom_value_fact_table_define_uniform_scale_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_uniform_scale_origin_t origin);

// Returns true when |value_id| has a known uniform scalar-scale origin. The
// query validates that the result/source are same-typed vectors and the scale
// is a same-element scalar.
bool loom_value_fact_table_query_uniform_scale_origin(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id,
    loom_value_fact_uniform_scale_origin_t* out_origin);

// Defines |value_id| as the SSA materialization of one contextual query. The
// family is the descriptor-backed static applicability family with the same
// public dotted name. The optional structured key distinguishes projections
// within that family without flattening them into strings or feature bits.
// A family of ANY withdraws the proof. Each value owns its descriptor slot,
// so updating a producer leaves prior alias publications unchanged.
iree_status_t loom_value_fact_table_define_contextual_query_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_contextual_query_origin_t origin);

// Returns true when scalar |value_id| materializes a contextual query. The
// returned origin borrows its structured key from the current module and is
// valid until the fact-table scope is cleared.
bool loom_value_fact_table_query_contextual_query_origin(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id,
    loom_value_fact_contextual_query_origin_t* out_origin);

// Returns the sparse value index for retained contextual query origins. The
// borrowed values are unique, appear in first-definition order, and remain
// valid until the fact-table scope is cleared. Withdrawn values retain slots;
// the query API distinguishes those slots from live proofs.
void loom_value_fact_table_contextual_query_values(
    const loom_value_fact_table_t* table, const loom_value_id_t** out_value_ids,
    iree_host_size_t* out_value_count);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_FACT_ORIGIN_H_
