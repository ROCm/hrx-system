// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Scope-owned integer interpretations published by conversion inference.

#ifndef LOOM_UTIL_FACT_CAST_H_
#define LOOM_UTIL_FACT_CAST_H_

#include "loom/ir/integer_cast.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_value_fact_table_t loom_value_fact_table_t;

// Exact conversion functions, independent of the current numeric interval.
// Keeping both endpoints permits refinement by conditions on an intermediate
// payload or on its transitive source. Zero bit counts denote absent proofs.
typedef struct loom_value_fact_integer_cast_t {
  // Interpretation of the direct numeric input to the conversion.
  loom_integer_cast_t immediate;
  // Composition through the input's retained interpretation.
  loom_integer_cast_t composed;
} loom_value_fact_integer_cast_t;

// Returns the interpretation published for this value, or an empty proof.
// The query records an inference dependency and never inspects defining IR.
loom_value_fact_integer_cast_t loom_value_fact_table_query_integer_cast(
    const loom_value_fact_table_t* table, loom_value_id_t value);

// Visits populated interpretations in the current scope, without visiting IR
// or scanning the value-ID span. A callback may allocate an immutable
// projection but must not mutate the table being visited.
iree_status_t loom_value_fact_table_for_each_integer_cast(
    const loom_value_fact_table_t* table,
    iree_status_t (*callback)(void* user_data, loom_value_id_t value,
                              loom_value_fact_integer_cast_t cast),
    void* user_data);

// Publishes complete conversion semantics for a result. An empty proof
// withdraws the prior result without allocating or losing its reusable slot.
iree_status_t loom_value_fact_table_define_integer_cast(
    loom_value_fact_table_t* table, loom_value_id_t value,
    loom_value_fact_integer_cast_t cast);

// Publishes the interpretation of one operand as a result, composing through
// producer-owned identities and cast summaries once. Conditional queries with
// no active publication compute numeric facts without mutating this index.
iree_status_t loom_value_fact_table_define_operand_cast(
    loom_fact_context_t* context, uint16_t result_ordinal,
    uint16_t operand_ordinal, loom_integer_cast_kind_t kind, uint8_t bit_count);

// Returns a conversion of source that equals value. This can use their shared
// payload root when source retains every low bit required by value. A false
// result means no such retained relation is available, not numeric inequality.
bool loom_value_fact_table_query_integer_cast_from(
    const loom_value_fact_table_t* table, loom_value_id_t value,
    loom_value_id_t source, loom_integer_cast_t* out_cast);

// Compares complete semantic publications, including absence.
bool loom_value_fact_integer_cast_equal(loom_value_fact_integer_cast_t lhs,
                                        loom_value_fact_integer_cast_t rhs);

// Withdraws the cast family during ordinary result publication or erasure.
void loom_value_fact_table_clear_integer_cast(loom_value_fact_table_t* table,
                                              loom_value_id_t value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_UTIL_FACT_CAST_H_
