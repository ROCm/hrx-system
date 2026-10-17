// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Selected native types and emitted SSA bindings for source values.
//
// A source ordinal owns one four-byte slot. A producer publishes its selected
// type before emission. Structural values can inherit that type through a
// flattened producer ordinal recorded by definition-ordered planning. Emission
// replaces each slot with its own Low value. The state bits share the source
// plan's per-value flags.
// Consumers never recover a producer's choice from a preferred type mapping.

#ifndef LOOM_CODEGEN_LOW_LOWER_BINDINGS_H_
#define LOOM_CODEGEN_LOW_LOWER_BINDINGS_H_

#include "iree/base/api.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_context_t loom_low_lower_context_t;
typedef struct loom_op_t loom_op_t;

typedef union loom_low_lower_value_binding_t {
  // Canonical selected type, or INVALID before the producer publishes one.
  loom_type_id_t type;
  // Flattened structural source of an inherited native carrier.
  loom_value_ordinal_t type_source;
  // Low value identity after materialization, including the elided sentinel.
  loom_value_id_t value;
} loom_low_lower_value_binding_t;

// Records carrier inheritance for every same-type tied result on |source_op|.
// Type-changing ties retain independent target-selected carriers.
void loom_low_lower_plan_tied_result_types(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op);

// Publishes an already interned native type selected by a source producer.
void loom_low_lower_plan_value_type_id(loom_low_lower_context_t* context,
                                       loom_value_id_t source_value_id,
                                       loom_type_id_t type_id);

// Publishes the exact native carrier selected by a source value's producer.
// Canonical type interning is the only failure. This records a producer
// decision; it does not choose a preferred representation for another value.
iree_status_t loom_low_lower_plan_value_type(loom_low_lower_context_t* context,
                                             loom_value_id_t source_value_id,
                                             loom_type_t type);

// Records structural carrier inheritance during definition-ordered planning.
// Existing inherited sources are flattened here; the root producer publishes a
// concrete type before consumers query it. Each value keeps its own SSA
// binding.
void loom_low_lower_inherit_value_type(loom_low_lower_context_t* context,
                                       loom_value_id_t source_value_id,
                                       loom_value_id_t result_value_id);

// Returns true when definition-ordered planning has selected or inherited the
// native carrier for |source_value_id|.
bool loom_low_lower_value_binding_type_is_planned(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id);

// Returns the selected native type, or its equivalent emitted type after the
// producer has materialized. The value must have a non-elided binding.
loom_type_t loom_low_lower_value_binding_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id);

// Returns the Low SSA value already bound to |source_value_id|. The producer
// must have materialized a non-elided binding before this lookup.
loom_value_id_t loom_low_lower_lookup_value(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id);

// Returns true when the source value has a non-elided emitted binding. A
// selected type alone does not count as an available SSA value.
bool loom_low_lower_source_value_has_low_mapping(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id);

// Replaces the selected type with the corresponding Low value and copies the
// source display name. Rebinding the same materialized identity is allowed.
iree_status_t loom_low_lower_bind_value(loom_low_lower_context_t* context,
                                        loom_value_id_t source_value_id,
                                        loom_value_id_t low_value_id);

// Replaces an emitted binding when entry interposition remaps its SSA identity
// or adapts a physical ABI carrier to its body representation. Subsequent
// queries observe the replacement value's actual type.
iree_status_t loom_low_lower_replace_value_binding(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    loom_value_id_t low_value_id);

// Binds an alias to the producer's already materialized Low value.
iree_status_t loom_low_lower_bind_value_alias(loom_low_lower_context_t* context,
                                              loom_value_id_t source_value_id,
                                              loom_value_id_t result_value_id);

// Marks a source value as erased by its selected plan. Such values represent
// source sequencing or control and cannot be consumed as Low operands.
void loom_low_lower_elide_value(loom_low_lower_context_t* context,
                                loom_value_id_t source_value_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_BINDINGS_H_
