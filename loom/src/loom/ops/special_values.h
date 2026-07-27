// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Typed Loom special value helpers.
//
// This centralizes small cross-dialect value inventories that transforms need
// to recognize or materialize without string-matching op names. Poison is an
// invalid observation sentinel. Empty aggregates are valid values with a
// statically zero element footprint.

#ifndef LOOM_OPS_SPECIAL_VALUES_H_
#define LOOM_OPS_SPECIAL_VALUES_H_

#include "loom/ir/facts.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |op| is one of the typed poison materialization ops.
bool loom_op_is_poison(const loom_op_t* op);

// Returns true when |value_id| is defined by a typed poison op in |module|.
bool loom_value_is_poison(const loom_module_t* module,
                          loom_value_id_t value_id);

// Builds a typed poison value for supported scalar/vector result types.
iree_status_t loom_poison_build(loom_builder_t* builder,
                                loom_type_t result_type,
                                loom_location_id_t location,
                                loom_value_id_t* out_value_id);

// Returns true when |type| currently has a supported constant materializer.
bool loom_type_has_constant_materializer(loom_type_t type);

// Returns true when |facts| can be materialized as a constant of |type|.
bool loom_value_facts_can_materialize_constant(loom_value_facts_t facts,
                                               loom_type_t type);

// Builds a typed constant value from exact facts for supported result types.
iree_status_t loom_constant_build(loom_builder_t* builder,
                                  loom_value_facts_t facts,
                                  loom_type_t result_type,
                                  loom_location_id_t location,
                                  loom_value_id_t* out_value_id);

// Returns true when |op| materializes a typed empty aggregate value.
bool loom_op_is_empty(const loom_op_t* op);

// Returns true when |type| currently has a supported empty materializer.
bool loom_type_has_empty_materializer(loom_type_t type);

// Builds a typed empty aggregate value for supported result types.
iree_status_t loom_empty_build(loom_builder_t* builder, loom_type_t result_type,
                               loom_location_id_t location,
                               loom_value_id_t* out_value_id);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_SPECIAL_VALUES_H_
