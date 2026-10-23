// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cleanup special-value policy.
//
// Compiler compositions provide the dialect-backed classification and builders
// used by universal poison, empty-value, and constant cleanup. The generic
// canonicalizer owns propagation semantics while composition selects the
// concrete materializers.

#ifndef LOOM_TRANSFORMS_CLEANUP_SPECIAL_VALUE_POLICY_H_
#define LOOM_TRANSFORMS_CLEANUP_SPECIAL_VALUE_POLICY_H_

#include "iree/base/api.h"
#include "loom/ir/facts.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_cleanup_special_value_policy_t {
  // Returns true when |type| has a poison materializer.
  bool (*type_has_poison_materializer)(loom_type_t type);
  // Builds a poison value of |result_type|.
  iree_status_t (*materialize_poison)(loom_builder_t* builder,
                                      loom_type_t result_type,
                                      loom_location_id_t location,
                                      loom_value_id_t* out_value_id);
  // Returns true when |op| is an empty-value materializer.
  bool (*op_is_empty)(const loom_op_t* op);
  // Returns true when |type| has an empty-value materializer.
  bool (*type_has_empty_materializer)(loom_type_t type);
  // Builds an empty value of |result_type|.
  iree_status_t (*materialize_empty)(loom_builder_t* builder,
                                     loom_type_t result_type,
                                     loom_location_id_t location,
                                     loom_value_id_t* out_value_id);
  // Returns true when exact |facts| can be materialized as |result_type|.
  bool (*can_materialize_constant)(loom_value_facts_t facts,
                                   loom_type_t result_type);
  // Builds a constant of |result_type| from exact |facts|.
  iree_status_t (*materialize_constant)(loom_builder_t* builder,
                                        loom_value_facts_t facts,
                                        loom_type_t result_type,
                                        loom_location_id_t location,
                                        loom_value_id_t* out_value_id);
} loom_cleanup_special_value_policy_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_CLEANUP_SPECIAL_VALUE_POLICY_H_
