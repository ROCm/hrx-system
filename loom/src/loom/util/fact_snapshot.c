// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_snapshot.h"

#include <string.h>

#include "loom/util/fact_table.h"

void loom_value_fact_table_snapshot_auxiliary(
    const loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_auxiliary_snapshot_t* out_snapshot) {
  loom_value_fact_table_snapshot_origins(table, value_id,
                                         &out_snapshot->origins);
  out_snapshot->partition =
      loom_value_fact_table_query_vector_partition(table, value_id);
  out_snapshot->relations =
      loom_value_fact_integer_relations_snapshot(table, value_id);
  out_snapshot->identity =
      loom_value_fact_table_query_identity(table, value_id);
  out_snapshot->selectors =
      loom_value_fact_table_select_dependency_root(table, value_id);
  out_snapshot->definition_uniform_scope =
      (uint8_t)loom_value_fact_table_definition_uniform_scope(table, value_id);
}

bool loom_value_fact_auxiliary_snapshot_equal(
    const loom_value_fact_auxiliary_snapshot_t* lhs,
    const loom_value_fact_auxiliary_snapshot_t* rhs) {
  return lhs->identity == rhs->identity && lhs->selectors == rhs->selectors &&
         lhs->definition_uniform_scope == rhs->definition_uniform_scope &&
         memcmp(&lhs->relations, &rhs->relations, sizeof(lhs->relations)) ==
             0 &&
         loom_value_fact_origin_snapshot_equal(&lhs->origins, &rhs->origins) &&
         loom_value_fact_vector_partition_equal(lhs->partition, rhs->partition);
}
