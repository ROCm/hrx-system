// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/storage_identity.h"

#include "loom/codegen/low/storage_relation.h"

static loom_value_ordinal_t loom_low_storage_identity_origin(
    loom_value_ordinal_t* origins, loom_value_ordinal_t ordinal) {
  loom_value_ordinal_t origin = ordinal;
  while (origins[origin] != origin) {
    origin = origins[origin];
  }
  while (origins[ordinal] != origin) {
    const loom_value_ordinal_t parent = origins[ordinal];
    origins[ordinal] = origin;
    ordinal = parent;
  }
  return origin;
}

iree_status_t loom_low_storage_identity_build(
    const loom_local_value_domain_t* domain,
    iree_host_size_t additional_value_capacity, iree_arena_allocator_t* arena,
    loom_value_ordinal_t** out_origins) {
  *out_origins = NULL;
  loom_value_ordinal_t* origins = NULL;
  if (additional_value_capacity >
      LOOM_VALUE_ORDINAL_INVALID - domain->value_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function preparation exceeds value ordinal range");
  }
  const loom_value_ordinal_t capacity =
      domain->value_count + (loom_value_ordinal_t)additional_value_capacity;
  for (loom_value_ordinal_t i = 0; i < domain->definition_count; ++i) {
    const loom_value_id_t value_id = domain->value_ids[i];
    const loom_value_t* value = loom_module_value(domain->module, value_id);
    if (loom_value_is_block_arg(value)) {
      continue;
    }
    const loom_op_t* op = loom_value_def_op(value);
    // The acquired definition prefix covers every operation result, including
    // unused results and nested definitions when the domain covers the tree.
    // Result zero owns the operation's relation enumeration.
    if (loom_op_const_results(op)[0] != value_id) {
      continue;
    }
    loom_low_storage_relation_iterator_t iterator;
    loom_low_storage_relation_iterator_initialize(domain->module, op,
                                                  &iterator);
    loom_low_storage_relation_t relation;
    while (loom_low_storage_relation_iterator_next(&iterator, &relation)) {
      if (relation.cause != LOOM_LOW_STORAGE_RELATION_CAUSE_TIED_RESULT) {
        continue;
      }
      if (origins == NULL) {
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            arena, capacity, sizeof(*origins), (void**)&origins));
        for (loom_value_ordinal_t v = 0; v < capacity; ++v) {
          origins[v] = v;
        }
      }
      origins[loom_local_value_domain_ordinal(domain,
                                              relation.destination_value_id)] =
          loom_local_value_domain_ordinal(domain, relation.source_value_id);
    }
  }
  if (origins != NULL) {
    for (loom_value_ordinal_t i = 0; i < domain->value_count; ++i) {
      origins[i] = loom_low_storage_identity_origin(origins, i);
    }
  }
  *out_origins = origins;
  return iree_ok_status();
}
