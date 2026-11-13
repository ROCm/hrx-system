// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbolic_scope.h"

#include "loom/ir/structural_hash.h"

typedef struct loom_symbolic_scope_key_t {
  // Scope containing the indexed source identities.
  const loom_symbolic_scope_t* scope;
  // Source identity being queried or inserted.
  loom_value_id_t value;
} loom_symbolic_scope_key_t;

static bool loom_symbolic_scope_key_equal(const void* user_data,
                                          uint32_t ordinal) {
  const loom_symbolic_scope_key_t* key = user_data;
  return key->scope->values[ordinal] == key->value;
}

static uint32_t loom_symbolic_scope_hash(loom_value_id_t value) {
  return loom_structural_hash_finalize(
      loom_structural_hash_mix_u32(loom_structural_hash_initialize(), value));
}

loom_value_ordinal_t loom_symbolic_scope_find(
    const loom_symbolic_scope_t* scope, loom_value_id_t value) {
  const loom_symbolic_scope_key_t key = {.scope = scope, .value = value};
  return loom_intern_table_probe(&scope->index, loom_symbolic_scope_hash(value),
                                 loom_symbolic_scope_key_equal, &key)
      .index;
}

iree_status_t loom_symbolic_scope_register(loom_symbolic_scope_t* scope,
                                           iree_arena_allocator_t* arena,
                                           loom_value_id_t value,
                                           loom_value_ordinal_t* out_ordinal) {
  const loom_symbolic_scope_key_t key = {.scope = scope, .value = value};
  const uint32_t hash = loom_symbolic_scope_hash(value);
  loom_intern_probe_t probe = loom_intern_table_probe(
      &scope->index, hash, loom_symbolic_scope_key_equal, &key);
  if (probe.index != UINT32_MAX) {
    *out_ordinal = probe.index;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_intern_table_reserve_insert(arena, &scope->index,
                                                        hash, 1, &probe.slot));
  if (scope->index.count == scope->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, scope->index.count, scope->index.count + 1,
        sizeof(*scope->values), &scope->capacity, (void**)&scope->values));
  }
  const loom_value_ordinal_t ordinal = (loom_value_ordinal_t)scope->index.count;
  scope->values[ordinal] = value;
  loom_intern_table_insert(&scope->index, probe.slot, hash, ordinal);
  *out_ordinal = ordinal;
  return iree_ok_status();
}
