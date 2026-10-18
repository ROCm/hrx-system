// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_identity.h"

#include <string.h>

#include "loom/util/fact_table.h"

static loom_value_fact_identity_t* loom_value_fact_identity_lookup(
    const loom_value_fact_table_t* table, loom_value_id_t value_id) {
  return value_id < table->identities.capacity
             ? table->identities.entries[value_id]
             : NULL;
}

static iree_status_t loom_value_fact_identity_ensure_capacity(
    loom_value_fact_table_t* table, loom_value_id_t value_id) {
  const iree_host_size_t old_capacity = table->identities.capacity;
  if (value_id < old_capacity) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      table->arena, old_capacity, (iree_host_size_t)value_id + 1,
      sizeof(*table->identities.entries), &table->identities.capacity,
      (void**)&table->identities.entries));
  memset(table->identities.entries + old_capacity, 0,
         (table->identities.capacity - old_capacity) *
             sizeof(*table->identities.entries));
  return iree_ok_status();
}

static iree_status_t loom_value_fact_identity_allocate(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_identity_t** out_identity) {
  IREE_RETURN_IF_ERROR(
      loom_value_fact_identity_ensure_capacity(table, value_id));
  loom_value_fact_identity_t* identity = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      table->transient_arena, sizeof(*identity), (void**)&identity));
  *identity = (loom_value_fact_identity_t){
      .next = table->identities.nodes,
      .representative = value_id,
  };
  table->identities.nodes = identity;
  table->identities.entries[value_id] = identity;
  *out_identity = identity;
  return iree_ok_status();
}

// Only roots unite. Choosing by rank bounds every query independently of the
// order in which a rewrite replaces representatives.
static void loom_value_fact_identity_unite(loom_value_fact_identity_t* lhs,
                                           loom_value_fact_identity_t* rhs,
                                           loom_value_id_t representative) {
  if (lhs != rhs) {
    if (lhs->rank < rhs->rank) {
      lhs->parent = rhs;
      lhs = rhs;
    } else {
      rhs->parent = lhs;
      if (lhs->rank == rhs->rank) {
        ++lhs->rank;
      }
    }
  }
  lhs->representative = representative;
}

iree_status_t loom_value_fact_table_define_identity(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_id_t source_value_id, bool* inout_changed) {
  if (source_value_id == value_id) {
    const bool changed =
        loom_value_fact_table_query_identity(table, value_id) != value_id;
    loom_value_fact_table_undefine_identity(table, value_id);
    if (changed && inout_changed) {
      *inout_changed = true;
    }
    return iree_ok_status();
  }
  const loom_value_id_t canonical =
      loom_value_fact_table_query_identity(table, source_value_id);
  if (canonical == value_id) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_value_fact_identity_ensure_capacity(table, value_id));
  loom_value_fact_identity_t* source =
      loom_value_fact_identity_lookup(table, canonical);
  if (!source || loom_value_fact_identity_root(source)->representative ==
                     LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_value_fact_identity_allocate(table, canonical, &source));
  }
  loom_value_fact_identity_t* previous = table->identities.entries[value_id];
  // Compare publication tokens, not their current union roots. Replacement
  // changes the representative immediately, but deferred inference still needs
  // to report that an old publication has followed the replacement source.
  if (previous == source) {
    return iree_ok_status();
  }
  if (previous) {
    loom_value_fact_identity_t* root = loom_value_fact_identity_root(previous);
    if (root->representative == value_id) {
      loom_value_fact_identity_unite(
          root, loom_value_fact_identity_root(source), canonical);
    }
  }
  table->identities.entries[value_id] = source;
  if (inout_changed) {
    *inout_changed = true;
  }
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_replace_identity_value(
    loom_value_fact_table_t* table, loom_value_id_t old_value,
    loom_value_id_t new_value) {
  loom_value_fact_identity_t* previous =
      loom_value_fact_identity_lookup(table, old_value);
  if (!previous || old_value == new_value) {
    return iree_ok_status();
  }
  loom_value_fact_identity_t* root = loom_value_fact_identity_root(previous);
  if (root->representative != old_value) {
    return iree_ok_status();
  }
  loom_value_id_t canonical =
      loom_value_fact_table_query_identity(table, new_value);
  loom_value_fact_identity_t* replacement =
      loom_value_fact_identity_lookup(table, canonical);
  if (canonical == old_value || !replacement ||
      loom_value_fact_identity_root(replacement)->representative ==
          LOOM_VALUE_ID_INVALID) {
    canonical = new_value;
    IREE_RETURN_IF_ERROR(
        loom_value_fact_identity_allocate(table, canonical, &replacement));
  }
  loom_value_fact_identity_unite(
      root, loom_value_fact_identity_root(replacement), canonical);
  // Replacing uses does not redefine or erase the old SSA value. Existing
  // aliases keep the redirected token; subsequently created aliases of the old
  // value acquire a fresh token describing its still-independent definition.
  table->identities.entries[old_value] = NULL;
  return iree_ok_status();
}

void loom_value_fact_table_erase_identity_value(loom_value_fact_table_t* table,
                                                loom_value_id_t value_id) {
  loom_value_fact_identity_t* identity =
      loom_value_fact_identity_lookup(table, value_id);
  if (!identity) {
    return;
  }
  loom_value_fact_identity_t* root = loom_value_fact_identity_root(identity);
  if (root->representative == value_id) {
    root->representative = LOOM_VALUE_ID_INVALID;
  }
  table->identities.entries[value_id] = NULL;
}

void loom_value_fact_table_undefine_identity(loom_value_fact_table_t* table,
                                             loom_value_id_t value_id) {
  loom_value_fact_identity_t* identity =
      loom_value_fact_identity_lookup(table, value_id);
  if (identity &&
      loom_value_fact_identity_root(identity)->representative != value_id) {
    table->identities.entries[value_id] = NULL;
  }
}

void loom_value_fact_table_clear_identities(loom_value_fact_table_t* table) {
  if (!table->identities.capacity) {
    return;
  }
  for (iree_host_size_t i = 0; i < table->touched_count; ++i) {
    const loom_value_id_t value_id = table->touched_values[i];
    if (value_id < table->identities.capacity) {
      table->identities.entries[value_id] = NULL;
    }
  }
  for (loom_value_fact_identity_t* node = table->identities.nodes; node;
       node = node->next) {
    // Sources need not have numeric entries in this scope. Non-root nodes keep
    // their prior representative IDs, covering detached publications as well.
    if (node->representative != LOOM_VALUE_ID_INVALID) {
      table->identities.entries[node->representative] = NULL;
    }
  }
  table->identities.nodes = NULL;
}
