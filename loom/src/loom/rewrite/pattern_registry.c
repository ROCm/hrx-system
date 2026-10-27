// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/pattern_registry.h"

#include <stdint.h>

static loom_rewrite_pattern_op_entry_t loom_rewrite_pattern_op_entry_empty(
    void) {
  return (loom_rewrite_pattern_op_entry_t){
      /*.pattern_start=*/UINT16_MAX,
      /*.pattern_count=*/0,
  };
}

static iree_status_t loom_rewrite_pattern_registry_count_patterns(
    loom_rewrite_pattern_provider_list_t providers, uint16_t* dialect_op_counts,
    uint8_t* out_dialect_base_id, uint16_t* out_dialect_limit,
    uint16_t* out_pattern_count) {
  *out_dialect_base_id = UINT8_MAX;
  *out_dialect_limit = 0;
  *out_pattern_count = 0;
  if (providers.count > 0 && providers.values == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "rewrite pattern provider table is required");
  }
  for (iree_host_size_t provider_index = 0; provider_index < providers.count;
       ++provider_index) {
    const loom_rewrite_pattern_provider_t* provider =
        providers.values[provider_index];
    if (provider == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "rewrite pattern provider is required");
    }
    if (provider->pattern_count > 0 && provider->patterns == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "rewrite pattern provider '%.*s' has no pattern table",
          (int)provider->name.size, provider->name.data);
    }
    const uint32_t total_pattern_count =
        (uint32_t)(*out_pattern_count) + provider->pattern_count;
    if (total_pattern_count > UINT16_MAX) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "rewrite pattern registry exceeds uint16_t capacity");
    }
    *out_pattern_count = (uint16_t)total_pattern_count;
    for (uint16_t pattern_index = 0; pattern_index < provider->pattern_count;
         ++pattern_index) {
      const loom_rewrite_pattern_t* pattern =
          &provider->patterns[pattern_index];
      const uint8_t dialect_id = loom_op_dialect_id(pattern->root_kind);
      if (pattern->root_kind == LOOM_OP_KIND_UNKNOWN ||
          dialect_id == LOOM_DIALECT_UNKNOWN ||
          dialect_id == LOOM_DIALECT_RESERVED) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "rewrite pattern provider '%.*s' has an invalid root kind",
            (int)provider->name.size, provider->name.data);
      }
      if (pattern->match_and_rewrite == NULL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "rewrite pattern provider '%.*s' has no callback",
            (int)provider->name.size, provider->name.data);
      }
      const uint8_t op_index = loom_op_dialect_index(pattern->root_kind);
      if (dialect_id < *out_dialect_base_id) {
        *out_dialect_base_id = dialect_id;
      }
      const uint16_t dialect_limit = (uint16_t)dialect_id + 1;
      if (dialect_limit > *out_dialect_limit) {
        *out_dialect_limit = dialect_limit;
      }
      const uint16_t op_count = (uint16_t)op_index + 1;
      if (op_count > dialect_op_counts[dialect_id]) {
        dialect_op_counts[dialect_id] = op_count;
      }
    }
  }
  return iree_ok_status();
}

static void loom_rewrite_pattern_registry_initialize_dialects(
    const uint16_t* dialect_op_counts, uint8_t dialect_base_id,
    uint16_t dialect_count, loom_rewrite_pattern_dialect_table_t* dialects,
    loom_rewrite_pattern_op_entry_t* op_entries,
    loom_rewrite_pattern_op_entry_t** op_entries_by_dialect) {
  uint32_t op_entry_cursor = 0;
  for (uint16_t dialect_index = 0; dialect_index < dialect_count;
       ++dialect_index) {
    const uint8_t dialect_id = (uint8_t)(dialect_base_id + dialect_index);
    const uint16_t op_count = dialect_op_counts[dialect_id];
    if (op_count == 0) {
      dialects[dialect_index] = (loom_rewrite_pattern_dialect_table_t){0};
      continue;
    }
    loom_rewrite_pattern_op_entry_t* dialect_op_entries =
        &op_entries[op_entry_cursor];
    for (uint16_t op_index = 0; op_index < op_count; ++op_index) {
      dialect_op_entries[op_index] = loom_rewrite_pattern_op_entry_empty();
    }
    dialects[dialect_index] = (loom_rewrite_pattern_dialect_table_t){
        .op_count = op_count,
        .op_entries = dialect_op_entries,
    };
    op_entries_by_dialect[dialect_id] = dialect_op_entries;
    op_entry_cursor += op_count;
  }
}

static void loom_rewrite_pattern_registry_count_patterns_by_op(
    loom_rewrite_pattern_provider_list_t providers,
    loom_rewrite_pattern_op_entry_t** op_entries_by_dialect) {
  for (iree_host_size_t provider_index = 0; provider_index < providers.count;
       ++provider_index) {
    const loom_rewrite_pattern_provider_t* provider =
        providers.values[provider_index];
    for (uint16_t pattern_index = 0; pattern_index < provider->pattern_count;
         ++pattern_index) {
      const loom_op_kind_t root_kind =
          provider->patterns[pattern_index].root_kind;
      ++op_entries_by_dialect[loom_op_dialect_id(root_kind)]
                             [loom_op_dialect_index(root_kind)]
                                 .pattern_count;
    }
  }
}

static uint16_t loom_rewrite_pattern_registry_assign_spans(
    loom_rewrite_pattern_dialect_table_t* dialects, uint16_t dialect_count) {
  uint16_t pattern_cursor = 0;
  for (uint16_t dialect_index = 0; dialect_index < dialect_count;
       ++dialect_index) {
    loom_rewrite_pattern_dialect_table_t* dialect = &dialects[dialect_index];
    loom_rewrite_pattern_op_entry_t* op_entries =
        (loom_rewrite_pattern_op_entry_t*)dialect->op_entries;
    for (uint16_t op_index = 0; op_index < dialect->op_count; ++op_index) {
      loom_rewrite_pattern_op_entry_t* entry = &op_entries[op_index];
      if (entry->pattern_count == 0) {
        *entry = loom_rewrite_pattern_op_entry_empty();
        continue;
      }
      const uint16_t span_count = entry->pattern_count;
      entry->pattern_start = pattern_cursor;
      entry->pattern_count = 0;
      pattern_cursor = (uint16_t)(pattern_cursor + span_count);
    }
  }
  return pattern_cursor;
}

static void loom_rewrite_pattern_registry_fill_patterns(
    loom_rewrite_pattern_provider_list_t providers,
    loom_rewrite_pattern_op_entry_t** op_entries_by_dialect,
    const loom_rewrite_pattern_t** patterns) {
  for (iree_host_size_t provider_index = 0; provider_index < providers.count;
       ++provider_index) {
    const loom_rewrite_pattern_provider_t* provider =
        providers.values[provider_index];
    for (uint16_t pattern_index = 0; pattern_index < provider->pattern_count;
         ++pattern_index) {
      const loom_rewrite_pattern_t* pattern =
          &provider->patterns[pattern_index];
      loom_rewrite_pattern_op_entry_t* op_entry =
          &op_entries_by_dialect[loom_op_dialect_id(pattern->root_kind)]
                                [loom_op_dialect_index(pattern->root_kind)];
      patterns[op_entry->pattern_start + op_entry->pattern_count++] = pattern;
    }
  }
}

iree_status_t loom_rewrite_pattern_registry_storage_initialize(
    loom_rewrite_pattern_provider_list_t providers, iree_allocator_t allocator,
    loom_rewrite_pattern_registry_storage_t* out_storage) {
  IREE_ASSERT_ARGUMENT(out_storage);
  *out_storage = (loom_rewrite_pattern_registry_storage_t){0};

  uint16_t dialect_op_counts[UINT8_MAX + 1] = {0};
  uint8_t dialect_base_id = 0;
  uint16_t dialect_limit = 0;
  uint16_t pattern_count = 0;
  IREE_RETURN_IF_ERROR(loom_rewrite_pattern_registry_count_patterns(
      providers, dialect_op_counts, &dialect_base_id, &dialect_limit,
      &pattern_count));
  if (pattern_count == 0) {
    out_storage->allocator = allocator;
    return iree_ok_status();
  }

  const uint16_t dialect_count = dialect_limit - dialect_base_id;
  uint32_t op_entry_count = 0;
  for (uint16_t dialect_index = 0; dialect_index < dialect_count;
       ++dialect_index) {
    op_entry_count += dialect_op_counts[dialect_base_id + dialect_index];
  }

  iree_host_size_t allocation_size = 0;
  iree_host_size_t dialects_offset = 0;
  iree_host_size_t op_entries_offset = 0;
  iree_host_size_t patterns_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &allocation_size,
      IREE_STRUCT_FIELD_ALIGNED(dialect_count,
                                loom_rewrite_pattern_dialect_table_t,
                                iree_max_align_t, &dialects_offset),
      IREE_STRUCT_FIELD_ALIGNED(op_entry_count, loom_rewrite_pattern_op_entry_t,
                                iree_max_align_t, &op_entries_offset),
      IREE_STRUCT_FIELD_ALIGNED(pattern_count, const loom_rewrite_pattern_t*,
                                iree_max_align_t, &patterns_offset)));
  uint8_t* allocation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      allocator, allocation_size, (void**)&allocation));
  loom_rewrite_pattern_dialect_table_t* dialects =
      (loom_rewrite_pattern_dialect_table_t*)(allocation + dialects_offset);
  loom_rewrite_pattern_op_entry_t* op_entries =
      (loom_rewrite_pattern_op_entry_t*)(allocation + op_entries_offset);
  const loom_rewrite_pattern_t** patterns =
      (const loom_rewrite_pattern_t**)(allocation + patterns_offset);
  loom_rewrite_pattern_op_entry_t* op_entries_by_dialect[UINT8_MAX + 1] = {0};
  loom_rewrite_pattern_registry_initialize_dialects(
      dialect_op_counts, dialect_base_id, dialect_count, dialects, op_entries,
      op_entries_by_dialect);
  loom_rewrite_pattern_registry_count_patterns_by_op(providers,
                                                     op_entries_by_dialect);
  const uint16_t assigned_pattern_count =
      loom_rewrite_pattern_registry_assign_spans(dialects, dialect_count);
  loom_rewrite_pattern_registry_fill_patterns(providers, op_entries_by_dialect,
                                              patterns);

  *out_storage = (loom_rewrite_pattern_registry_storage_t){
      .allocator = allocator,
      .allocation = iree_make_byte_span(allocation, allocation_size),
      .registry =
          {
              .dialect_base_id = dialect_base_id,
              .dialect_count = dialect_count,
              .dialects = dialects,
              .patterns = patterns,
              .pattern_count = assigned_pattern_count,
          },
  };
  return iree_ok_status();
}

void loom_rewrite_pattern_registry_storage_deinitialize(
    loom_rewrite_pattern_registry_storage_t* storage) {
  if (storage == NULL) {
    return;
  }
  iree_allocator_free(storage->allocator, storage->allocation.data);
  *storage = (loom_rewrite_pattern_registry_storage_t){0};
}

const loom_rewrite_pattern_registry_t*
loom_rewrite_pattern_registry_storage_registry(
    const loom_rewrite_pattern_registry_storage_t* storage) {
  return &storage->registry;
}

iree_status_t loom_rewrite_pattern_registry_apply(
    const loom_rewrite_pattern_registry_t* registry, void* context,
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed) {
  *out_changed = false;
  const loom_rewrite_pattern_span_t span =
      loom_rewrite_pattern_registry_lookup_kind(registry, op->kind);
  for (uint16_t pattern_index = 0; pattern_index < span.count;
       ++pattern_index) {
    bool changed = false;
    IREE_RETURN_IF_ERROR(span.patterns[pattern_index]->match_and_rewrite(
        span.patterns[pattern_index], context, op, rewriter, &changed));
    if (changed) {
      *out_changed = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

iree_status_t loom_rewrite_pattern_registry_prepare_op(
    const loom_rewrite_pattern_registry_t* registry,
    const loom_module_t* module, const loom_value_fact_table_t* facts,
    const loom_op_t* op, iree_arena_allocator_t* arena,
    loom_rewrite_action_t** out_action) {
  *out_action = NULL;
  const loom_rewrite_pattern_span_t span =
      loom_rewrite_pattern_registry_lookup_kind(registry, op->kind);
  for (uint16_t i = 0; i < span.count; ++i) {
    const loom_rewrite_pattern_t* pattern = span.patterns[i];
    if (!pattern->prepare) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        pattern->prepare(pattern, module, facts, op, arena, out_action));
    if (*out_action) {
      break;
    }
  }
  return iree_ok_status();
}
