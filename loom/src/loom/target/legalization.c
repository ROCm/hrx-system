// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/legalization.h"

#include <stdint.h>

static iree_status_t loom_target_legalizer_registry_count_rows(
    const loom_target_legalizer_provider_t* const* providers,
    iree_host_size_t provider_count, uint16_t* dialect_op_counts,
    uint8_t* out_dialect_base_id, uint16_t* out_dialect_limit,
    uint16_t* out_entry_count) {
  *out_dialect_base_id = UINT8_MAX;
  *out_dialect_limit = 0;
  *out_entry_count = 0;
  for (iree_host_size_t provider_index = 0; provider_index < provider_count;
       ++provider_index) {
    const loom_target_legalizer_provider_t* provider =
        providers[provider_index];
    if (provider->entry_count == 0) {
      continue;
    }
    const uint32_t total_entry_count =
        (uint32_t)(*out_entry_count) + provider->entry_count;
    if (total_entry_count > UINT16_MAX) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "target legalizer registry entry count exceeds uint16_t capacity");
    }
    *out_entry_count = (uint16_t)total_entry_count;
    for (uint16_t entry_index = 0; entry_index < provider->entry_count;
         ++entry_index) {
      const loom_op_kind_t op_kind = provider->entries[entry_index].root_kind;
      const uint8_t dialect_id = loom_op_dialect_id(op_kind);
      const uint8_t op_index = loom_op_dialect_index(op_kind);
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

static iree_status_t loom_target_legalizer_registry_allocate_dialects(
    const uint16_t* dialect_op_counts, uint8_t dialect_base_id,
    uint8_t dialect_count, loom_target_legalizer_dialect_table_t** out_dialects,
    loom_target_legalizer_op_entry_t** op_entries_by_dialect,
    iree_arena_allocator_t* arena) {
  *out_dialects = NULL;

  loom_target_legalizer_dialect_table_t* dialects = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, dialect_count, sizeof(*dialects), (void**)&dialects));

  for (uint16_t i = 0; i <= UINT8_MAX; ++i) {
    op_entries_by_dialect[i] = NULL;
  }

  uint32_t op_entry_count = 0;
  for (uint8_t i = 0; i < dialect_count; ++i) {
    op_entry_count += dialect_op_counts[dialect_base_id + i];
  }
  loom_target_legalizer_op_entry_t* op_entries = NULL;
  if (op_entry_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, op_entry_count, sizeof(*op_entries), (void**)&op_entries));
  }

  uint32_t op_entry_cursor = 0;
  for (uint8_t i = 0; i < dialect_count; ++i) {
    const uint8_t dialect_id = (uint8_t)(dialect_base_id + i);
    const uint16_t op_count = dialect_op_counts[dialect_id];
    if (op_count == 0) {
      dialects[i] = (loom_target_legalizer_dialect_table_t){
          .op_count = 0,
          .op_entries = NULL,
      };
      continue;
    }
    loom_target_legalizer_op_entry_t* dialect_op_entries =
        &op_entries[op_entry_cursor];
    for (uint16_t op_index = 0; op_index < op_count; ++op_index) {
      dialect_op_entries[op_index] = loom_target_legalizer_op_entry_empty();
    }
    dialects[i] = (loom_target_legalizer_dialect_table_t){
        .op_count = op_count,
        .op_entries = dialect_op_entries,
    };
    op_entries_by_dialect[dialect_id] = dialect_op_entries;
    op_entry_cursor += op_count;
  }

  *out_dialects = dialects;
  return iree_ok_status();
}

static void loom_target_legalizer_registry_count_entries_by_op(
    const loom_target_legalizer_provider_t* const* providers,
    iree_host_size_t provider_count,
    loom_target_legalizer_op_entry_t** op_entries_by_dialect) {
  for (iree_host_size_t provider_index = 0; provider_index < provider_count;
       ++provider_index) {
    const loom_target_legalizer_provider_t* provider =
        providers[provider_index];
    for (uint16_t entry_index = 0; entry_index < provider->entry_count;
         ++entry_index) {
      const loom_op_kind_t op_kind = provider->entries[entry_index].root_kind;
      const uint8_t dialect_id = loom_op_dialect_id(op_kind);
      const uint8_t op_index = loom_op_dialect_index(op_kind);
      ++op_entries_by_dialect[dialect_id][op_index].entry_count;
    }
  }
}

static uint16_t loom_target_legalizer_registry_assign_entry_spans(
    loom_target_legalizer_dialect_table_t* dialects, uint8_t dialect_count) {
  uint16_t entry_cursor = 0;
  for (uint8_t dialect_index = 0; dialect_index < dialect_count;
       ++dialect_index) {
    loom_target_legalizer_dialect_table_t* dialect = &dialects[dialect_index];
    loom_target_legalizer_op_entry_t* op_entries =
        (loom_target_legalizer_op_entry_t*)dialect->op_entries;
    for (uint16_t op_index = 0; op_index < dialect->op_count; ++op_index) {
      loom_target_legalizer_op_entry_t* entry = &op_entries[op_index];
      if (entry->entry_count == 0) {
        *entry = loom_target_legalizer_op_entry_empty();
        continue;
      }
      const uint16_t span_count = entry->entry_count;
      entry->entry_start = entry_cursor;
      entry->entry_count = 0;
      entry_cursor = (uint16_t)(entry_cursor + span_count);
    }
  }
  return entry_cursor;
}

static void loom_target_legalizer_registry_fill_entries(
    const loom_target_legalizer_provider_t* const* providers,
    iree_host_size_t provider_count,
    loom_target_legalizer_op_entry_t** op_entries_by_dialect,
    loom_target_legalizer_entry_t* entries) {
  for (iree_host_size_t provider_index = 0; provider_index < provider_count;
       ++provider_index) {
    const loom_target_legalizer_provider_t* provider =
        providers[provider_index];
    for (uint16_t entry_index = 0; entry_index < provider->entry_count;
         ++entry_index) {
      const loom_target_legalizer_entry_t* source_entry =
          &provider->entries[entry_index];
      const uint8_t dialect_id = loom_op_dialect_id(source_entry->root_kind);
      const uint8_t op_index = loom_op_dialect_index(source_entry->root_kind);
      loom_target_legalizer_op_entry_t* op_entry =
          &op_entries_by_dialect[dialect_id][op_index];
      loom_target_legalizer_entry_t* target_entry =
          &entries[op_entry->entry_start + op_entry->entry_count++];
      *target_entry = *source_entry;
      target_entry->provider_name = provider->name;
      target_entry->provider_strategy = provider->strategy;
    }
  }
}

iree_status_t loom_target_legalizer_registry_compose(
    const loom_target_legalizer_provider_t* const* providers,
    iree_host_size_t provider_count,
    loom_target_legalizer_registry_t* out_registry,
    iree_arena_allocator_t* arena) {
  IREE_ASSERT_ARGUMENT(out_registry);
  IREE_ASSERT_ARGUMENT(arena);
  if (provider_count != 0) {
    IREE_ASSERT_ARGUMENT(providers);
  }
  *out_registry = (loom_target_legalizer_registry_t){0};
  if (provider_count == 0) {
    return iree_ok_status();
  }

  uint16_t dialect_op_counts[UINT8_MAX + 1] = {0};
  uint8_t dialect_base_id = 0;
  uint16_t dialect_limit = 0;
  uint16_t entry_count = 0;
  IREE_RETURN_IF_ERROR(loom_target_legalizer_registry_count_rows(
      providers, provider_count, dialect_op_counts, &dialect_base_id,
      &dialect_limit, &entry_count));
  if (entry_count == 0) {
    return iree_ok_status();
  }

  const uint16_t dialect_count_u16 = dialect_limit - dialect_base_id;
  if (dialect_count_u16 > UINT8_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "target legalizer registry dialect span exceeds uint8_t capacity");
  }
  const uint8_t dialect_count = (uint8_t)dialect_count_u16;

  loom_target_legalizer_dialect_table_t* dialects = NULL;
  loom_target_legalizer_op_entry_t* op_entries_by_dialect[UINT8_MAX + 1] = {0};
  IREE_RETURN_IF_ERROR(loom_target_legalizer_registry_allocate_dialects(
      dialect_op_counts, dialect_base_id, dialect_count, &dialects,
      op_entries_by_dialect, arena));
  loom_target_legalizer_registry_count_entries_by_op(providers, provider_count,
                                                     op_entries_by_dialect);
  const uint16_t assigned_entry_count =
      loom_target_legalizer_registry_assign_entry_spans(dialects,
                                                        dialect_count);

  loom_target_legalizer_entry_t* entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, assigned_entry_count, sizeof(*entries), (void**)&entries));
  loom_target_legalizer_registry_fill_entries(providers, provider_count,
                                              op_entries_by_dialect, entries);

  *out_registry = (loom_target_legalizer_registry_t){
      .dialect_base_id = dialect_base_id,
      .dialect_count = dialect_count,
      .dialects = dialects,
      .entries = entries,
      .entry_count = assigned_entry_count,
  };
  return iree_ok_status();
}

iree_status_t loom_target_legalization_query_contract(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    loom_target_contract_query_result_t* out_result) {
  *out_result = loom_target_contract_query_result_empty();
  if (loom_target_contract_query_callback_is_empty(context->contract_query)) {
    return iree_ok_status();
  }
  const loom_target_contract_query_environment_t environment = {
      .module = context->module,
      .function = context->function,
      .bundle = context->bundle,
      .target_data = context->target_data,
      .target_ref = context->target_ref,
      .descriptor_set = context->descriptor_set,
      .fact_table = context->fact_table,
      .view_regions = context->view_regions,
      .arena = context->arena,
  };
  return context->contract_query.fn(context->contract_query.user_data,
                                    &environment, op, out_result);
}
