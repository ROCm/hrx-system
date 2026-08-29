// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/legalization.h"

#include <stdint.h>

static iree_status_t loom_target_legalizer_registry_count_rows(
    const loom_target_legalizer_provider_list_t* provider_lists,
    iree_host_size_t provider_list_count, uint16_t* dialect_op_counts,
    uint8_t* out_dialect_base_id, uint16_t* out_dialect_limit,
    uint16_t* out_entry_count) {
  *out_dialect_base_id = UINT8_MAX;
  *out_dialect_limit = 0;
  *out_entry_count = 0;
  for (iree_host_size_t list_index = 0; list_index < provider_list_count;
       ++list_index) {
    const loom_target_legalizer_provider_list_t* provider_list =
        &provider_lists[list_index];
    if (provider_list->count != 0 && provider_list->values == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "target legalizer provider list is missing its "
                              "provider pointer table");
    }
    for (iree_host_size_t provider_index = 0;
         provider_index < provider_list->count; ++provider_index) {
      const loom_target_legalizer_provider_t* provider =
          provider_list->values[provider_index];
      if (provider == NULL) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "target legalizer provider must not be NULL");
      }
      if (provider->entry_count == 0) {
        continue;
      }
      if (provider->entries == NULL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "target legalizer provider has entries but no entry table");
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
  }
  return iree_ok_status();
}

static void loom_target_legalizer_registry_initialize_dialects(
    const uint16_t* dialect_op_counts, uint8_t dialect_base_id,
    uint8_t dialect_count, loom_target_legalizer_dialect_table_t* dialects,
    loom_target_legalizer_op_entry_t* op_entries,
    loom_target_legalizer_op_entry_t** op_entries_by_dialect,
    uint32_t op_entry_count) {
  for (uint16_t i = 0; i <= UINT8_MAX; ++i) {
    op_entries_by_dialect[i] = NULL;
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
  IREE_ASSERT_EQ(op_entry_cursor, op_entry_count);
}

static void loom_target_legalizer_registry_count_entries_by_op(
    const loom_target_legalizer_provider_list_t* provider_lists,
    iree_host_size_t provider_list_count,
    loom_target_legalizer_op_entry_t** op_entries_by_dialect) {
  for (iree_host_size_t list_index = 0; list_index < provider_list_count;
       ++list_index) {
    const loom_target_legalizer_provider_list_t* provider_list =
        &provider_lists[list_index];
    for (iree_host_size_t provider_index = 0;
         provider_index < provider_list->count; ++provider_index) {
      const loom_target_legalizer_provider_t* provider =
          provider_list->values[provider_index];
      for (uint16_t entry_index = 0; entry_index < provider->entry_count;
           ++entry_index) {
        const loom_op_kind_t op_kind = provider->entries[entry_index].root_kind;
        const uint8_t dialect_id = loom_op_dialect_id(op_kind);
        const uint8_t op_index = loom_op_dialect_index(op_kind);
        ++op_entries_by_dialect[dialect_id][op_index].entry_count;
      }
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
    const loom_target_legalizer_provider_list_t* provider_lists,
    iree_host_size_t provider_list_count,
    loom_target_legalizer_op_entry_t** op_entries_by_dialect,
    loom_target_legalizer_entry_t* entries) {
  for (iree_host_size_t list_index = 0; list_index < provider_list_count;
       ++list_index) {
    const loom_target_legalizer_provider_list_t* provider_list =
        &provider_lists[list_index];
    for (iree_host_size_t provider_index = 0;
         provider_index < provider_list->count; ++provider_index) {
      const loom_target_legalizer_provider_t* provider =
          provider_list->values[provider_index];
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
}

iree_status_t loom_target_legalizer_registry_storage_initialize(
    const loom_target_legalizer_provider_list_t* provider_lists,
    iree_host_size_t provider_list_count, iree_allocator_t allocator,
    loom_target_legalizer_registry_storage_t* out_storage) {
  IREE_ASSERT_ARGUMENT(out_storage);
  *out_storage = (loom_target_legalizer_registry_storage_t){0};
  if (provider_list_count != 0 && provider_lists == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target legalizer provider lists must not be NULL");
  }

  uint16_t dialect_op_counts[UINT8_MAX + 1] = {0};
  uint8_t dialect_base_id = 0;
  uint16_t dialect_limit = 0;
  uint16_t entry_count = 0;
  IREE_RETURN_IF_ERROR(loom_target_legalizer_registry_count_rows(
      provider_lists, provider_list_count, dialect_op_counts, &dialect_base_id,
      &dialect_limit, &entry_count));
  if (entry_count == 0) {
    out_storage->allocator = allocator;
    return iree_ok_status();
  }

  const uint16_t dialect_count_u16 = dialect_limit - dialect_base_id;
  if (dialect_count_u16 > UINT8_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "target legalizer registry dialect span exceeds uint8_t capacity");
  }
  const uint8_t dialect_count = (uint8_t)dialect_count_u16;

  uint32_t op_entry_count = 0;
  for (uint8_t i = 0; i < dialect_count; ++i) {
    op_entry_count += dialect_op_counts[dialect_base_id + i];
  }

  iree_host_size_t allocation_size = 0;
  iree_host_size_t dialects_offset = 0;
  iree_host_size_t op_entries_offset = 0;
  iree_host_size_t entries_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &allocation_size,
      IREE_STRUCT_FIELD_ALIGNED(dialect_count,
                                loom_target_legalizer_dialect_table_t,
                                iree_max_align_t, &dialects_offset),
      IREE_STRUCT_FIELD_ALIGNED(op_entry_count,
                                loom_target_legalizer_op_entry_t,
                                iree_max_align_t, &op_entries_offset),
      IREE_STRUCT_FIELD_ALIGNED(entry_count, loom_target_legalizer_entry_t,
                                iree_max_align_t, &entries_offset)));
  uint8_t* allocation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      allocator, allocation_size, (void**)&allocation));
  loom_target_legalizer_dialect_table_t* dialects =
      (loom_target_legalizer_dialect_table_t*)(allocation + dialects_offset);
  loom_target_legalizer_op_entry_t* op_entries =
      (loom_target_legalizer_op_entry_t*)(allocation + op_entries_offset);
  loom_target_legalizer_entry_t* entries =
      (loom_target_legalizer_entry_t*)(allocation + entries_offset);
  loom_target_legalizer_op_entry_t* op_entries_by_dialect[UINT8_MAX + 1] = {0};
  loom_target_legalizer_registry_initialize_dialects(
      dialect_op_counts, dialect_base_id, dialect_count, dialects, op_entries,
      op_entries_by_dialect, op_entry_count);
  loom_target_legalizer_registry_count_entries_by_op(
      provider_lists, provider_list_count, op_entries_by_dialect);
  const uint16_t assigned_entry_count =
      loom_target_legalizer_registry_assign_entry_spans(dialects,
                                                        dialect_count);
  IREE_ASSERT_EQ(assigned_entry_count, entry_count);
  loom_target_legalizer_registry_fill_entries(
      provider_lists, provider_list_count, op_entries_by_dialect, entries);

  *out_storage = (loom_target_legalizer_registry_storage_t){
      .allocator = allocator,
      .allocation = iree_make_byte_span(allocation, allocation_size),
      .registry =
          {
              .dialect_base_id = dialect_base_id,
              .dialect_count = dialect_count,
              .dialects = dialects,
              .entries = entries,
              .entry_count = assigned_entry_count,
          },
  };
  return iree_ok_status();
}

void loom_target_legalizer_registry_storage_deinitialize(
    loom_target_legalizer_registry_storage_t* storage) {
  if (storage == NULL) {
    return;
  }
  iree_allocator_free(storage->allocator, storage->allocation.data);
  *storage = (loom_target_legalizer_registry_storage_t){0};
}

const loom_target_legalizer_registry_t*
loom_target_legalizer_registry_storage_registry(
    const loom_target_legalizer_registry_storage_t* storage) {
  return &storage->registry;
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
      .target_facts = context->target_facts,
      .descriptor_set = context->descriptor_set,
      .fact_table = context->fact_table,
      .view_regions = context->view_regions,
      .arena = context->arena,
  };
  return context->contract_query.fn(context->contract_query.user_data,
                                    &environment, op, out_result);
}
