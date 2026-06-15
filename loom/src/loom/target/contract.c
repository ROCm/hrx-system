// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/contract.h"

#include <stddef.h>
#include <stdint.h>

static iree_status_t loom_target_contract_index_count_fragment_rows(
    const loom_target_contract_binding_t* bindings, uint16_t binding_count,
    uint16_t* dialect_op_counts, uint8_t* out_dialect_base_id,
    uint16_t* out_dialect_limit, uint16_t* out_case_count) {
  *out_dialect_base_id = UINT8_MAX;
  *out_dialect_limit = 0;
  *out_case_count = 0;
  for (uint16_t binding_index = 0; binding_index < binding_count;
       ++binding_index) {
    const loom_target_contract_fragment_t* fragment =
        bindings[binding_index].fragment;
    const uint32_t total_case_count =
        (uint32_t)(*out_case_count) + fragment->case_count;
    if (total_case_count > UINT16_MAX) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "target contract index case count exceeds uint16_t capacity");
    }
    *out_case_count = (uint16_t)total_case_count;
    for (uint8_t i = 0; i < fragment->dialect_count; ++i) {
      const uint8_t dialect_id = (uint8_t)(fragment->dialect_base_id + i);
      const loom_target_contract_dialect_table_t* dialect_table =
          &fragment->dialects[i];
      if (dialect_table->op_count == 0) {
        continue;
      }
      if (dialect_id < *out_dialect_base_id) {
        *out_dialect_base_id = dialect_id;
      }
      const uint16_t dialect_limit = (uint16_t)dialect_id + 1;
      if (dialect_limit > *out_dialect_limit) {
        *out_dialect_limit = dialect_limit;
      }
      if (dialect_table->op_count > dialect_op_counts[dialect_id]) {
        dialect_op_counts[dialect_id] = dialect_table->op_count;
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_contract_index_allocate_dialects(
    const uint16_t* dialect_op_counts, uint8_t dialect_base_id,
    uint8_t dialect_count, loom_target_contract_dialect_table_t** out_dialects,
    loom_target_contract_op_entry_t** op_entries_by_dialect,
    iree_arena_allocator_t* arena) {
  *out_dialects = NULL;

  loom_target_contract_dialect_table_t* dialects = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, dialect_count, sizeof(*dialects), (void**)&dialects));

  for (uint16_t i = 0; i <= UINT8_MAX; ++i) {
    op_entries_by_dialect[i] = NULL;
  }

  uint32_t op_entry_count = 0;
  for (uint8_t i = 0; i < dialect_count; ++i) {
    op_entry_count += dialect_op_counts[dialect_base_id + i];
  }
  loom_target_contract_op_entry_t* op_entries = NULL;
  if (op_entry_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, op_entry_count, sizeof(*op_entries), (void**)&op_entries));
  }

  uint32_t op_entry_cursor = 0;
  for (uint8_t i = 0; i < dialect_count; ++i) {
    const uint8_t dialect_id = (uint8_t)(dialect_base_id + i);
    const uint16_t op_count = dialect_op_counts[dialect_id];
    if (op_count == 0) {
      dialects[i] = (loom_target_contract_dialect_table_t){
          .op_count = 0,
          .op_entries = NULL,
      };
      continue;
    }
    loom_target_contract_op_entry_t* dialect_op_entries =
        &op_entries[op_entry_cursor];
    for (uint16_t op_index = 0; op_index < op_count; ++op_index) {
      dialect_op_entries[op_index] = loom_target_contract_op_entry_empty();
    }
    dialects[i] = (loom_target_contract_dialect_table_t){
        .op_count = op_count,
        .op_entries = dialect_op_entries,
    };
    op_entries_by_dialect[dialect_id] = dialect_op_entries;
    op_entry_cursor += op_count;
  }

  *out_dialects = dialects;
  return iree_ok_status();
}

static iree_status_t loom_target_contract_index_count_cases_by_op(
    const loom_target_contract_binding_t* bindings, uint16_t binding_count,
    loom_target_contract_op_entry_t** op_entries_by_dialect) {
  for (uint16_t binding_index = 0; binding_index < binding_count;
       ++binding_index) {
    const loom_target_contract_fragment_t* fragment =
        bindings[binding_index].fragment;
    for (uint8_t dialect_index = 0; dialect_index < fragment->dialect_count;
         ++dialect_index) {
      const uint8_t dialect_id =
          (uint8_t)(fragment->dialect_base_id + dialect_index);
      const loom_target_contract_dialect_table_t* fragment_dialect =
          &fragment->dialects[dialect_index];
      loom_target_contract_op_entry_t* root_op_entries =
          op_entries_by_dialect[dialect_id];
      for (uint16_t op_index = 0; op_index < fragment_dialect->op_count;
           ++op_index) {
        const loom_target_contract_op_entry_t fragment_entry =
            fragment_dialect->op_entries[op_index];
        if (loom_target_contract_op_entry_is_empty(fragment_entry)) {
          continue;
        }
        loom_target_contract_op_entry_t* root_entry =
            &root_op_entries[op_index];
        const uint32_t case_count =
            (uint32_t)root_entry->case_count + fragment_entry.case_count;
        if (case_count > UINT16_MAX) {
          return iree_make_status(
              IREE_STATUS_RESOURCE_EXHAUSTED,
              "target contract op case count exceeds uint16_t capacity");
        }
        root_entry->case_count = (uint16_t)case_count;
      }
    }
  }
  return iree_ok_status();
}

static uint16_t loom_target_contract_index_assign_case_spans(
    loom_target_contract_dialect_table_t* dialects, uint8_t dialect_count) {
  uint16_t case_cursor = 0;
  for (uint8_t dialect_index = 0; dialect_index < dialect_count;
       ++dialect_index) {
    loom_target_contract_dialect_table_t* dialect = &dialects[dialect_index];
    loom_target_contract_op_entry_t* op_entries =
        (loom_target_contract_op_entry_t*)dialect->op_entries;
    for (uint16_t op_index = 0; op_index < dialect->op_count; ++op_index) {
      const uint16_t case_count = op_entries[op_index].case_count;
      if (case_count == 0) {
        op_entries[op_index] = loom_target_contract_op_entry_empty();
        continue;
      }
      op_entries[op_index].case_start = case_cursor;
      op_entries[op_index].case_count = 0;
      case_cursor = (uint16_t)(case_cursor + case_count);
    }
  }
  return case_cursor;
}

static void loom_target_contract_index_fill_cases(
    const loom_target_contract_binding_t* bindings, uint16_t binding_count,
    loom_target_contract_op_entry_t** op_entries_by_dialect,
    loom_target_contract_case_t* cases) {
  for (uint16_t binding_index = 0; binding_index < binding_count;
       ++binding_index) {
    const loom_target_contract_fragment_t* fragment =
        bindings[binding_index].fragment;
    for (uint8_t dialect_index = 0; dialect_index < fragment->dialect_count;
         ++dialect_index) {
      const uint8_t dialect_id =
          (uint8_t)(fragment->dialect_base_id + dialect_index);
      const loom_target_contract_dialect_table_t* fragment_dialect =
          &fragment->dialects[dialect_index];
      loom_target_contract_op_entry_t* root_op_entries =
          op_entries_by_dialect[dialect_id];
      for (uint16_t op_index = 0; op_index < fragment_dialect->op_count;
           ++op_index) {
        const loom_target_contract_op_entry_t fragment_entry =
            fragment_dialect->op_entries[op_index];
        if (loom_target_contract_op_entry_is_empty(fragment_entry)) {
          continue;
        }
        loom_target_contract_op_entry_t* root_entry =
            &root_op_entries[op_index];
        const uint16_t case_start =
            (uint16_t)(root_entry->case_start + root_entry->case_count);
        for (uint16_t case_index = 0; case_index < fragment_entry.case_count;
             ++case_index) {
          const loom_target_contract_fragment_case_t* fragment_case =
              &fragment->cases[fragment_entry.case_start + case_index];
          cases[case_start + case_index] = (loom_target_contract_case_t){
              .system = fragment_case->system,
              .binding_index = (uint8_t)binding_index,
              .row_index = fragment_case->row_index,
          };
        }
        root_entry->case_count =
            (uint16_t)(root_entry->case_count + fragment_entry.case_count);
      }
    }
  }
}

iree_status_t loom_target_contract_index_compose(
    const loom_target_contract_binding_t* bindings, uint16_t binding_count,
    loom_target_contract_index_t* out_index, iree_arena_allocator_t* arena) {
  IREE_ASSERT_ARGUMENT(out_index);
  IREE_ASSERT_ARGUMENT(arena);
  if (binding_count != 0) {
    IREE_ASSERT_ARGUMENT(bindings);
  }
  *out_index = (loom_target_contract_index_t){0};
  if (binding_count == 0) {
    return iree_ok_status();
  }
  if (binding_count > UINT8_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "target contract index binding count exceeds uint8_t capacity");
  }

  uint16_t dialect_op_counts[UINT8_MAX + 1] = {0};
  uint8_t dialect_base_id = 0;
  uint16_t dialect_limit = 0;
  uint16_t case_count = 0;
  IREE_RETURN_IF_ERROR(loom_target_contract_index_count_fragment_rows(
      bindings, binding_count, dialect_op_counts, &dialect_base_id,
      &dialect_limit, &case_count));
  if (case_count == 0) {
    return iree_ok_status();
  }

  const uint16_t dialect_count_u16 = dialect_limit - dialect_base_id;
  if (dialect_count_u16 > UINT8_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "target contract index dialect span exceeds uint8_t capacity");
  }
  const uint8_t dialect_count = (uint8_t)dialect_count_u16;

  loom_target_contract_dialect_table_t* dialects = NULL;
  loom_target_contract_op_entry_t* op_entries_by_dialect[UINT8_MAX + 1] = {0};
  IREE_RETURN_IF_ERROR(loom_target_contract_index_allocate_dialects(
      dialect_op_counts, dialect_base_id, dialect_count, &dialects,
      op_entries_by_dialect, arena));
  IREE_RETURN_IF_ERROR(loom_target_contract_index_count_cases_by_op(
      bindings, binding_count, op_entries_by_dialect));
  const uint16_t assigned_case_count =
      loom_target_contract_index_assign_case_spans(dialects, dialect_count);

  loom_target_contract_case_t* cases = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, assigned_case_count, sizeof(*cases), (void**)&cases));
  loom_target_contract_index_fill_cases(bindings, binding_count,
                                        op_entries_by_dialect, cases);

  *out_index = (loom_target_contract_index_t){
      .dialect_base_id = dialect_base_id,
      .dialect_count = dialect_count,
      .dialects = dialects,
      .case_count = assigned_case_count,
      .cases = cases,
      .binding_count = (uint8_t)binding_count,
      .bindings = bindings,
  };
  return iree_ok_status();
}
