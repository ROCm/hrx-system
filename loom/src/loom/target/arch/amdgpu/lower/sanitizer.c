// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/sanitizer.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/bitfield.h"
#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/sanitizer/site_table.h"
#include "loom/target/arch/amdgpu/abi/tsan.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_access.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/registers.h"

typedef struct loom_amdgpu_sanitizer_site_chunk_t {
  // Next committed site-row chunk, or NULL for the final chunk.
  struct loom_amdgpu_sanitizer_site_chunk_t* next;
  // Rows copied from one successfully lowered source function.
  loom_sanitizer_site_row_t* rows;
  // Number of rows in rows.
  iree_host_size_t row_count;
} loom_amdgpu_sanitizer_site_chunk_t;

typedef struct loom_amdgpu_sanitizer_module_state_t {
  // First committed site-row chunk, or NULL when no sites were committed.
  loom_amdgpu_sanitizer_site_chunk_t* site_chunk_head;
  // Final committed site-row chunk, or NULL when no sites were committed.
  loom_amdgpu_sanitizer_site_chunk_t* site_chunk_tail;
  // Total number of committed site rows across all chunks.
  iree_host_size_t site_row_count;
} loom_amdgpu_sanitizer_module_state_t;

typedef struct loom_amdgpu_sanitizer_lower_state_t {
  // True once the source function sanitizer site collection has been built.
  bool has_site_collection;
  // First module-global site ID assigned to this function's collection.
  loom_sanitizer_site_id_t site_id_base;
  // Function-local sanitizer site rows in report-site order.
  loom_sanitizer_site_collection_t site_collection;
  // Next function-local site row expected by source-to-low planning.
  iree_host_size_t next_site_row_index;
  // True once the runtime feedback config symbol has been looked up or created.
  bool has_feedback_config_symbol;
  // Module-local feedback channel configuration symbol.
  loom_symbol_ref_t feedback_config_symbol;
  // True once the runtime ASAN shadow config symbol has been looked up or
  // created.
  bool has_asan_config_symbol;
  // Module-local ASAN shadow configuration symbol.
  loom_symbol_ref_t asan_config_symbol;
  // True once the runtime TSAN shadow config symbol has been looked up or
  // created.
  bool has_tsan_config_symbol;
  // Module-local TSAN shadow configuration symbol.
  loom_symbol_ref_t tsan_config_symbol;
  // True once trap-mode sanitizer failures have a shared cold island.
  bool has_trap_island;
  // Shared no-return cold island for trap-mode sanitizer failures.
  loom_amdgpu_sanitizer_trap_island_t trap_island;
  // True once read access failures have a shared cold report island.
  bool has_read_island;
  // Shared cold island for read access failures.
  loom_amdgpu_sanitizer_access_report_island_t read_island;
  // True once write access failures have a shared cold report island.
  bool has_write_island;
  // Shared cold island for write access failures.
  loom_amdgpu_sanitizer_access_report_island_t write_island;
  // True once atomic access failures have a shared cold report island.
  bool has_atomic_island;
  // Shared cold island for atomic access failures.
  loom_amdgpu_sanitizer_access_report_island_t atomic_island;
} loom_amdgpu_sanitizer_lower_state_t;

typedef struct loom_amdgpu_sanitizer_access_diagnostic_t {
  // Sanitizer-specific rejection key when the failure is not owned by memory
  // planning.
  iree_string_view_t sanitizer_constraint_key;
  // Target-independent source memory planning rejection bits.
  loom_low_source_memory_access_diagnostic_t source;
  // AMDGPU flat-address selection rejection bits.
  loom_amdgpu_memory_access_diagnostic_t memory;
} loom_amdgpu_sanitizer_access_diagnostic_t;

static int loom_amdgpu_sanitizer_lower_state_key;
static int loom_amdgpu_sanitizer_module_state_key;

static iree_status_t loom_amdgpu_sanitizer_emit_site_table_overflow_diagnostic(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_host_size_t current_site_count, iree_host_size_t function_site_count) {
  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_lower_context_target_key(context)),
      loom_param_string(loom_low_lower_context_export_name(context)),
      loom_param_string(loom_low_lower_context_config_key(context)),
      loom_param_string(loom_low_lower_context_function_name(context)),
      loom_param_string(loom_op_name(module, source_op)),
      loom_param_u64((uint64_t)current_site_count),
      loom_param_u64((uint64_t)function_site_count),
      loom_param_u64((uint64_t)LOOM_SANITIZER_SITE_ID_INVALID - 1),
  };
  return loom_low_lower_emit_error_ref(context, source_op,
                                       LOOM_ERR_AMDGPU_042_REF, params,
                                       IREE_ARRAYSIZE(params));
}

static iree_status_t loom_amdgpu_sanitizer_emit_site_table_symbol_diagnostic(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t symbol_name) {
  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_lower_context_target_key(context)),
      loom_param_string(loom_low_lower_context_export_name(context)),
      loom_param_string(loom_low_lower_context_config_key(context)),
      loom_param_string(loom_low_lower_context_function_name(context)),
      loom_param_string(loom_op_name(module, source_op)),
      loom_param_string(symbol_name),
  };
  return loom_low_lower_emit_error_ref(context, source_op,
                                       LOOM_ERR_AMDGPU_043_REF, params,
                                       IREE_ARRAYSIZE(params));
}

static iree_status_t loom_amdgpu_sanitizer_get_or_create_symbol(
    loom_module_t* module, iree_string_view_t name,
    loom_symbol_ref_t* out_symbol_ref) {
  *out_symbol_ref = loom_symbol_ref_null();
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, name, &name_id));
  uint16_t symbol_id = loom_module_find_symbol(module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name_id, &symbol_id));
  }
  *out_symbol_ref = (loom_symbol_ref_t){
      .module_id = 0,
      .symbol_id = symbol_id,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_lower_state(
    loom_low_lower_context_t* context,
    loom_amdgpu_sanitizer_lower_state_t** out_state) {
  *out_state = NULL;
  loom_amdgpu_sanitizer_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_sanitizer_lower_state_key, sizeof(*state),
      (void**)&state));
  *out_state = state;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_module_state_from_context(
    loom_low_lower_context_t* context,
    loom_amdgpu_sanitizer_module_state_t** out_state) {
  *out_state = NULL;
  loom_amdgpu_sanitizer_module_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_module_target_state(
      context, &loom_amdgpu_sanitizer_module_state_key, sizeof(*state),
      (void**)&state));
  *out_state = state;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_reserve_site_table_symbol(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_module_t* module = loom_low_lower_context_module(context);
  const iree_string_view_t symbol_name =
      IREE_SV(LOOM_SANITIZER_SITE_TABLE_SYMBOL_NAME);
  loom_symbol_ref_t symbol_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_get_or_create_symbol(
      module, symbol_name, &symbol_ref));
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->defining_op != NULL) {
    return loom_amdgpu_sanitizer_emit_site_table_symbol_diagnostic(
        context, source_op, symbol_name);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_module_state_from_module_state(
    loom_low_lower_module_state_t* module_state,
    loom_amdgpu_sanitizer_module_state_t** out_state) {
  *out_state = NULL;
  loom_amdgpu_sanitizer_module_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_module_state_get_or_allocate(
      module_state, &loom_amdgpu_sanitizer_module_state_key, sizeof(*state),
      (void**)&state));
  *out_state = state;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_ensure_feedback_config_symbol(
    loom_low_lower_context_t* context,
    loom_amdgpu_sanitizer_lower_state_t* state) {
  if (!state->has_feedback_config_symbol) {
    loom_module_t* module = loom_low_lower_context_module(context);
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_get_or_create_symbol(
        module, IREE_SV("iree_feedback_config"),
        &state->feedback_config_symbol));
    state->has_feedback_config_symbol = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_ensure_asan_config_symbol(
    loom_low_lower_context_t* context,
    loom_amdgpu_sanitizer_lower_state_t* state) {
  if (!state->has_asan_config_symbol) {
    loom_module_t* module = loom_low_lower_context_module(context);
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_get_or_create_symbol(
        module, IREE_SV("iree_asan_config"), &state->asan_config_symbol));
    state->has_asan_config_symbol = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_ensure_tsan_config_symbol(
    loom_low_lower_context_t* context,
    loom_amdgpu_sanitizer_lower_state_t* state) {
  if (!state->has_tsan_config_symbol) {
    loom_module_t* module = loom_low_lower_context_module(context);
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_get_or_create_symbol(
        module, IREE_SV(LOOM_AMDGPU_TSAN_CONFIG_GLOBAL_NAME),
        &state->tsan_config_symbol));
    state->has_tsan_config_symbol = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_ensure_config_symbols(
    loom_low_lower_context_t* context,
    loom_amdgpu_sanitizer_lower_state_t* state) {
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_ensure_feedback_config_symbol(context, state));
  return loom_amdgpu_sanitizer_ensure_asan_config_symbol(context, state);
}

iree_status_t loom_amdgpu_sanitizer_feedback_config_symbol(
    loom_low_lower_context_t* context, loom_symbol_ref_t* out_symbol_ref) {
  *out_symbol_ref = loom_symbol_ref_null();
  loom_amdgpu_sanitizer_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_lower_state(context, &state));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_ensure_feedback_config_symbol(context, state));
  *out_symbol_ref = state->feedback_config_symbol;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_sanitizer_tsan_config_symbol(
    loom_low_lower_context_t* context, loom_symbol_ref_t* out_symbol_ref) {
  *out_symbol_ref = loom_symbol_ref_null();
  loom_amdgpu_sanitizer_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_lower_state(context, &state));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_ensure_tsan_config_symbol(context, state));
  *out_symbol_ref = state->tsan_config_symbol;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_ensure_site_collection(
    loom_low_lower_context_t* context,
    loom_amdgpu_sanitizer_lower_state_t* state) {
  if (state->has_site_collection) return iree_ok_status();

  loom_amdgpu_sanitizer_module_state_t* module_state = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_module_state_from_context(context, &module_state));

  loom_module_t* module = loom_low_lower_context_module(context);
  IREE_RETURN_IF_ERROR(loom_sanitizer_site_collection_build_function(
      module, loom_low_lower_context_source_function(context),
      loom_low_lower_context_function_arena(context), &state->site_collection));
  const iree_host_size_t row_count = state->site_collection.row_count;
  iree_host_size_t total_row_count = 0;
  if (!iree_host_size_checked_add(module_state->site_row_count, row_count,
                                  &total_row_count) ||
      total_row_count > LOOM_SANITIZER_SITE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_sanitizer_emit_site_table_overflow_diagnostic(
            context, loom_low_lower_context_source_function(context).op,
            module_state->site_row_count, row_count));
    return iree_ok_status();
  }
  if (row_count != 0) {
    const uint32_t previous_error_count =
        loom_low_lower_context_error_count(context);
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_reserve_site_table_symbol(
        context, loom_low_lower_context_source_function(context).op));
    if (loom_low_lower_context_error_count(context) != previous_error_count) {
      return iree_ok_status();
    }
  }
  state->site_id_base = (loom_sanitizer_site_id_t)module_state->site_row_count;
  for (iree_host_size_t i = 0; i < row_count; ++i) {
    state->site_collection.rows[i].site_id =
        (loom_sanitizer_site_id_t)(state->site_id_base + i);
  }

  state->has_site_collection = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_finalize_sanitizer_function(
    loom_low_lower_context_t* context) {
  loom_amdgpu_sanitizer_lower_state_t* function_state = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_lower_state(context, &function_state));
  if (!function_state->has_site_collection ||
      function_state->site_collection.row_count == 0) {
    return iree_ok_status();
  }

  loom_amdgpu_sanitizer_module_state_t* module_state = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_module_state_from_context(context, &module_state));
  IREE_ASSERT(module_state->site_row_count == function_state->site_id_base,
              "AMDGPU sanitizer site rows committed out of order");

  const iree_host_size_t row_count = function_state->site_collection.row_count;
  loom_low_lower_module_state_t* lower_module_state =
      loom_low_lower_context_module_state(context);
  loom_amdgpu_sanitizer_site_chunk_t* chunk = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_module_state_allocate(
      lower_module_state, sizeof(*chunk), (void**)&chunk));
  memset(chunk, 0, sizeof(*chunk));
  IREE_RETURN_IF_ERROR(loom_low_lower_module_state_allocate_array(
      lower_module_state, row_count, sizeof(*chunk->rows),
      (void**)&chunk->rows));
  for (iree_host_size_t i = 0; i < row_count; ++i) {
    chunk->rows[i] = function_state->site_collection.rows[i];
    chunk->rows[i].op = NULL;
  }
  chunk->row_count = row_count;

  if (module_state->site_chunk_tail != NULL) {
    module_state->site_chunk_tail->next = chunk;
  } else {
    module_state->site_chunk_head = chunk;
  }
  module_state->site_chunk_tail = chunk;
  module_state->site_row_count += row_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_flatten_site_rows(
    const loom_amdgpu_sanitizer_module_state_t* module_state,
    iree_arena_allocator_t* scratch_arena,
    loom_sanitizer_site_collection_t* out_collection) {
  *out_collection = (loom_sanitizer_site_collection_t){0};
  const iree_host_size_t row_count = module_state->site_row_count;
  out_collection->row_count = row_count;
  if (row_count == 0) return iree_ok_status();

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, row_count, sizeof(*out_collection->rows),
      (void**)&out_collection->rows));
  iree_host_size_t row_index = 0;
  for (const loom_amdgpu_sanitizer_site_chunk_t* chunk =
           module_state->site_chunk_head;
       chunk != NULL; chunk = chunk->next) {
    memcpy(out_collection->rows + row_index, chunk->rows,
           chunk->row_count * sizeof(*out_collection->rows));
    row_index += chunk->row_count;
  }
  IREE_ASSERT(row_index == row_count,
              "AMDGPU sanitizer site chunk row count changed during "
              "finalization");
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_emit_site_table_rodata(
    loom_module_t* module, iree_const_byte_span_t site_table) {
  loom_symbol_ref_t symbol_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_get_or_create_symbol(
      module, IREE_SV(LOOM_SANITIZER_SITE_TABLE_SYMBOL_NAME), &symbol_ref));
  loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  IREE_ASSERT(symbol->defining_op == NULL,
              "AMDGPU sanitizer site table symbol was defined after "
              "function-local reservation");

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  loom_op_t* rodata_op = NULL;
  return loom_global_rodata_def_build(
      &builder, LOOM_GLOBAL_RODATA_DEF_BUILD_FLAG_HAS_ALIGNMENT, symbol_ref, 8,
      site_table, LOOM_LOCATION_UNKNOWN, &rodata_op);
}

iree_status_t loom_amdgpu_finalize_sanitizer_module(
    loom_module_t* module, loom_low_lower_module_state_t* module_state,
    iree_arena_allocator_t* scratch_arena) {
  loom_amdgpu_sanitizer_module_state_t* sanitizer_state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_module_state_from_module_state(
      module_state, &sanitizer_state));
  if (sanitizer_state->site_row_count == 0) {
    return iree_ok_status();
  }

  loom_sanitizer_site_collection_t collection = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_flatten_site_rows(
      sanitizer_state, scratch_arena, &collection));
  iree_const_byte_span_t site_table = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(loom_sanitizer_site_table_encode(
      module, &collection, scratch_arena, &site_table));
  return loom_amdgpu_sanitizer_emit_site_table_rodata(module, site_table);
}

iree_status_t loom_amdgpu_sanitizer_site_id_for_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_sanitizer_site_id_t* out_site_id) {
  *out_site_id = LOOM_SANITIZER_SITE_ID_INVALID;
  loom_amdgpu_sanitizer_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_lower_state(context, &state));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_ensure_site_collection(context, state));
  if (!state->has_site_collection) return iree_ok_status();

  for (iree_host_size_t i = state->next_site_row_index;
       i < state->site_collection.row_count; ++i) {
    const loom_sanitizer_site_row_t* row = &state->site_collection.rows[i];
    if (row->op == source_op) {
      state->next_site_row_index = i + 1;
      *out_site_id = row->site_id;
      return iree_ok_status();
    }
  }

  IREE_ASSERT_UNREACHABLE(
      "sanitizer site ID lookup must follow site collection order");
  IREE_BUILTIN_UNREACHABLE();
}

static bool loom_amdgpu_sanitizer_assert_access_kinds(
    loom_sanitizer_assert_access_kind_t kind,
    loom_low_source_memory_operation_kind_t* out_source_kind,
    loom_amdgpu_sanitizer_access_kind_t* out_report_kind) {
  switch (kind) {
    case LOOM_SANITIZER_ASSERT_ACCESS_KIND_READ:
      *out_source_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
      *out_report_kind = LOOM_AMDGPU_SANITIZER_ACCESS_KIND_READ;
      return true;
    case LOOM_SANITIZER_ASSERT_ACCESS_KIND_WRITE:
      *out_source_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE;
      *out_report_kind = LOOM_AMDGPU_SANITIZER_ACCESS_KIND_WRITE;
      return true;
    case LOOM_SANITIZER_ASSERT_ACCESS_KIND_READ_WRITE:
      *out_source_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_RMW;
      *out_report_kind = LOOM_AMDGPU_SANITIZER_ACCESS_KIND_ATOMIC;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_sanitizer_assert_accesses_kinds(
    loom_sanitizer_assert_accesses_kind_t kind,
    loom_low_source_memory_operation_kind_t* out_source_kind,
    loom_amdgpu_sanitizer_access_kind_t* out_report_kind) {
  switch (kind) {
    case LOOM_SANITIZER_ASSERT_ACCESSES_KIND_READ:
      *out_source_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
      *out_report_kind = LOOM_AMDGPU_SANITIZER_ACCESS_KIND_READ;
      return true;
    case LOOM_SANITIZER_ASSERT_ACCESSES_KIND_WRITE:
      *out_source_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE;
      *out_report_kind = LOOM_AMDGPU_SANITIZER_ACCESS_KIND_WRITE;
      return true;
    case LOOM_SANITIZER_ASSERT_ACCESSES_KIND_READ_WRITE:
      *out_source_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_RMW;
      *out_report_kind = LOOM_AMDGPU_SANITIZER_ACCESS_KIND_ATOMIC;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_sanitizer_access_byte_range(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t view_value_id, loom_attribute_t static_extents,
    loom_type_t* out_base_vector_type, uint32_t* out_access_size) {
  *out_base_vector_type = loom_type_none();
  *out_access_size = 0;
  if (view_value_id >= module->values.count) {
    return false;
  }
  const loom_type_t view_type = loom_module_value_type(module, view_value_id);
  if (!loom_type_is_view(view_type)) {
    return false;
  }

  const loom_type_t base_vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, loom_type_element_type(view_type),
                          loom_dim_pack_static(1), /*encoding_id=*/0);
  loom_vector_memory_access_t vector_access = {0};
  const loom_fact_context_t* fact_context =
      fact_table ? &fact_table->context : NULL;
  if (!loom_vector_memory_access_describe(fact_context, module, view_type,
                                          base_vector_type, &vector_access) ||
      vector_access.static_element_byte_count <= 0) {
    return false;
  }

  int64_t element_count = 1;
  if (!loom_attr_is_absent(static_extents)) {
    const uint8_t view_rank = loom_type_rank(view_type);
    if (static_extents.kind != LOOM_ATTR_I64_ARRAY ||
        static_extents.count != view_rank || view_rank == 0) {
      return false;
    }
    uint8_t range_axis = UINT8_MAX;
    for (uint8_t axis = 0; axis < view_rank; ++axis) {
      const int64_t extent = static_extents.i64_array[axis];
      if (extent <= 0) {
        return false;
      }
      if (extent == 1) continue;
      if (range_axis != UINT8_MAX) {
        return false;
      }
      int64_t axis_stride = 0;
      if (!loom_vector_memory_access_static_axis_stride(&vector_access, axis,
                                                        &axis_stride) ||
          axis_stride != 1) {
        return false;
      }
      range_axis = axis;
      element_count = extent;
    }
  }

  int64_t access_size = 0;
  if (!iree_checked_mul_i64(element_count,
                            vector_access.static_element_byte_count,
                            &access_size) ||
      access_size <= 0 || access_size > UINT32_MAX) {
    return false;
  }
  *out_base_vector_type = base_vector_type;
  *out_access_size = (uint32_t)access_size;
  return true;
}

static bool loom_amdgpu_sanitizer_access_byte_stride(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t view_value_id, loom_attribute_t static_strides,
    uint64_t* out_byte_stride) {
  *out_byte_stride = 0;
  if (view_value_id >= module->values.count) {
    return false;
  }
  const loom_type_t view_type = loom_module_value_type(module, view_value_id);
  if (!loom_type_is_view(view_type)) {
    return false;
  }
  const uint8_t view_rank = loom_type_rank(view_type);
  if (static_strides.kind != LOOM_ATTR_I64_ARRAY ||
      static_strides.count != view_rank) {
    return false;
  }

  const loom_type_t base_vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, loom_type_element_type(view_type),
                          loom_dim_pack_static(1), /*encoding_id=*/0);
  loom_vector_memory_access_t vector_access = {0};
  const loom_fact_context_t* fact_context =
      fact_table ? &fact_table->context : NULL;
  if (!loom_vector_memory_access_describe(fact_context, module, view_type,
                                          base_vector_type, &vector_access) ||
      vector_access.static_element_byte_count <= 0) {
    return false;
  }

  int64_t stride_elements = 0;
  for (uint8_t axis = 0; axis < view_rank; ++axis) {
    const int64_t static_stride = static_strides.i64_array[axis];
    if (static_stride < 0) {
      return false;
    }
    if (static_stride == 0) {
      continue;
    }
    int64_t axis_stride = 0;
    if (!loom_vector_memory_access_static_axis_stride(&vector_access, axis,
                                                      &axis_stride) ||
        axis_stride < 0) {
      return false;
    }
    int64_t stride_term = 0;
    if (!iree_checked_mul_i64(static_stride, axis_stride, &stride_term) ||
        !iree_checked_add_i64(stride_elements, stride_term, &stride_elements)) {
      return false;
    }
  }

  int64_t byte_stride = 0;
  if (!iree_checked_mul_i64(stride_elements,
                            vector_access.static_element_byte_count,
                            &byte_stride) ||
      byte_stride < 0) {
    return false;
  }
  *out_byte_stride = (uint64_t)byte_stride;
  return true;
}

static bool loom_amdgpu_sanitizer_assert_access_plan_build(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions, const loom_op_t* op,
    loom_amdgpu_sanitizer_access_plan_t* out_plan,
    loom_amdgpu_sanitizer_access_diagnostic_t* out_diagnostic) {
  *out_plan = (loom_amdgpu_sanitizer_access_plan_t){
      .site_id = LOOM_SANITIZER_SITE_ID_INVALID,
  };
  *out_diagnostic = (loom_amdgpu_sanitizer_access_diagnostic_t){0};

  loom_low_source_memory_operation_kind_t source_kind =
      LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
  loom_amdgpu_sanitizer_access_kind_t report_kind =
      LOOM_AMDGPU_SANITIZER_ACCESS_KIND_UNKNOWN;
  loom_value_id_t view_value_id = LOOM_VALUE_ID_INVALID;
  loom_value_slice_t indices = {0};
  loom_attribute_t static_indices = loom_attr_absent();
  loom_attribute_t static_extents = loom_attr_absent();
  uint16_t static_repeat_count = 1;
  uint64_t static_repeat_byte_stride = 0;
  if (loom_sanitizer_assert_access_isa(op)) {
    if (!loom_amdgpu_sanitizer_assert_access_kinds(
            loom_sanitizer_assert_access_kind(op), &source_kind,
            &report_kind)) {
      out_diagnostic->sanitizer_constraint_key =
          IREE_SV("target_contract.sanitizer_access.access_kind");
      return false;
    }
    view_value_id = loom_sanitizer_assert_access_view(op);
    indices = loom_sanitizer_assert_access_indices(op);
    static_indices = loom_sanitizer_assert_access_static_indices(op);
    static_extents = loom_sanitizer_assert_access_static_extents(op);
  } else if (loom_sanitizer_assert_accesses_isa(op)) {
    if (!loom_amdgpu_sanitizer_assert_accesses_kinds(
            loom_sanitizer_assert_accesses_kind(op), &source_kind,
            &report_kind)) {
      out_diagnostic->sanitizer_constraint_key =
          IREE_SV("target_contract.sanitizer_access.access_kind");
      return false;
    }
    const int64_t static_count =
        loom_sanitizer_assert_accesses_static_count(op);
    if (static_count <= 0 || static_count > UINT16_MAX) {
      out_diagnostic->sanitizer_constraint_key =
          IREE_SV("target_contract.sanitizer_access.repeat_count");
      return false;
    }
    static_repeat_count = (uint16_t)static_count;
    view_value_id = loom_sanitizer_assert_accesses_view(op);
    indices = loom_sanitizer_assert_accesses_indices(op);
    static_indices = loom_sanitizer_assert_accesses_static_indices(op);
    static_extents = loom_sanitizer_assert_accesses_static_extents(op);
    if (static_repeat_count > 1 &&
        (!loom_amdgpu_sanitizer_access_byte_stride(
             module, fact_table, view_value_id,
             loom_sanitizer_assert_accesses_static_strides(op),
             &static_repeat_byte_stride) ||
         static_repeat_byte_stride == 0)) {
      out_diagnostic->sanitizer_constraint_key =
          IREE_SV("target_contract.sanitizer_access.repeat_stride");
      return false;
    }
  } else {
    out_diagnostic->sanitizer_constraint_key =
        IREE_SV("target_contract.sanitizer_access.op");
    return false;
  }

  loom_type_t base_vector_type = loom_type_none();
  uint32_t access_size = 0;
  if (!loom_amdgpu_sanitizer_access_byte_range(
          module, fact_table, view_value_id, static_extents, &base_vector_type,
          &access_size)) {
    out_diagnostic->sanitizer_constraint_key =
        IREE_SV("target_contract.sanitizer_access.payload_type");
    return false;
  }

  loom_low_source_memory_access_plan_t source = {0};
  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_low_source_memory_access_plan_build_indexed(
          view_regions, source_kind, view_value_id, indices, static_indices,
          base_vector_type, cache_policy, &source, &out_diagnostic->source)) {
    return false;
  }

  if (!loom_amdgpu_memory_access_select_flat_global_address(
          module, &source, &out_plan->address, &out_diagnostic->memory)) {
    return false;
  }

  out_plan->report_access_kind = report_kind;
  out_plan->access_size = access_size;
  out_plan->minimum_alignment = source.minimum_alignment;
  out_plan->static_repeat_count = static_repeat_count;
  out_plan->static_repeat_byte_stride = static_repeat_byte_stride;
  return true;
}

static iree_status_t loom_amdgpu_sanitizer_assert_access_reject(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_amdgpu_sanitizer_access_plan_t* plan,
    const loom_amdgpu_sanitizer_access_diagnostic_t* diagnostic) {
  if (!iree_string_view_is_empty(diagnostic->sanitizer_constraint_key)) {
    return loom_amdgpu_low_legality_reject(
        context, op, diagnostic->sanitizer_constraint_key);
  }
  if (diagnostic->source.rejection_bits != 0) {
    return loom_amdgpu_low_legality_reject(
        context, op,
        loom_low_source_memory_access_rejection_key(
            diagnostic->source.rejection_bits));
  }
  if (diagnostic->memory.rejection_bits != 0) {
    bool handled = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_access_rejection_diagnostic(
        context, op, &plan->address.source, &diagnostic->memory, &handled));
    if (handled) {
      return iree_ok_status();
    }
    return loom_amdgpu_low_legality_reject(
        context, op,
        loom_amdgpu_memory_access_rejection_key(
            diagnostic->memory.rejection_bits));
  }
  return loom_amdgpu_low_legality_reject(
      context, op, IREE_SV("target_contract.sanitizer_access.address"));
}

iree_status_t loom_amdgpu_select_sanitizer_assert_access_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_sanitizer_access_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_sanitizer_access_plan_t){0};
  out_plan->site_id = LOOM_SANITIZER_SITE_ID_INVALID;
  *out_selected = false;
  if (!loom_sanitizer_assert_access_isa(source_op) &&
      !loom_sanitizer_assert_accesses_isa(source_op)) {
    return iree_ok_status();
  }

  loom_amdgpu_sanitizer_access_diagnostic_t diagnostic = {0};
  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_context_view_regions(context, &view_regions));
  if (!loom_amdgpu_sanitizer_assert_access_plan_build(
          loom_low_lower_context_module(context),
          loom_low_lower_context_fact_table(context), view_regions, source_op,
          out_plan, &diagnostic)) {
    return iree_ok_status();
  }
  const loom_sanitizer_reporting_mode_t reporting_mode =
      loom_low_lower_context_sanitizer_reporting_mode(context);
  if (reporting_mode == LOOM_SANITIZER_REPORTING_MODE_DEFAULT ||
      reporting_mode == LOOM_SANITIZER_REPORTING_MODE_REPORT_ONLY) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_site_id_for_op(
        context, source_op, &out_plan->site_id));
    if (out_plan->site_id == LOOM_SANITIZER_SITE_ID_INVALID) {
      return iree_ok_status();
    }
  }
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_sanitizer_assert_access(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  *out_handled = false;
  if (!loom_sanitizer_assert_access_isa(op) &&
      !loom_sanitizer_assert_accesses_isa(op)) {
    return iree_ok_status();
  }
  *out_handled = true;

  loom_amdgpu_sanitizer_access_plan_t plan = {0};
  loom_amdgpu_sanitizer_access_diagnostic_t diagnostic = {0};
  const loom_view_region_table_t* view_regions =
      loom_target_low_legality_view_regions(context);
  if (!loom_amdgpu_sanitizer_assert_access_plan_build(
          loom_target_low_legality_module(context),
          loom_target_low_legality_fact_table(context), view_regions, op, &plan,
          &diagnostic)) {
    return loom_amdgpu_sanitizer_assert_access_reject(context, op, &plan,
                                                      &diagnostic);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_emit_vgpr_u64_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t value, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t sgpr_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_constant_u64(
      context, source_op, value, &sgpr_value));
  return loom_amdgpu_materialize_low_vgpr_b32_registers(context, source_op,
                                                        sgpr_value, out_value);
}

static iree_status_t loom_amdgpu_sanitizer_emit_sgpr_u32_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t value, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, value,
                                    sgpr_type, out_value);
}

static iree_status_t loom_amdgpu_sanitizer_emit_vgpr_u32_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t value, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, value,
                                    vgpr_type, out_value);
}

static loom_amdgpu_sanitizer_access_report_island_t*
loom_amdgpu_sanitizer_island_for_kind(
    loom_amdgpu_sanitizer_lower_state_t* state,
    loom_amdgpu_sanitizer_access_kind_t access_kind, bool** out_has_island) {
  switch (access_kind) {
    case LOOM_AMDGPU_SANITIZER_ACCESS_KIND_READ:
      *out_has_island = &state->has_read_island;
      return &state->read_island;
    case LOOM_AMDGPU_SANITIZER_ACCESS_KIND_WRITE:
      *out_has_island = &state->has_write_island;
      return &state->write_island;
    case LOOM_AMDGPU_SANITIZER_ACCESS_KIND_ATOMIC:
      *out_has_island = &state->has_atomic_island;
      return &state->atomic_island;
    default:
      *out_has_island = NULL;
      return NULL;
  }
}

static iree_status_t loom_amdgpu_sanitizer_get_access_island(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_sanitizer_lower_state_t* state,
    loom_amdgpu_sanitizer_access_kind_t access_kind,
    loom_location_id_t location,
    const loom_amdgpu_sanitizer_access_report_island_t** out_island) {
  *out_island = NULL;
  bool* has_island = NULL;
  loom_amdgpu_sanitizer_access_report_island_t* island =
      loom_amdgpu_sanitizer_island_for_kind(state, access_kind, &has_island);
  if (island == NULL) {
    IREE_ASSERT_UNREACHABLE("unsupported AMDGPU sanitizer access kind");
    IREE_BUILTIN_UNREACHABLE();
  }
  if (!*has_island) {
    loom_builder_ip_t saved_ip = loom_builder_save(builder);
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_access_report_island(
        builder, descriptor_set, saved_ip.block, state->feedback_config_symbol,
        access_kind, LOOM_AMDGPU_SANITIZER_REPORT_FLAG_NONE, location, island));
    loom_builder_restore(builder, saved_ip);
    *has_island = true;
  }
  *out_island = island;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_get_trap_island(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_sanitizer_lower_state_t* state, loom_location_id_t location,
    const loom_amdgpu_sanitizer_trap_island_t** out_island) {
  *out_island = NULL;
  if (!state->has_trap_island) {
    loom_builder_ip_t saved_ip = loom_builder_save(builder);
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_trap_island(
        builder, descriptor_set, saved_ip.block, location,
        &state->trap_island));
    loom_builder_restore(builder, saved_ip);
    state->has_trap_island = true;
  }
  *out_island = &state->trap_island;
  return iree_ok_status();
}

static loom_value_id_t loom_amdgpu_sanitizer_access_view_value(
    const loom_op_t* op) {
  if (loom_sanitizer_assert_access_isa(op)) {
    return loom_sanitizer_assert_access_view(op);
  }
  if (loom_sanitizer_assert_accesses_isa(op)) {
    return loom_sanitizer_assert_accesses_view(op);
  }
  return LOOM_VALUE_ID_INVALID;
}

static void loom_amdgpu_sanitizer_access_plan_for_repeat(
    const loom_amdgpu_sanitizer_access_plan_t* plan, uint16_t repeat_ordinal,
    loom_amdgpu_memory_access_t* out_access) {
  *out_access = plan->address;
  if (repeat_ordinal == 0) {
    return;
  }
  const uint64_t byte_stride = plan->static_repeat_byte_stride;
  IREE_ASSERT(byte_stride == 0 || repeat_ordinal <= UINT64_MAX / byte_stride,
              "AMDGPU sanitizer repeated access byte offset overflow");
  const uint64_t repeat_byte_offset = byte_stride * repeat_ordinal;
  IREE_ASSERT(
      repeat_byte_offset <= UINT64_MAX - out_access->vaddr_static_byte_offset,
      "AMDGPU sanitizer repeated access static offset overflow");
  out_access->vaddr_static_byte_offset += repeat_byte_offset;
}

static iree_status_t loom_amdgpu_sanitizer_emit_repeat_fault_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_access_plan_t* plan,
    loom_value_id_t low_resource, uint16_t repeat_ordinal,
    loom_value_id_t* out_fault_address) {
  *out_fault_address = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_memory_access_t access = {0};
  loom_amdgpu_sanitizer_access_plan_for_repeat(plan, repeat_ordinal, &access);
  return loom_amdgpu_emit_memory_flat_vaddr(context, source_op, &access,
                                            low_resource, out_fault_address);
}

typedef uint32_t loom_amdgpu_sanitizer_repeat_failure_flags_t;

enum loom_amdgpu_sanitizer_repeat_failure_flag_bits_e {
  LOOM_AMDGPU_SANITIZER_REPEAT_FAILURE_FLAG_NONE = 0u,
  LOOM_AMDGPU_SANITIZER_REPEAT_FAILURE_FLAG_SELECT_FAULT_ADDRESS = 1u << 0,
};

typedef struct loom_amdgpu_sanitizer_repeat_failure_summary_t {
  // Native lane mask identifying lanes that failed any repeated access.
  loom_value_id_t failure_mask;
  // VGPRx2 per-lane fault address selected from the first failed repeat.
  loom_value_id_t selected_fault_address;
} loom_amdgpu_sanitizer_repeat_failure_summary_t;

static iree_status_t loom_amdgpu_sanitizer_build_repeat_failure_summary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_access_plan_t* plan,
    loom_value_id_t low_resource,
    const loom_amdgpu_sanitizer_access_config_t* access_config,
    uint32_t wavefront_size, loom_amdgpu_sanitizer_repeat_failure_flags_t flags,
    loom_amdgpu_sanitizer_repeat_failure_summary_t* out_summary) {
  *out_summary = (loom_amdgpu_sanitizer_repeat_failure_summary_t){
      .failure_mask = LOOM_VALUE_ID_INVALID,
      .selected_fault_address = LOOM_VALUE_ID_INVALID,
  };
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const bool select_fault_address = iree_any_bit_set(
      flags, LOOM_AMDGPU_SANITIZER_REPEAT_FAILURE_FLAG_SELECT_FAULT_ADDRESS);
  loom_type_t mask_type = loom_type_none();
  if (select_fault_address) {
    IREE_RETURN_IF_ERROR(loom_low_build_register_type(
        descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &mask_type));
  }
  for (uint16_t i = 0; i < plan->static_repeat_count; ++i) {
    loom_value_id_t fault_address = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_emit_repeat_fault_address(
        context, source_op, plan, low_resource, i, &fault_address));
    loom_value_id_t repeat_failure_mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_build_sanitizer_access_failure_mask_with_config(
            builder, descriptor_set, access_config, fault_address,
            plan->access_size, plan->minimum_alignment, wavefront_size,
            source_op->location, &repeat_failure_mask));
    if (out_summary->failure_mask == LOOM_VALUE_ID_INVALID) {
      out_summary->failure_mask = repeat_failure_mask;
      if (select_fault_address) {
        out_summary->selected_fault_address = fault_address;
      }
    } else {
      if (select_fault_address) {
        loom_value_id_t changed_failure_mask = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_binary(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64,
            repeat_failure_mask, out_summary->failure_mask, mask_type,
            &changed_failure_mask));
        loom_value_id_t first_repeat_failure_mask = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_binary(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
            repeat_failure_mask, changed_failure_mask, mask_type,
            &first_repeat_failure_mask));
        IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_access_vgpr64_select(
            builder, descriptor_set, out_summary->selected_fault_address,
            fault_address, first_repeat_failure_mask, source_op->location,
            &out_summary->selected_fault_address));
      }
      loom_value_id_t union_failure_mask = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_build_sanitizer_access_failure_mask_union(
              builder, descriptor_set, out_summary->failure_mask,
              repeat_failure_mask, source_op->location, &union_failure_mask));
      out_summary->failure_mask = union_failure_mask;
    }
  }
  IREE_ASSERT(out_summary->failure_mask != LOOM_VALUE_ID_INVALID,
              "AMDGPU sanitizer repeated access plan has no accesses");
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_sanitizer_assert_access(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_access_plan_t* plan) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);

  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_amdgpu_sanitizer_access_view_value(source_op),
      &low_resource));

  loom_amdgpu_sanitizer_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_lower_state(context, &state));
  const loom_sanitizer_reporting_mode_t reporting_mode =
      loom_low_lower_context_sanitizer_reporting_mode(context);
  switch (reporting_mode) {
    case LOOM_SANITIZER_REPORTING_MODE_DEFAULT: {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_sanitizer_ensure_config_symbols(context, state));
      break;
    }
    case LOOM_SANITIZER_REPORTING_MODE_REPORT_ONLY: {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_sanitizer_ensure_config_symbols(context, state));
      break;
    }
    case LOOM_SANITIZER_REPORTING_MODE_TRAP: {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_sanitizer_ensure_asan_config_symbol(context, state));
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("unsupported AMDGPU sanitizer reporting mode");
      IREE_BUILTIN_UNREACHABLE();
  }
  const uint32_t wavefront_size =
      loom_amdgpu_target_wavefront_size(loom_low_lower_context_bundle(context));
  loom_amdgpu_sanitizer_access_config_t access_config = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_access_config(
      builder, descriptor_set, state->asan_config_symbol, source_op->location,
      &access_config));
  const loom_amdgpu_sanitizer_repeat_failure_flags_t repeat_failure_flags =
      reporting_mode == LOOM_SANITIZER_REPORTING_MODE_TRAP
          ? LOOM_AMDGPU_SANITIZER_REPEAT_FAILURE_FLAG_NONE
          : LOOM_AMDGPU_SANITIZER_REPEAT_FAILURE_FLAG_SELECT_FAULT_ADDRESS;
  loom_amdgpu_sanitizer_repeat_failure_summary_t repeat_failure = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_build_repeat_failure_summary(
      context, source_op, plan, low_resource, &access_config, wavefront_size,
      repeat_failure_flags, &repeat_failure));

  if (reporting_mode == LOOM_SANITIZER_REPORTING_MODE_TRAP) {
    loom_amdgpu_sanitizer_access_report_failure_branch_t branch = {0};
    const loom_amdgpu_sanitizer_trap_island_t* island = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_get_trap_island(
        builder, descriptor_set, state, source_op->location, &island));
    return loom_amdgpu_build_sanitizer_trap_failure_mask_branch_to_island(
        builder, descriptor_set, island, repeat_failure.failure_mask,
        source_op->location, &branch);
  }

  loom_amdgpu_sanitizer_access_report_failure_branch_t branch = {0};
  const loom_amdgpu_sanitizer_access_report_island_t* island = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_get_access_island(
      builder, descriptor_set, state, plan->report_access_kind,
      source_op->location, &island));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_build_sanitizer_access_report_failure_mask_split(
          builder, descriptor_set, repeat_failure.failure_mask,
          source_op->location, &branch));

  loom_amdgpu_feedback_packet_source_t source = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_constant_u64(
      context, source_op, 0, &source.dispatch_ptr));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_emit_sgpr_u32_constant(
      context, source_op, 0, &source.workgroup_id_x));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_emit_vgpr_u32_constant(
      context, source_op, 0, &source.workitem_id_x));
  loom_value_id_t report_access_size = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_emit_vgpr_u64_constant(
      context, source_op, plan->access_size, &report_access_size));
  loom_value_id_t report_site_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_emit_vgpr_u64_constant(
      context, source_op, plan->site_id, &report_site_id));

  loom_amdgpu_sanitizer_access_check_t report_check = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_access_check_with_config(
      builder, descriptor_set, &access_config,
      repeat_failure.selected_fault_address, plan->access_size,
      plan->minimum_alignment, wavefront_size, source_op->location,
      &report_check));
  loom_amdgpu_sanitizer_access_report_t report = {
      .access_kind = plan->report_access_kind,
      .flags = LOOM_AMDGPU_SANITIZER_REPORT_FLAG_NONE,
      .fault_address = repeat_failure.selected_fault_address,
      .access_size = report_access_size,
      .site_id = report_site_id,
      .shadow_address = report_check.shadow_address,
      .shadow_value = report_check.shadow_value,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_access_report_branch(
      builder, descriptor_set, island, &source, &report, source_op->location));

  loom_builder_set_block(builder, branch.continuation_block);
  return iree_ok_status();
}
