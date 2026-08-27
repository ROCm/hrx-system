// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/parameters.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/command/ops.h"
#include "loom/util/walk.h"

enum {
  // Parameter starts use a deterministic source-order layout with enough
  // alignment for wide device loads and direct transfer operations.
  LOOM_CMD_PARAMETER_DEFAULT_ALIGNMENT = 256,
};

typedef struct loom_cmd_parameter_source_row_t {
  // command.parameter result mapped by the resulting placement row.
  loom_value_id_t result_value;

  // Source command-program launch-binding ordinal selecting the root group.
  uint16_t source_binding_ordinal;

  // Scratch-owned fully substituted parameter key.
  iree_string_view_t key;

  // Exact byte length derived from the typed result view.
  uint64_t byte_length;

  // Minimum placement alignment derived from view facts and policy.
  uint64_t minimum_alignment;
} loom_cmd_parameter_source_row_t;

typedef struct loom_cmd_parameter_source_range_t {
  // Source SSA value mapped by the resolved range.
  loom_value_id_t source_value;

  // Source command-program launch-binding ordinal owning the storage root.
  uint16_t source_binding_ordinal;

  // Byte offset relative to the source launch-binding range.
  uint64_t byte_offset;

  // Exact byte length of the derived range.
  uint64_t byte_length;
} loom_cmd_parameter_source_range_t;

typedef struct loom_cmd_parameter_build_t {
  // Immutable source module containing the command program.
  const loom_module_t* module;

  // Source program launch-binding values in signature order.
  const loom_value_id_t* binding_values;

  // Number of entries in |binding_values|.
  uint16_t binding_count;

  // Complete value facts for the source program.
  const loom_value_fact_table_t* fact_table;

  // Scratch storage for source rows and formatted keys.
  iree_arena_allocator_t* scratch_arena;

  // Parameter source rows in source traversal order.
  loom_cmd_parameter_source_row_t* rows;

  // Number of populated entries in |rows|.
  iree_host_size_t row_count;

  // Number of allocated entries in |rows|.
  iree_host_size_t row_capacity;

  // Total bytes required to persist every concrete key.
  iree_host_size_t key_storage_length;

  // Exact non-parameter view ranges in source traversal order.
  loom_cmd_parameter_source_range_t* buffer_ranges;

  // Number of populated entries in |buffer_ranges|.
  iree_host_size_t buffer_range_count;

  // Number of allocated entries in |buffer_ranges|.
  iree_host_size_t buffer_range_capacity;
} loom_cmd_parameter_build_t;

static iree_host_size_t loom_cmd_parameter_decimal_length(uint64_t value) {
  iree_host_size_t length = 1;
  while (value >= 10) {
    value /= 10;
    ++length;
  }
  return length;
}

static char* loom_cmd_parameter_write_decimal(uint64_t value, char* target,
                                              iree_host_size_t length) {
  char* cursor = target + length;
  do {
    *--cursor = (char)('0' + value % 10);
    value /= 10;
  } while (cursor != target);
  return target + length;
}

static iree_status_t loom_cmd_parameter_format_key(
    loom_cmd_parameter_build_t* build, const loom_op_t* op,
    iree_string_view_t* out_key) {
  *out_key = iree_string_view_empty();
  const iree_string_view_t pattern =
      build->module->strings.entries[loom_command_parameter_pattern(op)];
  const loom_value_slice_t substitutions =
      loom_command_parameter_substitutions(op);

  iree_host_size_t key_length = pattern.size;
  uint64_t* substitution_values = NULL;
  iree_host_size_t* substitution_lengths = NULL;
  if (substitutions.count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        build->scratch_arena, substitutions.count, sizeof(*substitution_values),
        (void**)&substitution_values));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        build->scratch_arena, substitutions.count,
        sizeof(*substitution_lengths), (void**)&substitution_lengths));
  }
  for (uint16_t i = 0; i < substitutions.count; ++i) {
    int64_t exact_value = 0;
    if (!loom_value_facts_as_exact_i64(
            loom_value_fact_table_lookup(build->fact_table,
                                         substitutions.values[i]),
            &exact_value) ||
        exact_value < 0) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "command parameter substitution %" PRIu16
          " must be an exact nonnegative index before program preparation",
          i);
    }
    substitution_values[i] = (uint64_t)exact_value;
    substitution_lengths[i] =
        loom_cmd_parameter_decimal_length((uint64_t)exact_value);
    IREE_ASSERT_GE(key_length, 2u);
    key_length -= 2;
    if (!iree_host_size_checked_add(key_length, substitution_lengths[i],
                                    &key_length)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "command parameter key is too large");
    }
  }

  char* key_data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(build->scratch_arena, key_length, (void**)&key_data));
  const char* pattern_cursor = pattern.data;
  const char* pattern_end = pattern.data + pattern.size;
  char* key_cursor = key_data;
  uint16_t substitution_ordinal = 0;
  while (pattern_cursor != pattern_end) {
    if (*pattern_cursor != '{') {
      *key_cursor++ = *pattern_cursor++;
      continue;
    }
    IREE_ASSERT_LT(substitution_ordinal, substitutions.count);
    IREE_ASSERT_LT(pattern_cursor + 1, pattern_end);
    IREE_ASSERT_EQ(pattern_cursor[1], '}');
    key_cursor = loom_cmd_parameter_write_decimal(
        substitution_values[substitution_ordinal], key_cursor,
        substitution_lengths[substitution_ordinal]);
    ++substitution_ordinal;
    pattern_cursor += 2;
  }
  IREE_ASSERT_EQ(substitution_ordinal, substitutions.count);
  IREE_ASSERT_EQ(key_cursor, key_data + key_length);
  *out_key = iree_make_string_view(key_data, key_length);
  return iree_ok_status();
}

static uint16_t loom_cmd_parameter_find_root_binding(
    const loom_cmd_parameter_build_t* build, loom_value_id_t root_value) {
  for (uint16_t i = 0; i < build->binding_count; ++i) {
    if (build->binding_values[i] == root_value) return i;
  }
  return UINT16_MAX;
}

static uint16_t loom_cmd_parameter_find_source_binding(
    const loom_cmd_parameter_build_t* build, loom_value_id_t source_value) {
  loom_value_fact_buffer_reference_t reference = {0};
  const bool has_reference = loom_value_facts_query_buffer_reference(
      &build->fact_table->context,
      loom_value_fact_table_lookup(build->fact_table, source_value),
      &reference);
  IREE_ASSERT(has_reference, "verified buffer values have reference facts");
  const loom_value_id_t root_value =
      loom_value_fact_buffer_reference_resolve_root_value(reference,
                                                          source_value);
  return loom_cmd_parameter_find_root_binding(build, root_value);
}

static iree_status_t loom_cmd_parameter_reserve_row(
    loom_cmd_parameter_build_t* build) {
  if (build->row_count < build->row_capacity) return iree_ok_status();
  return iree_arena_grow_array(build->scratch_arena, build->row_count,
                               iree_max(build->row_count + 1, 16u),
                               sizeof(*build->rows), &build->row_capacity,
                               (void**)&build->rows);
}

static iree_status_t loom_cmd_parameter_reserve_buffer_range(
    loom_cmd_parameter_build_t* build) {
  if (build->buffer_range_count < build->buffer_range_capacity) {
    return iree_ok_status();
  }
  return iree_arena_grow_array(build->scratch_arena, build->buffer_range_count,
                               iree_max(build->buffer_range_count + 1, 16u),
                               sizeof(*build->buffer_ranges),
                               &build->buffer_range_capacity,
                               (void**)&build->buffer_ranges);
}

static iree_status_t loom_cmd_parameter_append_exact_view_ranges(
    loom_cmd_parameter_build_t* build, const loom_op_t* op) {
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_value_id_t result = results[i];
    loom_value_fact_view_reference_t view_reference = {0};
    if (!loom_value_facts_query_view_reference(
            &build->fact_table->context,
            loom_value_fact_table_lookup(build->fact_table, result),
            &view_reference)) {
      continue;
    }
    int64_t exact_byte_offset = 0;
    int64_t exact_byte_length = 0;
    if (!loom_value_facts_as_exact_i64(view_reference.base_byte_offset,
                                       &exact_byte_offset) ||
        !loom_value_facts_as_exact_i64(view_reference.footprint_byte_length,
                                       &exact_byte_length) ||
        exact_byte_offset < 0 || exact_byte_length < 0) {
      continue;
    }
    const uint16_t source_binding_ordinal =
        loom_cmd_parameter_find_root_binding(build,
                                             view_reference.root_value_id);
    if (source_binding_ordinal == UINT16_MAX) continue;
    IREE_RETURN_IF_ERROR(loom_cmd_parameter_reserve_buffer_range(build));
    build->buffer_ranges[build->buffer_range_count++] =
        (loom_cmd_parameter_source_range_t){
            .source_value = result,
            .source_binding_ordinal = source_binding_ordinal,
            .byte_offset = (uint64_t)exact_byte_offset,
            .byte_length = (uint64_t)exact_byte_length,
        };
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_parameter_visit(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_cmd_parameter_build_t* build = (loom_cmd_parameter_build_t*)user_data;
  if (!loom_command_parameter_isa(op)) {
    return loom_cmd_parameter_append_exact_view_ranges(build, op);
  }

  const uint16_t source_binding_ordinal =
      loom_cmd_parameter_find_source_binding(build,
                                             loom_command_parameter_source(op));
  if (source_binding_ordinal == UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "command.parameter source must resolve to a command-program launch "
        "binding");
  }

  const loom_value_id_t result_value = loom_command_parameter_result(op);
  loom_value_fact_view_reference_t view_reference = {0};
  const bool has_view_reference = loom_value_facts_query_view_reference(
      &build->fact_table->context,
      loom_value_fact_table_lookup(build->fact_table, result_value),
      &view_reference);
  IREE_ASSERT(has_view_reference,
              "command.parameter results have typed view-reference facts");
  int64_t exact_byte_length = 0;
  if (!loom_value_facts_as_exact_i64(view_reference.footprint_byte_length,
                                     &exact_byte_length) ||
      exact_byte_length < 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "command parameter view must have an exact byte footprint before "
        "program preparation");
  }

  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_cmd_parameter_format_key(build, op, &key));
  IREE_RETURN_IF_ERROR(loom_cmd_parameter_reserve_row(build));
  build->rows[build->row_count++] = (loom_cmd_parameter_source_row_t){
      .result_value = result_value,
      .source_binding_ordinal = source_binding_ordinal,
      .key = key,
      .byte_length = (uint64_t)exact_byte_length,
      .minimum_alignment =
          iree_max(view_reference.minimum_alignment,
                   (uint64_t)LOOM_CMD_PARAMETER_DEFAULT_ALIGNMENT),
  };
  if (!iree_host_size_checked_add(build->key_storage_length, key.size,
                                  &build->key_storage_length)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command parameter key table is too large");
  }
  return iree_ok_status();
}

static bool loom_cmd_parameter_align_offset(uint64_t value, uint64_t alignment,
                                            uint64_t* out_value) {
  IREE_ASSERT(iree_is_power_of_two_uint64(alignment));
  const uint64_t mask = alignment - 1;
  if (value > UINT64_MAX - mask) return false;
  *out_value = (value + mask) & ~mask;
  return true;
}

static iree_status_t loom_cmd_parameter_allocate_requirement_table(
    const loom_cmd_parameter_build_t* build, iree_host_size_t root_count,
    iree_allocator_t host_allocator,
    loom_cmd_parameter_requirement_table_t* table) {
  iree_host_size_t root_table_size = 0;
  iree_host_size_t entry_table_size = 0;
  if (!iree_host_size_checked_mul(root_count, sizeof(*table->roots),
                                  &root_table_size) ||
      !iree_host_size_checked_mul(build->row_count, sizeof(*table->entries),
                                  &entry_table_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command parameter requirement table is too large");
  }

  table->root_count = root_count;
  table->count = build->row_count;
  iree_status_t status = iree_ok_status();
  if (root_table_size != 0) {
    status = iree_allocator_malloc(host_allocator, root_table_size,
                                   (void**)&table->roots);
  }
  if (iree_status_is_ok(status) && entry_table_size != 0) {
    status = iree_allocator_malloc(host_allocator, entry_table_size,
                                   (void**)&table->entries);
  }
  if (iree_status_is_ok(status) && build->key_storage_length != 0) {
    status = iree_allocator_malloc(host_allocator, build->key_storage_length,
                                   (void**)&table->key_storage);
  }
  if (!iree_status_is_ok(status)) {
    loom_cmd_parameter_requirement_table_deinitialize(table, host_allocator);
  }
  return status;
}

iree_status_t loom_cmd_parameter_layout_build(
    const loom_module_t* module, loom_func_like_t program,
    const loom_value_fact_table_t* fact_table,
    iree_arena_allocator_t* scratch_arena, iree_allocator_t host_allocator,
    loom_cmd_buffer_binding_t* bindings, iree_host_size_t binding_count,
    loom_cmd_parameter_requirement_table_t* out_requirements,
    loom_cmd_parameter_layout_t* out_layout) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT(loom_func_like_isa(program));
  IREE_ASSERT_ARGUMENT(fact_table);
  IREE_ASSERT_ARGUMENT(scratch_arena);
  IREE_ASSERT(binding_count == 0 || bindings != NULL);
  IREE_ASSERT_ARGUMENT(out_requirements);
  IREE_ASSERT_ARGUMENT(out_layout);
  memset(out_requirements, 0, sizeof(*out_requirements));
  memset(out_layout, 0, sizeof(*out_layout));

  uint16_t argument_count = 0;
  const loom_value_id_t* argument_values =
      loom_func_like_arg_ids(program, &argument_count);
  const int64_t specialization_count_i64 =
      loom_func_like_specialization_count(program);
  IREE_ASSERT_GE(specialization_count_i64, 0);
  IREE_ASSERT_LE(specialization_count_i64, argument_count);
  const uint16_t specialization_count = (uint16_t)specialization_count_i64;
  IREE_ASSERT_EQ(binding_count, argument_count - specialization_count);

  loom_cmd_parameter_build_t build = {
      .module = module,
      .binding_values =
          binding_count != 0 ? argument_values + specialization_count : NULL,
      .binding_count = (uint16_t)binding_count,
      .fact_table = fact_table,
      .scratch_arena = scratch_arena,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_RETURN_IF_ERROR(loom_walk_function(
      module, program, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_cmd_parameter_visit, &build}, scratch_arena,
      &walk_result));
  IREE_ASSERT_EQ(walk_result, LOOM_WALK_CONTINUE);

  bool* fixed_bindings = NULL;
  uint64_t* placement_cursors = NULL;
  if (binding_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, binding_count,
                                                   sizeof(*fixed_bindings),
                                                   (void**)&fixed_bindings));
    memset(fixed_bindings, 0, binding_count * sizeof(*fixed_bindings));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, binding_count,
                                                   sizeof(*placement_cursors),
                                                   (void**)&placement_cursors));
    memset(placement_cursors, 0, binding_count * sizeof(*placement_cursors));
  }
  for (iree_host_size_t i = 0; i < build.row_count; ++i) {
    fixed_bindings[build.rows[i].source_binding_ordinal] = true;
  }

  uint32_t fixed_buffer_count = 0;
  uint32_t rebindable_binding_count = 0;
  for (uint16_t i = 0; i < binding_count; ++i) {
    const loom_cmd_buffer_role_t role = fixed_bindings[i]
                                            ? LOOM_CMD_BUFFER_ROLE_FIXED
                                            : LOOM_CMD_BUFFER_ROLE_REBINDABLE;
    const uint32_t resource_index =
        fixed_bindings[i] ? fixed_buffer_count++ : rebindable_binding_count++;
    bindings[i] = (loom_cmd_buffer_binding_t){
        .role = role,
        .resource_index = resource_index,
        .byte_offset = 0,
        .byte_length = UINT64_MAX,
    };
  }

  iree_host_size_t required_buffer_range_count = 0;
  if (!iree_host_size_checked_add(build.buffer_range_count, build.row_count,
                                  &required_buffer_range_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command buffer range table is too large");
  }
  if (required_buffer_range_count > build.buffer_range_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        scratch_arena, build.buffer_range_count, required_buffer_range_count,
        sizeof(*build.buffer_ranges), &build.buffer_range_capacity,
        (void**)&build.buffer_ranges));
  }
  loom_cmd_buffer_range_t* resolved_buffer_ranges = NULL;
  if (required_buffer_range_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, required_buffer_range_count,
        sizeof(*resolved_buffer_ranges), (void**)&resolved_buffer_ranges));
  }
  IREE_RETURN_IF_ERROR(loom_cmd_parameter_allocate_requirement_table(
      &build, fixed_buffer_count, host_allocator, out_requirements));

  for (uint16_t i = 0; i < binding_count; ++i) {
    if (!fixed_bindings[i]) continue;
    const uint32_t fixed_buffer_index = bindings[i].resource_index;
    out_requirements->roots[fixed_buffer_index] =
        (loom_cmd_parameter_root_requirement_t){
            .source_binding_ordinal = i,
            .fixed_buffer_index = fixed_buffer_index,
            .required_byte_length = 0,
            .minimum_alignment = LOOM_CMD_PARAMETER_DEFAULT_ALIGNMENT,
        };
  }

  char* key_cursor = out_requirements->key_storage;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < build.row_count && iree_status_is_ok(status);
       ++i) {
    const loom_cmd_parameter_source_row_t* source = &build.rows[i];
    uint64_t byte_offset = 0;
    if (!loom_cmd_parameter_align_offset(
            placement_cursors[source->source_binding_ordinal],
            source->minimum_alignment, &byte_offset) ||
        source->byte_length > UINT64_MAX - byte_offset) {
      status = iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "command parameter placement exceeds the 64-bit buffer range");
      break;
    }
    const uint64_t byte_end = byte_offset + source->byte_length;
    placement_cursors[source->source_binding_ordinal] = byte_end;

    const uint32_t fixed_buffer_index =
        bindings[source->source_binding_ordinal].resource_index;
    loom_cmd_parameter_root_requirement_t* root =
        &out_requirements->roots[fixed_buffer_index];
    root->required_byte_length = iree_max(root->required_byte_length, byte_end);
    root->minimum_alignment =
        iree_max(root->minimum_alignment, source->minimum_alignment);

    memcpy(key_cursor, source->key.data, source->key.size);
    const iree_string_view_t owned_key =
        iree_make_string_view(key_cursor, source->key.size);
    key_cursor += source->key.size;
    out_requirements->entries[i] = (loom_cmd_parameter_requirement_t){
        .key = owned_key,
        .source_binding_ordinal = source->source_binding_ordinal,
        .fixed_buffer_index = fixed_buffer_index,
        .byte_offset = byte_offset,
        .byte_length = source->byte_length,
        .minimum_alignment = source->minimum_alignment,
    };
    build.buffer_ranges[build.buffer_range_count++] =
        (loom_cmd_parameter_source_range_t){
            .source_value = source->result_value,
            .source_binding_ordinal = source->source_binding_ordinal,
            .byte_offset = byte_offset,
            .byte_length = source->byte_length,
        };
  }

  if (!iree_status_is_ok(status)) {
    loom_cmd_parameter_requirement_table_deinitialize(out_requirements,
                                                      host_allocator);
    return status;
  }
  if (build.key_storage_length != 0) {
    IREE_ASSERT_EQ(key_cursor,
                   out_requirements->key_storage + build.key_storage_length);
  }
  IREE_ASSERT_EQ(build.buffer_range_count, required_buffer_range_count);
  for (iree_host_size_t i = 0; i < build.buffer_range_count; ++i) {
    const loom_cmd_parameter_source_range_t* source = &build.buffer_ranges[i];
    IREE_ASSERT_LT(source->source_binding_ordinal, binding_count);
    const loom_cmd_buffer_binding_t binding =
        bindings[source->source_binding_ordinal];
    if (binding.role == LOOM_CMD_BUFFER_ROLE_FIXED) {
      loom_cmd_parameter_root_requirement_t* root =
          &out_requirements->roots[binding.resource_index];
      root->required_byte_length =
          iree_max(root->required_byte_length,
                   source->byte_offset + source->byte_length);
    }
    IREE_ASSERT_LE(binding.byte_offset, UINT64_MAX - source->byte_offset);
    if (binding.byte_length != UINT64_MAX) {
      IREE_ASSERT_LE(source->byte_offset, binding.byte_length);
      IREE_ASSERT_LE(source->byte_length,
                     binding.byte_length - source->byte_offset);
    }
    resolved_buffer_ranges[i] = (loom_cmd_buffer_range_t){
        .source_value = source->source_value,
        .role = binding.role,
        .resource_index = binding.resource_index,
        .byte_offset = binding.byte_offset + source->byte_offset,
        .byte_length = source->byte_length,
    };
  }
  *out_layout = (loom_cmd_parameter_layout_t){
      .buffer_ranges = resolved_buffer_ranges,
      .buffer_range_count = build.buffer_range_count,
      .fixed_buffer_count = fixed_buffer_count,
      .rebindable_binding_count = rebindable_binding_count,
  };
  return iree_ok_status();
}

void loom_cmd_parameter_requirement_table_deinitialize(
    loom_cmd_parameter_requirement_table_t* table,
    iree_allocator_t host_allocator) {
  if (!table) return;
  iree_allocator_free(host_allocator, table->key_storage);
  iree_allocator_free(host_allocator, table->entries);
  iree_allocator_free(host_allocator, table->roots);
  memset(table, 0, sizeof(*table));
}
