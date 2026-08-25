// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/lower.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/builder.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/cmd/abi_layout.h"
#include "loom/target/arch/cmd/descriptors/descriptors.h"
#include "loom/target/types.h"

typedef struct loom_cmd_lower_constant_t {
  // Source constant bit pattern used to deduplicate materializations.
  uint64_t value;
  // Low SSA value produced by the materialized constant.
  loom_value_id_t low_value;
} loom_cmd_lower_constant_t;

typedef struct loom_cmd_lower_constant_table_t {
  // Arena-owned constant entries.
  loom_cmd_lower_constant_t* entries;
  // Number of populated entries.
  iree_host_size_t count;
  // Number of allocated entries.
  iree_host_size_t capacity;
} loom_cmd_lower_constant_table_t;

typedef struct loom_cmd_lower_types_t {
  // Interned attribute name used by low constant descriptors.
  loom_string_id_t value_attr_name;
  // Source buffer type used by buffer resource imports.
  loom_type_id_t buffer_source;
  // Source index type used by executable and entry resource imports.
  loom_type_id_t index_source;
  // Portable unsigned 32-bit register type.
  loom_type_t u32;
  // Portable unsigned 64-bit register type.
  loom_type_t u64;
  // Portable tagless 8-bit argument register type.
  loom_type_t b8;
  // Portable tagless 16-bit argument register type.
  loom_type_t b16;
  // Portable tagless 32-bit argument register type.
  loom_type_t b32;
  // Portable tagless 64-bit argument register type.
  loom_type_t b64;
  // Fixed buffer resource register type.
  loom_type_t fixed_buffer;
  // Issue-time binding resource register type.
  loom_type_t binding;
  // Resolved buffer-range register type.
  loom_type_t buffer_ref;
  // Loaded executable resource register type.
  loom_type_t executable;
  // Executable-local entry-token resource register type.
  loom_type_t entry;
} loom_cmd_lower_types_t;

typedef struct loom_cmd_lower_buffer_tuple_t {
  // Fixed or rebindable buffer root.
  loom_value_id_t root;
  // Root-relative byte offset.
  loom_value_id_t byte_offset;
  // Byte length or the remaining-range sentinel.
  loom_value_id_t byte_length;
} loom_cmd_lower_buffer_tuple_t;

// Prepared low representations associated with one source SSA value.
typedef struct loom_cmd_lower_source_value_t {
  // Resolved buffer tuple when the source is a buffer or view.
  loom_cmd_lower_buffer_tuple_t buffer;
  // Materialized low scalar constant when the source is an exact scalar.
  loom_value_id_t scalar;
} loom_cmd_lower_source_value_t;

typedef struct loom_cmd_lower_resources_t {
  // Prepared representations indexed directly by source value ID.
  loom_cmd_lower_source_value_t* source_values;
  // Number of entries in |source_values|.
  iree_host_size_t source_value_count;
  // Fixed buffer resources indexed by plan resource ordinal.
  loom_value_id_t* fixed_buffers;
  // Rebindable resources indexed by issue-time binding slot.
  loom_value_id_t* bindings;
  // Buffer references indexed by aggregate host launch tuple ordinal.
  loom_value_id_t* launch_counts;
  // Executable resources indexed by package executable ordinal.
  loom_value_id_t* executables;
  // Entry-token resources indexed by program entry ordinal.
  loom_value_id_t* entries;
} loom_cmd_lower_resources_t;

typedef struct loom_cmd_lower_state_t {
  // Module containing both the source and replacement operations.
  loom_module_t* module;
  // Source command program being converted.
  loom_func_like_t source_program;
  // Compiler-owned placement and dispatch facts.
  const loom_cmd_lower_plan_t* plan;
  // Generated descriptor set for the portable command ISA.
  const loom_low_descriptor_set_t* descriptor_set;
  // Scratch storage discarded after conversion.
  iree_arena_allocator_t* scratch_arena;
  // Builder positioned in the replacement low function.
  loom_builder_t builder;
  // Partially or completely built replacement function.
  loom_op_t* low_function;
  // Interned source and portable register types.
  loom_cmd_lower_types_t types;
  // Source-to-low value map and imported ABI resources.
  loom_cmd_lower_resources_t resources;
  // Deduplicated portable constant tables grouped by representation.
  struct {
    // Unsigned 32-bit structural constants.
    loom_cmd_lower_constant_table_t u32;
    // Unsigned 64-bit structural constants.
    loom_cmd_lower_constant_table_t u64;
  } constants;
} loom_cmd_lower_state_t;

static const loom_low_descriptor_t* loom_cmd_lower_descriptor(
    const loom_cmd_lower_state_t* state, uint32_t descriptor_ordinal) {
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(state->descriptor_set,
                                            descriptor_ordinal);
  IREE_ASSERT(descriptor != NULL, "generated cmd descriptor refs exist");
  return descriptor;
}

static iree_status_t loom_cmd_lower_initialize_types(
    loom_cmd_lower_state_t* state) {
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      state->descriptor_set, CMD_CORE_REG_CLASS_ID_U32, 1, &state->types.u32));
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      state->descriptor_set, CMD_CORE_REG_CLASS_ID_U64, 1, &state->types.u64));
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      state->descriptor_set, CMD_CORE_REG_CLASS_ID_BUFFER, 1,
      &state->types.fixed_buffer));
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      state->descriptor_set, CMD_CORE_REG_CLASS_ID_BINDING, 1,
      &state->types.binding));
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      state->descriptor_set, CMD_CORE_REG_CLASS_ID_BUFFER_REF, 1,
      &state->types.buffer_ref));
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      state->descriptor_set, CMD_CORE_REG_CLASS_ID_EXECUTABLE, 1,
      &state->types.executable));
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(state->descriptor_set,
                                                    CMD_CORE_REG_CLASS_ID_ENTRY,
                                                    1, &state->types.entry));
  IREE_RETURN_IF_ERROR(loom_module_intern_type_id(
      state->module, loom_type_buffer(), &state->types.buffer_source));
  IREE_RETURN_IF_ERROR(loom_module_intern_type_id(
      state->module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      &state->types.index_source));
  return loom_module_intern_string(state->module, IREE_SV("value"),
                                   &state->types.value_attr_name);
}

static iree_status_t loom_cmd_lower_build_resource(
    loom_cmd_lower_state_t* state, uint32_t index, loom_type_id_t source_type,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_op_t* resource_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_resource_build(
      &state->builder, /*build_flags=*/0,
      LOOM_LOW_RESOURCE_IMPORT_KIND_COMMAND_INPUT, LOOM_VALUE_ID_INVALID,
      (int64_t)index, source_type, /*extent=*/0,
      /*cache_swizzle_stride=*/0, result_type, location, &resource_op));
  *out_value = loom_low_resource_result(resource_op);
  return iree_ok_status();
}

static iree_status_t loom_cmd_lower_reserve_constant(
    loom_cmd_lower_state_t* state, loom_cmd_lower_constant_table_t* table) {
  if (table->count < table->capacity) return iree_ok_status();
  return iree_arena_grow_array(
      state->scratch_arena, table->count, iree_max(table->count + 1, 8u),
      sizeof(*table->entries), &table->capacity, (void**)&table->entries);
}

static int64_t loom_cmd_lower_u64_attr_value(uint64_t value) {
  int64_t attr_value = 0;
  memcpy(&attr_value, &value, sizeof(attr_value));
  return attr_value;
}

static iree_status_t loom_cmd_lower_emit_constant(loom_cmd_lower_state_t* state,
                                                  uint64_t value,
                                                  uint32_t descriptor_ordinal,
                                                  loom_type_t result_type,
                                                  loom_location_id_t location,
                                                  loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_named_attr_t value_attr = {
      .name_id = state->types.value_attr_name,
      .value = loom_attr_i64(loom_cmd_lower_u64_attr_value(value)),
  };
  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      &state->builder, state->descriptor_set,
      loom_cmd_lower_descriptor(state, descriptor_ordinal),
      loom_make_named_attr_slice(&value_attr, 1), result_type, location,
      &constant_op));
  *out_value = loom_low_const_result(constant_op);
  return iree_ok_status();
}

static iree_status_t loom_cmd_lower_build_constant(
    loom_cmd_lower_state_t* state, uint64_t value, uint32_t descriptor_ordinal,
    loom_type_t result_type, loom_cmd_lower_constant_table_t* table,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  for (iree_host_size_t i = 0; i < table->count; ++i) {
    if (table->entries[i].value == value) {
      *out_value = table->entries[i].low_value;
      return iree_ok_status();
    }
  }

  IREE_RETURN_IF_ERROR(loom_cmd_lower_reserve_constant(state, table));
  IREE_RETURN_IF_ERROR(loom_cmd_lower_emit_constant(
      state, value, descriptor_ordinal, result_type, location, out_value));
  table->entries[table->count++] = (loom_cmd_lower_constant_t){
      .value = value,
      .low_value = *out_value,
  };
  return iree_ok_status();
}

static iree_status_t loom_cmd_lower_build_u32_constant(
    loom_cmd_lower_state_t* state, uint32_t value, loom_location_id_t location,
    loom_value_id_t* out_value) {
  return loom_cmd_lower_build_constant(
      state, value, CMD_CORE_DESCRIPTOR_REF_CONSTANT_U32, state->types.u32,
      &state->constants.u32, location, out_value);
}

static iree_status_t loom_cmd_lower_build_u64_constant(
    loom_cmd_lower_state_t* state, uint64_t value, loom_location_id_t location,
    loom_value_id_t* out_value) {
  return loom_cmd_lower_build_constant(
      state, value, CMD_CORE_DESCRIPTOR_REF_CONSTANT_U64, state->types.u64,
      &state->constants.u64, location, out_value);
}

static iree_status_t loom_cmd_lower_build_dispatch_argument_constant(
    loom_cmd_lower_state_t* state,
    const loom_cmd_lower_dispatch_argument_t* argument,
    loom_location_id_t location, loom_value_id_t* out_value) {
  uint32_t descriptor_ordinal = 0;
  loom_type_t* cached_type = NULL;
  uint16_t reg_class_id = 0;
  switch (argument->kind) {
    case LOOM_CMD_LOWER_DISPATCH_ARGUMENT_KIND_B8:
      descriptor_ordinal = CMD_CORE_DESCRIPTOR_REF_CONSTANT_B8;
      cached_type = &state->types.b8;
      reg_class_id = CMD_CORE_REG_CLASS_ID_B8;
      break;
    case LOOM_CMD_LOWER_DISPATCH_ARGUMENT_KIND_B16:
      descriptor_ordinal = CMD_CORE_DESCRIPTOR_REF_CONSTANT_B16;
      cached_type = &state->types.b16;
      reg_class_id = CMD_CORE_REG_CLASS_ID_B16;
      break;
    case LOOM_CMD_LOWER_DISPATCH_ARGUMENT_KIND_B32:
      descriptor_ordinal = CMD_CORE_DESCRIPTOR_REF_CONSTANT_B32;
      cached_type = &state->types.b32;
      reg_class_id = CMD_CORE_REG_CLASS_ID_B32;
      break;
    case LOOM_CMD_LOWER_DISPATCH_ARGUMENT_KIND_B64:
      descriptor_ordinal = CMD_CORE_DESCRIPTOR_REF_CONSTANT_B64;
      cached_type = &state->types.b64;
      reg_class_id = CMD_CORE_REG_CLASS_ID_B64;
      break;
    default:
      IREE_ASSERT_UNREACHABLE("dispatch scalar argument kind is valid");
      IREE_BUILTIN_UNREACHABLE();
  }
  IREE_ASSERT_LT(argument->source_value, state->resources.source_value_count);
  loom_value_id_t* cached_value =
      &state->resources.source_values[argument->source_value].scalar;
  if (*cached_value != LOOM_VALUE_ID_INVALID) {
    *out_value = *cached_value;
    return iree_ok_status();
  }
  if (loom_type_kind(*cached_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(loom_low_build_register_type(
        state->descriptor_set, reg_class_id, 1, cached_type));
  }
  IREE_RETURN_IF_ERROR(loom_cmd_lower_emit_constant(
      state, argument->scalar_bits, descriptor_ordinal, *cached_type, location,
      cached_value));
  *out_value = *cached_value;
  return iree_ok_status();
}

static iree_status_t loom_cmd_lower_build_descriptor_op(
    loom_cmd_lower_state_t* state, uint32_t descriptor_ordinal,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    const loom_type_t* result_types, iree_host_size_t result_count,
    loom_location_id_t location, loom_op_t** out_op) {
  return loom_low_build_resolved_descriptor_op(
      &state->builder, state->descriptor_set,
      loom_cmd_lower_descriptor(state, descriptor_ordinal), operands,
      operand_count, loom_named_attr_slice_empty(), result_types, result_count,
      /*tied_results=*/NULL, /*tied_result_count=*/0, location, out_op);
}

static iree_status_t loom_cmd_lower_allocate_value_array(
    loom_cmd_lower_state_t* state, iree_host_size_t count,
    loom_value_id_t** out_values) {
  *out_values = NULL;
  if (count == 0) return iree_ok_status();
  return iree_arena_allocate_array(state->scratch_arena, count,
                                   sizeof(**out_values), (void**)out_values);
}

static iree_status_t loom_cmd_lower_build_dense_resources(
    loom_cmd_lower_state_t* state, uint32_t count, loom_type_id_t source_type,
    loom_type_t result_type, loom_value_id_t** out_values) {
  IREE_RETURN_IF_ERROR(
      loom_cmd_lower_allocate_value_array(state, count, out_values));
  for (uint32_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(loom_cmd_lower_build_resource(
        state, i, source_type, result_type, state->source_program.op->location,
        &(*out_values)[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_lower_build_abi_resources(
    loom_cmd_lower_state_t* state) {
  IREE_RETURN_IF_ERROR(loom_cmd_lower_build_dense_resources(
      state, state->plan->abi_layout.fixed_buffer_count,
      state->types.buffer_source, state->types.fixed_buffer,
      &state->resources.fixed_buffers));
  IREE_RETURN_IF_ERROR(loom_cmd_lower_build_dense_resources(
      state, state->plan->abi_layout.rebindable_binding_count,
      state->types.buffer_source, state->types.binding,
      &state->resources.bindings));
  IREE_RETURN_IF_ERROR(loom_cmd_lower_build_dense_resources(
      state, state->plan->abi_layout.executable_count,
      state->types.index_source, state->types.executable,
      &state->resources.executables));
  return loom_cmd_lower_build_dense_resources(
      state, state->plan->abi_layout.entry_count, state->types.index_source,
      state->types.entry, &state->resources.entries);
}

static iree_status_t loom_cmd_lower_build_buffer_tuple(
    loom_cmd_lower_state_t* state, const loom_cmd_buffer_binding_t* binding,
    uint64_t byte_offset, uint64_t byte_length, loom_location_id_t location,
    loom_cmd_lower_buffer_tuple_t* out_tuple) {
  *out_tuple = (loom_cmd_lower_buffer_tuple_t){
      .root = LOOM_VALUE_ID_INVALID,
      .byte_offset = LOOM_VALUE_ID_INVALID,
      .byte_length = LOOM_VALUE_ID_INVALID,
  };
  if (binding->role == LOOM_CMD_BUFFER_ROLE_FIXED) {
    IREE_ASSERT_LT(binding->resource_index,
                   state->plan->abi_layout.fixed_buffer_count);
    out_tuple->root = state->resources.fixed_buffers[binding->resource_index];
  } else {
    IREE_ASSERT_EQ(binding->role, LOOM_CMD_BUFFER_ROLE_REBINDABLE);
    IREE_ASSERT_LT(binding->resource_index,
                   state->plan->abi_layout.rebindable_binding_count);
    out_tuple->root = state->resources.bindings[binding->resource_index];
  }
  IREE_RETURN_IF_ERROR(loom_cmd_lower_build_u64_constant(
      state, byte_offset, location, &out_tuple->byte_offset));
  return loom_cmd_lower_build_u64_constant(state, byte_length, location,
                                           &out_tuple->byte_length);
}

static iree_status_t loom_cmd_lower_build_buffer_ref(
    loom_cmd_lower_state_t* state, const loom_cmd_buffer_binding_t* binding,
    uint64_t byte_offset, uint64_t byte_length, loom_location_id_t location,
    loom_value_id_t* out_buffer_ref) {
  loom_cmd_lower_buffer_tuple_t tuple = {0};
  IREE_RETURN_IF_ERROR(loom_cmd_lower_build_buffer_tuple(
      state, binding, byte_offset, byte_length, location, &tuple));
  const loom_value_id_t operands[] = {
      tuple.root,
      tuple.byte_offset,
      tuple.byte_length,
  };
  const uint32_t descriptor_ordinal =
      binding->role == LOOM_CMD_BUFFER_ROLE_FIXED
          ? CMD_CORE_DESCRIPTOR_REF_BUFFER_REF_DIRECT
          : CMD_CORE_DESCRIPTOR_REF_BUFFER_REF_BINDING;
  loom_op_t* buffer_ref_op = NULL;
  IREE_RETURN_IF_ERROR(loom_cmd_lower_build_descriptor_op(
      state, descriptor_ordinal, operands, IREE_ARRAYSIZE(operands),
      &state->types.buffer_ref, 1, location, &buffer_ref_op));
  *out_buffer_ref = loom_low_op_results(buffer_ref_op).values[0];
  return iree_ok_status();
}

static iree_status_t loom_cmd_lower_map_source_bindings(
    loom_cmd_lower_state_t* state) {
  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(state->source_program, &argument_count);
  const int64_t specialization_count_i64 =
      loom_func_like_specialization_count(state->source_program);
  IREE_ASSERT_GE(specialization_count_i64, 0);
  IREE_ASSERT_LE(specialization_count_i64, argument_count);
  const uint16_t specialization_count = (uint16_t)specialization_count_i64;
  const uint16_t binding_count = argument_count - specialization_count;
  IREE_ASSERT_EQ(state->plan->binding_count, binding_count);
  IREE_ASSERT(state->plan->binding_count == 0 || state->plan->bindings != NULL);

  for (uint16_t i = 0; i < binding_count; ++i) {
    const loom_value_id_t source_value = argument_ids[specialization_count + i];
    IREE_ASSERT_LT(source_value, state->resources.source_value_count);
    IREE_ASSERT(loom_type_is_buffer(
        loom_module_value_type(state->module, source_value)));
    const loom_cmd_buffer_binding_t* binding = &state->plan->bindings[i];
    loom_cmd_lower_buffer_tuple_t tuple = {0};
    IREE_RETURN_IF_ERROR(loom_cmd_lower_build_buffer_tuple(
        state, binding, binding->byte_offset, binding->byte_length,
        state->source_program.op->location, &tuple));
    state->resources.source_values[source_value].buffer = tuple;
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_lower_map_source_buffer_ranges(
    loom_cmd_lower_state_t* state) {
  IREE_ASSERT(state->plan->buffer_range_count == 0 ||
              state->plan->buffer_ranges != NULL);
  for (iree_host_size_t i = 0; i < state->plan->buffer_range_count; ++i) {
    const loom_cmd_buffer_range_t* range = &state->plan->buffer_ranges[i];
    IREE_ASSERT_LT(range->source_value, state->resources.source_value_count);
    IREE_ASSERT_EQ(
        state->resources.source_values[range->source_value].buffer.root,
        LOOM_VALUE_ID_INVALID);
    const loom_cmd_buffer_binding_t binding = {
        .role = range->role,
        .resource_index = range->resource_index,
        .byte_offset = range->byte_offset,
        .byte_length = range->byte_length,
    };
    loom_cmd_lower_buffer_tuple_t tuple = {0};
    IREE_RETURN_IF_ERROR(loom_cmd_lower_build_buffer_tuple(
        state, &binding, range->byte_offset, range->byte_length,
        state->source_program.op->location, &tuple));
    state->resources.source_values[range->source_value].buffer = tuple;
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_lower_build_launch_count_refs(
    loom_cmd_lower_state_t* state) {
  const loom_cmd_launch_graph_t* graph = state->plan->launch_graph;
  if (graph->host_tuple_count == 0) return iree_ok_status();

  const loom_cmd_lower_launch_count_binding_t binding =
      state->plan->launch_count_binding;
  IREE_ASSERT_LT(binding.resource_index,
                 state->plan->abi_layout.rebindable_binding_count);
  IREE_ASSERT_EQ(binding.byte_offset % sizeof(uint32_t), 0u);
  IREE_RETURN_IF_ERROR(loom_cmd_lower_allocate_value_array(
      state, graph->host_tuple_count, &state->resources.launch_counts));

  const uint64_t tuple_byte_length =
      LOOM_CMD_PROGRAM_LAUNCH_COUNT_TUPLE_BYTE_LENGTH;
  const loom_cmd_buffer_binding_t range_binding = {
      .role = LOOM_CMD_BUFFER_ROLE_REBINDABLE,
      .resource_index = binding.resource_index,
  };
  for (uint32_t i = 0; i < graph->host_tuple_count; ++i) {
    const uint64_t tuple_byte_offset = (uint64_t)i * tuple_byte_length;
    IREE_ASSERT_LE(binding.byte_offset, UINT64_MAX - tuple_byte_offset);
    IREE_RETURN_IF_ERROR(loom_cmd_lower_build_buffer_ref(
        state, &range_binding, binding.byte_offset + tuple_byte_offset,
        tuple_byte_length, state->source_program.op->location,
        &state->resources.launch_counts[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_lower_build_dispatch_operands(
    loom_cmd_lower_state_t* state, const loom_op_t* source_op,
    const loom_cmd_lower_dispatch_t* dispatch,
    const loom_value_id_t* prefix_operands,
    iree_host_size_t prefix_operand_count, loom_value_id_t** out_operands,
    iree_host_size_t* out_operand_count) {
  *out_operands = NULL;
  const iree_host_size_t operand_capacity =
      prefix_operand_count + dispatch->operand_value_count;
  IREE_RETURN_IF_ERROR(loom_cmd_lower_allocate_value_array(
      state, operand_capacity, out_operands));
  memcpy(*out_operands, prefix_operands,
         prefix_operand_count * sizeof(**out_operands));
  iree_host_size_t operand_count = prefix_operand_count;
  IREE_ASSERT(dispatch->argument_count == 0 || dispatch->arguments != NULL);
  for (uint16_t i = 0; i < dispatch->argument_count; ++i) {
    const loom_cmd_lower_dispatch_argument_t* argument =
        &dispatch->arguments[i];
    if (argument->kind != LOOM_CMD_LOWER_DISPATCH_ARGUMENT_KIND_BUFFER) {
      IREE_RETURN_IF_ERROR(loom_cmd_lower_build_dispatch_argument_constant(
          state, argument, source_op->location,
          &(*out_operands)[operand_count++]));
      continue;
    }

    const loom_value_id_t source_value = argument->source_value;
    IREE_ASSERT_LT(source_value, state->resources.source_value_count);
    if (state->resources.source_values[source_value].buffer.root ==
        LOOM_VALUE_ID_INVALID) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "kernel dispatch argument %" PRIu16
          " does not have a resolved command-program buffer range",
          i);
    }
    const loom_cmd_lower_buffer_tuple_t tuple =
        state->resources.source_values[source_value].buffer;
    (*out_operands)[operand_count++] = tuple.root;
    (*out_operands)[operand_count++] = tuple.byte_offset;
    (*out_operands)[operand_count++] = tuple.byte_length;
  }
  IREE_ASSERT_EQ(operand_count, operand_capacity);
  *out_operand_count = operand_count;
  return iree_ok_status();
}

static iree_status_t loom_cmd_lower_build_direct_dispatch(
    loom_cmd_lower_state_t* state, const loom_op_t* source_op,
    const loom_cmd_lower_dispatch_t* dispatch,
    loom_target_dispatch_workgroup_count_t workgroup_count, bool has_barrier) {
  IREE_ASSERT_LT(dispatch->executable_index,
                 state->plan->abi_layout.executable_count);
  IREE_ASSERT_LT(dispatch->entry_index, state->plan->abi_layout.entry_count);

  loom_value_id_t workgroup_count_x = LOOM_VALUE_ID_INVALID;
  loom_value_id_t workgroup_count_y = LOOM_VALUE_ID_INVALID;
  loom_value_id_t workgroup_count_z = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_cmd_lower_build_u32_constant(
      state, workgroup_count.x, source_op->location, &workgroup_count_x));
  IREE_RETURN_IF_ERROR(loom_cmd_lower_build_u32_constant(
      state, workgroup_count.y, source_op->location, &workgroup_count_y));
  IREE_RETURN_IF_ERROR(loom_cmd_lower_build_u32_constant(
      state, workgroup_count.z, source_op->location, &workgroup_count_z));
  const loom_value_id_t prefix_operands[] = {
      state->resources.executables[dispatch->executable_index],
      state->resources.entries[dispatch->entry_index],
      workgroup_count_x,
      workgroup_count_y,
      workgroup_count_z,
  };
  loom_value_id_t* operands = NULL;
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_cmd_lower_build_dispatch_operands(
      state, source_op, dispatch, prefix_operands,
      IREE_ARRAYSIZE(prefix_operands), &operands, &operand_count));
  loom_op_t* dispatch_op = NULL;
  const uint32_t descriptor_ordinal =
      has_barrier ? CMD_CORE_DESCRIPTOR_REF_DISPATCH_DIRECT_BARRIER
                  : CMD_CORE_DESCRIPTOR_REF_DISPATCH_DIRECT;
  return loom_cmd_lower_build_descriptor_op(
      state, descriptor_ordinal, operands, operand_count,
      /*result_types=*/NULL, /*result_count=*/0, source_op->location,
      &dispatch_op);
}

static iree_status_t loom_cmd_lower_build_indirect_dispatch(
    loom_cmd_lower_state_t* state, const loom_op_t* source_op,
    const loom_cmd_lower_dispatch_t* dispatch,
    loom_value_id_t workgroup_count_ref, bool has_barrier) {
  IREE_ASSERT_LT(dispatch->executable_index,
                 state->plan->abi_layout.executable_count);
  IREE_ASSERT_LT(dispatch->entry_index, state->plan->abi_layout.entry_count);

  const loom_value_id_t prefix_operands[] = {
      state->resources.executables[dispatch->executable_index],
      state->resources.entries[dispatch->entry_index],
      workgroup_count_ref,
  };
  loom_value_id_t* operands = NULL;
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_cmd_lower_build_dispatch_operands(
      state, source_op, dispatch, prefix_operands,
      IREE_ARRAYSIZE(prefix_operands), &operands, &operand_count));
  loom_op_t* dispatch_op = NULL;
  const uint32_t descriptor_ordinal =
      has_barrier ? CMD_CORE_DESCRIPTOR_REF_DISPATCH_INDIRECT_STATIC_BARRIER
                  : CMD_CORE_DESCRIPTOR_REF_DISPATCH_INDIRECT_STATIC;
  return loom_cmd_lower_build_descriptor_op(
      state, descriptor_ordinal, operands, operand_count,
      /*result_types=*/NULL, /*result_count=*/0, source_op->location,
      &dispatch_op);
}

static iree_status_t loom_cmd_lower_build_host_dispatch(
    loom_cmd_lower_state_t* state, const loom_op_t* source_op,
    const loom_cmd_lower_dispatch_t* dispatch, uint32_t host_tuple_ordinal,
    bool has_barrier) {
  IREE_ASSERT_LT(host_tuple_ordinal,
                 state->plan->launch_graph->host_tuple_count);
  return loom_cmd_lower_build_indirect_dispatch(
      state, source_op, dispatch,
      state->resources.launch_counts[host_tuple_ordinal], has_barrier);
}

static iree_status_t loom_cmd_lower_build_dispatch(
    loom_cmd_lower_state_t* state, iree_host_size_t dispatch_index,
    bool has_barrier) {
  const loom_cmd_lower_dispatch_t* dispatch =
      &state->plan->dispatches[dispatch_index];
  const loom_cmd_launch_count_t* launch_count =
      &state->plan->launch_graph->launches[dispatch_index];
  switch (launch_count->kind) {
    case LOOM_CMD_LAUNCH_COUNT_KIND_DIRECT:
      return loom_cmd_lower_build_direct_dispatch(
          state, launch_count->source_op, dispatch,
          launch_count->payload.direct, has_barrier);
    case LOOM_CMD_LAUNCH_COUNT_KIND_HOST:
      return loom_cmd_lower_build_host_dispatch(
          state, launch_count->source_op, dispatch,
          launch_count->payload.host_tuple_ordinal, has_barrier);
    default:
      IREE_ASSERT_UNREACHABLE("aggregate launch count kind is valid");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_cmd_lower_build_commands(
    loom_cmd_lower_state_t* state) {
  const loom_cmd_launch_graph_t* graph = state->plan->launch_graph;
  IREE_ASSERT(graph->launch_count == 0 || graph->launches != NULL);
  IREE_ASSERT(graph->launch_count == 0 || state->plan->dispatches != NULL);
  IREE_ASSERT(graph->wave_count == 0 || graph->waves != NULL);
  for (iree_host_size_t wave_index = 0; wave_index < graph->wave_count;
       ++wave_index) {
    const loom_cmd_schedule_wave_t wave = graph->waves[wave_index];
    IREE_ASSERT_GT(wave.command_count, 0u);
    for (iree_host_size_t i = 0; i < wave.command_count; ++i) {
      const iree_host_size_t dispatch_index = wave.command_offset + i;
      const bool has_barrier = wave_index != 0 && i == 0;
      IREE_RETURN_IF_ERROR(
          loom_cmd_lower_build_dispatch(state, dispatch_index, has_barrier));
    }
  }
  loom_op_t* return_op = NULL;
  return loom_low_return_build(&state->builder, /*values=*/NULL,
                               /*values_count=*/0,
                               state->source_program.op->location, &return_op);
}

static iree_status_t loom_cmd_lower_create_function(
    loom_cmd_lower_state_t* state) {
  loom_symbol_ref_t callee = loom_func_like_callee(state->source_program);
  IREE_ASSERT(loom_symbol_ref_is_valid(callee));
  IREE_ASSERT_EQ(callee.module_id, 0u);
  IREE_ASSERT_LT(callee.symbol_id, state->module->symbols.count);

  loom_string_id_t descriptor_set_key = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      state->module,
      loom_low_descriptor_set_string(state->descriptor_set,
                                     state->descriptor_set->key_string_offset),
      &descriptor_set_key));

  loom_low_func_def_build_flags_t build_flags =
      LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_ABI |
      LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_ABI_LAYOUT;
  uint8_t visibility = 0;
  if (loom_func_like_visibility(state->source_program) != 0) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY;
    visibility = LOOM_LOW_VISIBILITY_PUBLIC;
  }
  uint8_t retain = 0;
  if (iree_any_bit_set(state->module->symbols.entries[callee.symbol_id].flags,
                       LOOM_SYMBOL_FLAG_RETAIN)) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_RETAIN;
    retain = LOOM_LOW_RETAIN_RETAIN;
  }

  loom_builder_initialize(state->module, &state->module->arena,
                          loom_module_block(state->module), &state->builder);
  loom_builder_set_before(&state->builder, state->source_program.op);
  loom_attribute_t abi_layout_attr = loom_attr_absent();
  IREE_RETURN_IF_ERROR(loom_cmd_abi_layout_make_attr(
      state->module, &state->plan->abi_layout, &abi_layout_attr));
  IREE_RETURN_IF_ERROR(loom_low_func_def_build(
      &state->builder, build_flags, visibility, retain,
      /*cc=*/0, /*purity=*/0, /*allocation=*/0, /*schedule=*/0,
      descriptor_set_key, loom_symbol_ref_null(),
      LOOM_TARGET_ABI_COMMAND_PROGRAM, loom_named_attr_slice_empty(),
      loom_attr_as_dict(abi_layout_attr), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(), callee,
      /*arg_types=*/NULL, /*arg_types_count=*/0,
      /*result_types=*/NULL, /*result_count=*/0,
      /*tied_results=*/NULL, /*tied_result_count=*/0,
      /*predicates=*/NULL, /*predicates_count=*/0,
      state->source_program.op->location, &state->low_function));
  loom_builder_enter_region(&state->builder, state->low_function,
                            loom_low_func_def_body(state->low_function));
  return iree_ok_status();
}

static iree_status_t loom_cmd_lower_convert(loom_cmd_lower_state_t* state) {
  IREE_RETURN_IF_ERROR(loom_cmd_lower_initialize_types(state));
  IREE_RETURN_IF_ERROR(loom_cmd_lower_create_function(state));
  IREE_RETURN_IF_ERROR(loom_cmd_lower_build_abi_resources(state));
  IREE_RETURN_IF_ERROR(loom_cmd_lower_map_source_bindings(state));
  IREE_RETURN_IF_ERROR(loom_cmd_lower_map_source_buffer_ranges(state));
  IREE_RETURN_IF_ERROR(loom_cmd_lower_build_launch_count_refs(state));
  return loom_cmd_lower_build_commands(state);
}

iree_status_t loom_cmd_lower_program_to_low(loom_module_t* module,
                                            loom_op_t* program_op,
                                            const loom_cmd_lower_plan_t* plan,
                                            loom_op_t** out_low_function) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(program_op);
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(out_low_function);
  *out_low_function = NULL;

  loom_func_like_t source_program = loom_func_like_cast(module, program_op);
  IREE_ASSERT(loom_func_like_isa(source_program));
  uint16_t argument_count = 0;
  loom_func_like_arg_ids(source_program, &argument_count);
  const int64_t specialization_count =
      loom_func_like_specialization_count(source_program);
  IREE_ASSERT_GE(specialization_count, 0);
  IREE_ASSERT_LE(specialization_count, argument_count);
  IREE_ASSERT_EQ(plan->binding_count,
                 argument_count - (uint16_t)specialization_count);
  IREE_ASSERT(plan->binding_count == 0 || plan->bindings != NULL);
  IREE_ASSERT(plan->buffer_range_count == 0 || plan->buffer_ranges != NULL);
  IREE_ASSERT(plan->launch_graph != NULL);
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(module->arena.block_pool, &scratch_arena);
  iree_status_t status = iree_ok_status();

  loom_cmd_lower_state_t state = {
      .module = module,
      .source_program = source_program,
      .plan = plan,
      .descriptor_set = loom_cmd_core_descriptor_set(),
      .scratch_arena = &scratch_arena,
      .resources.source_value_count = module->values.count,
  };
  if (iree_status_is_ok(status) && state.resources.source_value_count != 0) {
    status = iree_arena_allocate_array(&scratch_arena,
                                       state.resources.source_value_count,
                                       sizeof(*state.resources.source_values),
                                       (void**)&state.resources.source_values);
  }
  if (iree_status_is_ok(status) && state.resources.source_value_count != 0) {
    memset(state.resources.source_values, 0xFF,
           state.resources.source_value_count *
               sizeof(*state.resources.source_values));
  }
  if (iree_status_is_ok(status)) {
    status = loom_cmd_lower_convert(&state);
  }

  if (!iree_status_is_ok(status) && state.low_function != NULL) {
    status =
        iree_status_join(status, loom_op_erase(module, state.low_function));
    state.low_function = NULL;
  }
  if (iree_status_is_ok(status)) {
    status = loom_op_erase(module, program_op);
    if (!iree_status_is_ok(status)) {
      status =
          iree_status_join(status, loom_op_erase(module, state.low_function));
      state.low_function = NULL;
    }
  }
  if (iree_status_is_ok(status)) {
    loom_module_link_symbol_defining_op(
        module, state.low_function, loom_op_vtable(module, state.low_function));
    *out_low_function = state.low_function;
  }

  iree_arena_deinitialize(&scratch_arena);
  return status;
}
