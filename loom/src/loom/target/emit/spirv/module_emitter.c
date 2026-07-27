// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/module_emitter.h"

#include <inttypes.h>

#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/module_contract.h"
#include "loom/target/arch/spirv/packet_rows.h"
#include "loom/target/emit/spirv/binary_format.h"
#include "loom/target/emit/spirv/module_abi.h"
#include "loom/target/emit/spirv/module_instructions.h"
#include "loom/target/emit/spirv/module_storage.h"
#include "loom/target/emit/spirv/module_types.h"
#include "loom/target/emit/spirv/module_values.h"
#include "loom/target/types.h"

enum {
  LOOM_SPIRV_BUILTIN_VARIABLE_COUNT = 3,
};

typedef enum loom_spirv_emit_module_state_flag_bits_e {
  LOOM_SPIRV_EMIT_MODULE_STATE_FLAG_NONE = 0u,
  LOOM_SPIRV_EMIT_MODULE_STATE_FLAG_BUILDER_INITIALIZED = 1u << 0,
} loom_spirv_emit_module_state_flag_bits_t;
typedef uint32_t loom_spirv_emit_module_state_flags_t;

typedef struct loom_spirv_emit_module_state_t {
  // Module containing the emitted low functions.
  loom_module_t* module;
  // Low descriptor registry used to resolve target-bound packets.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Runtime-selected target overlay, or empty for source-selected targets.
  loom_target_selection_t target_selection;
  // Structured diagnostic emitter for target-resolution failures.
  iree_diagnostic_emitter_t diagnostic_emitter;
  // Case/module scratch arena.
  iree_arena_allocator_t* scratch_arena;
  // Cached symbol facts shared by target resolution for every function.
  loom_symbol_fact_table_t symbol_facts;
  // Sectioned SPIR-V module builder shared by every emitted function.
  loom_spirv_module_builder_t builder;
  // SPIR-V type and constant emission cache shared by the module.
  loom_spirv_type_context_t type_context;
  // Module-level raw-BDA ABI layout shared by HAL kernel entries.
  loom_spirv_module_raw_bda_layout_t raw_bda_layout;
  // Shared Input variables for workgroup/local/global invocation builtins.
  uint32_t builtin_variable_ids[LOOM_SPIRV_BUILTIN_VARIABLE_COUNT];
  // First function's module-level target contract.
  loom_spirv_module_contract_t contract;
  // Module-emission state flags.
  loom_spirv_emit_module_state_flags_t flags;
  // Number of functions emitted into the current module.
  iree_host_size_t function_count;
} loom_spirv_emit_module_state_t;

typedef struct loom_spirv_emit_state_t {
  // Module-emission state shared by every emitted function.
  loom_spirv_emit_module_state_t* module_state;
  // Module containing the emitted low function.
  loom_module_t* module;
  // Target-low function definition being emitted.
  loom_op_t* function_op;
  // Target-low function body being emitted.
  const loom_region_t* body;
  // Resolved target record and descriptor set for |function_op|.
  const loom_low_resolved_target_t* target;
  // Case/function scratch arena.
  iree_arena_allocator_t* scratch_arena;
  // Sectioned SPIR-V module builder.
  loom_spirv_module_builder_t* builder;
  // Function-local value domain for dense value tables.
  loom_local_value_domain_t value_domain;
  // Function-local Loom value to SPIR-V value-ref table.
  loom_spirv_module_value_table_t value_table;
  // Module string IDs for descriptor-set immediate rows.
  loom_string_id_t* immediate_name_ids;
  // Number of entries in |immediate_name_ids|.
  iree_host_size_t immediate_name_id_count;
  // SPIR-V type and constant emission cache shared by the module.
  loom_spirv_type_context_t* type_context;
  // SPIR-V ID assigned to the function.
  uint32_t function_id;
  // SPIR-V label ID of the currently open function-section block.
  uint32_t current_label_id;
  // Selected ABI plan for entry materialization.
  loom_spirv_module_abi_plan_t abi_plan;
  // Function-local Workgroup storage materialization state.
  loom_spirv_module_workgroup_storage_state_t workgroup_storage;
  // Input variables referenced by this function's entry-point interface.
  uint32_t builtin_interface_variable_ids[LOOM_SPIRV_BUILTIN_VARIABLE_COUNT];
} loom_spirv_emit_state_t;

static loom_spirv_binary_writer_t* loom_spirv_emit_section(
    loom_spirv_emit_state_t* state, loom_spirv_module_section_t section) {
  return loom_spirv_module_builder_section(state->builder, section);
}

static uint32_t loom_spirv_emit_allocate_id(loom_spirv_emit_state_t* state) {
  return loom_spirv_module_builder_allocate_id(state->builder);
}

static iree_string_view_t loom_spirv_emit_string_or_empty(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static iree_string_view_t loom_spirv_emit_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  return loom_spirv_emit_string_or_empty(module, symbol->name_id);
}

static iree_string_view_t loom_spirv_emit_function_name(
    const loom_spirv_emit_state_t* state) {
  return loom_spirv_emit_symbol_name(
      state->module, loom_low_function_callee(state->function_op));
}

static iree_string_view_t loom_spirv_emit_export_name(
    const loom_spirv_emit_state_t* state) {
  const iree_string_view_t export_symbol =
      state->target->bundle_storage.export_plan.export_symbol;
  if (!iree_string_view_is_empty(export_symbol)) {
    return export_symbol;
  }
  return loom_spirv_emit_function_name(state);
}

static iree_status_t loom_spirv_emit_op_name(loom_spirv_emit_state_t* state,
                                             uint32_t id,
                                             iree_string_view_t name) {
  if (id == 0 || iree_string_view_is_empty(name)) {
    return iree_ok_status();
  }
  const uint32_t prefix_operands[] = {id};
  return loom_spirv_binary_write_string_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DEBUG),
      LOOM_SPIRV_OP_NAME, prefix_operands, IREE_ARRAYSIZE(prefix_operands),
      name, NULL, 0);
}

static iree_status_t loom_spirv_emit_value_name(loom_spirv_emit_state_t* state,
                                                loom_value_id_t value_id,
                                                uint32_t id) {
  if (value_id >= state->module->values.count) {
    return iree_ok_status();
  }
  const loom_value_t* value = loom_module_value(state->module, value_id);
  return loom_spirv_emit_op_name(
      state, id,
      loom_spirv_emit_string_or_empty(state->module, value->name_id));
}

static iree_status_t loom_spirv_emit_prepare_immediate_name_ids(
    loom_spirv_emit_state_t* state) {
  const iree_host_size_t immediate_count =
      state->target->descriptor_set->immediate_count;
  if (immediate_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, immediate_count, sizeof(*state->immediate_name_ids),
      (void**)&state->immediate_name_ids));
  state->immediate_name_id_count = immediate_count;
  for (iree_host_size_t i = 0; i < immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        &state->target->descriptor_set->immediates[i];
    const iree_string_view_t name = loom_low_descriptor_set_string(
        state->target->descriptor_set, immediate->field_name_string_offset);
    state->immediate_name_ids[i] =
        loom_module_lookup_string(state->module, name);
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_define_value(
    loom_spirv_emit_state_t* state, loom_value_id_t value_id,
    loom_spirv_module_value_ref_t value_ref, bool emit_name) {
  loom_spirv_module_value_table_define(&state->value_table, value_id,
                                       value_ref);
  if (emit_name) {
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_value_name(state, value_id, value_ref.id));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_reserve_value_ref(
    loom_spirv_emit_state_t* state, loom_value_id_t value_id, uint32_t type_id,
    loom_spirv_value_type_t value_type, uint32_t* out_result_id) {
  *out_result_id = loom_spirv_module_value_table_reserve(
      &state->value_table, state->builder, value_id, type_id, value_type);
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_prepare_packet_result(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet, uint32_t type_id,
    loom_spirv_value_type_t value_type, uint32_t* out_result_id) {
  return loom_spirv_emit_reserve_value_ref(state,
                                           loom_op_const_results(packet->op)[0],
                                           type_id, value_type, out_result_id);
}

static iree_status_t loom_spirv_emit_define_packet_result(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet, uint32_t id,
    uint32_t type_id, loom_spirv_value_type_t value_type) {
  const loom_spirv_module_value_ref_t value_ref = {
      .id = id, .type_id = type_id, .value_type = value_type};
  return loom_spirv_emit_define_value(
      state, loom_op_const_results(packet->op)[0], value_ref, true);
}

static iree_status_t loom_spirv_emit_lookup_value(
    loom_spirv_emit_state_t* state, loom_value_id_t value_id,
    loom_spirv_module_value_ref_t* out_value_ref) {
  *out_value_ref =
      loom_spirv_module_value_table_lookup(&state->value_table, value_id);
  return iree_ok_status();
}

static bool loom_spirv_emit_value_ref_exists(loom_spirv_emit_state_t* state,
                                             loom_value_id_t value_id) {
  return loom_spirv_module_value_table_exists(&state->value_table, value_id);
}

static loom_spirv_module_abi_context_t loom_spirv_emit_abi_context(
    loom_spirv_emit_state_t* state) {
  return (loom_spirv_module_abi_context_t){
      .module = state->module,
      .function_op = state->function_op,
      .target = state->target,
      .scratch_arena = state->scratch_arena,
      .builder = state->builder,
      .type_context = state->type_context,
      .raw_bda_layout = &state->module_state->raw_bda_layout,
      .value_table = &state->value_table,
  };
}

static bool loom_spirv_emit_builtin_variable_info(
    uint32_t builtin, uint8_t* out_slot, iree_string_view_t* out_name) {
  switch (builtin) {
    case LOOM_SPIRV_BUILT_IN_WORKGROUP_ID:
      *out_slot = 0;
      *out_name = IREE_SV("workgroup_id");
      return true;
    case LOOM_SPIRV_BUILT_IN_LOCAL_INVOCATION_ID:
      *out_slot = 1;
      *out_name = IREE_SV("local_invocation_id");
      return true;
    case LOOM_SPIRV_BUILT_IN_GLOBAL_INVOCATION_ID:
      *out_slot = 2;
      *out_name = IREE_SV("global_invocation_id");
      return true;
  }
  return false;
}

static iree_status_t loom_spirv_emit_builtin_variable(
    loom_spirv_emit_state_t* state, uint32_t builtin,
    uint32_t* out_variable_id) {
  uint8_t slot = 0;
  iree_string_view_t name = iree_string_view_empty();
  if (!loom_spirv_emit_builtin_variable_info(builtin, &slot, &name)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "unknown SPIR-V builtin packet row %" PRIu32,
                            builtin);
  }
  uint32_t variable_id = state->module_state->builtin_variable_ids[slot];
  if (variable_id != 0) {
    state->builtin_interface_variable_ids[slot] = variable_id;
    *out_variable_id = variable_id;
    return iree_ok_status();
  }

  uint32_t u32_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_u32(state->type_context, &u32_type_id));
  uint32_t vector_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_vector(
      state->type_context, u32_type_id, 3, &vector_type_id));
  uint32_t pointer_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_pointer(
      state->type_context, LOOM_SPIRV_STORAGE_CLASS_INPUT, vector_type_id,
      /*pointer_array_stride=*/0, &pointer_type_id));

  variable_id = loom_spirv_emit_allocate_id(state);
  const uint32_t decoration_operands[] = {
      variable_id,
      LOOM_SPIRV_DECORATION_BUILT_IN,
      builtin,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_ANNOTATION),
      LOOM_SPIRV_OP_DECORATE, decoration_operands,
      IREE_ARRAYSIZE(decoration_operands)));
  const uint32_t variable_operands[] = {
      pointer_type_id,
      variable_id,
      LOOM_SPIRV_STORAGE_CLASS_INPUT,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
      LOOM_SPIRV_OP_VARIABLE, variable_operands,
      IREE_ARRAYSIZE(variable_operands)));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_op_name(state, variable_id, name));
  state->module_state->builtin_variable_ids[slot] = variable_id;
  state->builtin_interface_variable_ids[slot] = variable_id;
  *out_variable_id = variable_id;
  return iree_ok_status();
}

static loom_named_attr_slice_t loom_spirv_emit_packet_attrs(
    const loom_low_resolved_descriptor_packet_t* packet) {
  switch (packet->kind) {
    case LOOM_LOW_DESCRIPTOR_PACKET_CONST:
      return loom_low_const_attrs(packet->op);
    case LOOM_LOW_DESCRIPTOR_PACKET_OP:
      return loom_low_op_attrs(packet->op);
    case LOOM_LOW_DESCRIPTOR_PACKET_NONE:
      break;
  }
  return loom_make_named_attr_slice(NULL, 0);
}

static const loom_low_immediate_t* loom_spirv_emit_descriptor_immediate(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    uint8_t descriptor_immediate_index) {
  if (descriptor_immediate_index >= packet->descriptor->immediate_count) {
    return NULL;
  }
  const uint32_t immediate_row =
      packet->descriptor->immediate_start + descriptor_immediate_index;
  if (immediate_row >= state->target->descriptor_set->immediate_count) {
    return NULL;
  }
  return &state->target->descriptor_set->immediates[immediate_row];
}

static iree_string_view_t loom_spirv_emit_immediate_name(
    loom_spirv_emit_state_t* state, const loom_low_immediate_t* immediate) {
  if (immediate == NULL) {
    return IREE_SV("<unknown>");
  }
  return loom_low_descriptor_set_string(state->target->descriptor_set,
                                        immediate->field_name_string_offset);
}

static iree_status_t loom_spirv_emit_lookup_i64_immediate(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row, int64_t* out_value) {
  const loom_low_immediate_t* immediate =
      loom_spirv_emit_descriptor_immediate(state, packet, row->immediate_index);
  IREE_ASSERT(immediate != NULL);
  const uint32_t immediate_row =
      packet->descriptor->immediate_start + row->immediate_index;
  IREE_ASSERT_LT(immediate_row, state->immediate_name_id_count);
  const loom_string_id_t expected_name_id =
      state->immediate_name_ids[immediate_row];
  const loom_named_attr_slice_t attrs = loom_spirv_emit_packet_attrs(packet);
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (attr->name_id != expected_name_id) {
      continue;
    }
    IREE_ASSERT_EQ(attr->value.kind, LOOM_ATTR_I64);
    *out_value = loom_attr_as_i64(attr->value);
    return iree_ok_status();
  }
  if (iree_any_bit_set(immediate->flags,
                       LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE)) {
    *out_value = immediate->default_value;
    return iree_ok_status();
  }
  IREE_CHECK_UNREACHABLE("verified low descriptor immediate attribute");
  return iree_ok_status();
}

static void loom_spirv_emit_validate_packet_shape(
    const loom_op_t* op, const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  (void)packet;
  IREE_ASSERT_EQ(op->result_count, row->result_count);
  IREE_ASSERT_EQ(op->operand_count, row->operand_count);
}

static iree_status_t loom_spirv_emit_load_packet_operands(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row,
    loom_spirv_module_value_ref_t* out_operands) {
  const loom_value_id_t* operand_values = loom_op_const_operands(packet->op);
  for (uint8_t i = 0; i < row->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(state, operand_values[i],
                                                      &out_operands[i]));
    IREE_ASSERT(loom_spirv_value_type_equal(out_operands[i].value_type,
                                            row->operand_types[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_integer_constant_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_i64_immediate(state, packet, row, &value));
  uint32_t type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      state->type_context, row->result_type, &type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, type_id, row->result_type, &result_id));
  const uint64_t literal = (uint64_t)value;
  uint32_t operands[] = {
      type_id,
      result_id,
      (uint32_t)literal,
      (uint32_t)(literal >> 32),
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
      row->opcode, operands, 2 + row->literal_word_count));
  return loom_spirv_emit_define_packet_result(state, packet, result_id, type_id,
                                              row->result_type);
}

static iree_status_t loom_spirv_emit_binary_same_type_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  IREE_ASSERT_EQ(operands[0].type_id, operands[1].type_id);
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, operands[0].type_id, row->result_type, &result_id));
  const uint32_t instruction_operands[] = {
      operands[0].type_id,
      result_id,
      operands[0].id,
      operands[1].id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(
      state, packet, result_id, operands[0].type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_unary_convert_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[1] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      state->type_context, row->result_type, &result_type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, row->result_type, &result_id));
  const uint32_t instruction_operands[] = {
      result_type_id,
      result_id,
      operands[0].id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(state, packet, result_id,
                                              result_type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_load_builtin_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  uint32_t variable_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_builtin_variable(state, row->builtin, &variable_id));
  uint32_t u32_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_u32(state->type_context, &u32_type_id));
  uint32_t vector_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_vector(
      state->type_context, u32_type_id, 3, &vector_type_id));

  uint32_t vector_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_unary_result(
      state->builder, LOOM_SPIRV_OP_LOAD, vector_type_id, variable_id,
      &vector_id));

  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      state->type_context, row->result_type, &result_type_id));
  uint32_t result_id = 0;
  uint32_t component_id = 0;
  if (result_type_id == u32_type_id) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
        state, packet, result_type_id, row->result_type, &result_id));
    component_id = result_id;
  } else {
    component_id = loom_spirv_emit_allocate_id(state);
  }
  const uint32_t extract_operands[] = {
      u32_type_id,
      component_id,
      vector_id,
      row->component_index,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_COMPOSITE_EXTRACT, extract_operands,
      IREE_ARRAYSIZE(extract_operands)));

  if (result_type_id != u32_type_id) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
        state, packet, result_type_id, row->result_type, &result_id));
    const uint32_t bitcast_operands[] = {
        result_type_id,
        result_id,
        component_id,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
        LOOM_SPIRV_OP_BITCAST, bitcast_operands,
        IREE_ARRAYSIZE(bitcast_operands)));
  }
  return loom_spirv_emit_define_packet_result(state, packet, result_id,
                                              result_type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_integer_mul_add_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[3] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  IREE_ASSERT_EQ(operands[0].type_id, operands[1].type_id);
  IREE_ASSERT_EQ(operands[0].type_id, operands[2].type_id);
  uint32_t product_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_binary_result(
      state->builder, LOOM_SPIRV_OP_I_MUL, operands[0].type_id, operands[0].id,
      operands[1].id, &product_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, operands[0].type_id, row->result_type, &result_id));
  const uint32_t instruction_operands[] = {
      operands[0].type_id,
      result_id,
      product_id,
      operands[2].id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_I_ADD, instruction_operands,
      IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(
      state, packet, result_id, operands[0].type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_compare_same_type_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  IREE_ASSERT_EQ(operands[0].type_id, operands[1].type_id);
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      state->type_context, row->result_type, &result_type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, row->result_type, &result_id));
  const uint32_t instruction_operands[] = {
      result_type_id,
      result_id,
      operands[0].id,
      operands[1].id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(state, packet, result_id,
                                              result_type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_select_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[3] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  IREE_ASSERT_EQ(operands[1].type_id, operands[2].type_id);
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      state->type_context, row->result_type, &result_type_id));
  IREE_ASSERT_EQ(result_type_id, operands[1].type_id);
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, row->result_type, &result_id));
  const uint32_t instruction_operands[] = {
      result_type_id, result_id, operands[0].id, operands[1].id, operands[2].id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(state, packet, result_id,
                                              result_type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_typed_physical_storage_buffer_pointer(
    loom_spirv_emit_state_t* state, loom_spirv_module_value_ref_t address,
    loom_spirv_value_type_t pointer_type, uint32_t pointer_type_id,
    uint32_t* out_pointer_id) {
  if (address.value_type.value_class ==
      LOOM_SPIRV_VALUE_CLASS_PTR_PHYSICAL_STORAGE_BUFFER) {
    IREE_ASSERT(loom_spirv_value_type_equal(address.value_type, pointer_type));
    *out_pointer_id = address.id;
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(address.value_type.value_class,
                 LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS);
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
  const uint32_t operands[] = {
      pointer_type_id,
      result_id,
      address.id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_CONVERT_U_TO_PTR, operands, IREE_ARRAYSIZE(operands)));
  *out_pointer_id = result_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_ptr_access_chain_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      state->type_context, row->result_type, &result_type_id));
  uint32_t base_pointer_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_typed_physical_storage_buffer_pointer(
      state, operands[0], row->result_type, result_type_id, &base_pointer_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, row->result_type, &result_id));
  const uint32_t instruction_operands[] = {
      result_type_id,
      result_id,
      base_pointer_id,
      operands[1].id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(state, packet, result_id,
                                              result_type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_access_chain_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      state->type_context, row->result_type, &result_type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, row->result_type, &result_id));
  const uint32_t instruction_operands[] = {
      result_type_id,
      result_id,
      operands[0].id,
      operands[1].id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(state, packet, result_id,
                                              result_type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_load_aligned_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[1] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      state->type_context, row->result_type, &result_type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, row->result_type, &result_id));
  const uint32_t instruction_operands[] = {
      result_type_id,        result_id,
      operands[0].id,        LOOM_SPIRV_MEMORY_ACCESS_ALIGNED_MASK,
      row->memory_alignment,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(state, packet, result_id,
                                              result_type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_store_aligned_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  const uint32_t instruction_operands[] = {
      operands[0].id,
      operands[1].id,
      LOOM_SPIRV_MEMORY_ACCESS_ALIGNED_MASK,
      row->memory_alignment,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands));
}

static iree_status_t loom_spirv_emit_cooperative_matrix_layout_operands(
    loom_spirv_emit_state_t* state, const loom_spirv_packet_row_t* row,
    uint32_t* out_layout_id, uint32_t* out_stride_id) {
  IREE_RETURN_IF_ERROR(loom_spirv_emit_u32_constant(
      state->type_context, row->cooperative_matrix_layout, out_layout_id));
  return loom_spirv_emit_u32_constant(
      state->type_context, row->cooperative_matrix_stride, out_stride_id);
}

static iree_status_t loom_spirv_emit_cooperative_matrix_load_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[1] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      state->type_context, row->result_type, &result_type_id));
  uint32_t layout_id = 0;
  uint32_t stride_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_cooperative_matrix_layout_operands(
      state, row, &layout_id, &stride_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, row->result_type, &result_id));
  const uint32_t instruction_operands[] = {
      result_type_id,
      result_id,
      operands[0].id,
      layout_id,
      stride_id,
      LOOM_SPIRV_MEMORY_ACCESS_ALIGNED_MASK,
      row->memory_alignment,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(state, packet, result_id,
                                              result_type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_cooperative_matrix_store_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t layout_id = 0;
  uint32_t stride_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_cooperative_matrix_layout_operands(
      state, row, &layout_id, &stride_id));
  const uint32_t instruction_operands[] = {
      operands[0].id,
      operands[1].id,
      layout_id,
      stride_id,
      LOOM_SPIRV_MEMORY_ACCESS_ALIGNED_MASK,
      row->memory_alignment,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands));
}

static iree_status_t loom_spirv_emit_cooperative_matrix_mul_add_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[3] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      state->type_context, row->result_type, &result_type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, row->result_type, &result_id));
  uint32_t instruction_operands[6] = {
      result_type_id, result_id, operands[0].id, operands[1].id, operands[2].id,
  };
  uint8_t operand_count = 5;
  if (row->cooperative_matrix_operands != 0) {
    instruction_operands[operand_count++] = row->cooperative_matrix_operands;
  }
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, operand_count));
  return loom_spirv_emit_define_packet_result(state, packet, result_id,
                                              result_type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_control_barrier_packet(
    loom_spirv_emit_state_t* state, const loom_spirv_packet_row_t* row) {
  uint32_t execution_scope_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_u32_constant(
      state->type_context, row->execution_scope, &execution_scope_id));
  uint32_t memory_scope_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_u32_constant(
      state->type_context, row->memory_scope, &memory_scope_id));
  uint32_t memory_semantics_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_u32_constant(
      state->type_context, row->memory_semantics, &memory_semantics_id));
  const uint32_t instruction_operands[] = {
      execution_scope_id,
      memory_scope_id,
      memory_semantics_id,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands));
}

static iree_status_t loom_spirv_emit_copy(loom_spirv_emit_state_t* state,
                                          const loom_op_t* op) {
  loom_spirv_module_value_ref_t source = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, loom_low_copy_source(op), &source));
  return loom_spirv_emit_define_value(state, loom_low_copy_result(op), source,
                                      false);
}

static iree_status_t loom_spirv_emit_descriptor_packet(
    loom_spirv_emit_state_t* state, const loom_op_t* op,
    const loom_low_resolved_descriptor_packet_t* packet) {
  for (uint16_t i = 0; i < packet->descriptor->feature_mask_word_count; ++i) {
    const uint32_t feature_mask_row =
        packet->descriptor->feature_mask_word_start + i;
    const uint64_t feature_bits =
        state->target->descriptor_set->feature_mask_words[feature_mask_row];
    IREE_ASSERT(i == 0 || feature_bits == 0);
    loom_spirv_module_builder_require_feature_bits(
        state->builder, (loom_spirv_feature_bits_t)feature_bits);
  }

  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_descriptor_ordinal(state->target->descriptor_set,
                                                 packet->descriptor);
  const loom_spirv_packet_row_t* row =
      loom_spirv_packet_row_for_descriptor_ordinal(descriptor_ordinal);
  IREE_ASSERT(row != NULL);
  loom_spirv_emit_validate_packet_shape(op, packet, row);
  switch (row->form) {
    case LOOM_SPIRV_PACKET_FORM_INTEGER_CONSTANT:
      return loom_spirv_emit_integer_constant_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE:
      return loom_spirv_emit_binary_same_type_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_UNARY_CONVERT:
      return loom_spirv_emit_unary_convert_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_LOAD_BUILTIN:
      return loom_spirv_emit_load_builtin_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_INTEGER_MUL_ADD:
      return loom_spirv_emit_integer_mul_add_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_COMPARE_SAME_TYPE:
      return loom_spirv_emit_compare_same_type_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_SELECT:
      return loom_spirv_emit_select_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_PTR_ACCESS_CHAIN:
      return loom_spirv_emit_ptr_access_chain_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_ACCESS_CHAIN:
      return loom_spirv_emit_access_chain_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_LOAD_ALIGNED:
      return loom_spirv_emit_load_aligned_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_STORE_ALIGNED:
      return loom_spirv_emit_store_aligned_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_COOPERATIVE_MATRIX_LOAD:
      return loom_spirv_emit_cooperative_matrix_load_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_COOPERATIVE_MATRIX_STORE:
      return loom_spirv_emit_cooperative_matrix_store_packet(state, packet,
                                                             row);
    case LOOM_SPIRV_PACKET_FORM_COOPERATIVE_MATRIX_MUL_ADD:
      return loom_spirv_emit_cooperative_matrix_mul_add_packet(state, packet,
                                                               row);
    case LOOM_SPIRV_PACKET_FORM_CONTROL_BARRIER:
      return loom_spirv_emit_control_barrier_packet(state, row);
    case LOOM_SPIRV_PACKET_FORM_UNSUPPORTED:
      break;
  }
  IREE_CHECK_UNREACHABLE("verified SPIR-V binary packet row");
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_low_op(loom_spirv_emit_state_t* state,
                                            const loom_op_t* op);

typedef struct loom_spirv_emit_region_yield_t {
  // Region terminator op when present.
  const loom_op_t* op;
  // Label ID of the block that reaches the parent merge/continue edge.
  uint32_t predecessor_label_id;
} loom_spirv_emit_region_yield_t;

static iree_status_t loom_spirv_emit_label_id(loom_spirv_emit_state_t* state,
                                              uint32_t label_id) {
  const uint32_t operands[] = {label_id};
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_LABEL, operands, IREE_ARRAYSIZE(operands)));
  state->current_label_id = label_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_branch_label(
    loom_spirv_emit_state_t* state, uint32_t label_id) {
  const uint32_t operands[] = {label_id};
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_BRANCH, operands, IREE_ARRAYSIZE(operands));
}

static bool loom_spirv_emit_loop_counter_is_unsigned(
    loom_spirv_value_type_t value_type) {
  if (value_type.value_class == LOOM_SPIRV_VALUE_CLASS_OFFSET64) {
    return true;
  }
  if (value_type.value_class != LOOM_SPIRV_VALUE_CLASS_SCALAR) {
    IREE_CHECK_UNREACHABLE("verified SPIR-V low.scf.for counter type");
    return false;
  }
  const loom_spirv_scalar_type_descriptor_t* descriptor =
      loom_spirv_scalar_type_descriptor(value_type.scalar_type);
  if (descriptor == NULL ||
      descriptor->kind == LOOM_SPIRV_SCALAR_TYPE_KIND_FLOAT) {
    IREE_CHECK_UNREACHABLE("verified SPIR-V low.scf.for counter type");
    return false;
  }
  return descriptor->kind == LOOM_SPIRV_SCALAR_TYPE_KIND_UNSIGNED_INT;
}

static uint32_t loom_spirv_emit_for_loop_control(const loom_op_t* op) {
  if (loom_low_scf_for_unroll_factor_is_present(op) ||
      !loom_attr_is_absent(
          loom_op_attrs(op)[loom_low_scf_for_unroll_policy_ATTR_INDEX])) {
    return LOOM_SPIRV_LOOP_CONTROL_UNROLL_MASK;
  }
  return LOOM_SPIRV_LOOP_CONTROL_NONE;
}

static const loom_op_t* loom_spirv_emit_region_terminator(
    const loom_region_t* region) {
  const loom_block_t* block = loom_region_const_entry_block(region);
  return block != NULL ? block->last_op : NULL;
}

static iree_status_t loom_spirv_emit_region_ops(
    loom_spirv_emit_state_t* state, const loom_region_t* region,
    loom_spirv_emit_region_yield_t* out_yield) {
  *out_yield = (loom_spirv_emit_region_yield_t){
      .predecessor_label_id = state->current_label_id,
  };
  const loom_block_t* block = loom_region_const_entry_block(region);
  if (block == NULL) {
    return iree_ok_status();
  }
  for (const loom_op_t* op = block->first_op; op != NULL; op = op->next_op) {
    if (loom_low_scf_yield_isa(op)) {
      *out_yield = (loom_spirv_emit_region_yield_t){
          .op = op,
          .predecessor_label_id = state->current_label_id,
      };
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_spirv_emit_low_op(state, op));
  }
  out_yield->predecessor_label_id = state->current_label_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_scf_result_phi(
    loom_spirv_emit_state_t* state, loom_value_id_t result_value_id,
    loom_value_id_t then_value_id, uint32_t then_predecessor_label_id,
    loom_value_id_t else_value_id, uint32_t else_predecessor_label_id) {
  loom_spirv_module_value_ref_t then_ref = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, then_value_id, &then_ref));
  loom_spirv_module_value_ref_t else_ref = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, else_value_id, &else_ref));
  IREE_ASSERT_EQ(then_ref.type_id, else_ref.type_id);
  IREE_ASSERT(
      loom_spirv_value_type_equal(then_ref.value_type, else_ref.value_type));

  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_reserve_value_ref(
      state, result_value_id, then_ref.type_id, then_ref.value_type,
      &result_id));
  const uint32_t operands[] = {
      then_ref.type_id,          result_id,   then_ref.id,
      then_predecessor_label_id, else_ref.id, else_predecessor_label_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_PHI, operands, IREE_ARRAYSIZE(operands)));
  const loom_spirv_module_value_ref_t result_ref = {
      .id = result_id,
      .type_id = then_ref.type_id,
      .value_type = then_ref.value_type,
  };
  return loom_spirv_emit_define_value(state, result_value_id, result_ref, true);
}

static iree_status_t loom_spirv_emit_scf_if(loom_spirv_emit_state_t* state,
                                            const loom_op_t* op) {
  loom_spirv_module_value_ref_t condition = {0};
  IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(
      state, loom_low_scf_if_condition(op), &condition));

  const loom_value_slice_t results = loom_low_scf_if_results(op);
  const loom_region_t* then_region = loom_low_scf_if_then_region(op);
  const loom_region_t* else_region = loom_low_scf_if_else_region(op);
  const uint32_t condition_label_id = state->current_label_id;
  const uint32_t then_label_id = loom_spirv_emit_allocate_id(state);
  const uint32_t merge_label_id = loom_spirv_emit_allocate_id(state);
  const uint32_t else_label_id =
      else_region != NULL ? loom_spirv_emit_allocate_id(state) : merge_label_id;

  const uint32_t selection_merge_operands[] = {
      merge_label_id,
      LOOM_SPIRV_SELECTION_CONTROL_NONE,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_SELECTION_MERGE, selection_merge_operands,
      IREE_ARRAYSIZE(selection_merge_operands)));
  const uint32_t branch_operands[] = {
      condition.id,
      then_label_id,
      else_label_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_BRANCH_CONDITIONAL, branch_operands,
      IREE_ARRAYSIZE(branch_operands)));

  IREE_RETURN_IF_ERROR(loom_spirv_emit_label_id(state, then_label_id));
  loom_spirv_emit_region_yield_t then_yield = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_region_ops(state, then_region, &then_yield));
  const uint32_t then_predecessor_label_id = state->current_label_id;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_branch_label(state, merge_label_id));

  loom_spirv_emit_region_yield_t else_yield = {
      .predecessor_label_id = condition_label_id,
  };
  if (else_region != NULL) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_label_id(state, else_label_id));
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_region_ops(state, else_region, &else_yield));
    else_yield.predecessor_label_id = state->current_label_id;
    IREE_RETURN_IF_ERROR(loom_spirv_emit_branch_label(state, merge_label_id));
  }

  IREE_RETURN_IF_ERROR(loom_spirv_emit_label_id(state, merge_label_id));
  if (results.count == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT(then_yield.op != NULL);
  IREE_ASSERT(else_yield.op != NULL);
  const loom_value_slice_t then_values =
      loom_low_scf_yield_values(then_yield.op);
  const loom_value_slice_t else_values =
      loom_low_scf_yield_values(else_yield.op);
  IREE_ASSERT_EQ(then_values.count, results.count);
  IREE_ASSERT_EQ(else_values.count, results.count);
  for (uint16_t i = 0; i < results.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_scf_result_phi(
        state, results.values[i], then_values.values[i],
        then_predecessor_label_id, else_values.values[i],
        else_yield.predecessor_label_id));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_lookup_or_reserve_typed_value(
    loom_spirv_emit_state_t* state, loom_value_id_t value_id, uint32_t type_id,
    loom_spirv_value_type_t value_type,
    loom_spirv_module_value_ref_t* out_value_ref) {
  if (loom_spirv_emit_value_ref_exists(state, value_id)) {
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_lookup_value(state, value_id, out_value_ref));
    IREE_ASSERT_EQ(out_value_ref->type_id, type_id);
    IREE_ASSERT(
        loom_spirv_value_type_equal(out_value_ref->value_type, value_type));
    return iree_ok_status();
  }
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_reserve_value_ref(
      state, value_id, type_id, value_type, &result_id));
  *out_value_ref = (loom_spirv_module_value_ref_t){
      .id = result_id,
      .type_id = type_id,
      .value_type = value_type,
  };
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_scf_for(loom_spirv_emit_state_t* state,
                                             const loom_op_t* op) {
  loom_spirv_module_value_ref_t lower_bound = {0};
  IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(
      state, loom_low_scf_for_lower_bound(op), &lower_bound));
  loom_spirv_module_value_ref_t upper_bound = {0};
  IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(
      state, loom_low_scf_for_upper_bound(op), &upper_bound));
  loom_spirv_module_value_ref_t step = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, loom_low_scf_for_step(op), &step));
  IREE_ASSERT_EQ(lower_bound.type_id, upper_bound.type_id);
  IREE_ASSERT_EQ(lower_bound.type_id, step.type_id);
  IREE_ASSERT(loom_spirv_value_type_equal(lower_bound.value_type,
                                          upper_bound.value_type));
  IREE_ASSERT(
      loom_spirv_value_type_equal(lower_bound.value_type, step.value_type));

  const uint32_t preheader_label_id = state->current_label_id;
  const uint32_t header_label_id = loom_spirv_emit_allocate_id(state);
  const uint32_t body_label_id = loom_spirv_emit_allocate_id(state);
  const uint32_t merge_label_id = loom_spirv_emit_allocate_id(state);
  const uint32_t continue_label_id = loom_spirv_emit_allocate_id(state);
  const uint32_t next_iv_id = loom_spirv_emit_allocate_id(state);
  IREE_RETURN_IF_ERROR(loom_spirv_emit_branch_label(state, header_label_id));

  const loom_region_t* body_region = loom_low_scf_for_body(op);
  const loom_block_t* body_block = loom_region_const_entry_block(body_region);
  const loom_value_slice_t iter_args = loom_low_scf_for_iter_args(op);
  const loom_value_slice_t results = loom_low_scf_for_results(op);
  const loom_op_t* yield_op = loom_spirv_emit_region_terminator(body_region);
  loom_value_slice_t yielded_values = {0};
  if (yield_op != NULL && loom_low_scf_yield_isa(yield_op)) {
    yielded_values = loom_low_scf_yield_values(yield_op);
  }
  IREE_ASSERT(body_block != NULL);
  IREE_ASSERT_EQ(body_block->arg_count, (uint16_t)(iter_args.count + 1));
  IREE_ASSERT_EQ(results.count, iter_args.count);
  IREE_ASSERT_EQ(yielded_values.count, iter_args.count);

  const loom_value_id_t iv_value_id = loom_block_arg_id(body_block, 0);
  const uint32_t iv_id = loom_spirv_emit_allocate_id(state);
  const loom_spirv_module_value_ref_t iv_ref = {
      .id = iv_id,
      .type_id = lower_bound.type_id,
      .value_type = lower_bound.value_type,
  };
  loom_spirv_module_value_ref_t* initial_refs = NULL;
  loom_spirv_module_value_ref_t* next_refs = NULL;
  uint32_t* carried_phi_ids = NULL;
  if (iter_args.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->scratch_arena, iter_args.count, sizeof(*initial_refs),
        (void**)&initial_refs));
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->scratch_arena, iter_args.count,
                                  sizeof(*next_refs), (void**)&next_refs));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->scratch_arena, iter_args.count, sizeof(*carried_phi_ids),
        (void**)&carried_phi_ids));
  }
  for (uint16_t i = 0; i < iter_args.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(
        state, iter_args.values[i], &initial_refs[i]));
    carried_phi_ids[i] = loom_spirv_emit_allocate_id(state);
  }
  for (uint16_t i = 0; i < iter_args.count; ++i) {
    const loom_value_id_t yielded_value_id = yielded_values.values[i];
    if (yielded_value_id == iv_value_id) {
      IREE_ASSERT_EQ(initial_refs[i].type_id, iv_ref.type_id);
      IREE_ASSERT(loom_spirv_value_type_equal(initial_refs[i].value_type,
                                              iv_ref.value_type));
      next_refs[i] = iv_ref;
      continue;
    }
    bool yielded_block_arg = false;
    for (uint16_t j = 0; j < iter_args.count; ++j) {
      if (yielded_value_id != loom_block_arg_id(body_block, j + 1)) {
        continue;
      }
      IREE_ASSERT_EQ(initial_refs[i].type_id, initial_refs[j].type_id);
      IREE_ASSERT(loom_spirv_value_type_equal(initial_refs[i].value_type,
                                              initial_refs[j].value_type));
      next_refs[i] = (loom_spirv_module_value_ref_t){
          .id = carried_phi_ids[j],
          .type_id = initial_refs[j].type_id,
          .value_type = initial_refs[j].value_type,
      };
      yielded_block_arg = true;
      break;
    }
    if (yielded_block_arg) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_or_reserve_typed_value(
        state, yielded_value_id, initial_refs[i].type_id,
        initial_refs[i].value_type, &next_refs[i]));
  }

  IREE_RETURN_IF_ERROR(loom_spirv_emit_label_id(state, header_label_id));
  const uint32_t iv_phi_operands[] = {
      lower_bound.type_id, iv_id,      lower_bound.id,
      preheader_label_id,  next_iv_id, continue_label_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_PHI, iv_phi_operands, IREE_ARRAYSIZE(iv_phi_operands)));
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_define_value(state, iv_value_id, iv_ref, true));

  for (uint16_t i = 0; i < iter_args.count; ++i) {
    const loom_value_id_t body_arg_id = loom_block_arg_id(body_block, i + 1);
    const uint32_t body_arg_id_spirv = carried_phi_ids[i];
    const uint32_t operands[] = {
        initial_refs[i].type_id, body_arg_id_spirv, initial_refs[i].id,
        preheader_label_id,      next_refs[i].id,   continue_label_id,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
        LOOM_SPIRV_OP_PHI, operands, IREE_ARRAYSIZE(operands)));
    const loom_spirv_module_value_ref_t body_arg_ref = {
        .id = body_arg_id_spirv,
        .type_id = initial_refs[i].type_id,
        .value_type = initial_refs[i].value_type,
    };
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_define_value(state, body_arg_id, body_arg_ref, true));
  }

  uint32_t bool_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_bool(state->type_context, &bool_type_id));
  const uint32_t condition_id = loom_spirv_emit_allocate_id(state);
  const uint32_t compare_opcode =
      loom_spirv_emit_loop_counter_is_unsigned(lower_bound.value_type)
          ? LOOM_SPIRV_OP_U_LESS_THAN
          : LOOM_SPIRV_OP_S_LESS_THAN;
  const uint32_t compare_operands[] = {
      bool_type_id,
      condition_id,
      iv_id,
      upper_bound.id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      compare_opcode, compare_operands, IREE_ARRAYSIZE(compare_operands)));
  const uint32_t loop_merge_operands[] = {
      merge_label_id,
      continue_label_id,
      loom_spirv_emit_for_loop_control(op),
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_LOOP_MERGE, loop_merge_operands,
      IREE_ARRAYSIZE(loop_merge_operands)));
  const uint32_t branch_operands[] = {
      condition_id,
      body_label_id,
      merge_label_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_BRANCH_CONDITIONAL, branch_operands,
      IREE_ARRAYSIZE(branch_operands)));

  IREE_RETURN_IF_ERROR(loom_spirv_emit_label_id(state, body_label_id));
  loom_spirv_emit_region_yield_t body_yield = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_region_ops(state, body_region, &body_yield));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_branch_label(state, continue_label_id));

  IREE_RETURN_IF_ERROR(loom_spirv_emit_label_id(state, continue_label_id));
  const uint32_t add_operands[] = {
      lower_bound.type_id,
      next_iv_id,
      iv_id,
      step.id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_I_ADD, add_operands, IREE_ARRAYSIZE(add_operands)));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_branch_label(state, header_label_id));

  IREE_RETURN_IF_ERROR(loom_spirv_emit_label_id(state, merge_label_id));
  for (uint16_t i = 0; i < results.count; ++i) {
    loom_spirv_module_value_ref_t body_arg_ref = {0};
    IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(
        state, loom_block_arg_id(body_block, i + 1), &body_arg_ref));
    uint32_t result_id = 0;
    IREE_RETURN_IF_ERROR(loom_spirv_emit_reserve_value_ref(
        state, results.values[i], body_arg_ref.type_id, body_arg_ref.value_type,
        &result_id));
    const uint32_t operands[] = {
        body_arg_ref.type_id,
        result_id,
        body_arg_ref.id,
        header_label_id,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
        LOOM_SPIRV_OP_PHI, operands, IREE_ARRAYSIZE(operands)));
    const loom_spirv_module_value_ref_t result_ref = {
        .id = result_id,
        .type_id = body_arg_ref.type_id,
        .value_type = body_arg_ref.value_type,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_emit_define_value(state, results.values[i],
                                                      result_ref, true));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_low_op(loom_spirv_emit_state_t* state,
                                            const loom_op_t* op) {
  if (loom_low_scf_if_isa(op)) {
    return loom_spirv_emit_scf_if(state, op);
  }
  if (loom_low_scf_for_isa(op)) {
    return loom_spirv_emit_scf_for(state, op);
  }
  if (loom_low_scf_yield_isa(op)) {
    IREE_CHECK_UNREACHABLE("verified SPIR-V low.scf.yield parent");
    return iree_ok_status();
  }
  if (loom_low_resource_isa(op)) {
    IREE_ASSERT_EQ(state->abi_plan.kind,
                   LOOM_SPIRV_MODULE_ABI_PLAN_HAL_KERNEL_RAW_BDA);
    loom_spirv_module_abi_context_t context =
        loom_spirv_emit_abi_context(state);
    return loom_spirv_module_abi_materialize_resource(&context,
                                                      &state->abi_plan, op);
  }
  if (loom_low_storage_reserve_isa(op)) {
    return loom_spirv_module_workgroup_storage_emit_reserve(
        &state->value_domain, &state->workgroup_storage, op);
  }
  if (loom_low_storage_address_isa(op)) {
    loom_spirv_module_value_ref_t value_ref = {0};
    IREE_RETURN_IF_ERROR(loom_spirv_module_workgroup_storage_emit_address(
        &state->value_domain, &state->workgroup_storage, op,
        state->type_context, state->builder, &value_ref));
    return loom_spirv_emit_define_value(
        state, loom_low_storage_address_result(op), value_ref, true);
  }
  if (loom_low_copy_isa(op)) {
    return loom_spirv_emit_copy(state, op);
  }

  loom_low_resolved_descriptor_packet_t packet = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_descriptor_packet(
      state->module, state->target, op, &packet));
  if (packet.kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
    IREE_CHECK_UNREACHABLE("verified SPIR-V structural op");
    return iree_ok_status();
  }
  IREE_ASSERT(packet.descriptor != NULL);
  return loom_spirv_emit_descriptor_packet(state, op, &packet);
}

static iree_status_t loom_spirv_emit_return(loom_spirv_emit_state_t* state,
                                            const loom_op_t* op) {
  loom_spirv_module_abi_context_t context = loom_spirv_emit_abi_context(state);
  IREE_RETURN_IF_ERROR(loom_spirv_module_abi_store_return_values(
      &context, &state->abi_plan, op));
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_RETURN, NULL, 0);
}

static iree_status_t loom_spirv_emit_entry_point(
    loom_spirv_emit_state_t* state) {
  const uint32_t prefix_operands[] = {
      LOOM_SPIRV_EXECUTION_MODEL_GL_COMPUTE,
      state->function_id,
  };
  uint32_t interface_operands[LOOM_SPIRV_BUILTIN_VARIABLE_COUNT] = {0};
  iree_host_size_t interface_operand_count = 0;
  for (iree_host_size_t i = 0; i < LOOM_SPIRV_BUILTIN_VARIABLE_COUNT; ++i) {
    if (state->builtin_interface_variable_ids[i] != 0) {
      interface_operands[interface_operand_count++] =
          state->builtin_interface_variable_ids[i];
    }
  }
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_string_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_ENTRY_POINT),
      LOOM_SPIRV_OP_ENTRY_POINT, prefix_operands,
      IREE_ARRAYSIZE(prefix_operands), loom_spirv_emit_export_name(state),
      interface_operands, interface_operand_count));
  uint32_t workgroup_size_x = 1;
  uint32_t workgroup_size_y = 1;
  uint32_t workgroup_size_z = 1;
  if (loom_low_kernel_def_isa(state->function_op)) {
    const int64_t x = loom_low_kernel_def_workgroup_size_x(state->function_op);
    const int64_t y = loom_low_kernel_def_workgroup_size_y(state->function_op);
    const int64_t z = loom_low_kernel_def_workgroup_size_z(state->function_op);
    workgroup_size_x = x > 0 ? (uint32_t)x : 1u;
    workgroup_size_y = y > 0 ? (uint32_t)y : 1u;
    workgroup_size_z = z > 0 ? (uint32_t)z : 1u;
  }
  const uint32_t execution_mode_operands[] = {
      state->function_id, LOOM_SPIRV_EXECUTION_MODE_LOCAL_SIZE,
      workgroup_size_x,   workgroup_size_y,
      workgroup_size_z,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_EXECUTION_MODE),
      LOOM_SPIRV_OP_EXECUTION_MODE, execution_mode_operands,
      IREE_ARRAYSIZE(execution_mode_operands));
}

static iree_status_t loom_spirv_emit_function_signature(
    loom_spirv_emit_state_t* state, uint32_t* out_result_type_id,
    uint32_t* out_function_type_id) {
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_void(state->type_context, &result_type_id));
  uint32_t function_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_function(
      state->type_context, result_type_id, /*parameter_type_ids=*/NULL,
      /*parameter_count=*/0, &function_type_id));
  *out_result_type_id = result_type_id;
  *out_function_type_id = function_type_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_cfg_op_error(
    loom_spirv_emit_state_t* state, const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(state->module, op);
  const iree_string_view_t op_name =
      vtable != NULL ? loom_op_vtable_name(vtable) : IREE_SV("<unknown>");
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "verified SPIR-V low function still contains CFG op '%.*s'",
      (int)op_name.size, op_name.data);
}

static iree_status_t loom_spirv_emit_function_entry_block(
    loom_spirv_emit_state_t* state, const loom_block_t* block) {
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_label_id(state, loom_spirv_emit_allocate_id(state)));
  loom_spirv_module_abi_context_t context = loom_spirv_emit_abi_context(state);
  IREE_RETURN_IF_ERROR(
      loom_spirv_module_abi_materialize_entry_args(&context, &state->abi_plan));
  for (const loom_op_t* op = block->first_op; op != NULL; op = op->next_op) {
    if (loom_low_return_isa(op)) {
      IREE_RETURN_IF_ERROR(loom_spirv_emit_return(state, op));
    } else if (loom_low_br_isa(op) || loom_low_cond_br_isa(op)) {
      return loom_spirv_emit_cfg_op_error(state, op);
    } else {
      IREE_RETURN_IF_ERROR(loom_spirv_emit_low_op(state, op));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_function_body(
    loom_spirv_emit_state_t* state) {
  if (state->body->block_count != 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "verified SPIR-V low function has %u top-level blocks; expected one",
        (unsigned)state->body->block_count);
  }
  const loom_block_t* block = loom_region_const_entry_block(state->body);
  IREE_RETURN_IF_ERROR(loom_spirv_emit_function_entry_block(state, block));
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_FUNCTION_END, NULL, 0);
}

static iree_status_t loom_spirv_emit_function(loom_spirv_emit_state_t* state) {
  const loom_region_t* body = loom_low_function_const_body(state->function_op);
  IREE_ASSERT(body != NULL);
  IREE_ASSERT_NE(body->block_count, 0u);
  state->body = body;
  const loom_block_t* entry_block = loom_region_const_entry_block(body);
  state->function_id = loom_spirv_emit_allocate_id(state);
  IREE_RETURN_IF_ERROR(loom_spirv_emit_op_name(
      state, state->function_id, loom_spirv_emit_function_name(state)));
  loom_spirv_module_abi_context_t context = loom_spirv_emit_abi_context(state);
  IREE_RETURN_IF_ERROR(loom_spirv_module_abi_build_plan(&context, entry_block,
                                                        &state->abi_plan));

  uint32_t result_type_id = 0;
  uint32_t function_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_function_signature(
      state, &result_type_id, &function_type_id));

  const uint32_t function_operands[] = {
      result_type_id,
      state->function_id,
      LOOM_SPIRV_FUNCTION_CONTROL_NONE,
      function_type_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_FUNCTION, function_operands,
      IREE_ARRAYSIZE(function_operands)));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_function_body(state));
  return loom_spirv_emit_entry_point(state);
}

static iree_status_t loom_spirv_emit_validate_target(
    const loom_low_resolved_target_t* target) {
  if (target->descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "SPIR-V low function target did not resolve to a "
                            "descriptor set");
  }
  if (target->descriptor_set->stable_id !=
      SPIRV_LOGICAL_CORE_DESCRIPTOR_SET_ID) {
    const iree_string_view_t descriptor_set_key =
        loom_low_descriptor_set_string(
            target->descriptor_set, target->descriptor_set->key_string_offset);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "verified SPIR-V low function selected descriptor set '%.*s'; "
        "expected 'spirv.logical.core'",
        (int)descriptor_set_key.size, descriptor_set_key.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_module_prepare_contract(
    loom_spirv_emit_module_state_t* state,
    const loom_low_resolved_target_t* target, iree_allocator_t allocator) {
  const loom_spirv_module_contract_t contract =
      loom_spirv_module_contract_from_target(target);
  if (!iree_any_bit_set(
          state->flags,
          LOOM_SPIRV_EMIT_MODULE_STATE_FLAG_BUILDER_INITIALIZED)) {
    IREE_RETURN_IF_ERROR(loom_spirv_module_builder_initialize(
        &target->bundle_storage.bundle, allocator, &state->builder));
    loom_spirv_type_context_initialize(&state->builder, state->scratch_arena,
                                       &state->type_context);
    state->contract = contract;
    state->flags |= LOOM_SPIRV_EMIT_MODULE_STATE_FLAG_BUILDER_INITIALIZED;
    return iree_ok_status();
  }
  if (!loom_spirv_module_contract_equal(&state->contract, &contract)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "verified SPIR-V low module mixes target contract '%.*s' with "
        "target contract '%.*s'",
        (int)state->contract.target_name.size, state->contract.target_name.data,
        (int)contract.target_name.size, contract.target_name.data);
  }
  return iree_ok_status();
}

static void loom_spirv_emit_function_state_deinitialize(
    loom_spirv_emit_state_t* state) {
  if (loom_local_value_domain_is_acquired(&state->value_domain)) {
    loom_local_value_domain_release(&state->value_domain);
  }
}

static iree_status_t loom_spirv_emit_function_state_initialize(
    loom_spirv_emit_module_state_t* module_state, loom_op_t* low_function_op,
    const loom_low_resolved_target_t* target,
    loom_spirv_emit_state_t* out_state) {
  *out_state = (loom_spirv_emit_state_t){
      .module_state = module_state,
      .module = module_state->module,
      .function_op = low_function_op,
      .target = target,
      .scratch_arena = module_state->scratch_arena,
      .builder = &module_state->builder,
      .type_context = &module_state->type_context,
  };

  const loom_region_t* body = loom_low_function_const_body(low_function_op);
  iree_status_t status = iree_ok_status();
  if (body == NULL || body->block_count == 0) {
    status =
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "verified SPIR-V low function has no function body");
  }
  if (iree_status_is_ok(status)) {
    status = loom_local_value_domain_acquire_for_region_tree(
        module_state->module, body, module_state->scratch_arena,
        &out_state->value_domain);
  }
  if (iree_status_is_ok(status)) {
    status = loom_spirv_module_workgroup_storage_initialize(
        &out_state->value_domain, &out_state->workgroup_storage,
        module_state->scratch_arena);
  }
  if (iree_status_is_ok(status)) {
    status = loom_spirv_module_value_table_initialize(
        &out_state->value_domain, &out_state->value_table,
        module_state->scratch_arena);
  }
  if (iree_status_is_ok(status)) {
    loom_spirv_module_abi_context_t context =
        loom_spirv_emit_abi_context(out_state);
    status = loom_spirv_module_abi_prepare_value_types(
        &context, loom_region_const_entry_block(body), &out_state->abi_plan);
  }
  if (iree_status_is_ok(status)) {
    status = loom_spirv_emit_prepare_immediate_name_ids(out_state);
  }
  if (!iree_status_is_ok(status)) {
    loom_spirv_emit_function_state_deinitialize(out_state);
  }
  return status;
}

static iree_status_t loom_spirv_emit_low_function_into_module(
    loom_spirv_emit_module_state_t* module_state, loom_op_t* low_function_op,
    iree_allocator_t allocator) {
  if (!loom_low_function_def_isa(low_function_op)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "SPIR-V emission requires a low function "
                            "definition");
  }

  loom_low_resolved_target_t target = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target_with_facts(
      module_state->module, &module_state->symbol_facts, low_function_op,
      module_state->descriptor_registry, module_state->target_selection,
      module_state->diagnostic_emitter, &target));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_validate_target(&target));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_module_prepare_contract(
      module_state, &target, allocator));

  loom_spirv_emit_state_t function_state = {0};
  iree_status_t status = loom_spirv_emit_function_state_initialize(
      module_state, low_function_op, &target, &function_state);
  if (iree_status_is_ok(status)) {
    status = loom_spirv_emit_function(&function_state);
  }
  loom_spirv_emit_function_state_deinitialize(&function_state);
  if (iree_status_is_ok(status)) {
    ++module_state->function_count;
  }
  return status;
}

static void loom_spirv_emit_module_state_initialize(
    loom_module_t* module,
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_target_selection_t target_selection,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* scratch_arena,
    loom_spirv_emit_module_state_t* out_state) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(descriptor_registry);
  IREE_ASSERT_ARGUMENT(scratch_arena);

  *out_state = (loom_spirv_emit_module_state_t){
      .module = module,
      .descriptor_registry = descriptor_registry,
      .target_selection = target_selection,
      .diagnostic_emitter = diagnostic_emitter,
      .scratch_arena = scratch_arena,
  };
  loom_symbol_fact_table_initialize(&out_state->symbol_facts, scratch_arena);
}

static void loom_spirv_emit_module_state_deinitialize(
    loom_spirv_emit_module_state_t* state) {
  if (iree_any_bit_set(state->flags,
                       LOOM_SPIRV_EMIT_MODULE_STATE_FLAG_BUILDER_INITIALIZED)) {
    loom_spirv_module_builder_deinitialize(&state->builder);
  }
}

static iree_status_t loom_spirv_emit_module_state_finalize(
    loom_spirv_emit_module_state_t* state,
    loom_spirv_module_binary_t* out_module) {
  if (state->function_count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "SPIR-V low module has no low function definitions");
  }
  loom_spirv_module_abi_context_t context = {
      .module = state->module,
      .scratch_arena = state->scratch_arena,
      .builder = &state->builder,
      .type_context = &state->type_context,
      .raw_bda_layout = &state->raw_bda_layout,
  };
  IREE_RETURN_IF_ERROR(
      loom_spirv_module_abi_emit_metadata(&context, &state->raw_bda_layout));
  return loom_spirv_module_builder_finalize(&state->builder, out_module);
}

static iree_status_t loom_spirv_emit_low_module_initialize(
    loom_module_t* module,
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_target_selection_t target_selection,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* scratch_arena,
    loom_spirv_module_binary_t* out_module,
    loom_spirv_emit_module_state_t* out_state) {
  IREE_ASSERT_ARGUMENT(out_module);

  *out_module = (loom_spirv_module_binary_t){0};
  loom_spirv_emit_module_state_initialize(module, descriptor_registry,
                                          target_selection, diagnostic_emitter,
                                          scratch_arena, out_state);
  return iree_ok_status();
}

void loom_spirv_emit_low_module_options_initialize(
    loom_spirv_emit_low_module_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  *out_options = (loom_spirv_emit_low_module_options_t){0};
}

static iree_status_t loom_spirv_emit_low_module_options_validate(
    const loom_spirv_emit_low_module_options_t* options) {
  if (options == NULL || options->entry_count == 0) {
    return iree_ok_status();
  }
  if (options->entry_ops == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selected SPIR-V low module entries require an entry op list");
  }
  for (iree_host_size_t i = 0; i < options->entry_count; ++i) {
    if (options->entry_ops[i] == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "selected SPIR-V low module entry list contains a null op");
    }
  }
  return iree_ok_status();
}

static bool loom_spirv_emit_low_module_options_selects_entry(
    const loom_spirv_emit_low_module_options_t* options,
    loom_op_t* low_function_op) {
  if (options == NULL || options->entry_count == 0) {
    return true;
  }
  for (iree_host_size_t i = 0; i < options->entry_count; ++i) {
    if (options->entry_ops[i] == low_function_op) {
      return true;
    }
  }
  return false;
}

iree_status_t loom_spirv_emit_low_module(
    loom_module_t* module,
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_target_selection_t target_selection,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* scratch_arena,
    const loom_spirv_emit_low_module_options_t* options,
    loom_spirv_module_binary_t* out_module, iree_allocator_t allocator) {
  loom_spirv_emit_module_state_t state = {0};
  iree_status_t status = iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_spirv_emit_low_module_options_validate(options));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_low_module_initialize(
      module, descriptor_registry, target_selection, diagnostic_emitter,
      scratch_arena, out_module, &state));
  loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(module, symbol) {
    if (!loom_low_function_def_isa(symbol->defining_op) ||
        !loom_spirv_emit_low_module_options_selects_entry(
            options, symbol->defining_op)) {
      continue;
    }
    status = loom_spirv_emit_low_function_into_module(
        &state, symbol->defining_op, allocator);
    if (!iree_status_is_ok(status)) {
      break;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_spirv_emit_module_state_finalize(&state, out_module);
  }
  loom_spirv_emit_module_state_deinitialize(&state);
  if (!iree_status_is_ok(status)) {
    loom_spirv_module_binary_deinitialize(out_module, allocator);
  }
  return status;
}
