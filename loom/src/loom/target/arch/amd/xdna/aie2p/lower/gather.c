// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/lower/gather.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "loom/codegen/low/lower/module_state.h"
#include "loom/codegen/low/lower/source_memory.h"
#include "loom/ir/module.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/gather.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/rodata.h"

typedef struct loom_aie2p_gather_table_state_t {
  // True once this original symbol has one planned derived pair.
  bool initialized;
  // Interned source name used to name the emitted pair.
  iree_string_view_t source_name;
  // Derived table copy used by the AB pointer lanes, null until emission.
  loom_symbol_ref_t ab_symbol;
  // Derived table copy used by the CD pointer lanes, null until emission.
  loom_symbol_ref_t cd_symbol;
  // Copied block-replicated bytes shared by both derived definitions.
  iree_const_byte_span_t derived_contents;
  // Source location retained for both derived definitions.
  loom_location_id_t location;
} loom_aie2p_gather_table_state_t;

typedef struct loom_aie2p_gather_plan_t {
  // Module-owned derived pair shared by all lookups of the same source table.
  loom_aie2p_gather_table_state_t* table;
  // Bit width selected by the source payload type.
  uint8_t element_bit_count;
} loom_aie2p_gather_plan_t;

typedef struct loom_aie2p_gather_module_state_t {
  // Entries indexed by original module symbol ID.
  loom_aie2p_gather_table_state_t* tables;
  // Number of addressable original module symbol IDs in tables.
  iree_host_size_t table_count;
} loom_aie2p_gather_module_state_t;

static int loom_aie2p_gather_module_state_key;

static iree_status_t loom_aie2p_gather_module_state_from_context(
    loom_low_lower_context_t* context,
    loom_aie2p_gather_module_state_t** out_state) {
  *out_state = NULL;
  loom_aie2p_gather_module_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_module_target_state(
      context, &loom_aie2p_gather_module_state_key, sizeof(*state),
      (void**)&state));
  if (state->tables == NULL) {
    loom_module_t* module = loom_low_lower_context_module(context);
    state->table_count = module->symbols.count;
    if (state->table_count != 0) {
      IREE_RETURN_IF_ERROR(loom_low_lower_module_state_allocate_array(
          loom_low_lower_context_module_state(context), state->table_count,
          sizeof(*state->tables), (void**)&state->tables));
      memset(state->tables, 0, state->table_count * sizeof(*state->tables));
    }
  }
  *out_state = state;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_gather_module_state_from_module_state(
    loom_low_lower_module_state_t* module_state,
    loom_aie2p_gather_module_state_t** out_state) {
  *out_state = NULL;
  return loom_low_lower_module_state_get_or_allocate(
      module_state, &loom_aie2p_gather_module_state_key,
      sizeof(loom_aie2p_gather_module_state_t), (void**)out_state);
}

static iree_status_t loom_aie2p_gather_reserve_derived_symbol(
    loom_module_t* module, loom_low_lower_module_state_t* module_state,
    iree_string_view_t source_name, iree_string_view_t suffix,
    loom_symbol_ref_t* out_symbol) {
  *out_symbol = loom_symbol_ref_null();
  iree_host_size_t name_capacity = 0;
  if (!iree_host_size_checked_add(source_name.size, suffix.size + 16,
                                  &name_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P gather symbol name overflow");
  }
  char* name_storage = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_module_state_allocate_bytes(
      module_state, name_capacity, (void**)&name_storage));
  memcpy(name_storage, source_name.data, source_name.size);
  memcpy(name_storage + source_name.size, suffix.data, suffix.size);

  for (uint32_t discriminator = 0; discriminator < LOOM_SYMBOL_ID_INVALID;
       ++discriminator) {
    iree_host_size_t name_length = source_name.size + suffix.size;
    if (discriminator != 0) {
      const int suffix_length =
          snprintf(name_storage + name_length, name_capacity - name_length,
                   "$%" PRIu32, discriminator);
      if (suffix_length < 0 ||
          (iree_host_size_t)suffix_length >= name_capacity - name_length) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "AIE2P gather symbol suffix overflow");
      }
      name_length += (iree_host_size_t)suffix_length;
    }
    const iree_string_view_t name =
        iree_make_string_view(name_storage, name_length);
    const loom_string_id_t existing_name_id =
        loom_module_lookup_string(module, name);
    if (existing_name_id != LOOM_STRING_ID_INVALID &&
        loom_module_find_symbol(module, existing_name_id) !=
            LOOM_SYMBOL_ID_INVALID) {
      continue;
    }
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(module, name, &name_id));
    out_symbol->module_id = 0;
    return loom_module_add_symbol(module, name_id, &out_symbol->symbol_id);
  }
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "could not reserve a unique AIE2P gather symbol");
}

static iree_status_t loom_aie2p_gather_derive_contents(
    loom_low_lower_module_state_t* module_state,
    iree_const_byte_span_t source_contents,
    iree_const_byte_span_t* out_derived_contents) {
  *out_derived_contents = iree_const_byte_span_empty();
  IREE_ASSERT_EQ(source_contents.data_length % 32, 0u);
  const iree_host_size_t derived_length = source_contents.data_length * 2;
  uint8_t* derived_data = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_module_state_allocate_bytes(
      module_state, derived_length, (void**)&derived_data));
  for (iree_host_size_t source_offset = 0;
       source_offset < source_contents.data_length; source_offset += 32) {
    const iree_host_size_t derived_offset = source_offset * 2;
    memcpy(derived_data + derived_offset, source_contents.data + source_offset,
           32);
    memcpy(derived_data + derived_offset + 32,
           source_contents.data + source_offset, 32);
  }
  *out_derived_contents =
      iree_make_const_byte_span(derived_data, derived_length);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_gather_get_or_create_table(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_aie2p_immutable_gather_match_t* match,
    loom_aie2p_gather_table_state_t** out_table) {
  *out_table = NULL;
  loom_aie2p_gather_module_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_gather_module_state_from_context(context, &state));
  IREE_ASSERT_LT(match->source_symbol.symbol_id, state->table_count);
  loom_aie2p_gather_table_state_t* table =
      &state->tables[match->source_symbol.symbol_id];
  if (!table->initialized) {
    loom_module_t* module = loom_low_lower_context_module(context);
    loom_low_lower_module_state_t* module_state =
        loom_low_lower_context_module_state(context);
    IREE_RETURN_IF_ERROR(loom_aie2p_gather_derive_contents(
        module_state, match->source_contents, &table->derived_contents));
    const loom_symbol_t* source_symbol =
        &module->symbols.entries[match->source_symbol.symbol_id];
    table->source_name =
        loom_string_table_get(&module->strings, source_symbol->name_id);
    table->ab_symbol = loom_symbol_ref_null();
    table->cd_symbol = loom_symbol_ref_null();
    table->location = source_op->location;
    table->initialized = true;
  }
  *out_table = table;
  return iree_ok_status();
}

iree_status_t loom_aie2p_select_gather_plan(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  if (!loom_vector_gather_isa(source_op)) {
    return iree_ok_status();
  }

  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  const loom_low_source_memory_access_plan_t* access =
      loom_low_lower_source_memory_access(context, source_op, &diagnostic);
  loom_aie2p_immutable_gather_match_t match = {0};
  if (!loom_aie2p_match_immutable_gather(
          loom_low_lower_context_module(context),
          loom_low_lower_context_fact_table(context), source_op, access,
          &match)) {
    return iree_ok_status();
  }

  loom_aie2p_gather_table_state_t* table = NULL;
  IREE_RETURN_IF_ERROR(loom_aie2p_gather_get_or_create_table(context, source_op,
                                                             &match, &table));
  loom_aie2p_gather_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  *plan = (loom_aie2p_gather_plan_t){
      .table = table,
      .element_bit_count = match.element_bit_count,
  };
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, AIE2P_CORE_REG_CLASS_ID_AIE2P_VEC256, 2, &result_type));
  IREE_RETURN_IF_ERROR(loom_low_lower_plan_value_type(
      context, loom_vector_gather_result(source_op), result_type));
  *out_plan = loom_low_lower_plan_make(LOOM_AIE2P_PLAN_GATHER_IMMUTABLE, plan);
  return iree_ok_status();
}

void loom_aie2p_mark_gather_plan_demands(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_low_lower_plan_t plan) {
  (void)plan;
  loom_low_lower_require_source_value_storage(
      context, loom_vector_gather_offsets(source_op));
}

void loom_aie2p_describe_gather_plan(loom_low_lower_context_t* context,
                                     const loom_op_t* source_op,
                                     loom_low_lower_plan_t plan,
                                     loom_low_lower_plan_report_t* out_report) {
  (void)context;
  (void)source_op;
  const loom_aie2p_gather_plan_t* gather_plan =
      (const loom_aie2p_gather_plan_t*)plan.target_data;
  iree_string_view_t plan_key = iree_string_view_empty();
  switch (gather_plan->element_bit_count) {
    case 16:
      plan_key = IREE_SV("read-only-data.gather.vldb16x32");
      break;
    case 32:
      plan_key = IREE_SV("read-only-data.gather.vldb32x16");
      break;
    case 64:
      plan_key = IREE_SV("read-only-data.gather.vldb64x8");
      break;
    default:
      IREE_ASSERT_UNREACHABLE("matched immutable gather width");
      break;
  }
  *out_report = (loom_low_lower_plan_report_t){
      .plan_key = plan_key,
  };
}

static iree_status_t loom_aie2p_gather_make_register_type(
    loom_low_lower_context_t* context, uint16_t register_class_id,
    uint32_t unit_count, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(context, register_class_id,
                                           unit_count, out_type);
}

static iree_status_t loom_aie2p_gather_emit_descriptor(
    loom_low_lower_context_t* context, uint32_t descriptor_ordinal,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor = &loom_low_lower_context_descriptor_set(context)
                         ->descriptors[descriptor_ordinal],
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, operand_count,
      loom_named_attr_slice_empty(), &result_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, location, &low_op));
  *out_result = loom_low_op_results(low_op).values[0];
  return iree_ok_status();
}

static iree_status_t loom_aie2p_gather_emit_constant(
    loom_low_lower_context_t* context, uint32_t descriptor_ordinal,
    int64_t value, loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  loom_string_id_t immediate_name = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      loom_low_lower_context_module(context), IREE_SV("i"), &immediate_name));
  const loom_named_attr_t immediate = {
      .name_id = immediate_name,
      .value = loom_attr_i64(value),
  };
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor = &loom_low_lower_context_descriptor_set(context)
                         ->descriptors[descriptor_ordinal],
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
      context, &descriptor, loom_make_named_attr_slice(&immediate, 1),
      result_type, location, &low_op));
  *out_result = loom_low_const_result(low_op);
  return iree_ok_status();
}

typedef struct loom_aie2p_gather_physical_recipe_t {
  // Number of doublings that convert logical indices to lookup addresses.
  uint8_t address_shift;
  // Number of 16-index groups consumed from the offsets carrier.
  uint8_t index_group_count;
  // Number of 256-bit address halves read for each index group.
  uint8_t address_half_count;
  // Descriptor used to load the lower four 64-bit lookup windows.
  uint32_t load_lo_descriptor;
  // Descriptor used to load the upper four 64-bit lookup windows.
  uint32_t load_hi_descriptor;
  // Shuffle mode selecting payload elements, or -1 for direct concatenation.
  int8_t shuffle_mode;
} loom_aie2p_gather_physical_recipe_t;

static loom_aie2p_gather_physical_recipe_t loom_aie2p_gather_physical_recipe(
    uint8_t element_bit_count) {
  switch (element_bit_count) {
    case 16:
      return (loom_aie2p_gather_physical_recipe_t){
          .address_shift = 2,
          .index_group_count = 2,
          .address_half_count = 2,
          .load_lo_descriptor = AIE2P_CORE_DESCRIPTOR_REF_LOAD_B_LOOKUP_4X16_LO,
          .load_hi_descriptor = AIE2P_CORE_DESCRIPTOR_REF_LOAD_B_LOOKUP_4X16_HI,
          .shuffle_mode = 24,
      };
    case 32:
      return (loom_aie2p_gather_physical_recipe_t){
          .address_shift = 3,
          .index_group_count = 1,
          .address_half_count = 2,
          .load_lo_descriptor = AIE2P_CORE_DESCRIPTOR_REF_LOAD_B_LOOKUP_4X32_LO,
          .load_hi_descriptor = AIE2P_CORE_DESCRIPTOR_REF_LOAD_B_LOOKUP_4X32_HI,
          .shuffle_mode = 4,
      };
    case 64:
      return (loom_aie2p_gather_physical_recipe_t){
          .address_shift = 4,
          .index_group_count = 1,
          .address_half_count = 1,
          .load_lo_descriptor = AIE2P_CORE_DESCRIPTOR_REF_LOAD_B_LOOKUP_4X64_LO,
          .load_hi_descriptor = AIE2P_CORE_DESCRIPTOR_REF_LOAD_B_LOOKUP_4X64_HI,
          .shuffle_mode = -1,
      };
    default:
      IREE_ASSERT_UNREACHABLE("matched immutable gather width");
      return (loom_aie2p_gather_physical_recipe_t){0};
  }
}

// Symbol identities are emission results. Planning retains the table contents
// and placement relationship without publishing symbols in the source module.
static iree_status_t loom_aie2p_gather_publish_symbols(
    loom_module_t* module, loom_low_lower_module_state_t* module_state,
    loom_aie2p_gather_table_state_t* table) {
  if (table->ab_symbol.symbol_id != LOOM_SYMBOL_ID_INVALID) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_gather_reserve_derived_symbol(
      module, module_state, table->source_name, IREE_SV("$aie2p$lookup$ab"),
      &table->ab_symbol));
  return loom_aie2p_gather_reserve_derived_symbol(
      module, module_state, table->source_name, IREE_SV("$aie2p$lookup$cd"),
      &table->cd_symbol);
}

iree_status_t loom_aie2p_emit_gather_plan(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_low_lower_plan_t plan) {
  const loom_aie2p_gather_plan_t* gather_plan =
      (const loom_aie2p_gather_plan_t*)plan.target_data;
  IREE_RETURN_IF_ERROR(loom_aie2p_gather_publish_symbols(
      loom_low_lower_context_module(context),
      loom_low_lower_context_module_state(context), gather_plan->table));
  const loom_aie2p_gather_physical_recipe_t physical_recipe =
      loom_aie2p_gather_physical_recipe(gather_plan->element_bit_count);
  const loom_location_id_t location = source_op->location;
  loom_type_t scalar_type = loom_type_none();
  loom_type_t select_type = loom_type_none();
  loom_type_t vector_half_type = loom_type_none();
  const loom_type_t vector_type = loom_low_lower_value_binding_type(
      context, loom_vector_gather_result(source_op));
  IREE_RETURN_IF_ERROR(loom_aie2p_gather_make_register_type(
      context, AIE2P_CORE_REG_CLASS_ID_AIE2P_ER, 1, &scalar_type));
  IREE_RETURN_IF_ERROR(loom_aie2p_gather_make_register_type(
      context, AIE2P_CORE_REG_CLASS_ID_AIE2P_ERS16, 1, &select_type));
  IREE_RETURN_IF_ERROR(loom_aie2p_gather_make_register_type(
      context, AIE2P_CORE_REG_CLASS_ID_AIE2P_VEC256, 1, &vector_half_type));

  loom_value_id_t ab_pointer = LOOM_VALUE_ID_INVALID;
  loom_value_id_t cd_pointer = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_emit_rodata_address(
      context, gather_plan->table->ab_symbol, location, &ab_pointer));
  IREE_RETURN_IF_ERROR(loom_aie2p_emit_rodata_address(
      context, gather_plan->table->cd_symbol, location, &cd_pointer));

  loom_value_id_t ab_scalar = LOOM_VALUE_ID_INVALID;
  loom_value_id_t cd_scalar = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_gather_emit_descriptor(
      context, AIE2P_CORE_DESCRIPTOR_REF_MOVE_LOCAL_ADDRESS_TO_SCALAR,
      &ab_pointer, 1, scalar_type, location, &ab_scalar));
  IREE_RETURN_IF_ERROR(loom_aie2p_gather_emit_descriptor(
      context, AIE2P_CORE_DESCRIPTOR_REF_MOVE_LOCAL_ADDRESS_TO_SCALAR,
      &cd_pointer, 1, scalar_type, location, &cd_scalar));

  loom_value_id_t ab_vector = LOOM_VALUE_ID_INVALID;
  loom_value_id_t cd_vector = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_gather_emit_descriptor(
      context, AIE2P_CORE_DESCRIPTOR_REF_SPLAT_I32X16, &ab_scalar, 1,
      vector_type, location, &ab_vector));
  IREE_RETURN_IF_ERROR(loom_aie2p_gather_emit_descriptor(
      context, AIE2P_CORE_DESCRIPTOR_REF_SPLAT_I32X16, &cd_scalar, 1,
      vector_type, location, &cd_vector));

  loom_value_id_t select_control_scalar = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_gather_emit_constant(
      context, AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32, 0xCCCC, scalar_type,
      location, &select_control_scalar));
  loom_op_t* select_control_copy = NULL;
  IREE_RETURN_IF_ERROR(loom_low_copy_build(
      loom_low_lower_context_builder(context), select_control_scalar,
      /*detached=*/false, select_type, location, &select_control_copy));
  const loom_value_id_t select_control =
      loom_low_copy_result(select_control_copy);
  const loom_value_id_t select_operands[] = {ab_vector, cd_vector,
                                             select_control};
  loom_value_id_t lane_bases = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_gather_emit_descriptor(
      context, AIE2P_CORE_DESCRIPTOR_REF_SELECT_I32X16, select_operands,
      IREE_ARRAYSIZE(select_operands), vector_type, location, &lane_bases));

  loom_value_id_t source_offsets = loom_low_lower_lookup_value(
      context, loom_vector_gather_offsets(source_op));

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_value_id_t shuffle_control = LOOM_VALUE_ID_INVALID;
  if (physical_recipe.shuffle_mode >= 0) {
    IREE_RETURN_IF_ERROR(loom_aie2p_gather_emit_constant(
        context, AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_MOVA,
        physical_recipe.shuffle_mode, scalar_type, location, &shuffle_control));
  }

  loom_value_id_t group_results[2] = {LOOM_VALUE_ID_INVALID,
                                      LOOM_VALUE_ID_INVALID};
  for (uint32_t group = 0; group < physical_recipe.index_group_count; ++group) {
    loom_value_id_t scaled_offsets = source_offsets;
    if (physical_recipe.index_group_count > 1) {
      loom_op_t* offset_slice = NULL;
      IREE_RETURN_IF_ERROR(loom_low_slice_build(builder, source_offsets,
                                                group * 2, vector_type,
                                                location, &offset_slice));
      scaled_offsets = loom_low_slice_result(offset_slice);
    }
    for (uint32_t shift = 0; shift < physical_recipe.address_shift; ++shift) {
      const loom_value_id_t add_operands[] = {scaled_offsets, scaled_offsets};
      IREE_RETURN_IF_ERROR(loom_aie2p_gather_emit_descriptor(
          context, AIE2P_CORE_DESCRIPTOR_REF_ADD_I32X16, add_operands,
          IREE_ARRAYSIZE(add_operands), vector_type, location,
          &scaled_offsets));
    }
    const loom_value_id_t address_operands[] = {lane_bases, scaled_offsets};
    loom_value_id_t addresses = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_gather_emit_descriptor(
        context, AIE2P_CORE_DESCRIPTOR_REF_ADD_I32X16, address_operands,
        IREE_ARRAYSIZE(address_operands), vector_type, location, &addresses));

    loom_value_id_t loaded_vectors[2] = {LOOM_VALUE_ID_INVALID,
                                         LOOM_VALUE_ID_INVALID};
    for (uint32_t half = 0; half < physical_recipe.address_half_count; ++half) {
      loom_op_t* address_slice = NULL;
      IREE_RETURN_IF_ERROR(loom_low_slice_build(builder, addresses, half,
                                                vector_half_type, location,
                                                &address_slice));
      const loom_value_id_t address_half = loom_low_slice_result(address_slice);
      loom_value_id_t loaded_halves[2] = {LOOM_VALUE_ID_INVALID,
                                          LOOM_VALUE_ID_INVALID};
      IREE_RETURN_IF_ERROR(loom_aie2p_gather_emit_descriptor(
          context, physical_recipe.load_lo_descriptor, &address_half, 1,
          vector_half_type, location, &loaded_halves[0]));
      IREE_RETURN_IF_ERROR(loom_aie2p_gather_emit_descriptor(
          context, physical_recipe.load_hi_descriptor, &address_half, 1,
          vector_half_type, location, &loaded_halves[1]));
      loom_op_t* loaded_concat = NULL;
      IREE_RETURN_IF_ERROR(loom_low_concat_build(
          builder, loaded_halves, IREE_ARRAYSIZE(loaded_halves), vector_type,
          location, &loaded_concat));
      loaded_vectors[half] = loom_low_concat_result(loaded_concat);
    }

    if (physical_recipe.shuffle_mode >= 0) {
      const loom_value_id_t shuffle_operands[] = {
          loaded_vectors[0], loaded_vectors[1], shuffle_control};
      IREE_RETURN_IF_ERROR(loom_aie2p_gather_emit_descriptor(
          context, AIE2P_CORE_DESCRIPTOR_REF_SHUFFLE_X_CONFIGURED,
          shuffle_operands, IREE_ARRAYSIZE(shuffle_operands), vector_type,
          location, &group_results[group]));
    } else {
      group_results[group] = loaded_vectors[0];
    }
  }

  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  if (physical_recipe.index_group_count == 1) {
    result = group_results[0];
  } else {
    loom_value_id_t result_halves[2] = {LOOM_VALUE_ID_INVALID,
                                        LOOM_VALUE_ID_INVALID};
    for (int64_t group = 0; group < 2; ++group) {
      loom_op_t* result_slice = NULL;
      IREE_RETURN_IF_ERROR(loom_low_slice_build(builder, group_results[group],
                                                /*offset=*/0, vector_half_type,
                                                location, &result_slice));
      result_halves[group] = loom_low_slice_result(result_slice);
    }
    loom_op_t* result_concat = NULL;
    IREE_RETURN_IF_ERROR(loom_low_concat_build(
        builder, result_halves, IREE_ARRAYSIZE(result_halves), vector_type,
        location, &result_concat));
    result = loom_low_concat_result(result_concat);
  }
  return loom_low_lower_bind_value(
      context, loom_vector_gather_result(source_op), result);
}

static iree_status_t loom_aie2p_gather_emit_derived_definition(
    loom_builder_t* builder, loom_symbol_ref_t symbol,
    loom_symbol_ref_t conflicting_symbol, iree_const_byte_span_t contents,
    loom_location_id_t location) {
  // Each lookup block contains two duplicated 32-byte bank halves. The pair
  // must begin at a 64-byte boundary to preserve lookup addressing.
  loom_op_t* definition = NULL;
  return loom_global_rodata_def_build(
      builder,
      LOOM_GLOBAL_RODATA_DEF_BUILD_FLAG_HAS_ALIGNMENT |
          LOOM_GLOBAL_RODATA_DEF_BUILD_FLAG_HAS_BANK_CONFLICTS,
      /*visibility=*/0, /*linkage=*/0, symbol, /*alignment=*/64,
      loom_make_symbol_ref_array(&conflicting_symbol, 1), contents, location,
      &definition);
}

iree_status_t loom_aie2p_finalize_gather_module(
    loom_module_t* module, loom_low_lower_module_state_t* module_state,
    iree_arena_allocator_t* scratch_arena) {
  (void)scratch_arena;
  loom_aie2p_gather_module_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_gather_module_state_from_module_state(module_state, &state));
  if (state->tables == NULL) {
    return iree_ok_status();
  }

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  for (iree_host_size_t i = 0; i < state->table_count; ++i) {
    loom_aie2p_gather_table_state_t* table = &state->tables[i];
    if (!table->initialized) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_aie2p_gather_publish_symbols(module, module_state, table));
    IREE_ASSERT(
        module->symbols.entries[table->ab_symbol.symbol_id].defining_op ==
        NULL);
    IREE_ASSERT(
        module->symbols.entries[table->cd_symbol.symbol_id].defining_op ==
        NULL);
    IREE_RETURN_IF_ERROR(loom_aie2p_gather_emit_derived_definition(
        &builder, table->ab_symbol, table->cd_symbol, table->derived_contents,
        table->location));
    IREE_RETURN_IF_ERROR(loom_aie2p_gather_emit_derived_definition(
        &builder, table->cd_symbol, table->ab_symbol, table->derived_contents,
        table->location));
  }
  return iree_ok_status();
}
