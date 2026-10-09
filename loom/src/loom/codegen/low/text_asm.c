// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/text_asm.h"

#include <inttypes.h>

#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/repr.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"

static const loom_low_descriptor_registry_t*
loom_low_descriptor_text_asm_state_registry(
    const loom_text_low_asm_environment_state_t* state) {
  return (const loom_low_descriptor_registry_t*)state;
}

static const loom_low_descriptor_text_asm_environment_storage_t*
loom_low_descriptor_text_asm_state_storage(
    const loom_text_low_asm_environment_state_t* state) {
  return (const loom_low_descriptor_text_asm_environment_storage_t*)state;
}

static const loom_text_low_asm_descriptor_set_t*
loom_low_descriptor_text_asm_descriptor_set_handle(
    const loom_low_descriptor_set_t* descriptor_set) {
  return (const loom_text_low_asm_descriptor_set_t*)descriptor_set;
}

static const loom_low_descriptor_set_t*
loom_low_descriptor_text_asm_descriptor_set(
    const loom_text_low_asm_descriptor_set_t* descriptor_set) {
  return (const loom_low_descriptor_set_t*)descriptor_set;
}

static const loom_text_low_asm_form_t* loom_low_descriptor_text_asm_form_handle(
    const loom_low_asm_form_t* asm_form) {
  return (const loom_text_low_asm_form_t*)asm_form;
}

static const loom_low_asm_form_t* loom_low_descriptor_text_asm_form(
    const loom_text_low_asm_form_t* asm_form) {
  return (const loom_low_asm_form_t*)asm_form;
}

static const loom_text_low_asm_descriptor_handle_t*
loom_low_descriptor_text_asm_descriptor_handle(
    const loom_low_descriptor_t* descriptor) {
  return (const loom_text_low_asm_descriptor_handle_t*)descriptor;
}

static const loom_low_descriptor_t* loom_low_descriptor_text_asm_descriptor(
    const loom_text_low_asm_descriptor_handle_t* descriptor) {
  return (const loom_low_descriptor_t*)descriptor;
}

static void loom_low_descriptor_text_asm_make_packet(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_asm_form_t* asm_form,
    const loom_low_descriptor_t* descriptor,
    loom_text_low_asm_packet_descriptor_t* out_packet) {
  const loom_low_asm_layout_t* layout =
      &descriptor_set->asm_layouts[asm_form->layout_index];
  const bool builds_as_const =
      descriptor->op_kind == LOOM_LOW_DESCRIPTOR_OP_KIND_CONST;
  *out_packet = (loom_text_low_asm_packet_descriptor_t){
      .descriptor_set =
          loom_low_descriptor_text_asm_descriptor_set_handle(descriptor_set),
      .form = loom_low_descriptor_text_asm_form_handle(asm_form),
      .descriptor = loom_low_descriptor_text_asm_descriptor_handle(descriptor),
      .descriptor_key = loom_low_descriptor_set_string(
          descriptor_set, descriptor->key_string_ref),
      .mnemonic = loom_low_descriptor_set_string(descriptor_set,
                                                 asm_form->mnemonic_string_ref),
      .result_count = layout->result_operand_index_count,
      .minimum_operand_count = descriptor->minimum_packet_operand_count,
      .operand_segment_count = layout->operand_segment_count,
      .has_variadic_operands =
          loom_low_descriptor_has_variadic_operands(descriptor),
      .asm_immediate_count = layout->explicit_immediate_count,
      .immediate_count = descriptor->immediate_count,
      .immediate_attribute_field_index =
          builds_as_const ? loom_low_const_attrs_diagnostic_ref().index
                          : loom_low_op_attrs_diagnostic_ref().index,
      .has_named_immediates = iree_any_bit_set(
          layout->flags, LOOM_LOW_ASM_LAYOUT_FLAG_NAMED_IMMEDIATES),
      .requires_named_immediates = iree_any_bit_set(
          layout->flags, LOOM_LOW_ASM_LAYOUT_FLAG_REQUIRED_NAMED_IMMEDIATES),
      .operation_kind = builds_as_const ? LOOM_OP_LOW_CONST : LOOM_OP_LOW_OP,
  };
}

static void loom_low_descriptor_text_asm_lookup_packet(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_descriptor_set_t* descriptor_set_handle,
    iree_string_view_t mnemonic,
    loom_text_low_asm_packet_descriptor_t* out_packet) {
  (void)state;
  *out_packet = (loom_text_low_asm_packet_descriptor_t){0};
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(descriptor_set_handle);

  uint32_t asm_form_ordinal =
      loom_low_descriptor_set_lookup_asm_form(descriptor_set, mnemonic);
  if (asm_form_ordinal == LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
    return;
  }
  const loom_low_asm_form_t* asm_form =
      &descriptor_set->asm_forms[asm_form_ordinal];
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[asm_form->descriptor_ordinal];
  loom_low_descriptor_text_asm_make_packet(descriptor_set, asm_form, descriptor,
                                           out_packet);
}

static iree_status_t loom_low_descriptor_text_asm_diagnose_unknown_packet(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_descriptor_set_t* descriptor_set_handle,
    iree_string_view_t packet_name,
    loom_text_low_asm_diagnostic_t* out_diagnostic) {
  *out_diagnostic = (loom_text_low_asm_diagnostic_t){0};
  const loom_low_descriptor_text_asm_environment_storage_t* storage =
      loom_low_descriptor_text_asm_state_storage(state);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(descriptor_set_handle);
  if (storage->diagnostic_provider_list.count != 0 &&
      storage->diagnostic_provider_list.values == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low asm diagnostic provider list has no provider table");
  }
  for (iree_host_size_t i = 0; i < storage->diagnostic_provider_list.count;
       ++i) {
    const loom_target_low_asm_diagnostic_provider_t* provider =
        storage->diagnostic_provider_list.values[i];
    if (provider == NULL || provider->try_unknown_packet == NULL) {
      continue;
    }
    IREE_RETURN_IF_ERROR(provider->try_unknown_packet(
        provider, descriptor_set, packet_name, out_diagnostic));
    if (out_diagnostic->error != NULL) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static const loom_low_operand_t* loom_low_descriptor_text_asm_result_operand(
    const loom_text_low_asm_packet_descriptor_t* packet, uint16_t result_index,
    uint16_t* out_descriptor_operand_index) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(packet->descriptor_set);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_text_asm_descriptor(packet->descriptor);
  const loom_low_asm_form_t* asm_form =
      loom_low_descriptor_text_asm_form(packet->form);
  const loom_low_asm_layout_t* layout =
      &descriptor_set->asm_layouts[asm_form->layout_index];

  const uint32_t asm_operand_index =
      layout->result_operand_index_start + result_index;
  const uint16_t descriptor_operand_index =
      descriptor_set->asm_operand_indices[asm_operand_index];
  *out_descriptor_operand_index = descriptor_operand_index;
  return loom_low_descriptor_operand(descriptor_set, descriptor,
                                     descriptor_operand_index);
}

static bool loom_low_descriptor_text_asm_find_packet_operand_index(
    const loom_text_low_asm_packet_descriptor_t* packet,
    uint16_t descriptor_operand_index, uint16_t* out_packet_operand_index) {
  *out_packet_operand_index = 0;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(packet->descriptor_set);
  const loom_low_asm_form_t* asm_form =
      loom_low_descriptor_text_asm_form(packet->form);
  const loom_low_asm_layout_t* layout =
      &descriptor_set->asm_layouts[asm_form->layout_index];
  for (uint16_t i = 0; i < layout->operand_index_count; ++i) {
    const uint32_t asm_operand_index = layout->operand_index_start + i;
    if (descriptor_set->asm_operand_indices[asm_operand_index] ==
        descriptor_operand_index) {
      *out_packet_operand_index = i;
      return true;
    }
  }
  return false;
}

static bool loom_low_descriptor_text_asm_find_packet_result_index(
    const loom_text_low_asm_packet_descriptor_t* packet,
    uint16_t descriptor_operand_index, uint16_t* out_packet_result_index) {
  *out_packet_result_index = 0;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(packet->descriptor_set);
  const loom_low_asm_form_t* asm_form =
      loom_low_descriptor_text_asm_form(packet->form);
  const loom_low_asm_layout_t* layout =
      &descriptor_set->asm_layouts[asm_form->layout_index];
  for (uint16_t i = 0; i < packet->result_count; ++i) {
    const uint32_t asm_operand_index = layout->result_operand_index_start + i;
    if (descriptor_set->asm_operand_indices[asm_operand_index] ==
        descriptor_operand_index) {
      *out_packet_result_index = i;
      return true;
    }
  }
  return false;
}

static bool loom_low_descriptor_text_asm_constraint_ties_result_type(
    loom_low_constraint_kind_t kind) {
  return kind == LOOM_LOW_CONSTRAINT_KIND_TIED ||
         kind == LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE;
}

static iree_status_t loom_low_descriptor_text_asm_find_tied_packet_operand(
    const loom_text_low_asm_packet_descriptor_t* packet,
    uint16_t result_descriptor_operand_index, bool* out_found,
    uint16_t* out_packet_operand_index) {
  *out_found = false;
  *out_packet_operand_index = 0;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(packet->descriptor_set);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_text_asm_descriptor(packet->descriptor);
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const uint32_t constraint_index = descriptor->constraint_start + i;
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[constraint_index];
    if (!loom_low_descriptor_text_asm_constraint_ties_result_type(
            constraint->kind)) {
      continue;
    }
    if (constraint->lhs_operand_index != result_descriptor_operand_index) {
      continue;
    }
    uint16_t packet_operand_index = 0;
    if (!loom_low_descriptor_text_asm_find_packet_operand_index(
            packet, constraint->rhs_operand_index, &packet_operand_index)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low asm result tie references an operand outside the asm form");
    }
    *out_found = true;
    *out_packet_operand_index = packet_operand_index;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_append_reg_type(
    iree_string_builder_t* builder,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_operand_t* operand, uint16_t reg_class_id) {
  const iree_string_view_t reg_class_name = loom_low_descriptor_set_string(
      descriptor_set,
      descriptor_set->reg_classes[reg_class_id].name_string_ref);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "reg<"));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, reg_class_name));
  if (operand->unit_count != 1) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " x%u", (unsigned)operand->unit_count));
  }
  return iree_string_builder_append_cstring(builder, ">");
}

static iree_status_t loom_low_descriptor_text_asm_append_expected_result_types(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_operand_t* operand, iree_string_view_t prefix,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(builder, prefix));
  uint16_t appended_count = 0;
  for (uint16_t i = 0; i < operand->reg_class_alt_count; ++i) {
    const uint32_t alt_index = operand->reg_class_alt_start + i;
    const loom_low_reg_class_alt_t* alternative =
        &descriptor_set->reg_class_alts[alt_index];
    if (alternative->reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
        iree_all_bits_set(alternative->flags,
                          LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
      continue;
    }
    if (appended_count > 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, " | "));
    }
    IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_append_reg_type(
        builder, descriptor_set, operand, alternative->reg_class_id));
    ++appended_count;
  }
  if (appended_count == 0) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "<none>"));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_format_expected_result_types(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_operand_t* operand, iree_string_view_t prefix,
    iree_string_view_t* out_detail) {
  *out_detail = iree_string_view_empty();
  // Measure this cold-path diagnostic before allocating its exact retained
  // storage. No temporary buffer or growth slack survives in the module.
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_null(), &builder);
  IREE_RETURN_IF_ERROR(
      loom_low_descriptor_text_asm_append_expected_result_types(
          descriptor_set, operand, prefix, &builder));
  const iree_host_size_t storage_size = iree_string_builder_size(&builder) + 1;
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&module->arena, storage_size, (void**)&storage));
  iree_string_builder_initialize_with_storage(storage, storage_size, &builder);
  iree_status_t status =
      loom_low_descriptor_text_asm_append_expected_result_types(
          descriptor_set, operand, prefix, &builder);
  if (iree_status_is_ok(status)) {
    *out_detail = iree_string_builder_view(&builder);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static uint16_t loom_low_descriptor_text_asm_find_single_register_alt(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_operand_t* operand) {
  uint16_t reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  for (uint16_t i = 0; i < operand->reg_class_alt_count; ++i) {
    const uint32_t alt_index = operand->reg_class_alt_start + i;
    const loom_low_reg_class_alt_t* alternative =
        &descriptor_set->reg_class_alts[alt_index];
    if (alternative->reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
        iree_all_bits_set(alternative->flags,
                          LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
      continue;
    }
    if (reg_class_id != LOOM_LOW_REG_CLASS_NONE) {
      return LOOM_LOW_REG_CLASS_NONE;
    }
    reg_class_id = alternative->reg_class_id;
  }
  return reg_class_id;
}

static const loom_low_asm_result_value_type_t*
loom_low_descriptor_text_asm_result_value_type(
    const loom_text_low_asm_packet_descriptor_t* packet,
    uint16_t result_index) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(packet->descriptor_set);
  const loom_low_asm_form_t* asm_form =
      loom_low_descriptor_text_asm_form(packet->form);
  const loom_low_asm_layout_t* layout =
      &descriptor_set->asm_layouts[asm_form->layout_index];
  if (layout->result_value_type_start ==
      LOOM_LOW_ASM_RESULT_VALUE_TYPE_START_NONE) {
    return NULL;
  }
  const uint32_t value_type_index =
      layout->result_value_type_start + result_index;
  const loom_low_asm_result_value_type_t* value_type =
      &descriptor_set->asm_result_value_types[value_type_index];
  return value_type->kind == LOOM_LOW_ASM_RESULT_VALUE_TYPE_KIND_NONE
             ? NULL
             : value_type;
}

static loom_type_t loom_low_descriptor_text_asm_make_inferred_register_type(
    const loom_text_low_asm_packet_descriptor_t* packet, uint16_t result_index,
    const loom_low_operand_t* operand, uint16_t reg_class_id,
    loom_register_type_data_t* out_register_data) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(packet->descriptor_set);
  const loom_low_asm_result_value_type_t* value_type =
      loom_low_descriptor_text_asm_result_value_type(packet, result_index);
  if (value_type == NULL) {
    return loom_low_register_type(descriptor_set->stable_id, reg_class_id,
                                  operand->unit_count);
  }

  loom_type_t semantic_type = loom_type_none();
  switch (value_type->kind) {
    case LOOM_LOW_ASM_RESULT_VALUE_TYPE_KIND_SCALAR:
      semantic_type = loom_type_scalar(value_type->element_type);
      break;
    case LOOM_LOW_ASM_RESULT_VALUE_TYPE_KIND_VECTOR:
      semantic_type = loom_type_shaped_1d(
          LOOM_TYPE_VECTOR, value_type->element_type,
          loom_dim_pack_static(value_type->vector_lane_count), 0);
      break;
    default:
      IREE_ASSERT_UNREACHABLE("verified asm result value type kind");
      IREE_BUILTIN_UNREACHABLE();
  }
  *out_register_data = (loom_register_type_data_t){
      .carrier_payload0 = descriptor_set->stable_id,
      .carrier_payload1 = loom_low_register_type_pack_payload1(
          reg_class_id, operand->unit_count),
      .value_type = semantic_type,
  };
  return loom_type_register_payload_with_value_type(out_register_data);
}

static bool loom_low_descriptor_text_asm_result_accepts_type(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_operand_t* operand, loom_type_t type) {
  if (!loom_low_type_is_register(type)) {
    return false;
  }
  if (loom_low_register_type_unit_count(type) != operand->unit_count) {
    return false;
  }
  if (loom_low_register_type_descriptor_set_stable_id(type) !=
      descriptor_set->stable_id) {
    return false;
  }
  const uint16_t actual_class_id = loom_low_register_type_class_id(type);
  for (uint16_t i = 0; i < operand->reg_class_alt_count; ++i) {
    const uint32_t alt_index = operand->reg_class_alt_start + i;
    const loom_low_reg_class_alt_t* alternative =
        &descriptor_set->reg_class_alts[alt_index];
    if (alternative->reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
        iree_all_bits_set(alternative->flags,
                          LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
      continue;
    }
    if (actual_class_id == alternative->reg_class_id) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_low_descriptor_text_asm_infer_result_type(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_packet_descriptor_t* packet,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    uint16_t result_index, loom_module_t* module, loom_type_t* out_type,
    iree_string_view_t* out_diagnostic_detail) {
  (void)state;
  *out_type = loom_type_none();
  *out_diagnostic_detail = iree_string_view_empty();

  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(packet->descriptor_set);
  uint16_t descriptor_operand_index = 0;
  const loom_low_operand_t* operand =
      loom_low_descriptor_text_asm_result_operand(packet, result_index,
                                                  &descriptor_operand_index);

  bool has_tied_operand = false;
  uint16_t tied_operand_index = 0;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_find_tied_packet_operand(
      packet, descriptor_operand_index, &has_tied_operand,
      &tied_operand_index));
  if (has_tied_operand) {
    if (tied_operand_index >= operand_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low asm tied operand index is out of range");
    }
    loom_type_t tied_type =
        loom_module_value_type(module, operands[tied_operand_index]);
    if (loom_low_descriptor_text_asm_result_accepts_type(descriptor_set,
                                                         operand, tied_type)) {
      *out_type = tied_type;
      return iree_ok_status();
    }
    return loom_low_descriptor_text_asm_format_expected_result_types(
        module, descriptor_set, operand,
        IREE_SV("tied operand type must be one of: "), out_diagnostic_detail);
  }

  const uint16_t reg_class_id =
      loom_low_descriptor_text_asm_find_single_register_alt(descriptor_set,
                                                            operand);
  if (reg_class_id == LOOM_LOW_REG_CLASS_NONE) {
    return loom_low_descriptor_text_asm_format_expected_result_types(
        module, descriptor_set, operand,
        IREE_SV("result type annotation is required for one of: "),
        out_diagnostic_detail);
  }

  loom_register_type_data_t register_data = {0};
  loom_type_t inferred_type =
      loom_low_descriptor_text_asm_make_inferred_register_type(
          packet, result_index, operand, reg_class_id, &register_data);
  return loom_module_intern_type(module, inferred_type, out_type);
}

static iree_status_t loom_low_descriptor_text_asm_validate_result_type(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_packet_descriptor_t* packet,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    uint16_t result_index, loom_module_t* module, loom_type_t type,
    iree_string_view_t* out_diagnostic_detail) {
  (void)state;
  *out_diagnostic_detail = iree_string_view_empty();
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(packet->descriptor_set);
  uint16_t descriptor_operand_index = 0;
  const loom_low_operand_t* operand =
      loom_low_descriptor_text_asm_result_operand(packet, result_index,
                                                  &descriptor_operand_index);

  if (!loom_low_descriptor_text_asm_result_accepts_type(descriptor_set, operand,
                                                        type)) {
    return loom_low_descriptor_text_asm_format_expected_result_types(
        module, descriptor_set, operand,
        IREE_SV("result type annotation must be one of: "),
        out_diagnostic_detail);
  }

  bool has_tied_operand = false;
  uint16_t tied_operand_index = 0;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_find_tied_packet_operand(
      packet, descriptor_operand_index, &has_tied_operand,
      &tied_operand_index));
  if (!has_tied_operand) {
    return iree_ok_status();
  }
  if (tied_operand_index >= operand_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low asm tied operand index is out of range");
  }
  loom_type_t tied_type =
      loom_module_value_type(module, operands[tied_operand_index]);
  if (!loom_type_equal(type, tied_type)) {
    *out_diagnostic_detail =
        IREE_SV("result type annotation must match tied operand type");
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t
loom_low_descriptor_text_asm_result_type_annotation_required(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_packet_descriptor_t* packet,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    uint16_t result_index, const loom_module_t* module, loom_type_t type,
    bool* out_required, iree_string_view_t* out_diagnostic_detail) {
  (void)state;
  *out_required = true;
  *out_diagnostic_detail = iree_string_view_empty();
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(packet->descriptor_set);
  uint16_t descriptor_operand_index = 0;
  const loom_low_operand_t* operand =
      loom_low_descriptor_text_asm_result_operand(packet, result_index,
                                                  &descriptor_operand_index);

  if (!loom_low_descriptor_text_asm_result_accepts_type(descriptor_set, operand,
                                                        type)) {
    *out_diagnostic_detail =
        IREE_SV("result type annotation is not accepted by low descriptor");
    return iree_ok_status();
  }

  bool has_tied_operand = false;
  uint16_t tied_operand_index = 0;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_find_tied_packet_operand(
      packet, descriptor_operand_index, &has_tied_operand,
      &tied_operand_index));
  if (has_tied_operand) {
    if (tied_operand_index >= operand_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low asm tied operand index is out of range");
    }
    loom_type_t tied_type =
        loom_module_value_type(module, operands[tied_operand_index]);
    if (!loom_type_equal(type, tied_type)) {
      *out_diagnostic_detail =
          IREE_SV("result type annotation must match tied operand type");
      return iree_ok_status();
    }
    *out_required = false;
    return iree_ok_status();
  }

  const uint16_t reg_class_id =
      loom_low_descriptor_text_asm_find_single_register_alt(descriptor_set,
                                                            operand);
  if (reg_class_id == LOOM_LOW_REG_CLASS_NONE) {
    return iree_ok_status();
  }
  loom_register_type_data_t register_data = {0};
  loom_type_t inferred_type =
      loom_low_descriptor_text_asm_make_inferred_register_type(
          packet, result_index, operand, reg_class_id, &register_data);
  *out_required = !loom_type_equal(type, inferred_type);
  return iree_ok_status();
}

static void loom_low_descriptor_text_asm_immediate_descriptor(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_packet_descriptor_t* packet,
    uint16_t immediate_index,
    loom_text_low_asm_immediate_descriptor_t* out_immediate) {
  (void)state;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(packet->descriptor_set);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_text_asm_descriptor(packet->descriptor);
  const loom_low_asm_form_t* asm_form =
      loom_low_descriptor_text_asm_form(packet->form);
  const loom_low_asm_layout_t* layout =
      &descriptor_set->asm_layouts[asm_form->layout_index];
  const loom_low_asm_immediate_t* asm_immediate =
      &descriptor_set
           ->asm_immediates[layout->immediate_start + immediate_index];
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[descriptor->immediate_start +
                                  asm_immediate->immediate_index];
  out_immediate->field_name = loom_low_descriptor_set_string(
      descriptor_set, immediate->field_name_string_ref);
  out_immediate->has_default_value =
      iree_any_bit_set(immediate->flags, LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE);
  out_immediate->default_value = immediate->default_value;
  out_immediate->enum_domain = immediate->kind == LOOM_LOW_IMMEDIATE_KIND_ENUM
                                   ? immediate->enum_domain_id
                                   : UINT16_MAX;
  out_immediate->spelling =
      asm_immediate->name_string_ref != LOOM_STRING_REF_NONE
          ? loom_low_descriptor_set_string(descriptor_set,
                                           asm_immediate->name_string_ref)
          : out_immediate->field_name;
}

static iree_host_size_t loom_low_descriptor_text_asm_tied_result_count(
    const loom_text_low_asm_packet_descriptor_t* packet) {
  iree_host_size_t tied_result_count = 0;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(packet->descriptor_set);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_text_asm_descriptor(packet->descriptor);
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const uint32_t constraint_index = descriptor->constraint_start + i;
    if (descriptor_set->constraints[constraint_index].kind ==
        LOOM_LOW_CONSTRAINT_KIND_TIED) {
      ++tied_result_count;
    }
  }
  return tied_result_count;
}

static iree_status_t loom_low_descriptor_text_asm_tied_result_at(
    const loom_text_low_asm_packet_descriptor_t* packet,
    iree_host_size_t tied_result_ordinal, loom_tied_result_t* out_tied_result) {
  *out_tied_result = (loom_tied_result_t){0};
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(packet->descriptor_set);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_text_asm_descriptor(packet->descriptor);

  iree_host_size_t current_ordinal = 0;
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const uint32_t constraint_index = descriptor->constraint_start + i;
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[constraint_index];
    if (constraint->kind != LOOM_LOW_CONSTRAINT_KIND_TIED) {
      continue;
    }
    if (current_ordinal++ != tied_result_ordinal) {
      continue;
    }

    uint16_t result_index = 0;
    if (!loom_low_descriptor_text_asm_find_packet_result_index(
            packet, constraint->lhs_operand_index, &result_index)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low asm tied result references a result outside the asm form");
    }

    uint16_t operand_index = 0;
    if (!loom_low_descriptor_text_asm_find_packet_operand_index(
            packet, constraint->rhs_operand_index, &operand_index)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low asm tied result references an operand outside the asm form");
    }

    *out_tied_result = (loom_tied_result_t){
        .result_index = result_index,
        .operand_index = operand_index,
        .has_type_change = false,
    };
    return iree_ok_status();
  }

  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_low_descriptor_text_asm_build_tied_results(
    loom_builder_t* builder,
    const loom_text_low_asm_packet_descriptor_t* packet,
    const loom_tied_result_t** out_tied_results,
    iree_host_size_t* out_tied_result_count) {
  *out_tied_results = NULL;
  *out_tied_result_count = 0;
  const iree_host_size_t tied_result_count =
      loom_low_descriptor_text_asm_tied_result_count(packet);
  if (tied_result_count == 0) {
    return iree_ok_status();
  }

  loom_tied_result_t* tied_results = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&builder->module->arena, tied_result_count,
                                sizeof(*tied_results), (void**)&tied_results));

  for (iree_host_size_t i = 0; i < tied_result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_tied_result_at(
        packet, i, &tied_results[i]));
  }
  *out_tied_results = tied_results;
  *out_tied_result_count = tied_result_count;
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_build_packet(
    const loom_text_low_asm_environment_state_t* state, loom_builder_t* builder,
    const loom_text_low_asm_packet_descriptor_t* packet,
    loom_text_low_asm_packet_build_flags_t build_flags, uint8_t instance_flags,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attributes, const loom_type_t* result_types,
    iree_host_size_t result_count, const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count, loom_location_id_t location,
    loom_op_t** out_op) {
  (void)state;
  const bool operand_count_matches =
      packet->has_variadic_operands
          ? operand_count >= packet->minimum_operand_count
          : operand_count == packet->minimum_operand_count;
  if (!operand_count_matches || result_count != packet->result_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm packet build shape does not match the "
                            "packet descriptor");
  }
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(packet->descriptor_set);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_text_asm_descriptor(packet->descriptor);
  if (packet->operation_kind == LOOM_OP_LOW_CONST) {
    if (result_count != 1) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low asm const packet must have one result");
    }
    return loom_low_build_resolved_descriptor_const(
        builder, descriptor_set, descriptor, attributes, result_types[0],
        location, out_op);
  }
  if (iree_any_bit_set(build_flags,
                       LOOM_TEXT_LOW_ASM_PACKET_BUILD_FLAG_INFER_TIES)) {
    IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_build_tied_results(
        builder, packet, &tied_results, &tied_result_count));
  }
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, instance_flags, operands,
      operand_count, attributes, result_types, result_count, tied_results,
      tied_result_count, location, out_op);
}

static void loom_low_descriptor_text_asm_operand_segment_descriptor(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_packet_descriptor_t* packet, uint16_t segment_index,
    loom_text_low_asm_operand_segment_descriptor_t* out_segment) {
  (void)state;
  *out_segment = (loom_text_low_asm_operand_segment_descriptor_t){0};
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(packet->descriptor_set);
  const loom_low_asm_form_t* asm_form =
      loom_low_descriptor_text_asm_form(packet->form);
  const loom_low_asm_layout_t* layout =
      &descriptor_set->asm_layouts[asm_form->layout_index];
  const uint32_t row_index = layout->operand_segment_start + segment_index;
  const loom_low_asm_operand_segment_t* segment =
      &descriptor_set->asm_operand_segments[row_index];
  switch (segment->delimiter) {
    case LOOM_LOW_ASM_OPERAND_SEGMENT_DELIMITER_ANGLE:
      out_segment->delimiter =
          LOOM_TEXT_LOW_ASM_OPERAND_SEGMENT_DELIMITER_ANGLE;
      break;
    case LOOM_LOW_ASM_OPERAND_SEGMENT_DELIMITER_SQUARE:
      out_segment->delimiter =
          LOOM_TEXT_LOW_ASM_OPERAND_SEGMENT_DELIMITER_SQUARE;
      break;
    case LOOM_LOW_ASM_OPERAND_SEGMENT_DELIMITER_PAREN:
      out_segment->delimiter =
          LOOM_TEXT_LOW_ASM_OPERAND_SEGMENT_DELIMITER_PAREN;
      break;
    default:
      IREE_ASSERT_UNREACHABLE("generated low asm operand segment delimiter");
      IREE_BUILTIN_UNREACHABLE();
  }
  out_segment->is_variadic = iree_any_bit_set(
      segment->flags, LOOM_LOW_ASM_OPERAND_SEGMENT_FLAG_VARIADIC);
  out_segment->fixed_operand_count =
      segment->operand_count - (out_segment->is_variadic ? 1 : 0);
}

static iree_status_t loom_low_descriptor_text_asm_attr_slice(
    loom_attribute_t attr, loom_named_attr_slice_t* out_attrs) {
  *out_attrs = loom_make_named_attr_slice(NULL, 0);
  if (loom_attr_is_absent(attr)) {
    return iree_ok_status();
  }
  if (attr.kind != LOOM_ATTR_DICT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm packet immediate dictionary is invalid");
  }
  if (attr.count > 0 && attr.dict_entries == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low asm packet immediate dictionary has NULL entries");
  }
  *out_attrs = loom_attr_as_dict(attr);
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_attr_name(
    const loom_module_t* module, const loom_named_attr_t* attr,
    iree_string_view_t* out_name) {
  *out_name = iree_string_view_empty();
  if (attr->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm packet immediate name is invalid");
  }
  *out_name = loom_string_table_get(&module->strings, attr->name_id);
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_find_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t field_name, bool* out_found) {
  *out_found = false;
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    iree_string_view_t attr_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_attr_name(
        module, &attrs.entries[i], &attr_name));
    if (iree_string_view_equal(attr_name, field_name)) {
      *out_found = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_validate_immediates(
    const loom_module_t* module,
    const loom_text_low_asm_packet_descriptor_t* packet,
    loom_named_attr_slice_t attrs) {
  if (attrs.count > packet->immediate_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low asm packet '%.*s' expects at most %u immediate attributes but op "
        "has "
        "%" PRIhsz,
        (int)packet->descriptor_key.size, packet->descriptor_key.data,
        packet->immediate_count, attrs.count);
  }

  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    iree_string_view_t attr_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_attr_name(
        module, &attrs.entries[i], &attr_name));
    bool expected = false;
    for (uint16_t j = 0; j < packet->immediate_count; ++j) {
      loom_text_low_asm_immediate_descriptor_t immediate = {0};
      loom_low_descriptor_text_asm_immediate_descriptor(NULL, packet, j,
                                                        &immediate);
      if (iree_string_view_equal(attr_name, immediate.field_name)) {
        expected = true;
        break;
      }
    }
    if (!expected) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low asm packet '%.*s' has unexpected "
                              "immediate attribute '%.*s'",
                              (int)packet->descriptor_key.size,
                              packet->descriptor_key.data, (int)attr_name.size,
                              attr_name.data);
    }
  }

  for (uint16_t i = 0; i < packet->immediate_count; ++i) {
    loom_text_low_asm_immediate_descriptor_t immediate = {0};
    loom_low_descriptor_text_asm_immediate_descriptor(NULL, packet, i,
                                                      &immediate);
    bool found = false;
    IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_find_attr(
        module, attrs, immediate.field_name, &found));
    if (!found && !immediate.has_default_value) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low asm packet '%.*s' is missing immediate "
          "attribute '%.*s'",
          (int)packet->descriptor_key.size, packet->descriptor_key.data,
          (int)immediate.field_name.size, immediate.field_name.data);
    }
  }
  return iree_ok_status();
}

static bool loom_low_descriptor_text_asm_tied_result_equal(
    loom_tied_result_t lhs, loom_tied_result_t rhs) {
  return lhs.result_index == rhs.result_index &&
         lhs.operand_index == rhs.operand_index &&
         lhs.has_type_change == rhs.has_type_change;
}

static iree_status_t
loom_low_descriptor_text_asm_tied_results_require_annotation(
    const loom_text_low_asm_packet_descriptor_t* packet, const loom_op_t* op,
    bool* out_required) {
  *out_required = false;
  const iree_host_size_t expected_count =
      loom_low_descriptor_text_asm_tied_result_count(packet);
  if (op->tied_result_count != expected_count) {
    *out_required = true;
    return iree_ok_status();
  }

  const loom_tied_result_t* actual_ties = loom_op_tied_results(op);
  for (iree_host_size_t i = 0; i < expected_count; ++i) {
    loom_tied_result_t expected_tie = {0};
    IREE_RETURN_IF_ERROR(
        loom_low_descriptor_text_asm_tied_result_at(packet, i, &expected_tie));
    bool found = false;
    for (uint16_t j = 0; j < op->tied_result_count; ++j) {
      if (loom_low_descriptor_text_asm_tied_result_equal(expected_tie,
                                                         actual_ties[j])) {
        found = true;
        break;
      }
    }
    if (!found) {
      *out_required = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_describe_packet(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_module_t* module, const loom_op_t* op, bool is_const,
    loom_text_low_asm_statement_t* out_statement) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (op->attribute_count != vtable->attribute_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm packet has an invalid attribute count");
  }
  const loom_attribute_t descriptor_attr =
      is_const ? loom_low_const_descriptor_attr(op)
               : loom_low_op_descriptor_attr(op);
  if (descriptor_attr.kind != LOOM_ATTR_SCOPED_ENUM) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low asm packet descriptor must be a representation-scoped enum");
  }
  const uint32_t descriptor_ordinal = loom_attr_as_scoped_enum(descriptor_attr);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ordinal);
  if (descriptor == NULL) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low asm packet descriptor is outside the active representation "
        "contract");
  }

  out_statement->kind = LOOM_TEXT_LOW_ASM_STATEMENT_UNAVAILABLE;
  out_statement->packet.descriptor_key = loom_low_descriptor_set_string(
      descriptor_set, descriptor->key_string_ref);
  const loom_low_descriptor_view_t* descriptor_view =
      loom_low_descriptor_set_descriptor_view_at(descriptor_set,
                                                 descriptor_ordinal);
  if (descriptor_view->canonical_asm_form_ordinal ==
      LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
    return iree_ok_status();
  }
  loom_text_low_asm_packet_descriptor_t packet = {0};
  loom_low_descriptor_text_asm_make_packet(
      descriptor_set,
      &descriptor_set->asm_forms[descriptor_view->canonical_asm_form_ordinal],
      descriptor, &packet);
  if (is_const != (packet.operation_kind == LOOM_OP_LOW_CONST)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low asm packet '%.*s' has the wrong canonical operation form",
        (int)packet.descriptor_key.size, packet.descriptor_key.data);
  }

  loom_named_attr_slice_t attrs = loom_make_named_attr_slice(NULL, 0);
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_attr_slice(
      is_const ? loom_low_const_attrs_attr(op) : loom_low_op_attrs_attr(op),
      &attrs));
  IREE_RETURN_IF_ERROR(
      loom_low_descriptor_text_asm_validate_immediates(module, &packet, attrs));

  const loom_value_id_t* results = loom_op_const_results(op);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  bool requires_result_tie_annotation = false;
  if (is_const) {
    if (op->result_count != 1 || op->operand_count != 0 ||
        op->tied_result_count != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low const packet '%.*s' has invalid shape",
                              (int)packet.descriptor_key.size,
                              packet.descriptor_key.data);
    }
  } else {
    const bool operand_count_matches =
        packet.has_variadic_operands
            ? op->operand_count >= packet.minimum_operand_count
            : op->operand_count == packet.minimum_operand_count;
    if (op->result_count != packet.result_count || !operand_count_matches) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        loom_low_descriptor_text_asm_tied_results_require_annotation(
            &packet, op, &requires_result_tie_annotation));
  }

  *out_statement = (loom_text_low_asm_statement_t){
      .kind = LOOM_TEXT_LOW_ASM_STATEMENT_PACKET,
      .op = op,
      .packet = packet,
      .results = results,
      .result_count = op->result_count,
      .requires_result_tie_annotation = requires_result_tie_annotation,
      .operands = operands,
      .operand_count = op->operand_count,
      .attributes = attrs,
      .has_immediate_attribute_field = true,
      .immediate_attribute_field_index = packet.immediate_attribute_field_index,
      .location = op->location,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_describe_operation(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_descriptor_set_t* descriptor_set_handle,
    const loom_module_t* module, const loom_op_t* op,
    loom_text_low_asm_statement_t* out_statement) {
  (void)state;
  *out_statement = (loom_text_low_asm_statement_t){0};
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(descriptor_set_handle);
  if (loom_low_const_isa(op)) {
    return loom_low_descriptor_text_asm_describe_packet(
        descriptor_set, module, op, /*is_const=*/true, out_statement);
  }
  if (loom_low_op_isa(op)) {
    return loom_low_descriptor_text_asm_describe_packet(
        descriptor_set, module, op, /*is_const=*/false, out_statement);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_resolve_register_type(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_descriptor_set_t* descriptor_set_handle,
    iree_string_view_t register_class_name, uint32_t unit_count,
    loom_type_t* out_type, bool* out_found) {
  (void)state;
  *out_type = loom_type_none();
  *out_found = false;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(descriptor_set_handle);
  uint16_t register_class_id = LOOM_LOW_REG_CLASS_NONE;
  if (loom_low_descriptor_set_lookup_register_class(
          descriptor_set, register_class_name, &register_class_id, NULL)) {
    *out_type = loom_low_register_type(descriptor_set->stable_id,
                                       register_class_id, unit_count);
    *out_found = true;
  }
  return iree_ok_status();
}

static iree_status_t
loom_low_descriptor_text_asm_lookup_register_descriptor_set_from_registry(
    const loom_low_descriptor_registry_t* registry, loom_type_t type,
    const loom_text_low_asm_descriptor_set_t** out_descriptor_set) {
  *out_descriptor_set = NULL;
  if (!loom_low_type_is_register(type)) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_registry_lookup_by_id(
          registry, loom_low_register_type_descriptor_set_stable_id(type));
  if (descriptor_set == NULL) {
    return iree_ok_status();
  }
  *out_descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set_handle(descriptor_set);
  return iree_ok_status();
}

static iree_status_t
loom_low_descriptor_text_asm_lookup_register_descriptor_set_default(
    const loom_text_low_asm_environment_state_t* state, loom_type_t type,
    const loom_text_low_asm_descriptor_set_t** out_descriptor_set) {
  return loom_low_descriptor_text_asm_lookup_register_descriptor_set_from_registry(
      loom_low_descriptor_text_asm_state_registry(state), type,
      out_descriptor_set);
}

static iree_status_t
loom_low_descriptor_text_asm_lookup_register_descriptor_set_with_diagnostics(
    const loom_text_low_asm_environment_state_t* state, loom_type_t type,
    const loom_text_low_asm_descriptor_set_t** out_descriptor_set) {
  const loom_low_descriptor_text_asm_environment_storage_t* storage =
      loom_low_descriptor_text_asm_state_storage(state);
  return loom_low_descriptor_text_asm_lookup_register_descriptor_set_from_registry(
      storage->descriptor_registry, type, out_descriptor_set);
}

static iree_status_t loom_low_descriptor_text_asm_describe_register_type(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_descriptor_set_t* descriptor_set_handle,
    loom_type_t type, iree_string_view_t* out_register_class_name,
    uint32_t* out_unit_count, bool* out_found) {
  (void)state;
  *out_register_class_name = iree_string_view_empty();
  *out_unit_count = 0;
  *out_found = false;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(descriptor_set_handle);
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_descriptor_set_stable_id(type) !=
          descriptor_set->stable_id) {
    return iree_ok_status();
  }
  const uint16_t register_class_id = loom_low_register_type_class_id(type);
  if (register_class_id >= descriptor_set->reg_class_count) {
    return iree_ok_status();
  }
  const loom_low_reg_class_t* register_class =
      &descriptor_set->reg_classes[register_class_id];
  *out_register_class_name = loom_low_descriptor_set_string(
      descriptor_set, register_class->name_string_ref);
  *out_unit_count = loom_low_register_type_unit_count(type);
  *out_found = true;
  return iree_ok_status();
}

static const loom_text_low_asm_vtable_t kLowDescriptorTextAsmVtable = {
    .lookup_packet = loom_low_descriptor_text_asm_lookup_packet,
    .infer_result_type = loom_low_descriptor_text_asm_infer_result_type,
    .validate_result_type = loom_low_descriptor_text_asm_validate_result_type,
    .result_type_annotation_required =
        loom_low_descriptor_text_asm_result_type_annotation_required,
    .immediate_descriptor = loom_low_descriptor_text_asm_immediate_descriptor,
    .operand_segment_descriptor =
        loom_low_descriptor_text_asm_operand_segment_descriptor,
    .build_packet = loom_low_descriptor_text_asm_build_packet,
    .describe_operation = loom_low_descriptor_text_asm_describe_operation,
    .resolve_register_type = loom_low_descriptor_text_asm_resolve_register_type,
    .lookup_register_descriptor_set =
        loom_low_descriptor_text_asm_lookup_register_descriptor_set_default,
    .describe_register_type =
        loom_low_descriptor_text_asm_describe_register_type,
};

static const loom_text_low_asm_vtable_t kLowDescriptorTextAsmDiagnosticVtable = {
    .lookup_packet = loom_low_descriptor_text_asm_lookup_packet,
    .diagnose_unknown_packet =
        loom_low_descriptor_text_asm_diagnose_unknown_packet,
    .infer_result_type = loom_low_descriptor_text_asm_infer_result_type,
    .validate_result_type = loom_low_descriptor_text_asm_validate_result_type,
    .result_type_annotation_required =
        loom_low_descriptor_text_asm_result_type_annotation_required,
    .immediate_descriptor = loom_low_descriptor_text_asm_immediate_descriptor,
    .operand_segment_descriptor =
        loom_low_descriptor_text_asm_operand_segment_descriptor,
    .build_packet = loom_low_descriptor_text_asm_build_packet,
    .describe_operation = loom_low_descriptor_text_asm_describe_operation,
    .resolve_register_type = loom_low_descriptor_text_asm_resolve_register_type,
    .lookup_register_descriptor_set =
        loom_low_descriptor_text_asm_lookup_register_descriptor_set_with_diagnostics,
    .describe_register_type =
        loom_low_descriptor_text_asm_describe_register_type,
};

void loom_low_descriptor_text_asm_environment_initialize(
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_text_low_asm_environment_t* out_environment) {
  *out_environment = (loom_text_low_asm_environment_t){
      .operation_formats = &loom_low_assembly_formats,
      .vtable = &kLowDescriptorTextAsmVtable,
      .state =
          (const loom_text_low_asm_environment_state_t*)descriptor_registry,
  };
  loom_low_repr_environment_initialize(descriptor_registry,
                                       &out_environment->low_repr);
}

void loom_low_descriptor_text_asm_environment_initialize_with_diagnostics(
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_target_low_asm_diagnostic_provider_list_t diagnostic_provider_list,
    loom_low_descriptor_text_asm_environment_storage_t* out_storage,
    loom_text_low_asm_environment_t* out_environment) {
  *out_storage = (loom_low_descriptor_text_asm_environment_storage_t){
      .descriptor_registry = descriptor_registry,
      .diagnostic_provider_list = diagnostic_provider_list,
  };
  *out_environment = (loom_text_low_asm_environment_t){
      .operation_formats = &loom_low_assembly_formats,
      .vtable = &kLowDescriptorTextAsmDiagnosticVtable,
      .state = (const loom_text_low_asm_environment_state_t*)out_storage,
  };
  loom_low_repr_environment_initialize(descriptor_registry,
                                       &out_environment->low_repr);
}

void loom_low_descriptor_text_print_context_initialize(
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_low_descriptor_text_print_context_t* out_context) {
  *out_context = (loom_low_descriptor_text_print_context_t){0};
  out_context->options.flags = LOOM_TEXT_PRINT_DEFAULT;
  loom_low_descriptor_text_asm_environment_initialize(
      descriptor_registry, &out_context->options.low_asm_environment);
}

void loom_low_descriptor_text_print_context_initialize_for_set(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_low_descriptor_text_print_context_t* out_context) {
  *out_context = (loom_low_descriptor_text_print_context_t){0};
  if (descriptor_set != NULL) {
    out_context->descriptor_sets[0] = descriptor_set;
    out_context->descriptor_registry.descriptor_sets =
        out_context->descriptor_sets;
    out_context->descriptor_registry.descriptor_set_count = 1;
  }
  out_context->options.flags = LOOM_TEXT_PRINT_DEFAULT;
  loom_low_descriptor_text_asm_environment_initialize(
      &out_context->descriptor_registry,
      &out_context->options.low_asm_environment);
}
