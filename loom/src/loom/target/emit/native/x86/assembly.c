// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/x86/assembly.h"

#include <inttypes.h>

#include "loom/codegen/low/move_sequence.h"
#include "loom/codegen/low/packet.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/x86/register_classes.h"
#include "loom/target/emit/native/assembly.h"

static const char* const kX86Gpr64Names[] = {
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
    "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15",
};

static const char* const kX86Gpr32Names[] = {
    "eax", "ecx", "edx",  "ebx",  "esp",  "ebp",  "esi",  "edi",
    "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d",
};

static const char* const kX86Gpr8Names[] = {
    "al",  "cl",  "dl",   "bl",   "spl",  "bpl",  "sil",  "dil",
    "r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b", "r15b",
};

static iree_status_t loom_x86_descriptor_key(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t* out_key) {
  return loom_native_assembly_descriptor_string(
      context->schedule->target.descriptor_set,
      context->packet->descriptor->key_string_offset, out_key);
}

static iree_status_t loom_x86_descriptor_mnemonic(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t* out_mnemonic) {
  return loom_native_assembly_descriptor_string(
      context->schedule->target.descriptor_set,
      context->packet->descriptor->mnemonic_string_offset, out_mnemonic);
}

static iree_status_t loom_x86_append_mnemonic(
    const loom_native_assembly_packet_context_t* context) {
  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_mnemonic(context, &mnemonic));
  return iree_string_builder_append_string(context->builder, mnemonic);
}

static const loom_low_allocation_assignment_t* loom_x86_map_assignment(
    const loom_native_assembly_packet_context_t* context,
    loom_value_id_t value_id) {
  return loom_low_allocation_map_active_value_assignment(context->allocation,
                                                         value_id, NULL);
}

static bool loom_x86_assignments_match(
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  return lhs->location_kind == rhs->location_kind &&
         lhs->descriptor_reg_class_id == rhs->descriptor_reg_class_id &&
         lhs->location_base == rhs->location_base &&
         lhs->location_count == rhs->location_count;
}

static iree_status_t loom_x86_register_class_kind(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_allocation_assignment_t* assignment,
    loom_x86_register_class_t* out_kind) {
  return loom_x86_descriptor_set_logical_register_class(
      context->schedule->target.descriptor_set,
      assignment->descriptor_reg_class_id, out_kind);
}

static iree_status_t loom_x86_append_assignment(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_allocation_assignment_t* assignment) {
  if (assignment->location_count != 1) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 assembly multi-register value %" PRIu32
                            " is not supported",
                            assignment->value_id);
  }
  loom_x86_register_class_t register_class_kind = 0;
  IREE_RETURN_IF_ERROR(
      loom_x86_register_class_kind(context, assignment, &register_class_kind));
  if (register_class_kind == LOOM_X86_REGISTER_CLASS_GPR32) {
    if (assignment->location_base >= IREE_ARRAYSIZE(kX86Gpr32Names)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "x86 GPR index %" PRIu32 " is out of range",
                              assignment->location_base);
    }
    return iree_string_builder_append_cstring(
        context->builder, kX86Gpr32Names[assignment->location_base]);
  }
  if (register_class_kind == LOOM_X86_REGISTER_CLASS_GPR64) {
    if (assignment->location_base >= IREE_ARRAYSIZE(kX86Gpr64Names)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "x86 GPR index %" PRIu32 " is out of range",
                              assignment->location_base);
    }
    return iree_string_builder_append_cstring(
        context->builder, kX86Gpr64Names[assignment->location_base]);
  }
  if (register_class_kind == LOOM_X86_REGISTER_CLASS_XMM) {
    return iree_string_builder_append_format(context->builder, "xmm%" PRIu32,
                                             assignment->location_base);
  }
  if (register_class_kind == LOOM_X86_REGISTER_CLASS_YMM) {
    return iree_string_builder_append_format(context->builder, "ymm%" PRIu32,
                                             assignment->location_base);
  }
  if (register_class_kind == LOOM_X86_REGISTER_CLASS_ZMM) {
    return iree_string_builder_append_format(context->builder, "zmm%" PRIu32,
                                             assignment->location_base);
  }
  if (register_class_kind == LOOM_X86_REGISTER_CLASS_K) {
    return iree_string_builder_append_format(context->builder, "k%" PRIu32,
                                             assignment->location_base);
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "x86 assembly register class kind %u is unsupported",
                          (unsigned)register_class_kind);
}

static iree_status_t loom_x86_append_copy_mnemonic(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_allocation_assignment_t* destination_assignment) {
  loom_x86_register_class_t register_class_kind = 0;
  IREE_RETURN_IF_ERROR(loom_x86_register_class_kind(
      context, destination_assignment, &register_class_kind));
  switch (register_class_kind) {
    case LOOM_X86_REGISTER_CLASS_XMM:
    case LOOM_X86_REGISTER_CLASS_YMM:
    case LOOM_X86_REGISTER_CLASS_ZMM:
      return iree_string_builder_append_cstring(context->builder, "vmovdqa32 ");
    case LOOM_X86_REGISTER_CLASS_GPR32:
    case LOOM_X86_REGISTER_CLASS_GPR64:
      return iree_string_builder_append_cstring(context->builder, "mov ");
    case LOOM_X86_REGISTER_CLASS_K:
      return iree_string_builder_append_cstring(context->builder, "kmovq ");
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "x86 assembly copy register class kind %u is "
                          "unsupported",
                          (unsigned)register_class_kind);
}

static iree_status_t loom_x86_append_move_location(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_move_location_t* location) {
  const loom_low_allocation_assignment_t assignment = {
      .location_kind = location->location_kind,
      .value_class = location->value_class,
      .descriptor_reg_class_id = location->descriptor_reg_class_id,
      .location_base = location->location,
      .location_count = 1,
  };
  return loom_x86_append_assignment(context, &assignment);
}

static iree_status_t loom_x86_append_value(
    const loom_native_assembly_packet_context_t* context,
    loom_value_id_t value_id) {
  const loom_low_allocation_assignment_t* assignment =
      loom_x86_map_assignment(context, value_id);
  return loom_x86_append_assignment(context, assignment);
}

static iree_status_t loom_x86_append_result(
    const loom_native_assembly_packet_context_t* context,
    iree_host_size_t result_index) {
  const loom_op_t* op = context->packet->node->op;
  if (result_index >= op->result_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 assembly result index is out of range");
  }
  return loom_x86_append_value(context,
                               loom_op_const_results(op)[result_index]);
}

static iree_status_t loom_x86_append_operand(
    const loom_native_assembly_packet_context_t* context,
    iree_host_size_t operand_index) {
  const loom_op_t* op = context->packet->node->op;
  if (operand_index >= op->operand_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 assembly operand index is out of range");
  }
  return loom_x86_append_value(context,
                               loom_op_const_operands(op)[operand_index]);
}

static iree_status_t loom_x86_append_gpr8_value(
    const loom_native_assembly_packet_context_t* context,
    loom_value_id_t value_id) {
  const loom_low_allocation_assignment_t* assignment =
      loom_x86_map_assignment(context, value_id);
  if (assignment->location_count != 1) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 assembly multi-register byte value %" PRIu32
                            " is not supported",
                            assignment->value_id);
  }
  loom_x86_register_class_t register_class_kind = 0;
  IREE_RETURN_IF_ERROR(
      loom_x86_register_class_kind(context, assignment, &register_class_kind));
  if (register_class_kind != LOOM_X86_REGISTER_CLASS_GPR32) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "x86 assembly byte result requires a GPR32 allocation");
  }
  if (assignment->location_base >= IREE_ARRAYSIZE(kX86Gpr8Names)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 GPR index %" PRIu32 " is out of range",
                            assignment->location_base);
  }
  return iree_string_builder_append_cstring(
      context->builder, kX86Gpr8Names[assignment->location_base]);
}

static iree_status_t loom_x86_append_gpr8_result(
    const loom_native_assembly_packet_context_t* context,
    iree_host_size_t result_index) {
  const loom_op_t* op = context->packet->node->op;
  if (result_index >= op->result_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 assembly result index is out of range");
  }
  return loom_x86_append_gpr8_value(context,
                                    loom_op_const_results(op)[result_index]);
}

static loom_named_attr_slice_t loom_x86_packet_attrs(
    const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  if (loom_low_op_isa(op)) {
    return loom_low_op_attrs(op);
  }
  if (loom_low_const_isa(op)) {
    return loom_low_const_attrs(op);
  }
  return loom_make_named_attr_slice(NULL, 0);
}

static iree_status_t loom_x86_read_packet_i64_attr(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t name, int64_t* out_value) {
  return loom_native_assembly_read_i64_attr(context->schedule->module,
                                            loom_x86_packet_attrs(context),
                                            name, out_value);
}

static iree_status_t loom_x86_read_packet_immediate(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_descriptor_t* descriptor, uint16_t immediate_index,
    int64_t* out_value) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  if (immediate_index >= descriptor->immediate_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "x86 assembly immediate field is outside the descriptor");
  }
  const uint32_t immediate_row = descriptor->immediate_start + immediate_index;
  if (immediate_row >= descriptor_set->immediate_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 assembly immediate row is outside the "
                            "descriptor set");
  }
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[immediate_row];
  iree_string_view_t field_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
      descriptor_set, immediate->field_name_string_offset, &field_name));
  const loom_named_attr_t* attr = loom_native_assembly_find_attr(
      context->schedule->module, loom_x86_packet_attrs(context), field_name);
  if (attr == NULL) {
    if (iree_any_bit_set(immediate->flags,
                         LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE)) {
      *out_value = immediate->default_value;
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "x86 assembly requires immediate '%.*s'",
                            (int)field_name.size, field_name.data);
  }
  if (attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "x86 assembly immediate '%.*s' must be i64",
                            (int)field_name.size, field_name.data);
  }
  *out_value = loom_attr_as_i64(attr->value);
  return iree_ok_status();
}

static iree_status_t loom_x86_descriptor_immediate_index_by_name(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_descriptor_t* descriptor, iree_string_view_t field_name,
    uint16_t* out_immediate_index, bool* out_found) {
  *out_immediate_index = UINT16_MAX;
  *out_found = false;
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  if (descriptor->immediate_start > descriptor_set->immediate_count ||
      descriptor->immediate_count >
          descriptor_set->immediate_count - descriptor->immediate_start) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 assembly immediate range is outside the "
                            "descriptor set");
  }
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[descriptor->immediate_start + i];
    iree_string_view_t candidate_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
        descriptor_set, immediate->field_name_string_offset, &candidate_name));
    if (!iree_string_view_equal(candidate_name, field_name)) {
      continue;
    }
    *out_immediate_index = i;
    *out_found = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_read_optional_packet_immediate(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_descriptor_t* descriptor, iree_string_view_t field_name,
    int64_t default_value, int64_t* out_value, bool* out_found) {
  uint16_t immediate_index = UINT16_MAX;
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_immediate_index_by_name(
      context, descriptor, field_name, &immediate_index, out_found));
  if (!*out_found) {
    *out_value = default_value;
    return iree_ok_status();
  }
  return loom_x86_read_packet_immediate(context, descriptor, immediate_index,
                                        out_value);
}

static iree_status_t loom_x86_validate_address_scale(int64_t scale) {
  switch (scale) {
    case 1:
    case 2:
    case 4:
    case 8:
      return iree_ok_status();
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "x86 indexed memory scale %" PRId64 " is unsupported", scale);
  }
}

static iree_status_t loom_x86_read_packet_address_scale_attr(
    const loom_native_assembly_packet_context_t* context, int64_t* out_scale) {
  IREE_RETURN_IF_ERROR(
      loom_x86_read_packet_i64_attr(context, IREE_SV("scale"), out_scale));
  return loom_x86_validate_address_scale(*out_scale);
}

static iree_status_t loom_x86_append_memory_operand(
    const loom_native_assembly_packet_context_t* context,
    loom_value_id_t base_value_id, loom_value_id_t index_value_id,
    int64_t scale, int64_t displacement) {
  if (base_value_id == LOOM_VALUE_ID_INVALID &&
      index_value_id == LOOM_VALUE_ID_INVALID && displacement == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "x86 memory operand requires a base register, "
                            "index register, or displacement");
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "["));
  bool has_component = false;
  if (base_value_id != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_x86_append_value(context, base_value_id));
    has_component = true;
  }
  if (index_value_id != LOOM_VALUE_ID_INVALID) {
    if (has_component) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(context->builder, " + "));
    }
    IREE_RETURN_IF_ERROR(loom_x86_append_value(context, index_value_id));
    if (scale != 1) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          context->builder, " * %" PRId64, scale));
    }
    has_component = true;
  }
  if (displacement > 0) {
    if (has_component) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          context->builder, " + %" PRId64, displacement));
    } else {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          context->builder, "%" PRId64, displacement));
    }
  } else if (displacement < 0) {
    const uint64_t magnitude = (uint64_t)(-(displacement + 1)) + 1u;
    if (has_component) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          context->builder, " - %" PRIu64, magnitude));
    } else {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          context->builder, "-%" PRIu64, magnitude));
    }
  }
  return iree_string_builder_append_cstring(context->builder, "]");
}

static iree_status_t loom_x86_append_asm_form_separator(
    const loom_native_assembly_packet_context_t* context, bool* in_list) {
  if (*in_list) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, ", "));
  } else {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, " "));
    *in_list = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_append_asm_form_value(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_descriptor_t* descriptor, uint16_t descriptor_operand_index,
    bool is_result) {
  const loom_op_t* op = context->packet->node->op;
  if (descriptor_operand_index >= descriptor->operand_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "x86 assembly asm-form operand index is outside the descriptor");
  }
  if (is_result) {
    if (descriptor_operand_index >= descriptor->result_count ||
        descriptor_operand_index >= op->result_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "x86 assembly asm-form result field does not "
                              "name an emitted result");
    }
    return loom_x86_append_result(context, descriptor_operand_index);
  }
  if (descriptor_operand_index < descriptor->result_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "x86 assembly asm-form operand field unexpectedly "
                            "names a descriptor result");
  }
  const uint16_t operand_index =
      descriptor_operand_index - descriptor->result_count;
  if (operand_index >= op->operand_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "x86 assembly asm-form operand field does not "
                            "name an emitted operand");
  }
  return loom_x86_append_operand(context, operand_index);
}

static iree_status_t loom_x86_append_asm_form_values(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_descriptor_t* descriptor, uint32_t start, uint16_t count,
    bool is_result, bool* in_list) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  for (uint16_t i = 0; i < count; ++i) {
    const uint32_t asm_operand_index = start + i;
    if (asm_operand_index >= descriptor_set->asm_operand_index_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "x86 assembly asm-form operand row is outside the descriptor set");
    }
    IREE_RETURN_IF_ERROR(loom_x86_append_asm_form_separator(context, in_list));
    IREE_RETURN_IF_ERROR(loom_x86_append_asm_form_value(
        context, descriptor,
        descriptor_set->asm_operand_indices[asm_operand_index], is_result));
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_append_asm_form_immediates(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_descriptor_t* descriptor, const loom_low_asm_form_t* form,
    bool* in_list) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  for (uint16_t i = 0; i < form->immediate_count; ++i) {
    const uint32_t asm_immediate_index = form->immediate_start + i;
    if (asm_immediate_index >= descriptor_set->asm_immediate_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "x86 assembly asm-form immediate row is outside the descriptor set");
    }
    const loom_low_asm_immediate_t* asm_immediate =
        &descriptor_set->asm_immediates[asm_immediate_index];
    int64_t value = 0;
    IREE_RETURN_IF_ERROR(loom_x86_read_packet_immediate(
        context, descriptor, asm_immediate->immediate_index, &value));
    IREE_RETURN_IF_ERROR(loom_x86_append_asm_form_separator(context, in_list));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_format(context->builder, "%" PRId64, value));
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_canonical_asm_form(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_asm_form_t** out_form) {
  *out_form = NULL;
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  if (descriptor->canonical_asm_form_ordinal ==
      LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_x86_descriptor_key(context, &key));
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "x86 assembly descriptor '%.*s' has no canonical asm form",
        (int)key.size, key.data);
  }
  const loom_low_asm_form_t* form = loom_low_descriptor_set_asm_form_at(
      descriptor_set, descriptor->canonical_asm_form_ordinal);
  if (form == NULL) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "x86 assembly canonical asm form is outside the descriptor set");
  }
  *out_form = form;
  return iree_ok_status();
}

static iree_status_t loom_x86_append_canonical_asm_form_packet(
    const loom_native_assembly_packet_context_t* context) {
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  const loom_low_asm_form_t* form = NULL;
  IREE_RETURN_IF_ERROR(loom_x86_canonical_asm_form(context, &form));
  IREE_RETURN_IF_ERROR(loom_x86_append_mnemonic(context));
  bool in_list = false;
  IREE_RETURN_IF_ERROR(loom_x86_append_asm_form_values(
      context, descriptor, form->result_operand_index_start,
      form->result_operand_index_count, /*is_result=*/true, &in_list));
  IREE_RETURN_IF_ERROR(loom_x86_append_asm_form_values(
      context, descriptor, form->operand_index_start, form->operand_index_count,
      /*is_result=*/false, &in_list));
  return loom_x86_append_asm_form_immediates(context, descriptor, form,
                                             &in_list);
}

static iree_status_t loom_x86_append_tied_unary_packet(
    const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  const loom_low_allocation_assignment_t* result_assignment =
      loom_x86_map_assignment(context, loom_op_const_results(op)[0]);
  const loom_low_allocation_assignment_t* source_assignment =
      loom_x86_map_assignment(context, loom_op_const_operands(op)[0]);
  if (!loom_x86_assignments_match(result_assignment, source_assignment)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "x86 tied unary result must share the source physical register");
  }
  const loom_low_asm_form_t* form = NULL;
  IREE_RETURN_IF_ERROR(loom_x86_canonical_asm_form(context, &form));
  if (form->result_operand_index_count != 1 || form->operand_index_count != 1) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_x86_descriptor_key(context, &key));
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 tied unary descriptor '%.*s' has an "
                            "unsupported asm form",
                            (int)key.size, key.data);
  }

  IREE_RETURN_IF_ERROR(loom_x86_append_mnemonic(context));
  bool in_list = false;
  IREE_RETURN_IF_ERROR(loom_x86_append_asm_form_values(
      context, descriptor, form->result_operand_index_start,
      form->result_operand_index_count, /*is_result=*/true, &in_list));
  return loom_x86_append_asm_form_immediates(context, descriptor, form,
                                             &in_list);
}

static iree_status_t loom_x86_append_tied_ternary_packet(
    const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  const loom_low_allocation_assignment_t* result_assignment =
      loom_x86_map_assignment(context, loom_op_const_results(op)[0]);
  const loom_low_allocation_assignment_t* accumulator_assignment =
      loom_x86_map_assignment(context, loom_op_const_operands(op)[0]);
  if (!loom_x86_assignments_match(result_assignment, accumulator_assignment)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "x86 tied ternary result must share the accumulator physical register");
  }
  IREE_RETURN_IF_ERROR(loom_x86_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_x86_append_result(context, 0));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  IREE_RETURN_IF_ERROR(loom_x86_append_operand(context, 1));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  return loom_x86_append_operand(context, 2);
}

static iree_status_t loom_x86_append_tied_binary_packet(
    const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  const loom_low_allocation_assignment_t* result_assignment =
      loom_x86_map_assignment(context, loom_op_const_results(op)[0]);
  const loom_low_allocation_assignment_t* lhs_assignment =
      loom_x86_map_assignment(context, loom_op_const_operands(op)[0]);
  if (!loom_x86_assignments_match(result_assignment, lhs_assignment)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "x86 tied binary result must share the left-hand physical register");
  }
  IREE_RETURN_IF_ERROR(loom_x86_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_x86_append_result(context, 0));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  return loom_x86_append_operand(context, 1);
}

static iree_status_t loom_x86_append_cmp_setcc_packet(
    const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  if (op->result_count != 1 || op->operand_count != 2) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_x86_descriptor_key(context, &key));
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 compare descriptor '%.*s' has an unsupported "
                            "operand shape",
                            (int)key.size, key.data);
  }
  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_mnemonic(context, &mnemonic));
  if (!iree_string_view_consume_prefix(&mnemonic, IREE_SV("cmp.")) ||
      !iree_string_view_starts_with(mnemonic, IREE_SV("set"))) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_x86_descriptor_key(context, &key));
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 compare descriptor '%.*s' has an unsupported "
                            "mnemonic",
                            (int)key.size, key.data);
  }

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "cmp "));
  IREE_RETURN_IF_ERROR(loom_x86_append_operand(context, 0));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  IREE_RETURN_IF_ERROR(loom_x86_append_operand(context, 1));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "\n  "));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(context->builder, mnemonic));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_x86_append_gpr8_result(context, 0));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "\n  movzx "));
  IREE_RETURN_IF_ERROR(loom_x86_append_result(context, 0));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  return loom_x86_append_gpr8_result(context, 0);
}

static iree_status_t loom_x86_append_lea_packet(
    const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  if (op->result_count != 1 || op->operand_count < 1 || op->operand_count > 2) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_x86_descriptor_key(context, &key));
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 assembly LEA descriptor '%.*s' has an "
                            "unsupported operand shape",
                            (int)key.size, key.data);
  }

  int64_t displacement = 0;
  bool has_displacement = false;
  IREE_RETURN_IF_ERROR(loom_x86_read_optional_packet_immediate(
      context, descriptor, IREE_SV("disp32"), 0, &displacement,
      &has_displacement));
  int64_t scale = 1;
  bool has_scale = false;
  IREE_RETURN_IF_ERROR(loom_x86_read_optional_packet_immediate(
      context, descriptor, IREE_SV("scale"), 1, &scale, &has_scale));
  IREE_RETURN_IF_ERROR(loom_x86_validate_address_scale(scale));

  loom_value_id_t base_value_id = LOOM_VALUE_ID_INVALID;
  loom_value_id_t index_value_id = LOOM_VALUE_ID_INVALID;
  if (op->operand_count == 2) {
    base_value_id = loom_op_const_operands(op)[0];
    index_value_id = loom_op_const_operands(op)[1];
  } else if (has_scale) {
    index_value_id = loom_op_const_operands(op)[0];
  } else {
    base_value_id = loom_op_const_operands(op)[0];
  }
  if (op->operand_count == 1 && !has_scale && !has_displacement) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_x86_descriptor_key(context, &key));
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 assembly LEA descriptor '%.*s' does not "
                            "define an address transform",
                            (int)key.size, key.data);
  }

  IREE_RETURN_IF_ERROR(loom_x86_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_x86_append_result(context, 0));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  return loom_x86_append_memory_operand(context, base_value_id, index_value_id,
                                        scale, displacement);
}

static iree_status_t loom_x86_descriptor_has_constraint(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_low_constraint_kind_t kind,
    uint16_t lhs_operand_index, uint16_t rhs_operand_index,
    bool* out_has_constraint) {
  *out_has_constraint = false;
  if (descriptor->constraint_count == 0) {
    return iree_ok_status();
  }
  if (descriptor->constraint_start > descriptor_set->constraint_count ||
      descriptor->constraint_count >
          descriptor_set->constraint_count - descriptor->constraint_start) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 descriptor constraint range is out of range");
  }
  if (descriptor_set->constraints == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "x86 descriptor constraints table is missing");
  }
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (constraint->kind == kind &&
        constraint->lhs_operand_index == lhs_operand_index &&
        constraint->rhs_operand_index == rhs_operand_index) {
      *out_has_constraint = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_descriptor_has_effect(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_low_effect_kind_t kind,
    bool* out_has_effect) {
  *out_has_effect = false;
  if (descriptor->effect_count == 0) {
    return iree_ok_status();
  }
  if (descriptor->effect_start > descriptor_set->effect_count ||
      descriptor->effect_count >
          descriptor_set->effect_count - descriptor->effect_start) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 descriptor effect range is out of range");
  }
  if (descriptor_set->effects == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "x86 descriptor effects table is missing");
  }
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (effect->kind == kind) {
      *out_has_effect = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_descriptor_uses_tied_binary_form(
    const loom_native_assembly_packet_context_t* context,
    bool* out_uses_tied_binary_form) {
  *out_uses_tied_binary_form = false;
  const loom_op_t* op = context->packet->node->op;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  if (descriptor->result_count != 1 || descriptor->operand_count != 3 ||
      op->result_count != 1 || op->operand_count != 2) {
    return iree_ok_status();
  }
  const uint16_t result_operand_index = 0;
  const uint16_t lhs_operand_index = descriptor->result_count;
  bool is_tied = false;
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_has_constraint(
      descriptor_set, descriptor, LOOM_LOW_CONSTRAINT_KIND_TIED,
      result_operand_index, lhs_operand_index, &is_tied));
  if (!is_tied) {
    return iree_ok_status();
  }
  bool is_destructive = false;
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_has_constraint(
      descriptor_set, descriptor, LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE,
      result_operand_index, lhs_operand_index, &is_destructive));
  *out_uses_tied_binary_form = is_destructive;
  return iree_ok_status();
}

static iree_status_t loom_x86_descriptor_uses_tied_unary_form(
    const loom_native_assembly_packet_context_t* context,
    bool* out_uses_tied_unary_form) {
  *out_uses_tied_unary_form = false;
  const loom_op_t* op = context->packet->node->op;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  if (descriptor->result_count != 1 || descriptor->operand_count != 2 ||
      op->result_count != 1 || op->operand_count != 1) {
    return iree_ok_status();
  }
  const uint16_t result_operand_index = 0;
  const uint16_t source_operand_index = descriptor->result_count;
  bool is_tied = false;
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_has_constraint(
      descriptor_set, descriptor, LOOM_LOW_CONSTRAINT_KIND_TIED,
      result_operand_index, source_operand_index, &is_tied));
  if (!is_tied) {
    return iree_ok_status();
  }
  bool is_destructive = false;
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_has_constraint(
      descriptor_set, descriptor, LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE,
      result_operand_index, source_operand_index, &is_destructive));
  *out_uses_tied_unary_form = is_destructive;
  return iree_ok_status();
}

static iree_status_t loom_x86_descriptor_uses_tied_ternary_form(
    const loom_native_assembly_packet_context_t* context,
    bool* out_uses_tied_ternary_form) {
  *out_uses_tied_ternary_form = false;
  const loom_op_t* op = context->packet->node->op;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  if (descriptor->result_count != 1 || descriptor->operand_count != 4 ||
      op->result_count != 1 || op->operand_count != 3) {
    return iree_ok_status();
  }
  const uint16_t result_operand_index = 0;
  const uint16_t accumulator_operand_index = descriptor->result_count;
  bool is_tied = false;
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_has_constraint(
      descriptor_set, descriptor, LOOM_LOW_CONSTRAINT_KIND_TIED,
      result_operand_index, accumulator_operand_index, &is_tied));
  if (!is_tied) {
    return iree_ok_status();
  }
  bool is_destructive = false;
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_has_constraint(
      descriptor_set, descriptor, LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE,
      result_operand_index, accumulator_operand_index, &is_destructive));
  *out_uses_tied_ternary_form = is_destructive;
  return iree_ok_status();
}

static iree_status_t loom_x86_append_load_packet(
    const loom_native_assembly_packet_context_t* context) {
  int64_t displacement = 0;
  IREE_RETURN_IF_ERROR(
      loom_x86_read_packet_i64_attr(context, IREE_SV("disp32"), &displacement));
  const loom_op_t* op = context->packet->node->op;
  loom_value_id_t index_value_id = LOOM_VALUE_ID_INVALID;
  int64_t scale = 1;
  if (op->operand_count == 2) {
    index_value_id = loom_op_const_operands(op)[1];
    IREE_RETURN_IF_ERROR(
        loom_x86_read_packet_address_scale_attr(context, &scale));
  } else if (op->operand_count != 1) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_x86_descriptor_key(context, &key));
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 assembly load descriptor '%.*s' has an "
                            "unsupported operand shape",
                            (int)key.size, key.data);
  }
  IREE_RETURN_IF_ERROR(loom_x86_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_x86_append_result(context, 0));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  return loom_x86_append_memory_operand(context, loom_op_const_operands(op)[0],
                                        index_value_id, scale, displacement);
}

static iree_status_t loom_x86_append_store_packet(
    const loom_native_assembly_packet_context_t* context) {
  int64_t displacement = 0;
  IREE_RETURN_IF_ERROR(
      loom_x86_read_packet_i64_attr(context, IREE_SV("disp32"), &displacement));
  const loom_op_t* op = context->packet->node->op;
  loom_value_id_t base_value_id = LOOM_VALUE_ID_INVALID;
  loom_value_id_t index_value_id = LOOM_VALUE_ID_INVALID;
  int64_t scale = 1;
  if (op->operand_count == 2) {
    base_value_id = loom_op_const_operands(op)[1];
  } else if (op->operand_count == 3) {
    base_value_id = loom_op_const_operands(op)[1];
    index_value_id = loom_op_const_operands(op)[2];
    IREE_RETURN_IF_ERROR(
        loom_x86_read_packet_address_scale_attr(context, &scale));
  } else {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_x86_descriptor_key(context, &key));
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 assembly store descriptor '%.*s' has an "
                            "unsupported operand shape",
                            (int)key.size, key.data);
  }
  IREE_RETURN_IF_ERROR(loom_x86_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_x86_append_memory_operand(
      context, base_value_id, index_value_id, scale, displacement));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  return loom_x86_append_operand(context, 0);
}

static iree_status_t loom_x86_append_const_packet(
    const loom_native_assembly_packet_context_t* context) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  if (descriptor->result_count != 1 || descriptor->operand_count != 1 ||
      descriptor->immediate_count != 1) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_x86_descriptor_key(context, &key));
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 assembly const descriptor '%.*s' is "
                            "unsupported",
                            (int)key.size, key.data);
  }
  if (descriptor->immediate_start >= descriptor_set->immediate_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 assembly const immediate is out of range");
  }
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[descriptor->immediate_start];
  if (immediate->kind != LOOM_LOW_IMMEDIATE_KIND_SIGNED) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_x86_descriptor_key(context, &key));
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 assembly const descriptor '%.*s' has an "
                            "unsupported immediate kind",
                            (int)key.size, key.data);
  }
  iree_string_view_t immediate_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
      descriptor_set, immediate->field_name_string_offset, &immediate_name));
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(
      loom_x86_read_packet_i64_attr(context, immediate_name, &value));

  IREE_RETURN_IF_ERROR(loom_x86_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_x86_append_result(context, 0));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  return iree_string_builder_append_format(context->builder, "%" PRId64, value);
}

static iree_status_t loom_x86_append_copy_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  const loom_low_allocation_assignment_t* source_assignment =
      loom_x86_map_assignment(context, loom_low_copy_source(op));
  const loom_low_allocation_assignment_t* result_assignment =
      loom_x86_map_assignment(context, loom_low_copy_result(op));
  if (loom_x86_assignments_match(source_assignment, result_assignment)) {
    return iree_ok_status();
  }

  if (source_assignment->descriptor_reg_class_id !=
      result_assignment->descriptor_reg_class_id) {
    iree_string_view_t source_register_class = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_register_class_name(
        context->allocation, source_assignment, &source_register_class));
    iree_string_view_t result_register_class = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_register_class_name(
        context->allocation, result_assignment, &result_register_class));
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "x86 assembly copy between register classes '%.*s' and '%.*s' is "
        "unsupported",
        (int)source_register_class.size, source_register_class.data,
        (int)result_register_class.size, result_register_class.data);
  }

  IREE_RETURN_IF_ERROR(
      loom_x86_append_copy_mnemonic(context, result_assignment));
  IREE_RETURN_IF_ERROR(loom_x86_append_assignment(context, result_assignment));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  return loom_x86_append_assignment(context, source_assignment);
}

typedef struct loom_x86_assembly_move_state_t {
  // Packet context receiving assembly text.
  const loom_native_assembly_packet_context_t* context;
  // Number of non-identity moves emitted so far.
  uint32_t emitted_count;
} loom_x86_assembly_move_state_t;

static iree_status_t loom_x86_append_move(
    void* user_data, const loom_low_move_location_t* destination,
    const loom_low_move_location_t* source) {
  loom_x86_assembly_move_state_t* state =
      (loom_x86_assembly_move_state_t*)user_data;
  const loom_native_assembly_packet_context_t* context = state->context;
  if (state->emitted_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "\n  "));
  }
  const loom_low_allocation_assignment_t destination_assignment = {
      .location_kind = destination->location_kind,
      .value_class = destination->value_class,
      .descriptor_reg_class_id = destination->descriptor_reg_class_id,
      .location_base = destination->location,
      .location_count = 1,
  };
  IREE_RETURN_IF_ERROR(
      loom_x86_append_copy_mnemonic(context, &destination_assignment));
  IREE_RETURN_IF_ERROR(loom_x86_append_move_location(context, destination));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  IREE_RETURN_IF_ERROR(loom_x86_append_move_location(context, source));
  ++state->emitted_count;
  return iree_ok_status();
}

static iree_status_t loom_x86_emit_edge_copy_group(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_allocation_edge_copy_group_t* group) {
  iree_host_size_t move_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_count_edge_copy_units(
      context->allocation, group, &move_count));
  if (move_count == 0) {
    return iree_ok_status();
  }
  loom_low_move_t* moves = NULL;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_reserve_moves(
      context->move_scratch, move_count, &moves));
  loom_low_move_location_t* temporaries = NULL;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_reserve_temporaries(
      context->move_scratch, group->temporary_count, &temporaries));
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_populate_edge_copy_units(
      context->allocation, group, moves, move_count));
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_populate_edge_copy_temporaries(
      context->allocation, group, temporaries, group->temporary_count));
  loom_x86_assembly_move_state_t move_state = {
      .context = context,
  };
  loom_low_move_sequence_options_t options = {
      .descriptor_set = context->allocation->target.descriptor_set,
      .temporary_locations = temporaries,
      .temporary_location_count = group->temporary_count,
      .emit_move =
          {
              .fn = loom_x86_append_move,
              .user_data = &move_state,
          },
  };
  IREE_RETURN_IF_ERROR(
      loom_low_move_sequence_emit(context->move_scratch, move_count, &options));
  if (move_state.emitted_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "\n  "));
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_append_descriptor_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  if (loom_low_const_isa(context->packet->node->op)) {
    return loom_x86_append_const_packet(context);
  }
  bool has_read_effect = false;
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_has_effect(
      descriptor_set, descriptor, LOOM_LOW_EFFECT_KIND_READ, &has_read_effect));
  bool has_write_effect = false;
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_has_effect(
      descriptor_set, descriptor, LOOM_LOW_EFFECT_KIND_WRITE,
      &has_write_effect));
  if (has_read_effect && has_write_effect) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_x86_descriptor_key(context, &key));
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 assembly descriptor '%.*s' has both read and "
                            "write effects",
                            (int)key.size, key.data);
  }
  if (has_read_effect) {
    return loom_x86_append_load_packet(context);
  }
  if (has_write_effect) {
    return loom_x86_append_store_packet(context);
  }
  bool uses_tied_unary_form = false;
  IREE_RETURN_IF_ERROR(
      loom_x86_descriptor_uses_tied_unary_form(context, &uses_tied_unary_form));
  if (uses_tied_unary_form) {
    return loom_x86_append_tied_unary_packet(context);
  }
  bool uses_tied_binary_form = false;
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_uses_tied_binary_form(
      context, &uses_tied_binary_form));
  if (uses_tied_binary_form) {
    return loom_x86_append_tied_binary_packet(context);
  }
  bool uses_tied_ternary_form = false;
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_uses_tied_ternary_form(
      context, &uses_tied_ternary_form));
  if (uses_tied_ternary_form) {
    return loom_x86_append_tied_ternary_packet(context);
  }
  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_mnemonic(context, &mnemonic));
  if (iree_string_view_starts_with(mnemonic, IREE_SV("cmp.set"))) {
    return loom_x86_append_cmp_setcc_packet(context);
  }
  if (iree_string_view_equal(mnemonic, IREE_SV("lea"))) {
    return loom_x86_append_lea_packet(context);
  }
  return loom_x86_append_canonical_asm_form_packet(context);
}

static iree_status_t loom_x86_append_return_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  return iree_string_builder_append_cstring(context->builder, "ret");
}

static iree_status_t loom_x86_append_branch_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  loom_value_slice_t args = loom_low_br_args(op);
  if (args.count != 0) {
    const loom_low_allocation_edge_copy_group_t* group =
        loom_low_allocation_find_edge_copy_group_by_source_ordinal(
            context->allocation, context->packet->node->source_ordinal);
    if (!group) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "x86 assembly branch edge copies are missing from allocation");
    }
    IREE_RETURN_IF_ERROR(loom_x86_emit_edge_copy_group(context, group));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "jmp "));
  return loom_native_assembly_append_block_label(
      context->schedule, loom_low_br_dest(op), context->builder);
}

static iree_status_t loom_x86_append_cond_branch_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  const loom_low_allocation_assignment_t* condition_assignment =
      loom_x86_map_assignment(context, loom_low_cond_br_condition(op));
  loom_x86_register_class_t register_class_kind = 0;
  IREE_RETURN_IF_ERROR(loom_x86_register_class_kind(
      context, condition_assignment, &register_class_kind));
  if (register_class_kind == LOOM_X86_REGISTER_CLASS_K) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "kortestq "));
    IREE_RETURN_IF_ERROR(
        loom_x86_append_assignment(context, condition_assignment));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, ", "));
    IREE_RETURN_IF_ERROR(
        loom_x86_append_assignment(context, condition_assignment));
  } else if (register_class_kind == LOOM_X86_REGISTER_CLASS_GPR32 ||
             register_class_kind == LOOM_X86_REGISTER_CLASS_GPR64) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "test "));
    IREE_RETURN_IF_ERROR(
        loom_x86_append_assignment(context, condition_assignment));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, ", "));
    IREE_RETURN_IF_ERROR(
        loom_x86_append_assignment(context, condition_assignment));
  } else {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "x86 assembly conditional branch requires a mask or GPR predicate");
  }

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "\n  jnz "));
  IREE_RETURN_IF_ERROR(loom_native_assembly_append_block_label(
      context->schedule, loom_low_cond_br_true_dest(op), context->builder));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "\n  jmp "));
  return loom_native_assembly_append_block_label(
      context->schedule, loom_low_cond_br_false_dest(op), context->builder);
}

iree_status_t loom_x86_emit_assembly_fragment(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_string_builder_t* builder, iree_arena_allocator_t* scratch_arena) {
  iree_string_view_t target_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
      schedule->target.descriptor_set,
      schedule->target.descriptor_set->target_key_string_offset, &target_key));
  if (!iree_string_view_equal(target_key, IREE_SV("x86"))) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "x86 assembly emitter received target '%.*s'",
                            (int)target_key.size, target_key.data);
  }
  const loom_native_assembly_structural_packet_callback_t
      structural_packet_callbacks[] = {
          {
              .op_kind = LOOM_OP_LOW_COPY,
              .append_packet =
                  {
                      .fn = loom_x86_append_copy_packet,
                      .user_data = NULL,
                  },
          },
          {
              .op_kind = LOOM_OP_LOW_RETURN,
              .append_packet =
                  {
                      .fn = loom_x86_append_return_packet,
                      .user_data = NULL,
                  },
          },
          {
              .op_kind = LOOM_OP_LOW_BR,
              .append_packet =
                  {
                      .fn = loom_x86_append_branch_packet,
                      .user_data = NULL,
                  },
          },
          {
              .op_kind = LOOM_OP_LOW_COND_BR,
              .append_packet =
                  {
                      .fn = loom_x86_append_cond_branch_packet,
                      .user_data = NULL,
                  },
          },
      };
  const loom_native_assembly_format_options_t options = {
      .append_descriptor_packet =
          {
              .fn = loom_x86_append_descriptor_packet,
              .user_data = NULL,
          },
      .structural_packet_callbacks = structural_packet_callbacks,
      .structural_packet_callback_count =
          IREE_ARRAYSIZE(structural_packet_callbacks),
  };
  return loom_native_assembly_format_fragment(schedule, allocation, &options,
                                              builder, scratch_arena);
}
