// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/hal_materialization.h"

#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/transforms/task_abi.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/x86/call_abi.h"
#include "loom/target/arch/x86/descriptors/entry.h"
#include "loom/target/arch/x86/facts.h"
#include "loom/target/arch/x86/ops/ops.h"
#include "loom/target/arch/x86/register_classes.h"
#include "loom/target/registers.h"

typedef struct loom_x86_task_entry_t {
  // Resolved instruction vocabulary for the concrete function version.
  const loom_low_descriptor_set_t* descriptors;
  // View-local instruction identities used to realize task entry actions.
  const loom_x86_entry_descriptors_t* entry;
  // Reusable zero value in the XMM register class.
  loom_value_id_t xmm_zero;
  // Reusable zero value in the YMM register class.
  loom_value_id_t ymm_zero;
  // Reusable zero value in the ZMM register class.
  loom_value_id_t zmm_zero;
} loom_x86_task_entry_t;

static bool loom_x86_task_entry_initialize(
    const loom_low_descriptor_set_t* descriptors,
    loom_low_task_entry_builder_t* builder) {
  loom_x86_task_entry_t* target = builder->target_data;
  target->descriptors = descriptors;
  target->entry = loom_x86_entry_descriptors(descriptors);
  if (!target->entry) {
    return false;
  }
  target->xmm_zero = LOOM_VALUE_ID_INVALID;
  target->ymm_zero = LOOM_VALUE_ID_INVALID;
  target->zmm_zero = LOOM_VALUE_ID_INVALID;
  builder->word_type = loom_low_register_type(descriptors->stable_id,
                                              LOOM_X86_REGISTER_CLASS_GPR32, 1);
  builder->pointer_type = loom_low_register_type(
      descriptors->stable_id, LOOM_X86_REGISTER_CLASS_GPR64, 1);
  return true;
}

static iree_status_t loom_x86_hal_build_packet(
    loom_low_task_entry_builder_t* builder, uint32_t descriptor_ref,
    const loom_value_id_t* operands, uint16_t operand_count,
    iree_string_view_t immediate_name, loom_attribute_t immediate,
    loom_type_t result_type, loom_value_id_t* out_value) {
  const loom_x86_task_entry_t* target = builder->target_data;
  loom_named_attr_t attr = {.value = immediate};
  const uint16_t attr_count = iree_string_view_is_empty(immediate_name) ? 0 : 1;
  if (attr_count) {
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        builder->ir.module, immediate_name, &attr.name_id));
  }
  const loom_low_descriptor_set_t* set = target->descriptors;
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(set, descriptor_ref);
  loom_op_t* op = NULL;
  if (loom_low_descriptor_layout(set, descriptor)->op_kind ==
      LOOM_LOW_DESCRIPTOR_OP_KIND_CONST) {
    IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
        &builder->ir, set, descriptor,
        loom_make_named_attr_slice(&attr, attr_count), result_type,
        builder->location, &op));
  } else {
    IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
        &builder->ir, set, descriptor, 0, operands, operand_count,
        loom_make_named_attr_slice(&attr, attr_count), &result_type, 1, NULL, 0,
        builder->location, &op));
  }
  *out_value = loom_op_results(op)[0];
  return iree_ok_status();
}

static iree_status_t loom_x86_hal_build_gpr_load(
    loom_low_task_entry_builder_t* builder, loom_value_id_t base,
    uint32_t offset, uint8_t size, loom_type_t type,
    loom_value_id_t* out_value) {
  const loom_x86_task_entry_t* target = builder->target_data;
  // Parameter admission and the fixed invocation-state schema admit exactly
  // these load widths before any entry instructions are built.
  const uint32_t descriptor_ref = size == 1   ? target->entry->load_u8
                                  : size == 2 ? target->entry->load_u16
                                  : size == 4 ? target->entry->load_u32
                                              : target->entry->load_u64;
  return loom_x86_hal_build_packet(builder, descriptor_ref, &base, 1,
                                   IREE_SV("disp32"), loom_attr_i64(offset),
                                   type, out_value);
}

static iree_status_t loom_x86_hal_build_success(
    loom_low_task_entry_builder_t* builder, loom_value_id_t* out_value) {
  const loom_x86_task_entry_t* target = builder->target_data;
  return loom_x86_hal_build_packet(builder, target->entry->constant_u32, NULL,
                                   0, IREE_SV("imm32"), loom_attr_i64(0),
                                   builder->word_type, out_value);
}

static bool loom_x86_task_source_is_predicate_vector(loom_type_t source_type) {
  return loom_type_is_vector(source_type) &&
         loom_type_element_type(source_type) == LOOM_SCALAR_TYPE_I1;
}

static iree_string_view_t loom_x86_task_parameter_constraint(
    const loom_low_task_entry_builder_t* builder, loom_type_t source_type,
    loom_type_t carrier_type, uint8_t byte_length) {
  static const iree_string_view_t kConstraint = IREE_SVL(
      "a source-compatible single GPR32, GPR64, XMM, YMM, or ZMM carrier "
      "with an exact task constant load");
  if (!loom_low_type_is_register(carrier_type) ||
      loom_low_register_type_unit_count(carrier_type) != 1) {
    return kConstraint;
  }
  const uint16_t register_class = loom_low_register_type_class_id(carrier_type);
  loom_x86_call_abi_classification_t classification;
  if (!loom_x86_call_abi_classify_source_carrier(source_type, register_class,
                                                 &classification)) {
    return kConstraint;
  }
  const loom_x86_task_entry_t* target = builder->target_data;
  switch (register_class) {
    case LOOM_X86_REGISTER_CLASS_GPR32:
      return byte_length == 1 || byte_length == 2 || byte_length == 4
                 ? iree_string_view_empty()
                 : kConstraint;
    case LOOM_X86_REGISTER_CLASS_GPR64:
      return byte_length == 8 ? iree_string_view_empty() : kConstraint;
    case LOOM_X86_REGISTER_CLASS_XMM:
      return target->entry->maximum_simd_byte_length >= 16 &&
                     (byte_length == 1 || byte_length == 2 ||
                      byte_length == 4 || byte_length == 8 || byte_length == 16)
                 ? iree_string_view_empty()
                 : kConstraint;
    case LOOM_X86_REGISTER_CLASS_YMM:
      return target->entry->maximum_simd_byte_length >= 32 && byte_length == 32
                 ? iree_string_view_empty()
                 : kConstraint;
    case LOOM_X86_REGISTER_CLASS_ZMM:
      return target->entry->maximum_simd_byte_length >= 64 && byte_length == 64
                 ? iree_string_view_empty()
                 : kConstraint;
    default:
      return kConstraint;
  }
}

static iree_status_t loom_x86_task_emit_simd_zero(
    loom_low_task_entry_builder_t* builder, loom_type_t carrier_type,
    loom_value_id_t* out_value) {
  loom_x86_task_entry_t* target = builder->target_data;
  uint32_t descriptor_ref = target->entry->zero_xmm;
  loom_value_id_t* cached_value = &target->xmm_zero;
  switch (loom_low_register_type_class_id(carrier_type)) {
    case LOOM_X86_REGISTER_CLASS_XMM:
      descriptor_ref = target->entry->zero_xmm;
      cached_value = &target->xmm_zero;
      break;
    case LOOM_X86_REGISTER_CLASS_YMM:
      descriptor_ref = target->entry->zero_ymm;
      cached_value = &target->ymm_zero;
      break;
    case LOOM_X86_REGISTER_CLASS_ZMM:
      descriptor_ref = target->entry->zero_zmm;
      cached_value = &target->zmm_zero;
      break;
    default:
      IREE_ASSERT_UNREACHABLE("admitted x86 task SIMD carrier");
      break;
  }
  if (*cached_value == LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_x86_hal_build_packet(
        builder, descriptor_ref, NULL, 0, iree_string_view_empty(),
        loom_attr_absent(), carrier_type, cached_value));
  }
  *out_value = *cached_value;
  return iree_ok_status();
}

static iree_status_t loom_x86_task_emit_simd_load(
    loom_low_task_entry_builder_t* builder, loom_value_id_t base,
    uint32_t offset, uint8_t byte_length, loom_type_t carrier_type,
    loom_value_id_t* out_value) {
  const loom_x86_task_entry_t* target = builder->target_data;
  uint32_t descriptor_ref = target->entry->load_128_xmm;
  const uint16_t register_class = loom_low_register_type_class_id(carrier_type);
  if (register_class == LOOM_X86_REGISTER_CLASS_XMM && byte_length <= 2) {
    loom_value_id_t bits;
    IREE_RETURN_IF_ERROR(loom_x86_hal_build_gpr_load(
        builder, base, offset, byte_length, builder->word_type, &bits));
    descriptor_ref = target->entry->move_u32_xmm;
    return loom_x86_hal_build_packet(
        builder, descriptor_ref, &bits, 1, iree_string_view_empty(),
        loom_attr_absent(), carrier_type, out_value);
  }
  if (register_class == LOOM_X86_REGISTER_CLASS_XMM) {
    descriptor_ref = byte_length == 4   ? target->entry->load_32_xmm
                     : byte_length == 8 ? target->entry->load_64_xmm
                                        : target->entry->load_128_xmm;
  } else if (register_class == LOOM_X86_REGISTER_CLASS_YMM) {
    descriptor_ref = target->entry->load_256_ymm;
  } else if (register_class == LOOM_X86_REGISTER_CLASS_ZMM) {
    descriptor_ref = target->entry->load_512_zmm;
  } else {
    IREE_ASSERT_UNREACHABLE("admitted x86 task SIMD carrier");
  }
  return loom_x86_hal_build_packet(builder, descriptor_ref, &base, 1,
                                   IREE_SV("disp32"), loom_attr_i64(offset),
                                   carrier_type, out_value);
}

static iree_status_t loom_x86_task_emit_predicate_import(
    loom_low_task_entry_builder_t* builder, loom_value_id_t table,
    const loom_low_task_parameter_access_t* access,
    loom_value_id_t* out_value) {
  const loom_x86_task_entry_t* target = builder->target_data;
  loom_value_id_t bytes;
  IREE_RETURN_IF_ERROR(loom_x86_task_emit_simd_load(
      builder, table, access->value_offset, access->byte_length,
      access->carrier_type, &bytes));
  loom_value_id_t zero;
  IREE_RETURN_IF_ERROR(
      loom_x86_task_emit_simd_zero(builder, access->carrier_type, &zero));
  const uint16_t register_class =
      loom_low_register_type_class_id(access->carrier_type);
  const uint32_t subtract_ref = register_class == LOOM_X86_REGISTER_CLASS_XMM
                                    ? target->entry->subtract_i8_xmm
                                : register_class == LOOM_X86_REGISTER_CLASS_YMM
                                    ? target->entry->subtract_i8_ymm
                                    : target->entry->subtract_i8_zmm;
  const loom_value_id_t subtract_operands[] = {zero, bytes};
  loom_value_id_t lanes;
  IREE_RETURN_IF_ERROR(loom_x86_hal_build_packet(
      builder, subtract_ref, subtract_operands,
      IREE_ARRAYSIZE(subtract_operands), iree_string_view_empty(),
      loom_attr_absent(), access->carrier_type, &lanes));
  if (register_class != LOOM_X86_REGISTER_CLASS_XMM ||
      access->byte_length == 16) {
    *out_value = lanes;
    return iree_ok_status();
  }
  const uint32_t widen_ref =
      access->byte_length == 2   ? target->entry->widen_i8_i64_xmm
      : access->byte_length == 4 ? target->entry->widen_i8_i32_xmm
                                 : target->entry->widen_i8_i16_xmm;
  return loom_x86_hal_build_packet(builder, widen_ref, &lanes, 1,
                                   iree_string_view_empty(), loom_attr_absent(),
                                   access->carrier_type, out_value);
}

static iree_status_t loom_x86_task_emit_parameter(
    loom_low_task_entry_builder_t* builder,
    const loom_low_task_parameter_access_t* access, loom_value_id_t dispatch,
    loom_value_id_t* table, loom_value_id_t* out_value) {
  const loom_x86_task_entry_t* target = builder->target_data;
  const uint16_t register_class =
      loom_low_register_type_class_id(access->carrier_type);
  if (!access->used) {
    if (register_class == LOOM_X86_REGISTER_CLASS_XMM ||
        register_class == LOOM_X86_REGISTER_CLASS_YMM ||
        register_class == LOOM_X86_REGISTER_CLASS_ZMM) {
      return loom_x86_task_emit_simd_zero(builder, access->carrier_type,
                                          out_value);
    }
    return loom_x86_hal_build_packet(
        builder,
        access->byte_length <= 4 ? target->entry->constant_u32
                                 : target->entry->constant_u64,
        NULL, 0, access->byte_length <= 4 ? IREE_SV("imm32") : IREE_SV("imm64"),
        loom_attr_i64(0), access->carrier_type, out_value);
  }
  if (*table == LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_x86_hal_build_gpr_load(
        builder, dispatch, access->table_offset, LOOM_TASK_ABI64_POINTER_SIZE,
        builder->pointer_type, table));
  }
  if (register_class == LOOM_X86_REGISTER_CLASS_GPR32 ||
      register_class == LOOM_X86_REGISTER_CLASS_GPR64) {
    return loom_x86_hal_build_gpr_load(builder, *table, access->value_offset,
                                       access->byte_length,
                                       access->carrier_type, out_value);
  }
  if (loom_x86_task_source_is_predicate_vector(access->source_type)) {
    return loom_x86_task_emit_predicate_import(builder, *table, access,
                                               out_value);
  }
  return loom_x86_task_emit_simd_load(builder, *table, access->value_offset,
                                      access->byte_length, access->carrier_type,
                                      out_value);
}

static iree_status_t loom_x86_task_emit_builtin(
    loom_low_task_entry_builder_t* builder,
    const loom_task_builtin_info_t* builtin, loom_value_id_t state,
    loom_value_id_t* out_value) {
  const loom_x86_task_entry_t* target = builder->target_data;
  loom_value_id_t value;
  IREE_RETURN_IF_ERROR(
      loom_x86_hal_build_gpr_load(builder, state, builtin->offset,
                                  builtin->size, builder->word_type, &value));
  return loom_x86_hal_build_packet(builder, target->entry->widen_u32, &value, 1,
                                   iree_string_view_empty(), loom_attr_absent(),
                                   builder->pointer_type, out_value);
}

static iree_status_t loom_x86_task_emit_query(
    loom_low_task_entry_builder_t* builder, loom_region_t* body,
    loom_symbol_ref_t library_symbol, uint32_t required_version) {
  const loom_x86_task_entry_t* target = builder->target_data;
  loom_value_id_t compatible;
  const loom_value_id_t version = loom_region_entry_block(body)->arg_ids[0];
  IREE_RETURN_IF_ERROR(loom_x86_hal_build_packet(
      builder, target->entry->compare_uge_u32, &version, 1, IREE_SV("imm32"),
      loom_attr_i64(required_version), builder->word_type, &compatible));
  loom_block_t* supported = NULL;
  loom_block_t* unsupported = NULL;
  IREE_RETURN_IF_ERROR(
      loom_region_append_block(builder->ir.module, body, &supported));
  IREE_RETURN_IF_ERROR(
      loom_region_append_block(builder->ir.module, body, &unsupported));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(&builder->ir, compatible,
                                              supported, unsupported,
                                              builder->location, &op));
  loom_builder_set_block(&builder->ir, supported);
  loom_value_id_t address;
  IREE_RETURN_IF_ERROR(loom_x86_hal_build_packet(
      builder, target->entry->symbol_address, NULL, 0, IREE_SV("symbol"),
      loom_attr_symbol(library_symbol), builder->pointer_type, &address));
  IREE_RETURN_IF_ERROR(
      loom_low_return_build(&builder->ir, &address, 1, builder->location, &op));
  loom_builder_set_block(&builder->ir, unsupported);
  loom_value_id_t null_value;
  IREE_RETURN_IF_ERROR(loom_x86_hal_build_packet(
      builder, target->entry->constant_u64, NULL, 0, IREE_SV("imm64"),
      loom_attr_i64(0), builder->pointer_type, &null_value));
  IREE_RETURN_IF_ERROR(loom_low_return_build(&builder->ir, &null_value, 1,
                                             builder->location, &op));
  return iree_ok_status();
}

static const loom_low_task_entry_lowering_t loom_x86_task_entry_lowering = {
    .fact_type = &loom_x86_target_fact_type,
    .initialize = loom_x86_task_entry_initialize,
    .parameter_constraint = loom_x86_task_parameter_constraint,
    .emit_parameter = loom_x86_task_emit_parameter,
    .emit_builtin = loom_x86_task_emit_builtin,
    .emit_success = loom_x86_hal_build_success,
    .emit_query = loom_x86_task_emit_query,
};

iree_status_t loom_x86_materialize_hal_kernel_run(loom_pass_t* pass,
                                                  loom_module_t* module,
                                                  loom_func_like_t function) {
  loom_x86_task_entry_t target;
  return loom_low_task_materialize_kernel(
      pass, module, function, &loom_x86_task_entry_lowering, &target);
}

iree_status_t loom_x86_materialize_hal_query_run(loom_pass_t* pass,
                                                 loom_module_t* module) {
  loom_x86_task_entry_t target;
  return loom_low_task_materialize_query(
      pass, module, &loom_x86_task_entry_lowering, &target);
}
