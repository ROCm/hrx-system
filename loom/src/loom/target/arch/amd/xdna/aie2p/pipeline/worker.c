// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/worker.h"

#include "iree/base/internal/math.h"

iree_status_t loom_aie2p_worker_builder_initialize(
    loom_module_t* module, iree_arena_allocator_t* arena,
    loom_aie2p_worker_builder_t* out_builder) {
  *out_builder = (loom_aie2p_worker_builder_t){
      .module = module,
      .arena = arena,
      .descriptors = loom_aie2p_core_descriptor_set(),
  };
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    if (module->symbols.entries[i].name_id == LOOM_STRING_ID_INVALID) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_symbol_map_insert(
        &out_builder->names, arena, module->symbols.entries[i].name_id,
        (loom_symbol_id_t)i));
  }
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("amd.xdna.aie2p.core"), &out_builder->contract));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("i"),
                                                 &out_builder->integer_name));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("imm"), &out_builder->displacement_name));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("id"),
                                                 &out_builder->lock_name));
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      out_builder->descriptors, AIE2P_CORE_REG_CLASS_ID_AIE2P_ER, 1,
      &out_builder->scalar_type));
  return loom_low_build_register_type(out_builder->descriptors,
                                      AIE2P_CORE_REG_CLASS_ID_AIE2P_EP, 1,
                                      &out_builder->address_type);
}

iree_status_t loom_aie2p_worker_symbol(loom_aie2p_worker_builder_t* context,
                                       loom_symbol_ref_t* out_symbol) {
  loom_string_id_t name = LOOM_STRING_ID_INVALID;
  do {
    char text[48];
    iree_snprintf(text, sizeof(text), "__aie2p_worker_helper_%u",
                  context->next_name++);
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        context->module, iree_make_cstring_view(text), &name));
  } while (loom_symbol_map_find(&context->names, name) !=
           LOOM_SYMBOL_ID_INVALID);
  *out_symbol = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(
      loom_module_add_symbol(context->module, name, &out_symbol->symbol_id));
  out_symbol->module_id = 0;
  IREE_RETURN_IF_ERROR(loom_symbol_map_insert(&context->names, context->arena,
                                              name, out_symbol->symbol_id));
  return iree_ok_status();
}

iree_status_t loom_aie2p_worker_helper(
    loom_aie2p_worker_builder_t* context, const loom_type_t* arguments,
    iree_host_size_t argument_count, const loom_type_t* results,
    iree_host_size_t result_count, loom_builder_t* builder,
    loom_symbol_ref_t* out_symbol, loom_op_t** out_function) {
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_symbol(context, out_symbol));
  loom_builder_initialize(context->module, &context->module->arena,
                          loom_module_block(context->module), builder);
  IREE_RETURN_IF_ERROR(loom_low_func_def_build(
      builder, 0, 0, 0, 0, 0, 0, 0, 0, 0, context->contract,
      loom_symbol_ref_null(), 0, loom_named_attr_slice_empty(),
      loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(), *out_symbol, arguments, argument_count,
      results, result_count, NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN,
      out_function));
  IREE_RETURN_IF_ERROR(loom_pipeline_realization_register_helper(
      context->module, context->realization, context->worker_index,
      loom_func_like_cast(context->module, *out_function),
      context->diagnostic_emitter));
  loom_builder_enter_region(builder, *out_function,
                            loom_low_func_def_body(*out_function));
  return iree_ok_status();
}

iree_status_t loom_aie2p_worker_constant(loom_aie2p_worker_builder_t* context,
                                         loom_builder_t* builder, int32_t value,
                                         loom_value_id_t* out_value) {
  const uint32_t descriptor = value >= -1024 && value <= 1023
                                  ? AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_SHORT
                                  : AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32;
  const loom_named_attr_t attribute = {.name_id = context->integer_name,
                                       .value = loom_attr_i64(value)};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      builder, context->descriptors,
      &context->descriptors->descriptors[descriptor],
      loom_make_named_attr_slice(&attribute, 1), context->scalar_type,
      LOOM_LOCATION_UNKNOWN, &op));
  *out_value = loom_low_const_result(op);
  return iree_ok_status();
}

iree_status_t loom_aie2p_worker_op(loom_aie2p_worker_builder_t* context,
                                   loom_builder_t* builder, uint32_t descriptor,
                                   const loom_value_id_t* operands,
                                   iree_host_size_t operand_count,
                                   loom_named_attr_slice_t attributes,
                                   const loom_type_t* result_type,
                                   loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, context->descriptors,
      &context->descriptors->descriptors[descriptor], 0, operands,
      operand_count, attributes, result_type, result_type ? 1 : 0, NULL, 0,
      LOOM_LOCATION_UNKNOWN, &op));
  if (out_value) {
    *out_value = loom_op_results(op)[0];
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_worker_binary(loom_aie2p_worker_builder_t* context,
                                       loom_builder_t* builder,
                                       uint32_t descriptor, loom_value_id_t lhs,
                                       loom_value_id_t rhs,
                                       loom_value_id_t* out_value) {
  const loom_value_id_t operands[] = {lhs, rhs};
  return loom_aie2p_worker_op(context, builder, descriptor, operands, 2,
                              loom_named_attr_slice_empty(),
                              &context->scalar_type, out_value);
}

iree_status_t loom_aie2p_worker_return(loom_builder_t* builder,
                                       const loom_value_id_t* values,
                                       iree_host_size_t count) {
  loom_op_t* op = NULL;
  return loom_low_return_build(builder, values, count, LOOM_LOCATION_UNKNOWN,
                               &op);
}

iree_status_t loom_aie2p_worker_lock(loom_aie2p_worker_builder_t* context,
                                     loom_builder_t* builder,
                                     uint32_t descriptor, uint16_t selector,
                                     int32_t delta) {
  loom_value_id_t value;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_worker_constant(context, builder, delta, &value));
  const loom_named_attr_t attribute = {.name_id = context->lock_name,
                                       .value = loom_attr_i64(selector)};
  return loom_aie2p_worker_op(context, builder, descriptor, &value, 1,
                              loom_make_named_attr_slice(&attribute, 1), NULL,
                              NULL);
}

static uint32_t loom_aie2p_worker_odd_parity(uint32_t word) {
  return word | ((iree_math_count_ones_u32(word) & 1u) ^ 1u) << 31;
}

static iree_status_t loom_aie2p_worker_control_write_packet(
    loom_aie2p_worker_builder_t* context, loom_builder_t* builder,
    loom_value_id_t route, loom_value_id_t command,
    const loom_value_id_t* words, iree_host_size_t count) {
  const loom_value_id_t headers[] = {route, command};
  for (iree_host_size_t i = 0; i < 2; ++i) {
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_op(
        context, builder, AIE2P_CORE_DESCRIPTOR_REF_STREAM_WRITE_I32,
        &headers[i], 1, loom_named_attr_slice_empty(), NULL, NULL));
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_op(
        context, builder,
        i + 1 == count ? AIE2P_CORE_DESCRIPTOR_REF_STREAM_WRITE_LAST_I32
                       : AIE2P_CORE_DESCRIPTOR_REF_STREAM_WRITE_I32,
        &words[i], 1, loom_named_attr_slice_empty(), NULL, NULL));
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_worker_control_write(
    loom_aie2p_worker_builder_t* context, loom_builder_t* builder,
    uint8_t packet_id, uint32_t register_offset, const loom_value_id_t* words,
    iree_host_size_t count) {
  loom_value_id_t route, command;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(
      context, builder, (int32_t)loom_aie2p_worker_odd_parity(packet_id),
      &route));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(
      context, builder,
      (int32_t)loom_aie2p_worker_control_write_command(register_offset, count),
      &command));
  return loom_aie2p_worker_control_write_packet(context, builder, route,
                                                command, words, count);
}

uint32_t loom_aie2p_worker_control_write_command(uint32_t register_offset,
                                                 iree_host_size_t count) {
  return loom_aie2p_worker_odd_parity(register_offset |
                                      ((uint32_t)(count - 1) << 20));
}

iree_status_t loom_aie2p_worker_control_write_encoded(
    loom_aie2p_worker_builder_t* context, loom_builder_t* builder,
    uint8_t packet_id, loom_value_id_t command, const loom_value_id_t* words,
    iree_host_size_t count) {
  loom_value_id_t route;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(
      context, builder, (int32_t)loom_aie2p_worker_odd_parity(packet_id),
      &route));
  return loom_aie2p_worker_control_write_packet(context, builder, route,
                                                command, words, count);
}

iree_status_t loom_aie2p_worker_control_read(
    loom_aie2p_worker_builder_t* context, loom_builder_t* builder,
    uint8_t packet_id, uint8_t reply_id, uint32_t register_offset,
    iree_host_size_t count, loom_value_id_t* out_values) {
  loom_value_id_t route, command;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(
      context, builder, (int32_t)loom_aie2p_worker_odd_parity(packet_id),
      &route));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(
      context, builder,
      (int32_t)loom_aie2p_worker_odd_parity(
          register_offset | (uint32_t)(count - 1) << 20 | UINT32_C(1) << 22 |
          (uint32_t)reply_id << 24),
      &command));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_op(
      context, builder, AIE2P_CORE_DESCRIPTOR_REF_STREAM_WRITE_I32, &route, 1,
      loom_named_attr_slice_empty(), NULL, NULL));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_op(
      context, builder, AIE2P_CORE_DESCRIPTOR_REF_STREAM_WRITE_LAST_I32,
      &command, 1, loom_named_attr_slice_empty(), NULL, NULL));
  loom_value_id_t header;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_op(
      context, builder, AIE2P_CORE_DESCRIPTOR_REF_STREAM_READ_I32, NULL, 0,
      loom_named_attr_slice_empty(), &context->scalar_type, &header));
  for (iree_host_size_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_op(
        context, builder, AIE2P_CORE_DESCRIPTOR_REF_STREAM_READ_I32, NULL, 0,
        loom_named_attr_slice_empty(), &context->scalar_type, &out_values[i]));
  }
  return iree_ok_status();
}
