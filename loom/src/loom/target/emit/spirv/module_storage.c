// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/module_storage.h"

#include "loom/ops/low/ops.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/isa.h"
#include "loom/target/arch/spirv/registers.h"
#include "loom/target/registers.h"

static uint32_t loom_spirv_module_workgroup_storage_element_byte_count(
    loom_spirv_scalar_type_t scalar_type) {
  const loom_spirv_scalar_type_descriptor_t* scalar_descriptor =
      loom_spirv_scalar_type_descriptor(scalar_type);
  IREE_ASSERT(scalar_descriptor != NULL);
  IREE_ASSERT_NE(scalar_descriptor->bit_width, 0u);
  IREE_ASSERT_EQ(scalar_descriptor->bit_width % 8, 0u);
  return scalar_descriptor->bit_width / 8;
}

static iree_status_t loom_spirv_module_workgroup_storage_emit_type(
    const loom_op_t* reserve_op, loom_spirv_scalar_type_t scalar_type,
    loom_spirv_type_context_t* type_context, uint32_t* out_array_type_id,
    uint32_t* out_array_pointer_type_id) {
  const uint32_t element_byte_count =
      loom_spirv_module_workgroup_storage_element_byte_count(scalar_type);
  const int64_t byte_length = loom_low_storage_reserve_byte_length(reserve_op);
  const int64_t byte_alignment =
      loom_low_storage_reserve_byte_alignment(reserve_op);
  IREE_ASSERT_GT(byte_length, 0);
  IREE_ASSERT_EQ(byte_length % (int64_t)element_byte_count, 0);
  IREE_ASSERT_GE(byte_alignment, (int64_t)element_byte_count);
  const uint64_t element_count = (uint64_t)byte_length / element_byte_count;
  if (element_count == 0 || element_count > UINT32_MAX) {
    IREE_CHECK_UNREACHABLE("verified SPIR-V Workgroup storage element count");
  }

  uint32_t element_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_scalar(type_context, scalar_type, &element_type_id));
  uint32_t length_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_u32_constant(
      type_context, (uint32_t)element_count, &length_id));
  // Ordinary Workgroup arrays use implicit layout. ArrayStride is only valid
  // for explicitly laid-out storage, which requires a Block and its capability.
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_array(type_context, element_type_id,
                                                  length_id, /*array_stride=*/0,
                                                  out_array_type_id));
  return loom_spirv_emit_type_pointer(
      type_context, LOOM_SPIRV_STORAGE_CLASS_WORKGROUP, *out_array_type_id,
      /*pointer_array_stride=*/0, out_array_pointer_type_id);
}

static iree_status_t loom_spirv_module_workgroup_storage_emit_variable(
    const loom_op_t* reserve_op, loom_spirv_value_type_t value_type,
    loom_spirv_type_context_t* type_context,
    loom_spirv_module_builder_t* builder,
    loom_spirv_module_value_ref_t* out_value_ref) {
  uint32_t array_type_id = 0;
  uint32_t array_pointer_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_workgroup_storage_emit_type(
      reserve_op, value_type.scalar_type, type_context, &array_type_id,
      &array_pointer_type_id));
  const uint32_t variable_id = loom_spirv_module_builder_allocate_id(builder);
  const uint32_t operands[] = {
      array_pointer_type_id,
      variable_id,
      LOOM_SPIRV_STORAGE_CLASS_WORKGROUP,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_module_builder_section(builder,
                                        LOOM_SPIRV_MODULE_SECTION_DECLARATION),
      LOOM_SPIRV_OP_VARIABLE, operands, IREE_ARRAYSIZE(operands)));
  *out_value_ref = (loom_spirv_module_value_ref_t){
      .id = variable_id,
      .type_id = array_pointer_type_id,
      .value_type = value_type,
  };
  return iree_ok_status();
}

iree_status_t loom_spirv_module_workgroup_storage_emit_address(
    loom_spirv_module_value_table_t* value_table, const loom_op_t* op,
    loom_spirv_type_context_t* type_context,
    loom_spirv_module_builder_t* builder,
    loom_spirv_module_value_ref_t* out_value_ref) {
  IREE_ASSERT_ARGUMENT(value_table);
  IREE_ASSERT_ARGUMENT(op);
  IREE_ASSERT_ARGUMENT(type_context);
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_value_ref);

  *out_value_ref = (loom_spirv_module_value_ref_t){0};
  IREE_ASSERT_EQ(loom_low_storage_address_offset(op), 0);
  const loom_local_value_domain_t* value_domain = value_table->value_domain;
  const loom_value_id_t storage = loom_low_storage_address_storage(op);
  const loom_value_t* storage_value =
      loom_module_value(value_domain->module, storage);
  IREE_ASSERT(storage_value != NULL);
  IREE_ASSERT(!loom_value_is_block_arg(storage_value));
  const loom_op_t* reserve_op = loom_value_def_op(storage_value);
  IREE_ASSERT(reserve_op != NULL);
  IREE_ASSERT(loom_low_storage_reserve_isa(reserve_op));

  loom_spirv_value_type_t value_type = {0};
  const loom_type_t result_type = loom_module_value_type(
      value_domain->module, loom_low_storage_address_result(op));
  IREE_ASSERT(loom_low_type_is_register(result_type));
  IREE_ASSERT_EQ(loom_low_register_type_descriptor_set_stable_id(result_type),
                 SPIRV_LOGICAL_CORE_DESCRIPTOR_SET_ID);
  const bool resolved = loom_spirv_value_type_from_reg_class_id(
      loom_low_register_type_class_id(result_type), &value_type);
  IREE_ASSERT(resolved);
  IREE_ASSERT_EQ(value_type.value_class,
                 LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY);
  if (loom_spirv_module_value_table_exists(value_table, storage)) {
    const loom_spirv_module_value_ref_t value_ref =
        loom_spirv_module_value_table_lookup(value_table, storage);
    IREE_ASSERT(loom_spirv_value_type_equal(value_ref.value_type, value_type));
    *out_value_ref = value_ref;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_spirv_module_workgroup_storage_emit_variable(
      reserve_op, value_type, type_context, builder, out_value_ref));
  loom_spirv_module_value_table_define(value_table, storage, *out_value_ref);
  return iree_ok_status();
}
