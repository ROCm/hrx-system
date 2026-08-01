// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/data_symbol.h"

#include <stdint.h>

#include "loom/codegen/low/builder.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static iree_status_t loom_amdgpu_data_symbol_build_rel32_attrs(
    loom_builder_t* builder, loom_amdgpu_data_symbol_address_t target,
    loom_named_attr_t* out_attrs) {
  loom_string_id_t byte_offset_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(
      builder, IREE_SV("byte_offset"), &byte_offset_name_id));
  loom_string_id_t symbol_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_builder_intern_string(builder, IREE_SV("symbol"), &symbol_name_id));
  out_attrs[0] = (loom_named_attr_t){
      .name_id = symbol_name_id,
      .value = loom_attr_symbol(target.symbol),
  };
  out_attrs[1] = (loom_named_attr_t){
      .name_id = byte_offset_name_id,
      .value = loom_attr_i64((int64_t)target.byte_offset),
  };
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_data_symbol_address(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_data_symbol_address_t target, loom_location_id_t location,
    loom_value_id_t* out_low_address) {
  IREE_ASSERT_ARGUMENT(out_low_address);
  *out_low_address = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_LE(target.byte_offset, INT64_MAX);

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &sgpr_x2_type));

  const loom_low_descriptor_t* getpc_descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set,
                                        LOOM_AMDGPU_DESCRIPTOR_REF_S_GETPC_B64);
  loom_op_t* getpc_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, getpc_descriptor,
      /*operands=*/NULL, /*operand_count=*/0,
      loom_make_named_attr_slice(NULL, 0), &sgpr_x2_type, /*result_count=*/1,
      /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &getpc_op));

  const loom_value_id_t pc =
      loom_value_slice_get(loom_low_op_results(getpc_op), 0);
  loom_op_t* pc_low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(builder, pc, /*offset=*/0,
                                            sgpr_type, location, &pc_low_op));
  loom_op_t* pc_high_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(builder, pc, /*offset=*/1,
                                            sgpr_type, location, &pc_high_op));

  loom_named_attr_t rel32_attrs[2];
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_data_symbol_build_rel32_attrs(builder, target, rel32_attrs));

  const loom_low_descriptor_t* add_low_descriptor =
      loom_amdgpu_lookup_descriptor_ref(
          descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32_RHS_SYMBOL_REL32_LO);
  const loom_value_id_t pc_low = loom_low_slice_result(pc_low_op);
  loom_op_t* address_low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, add_low_descriptor, &pc_low,
      /*operand_count=*/1,
      loom_make_named_attr_slice(rel32_attrs, IREE_ARRAYSIZE(rel32_attrs)),
      &sgpr_type, /*result_count=*/1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &address_low_op));

  const loom_low_descriptor_t* add_high_descriptor =
      loom_amdgpu_lookup_descriptor_ref(
          descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32_RHS_SYMBOL_REL32_HI);
  const loom_value_id_t pc_high = loom_low_slice_result(pc_high_op);
  loom_op_t* address_high_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, add_high_descriptor, &pc_high,
      /*operand_count=*/1,
      loom_make_named_attr_slice(rel32_attrs, IREE_ARRAYSIZE(rel32_attrs)),
      &sgpr_type, /*result_count=*/1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &address_high_op));

  const loom_value_id_t address_parts[] = {
      loom_value_slice_get(loom_low_op_results(address_low_op), 0),
      loom_value_slice_get(loom_low_op_results(address_high_op), 0),
  };
  loom_op_t* address_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      builder, address_parts, IREE_ARRAYSIZE(address_parts), sgpr_x2_type,
      location, &address_op));
  *out_low_address = loom_low_concat_result(address_op);
  return iree_ok_status();
}
