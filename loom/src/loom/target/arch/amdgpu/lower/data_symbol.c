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

static iree_status_t loom_amdgpu_data_symbol_build_attrs(
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

  loom_type_t address_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &address_type));
  loom_named_attr_t attributes[2];
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_data_symbol_build_attrs(builder, target, attributes));
  const loom_low_descriptor_t* descriptor = loom_amdgpu_lookup_descriptor_ref(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_SYMBOL);
  loom_op_t* address = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, /*access_flags=*/0,
      /*operands=*/NULL, /*operand_count=*/0,
      loom_make_named_attr_slice(attributes, IREE_ARRAYSIZE(attributes)),
      &address_type, /*result_count=*/1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &address));
  *out_low_address = loom_value_slice_get(loom_low_op_results(address), 0);
  return iree_ok_status();
}
