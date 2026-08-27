// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/lower/constants.h"

#include "loom/codegen/low/builder.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/vm/descriptors.h"

uint16_t loom_vm_constant_descriptor_ordinal(uint64_t bits) {
  if (bits == 0) return VM_CORE_DESCRIPTOR_REF_CONSTANT_ZERO;
  const uint64_t sign_extension = iree_any_bit_set(bits, UINT64_C(0x8000))
                                      ? UINT64_C(0xFFFFFFFFFFFF0000)
                                      : 0;
  if (bits == ((bits & UINT64_C(0xFFFF)) | sign_extension)) {
    return VM_CORE_DESCRIPTOR_REF_CONSTANT_S16;
  }
  if (bits <= UINT32_MAX) return VM_CORE_DESCRIPTOR_REF_CONSTANT_I32;
  return VM_CORE_DESCRIPTOR_REF_CONSTANT_I64;
}

static int64_t loom_vm_s16_immediate(uint64_t bits) {
  const uint16_t low_bits = (uint16_t)bits;
  return iree_any_bit_set(low_bits, UINT16_C(0x8000))
             ? (int64_t)(low_bits & UINT16_C(0x7FFF)) - INT64_C(32768)
             : (int64_t)low_bits;
}

iree_status_t loom_vm_constant_build(loom_builder_t* builder, uint64_t bits,
                                     loom_type_t result_type,
                                     loom_location_id_t location,
                                     loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_vm_core_descriptor_set();
  const uint16_t descriptor_ordinal = loom_vm_constant_descriptor_ordinal(bits);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ordinal);
  IREE_ASSERT(descriptor != NULL);

  int64_t immediate_values[2] = {0};
  iree_host_size_t immediate_count = 0;
  switch (descriptor_ordinal) {
    case VM_CORE_DESCRIPTOR_REF_CONSTANT_ZERO:
      break;
    case VM_CORE_DESCRIPTOR_REF_CONSTANT_S16:
      immediate_values[0] = loom_vm_s16_immediate(bits);
      immediate_count = 1;
      break;
    case VM_CORE_DESCRIPTOR_REF_CONSTANT_I32:
      immediate_values[0] = (int64_t)(uint32_t)bits;
      immediate_count = 1;
      break;
    case VM_CORE_DESCRIPTOR_REF_CONSTANT_I64:
      immediate_values[0] = (int64_t)(uint32_t)bits;
      immediate_values[1] = (int64_t)(uint32_t)(bits >> 32);
      immediate_count = 2;
      break;
    default:
      IREE_ASSERT_UNREACHABLE("selected VM constant descriptor");
      return iree_ok_status();
  }

  IREE_ASSERT_EQ(immediate_count, descriptor->immediate_count);
  loom_named_attr_t attrs[2] = {0};
  for (iree_host_size_t i = 0; i < immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[descriptor->immediate_start + i];
    const iree_string_view_t immediate_name = loom_low_descriptor_set_string(
        descriptor_set, immediate->field_name_string_offset);
    IREE_RETURN_IF_ERROR(
        loom_builder_intern_string(builder, immediate_name, &attrs[i].name_id));
    attrs[i].value = loom_attr_i64(immediate_values[i]);
  }

  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      builder, descriptor_set, descriptor,
      loom_make_named_attr_slice(attrs, immediate_count), result_type, location,
      &constant_op));
  *out_value = loom_low_const_result(constant_op);
  return iree_ok_status();
}
