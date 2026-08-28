// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/lower/constants.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/builder.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/vm/descriptors.h"

static uint64_t loom_vm_float_constant_bits(loom_scalar_type_t scalar_type,
                                            double value) {
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_F8E4M3:
      return iree_math_f32_to_f8e4m3fn((float)value);
    case LOOM_SCALAR_TYPE_F8E5M2:
      return iree_math_f32_to_f8e5m2((float)value);
    case LOOM_SCALAR_TYPE_F16:
      return iree_math_f32_to_f16((float)value);
    case LOOM_SCALAR_TYPE_BF16:
      return iree_math_f32_to_bf16((float)value);
    case LOOM_SCALAR_TYPE_F32: {
      const float f32_value = (float)value;
      uint32_t bits = 0;
      memcpy(&bits, &f32_value, sizeof(bits));
      return bits;
    }
    case LOOM_SCALAR_TYPE_F64: {
      uint64_t bits = 0;
      memcpy(&bits, &value, sizeof(bits));
      return bits;
    }
    default:
      IREE_ASSERT_UNREACHABLE("verified VM float constant type");
      IREE_BUILTIN_UNREACHABLE();
  }
}

uint64_t loom_vm_constant_bits_from_scalar_attr(loom_scalar_type_t scalar_type,
                                                loom_attribute_t value) {
  if (loom_scalar_type_is_float(scalar_type)) {
    return loom_vm_float_constant_bits(scalar_type, loom_attr_as_f64(value));
  }
  const int64_t integer_value =
      scalar_type == LOOM_SCALAR_TYPE_I1 && value.kind == LOOM_ATTR_BOOL
          ? (loom_attr_as_bool(value) ? 1 : 0)
          : loom_attr_as_i64(value);
  const int32_t bit_width = loom_scalar_type_bitwidth(scalar_type);
  IREE_ASSERT_GT(bit_width, 0);
  return iree_math_mask_low_bits_u64((uint64_t)integer_value, bit_width);
}

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

static bool loom_vm_inline_constant_try_immediate(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t immediate_index,
    loom_named_attr_slice_t attrs, int64_t* out_value) {
  const loom_low_immediate_t* immediate =
      &descriptor_set
           ->immediates[descriptor->immediate_start + immediate_index];
  const iree_string_view_t immediate_name = loom_low_descriptor_set_string(
      descriptor_set, immediate->field_name_string_offset);
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (attr->name_id >= module->strings.count ||
        !iree_string_view_equal(module->strings.entries[attr->name_id],
                                immediate_name)) {
      continue;
    }
    if (attr->value.kind != LOOM_ATTR_I64) return false;
    *out_value = loom_attr_as_i64(attr->value);
    return true;
  }
  return false;
}

bool loom_vm_inline_constant_try_decode(
    const loom_module_t* module, const loom_op_t* op,
    loom_vm_inline_constant_t* out_constant) {
  *out_constant = (loom_vm_inline_constant_t){0};
  if (!loom_low_const_isa(op)) return false;

  const uint32_t descriptor_ordinal = loom_low_const_descriptor(op);
  if (descriptor_ordinal != VM_CORE_DESCRIPTOR_REF_CONSTANT_ZERO &&
      descriptor_ordinal != VM_CORE_DESCRIPTOR_REF_CONSTANT_S16 &&
      descriptor_ordinal != VM_CORE_DESCRIPTOR_REF_CONSTANT_I32 &&
      descriptor_ordinal != VM_CORE_DESCRIPTOR_REF_CONSTANT_I64) {
    return false;
  }
  const loom_low_descriptor_set_t* descriptor_set =
      loom_vm_core_descriptor_set();
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ordinal);
  if (descriptor == NULL) return false;

  int64_t immediate_values[2] = {0};
  const loom_named_attr_slice_t attrs = loom_low_const_attrs(op);
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    if (!loom_vm_inline_constant_try_immediate(module, descriptor_set,
                                               descriptor, i, attrs,
                                               &immediate_values[i])) {
      return false;
    }
  }

  uint64_t bits = 0;
  switch (descriptor_ordinal) {
    case VM_CORE_DESCRIPTOR_REF_CONSTANT_ZERO:
      break;
    case VM_CORE_DESCRIPTOR_REF_CONSTANT_S16:
      bits = (uint64_t)(int64_t)(int16_t)immediate_values[0];
      break;
    case VM_CORE_DESCRIPTOR_REF_CONSTANT_I32:
      bits = (uint32_t)immediate_values[0];
      break;
    case VM_CORE_DESCRIPTOR_REF_CONSTANT_I64:
      bits = (uint32_t)immediate_values[0] |
             ((uint64_t)(uint32_t)immediate_values[1] << 32);
      break;
    default:
      IREE_ASSERT_UNREACHABLE("recognized VM inline constant descriptor");
      IREE_BUILTIN_UNREACHABLE();
  }
  *out_constant = (loom_vm_inline_constant_t){
      .bits = bits,
      .descriptor_ordinal = (uint16_t)descriptor_ordinal,
  };
  return true;
}

static int64_t loom_vm_s16_immediate(uint64_t bits) {
  const uint16_t low_bits = (uint16_t)bits;
  return iree_any_bit_set(low_bits, UINT16_C(0x8000))
             ? (int64_t)(low_bits & UINT16_C(0x7FFF)) - INT64_C(32768)
             : (int64_t)low_bits;
}

iree_status_t loom_vm_inline_constant_build(loom_builder_t* builder,
                                            uint64_t bits,
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
      IREE_BUILTIN_UNREACHABLE();
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
