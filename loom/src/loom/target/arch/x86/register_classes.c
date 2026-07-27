// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/register_classes.h"

#include <inttypes.h>

iree_status_t loom_x86_register_class_name(
    loom_x86_register_class_t register_class, iree_string_view_t* out_name) {
  *out_name = iree_string_view_empty();
  switch (register_class) {
    case LOOM_X86_REGISTER_CLASS_GPR32:
      *out_name = IREE_SV("x86.gpr32");
      return iree_ok_status();
    case LOOM_X86_REGISTER_CLASS_GPR64:
      *out_name = IREE_SV("x86.gpr64");
      return iree_ok_status();
    case LOOM_X86_REGISTER_CLASS_XMM:
      *out_name = IREE_SV("x86.xmm");
      return iree_ok_status();
    case LOOM_X86_REGISTER_CLASS_YMM:
      *out_name = IREE_SV("x86.ymm");
      return iree_ok_status();
    case LOOM_X86_REGISTER_CLASS_ZMM:
      *out_name = IREE_SV("x86.zmm");
      return iree_ok_status();
    case LOOM_X86_REGISTER_CLASS_K:
      *out_name = IREE_SV("x86.k");
      return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown x86 register class %u",
                          (unsigned)register_class);
}

bool loom_x86_register_class_for_vector_bit_width(
    uint32_t vector_bit_width, loom_x86_register_class_t* out_register_class) {
  switch (vector_bit_width) {
    case 128:
      *out_register_class = LOOM_X86_REGISTER_CLASS_XMM;
      return true;
    case 256:
      *out_register_class = LOOM_X86_REGISTER_CLASS_YMM;
      return true;
    case 512:
      *out_register_class = LOOM_X86_REGISTER_CLASS_ZMM;
      return true;
    default:
      *out_register_class = LOOM_X86_REGISTER_CLASS_GPR32;
      return false;
  }
}

iree_status_t loom_x86_descriptor_set_register_class_id(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_x86_register_class_t register_class,
    uint16_t* out_descriptor_reg_class_id) {
  *out_descriptor_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  if (descriptor_set == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "x86 register class projection requires a descriptor set");
  }

  iree_string_view_t register_class_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_x86_register_class_name(register_class, &register_class_name));
  bool found = loom_low_descriptor_set_lookup_register_class(
      descriptor_set, register_class_name, out_descriptor_reg_class_id, NULL);
  if (found) {
    return iree_ok_status();
  }

  iree_string_view_t descriptor_set_key = loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset);
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "x86 descriptor set '%.*s' does not expose register class '%.*s'",
      (int)descriptor_set_key.size, descriptor_set_key.data,
      (int)register_class_name.size, register_class_name.data);
}

static iree_status_t loom_x86_register_class_from_name(
    iree_string_view_t register_class_name,
    loom_x86_register_class_t* out_register_class) {
  if (iree_string_view_equal(register_class_name, IREE_SV("x86.gpr32"))) {
    *out_register_class = LOOM_X86_REGISTER_CLASS_GPR32;
    return iree_ok_status();
  }
  if (iree_string_view_equal(register_class_name, IREE_SV("x86.gpr64"))) {
    *out_register_class = LOOM_X86_REGISTER_CLASS_GPR64;
    return iree_ok_status();
  }
  if (iree_string_view_equal(register_class_name, IREE_SV("x86.xmm"))) {
    *out_register_class = LOOM_X86_REGISTER_CLASS_XMM;
    return iree_ok_status();
  }
  if (iree_string_view_equal(register_class_name, IREE_SV("x86.ymm"))) {
    *out_register_class = LOOM_X86_REGISTER_CLASS_YMM;
    return iree_ok_status();
  }
  if (iree_string_view_equal(register_class_name, IREE_SV("x86.zmm"))) {
    *out_register_class = LOOM_X86_REGISTER_CLASS_ZMM;
    return iree_ok_status();
  }
  if (iree_string_view_equal(register_class_name, IREE_SV("x86.k"))) {
    *out_register_class = LOOM_X86_REGISTER_CLASS_K;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED, "unknown x86 descriptor register class '%.*s'",
      (int)register_class_name.size, register_class_name.data);
}

iree_status_t loom_x86_descriptor_set_logical_register_class(
    const loom_low_descriptor_set_t* descriptor_set,
    uint16_t descriptor_reg_class_id,
    loom_x86_register_class_t* out_register_class) {
  *out_register_class = LOOM_X86_REGISTER_CLASS_GPR32;
  if (descriptor_set == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "x86 register class projection requires a descriptor set");
  }
  if (descriptor_reg_class_id >= descriptor_set->reg_class_count) {
    iree_string_view_t descriptor_set_key = loom_low_descriptor_set_string(
        descriptor_set, descriptor_set->key_string_offset);
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "x86 descriptor set '%.*s' has no register class %" PRIu16,
        (int)descriptor_set_key.size, descriptor_set_key.data,
        descriptor_reg_class_id);
  }

  const loom_low_reg_class_t* register_class =
      &descriptor_set->reg_classes[descriptor_reg_class_id];
  iree_string_view_t register_class_name = loom_low_descriptor_set_string(
      descriptor_set, register_class->name_string_offset);
  return loom_x86_register_class_from_name(register_class_name,
                                           out_register_class);
}
