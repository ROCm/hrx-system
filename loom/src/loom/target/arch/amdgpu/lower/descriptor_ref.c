// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"

#include <stdint.h>

#include "loom/codegen/low/builder.h"
#include "loom/ops/op_defs.h"

bool loom_amdgpu_descriptor_set_has_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  if (descriptor_set == NULL) {
    return false;
  }
  return loom_amdgpu_descriptor_ref_ordinal(descriptor_set, descriptor_ref) !=
         LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
}

iree_string_view_t loom_amdgpu_descriptor_set_key(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (descriptor_set == NULL) {
    return IREE_SV("<missing>");
  }
  const iree_string_view_t descriptor_set_key = loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset);
  return iree_string_view_is_empty(descriptor_set_key) ? IREE_SV("<empty>")
                                                       : descriptor_set_key;
}

bool loom_amdgpu_descriptor_set_has_all_refs(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_descriptor_ref_t* descriptor_refs,
    iree_host_size_t descriptor_ref_count) {
  for (iree_host_size_t i = 0; i < descriptor_ref_count; ++i) {
    const loom_amdgpu_descriptor_ref_t descriptor_ref = descriptor_refs[i];
    if (descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
      continue;
    }
    if (!loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref)) {
      return false;
    }
  }
  return true;
}

bool loom_amdgpu_descriptor_requirements_present(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_descriptor_requirement_t* requirements,
    iree_host_size_t requirement_count,
    iree_string_view_t* out_constraint_key) {
  for (iree_host_size_t i = 0; i < requirement_count; ++i) {
    *out_constraint_key = requirements[i].constraint_key;
    if (!loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                            requirements[i].descriptor_ref)) {
      return false;
    }
  }
  return true;
}

bool loom_amdgpu_descriptor_requirement_present(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t constraint_key,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    iree_string_view_t* out_constraint_key) {
  *out_constraint_key = constraint_key;
  return loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref);
}

iree_status_t loom_amdgpu_lookup_descriptor_ref(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_low_descriptor_t** out_descriptor,
    loom_string_id_t* out_opcode_id) {
  *out_descriptor = NULL;
  *out_opcode_id = LOOM_STRING_ID_INVALID;
  const uint32_t descriptor_ordinal =
      loom_amdgpu_descriptor_ref_ordinal(descriptor_set, descriptor_ref);
  IREE_ASSERT(descriptor_ordinal != LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,
              "generated AMDGPU lowering references missing descriptor ref");
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ordinal);
  iree_string_view_t key = loom_low_descriptor_set_string(
      descriptor_set, descriptor->key_string_offset);
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, key, out_opcode_id));
  *out_descriptor = descriptor;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_make_descriptor_implicit_resource_type(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_type_t* out_type) {
  *out_type = loom_type_none();
  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand = &operands[i];
    if (operand->role != LOOM_LOW_OPERAND_ROLE_RESOURCE ||
        !iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_IMPLICIT)) {
      continue;
    }
    for (uint16_t j = 0; j < operand->reg_class_alt_count; ++j) {
      const uint32_t alt_index = operand->reg_class_alt_start + j;
      const loom_low_reg_class_alt_t* alt =
          &descriptor_set->reg_class_alts[alt_index];
      if (iree_any_bit_set(alt->flags, LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
        continue;
      }
      return loom_low_build_register_type(descriptor_set, alt->reg_class_id,
                                          operand->unit_count, out_type);
    }
  }
  IREE_ASSERT_UNREACHABLE(
      "selected AMDGPU descriptor has no implicit resource operand");
  IREE_BUILTIN_UNREACHABLE();
}

bool loom_amdgpu_descriptor_has_immediate(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, iree_string_view_t name) {
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[descriptor->immediate_start + i];
    if (iree_string_view_equal(
            loom_low_descriptor_set_string(descriptor_set,
                                           immediate->field_name_string_offset),
            name)) {
      return true;
    }
  }
  return false;
}

void loom_amdgpu_filter_descriptor_optional_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, iree_host_size_t required_count,
    loom_named_attr_t* attrs, iree_host_size_t* inout_attr_count) {
  iree_host_size_t filtered_count = required_count;
  for (iree_host_size_t i = required_count; i < *inout_attr_count; ++i) {
    const iree_string_view_t attr_name =
        builder->module->strings.entries[attrs[i].name_id];
    if (loom_amdgpu_descriptor_has_immediate(descriptor_set, descriptor,
                                             attr_name)) {
      attrs[filtered_count++] = attrs[i];
    }
  }
  *inout_attr_count = filtered_count;
}
