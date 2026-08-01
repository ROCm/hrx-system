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

const loom_low_descriptor_t* loom_amdgpu_lookup_descriptor_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_descriptor_ref_descriptor(descriptor_set, descriptor_ref);
  IREE_ASSERT(descriptor != NULL,
              "generated AMDGPU lowering references missing descriptor ref");
  return descriptor;
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
