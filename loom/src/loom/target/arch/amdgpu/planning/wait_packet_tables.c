// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_packet_tables.h"

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"

void loom_amdgpu_wait_packet_analyze_target(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_wait_packet_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(out_target);
  IREE_ASSERT_EQ(descriptor_set->target_stable_id,
                 LOOM_AMDGPU_TARGET_STABLE_ID);
  *out_target = (loom_amdgpu_wait_packet_target_t){0};
  const uint16_t descriptor_set_ordinal =
      descriptor_set->descriptor_set_ordinal;
  const loom_amdgpu_wait_packet_descriptor_range_t* range =
      &loom_amdgpu_wait_packet_descriptor_ranges[descriptor_set_ordinal];
  *out_target = (loom_amdgpu_wait_packet_target_t){
      .descriptors =
          &loom_amdgpu_wait_packet_descriptors[range->first_descriptor],
      .descriptor_count = range->descriptor_count,
      .descriptor_lookup = &loom_amdgpu_wait_packet_descriptor_lookups
                               [range->first_descriptor_lookup],
      .descriptor_lookup_count = range->descriptor_lookup_count,
      .selections = loom_amdgpu_wait_packet_selections[descriptor_set_ordinal],
      .selection_count = LOOM_AMDGPU_WAIT_COUNTER_MASK_ALL + 1,
      .max_descriptor_immediate_count = range->max_descriptor_immediate_count,
  };
}

const loom_amdgpu_wait_packet_descriptor_immediate_template_t*
loom_amdgpu_wait_packet_descriptor_immediate(
    const loom_amdgpu_wait_packet_descriptor_template_t* packet_descriptor,
    uint16_t immediate_index) {
  const uint32_t immediate_row =
      packet_descriptor->immediate_start + immediate_index;
  return &loom_amdgpu_wait_packet_immediates[immediate_row];
}

const loom_low_descriptor_t* loom_amdgpu_wait_packet_resolve_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_wait_packet_descriptor_template_t* packet_descriptor) {
  const uint32_t descriptor_ordinal = loom_amdgpu_descriptor_ref_ordinal(
      descriptor_set, packet_descriptor->descriptor_ref);
  IREE_ASSERT_NE(descriptor_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ordinal);
  IREE_ASSERT(descriptor != NULL);
  return descriptor;
}

static const loom_amdgpu_wait_packet_descriptor_template_t*
loom_amdgpu_wait_packet_find_descriptor_template(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    const loom_amdgpu_wait_packet_target_t* target) {
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_descriptor_ordinal(descriptor_set, descriptor);
  IREE_ASSERT_NE(descriptor_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  IREE_ASSERT_LT(descriptor_ordinal, target->descriptor_lookup_count);
  const uint16_t descriptor_index_plus_one =
      target->descriptor_lookup[descriptor_ordinal];
  IREE_ASSERT_NE(descriptor_index_plus_one, 0u);
  const uint16_t descriptor_index = descriptor_index_plus_one - 1;
  IREE_ASSERT_LT(descriptor_index, target->descriptor_count);
  return &target->descriptors[descriptor_index];
}

static const loom_named_attr_t* loom_amdgpu_wait_packet_find_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  const loom_string_id_t name_id = loom_module_lookup_string(module, name);
  if (name_id == LOOM_STRING_ID_INVALID) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    if (attrs.entries[i].name_id == name_id) {
      return &attrs.entries[i];
    }
  }
  return NULL;
}

static uint16_t loom_amdgpu_wait_packet_immediate_value(
    const loom_module_t* module, const loom_op_t* op,
    const loom_amdgpu_wait_packet_descriptor_immediate_template_t* immediate) {
  const loom_named_attr_t* attr = loom_amdgpu_wait_packet_find_attr(
      module, loom_low_op_attrs(op), immediate->name);
  if (attr == NULL) {
    return immediate->no_wait_value;
  }
  IREE_ASSERT_EQ(attr->value.kind, LOOM_ATTR_I64);
  IREE_ASSERT_GE(attr->value.i64, 0);
  IREE_ASSERT_LE((uint64_t)attr->value.i64, UINT16_MAX);
  return (uint16_t)attr->value.i64;
}

uint32_t loom_amdgpu_wait_packet_explicit_counter_mask(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    const loom_amdgpu_wait_packet_target_t* target, const loom_module_t* module,
    const loom_op_t* op) {
  const loom_amdgpu_wait_packet_descriptor_template_t* packet_descriptor =
      loom_amdgpu_wait_packet_find_descriptor_template(descriptor_set,
                                                       descriptor, target);
  uint32_t counter_mask = 0;
  for (uint16_t i = 0; i < packet_descriptor->immediate_count; ++i) {
    const loom_amdgpu_wait_packet_descriptor_immediate_template_t* immediate =
        loom_amdgpu_wait_packet_descriptor_immediate(packet_descriptor, i);
    const uint16_t value =
        loom_amdgpu_wait_packet_immediate_value(module, op, immediate);
    if (value < immediate->no_wait_value) {
      counter_mask |= immediate->counter_mask;
    }
  }
  return counter_mask;
}
