// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_packet_tables.h"

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/planning/wait_counters.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"

typedef struct loom_amdgpu_wait_packet_descriptor_range_t {
  // First descriptor template row for this descriptor-set ordinal.
  uint16_t first_descriptor;
  // Number of descriptor template rows for this descriptor-set ordinal.
  uint16_t descriptor_count;
  // First descriptor-ordinal lookup row for this descriptor-set ordinal.
  uint16_t first_descriptor_lookup;
  // Number of descriptor-ordinal lookup rows for this descriptor-set ordinal.
  uint16_t descriptor_lookup_count;
  // Maximum immediate template count owned by any descriptor row in the range.
  uint16_t max_descriptor_immediate_count;
} loom_amdgpu_wait_packet_descriptor_range_t;

#define LOOM_AMDGPU_WAIT_PACKET_DESCRIPTOR(                        \
    descriptor_ref_value, counter_mask_value, counter_count_value, \
    immediate_start_value, immediate_count_value)                  \
  {                                                                \
      .descriptor_ref = descriptor_ref_value,                      \
      .counter_mask = counter_mask_value,                          \
      .counter_count = counter_count_value,                        \
      .immediate_start = immediate_start_value,                    \
      .immediate_count = immediate_count_value,                    \
  },

static const loom_amdgpu_wait_packet_descriptor_template_t
    kAmdgpuWaitPacketDescriptors[] = {
#include "loom/target/arch/amdgpu/descriptors/wait_packet_descriptors.inl"
};

#undef LOOM_AMDGPU_WAIT_PACKET_DESCRIPTOR

#define LOOM_AMDGPU_WAIT_PACKET_DESCRIPTOR_LOOKUP( \
    descriptor_index_plus_one_value)               \
  descriptor_index_plus_one_value,

static const uint16_t kAmdgpuWaitPacketDescriptorLookups[] = {
#include "loom/target/arch/amdgpu/descriptors/wait_packet_descriptor_lookups.inl"
};

#undef LOOM_AMDGPU_WAIT_PACKET_DESCRIPTOR_LOOKUP

#define LOOM_AMDGPU_WAIT_PACKET_IMMEDIATE(descriptor_immediate_index_value, \
                                          name_value, counter_mask_value,   \
                                          no_wait_value_value)              \
  {                                                                         \
      .descriptor_immediate_index = descriptor_immediate_index_value,       \
      .name = IREE_SVL(name_value),                                         \
      .counter_mask = counter_mask_value,                                   \
      .no_wait_value = no_wait_value_value,                                 \
  },

static const loom_amdgpu_wait_packet_descriptor_immediate_template_t
    kAmdgpuWaitPacketImmediates[] = {
#include "loom/target/arch/amdgpu/descriptors/wait_packet_immediates.inl"
};

#undef LOOM_AMDGPU_WAIT_PACKET_IMMEDIATE

#define LOOM_AMDGPU_WAIT_PACKET_DESCRIPTOR_RANGE(                             \
    descriptor_set_ordinal_value, first_descriptor_value,                     \
    descriptor_count_value, first_descriptor_lookup_value,                    \
    descriptor_lookup_count_value, max_descriptor_immediate_count_value)      \
  [descriptor_set_ordinal_value] = {                                          \
      .first_descriptor = first_descriptor_value,                             \
      .descriptor_count = descriptor_count_value,                             \
      .first_descriptor_lookup = first_descriptor_lookup_value,               \
      .descriptor_lookup_count = descriptor_lookup_count_value,               \
      .max_descriptor_immediate_count = max_descriptor_immediate_count_value, \
  },

static const loom_amdgpu_wait_packet_descriptor_range_t
    kAmdgpuWaitPacketDescriptorRanges
        [LOOM_AMDGPU_TARGET_REF_DESCRIPTOR_SET_ORDINAL_COUNT] = {
#include "loom/target/arch/amdgpu/descriptors/wait_packet_descriptor_ranges.inl"
};

#undef LOOM_AMDGPU_WAIT_PACKET_DESCRIPTOR_RANGE

#define LOOM_AMDGPU_WAIT_PACKET_SELECTION(                                    \
    descriptor_set_ordinal_value, counter_mask_value, descriptor_index_value, \
    covered_counter_mask_value)                                               \
  [descriptor_set_ordinal_value][counter_mask_value] = {                      \
      .descriptor_index = descriptor_index_value,                             \
      .covered_counter_mask = covered_counter_mask_value,                     \
  },

static const loom_amdgpu_wait_packet_selection_template_t
    kAmdgpuWaitPacketSelections
        [LOOM_AMDGPU_TARGET_REF_DESCRIPTOR_SET_ORDINAL_COUNT]
        [LOOM_AMDGPU_WAIT_COUNTER_MASK_ALL + 1] = {
#include "loom/target/arch/amdgpu/descriptors/wait_packet_selections.inl"
};

#undef LOOM_AMDGPU_WAIT_PACKET_SELECTION

static_assert(sizeof(loom_amdgpu_wait_packet_selection_template_t) == 4,
              "wait-packet selection rows must stay compact");

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
  IREE_ASSERT_LT(descriptor_set_ordinal,
                 IREE_ARRAYSIZE(kAmdgpuWaitPacketDescriptorRanges));
  const loom_amdgpu_wait_packet_descriptor_range_t* range =
      &kAmdgpuWaitPacketDescriptorRanges[descriptor_set_ordinal];
  IREE_ASSERT_LE(range->first_descriptor,
                 IREE_ARRAYSIZE(kAmdgpuWaitPacketDescriptors));
  IREE_ASSERT_LE(
      range->descriptor_count,
      IREE_ARRAYSIZE(kAmdgpuWaitPacketDescriptors) - range->first_descriptor);
  IREE_ASSERT_LE(range->first_descriptor_lookup,
                 IREE_ARRAYSIZE(kAmdgpuWaitPacketDescriptorLookups));
  IREE_ASSERT_LE(range->descriptor_lookup_count,
                 IREE_ARRAYSIZE(kAmdgpuWaitPacketDescriptorLookups) -
                     range->first_descriptor_lookup);
  IREE_ASSERT_EQ(range->descriptor_lookup_count,
                 descriptor_set->descriptor_count);
  *out_target = (loom_amdgpu_wait_packet_target_t){
      .descriptors = &kAmdgpuWaitPacketDescriptors[range->first_descriptor],
      .descriptor_count = range->descriptor_count,
      .descriptor_lookup =
          &kAmdgpuWaitPacketDescriptorLookups[range->first_descriptor_lookup],
      .descriptor_lookup_count = range->descriptor_lookup_count,
      .selections = kAmdgpuWaitPacketSelections[descriptor_set_ordinal],
      .selection_count =
          IREE_ARRAYSIZE(kAmdgpuWaitPacketSelections[descriptor_set_ordinal]),
      .max_descriptor_immediate_count = range->max_descriptor_immediate_count,
  };
}

const loom_amdgpu_wait_packet_descriptor_immediate_template_t*
loom_amdgpu_wait_packet_descriptor_immediate(
    const loom_amdgpu_wait_packet_descriptor_template_t* packet_descriptor,
    uint16_t immediate_index) {
  const uint32_t immediate_row =
      packet_descriptor->immediate_start + immediate_index;
  IREE_ASSERT(immediate_row < IREE_ARRAYSIZE(kAmdgpuWaitPacketImmediates));
  return &kAmdgpuWaitPacketImmediates[immediate_row];
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
