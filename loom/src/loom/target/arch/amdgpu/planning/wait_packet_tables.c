// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_packet_tables.h"

#include <inttypes.h>

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/planning/wait_counters.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"

typedef struct loom_amdgpu_wait_packet_descriptor_range_t {
  // First descriptor template row for this descriptor-set ordinal.
  uint16_t first_descriptor;
  // Number of descriptor template rows for this descriptor-set ordinal.
  uint16_t descriptor_count;
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
#include "loom/target/arch/amdgpu/planning/wait_packet_descriptors.inl"
};

#undef LOOM_AMDGPU_WAIT_PACKET_DESCRIPTOR

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
#include "loom/target/arch/amdgpu/planning/wait_packet_immediates.inl"
};

#undef LOOM_AMDGPU_WAIT_PACKET_IMMEDIATE

#define LOOM_AMDGPU_WAIT_PACKET_DESCRIPTOR_RANGE(                             \
    descriptor_set_ordinal_value, first_descriptor_value,                     \
    descriptor_count_value, max_descriptor_immediate_count_value)             \
  [descriptor_set_ordinal_value] = {                                          \
      .first_descriptor = first_descriptor_value,                             \
      .descriptor_count = descriptor_count_value,                             \
      .max_descriptor_immediate_count = max_descriptor_immediate_count_value, \
  },

static const loom_amdgpu_wait_packet_descriptor_range_t
    kAmdgpuWaitPacketDescriptorRanges
        [LOOM_AMDGPU_TARGET_REF_DESCRIPTOR_SET_ORDINAL_COUNT] = {
#include "loom/target/arch/amdgpu/planning/wait_packet_descriptor_ranges.inl"
};

#undef LOOM_AMDGPU_WAIT_PACKET_DESCRIPTOR_RANGE

static iree_status_t loom_amdgpu_wait_packet_verify_target(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (descriptor_set == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU wait packet materialization requires a descriptor set");
  }
  if (descriptor_set->target_stable_id != LOOM_AMDGPU_TARGET_STABLE_ID) {
    iree_string_view_t target_key = loom_low_descriptor_set_string(
        descriptor_set, descriptor_set->target_key_string_offset);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU wait packet materialization received target '%.*s'",
        (int)target_key.size, target_key.data);
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_wait_packet_analyze_target(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_wait_packet_target_t* out_target) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_verify_target(descriptor_set));
  *out_target = (loom_amdgpu_wait_packet_target_t){0};
  const uint16_t descriptor_set_ordinal =
      descriptor_set->descriptor_set_ordinal;
  if (descriptor_set_ordinal >=
      IREE_ARRAYSIZE(kAmdgpuWaitPacketDescriptorRanges)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU descriptor set ordinal %" PRIu16
                            " has no generated wait-packet table",
                            descriptor_set_ordinal);
  }
  const loom_amdgpu_wait_packet_descriptor_range_t* range =
      &kAmdgpuWaitPacketDescriptorRanges[descriptor_set_ordinal];
  IREE_ASSERT_LE(range->first_descriptor,
                 IREE_ARRAYSIZE(kAmdgpuWaitPacketDescriptors));
  IREE_ASSERT_LE(
      range->descriptor_count,
      IREE_ARRAYSIZE(kAmdgpuWaitPacketDescriptors) - range->first_descriptor);
  *out_target = (loom_amdgpu_wait_packet_target_t){
      .descriptors = &kAmdgpuWaitPacketDescriptors[range->first_descriptor],
      .descriptor_count = range->descriptor_count,
      .max_descriptor_immediate_count = range->max_descriptor_immediate_count,
  };
  return iree_ok_status();
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

iree_status_t loom_amdgpu_wait_packet_resolve_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_wait_packet_descriptor_template_t* packet_descriptor,
    const loom_low_descriptor_t** out_descriptor) {
  *out_descriptor = NULL;
  const uint32_t descriptor_ordinal = loom_amdgpu_descriptor_ref_ordinal(
      descriptor_set, packet_descriptor->descriptor_ref);
  IREE_ASSERT_NE(descriptor_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ordinal);
  IREE_ASSERT(descriptor != NULL);
  *out_descriptor = descriptor;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_wait_packet_find_descriptor_template(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    const loom_amdgpu_wait_packet_target_t* target,
    const loom_amdgpu_wait_packet_descriptor_template_t**
        out_packet_descriptor) {
  *out_packet_descriptor = NULL;
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_descriptor_ordinal(descriptor_set, descriptor);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU explicit wait packet uses a descriptor outside the target "
        "descriptor set");
  }
  for (iree_host_size_t i = 0; i < target->descriptor_count; ++i) {
    const loom_amdgpu_wait_packet_descriptor_template_t* packet_descriptor =
        &target->descriptors[i];
    const uint32_t packet_descriptor_ordinal =
        loom_amdgpu_descriptor_ref_ordinal(descriptor_set,
                                           packet_descriptor->descriptor_ref);
    if (packet_descriptor_ordinal == descriptor_ordinal) {
      *out_packet_descriptor = packet_descriptor;
      return iree_ok_status();
    }
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "AMDGPU explicit wait packet descriptor has no generated wait-packet "
      "template");
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

static iree_status_t loom_amdgpu_wait_packet_immediate_value(
    const loom_module_t* module, const loom_op_t* op,
    const loom_amdgpu_wait_packet_descriptor_immediate_template_t* immediate,
    uint16_t* out_value) {
  *out_value = immediate->no_wait_value;
  const loom_named_attr_t* attr = loom_amdgpu_wait_packet_find_attr(
      module, loom_low_op_attrs(op), immediate->name);
  if (attr == NULL) {
    return iree_ok_status();
  }
  if (attr->value.kind != LOOM_ATTR_I64 || attr->value.i64 < 0 ||
      attr->value.i64 > UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU verified explicit wait packet immediate '%.*s' is malformed",
        (int)immediate->name.size, immediate->name.data);
  }
  *out_value = (uint16_t)attr->value.i64;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_wait_packet_explicit_counter_mask(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, const loom_module_t* module,
    const loom_op_t* op, uint32_t* out_counter_mask) {
  *out_counter_mask = 0;

  loom_amdgpu_wait_packet_target_t target = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_packet_analyze_target(descriptor_set, &target));

  const loom_amdgpu_wait_packet_descriptor_template_t* packet_descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_find_descriptor_template(
      descriptor_set, descriptor, &target, &packet_descriptor));
  for (uint16_t i = 0; i < packet_descriptor->immediate_count; ++i) {
    const loom_amdgpu_wait_packet_descriptor_immediate_template_t* immediate =
        loom_amdgpu_wait_packet_descriptor_immediate(packet_descriptor, i);
    uint16_t value = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_wait_packet_immediate_value(module, op, immediate, &value));
    if (value < immediate->no_wait_value) {
      *out_counter_mask |= immediate->counter_mask;
    }
  }
  return iree_ok_status();
}
