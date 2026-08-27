// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable descriptor-derived AMDGPU wait-packet data.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_PACKET_DATA_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_PACKET_DATA_H_

#include "iree/base/api.h"
#include "loom/target/arch/amdgpu/planning/wait_counters.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_wait_packet_descriptor_immediate_template_t {
  // Descriptor-local immediate index populated by this row.
  uint16_t descriptor_immediate_index;
  // Immediate field name populated by this row.
  iree_string_view_t name;
  // Logical counters controlled by this immediate field.
  uint32_t counter_mask;
  // Immediate value that leaves this field unconstrained.
  uint16_t no_wait_value;
} loom_amdgpu_wait_packet_descriptor_immediate_template_t;

typedef struct loom_amdgpu_wait_packet_descriptor_template_t {
  // Stable descriptor ref selected for this wait packet.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // Logical counters this descriptor can drain.
  uint32_t counter_mask;
  // Number of logical counters this descriptor can drain.
  uint8_t counter_count;
  // First immediate template row owned by this descriptor row.
  uint16_t immediate_start;
  // Number of immediate template rows owned by this descriptor row.
  uint16_t immediate_count;
} loom_amdgpu_wait_packet_descriptor_template_t;

typedef struct loom_amdgpu_wait_packet_selection_template_t {
  // Local descriptor-template index selected for this counter mask.
  uint16_t descriptor_index;
  // Logical counters covered by the selected descriptor.
  uint8_t covered_counter_mask;
} loom_amdgpu_wait_packet_selection_template_t;

typedef struct loom_amdgpu_wait_packet_descriptor_range_t {
  // First descriptor template row for this descriptor-set ordinal.
  uint16_t first_descriptor;
  // Number of descriptor template rows for this descriptor-set ordinal.
  uint16_t descriptor_count;
  // First descriptor-ordinal lookup row for this descriptor-set ordinal.
  uint16_t first_descriptor_lookup;
  // Number of descriptor-ordinal lookup rows for this descriptor-set ordinal.
  uint16_t descriptor_lookup_count;
  // Maximum immediate template count owned by any descriptor in this range.
  uint16_t max_descriptor_immediate_count;
} loom_amdgpu_wait_packet_descriptor_range_t;

// Descriptor templates ordered by descriptor set and descriptor ordinal.
extern const loom_amdgpu_wait_packet_descriptor_template_t
    loom_amdgpu_wait_packet_descriptors[];

// Immediate templates referenced by descriptor template ranges.
extern const loom_amdgpu_wait_packet_descriptor_immediate_template_t
    loom_amdgpu_wait_packet_immediates[];

// Dense descriptor-ordinal to local descriptor-template index+1 lookups.
extern const uint16_t loom_amdgpu_wait_packet_descriptor_lookups[];

// Descriptor and lookup ranges indexed by descriptor-set ordinal.
extern const loom_amdgpu_wait_packet_descriptor_range_t
    loom_amdgpu_wait_packet_descriptor_ranges[];

// Best descriptor template for each descriptor-set and counter-mask pair.
extern const loom_amdgpu_wait_packet_selection_template_t
    loom_amdgpu_wait_packet_selections[][LOOM_AMDGPU_WAIT_COUNTER_MASK_ALL + 1];

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_PACKET_DATA_H_
