// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generated AMDGPU wait-packet descriptor tables and compact lookup helpers.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_PACKET_TABLES_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_PACKET_TABLES_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/ir.h"
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

typedef struct loom_amdgpu_wait_packet_target_t {
  // Generated descriptor template rows available on this target.
  const loom_amdgpu_wait_packet_descriptor_template_t* descriptors;
  // Number of descriptor template rows available on this target.
  iree_host_size_t descriptor_count;
  // Generated best descriptor-template rows indexed by logical counter mask.
  const loom_amdgpu_wait_packet_selection_template_t* selections;
  // Number of generated counter-mask selection rows.
  iree_host_size_t selection_count;
  // Maximum immediate template count for any available wait descriptor.
  iree_host_size_t max_descriptor_immediate_count;
} loom_amdgpu_wait_packet_target_t;

// Populates the generated wait-packet rows available on |descriptor_set|.
iree_status_t loom_amdgpu_wait_packet_analyze_target(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_wait_packet_target_t* out_target);

// Returns the immediate template row for |packet_descriptor|'s local immediate
// |immediate_index|.
const loom_amdgpu_wait_packet_descriptor_immediate_template_t*
loom_amdgpu_wait_packet_descriptor_immediate(
    const loom_amdgpu_wait_packet_descriptor_template_t* packet_descriptor,
    uint16_t immediate_index);

// Resolves a generated wait-packet descriptor template to the descriptor row in
// |descriptor_set|.
const loom_low_descriptor_t* loom_amdgpu_wait_packet_resolve_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_wait_packet_descriptor_template_t* packet_descriptor);

// Finds the generated wait-packet descriptor template for |descriptor|.
iree_status_t loom_amdgpu_wait_packet_find_descriptor_template(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    const loom_amdgpu_wait_packet_target_t* target,
    const loom_amdgpu_wait_packet_descriptor_template_t**
        out_packet_descriptor);

// Returns the logical counters actually drained by an explicit wait packet
// already present in scheduled low IR. Descriptor effects describe the counters
// the packet can encode; this helper interprets the packet's concrete
// immediate attributes and omits counters left at their no-wait value.
iree_status_t loom_amdgpu_wait_packet_explicit_counter_mask(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, const loom_module_t* module,
    const loom_op_t* op, uint32_t* out_counter_mask);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_PACKET_TABLES_H_
