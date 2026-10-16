// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU retained synchronization packets and barrier emission contracts.
// Selection retains descriptor and shape decisions for emission.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_SYNCHRONIZATION_PLAN_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_SYNCHRONIZATION_PLAN_H_

#include <stdint.h>

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_AMDGPU_EXPLICIT_PACKET_IMMEDIATE_CAPACITY 4

typedef struct loom_amdgpu_explicit_packet_immediate_t {
  // Module string ID for the immediate field name.
  loom_string_id_t name_id;
  // Concrete immediate value emitted for the packet.
  uint16_t value;
} loom_amdgpu_explicit_packet_immediate_t;

typedef struct loom_amdgpu_explicit_packet_immediate_template_t {
  // Borrowed immediate field name resolved during packet planning.
  iree_string_view_t name;
  // Concrete immediate value emitted for the packet.
  uint16_t value;
} loom_amdgpu_explicit_packet_immediate_template_t;

typedef struct loom_amdgpu_explicit_packet_plan_t {
  // Descriptor row selected for the explicit packet.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Immediate rows emitted on the descriptor.
  loom_amdgpu_explicit_packet_immediate_t
      immediates[LOOM_AMDGPU_EXPLICIT_PACKET_IMMEDIATE_CAPACITY];
  // Number of populated immediate rows.
  iree_host_size_t immediate_count;
} loom_amdgpu_explicit_packet_plan_t;

typedef enum loom_amdgpu_kernel_barrier_lowering_kind_e {
  // No lowering has been selected.
  LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_NONE = 0,
  // Emit a full workgroup barrier packet.
  LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_S_BARRIER = 1,
  // Emit a memory-ordering wait that drains LDS effects without rendezvous.
  LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_LDS_WAIT = 2,
  // Emit the split signal/wait barrier packet pair used by GFX12+ targets.
  LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_SPLIT_BARRIER = 3,
  // Converged wave execution supplies the rendezvous; memory ordering is
  // separate.
  LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_CONVERGED_SUBGROUP = 4,
  // Emit the arrival packet for an authored split-barrier phase.
  LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_SPLIT_ARRIVE = 5,
  // Emit the wait packet for an authored split-barrier phase.
  LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_SPLIT_WAIT = 6,
} loom_amdgpu_kernel_barrier_lowering_kind_t;

typedef struct loom_amdgpu_kernel_barrier_plan_t {
  // Concrete synchronization packet path selected for a kernel barrier op.
  loom_amdgpu_kernel_barrier_lowering_kind_t kind;
  // Explicit wait packet selected when |kind| is LDS_WAIT.
  loom_amdgpu_explicit_packet_plan_t wait;
  // Explicit signal packet selected for SPLIT_BARRIER or SPLIT_ARRIVE.
  loom_amdgpu_explicit_packet_plan_t split_signal;
  // Explicit wait packet selected for SPLIT_BARRIER or SPLIT_WAIT.
  loom_amdgpu_explicit_packet_plan_t split_wait;
} loom_amdgpu_kernel_barrier_plan_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SYNCHRONIZATION_PLAN_H_
