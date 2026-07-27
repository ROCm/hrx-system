// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU logical wait-counter identifiers used by descriptor overlays.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_COUNTERS_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_COUNTERS_H_

#include "iree/base/api.h"

// AMDGPU target counter ids used by the current descriptor overlays.
enum loom_amdgpu_wait_counter_e {
  // Descriptor did not name a concrete AMDGPU wait counter.
  LOOM_AMDGPU_WAIT_COUNTER_NONE = 0,
  // VMEM/global load-result dependency counter.
  LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD = 1,
  // VMEM/global store completion counter.
  LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE = 2,
  // LDS/DS dependency and completion counter.
  LOOM_AMDGPU_WAIT_COUNTER_LDS = 3,
  // Scalar-memory dependency counter.
  LOOM_AMDGPU_WAIT_COUNTER_SMEM = 4,
  // ALU dependency counter used by depctr-style wait packets.
  LOOM_AMDGPU_WAIT_COUNTER_ALU = 5,
  // Gfx125x tensor-memory transfer completion counter.
  LOOM_AMDGPU_WAIT_COUNTER_TENSOR = 6,
  // Gfx125x asynchronous cluster/global transfer completion counter.
  LOOM_AMDGPU_WAIT_COUNTER_ASYNC = 7,
  // Gfx125x memory-source translation lifetime counter.
  LOOM_AMDGPU_WAIT_COUNTER_X = 8,
};

// Number of concrete AMDGPU wait-counter slots. Counter ids are one-based, so
// slot ids map to counter ids by adding one.
#define LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT LOOM_AMDGPU_WAIT_COUNTER_X

// Bit masks for AMDGPU wait counters. These are descriptor-overlay ids, not
// native instruction bit encodings.
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD ((uint32_t)1u << 0)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE ((uint32_t)1u << 1)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS ((uint32_t)1u << 2)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM ((uint32_t)1u << 3)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_ALU ((uint32_t)1u << 4)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_TENSOR ((uint32_t)1u << 5)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_ASYNC ((uint32_t)1u << 6)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_X ((uint32_t)1u << 7)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM   \
  (LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD | \
   LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_READ                                  \
  (LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD |                                \
   LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS | LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM | \
   LOOM_AMDGPU_WAIT_COUNTER_MASK_TENSOR | LOOM_AMDGPU_WAIT_COUNTER_MASK_ASYNC)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_WRITE                                 \
  (LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE |                               \
   LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS | LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM | \
   LOOM_AMDGPU_WAIT_COUNTER_MASK_TENSOR | LOOM_AMDGPU_WAIT_COUNTER_MASK_ASYNC)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_WORKGROUP \
  LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_MEMORY \
  (LOOM_AMDGPU_WAIT_COUNTER_MASK_READ | LOOM_AMDGPU_WAIT_COUNTER_MASK_WRITE)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_ALL                                     \
  (LOOM_AMDGPU_WAIT_COUNTER_MASK_MEMORY | LOOM_AMDGPU_WAIT_COUNTER_MASK_ALU | \
   LOOM_AMDGPU_WAIT_COUNTER_MASK_X)

// Returns true when |counter_id| names a concrete AMDGPU wait counter.
static inline bool loom_amdgpu_wait_counter_id_is_valid(uint16_t counter_id) {
  return counter_id >= LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD &&
         counter_id <= LOOM_AMDGPU_WAIT_COUNTER_X;
}

// Converts a concrete one-based counter id into its dense zero-based slot.
static inline uint32_t loom_amdgpu_wait_counter_slot_from_id(
    uint16_t counter_id) {
  IREE_ASSERT(loom_amdgpu_wait_counter_id_is_valid(counter_id));
  return (uint32_t)(counter_id - 1);
}

// Converts a concrete one-based counter id into its logical counter mask.
static inline uint32_t loom_amdgpu_wait_counter_mask(uint16_t counter_id) {
  return (uint32_t)1u << loom_amdgpu_wait_counter_slot_from_id(counter_id);
}

// Converts a dense zero-based counter slot into its concrete one-based id.
static inline uint16_t loom_amdgpu_wait_counter_id_from_slot(uint32_t slot) {
  IREE_ASSERT_LT(slot, LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT);
  return (uint16_t)(slot + 1);
}

// Converts a dense zero-based counter slot into its logical counter mask.
static inline uint32_t loom_amdgpu_wait_counter_mask_from_slot(uint32_t slot) {
  IREE_ASSERT_LT(slot, LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT);
  return (uint32_t)1u << slot;
}

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_COUNTERS_H_
