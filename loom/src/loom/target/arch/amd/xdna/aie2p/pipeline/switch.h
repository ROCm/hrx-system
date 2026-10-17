// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_SWITCH_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_SWITCH_H_

#include "loom/target/arch/amd/xdna/aie2p/array/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

struct loom_aie2p_native_context_t;

// One physical route edge carrying the packet identity used during admission.
typedef struct loom_aie2p_native_route_t {
  // Corresponding edge in the canonical route builder.
  const loom_aie2p_array_route_plan_t* edge;
  // Packet ID, or UINT8_MAX for an unframed circuit.
  uint8_t packet;
} loom_aie2p_native_route_t;

typedef enum loom_aie2p_native_switch_bank_e {
  LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS = 0,
  LOOM_AIE2P_NATIVE_SWITCH_BANK_SLAVES,
  LOOM_AIE2P_NATIVE_SWITCH_BANK_FILTERS,
  LOOM_AIE2P_NATIVE_SWITCH_BANK_COUNT,
} loom_aie2p_native_switch_bank_t;

// Complete valid register bank, with zero words for every disabled entry.
typedef struct loom_aie2p_native_register_bank_t {
  // Absolute address in the native transaction's 32-bit register space.
  uint32_t address;
  // Number of contiguous register words, excluding reserved address holes.
  uint32_t word_count;
  // Desired values retained by route admission in the pass arena.
  uint32_t* words;
} loom_aie2p_native_register_bank_t;

// A native pipeline owns route selection on each traversed switch. Its first
// invocation replaces complete banks after prior work has drained; externally
// prefixed routes are not an undeclared input to this materialization.
typedef struct loom_aie2p_native_switch_t {
  // Next routed switch in first-use order, excluding untouched tiles.
  struct loom_aie2p_native_switch_t* next;
  // Master, slave, and packet-filter banks from the generated register facts.
  loom_aie2p_native_register_bank_t banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_COUNT];
} loom_aie2p_native_switch_t;

// Admits physical route edges against packet/circuit ownership and switch
// resources, retaining the complete initialization banks in the pass arena.
// Allocation errors return a status; route conflicts emit the pipeline's
// target-support diagnostic and leave the caller's false |out_valid| intact.
iree_status_t loom_aie2p_native_switches_build(
    struct loom_aie2p_native_context_t* context, const loom_op_t* entry,
    bool* out_valid);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_SWITCH_H_
