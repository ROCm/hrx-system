// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_WORKER_UPLOAD_PLAN_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_WORKER_UPLOAD_PLAN_H_

#include "loom/target/arch/amd/xdna/aie2p/emit/program_broadcast.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/worker_upload.h"

// Simultaneous circuits. All destination credits and sending queues retire
// before a subsequent wave can replace the switch state.
typedef struct loom_aie2p_worker_wave_t {
  // Source, local receivers, and physical connectivity intent for each stream.
  loom_aie2p_native_connections_t connections;
  // Instruction writes in sender order.
  const loom_aie2p_worker_code_stream_t* streams;
  // Audited switch states establishing exactly these circuits.
  const loom_aie2p_native_switch_t* switches;
  // Destination semaphore value written by the final in-band packet.
  uint32_t credit;
} loom_aie2p_worker_wave_t;

typedef struct loom_aie2p_worker_upload_plan_t {
  // Entry-local indices of linked root sections delivered by the plan.
  bool* workers;
  // Physical shim columns owning a source engine borrowed during startup.
  bool* senders;
  // Whole-row broadcasts followed by individual correction waves.
  loom_aie2p_worker_wave_t* waves;
  // Number of admitted waves; zero selects inline initialization.
  iree_host_size_t wave_count;
  // Number of static source-address fixups across all waves.
  iree_host_size_t transfer_count;
  // Bound on native records before removing redundant switch writes.
  iree_host_size_t record_capacity;
  // Command bytes removed when the selected tile loads deliver their code by
  // DMA.
  iree_host_size_t inline_size;
} loom_aie2p_worker_upload_plan_t;

iree_status_t loom_aie2p_worker_upload_plan(
    const loom_aie2p_execution_plan_t* plan,
    const loom_aie2p_xdna_tile_t* tiles, iree_host_size_t tile_count,
    iree_arena_allocator_t* arena, loom_aie2p_worker_upload_plan_t* out_plan);

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_WORKER_UPLOAD_PLAN_H_
