// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_SHARING_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_SHARING_H_

#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

#ifdef __cplusplus
extern "C" {
#endif

// A physical payload path eligible to carry control packets. Every geometry
// on this path grants a complete destination record before queuing its sender.
typedef struct loom_aie2p_native_delivery_t {
  // Uniform physical destination and switch edges used by all granted phases.
  struct loom_aie2p_native_payload_class_t* payload;
  // Independent task/reader continuations for the path's geometry variants.
  struct loom_aie2p_native_grant_t* grants;
  // Directed lanes and actual-use result owned by route admission.
  loom_aie2p_array_route_sharing_t sharing;
  // Control packet IDs assigned to terminals on this granted path.
  uint32_t command_packets;
  // Terminal identity of each packet class crossing this shared path. Empty
  // entries have a NULL tile. Distinct core sources may reuse an ID only for
  // the same terminal, since their packets become indistinguishable on merge.
  struct {
    // Physical terminal tile, or NULL for an unassigned packet ID.
    const loom_aie2p_native_tile_t* tile;
    // Terminal stream port on that tile.
    loom_xdna_stream_port_t port;
    // Channel within the terminal port.
    uint8_t channel;
  } packets[32];
} loom_aie2p_native_delivery_t;

// One child's complete-record admission on a shared payload/control path.
typedef struct loom_aie2p_native_grant_t {
  // Next geometry's grant on the same physical path.
  struct loom_aie2p_native_grant_t* next;
  // Geometry retaining this child's exact receiver and repetition count.
  loom_aie2p_native_distribution_t* distribution;
  // Absolute sender task register, used by this receiver's continuation.
  uint32_t address;
  // One complete record task with no repeat, used for every receiver grant.
  uint32_t task;
  // Maximum granted/unreturned records, bounded by receiver slots and sender
  // task capacity so returns cannot fill the queue behind the startup burst.
  uint32_t window_capacity;
  // Compute receiver's control packet to the sender task register.
  uint8_t packet;
} loom_aie2p_native_grant_t;

// Selects a compatible core packet ID, excluding the reserved payload identity.
// When a delivery path is available, records the terminal before another core
// can assign that ID. A preferred ID is already bound to this endpoint in the
// caller's column namespace; it may be reused even when occupied, provided the
// delivery namespace agrees. UINT8_MAX requests a fresh ID. Returns UINT8_MAX
// when no compatible identity remains.
uint8_t loom_aie2p_native_delivery_packet(
    loom_aie2p_native_delivery_t* delivery, uint32_t occupied,
    const loom_aie2p_native_tile_t* tile, loom_xdna_stream_port_t port,
    uint8_t channel, uint8_t preferred);

// Reserves a scalar collective ID in the core's outgoing occupied set,
// excluding identities already carried by another source in its shared command
// column. Selects high IDs first; control routing reserves shared payload
// identities. Returns UINT8_MAX without changing the set when no identity
// remains.
uint8_t loom_aie2p_native_collective_packet(uint32_t* occupied,
                                            uint32_t shared);

// Selects at most one eligible payload path per control destination from the
// retained physical distributions and exact receiver credit returns.
// No payload framing or task policy changes until a route actually borrows it.
iree_status_t loom_aie2p_native_prepare_sharing(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization);

// Commits framing and receiver-granted admission for actually borrowed paths.
// Receiver emission consumes the channel's retained grant.
void loom_aie2p_native_commit_sharing(loom_aie2p_native_context_t* context);

// Retains a committed receiver-granted command path for a code loader whose
// SRAM column has no private command path. The retained scope owns all route
// and packet facts after native construction workers retire.
void loom_aie2p_native_retain_code_carriers(
    const loom_aie2p_native_context_t* context,
    loom_aie2p_execution_plan_t* plan);

// Emits the resident receiver's next-record task submission helper. The caller
// invokes it only after returning credit and before the last record retires.
iree_status_t loom_aie2p_native_prepare_grant(
    loom_aie2p_native_context_t* context,
    const loom_aie2p_native_grant_t* grant, loom_symbol_ref_t* out_symbol);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_SHARING_H_
