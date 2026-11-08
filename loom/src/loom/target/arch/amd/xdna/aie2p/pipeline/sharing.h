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

enum { LOOM_AIE2P_NATIVE_DELIVERY_PACKET_ID = 16 };

// A physical payload path eligible to carry control packets. Every geometry
// on this path grants a complete destination record before queuing its sender.
typedef struct loom_aie2p_native_delivery_t {
  // Uniform unicast path shared by all selected child geometries.
  loom_aie2p_native_distribution_path_t* path;
  // Independent task/reader continuations for the path's geometry variants.
  struct loom_aie2p_native_grant_t* grants;
  // Directed lanes and actual-use result owned by route admission.
  loom_aie2p_array_route_sharing_t sharing;
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
  // Compute receiver's control packet to the sender task register.
  uint8_t packet;
} loom_aie2p_native_grant_t;

// Selects a core packet ID outside its occupied set and the payload identity.
// When a delivery path is available, records the terminal before another core
// can assign that ID. Returns UINT8_MAX when no compatible identity remains.
uint8_t loom_aie2p_native_delivery_packet(
    loom_aie2p_native_delivery_t* delivery, uint32_t occupied,
    const loom_aie2p_native_tile_t* tile, loom_xdna_stream_port_t port,
    uint8_t channel);

// Reserves a scalar collective ID in the core's outgoing occupied set. Selects
// high IDs first; control routing reserves any shared payload identity there.
// Returns UINT8_MAX without changing the set when no identity remains.
uint8_t loom_aie2p_native_collective_packet(uint32_t* occupied);

// Selects at most one eligible payload path per control destination from the
// retained physical distributions and exact receiver credit returns.
// No payload framing or task policy changes until a route actually borrows it.
iree_status_t loom_aie2p_native_prepare_sharing(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization);

// Commits framing and receiver-granted admission for actually borrowed paths.
// Receiver emission consumes the channel's retained grant.
void loom_aie2p_native_commit_sharing(loom_aie2p_native_context_t* context);

// Emits the resident receiver's next-record task submission helper. The caller
// invokes it only after returning credit and before the last record retires.
iree_status_t loom_aie2p_native_prepare_grant(
    loom_aie2p_native_context_t* context,
    const loom_aie2p_native_grant_t* grant, loom_symbol_ref_t* out_symbol);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_SHARING_H_
