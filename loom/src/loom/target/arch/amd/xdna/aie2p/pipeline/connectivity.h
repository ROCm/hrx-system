// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_CONNECTIVITY_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_CONNECTIVITY_H_

#include "iree/base/internal/arena.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/switch.h"

#ifdef __cplusplus
extern "C" {
#endif

// Physical endpoint identity, independent of selected route edges. DMA channels
// name engines, including on shim tiles; other ports name stream channels.
typedef struct loom_aie2p_native_connection_endpoint_t {
  // Tile containing the endpoint.
  loom_xdna_tile_coordinate_t coordinate;
  // Endpoint's architectural port class.
  loom_xdna_stream_port_t port;
  // Direction-specific channel within the port.
  uint8_t channel;
} loom_aie2p_native_connection_endpoint_t;

// One requested control delivery, retained before physical routing. Equal
// source/packet pairs form one multicast destination set.
typedef struct loom_aie2p_native_control_connection_t {
  // Physical sender of the framed control record.
  loom_aie2p_native_connection_endpoint_t source;
  // Intended consumer of that record, independent of chosen switch paths.
  loom_aie2p_native_connection_endpoint_t destination;
  // Five-bit packet identity within the sender's namespace.
  uint8_t packet;
} loom_aie2p_native_control_connection_t;

typedef enum loom_aie2p_native_connectivity_failure_e {
  LOOM_AIE2P_NATIVE_CONNECTIVITY_NONE = 0,
  LOOM_AIE2P_NATIVE_CONNECTIVITY_DEAD_END,
  LOOM_AIE2P_NATIVE_CONNECTIVITY_REPEATED_ARRIVAL,
  LOOM_AIE2P_NATIVE_CONNECTIVITY_UNEXPECTED_DESTINATION,
  LOOM_AIE2P_NATIVE_CONNECTIVITY_MISSING_DESTINATION,
  LOOM_AIE2P_NATIVE_CONNECTIVITY_HEADER,
  LOOM_AIE2P_NATIVE_CONNECTIVITY_ARRAY_BOUNDARY,
} loom_aie2p_native_connectivity_failure_t;

typedef struct loom_aie2p_native_connectivity_result_t {
  // First violated connection invariant, or NONE for exact delivery.
  loom_aie2p_native_connectivity_failure_t failure;
  // Arrival or destination at which the invariant failed.
  loom_aie2p_native_connection_endpoint_t endpoint;
} loom_aie2p_native_connectivity_result_t;

typedef struct loom_aie2p_native_connectivity_t
    loom_aie2p_native_connectivity_t;

// Allocates an independent physical-network view from completed register banks
// and admitted shim mux connections. All storage belongs to |arena|. The audit
// is separate from route selection and admission. It does not inspect source
// IR, route-tree identities, DMA descriptors, credit lifetimes, or execution
// time.
iree_status_t loom_aie2p_native_connectivity_allocate(
    const loom_xdna_array_family_t* family,
    const loom_aie2p_native_switch_t* switches,
    const loom_aie2p_array_route_plan_t* routes, iree_host_size_t route_count,
    iree_arena_allocator_t* arena,
    loom_aie2p_native_connectivity_t** out_connectivity);

// Traces one sender and compares the exact delivered endpoint set with retained
// intent. Packet UINT8_MAX means an unframed source. Controls preserve headers;
// payload destinations remove them. Repeated arrivals reject both route cycles
// and duplicate multicast delivery. Success proves connectivity, not progress.
loom_aie2p_native_connectivity_result_t loom_aie2p_native_connectivity_check(
    loom_aie2p_native_connectivity_t* connectivity,
    loom_aie2p_native_connection_endpoint_t source, uint8_t packet,
    const loom_aie2p_native_connection_endpoint_t* destinations,
    iree_host_size_t destination_count, bool retain_header);

// Audits every completed native transport before configuration emission.
// Temporary scratch is released before returning. Allocation can fail; a
// connection mismatch terminates as a compiler invariant failure, never as
// unsupported user IR or a recoverable route-selection result.
iree_status_t loom_aie2p_native_connections_verify(
    const struct loom_aie2p_native_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_CONNECTIVITY_H_
