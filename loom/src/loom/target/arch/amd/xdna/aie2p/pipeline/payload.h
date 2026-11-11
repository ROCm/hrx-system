// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_PAYLOAD_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_PAYLOAD_H_

#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_aie2p_native_payload_endpoint_t {
  // Memory module owning the direction-specific DMA queue.
  loom_aie2p_native_tile_t* tile;
  // Assigned physical queue in that module.
  uint8_t engine;
} loom_aie2p_native_payload_endpoint_t;

// One immutable destination set on a physical DMA sender.
typedef struct loom_aie2p_native_payload_class_t {
  // Next class in stable first-use order.
  struct loom_aie2p_native_payload_class_t* next;
  // Physical sender whose framing and packet namespace this class uses.
  struct loom_aie2p_native_payload_source_t* source;
  // Canonically ordered destinations; order does not change broadcast
  // semantics.
  loom_aie2p_native_payload_endpoint_t* destinations;
  // Number of distinct receiver queues in that set.
  iree_host_size_t destination_count;
  // Packet identity in the physical sender's namespace.
  uint8_t packet;
  // First edge in the stable route table, assigned on first realization.
  iree_host_size_t route_first;
  // Number of edges retained for this destination class.
  iree_host_size_t route_count;
  // Physical routes have been realized for this class.
  bool routed;
  // Every use is a distribution with a receiver-owned admission protocol.
  bool distribution_only;
  // Distinct selected distribution paths using this physical destination set.
  uint32_t distribution_path_count;
} loom_aie2p_native_payload_class_t;

// All occurrence-owned uses of one physical memory-to-stream queue.
typedef struct loom_aie2p_native_payload_source_t {
  // Next physical sender in stable first-use order.
  struct loom_aie2p_native_payload_source_t* next;
  // Physical sender selected by common lifetime allocation.
  loom_aie2p_native_payload_endpoint_t endpoint;
  // Distinct destination sets in this sender's packet namespace.
  loom_aie2p_native_payload_class_t* classes;
  // Number of destination sets, bounded by the hardware packet ID space.
  uint8_t class_count;
  // Canonical routing tree identity, UINT32_MAX before first realization.
  uint32_t route;
  // Packet framing is required by any source use or receiver fan-in.
  bool framed;
} loom_aie2p_native_payload_source_t;

// Interns physical source/destination identities from formed native demand.
// Queue assignment precedes this operation; no source IR is traversed.
iree_status_t loom_aie2p_native_form_payloads(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, bool* out_valid);

// Realizes a class once, sharing the physical sender's route tree across all
// transport families and preserving its packet identity along every edge.
iree_status_t loom_aie2p_native_route_payload(
    loom_aie2p_native_context_t* context,
    loom_aie2p_native_payload_class_t* payload, const loom_op_t* op,
    uint32_t* next_source, bool* out_valid);

// Applies sender framing to descriptor words before descriptor interning.
void loom_aie2p_native_frame_payload(
    const loom_aie2p_native_payload_class_t* payload, uint32_t words[8]);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_PAYLOAD_H_
