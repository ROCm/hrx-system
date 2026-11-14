// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/payload.h"
#include "loom/util/adaptive_sort.h"

static uint64_t loom_aie2p_native_connection_key(
    const loom_aie2p_native_control_connection_t* connection) {
  const loom_aie2p_native_connection_endpoint_t source = connection->source;
  return (uint64_t)source.coordinate.column << 48 |
         (uint64_t)source.coordinate.row << 32 | (uint64_t)source.port << 16 |
         (uint64_t)source.channel << 8 | connection->packet;
}

static bool loom_aie2p_native_connection_less(
    const loom_aie2p_native_control_connection_t* connections,
    const iree_host_size_t* lhs, const iree_host_size_t* rhs) {
  const uint64_t lhs_key = loom_aie2p_native_connection_key(&connections[*lhs]);
  const uint64_t rhs_key = loom_aie2p_native_connection_key(&connections[*rhs]);
  return lhs_key != rhs_key ? lhs_key < rhs_key : *lhs < *rhs;
}

LOOM_DEFINE_ADAPTIVE_SORT_WITH_CONTEXT(
    loom_aie2p_native_connections_sort, iree_host_size_t,
    const loom_aie2p_native_control_connection_t*,
    loom_aie2p_native_connection_less)

iree_status_t loom_aie2p_native_connections_build(
    loom_aie2p_native_context_t* context) {
  iree_arena_allocator_t* arena = context->pass->arena;
  iree_host_size_t group_capacity = context->control_connection_count;
  iree_host_size_t destination_capacity = context->control_connection_count;
  for (const loom_aie2p_native_payload_source_t* source =
           context->payload_sources;
       source; source = source->next) {
    for (const loom_aie2p_native_payload_class_t* payload = source->classes;
         payload; payload = payload->next) {
      if (payload->routed) {
        ++group_capacity;
        destination_capacity += payload->destination_count;
      }
    }
  }
  loom_aie2p_native_connection_t* groups = NULL;
  loom_aie2p_native_connection_endpoint_t* destinations = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, group_capacity, sizeof(*groups), (void**)&groups));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, destination_capacity,
                                                 sizeof(*destinations),
                                                 (void**)&destinations));
  iree_host_size_t group_count = 0;
  iree_host_size_t destination_count = 0;
  for (const loom_aie2p_native_payload_source_t* source =
           context->payload_sources;
       source; source = source->next) {
    for (const loom_aie2p_native_payload_class_t* payload = source->classes;
         payload; payload = payload->next) {
      if (!payload->routed) {
        continue;
      }
      groups[group_count++] = (loom_aie2p_native_connection_t){
          .source = {source->endpoint.tile->coordinate,
                     LOOM_XDNA_STREAM_PORT_DMA, source->endpoint.engine},
          .packet = source->framed ? payload->packet : UINT8_MAX,
          .retain_header = false,
          .destination_offset = destination_count,
          .destination_count = payload->destination_count,
      };
      for (iree_host_size_t i = 0; i < payload->destination_count; ++i) {
        destinations[destination_count++] =
            (loom_aie2p_native_connection_endpoint_t){
                payload->destinations[i].tile->coordinate,
                LOOM_XDNA_STREAM_PORT_DMA, payload->destinations[i].engine};
      }
    }
  }

  // Control requests retain their producer's endpoint identities. Sorting a
  // sparse projection groups multicast requests without an array-wide table
  // for all possible source/packet pairs or any inspection of selected routes.
  const iree_arena_checkpoint_t checkpoint = iree_arena_checkpoint_save(arena);
  iree_host_size_t* order = NULL;
  iree_status_t status = iree_arena_allocate_array(
      arena, context->control_connection_count, sizeof(*order), (void**)&order);
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < context->control_connection_count; ++i) {
      order[i] = i;
    }
    loom_aie2p_native_connections_sort(context->control_connections, order,
                                       context->control_connection_count);
    for (iree_host_size_t i = 0; i < context->control_connection_count;) {
      const loom_aie2p_native_control_connection_t* request =
          &context->control_connections[order[i]];
      const uint64_t key = loom_aie2p_native_connection_key(request);
      loom_aie2p_native_connection_t* group = &groups[group_count++];
      *group = (loom_aie2p_native_connection_t){
          .source = request->source,
          .packet = request->packet,
          .retain_header = true,
          .destination_offset = destination_count,
      };
      do {
        destinations[destination_count++] =
            context->control_connections[order[i++]].destination;
        ++group->destination_count;
      } while (i < context->control_connection_count &&
               loom_aie2p_native_connection_key(
                   &context->control_connections[order[i]]) == key);
    }
    context->connections = (loom_aie2p_native_connections_t){
        .groups = groups,
        .group_count = group_count,
        .destinations = destinations,
        .destination_count = destination_count,
    };
  }
  iree_arena_checkpoint_restore(&checkpoint);
  return status;
}
