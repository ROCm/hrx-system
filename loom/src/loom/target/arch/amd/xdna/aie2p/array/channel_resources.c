// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/array/channel_resources.h"

#include <string.h>

static void loom_aie2p_array_channel_resources_initialize_probe(
    const loom_aie2p_array_tile_resources_t* resources,
    uint32_t* probe_bank_cursors,
    loom_aie2p_array_tile_resources_t* out_probe) {
  *out_probe = *resources;
  memcpy(probe_bank_cursors, resources->bank_cursors,
         resources->facts->memory.bank_count * sizeof(*probe_bank_cursors));
  out_probe->bank_cursors = probe_bank_cursors;
}

static loom_aie2p_array_channel_resource_failure_t
loom_aie2p_array_channel_resources_propose_ring_from_probe(
    loom_aie2p_array_tile_resources_t* probe,
    loom_xdna_tile_coordinate_t coordinate,
    const loom_xdna_address_window_t* load_window, uint8_t record_count,
    uint32_t record_byte_length,
    loom_aie2p_array_ring_resource_proposal_t* out_proposal) {
  loom_aie2p_array_ring_resource_proposal_t proposal = {
      .coordinate = coordinate,
      .load_window = load_window,
      .record_count = record_count,
  };
  for (uint16_t record = 0; record < record_count; ++record) {
    loom_aie2p_array_local_memory_proposal_t storage_proposal;
    if (!loom_aie2p_array_local_memory_propose_channel(
            probe->facts, probe->bank_cursors, probe->next_bank,
            record_byte_length, LOOM_AIE2P_ARRAY_CHANNEL_STORAGE_ALIGNMENT,
            &storage_proposal)) {
      return LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_RING_STORAGE;
    }
    loom_aie2p_array_local_memory_commit(probe->facts, &storage_proposal,
                                         probe->bank_cursors,
                                         &probe->next_bank);
    proposal.owner_offsets[record] = storage_proposal.owner_offset;
  }
  if ((uint16_t)probe->next_lock + 2u > probe->facts->lock_count) {
    return LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_LOCK_PAIR;
  }
  proposal.credit_lock = probe->next_lock++;
  proposal.ready_lock = probe->next_lock++;
  proposal.post_next_lock = probe->next_lock;
  proposal.post_next_bank = probe->next_bank;
  memcpy(proposal.post_bank_cursors, probe->bank_cursors,
         probe->facts->memory.bank_count * sizeof(*proposal.post_bank_cursors));
  *out_proposal = proposal;
  return LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_NONE;
}

loom_aie2p_array_channel_resource_failure_t
loom_aie2p_array_channel_resources_propose_neighbor(
    const loom_aie2p_array_tile_resources_t* resources,
    loom_xdna_tile_coordinate_t coordinate,
    const loom_xdna_address_window_t* load_window, uint8_t record_count,
    uint32_t record_byte_length,
    loom_aie2p_array_ring_resource_proposal_t* out_proposal) {
  uint32_t probe_bank_cursors[LOOM_AIE2P_ARRAY_CHANNEL_PROPOSAL_MAX_BANK_COUNT];
  loom_aie2p_array_tile_resources_t probe;
  loom_aie2p_array_channel_resources_initialize_probe(
      resources, probe_bank_cursors, &probe);
  return loom_aie2p_array_channel_resources_propose_ring_from_probe(
      &probe, coordinate, load_window, record_count, record_byte_length,
      out_proposal);
}

loom_aie2p_array_channel_resource_failure_t
loom_aie2p_array_channel_resources_propose_compute(
    const loom_aie2p_array_tile_resources_t* resources,
    const loom_aie2p_array_compute_endpoint_proposal_t* predecessor,
    const loom_aie2p_array_compute_endpoint_request_t* request,
    loom_aie2p_array_compute_endpoint_proposal_t* out_proposal) {
  uint32_t probe_bank_cursors[LOOM_AIE2P_ARRAY_CHANNEL_PROPOSAL_MAX_BANK_COUNT];
  loom_aie2p_array_tile_resources_t probe;
  loom_aie2p_array_channel_resources_initialize_probe(
      resources, probe_bank_cursors, &probe);
  if (predecessor != NULL) {
    loom_aie2p_array_channel_resources_commit_compute(predecessor, &probe);
  }

  uint8_t* next_channel =
      request->direction == LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM
          ? &probe.next_memory_to_stream_channel
          : &probe.next_stream_to_memory_channel;
  if (iree_any_bit_set(
          request->flags,
          LOOM_AIE2P_ARRAY_COMPUTE_ENDPOINT_REQUEST_FLAG_REQUIRE_LOOPBACK) &&
      (request->loopback_source_dma_channel >=
           probe.facts->dma.loopback_channel_count ||
       request->loopback_source_dma_channel != *next_channel)) {
    return LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_DMA_LOOPBACK;
  }
  if (*next_channel >= probe.facts->dma.channel_count_per_direction) {
    return LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_DMA_CHANNEL;
  }
  if ((uint32_t)probe.next_buffer_descriptor + request->record_count >
      probe.facts->dma.buffer_descriptor_count) {
    return LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_DMA_DESCRIPTORS;
  }

  loom_aie2p_array_compute_endpoint_proposal_t proposal = {
      .direction = request->direction,
      .dma_channel = (*next_channel)++,
      .buffer_descriptor_start = probe.next_buffer_descriptor,
      .buffer_descriptor_count = request->record_count,
  };
  probe.next_buffer_descriptor =
      (uint16_t)(probe.next_buffer_descriptor + request->record_count);
  if (!iree_any_bit_set(probe.flags,
                        LOOM_AIE2P_ARRAY_TILE_RESOURCE_FLAG_HAS_WORKER) &&
      !iree_any_bit_set(probe.flags,
                        LOOM_AIE2P_ARRAY_TILE_RESOURCE_FLAG_HAS_DMA_SERVICE)) {
    proposal.flags =
        LOOM_AIE2P_ARRAY_COMPUTE_ENDPOINT_PROPOSAL_FLAG_STARTS_DMA_SERVICE;
    probe.flags |= LOOM_AIE2P_ARRAY_TILE_RESOURCE_FLAG_HAS_DMA_SERVICE;
  }

  const loom_aie2p_array_channel_resource_failure_t failure =
      loom_aie2p_array_channel_resources_propose_ring_from_probe(
          &probe, request->coordinate, request->load_window,
          request->record_count, request->record_byte_length, &proposal.ring);
  if (failure != LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_NONE) {
    return failure;
  }
  *out_proposal = proposal;
  return LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_NONE;
}

loom_aie2p_array_channel_resource_failure_t
loom_aie2p_array_channel_resources_propose_shim(
    const loom_aie2p_array_tile_resources_t* resources,
    loom_xdna_tile_coordinate_t coordinate, loom_xdna_dma_direction_t direction,
    uint16_t descriptor_count,
    loom_aie2p_array_shim_endpoint_proposal_t* out_proposal) {
  const uint8_t next_channel =
      direction == LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM
          ? resources->next_memory_to_stream_channel
          : resources->next_stream_to_memory_channel;
  if (next_channel >= resources->facts->dma.channel_count_per_direction) {
    return LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_DMA_CHANNEL;
  }
  if ((uint32_t)resources->next_buffer_descriptor + descriptor_count >
      resources->facts->dma.buffer_descriptor_count) {
    return LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_DMA_DESCRIPTORS;
  }
  *out_proposal = (loom_aie2p_array_shim_endpoint_proposal_t){
      .coordinate = coordinate,
      .direction = direction,
      .dma_channel = next_channel,
      .buffer_descriptor_start = resources->next_buffer_descriptor,
      .buffer_descriptor_count = descriptor_count,
  };
  return LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_NONE;
}

void loom_aie2p_array_channel_resources_commit_ring(
    const loom_aie2p_array_ring_resource_proposal_t* proposal,
    loom_aie2p_array_tile_resources_t* resources) {
  memcpy(
      resources->bank_cursors, proposal->post_bank_cursors,
      resources->facts->memory.bank_count * sizeof(*resources->bank_cursors));
  resources->next_bank = proposal->post_next_bank;
  resources->next_lock = proposal->post_next_lock;
}

void loom_aie2p_array_channel_resources_commit_compute(
    const loom_aie2p_array_compute_endpoint_proposal_t* proposal,
    loom_aie2p_array_tile_resources_t* resources) {
  uint8_t* next_channel =
      proposal->direction == LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM
          ? &resources->next_memory_to_stream_channel
          : &resources->next_stream_to_memory_channel;
  *next_channel = proposal->dma_channel + 1u;
  resources->next_buffer_descriptor =
      proposal->buffer_descriptor_start + proposal->buffer_descriptor_count;
  if (iree_any_bit_set(
          proposal->flags,
          LOOM_AIE2P_ARRAY_COMPUTE_ENDPOINT_PROPOSAL_FLAG_STARTS_DMA_SERVICE)) {
    resources->flags |= LOOM_AIE2P_ARRAY_TILE_RESOURCE_FLAG_HAS_DMA_SERVICE;
  }
  loom_aie2p_array_channel_resources_commit_ring(&proposal->ring, resources);
}

void loom_aie2p_array_channel_resources_commit_shim(
    const loom_aie2p_array_shim_endpoint_proposal_t* proposal,
    loom_aie2p_array_tile_resources_t* resources) {
  uint8_t* next_channel =
      proposal->direction == LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM
          ? &resources->next_memory_to_stream_channel
          : &resources->next_stream_to_memory_channel;
  *next_channel = proposal->dma_channel + 1u;
  resources->next_buffer_descriptor =
      proposal->buffer_descriptor_start + proposal->buffer_descriptor_count;
}
