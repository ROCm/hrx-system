// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/dma_resources.h"

// Callers compare transfers sharing one admitted physical resource owner.
// Common execution proves distinct worker occurrences have disjoint lifetimes,
// even when repeated calls share source operations. Within one occurrence,
// async legality retires every group in its block, and the retained completion
// edge determines when that block can reuse the resource.
bool loom_aie2p_native_transfer_retired(
    const loom_aie2p_native_transfer_t* transfer, iree_host_size_t worker_index,
    const loom_op_t* issue) {
  return transfer->worker_index != worker_index ||
         transfer->completion->parent_block != issue->parent_block ||
         transfer->completion->block_ordinal <= issue->block_ordinal;
}

void loom_aie2p_native_task_descriptor_words(
    const loom_aie2p_native_transfer_task_t* task,
    const loom_aie2p_native_dma_descriptor_t* descriptor,
    const loom_aie2p_native_transfer_t* transfer, uint32_t out_words[8]) {
  memcpy(out_words,
         descriptor->shim ? transfer->shim_words : transfer->local_words,
         sizeof(transfer->local_words));
  if (transfer->descriptor_ordinal + 1 == task->transfer_count) {
    return;
  }
  const loom_aie2p_native_memory_layout_t* layout =
      descriptor->shim ? NULL : task->path->local->memory_layout;
  const uint8_t word = layout ? layout->chain.word : 7;
  const loom_xdna_register_field_id_t next =
      layout ? layout->chain.next
             : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD7_NEXT_BD;
  const loom_xdna_register_field_id_t enable =
      layout ? layout->chain.enable
             : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD7_USE_NEXT_BD;
  out_words[word] |= loom_xdna_register_field_encode(
                         next, descriptor->sequence.descriptor +
                                   transfer->descriptor_ordinal + 1) |
                     loom_xdna_register_field_encode(enable, 1);
}

static bool loom_aie2p_native_task_descriptor_matches(
    const loom_aie2p_native_transfer_task_t* task,
    const loom_aie2p_native_dma_descriptor_t* descriptor, uint32_t address_mask,
    bool* out_address_differs) {
  *out_address_differs = false;
  if (descriptor->sequence.descriptor_count != task->transfer_count) {
    return false;
  }
  for (const loom_aie2p_native_transfer_t* transfer = task->first_transfer;
       transfer; transfer = transfer->task_next) {
    uint32_t words[8];
    loom_aie2p_native_task_descriptor_words(task, descriptor, transfer, words);
    const uint32_t* previous =
        descriptor->sequence.words[transfer->descriptor_ordinal];
    if (address_mask) {
      const uint8_t word = task->path->local->memory_layout->address_word;
      const uint32_t difference = (words[word] ^ previous[word]) & address_mask;
      *out_address_differs |= difference != 0;
      words[word] ^= difference;
    }
    if (memcmp(previous, words, sizeof(words)) != 0) {
      return false;
    }
  }
  return true;
}

static bool loom_aie2p_native_task_external_address_differs(
    const loom_aie2p_native_transfer_task_t* task,
    const loom_aie2p_native_transfer_t* previous) {
  const loom_aie2p_native_transfer_t* transfer = task->first_transfer;
  for (; transfer && previous;
       transfer = transfer->task_next, previous = previous->task_next) {
    if (transfer->binding != previous->binding ||
        transfer->external_view->begin_byte_offset.constant !=
            previous->external_view->begin_byte_offset.constant) {
      return true;
    }
  }
  return transfer || previous;
}

// Each task endpoint has an immutable descriptor sequence and latest retained
// completion. Local and shim sequences can be reused independently.
loom_aie2p_native_dma_descriptor_t* loom_aie2p_native_reusable_descriptor(
    const loom_aie2p_native_worker_t* worker,
    const loom_aie2p_native_transfer_task_t* task, bool shim) {
  if (worker->execution == LOOM_AIE2P_NATIVE_EXECUTION_DMA) {
    // Autonomous descriptors retain iteration state until invocation
    // retirement.
    return NULL;
  }
  const loom_op_t* issue = task->first_transfer->source->request.op;
  const iree_host_size_t worker_index = task->first_transfer->worker_index;
  uint32_t address_mask = 0;
  if (!shim) {
    const loom_xdna_register_field_info_t address =
        loom_xdna_register_field_info(
            task->path->local->memory_layout->address);
    address_mask = (UINT32_MAX >> (32 - address.bit_width))
                   << address.least_significant_bit;
  }
  for (loom_aie2p_native_dma_descriptor_t* candidate = worker->descriptors;
       candidate; candidate = candidate->next) {
    const loom_aie2p_native_transfer_t* previous = candidate->last_transfer;
    bool address_differs;
    // Shim descriptors belong to the tile's shared bank. Direction and
    // completion routing belong to the issuing channel, so a retired sequence
    // can serve another path on that shim. Local descriptors retain their
    // path-specific completion semaphore and address aperture.
    const bool same_bank = shim ? candidate->path->shim == task->path->shim
                                : candidate->path == task->path;
    if (candidate->shim != shim || !same_bank ||
        !loom_aie2p_native_transfer_retired(previous, worker_index, issue) ||
        (candidate->completion &&
         !loom_aie2p_native_transfer_retired(
             candidate->completion->last_transfer, worker_index, issue)) ||
        !loom_aie2p_native_task_descriptor_matches(
            task, candidate, address_mask, &address_differs)) {
      continue;
    }
    if (shim) {
      candidate->address_dynamic |=
          task->shim_address_dynamic ||
          loom_aie2p_native_task_external_address_differs(task, previous);
    } else if (task->local_address_dynamic || address_differs) {
      candidate->address_dynamic = true;
      // All users patch their own address after the preceding task retires.
      // Keep length, dimensions, and chain links in the shared template.
      const uint8_t word = task->path->local->memory_layout->address_word;
      for (uint16_t i = 0; i < candidate->sequence.descriptor_count; ++i) {
        candidate->sequence.words[i][word] &= ~address_mask;
      }
    }
    return candidate;
  }
  return NULL;
}

// Both core and configuration waits consume arrival credits, so a retired
// credit can serve another descriptor geometry on the same tile. The search is
// bounded by the physical lock count.
loom_aie2p_native_dma_completion_t* loom_aie2p_native_reusable_completion(
    const loom_aie2p_native_worker_t* worker,
    const loom_aie2p_native_tile_t* tile,
    const loom_aie2p_native_transfer_t* next) {
  if (worker->execution == LOOM_AIE2P_NATIVE_EXECUTION_DMA) {
    return NULL;
  }
  for (loom_aie2p_native_dma_completion_t* completion = worker->dma_completions;
       completion; completion = completion->next) {
    if (completion->tile == tile &&
        loom_aie2p_native_transfer_retired(completion->last_transfer,
                                           next->worker_index,
                                           next->source->request.op)) {
      return completion;
    }
  }
  return NULL;
}
