// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/semaphores.h"

#include "loom/analysis/pipeline/lifetime_layout.h"
#include "loom/util/adaptive_sort.h"

typedef struct loom_aie2p_semaphore_requirement_t {
  // Channel in the retained construction inventory.
  iree_host_size_t channel;
  // Physical tile whose semaphore bank owns this pair.
  loom_aie2p_native_tile_t* tile;
  // Free credits at a fresh construction boundary; ready starts at zero.
  uint64_t credits;
} loom_aie2p_semaphore_requirement_t;

static bool loom_aie2p_semaphore_requirement_less(
    const loom_aie2p_semaphore_requirement_t* left,
    const loom_aie2p_semaphore_requirement_t* right) {
  if (left->tile != right->tile) {
    return (uintptr_t)left->tile < (uintptr_t)right->tile;
  }
  if (left->credits != right->credits) {
    return left->credits < right->credits;
  }
  return left->channel < right->channel;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_aie2p_semaphore_sort_requirements,
                          loom_aie2p_semaphore_requirement_t,
                          loom_aie2p_semaphore_requirement_less)

// Translates compatible physical requirements into common lifetime requests.
// Slot numbers are relative identities, not hardware lock IDs: assigning the
// latter in original channel order preserves nonaliasing programs' numbering.
static iree_status_t loom_aie2p_semaphore_slots(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_invocations_t* invocations,
    const iree_host_size_t** out_slots) {
  iree_arena_allocator_t* arena = context->pass->arena;
  loom_aie2p_semaphore_requirement_t* requirements = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, invocations->channel_count,
                                sizeof(*requirements), (void**)&requirements));
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < invocations->channel_count; ++i) {
    const loom_aie2p_native_channel_t* channel = &context->channels[i];
    // A root configuration may debit its first reservation during startup.
    // Root state coexists with every child and cannot share its pair, so that
    // debit does not affect compatibility between serial constructions.
    requirements[count++] = (loom_aie2p_semaphore_requirement_t){
        .channel = i,
        .tile = context->pool_tiles[channel->pool_index],
        .credits = channel->replay.record_count ? channel->replay.credits
                                                : channel->source->capacity};
  }
  loom_aie2p_semaphore_sort_requirements(requirements, count);
  loom_pipeline_lifetime_request_t* requests = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, count, sizeof(*requests), (void**)&requests));
  iree_host_size_t pool_count = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (!i || requirements[i].tile != requirements[i - 1].tile ||
        requirements[i].credits != requirements[i - 1].credits) {
      ++pool_count;
    }
    requests[i] = (loom_pipeline_lifetime_request_t){
        .lifetime = invocations->lifetimes
                        .owners[invocations->channels[requirements[i].channel]
                                    .invocation->ordinal],
        .pool = pool_count - 1,
        .length = 1,
        .alignment = 1};
  }
  const loom_pipeline_lifetime_extent_t* extents = NULL;
  IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_layout(
      &invocations->lifetimes, requests, count, pool_count, arena, &extents));
  iree_host_size_t* slots = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, invocations->channel_count, sizeof(*slots), (void**)&slots));
  *out_slots = slots;
  iree_host_size_t pool_offset = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (i && requests[i].pool != requests[i - 1].pool) {
      pool_offset += (iree_host_size_t)extents[requests[i - 1].pool].length;
    }
    slots[requirements[i].channel] =
        pool_offset + (iree_host_size_t)requests[i].offset;
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_assign_channel_locks(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, bool* out_valid) {
  *out_valid = false;
  const loom_pipeline_invocations_t* invocations = &realization->invocations;
  const iree_host_size_t* slots = NULL;
  uint16_t* locks = NULL;
  if (invocations->lifetimes.has_serial) {
    IREE_RETURN_IF_ERROR(
        loom_aie2p_semaphore_slots(context, invocations, &slots));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->pass->arena, invocations->channel_count, sizeof(*locks),
        (void**)&locks));
    memset(locks, 0xff, invocations->channel_count * sizeof(*locks));
  }
  for (iree_host_size_t i = 0; i < invocations->channel_count; ++i) {
    loom_aie2p_native_channel_t* channel = &context->channels[i];
    loom_aie2p_native_tile_t* owner = context->pool_tiles[channel->pool_index];
    const bool initialize = !slots || locks[slots[i]] == UINT16_MAX;
    if (initialize) {
      if (owner->next_lock + 2 > owner->facts->lock_count) {
        return loom_aie2p_native_reject(
            context, channel->source->binding,
            IREE_SV("a hardware semaphore pair with enough slot credits"));
      }
      channel->free_lock = owner->next_lock;
      owner->next_lock += 2;
      if (slots) {
        locks[slots[i]] = channel->free_lock;
      }
    } else {
      channel->free_lock = locks[slots[i]];
    }
    channel->ready_lock = channel->free_lock + 1;
    channel->initializes_locks = initialize;
  }
  *out_valid = true;
  return iree_ok_status();
}
