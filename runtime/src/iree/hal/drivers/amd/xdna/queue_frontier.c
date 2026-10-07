// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/queue_frontier.h"

#include <string.h>

void iree_hal_amd_xdna_frontier_state_initialize(
    iree_hal_amd_xdna_frontier_state_t* out_state) {
  iree_async_frontier_initialize(
      iree_async_fixed_frontier_as_frontier(&out_state->frontier), 0);
  out_state->exact = true;
}

void iree_hal_amd_xdna_frontier_state_copy(
    const iree_hal_amd_xdna_frontier_state_t* source,
    iree_hal_amd_xdna_frontier_state_t* target) {
  target->frontier.entry_count = source->frontier.entry_count;
  memcpy(target->frontier.reserved, source->frontier.reserved,
         sizeof(target->frontier.reserved));
  memcpy(target->frontier.entries, source->frontier.entries,
         source->frontier.entry_count * sizeof(iree_async_frontier_entry_t));
  target->exact = source->exact;
}

void iree_hal_amd_xdna_frontier_state_merge(
    iree_hal_amd_xdna_frontier_state_t* target,
    const iree_hal_amd_xdna_frontier_state_t* source) {
  if (!iree_async_frontier_merge(
          iree_async_fixed_frontier_as_frontier(&target->frontier),
          IREE_HAL_AMD_XDNA_FRONTIER_CAPACITY,
          iree_async_fixed_frontier_as_const_frontier(&source->frontier))) {
    target->exact = false;
  }
  target->exact &= source->exact;
}

void iree_hal_amd_xdna_frontier_state_advance(
    iree_hal_amd_xdna_frontier_state_t* state, iree_async_axis_t axis,
    uint64_t epoch) {
  iree_async_single_frontier_t self_frontier;
  iree_async_single_frontier_initialize(&self_frontier, axis, epoch);
  if (iree_async_frontier_merge(
          iree_async_fixed_frontier_as_frontier(&state->frontier),
          IREE_HAL_AMD_XDNA_FRONTIER_CAPACITY,
          iree_async_single_frontier_as_const_frontier(&self_frontier))) {
    return;
  }

  // The accepted self epoch is always a valid causal lower bound. Retaining it
  // lets the queue continue to describe its physical progress after losing the
  // complete transitive history.
  iree_async_frontier_initialize(
      iree_async_fixed_frontier_as_frontier(&state->frontier), 1);
  state->frontier.entries[0] = self_frontier.entries[0];
  state->exact = false;
}

bool iree_hal_amd_xdna_frontier_state_dominates(
    const iree_hal_amd_xdna_frontier_state_t* state, iree_async_axis_t axis,
    uint64_t epoch) {
  for (uint8_t i = 0; i < state->frontier.entry_count; ++i) {
    const iree_async_frontier_entry_t* entry = &state->frontier.entries[i];
    if (entry->axis < axis) {
      continue;
    }
    return entry->axis == axis && entry->epoch >= epoch;
  }
  return false;
}

const iree_async_frontier_t* iree_hal_amd_xdna_frontier_state_as_frontier(
    const iree_hal_amd_xdna_frontier_state_t* state) {
  return iree_async_fixed_frontier_as_const_frontier(&state->frontier);
}

// Merges the frontier attached to a reached wait. XDNA publishes native
// frontiers before timeline completion; a newer undominated publication must
// not be mistaken for history of the reached value.
static void iree_hal_amd_xdna_frontier_merge_reached_wait(
    iree_hal_semaphore_t* semaphore, uint64_t current_value,
    uint64_t minimum_value, iree_hal_amd_xdna_frontier_state_t* state) {
  iree_async_semaphore_t* async_semaphore = (iree_async_semaphore_t*)semaphore;
  if (iree_async_semaphore_is_value_tainted(async_semaphore, minimum_value)) {
    state->exact = false;
    return;
  }

  iree_hal_amd_xdna_frontier_t wait_frontier;
  const uint8_t actual_count = iree_async_semaphore_query_frontier(
      async_semaphore, iree_async_fixed_frontier_as_frontier(&wait_frontier),
      IREE_HAL_AMD_XDNA_FRONTIER_CAPACITY);
  if (actual_count == 0) {
    return;
  }

  // Only the XDNA semaphore exposes the submission snapshot needed to
  // distinguish completed causality from a later frontier published before
  // completion. A foreign frontier therefore cannot make this snapshot exact.
  if (!iree_hal_amd_xdna_semaphore_isa(semaphore)) {
    state->exact = false;
    return;
  }

  iree_hal_submitted_signal_flags_t signal_flags =
      IREE_HAL_SUBMITTED_SIGNAL_FLAG_NONE;
  iree_async_axis_t producer_axis = 0;
  uint64_t producer_epoch = 0;
  uint64_t producer_value = 0;
  const bool has_submitted_signal = iree_hal_submitted_signal_load(
      iree_hal_amd_xdna_semaphore_submitted_signal(semaphore), &signal_flags,
      &producer_axis, &producer_epoch, &producer_value);
  if (!has_submitted_signal) {
    if (!iree_any_bit_set(iree_hal_amd_xdna_semaphore_flags(semaphore),
                          IREE_HAL_SEMAPHORE_FLAG_SINGLE_PRODUCER)) {
      state->exact = false;
      return;
    }
  } else if (!iree_all_bits_set(
                 signal_flags,
                 IREE_HAL_SUBMITTED_SIGNAL_FLAG_PRODUCER_FRONTIER_EXACT)) {
    // The accumulated semaphore frontier contains independent producers. If
    // this exact cached value completed, its producer epoch is a safe lower
    // bound, but the other published axes may still be in flight.
    if (producer_value == current_value) {
      iree_hal_amd_xdna_frontier_state_t producer_state;
      iree_hal_amd_xdna_frontier_state_initialize(&producer_state);
      iree_hal_amd_xdna_frontier_state_advance(&producer_state, producer_axis,
                                               producer_epoch);
      producer_state.exact = false;
      iree_hal_amd_xdna_frontier_state_merge(state, &producer_state);
    } else {
      state->exact = false;
    }
    return;
  } else if (producer_value != current_value &&
             !iree_hal_amd_xdna_frontier_state_dominates(state, producer_axis,
                                                         producer_epoch)) {
    // The cache names a different value. It may be a later producer depending
    // on this consumer or a stale producer superseded by another completion.
    // Neither case permits installing its full frontier as new causality.
    state->exact = false;
    return;
  }

  iree_hal_amd_xdna_frontier_state_t source;
  iree_hal_amd_xdna_frontier_state_initialize(&source);
  source.frontier.entry_count = wait_frontier.entry_count;
  memcpy(source.frontier.entries, wait_frontier.entries,
         wait_frontier.entry_count * sizeof(iree_async_frontier_entry_t));
  if (actual_count > IREE_HAL_AMD_XDNA_FRONTIER_CAPACITY) {
    source.exact = false;
  }
  iree_hal_amd_xdna_frontier_state_merge(state, &source);
}

iree_status_t iree_hal_amd_xdna_frontier_resolve_waits(
    iree_hal_device_t* device, iree_hal_semaphore_list_t waits,
    const iree_hal_amd_xdna_frontier_state_t* accepted_state,
    const iree_hal_amd_xdna_frontier_state_t* initial_state,
    iree_hal_amd_xdna_wait_resolution_flags_t flags,
    iree_hal_amd_xdna_frontier_state_t* out_state,
    iree_hal_amd_xdna_wait_resolution_t* out_resolution,
    iree_host_size_t* out_deferred_wait_index) {
  if (initial_state) {
    if (initial_state != out_state) {
      iree_hal_amd_xdna_frontier_state_copy(initial_state, out_state);
    }
  } else {
    iree_hal_amd_xdna_frontier_state_initialize(out_state);
  }
  if (accepted_state) {
    iree_hal_amd_xdna_frontier_state_merge(out_state, accepted_state);
  }
  *out_resolution = IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_READY;
  if (out_deferred_wait_index) {
    *out_deferred_wait_index = IREE_HOST_SIZE_MAX;
  }

  for (iree_host_size_t i = 0; i < waits.count; ++i) {
    iree_hal_semaphore_t* semaphore = waits.semaphores[i];
    const uint64_t minimum_value = waits.payload_values[i];
    if (minimum_value == 0) {
      continue;
    }

    uint64_t current_value = 0;
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_query(semaphore, &current_value));
    if (current_value >= minimum_value) {
      iree_hal_amd_xdna_frontier_merge_reached_wait(semaphore, current_value,
                                                    minimum_value, out_state);
      continue;
    }

    bool fifo_resolved = false;
    if (accepted_state && accepted_state->exact && out_state->exact &&
        iree_any_bit_set(
            flags,
            IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_FLAG_ALLOW_ACCEPTED_FIFO) &&
        iree_hal_amd_xdna_semaphore_is_local(semaphore, device)) {
      iree_hal_submitted_signal_flags_t signal_flags =
          IREE_HAL_SUBMITTED_SIGNAL_FLAG_NONE;
      iree_async_axis_t producer_axis = 0;
      uint64_t producer_epoch = 0;
      uint64_t producer_value = 0;
      fifo_resolved =
          iree_hal_submitted_signal_load(
              iree_hal_amd_xdna_semaphore_submitted_signal(semaphore),
              &signal_flags, &producer_axis, &producer_epoch,
              &producer_value) &&
          producer_value >= minimum_value &&
          iree_all_bits_set(
              signal_flags,
              IREE_HAL_SUBMITTED_SIGNAL_FLAG_PRODUCER_FRONTIER_EXACT) &&
          iree_hal_amd_xdna_frontier_state_dominates(
              accepted_state, producer_axis, producer_epoch);
    }
    if (!fifo_resolved) {
      *out_resolution = IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_DEFER;
      if (out_deferred_wait_index) {
        *out_deferred_wait_index = i;
      }
      return iree_ok_status();
    }
  }

  return iree_ok_status();
}
