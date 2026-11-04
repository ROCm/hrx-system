// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/array/resident_plan.h"

static bool loom_aie2p_array_fold_span_equal(
    const loom_aie2p_array_fold_span_t* lhs,
    const loom_aie2p_array_fold_span_t* rhs) {
  return lhs->output_index == rhs->output_index &&
         lhs->output_byte_offset == rhs->output_byte_offset &&
         lhs->state_byte_offset == rhs->state_byte_offset &&
         lhs->byte_length == rhs->byte_length &&
         lhs->repeat_count == rhs->repeat_count;
}

static bool loom_aie2p_array_fold_state_equal(
    const loom_aie2p_array_fold_state_plan_t* lhs,
    const loom_aie2p_array_fold_state_plan_t* rhs) {
  if (lhs->load_address != rhs->load_address ||
      lhs->byte_length != rhs->byte_length ||
      lhs->span_count != rhs->span_count) {
    return false;
  }
  for (uint32_t i = 0; i < lhs->span_count; ++i) {
    if (!loom_aie2p_array_fold_span_equal(&lhs->spans[i], &rhs->spans[i])) {
      return false;
    }
  }
  return true;
}

static uint32_t loom_aie2p_array_worker_port_ordinal(
    const loom_aie2p_array_worker_plan_t* worker_plan, uint32_t port_index) {
  IREE_ASSERT_GE(port_index, worker_plan->first_port);
  const uint32_t ordinal = port_index - worker_plan->first_port;
  IREE_ASSERT_LT(ordinal, worker_plan->port_count);
  return ordinal;
}

static uint32_t loom_aie2p_array_worker_port_slot_address(
    const loom_aie2p_array_plan_t* plan,
    const loom_aie2p_array_worker_port_plan_t* port, uint32_t slot) {
  const loom_aie2p_array_channel_t* channel =
      &plan->channels[port->channel_index];
  IREE_ASSERT_LT(slot, channel->capacity);
  const loom_aie2p_array_channel_slot_t* channel_slot =
      &plan->channel_slots[channel->first_channel_slot + slot];
  return port->direction == LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND
             ? channel_slot->sender_storage.load_address
             : channel_slot->receiver_storage.load_address;
}

static bool loom_aie2p_array_worker_port_equal(
    const loom_aie2p_array_plan_t* plan,
    const loom_aie2p_array_worker_port_plan_t* lhs,
    const loom_aie2p_array_worker_port_plan_t* rhs) {
  const loom_aie2p_array_channel_t* lhs_channel =
      &plan->channels[lhs->channel_index];
  const loom_aie2p_array_channel_t* rhs_channel =
      &plan->channels[rhs->channel_index];
  if (lhs->port != rhs->port || lhs->direction != rhs->direction ||
      lhs->lock_selectors[0] != rhs->lock_selectors[0] ||
      lhs->lock_selectors[1] != rhs->lock_selectors[1] ||
      lhs->resident_state_ordinal != rhs->resident_state_ordinal ||
      lhs_channel->capacity != rhs_channel->capacity ||
      lhs_channel->record_byte_length != rhs_channel->record_byte_length) {
    return false;
  }
  for (uint32_t i = 0; i < lhs_channel->capacity; ++i) {
    if (loom_aie2p_array_worker_port_slot_address(plan, lhs, i) !=
        loom_aie2p_array_worker_port_slot_address(plan, rhs, i)) {
      return false;
    }
  }
  return true;
}

static bool loom_aie2p_array_worker_program_equal(
    const loom_aie2p_array_plan_t* plan, uint32_t lhs_worker_index,
    uint32_t rhs_worker_index) {
  const loom_aie2p_array_worker_t* lhs_worker =
      &plan->workers[lhs_worker_index];
  const loom_aie2p_array_worker_t* rhs_worker =
      &plan->workers[rhs_worker_index];
  const loom_aie2p_array_worker_plan_t* lhs_plan =
      &plan->worker_plans[lhs_worker_index];
  const loom_aie2p_array_worker_plan_t* rhs_plan =
      &plan->worker_plans[rhs_worker_index];
  if (lhs_worker->leaf != rhs_worker->leaf ||
      lhs_worker->fold_record_count != rhs_worker->fold_record_count ||
      lhs_worker->fold_output_port != rhs_worker->fold_output_port ||
      lhs_worker->fold_output_count != rhs_worker->fold_output_count ||
      lhs_worker->fold_kind != rhs_worker->fold_kind ||
      lhs_worker->fold_fast_math_flags != rhs_worker->fold_fast_math_flags ||
      lhs_plan->port_count != rhs_plan->port_count ||
      lhs_plan->resident_state_count != rhs_plan->resident_state_count ||
      !loom_aie2p_array_fold_state_equal(&lhs_plan->fold_state,
                                         &rhs_plan->fold_state)) {
    return false;
  }

  for (uint32_t i = 0; i < lhs_plan->port_count; ++i) {
    if (!loom_aie2p_array_worker_port_equal(
            plan, &plan->worker_ports[lhs_plan->first_port + i],
            &plan->worker_ports[rhs_plan->first_port + i])) {
      return false;
    }
  }

  const iree_host_size_t resource_count =
      lhs_worker->leaf->requirements.resource_count;
  for (iree_host_size_t i = 0; i < resource_count; ++i) {
    const uint32_t lhs_port =
        plan->worker_resource_ports[lhs_plan->first_port + i];
    const uint32_t rhs_port =
        plan->worker_resource_ports[rhs_plan->first_port + i];
    if (loom_aie2p_array_worker_port_ordinal(lhs_plan, lhs_port) !=
        loom_aie2p_array_worker_port_ordinal(rhs_plan, rhs_port)) {
      return false;
    }
  }
  for (uint32_t i = 0; i < lhs_worker->fold_output_count; ++i) {
    if (plan->worker_fold_output_states[lhs_plan->first_port + i] !=
        plan->worker_fold_output_states[rhs_plan->first_port + i]) {
      return false;
    }
  }
  return true;
}

iree_host_size_t loom_aie2p_array_resident_plan_build(
    const loom_aie2p_array_plan_t* plan,
    loom_aie2p_array_worker_plan_t* worker_plans,
    loom_aie2p_array_resident_program_plan_t* resident_programs) {
  iree_host_size_t resident_program_count = 0;
  for (iree_host_size_t worker_index = 0; worker_index < plan->worker_count;
       ++worker_index) {
    iree_host_size_t resident_program_index = 0;
    for (; resident_program_index < resident_program_count;
         ++resident_program_index) {
      const uint32_t representative_worker_index =
          resident_programs[resident_program_index].representative_worker_index;
      if (loom_aie2p_array_worker_program_equal(plan, (uint32_t)worker_index,
                                                representative_worker_index)) {
        break;
      }
    }
    if (resident_program_index == resident_program_count) {
      resident_programs[resident_program_count++] =
          (loom_aie2p_array_resident_program_plan_t){
              .representative_worker_index = (uint32_t)worker_index,
          };
    }
    worker_plans[worker_index].resident_program_index =
        (uint32_t)resident_program_index;
    ++resident_programs[resident_program_index].instance_count;
  }
  return resident_program_count;
}
