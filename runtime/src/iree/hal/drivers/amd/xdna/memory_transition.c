// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/memory_transition.h"

#include "iree/base/internal/math.h"

static iree_status_t iree_hal_amd_xdna_memory_translate_atomic_scope(
    amdf_atomic_scope_t source, iree_hal_atomic_reach_scope_t* out_scope) {
  switch (source) {
    case AMDF_ATOMIC_SCOPE_NONE:
      *out_scope = IREE_HAL_ATOMIC_REACH_NONE;
      return iree_ok_status();
    case AMDF_ATOMIC_SCOPE_DEVICE:
      *out_scope = IREE_HAL_ATOMIC_REACH_DEVICE;
      return iree_ok_status();
    case AMDF_ATOMIC_SCOPE_FABRIC:
      *out_scope = IREE_HAL_ATOMIC_REACH_FABRIC;
      return iree_ok_status();
    case AMDF_ATOMIC_SCOPE_SYSTEM:
      *out_scope = IREE_HAL_ATOMIC_REACH_SYSTEM;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "libamdf returned invalid atomic scope %u",
                              source);
  }
}

static iree_status_t iree_hal_amd_xdna_memory_translate_host_instruction(
    amdf_host_cache_instruction_t source,
    iree_hal_host_cache_instruction_t* out_instruction) {
  switch (source) {
    case AMDF_HOST_CACHE_INSTRUCTION_NONE:
      *out_instruction = IREE_HAL_HOST_CACHE_INSTRUCTION_NONE;
      return iree_ok_status();
    case AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH:
      *out_instruction = IREE_HAL_HOST_CACHE_INSTRUCTION_X86_CLFLUSH;
      return iree_ok_status();
    case AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSHOPT:
      *out_instruction = IREE_HAL_HOST_CACHE_INSTRUCTION_X86_CLFLUSHOPT;
      return iree_ok_status();
    case AMDF_HOST_CACHE_INSTRUCTION_X86_CLWB:
      *out_instruction = IREE_HAL_HOST_CACHE_INSTRUCTION_X86_CLWB;
      return iree_ok_status();
    default:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "libamdf returned invalid host cache instruction %u", source);
  }
}

static iree_status_t iree_hal_amd_xdna_memory_translate_host_fence(
    amdf_host_cache_fence_t source, iree_hal_host_cache_fence_t* out_fence) {
  switch (source) {
    case AMDF_HOST_CACHE_FENCE_NONE:
      *out_fence = IREE_HAL_HOST_CACHE_FENCE_NONE;
      return iree_ok_status();
    case AMDF_HOST_CACHE_FENCE_X86_SFENCE:
      *out_fence = IREE_HAL_HOST_CACHE_FENCE_X86_SFENCE;
      return iree_ok_status();
    case AMDF_HOST_CACHE_FENCE_X86_MFENCE:
      *out_fence = IREE_HAL_HOST_CACHE_FENCE_X86_MFENCE;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "libamdf returned invalid host cache fence %u",
                              source);
  }
}

static iree_status_t iree_hal_amd_xdna_memory_translate_transition(
    const amdf_cache_transition_t* source,
    iree_hal_memory_transition_action_t action,
    iree_hal_memory_transition_recipe_info_t* out_info) {
  memset(out_info, 0, sizeof(*out_info));
  const bool is_release = action == IREE_HAL_MEMORY_TRANSITION_RELEASE;
  switch (source->kind) {
    case AMDF_CACHE_TRANSITION_KIND_UNKNOWN:
      return iree_ok_status();
    case AMDF_CACHE_TRANSITION_KIND_NONE:
      if (source->executor != AMDF_CACHE_TRANSITION_EXECUTOR_NONE ||
          source->operation != AMDF_CACHE_OPERATION_NONE ||
          source->host_operation != AMDF_HOST_CACHE_OPERATION_NONE ||
          source->host_instruction != AMDF_HOST_CACHE_INSTRUCTION_NONE ||
          source->host_fence_before != AMDF_HOST_CACHE_FENCE_NONE ||
          source->host_fence_after != AMDF_HOST_CACHE_FENCE_NONE ||
          source->range_granularity != 0) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "libamdf returned payload fields for a no-op cache transition");
      }
      out_info->kind = IREE_HAL_MEMORY_TRANSITION_KIND_NONE;
      return iree_ok_status();
    case AMDF_CACHE_TRANSITION_KIND_RANGE:
      if (!iree_device_size_is_power_of_two(source->range_granularity)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "libamdf returned invalid range granularity %" PRIu64,
            source->range_granularity);
      }
      out_info->kind = IREE_HAL_MEMORY_TRANSITION_KIND_RANGE;
      out_info->range_granularity = source->range_granularity;
      break;
    case AMDF_CACHE_TRANSITION_KIND_GLOBAL:
      if (source->range_granularity != 0) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "libamdf returned a range granularity for a global transition");
      }
      out_info->kind = IREE_HAL_MEMORY_TRANSITION_KIND_GLOBAL;
      break;
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "libamdf returned invalid transition kind %u",
                              source->kind);
  }

  switch (source->executor) {
    case AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE:
      out_info->executor = IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE;
      break;
    case AMDF_CACHE_TRANSITION_EXECUTOR_PROGRAM:
      out_info->executor = IREE_HAL_MEMORY_TRANSITION_EXECUTOR_PROGRAM;
      break;
    case AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT:
      out_info->executor = IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_DIRECT;
      break;
    case AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API:
      out_info->executor = IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API;
      break;
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "libamdf returned invalid transition executor %u",
                              source->executor);
  }

  const bool is_host_executor =
      source->executor == AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT ||
      source->executor == AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API;
  if (is_host_executor) {
    if (source->kind != AMDF_CACHE_TRANSITION_KIND_RANGE ||
        source->operation != AMDF_CACHE_OPERATION_NONE ||
        source->host_operation !=
            (is_release ? AMDF_HOST_CACHE_OPERATION_FLUSH
                        : AMDF_HOST_CACHE_OPERATION_INVALIDATE)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "libamdf returned an invalid directional host transition");
    }
    out_info->operation =
        is_release ? IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH
                   : IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_INVALIDATE;
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_memory_translate_host_instruction(
        source->host_instruction, &out_info->host.instruction));
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_memory_translate_host_fence(
        source->host_fence_before, &out_info->host.fence_before));
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_memory_translate_host_fence(
        source->host_fence_after, &out_info->host.fence_after));
    if (source->executor == AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API &&
        (out_info->host.instruction != IREE_HAL_HOST_CACHE_INSTRUCTION_NONE ||
         out_info->host.fence_before != IREE_HAL_HOST_CACHE_FENCE_NONE ||
         out_info->host.fence_after != IREE_HAL_HOST_CACHE_FENCE_NONE)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "libamdf returned direct-host instructions for a "
                              "host API transition");
    }
    return iree_ok_status();
  }

  if (source->host_operation != AMDF_HOST_CACHE_OPERATION_NONE ||
      source->host_instruction != AMDF_HOST_CACHE_INSTRUCTION_NONE ||
      source->host_fence_before != AMDF_HOST_CACHE_FENCE_NONE ||
      source->host_fence_after != AMDF_HOST_CACHE_FENCE_NONE ||
      source->operation != (is_release
                                ? AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM
                                : AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "libamdf returned an invalid directional device transition");
  }
  out_info->operation =
      is_release ? IREE_HAL_MEMORY_TRANSITION_OPERATION_RELEASE_TO_SYSTEM
                 : IREE_HAL_MEMORY_TRANSITION_OPERATION_ACQUIRE_FROM_SYSTEM;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_memory_translate_pair(
    const amdf_memory_pair_info_t* source,
    iree_hal_memory_pair_info_t* out_pair) {
  memset(out_pair, 0, sizeof(*out_pair));
  if (iree_any_bit_set(source->flags,
                       AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE)) {
    out_pair->flags |= IREE_HAL_MEMORY_PAIR_SHARED_BACKING_REACHABLE;
  }
  if (iree_any_bit_set(source->flags, AMDF_MEMORY_PAIR_FLAG_MAPPING_SOURCE)) {
    out_pair->flags |= IREE_HAL_MEMORY_PAIR_MAPPING_SOURCE;
  }
  if (iree_any_bit_set(source->flags, AMDF_MEMORY_PAIR_FLAG_FIXED_COST_KNOWN)) {
    out_pair->flags |= IREE_HAL_MEMORY_PAIR_FIXED_COST_KNOWN;
    out_pair->estimated_fixed_cost_nanoseconds =
        source->estimated_fixed_cost_nanoseconds;
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_memory_translate_transition(
      &source->release, IREE_HAL_MEMORY_TRANSITION_RELEASE,
      &out_pair->release));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_memory_translate_transition(
      &source->acquire, IREE_HAL_MEMORY_TRANSITION_ACQUIRE,
      &out_pair->acquire));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_memory_translate_atomic_scope(
      source->atomic_reach.scope_32, &out_pair->atomic_reach.scope_32));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_memory_translate_atomic_scope(
      source->atomic_reach.scope_64, &out_pair->atomic_reach.scope_64));
  return iree_ok_status();
}
