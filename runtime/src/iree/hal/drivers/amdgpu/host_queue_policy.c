// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_policy.h"

#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/drivers/amdgpu/system.h"

// Returns true when |queue_family_affinity| names only the family containing
// |queue|. AMDGPU currently exposes one family per physical HSA agent, so this
// proves that all compatible queues share the agent. A multi-GPU logical
// device still needs SYSTEM scope for cross-family synchronization.
static bool iree_hal_amdgpu_host_queue_family_affinity_is_same_agent(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_queue_family_affinity_t queue_family_affinity) {
  const iree_hal_amdgpu_logical_device_t* logical_device =
      (const iree_hal_amdgpu_logical_device_t*)queue->logical_device;
  if (iree_hal_queue_family_affinity_is_any(queue_family_affinity)) {
    return logical_device->physical_device_count == 1;
  }
  return queue_family_affinity ==
         iree_hal_make_queue_family_affinity(
             (iree_hal_queue_family_ordinal_t)queue->device_ordinal);
}

// Returns true when a semaphore edge can be represented with HSA AGENT scope
// on |queue|. Public/exportable/host-visible semaphores and cross-agent
// affinities use SYSTEM scope.
static bool iree_hal_amdgpu_host_queue_can_use_agent_scope(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_semaphore_t* semaphore) {
  const iree_hal_amdgpu_logical_device_t* logical_device =
      (const iree_hal_amdgpu_logical_device_t*)queue->logical_device;
  if (!iree_hal_amdgpu_semaphore_is_local(semaphore, logical_device)) {
    return false;
  }

  const iree_hal_semaphore_flags_t flags =
      iree_hal_amdgpu_semaphore_flags(semaphore);
  if (!iree_all_bits_set(flags, IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL)) {
    return false;
  }
  const iree_hal_semaphore_flags_t public_flags =
      IREE_HAL_SEMAPHORE_FLAG_HOST_INTERRUPT |
      IREE_HAL_SEMAPHORE_FLAG_EXPORTABLE |
      IREE_HAL_SEMAPHORE_FLAG_EXPORTABLE_TIMEPOINTS;
  if (iree_any_bit_set(flags, public_flags)) return false;

  return iree_hal_amdgpu_host_queue_family_affinity_is_same_agent(
      queue, iree_hal_amdgpu_semaphore_queue_family_affinity(semaphore));
}

iree_hsa_fence_scope_t iree_hal_amdgpu_host_queue_wait_acquire_scope(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_semaphore_t* semaphore) {
  return iree_hal_amdgpu_host_queue_can_use_agent_scope(queue, semaphore)
             ? IREE_HSA_FENCE_SCOPE_AGENT
             : IREE_HSA_FENCE_SCOPE_SYSTEM;
}

iree_hsa_fence_scope_t iree_hal_amdgpu_host_queue_axis_acquire_scope(
    const iree_hal_amdgpu_host_queue_t* queue, iree_async_axis_t axis) {
  // Session, machine, domain, and logical-device identity occupy the upper
  // 32 bits of a queue axis. Queue indices are flattened across physical
  // devices, with each physical device owning one contiguous range.
  if ((axis >> 32) != (queue->axis >> 32)) {
    return IREE_HSA_FENCE_SCOPE_SYSTEM;
  }
  const iree_hal_amdgpu_logical_device_t* logical_device =
      (const iree_hal_amdgpu_logical_device_t*)queue->logical_device;
  const iree_host_size_t queue_count_per_physical_device =
      logical_device->system->topology.gpu_agent_queue_count;
  const iree_host_size_t candidate_device_ordinal =
      iree_async_axis_queue_index(axis) / queue_count_per_physical_device;
  return candidate_device_ordinal == queue->device_ordinal
             ? IREE_HSA_FENCE_SCOPE_AGENT
             : IREE_HSA_FENCE_SCOPE_SYSTEM;
}

iree_hsa_fence_scope_t iree_hal_amdgpu_host_queue_signal_release_scope(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_semaphore_t* semaphore) {
  return iree_hal_amdgpu_host_queue_can_use_agent_scope(queue, semaphore)
             ? IREE_HSA_FENCE_SCOPE_AGENT
             : IREE_HSA_FENCE_SCOPE_SYSTEM;
}

iree_hsa_fence_scope_t iree_hal_amdgpu_host_queue_signal_list_release_scope(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_semaphore_list_t semaphores) {
  iree_hsa_fence_scope_t release_scope = IREE_HSA_FENCE_SCOPE_AGENT;
  for (iree_host_size_t i = 0; i < semaphores.count; ++i) {
    release_scope = iree_hal_amdgpu_host_queue_max_fence_scope(
        release_scope, iree_hal_amdgpu_host_queue_signal_release_scope(
                           queue, semaphores.semaphores[i]));
  }
  return release_scope;
}

iree_hsa_fence_scope_t iree_hal_amdgpu_host_queue_buffer_acquire_scope(
    const iree_hal_buffer_t* source_buffer) {
  return iree_any_bit_set(iree_hal_buffer_memory_type(source_buffer),
                          IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)
             ? IREE_HSA_FENCE_SCOPE_SYSTEM
             : IREE_HSA_FENCE_SCOPE_NONE;
}

iree_hsa_fence_scope_t iree_hal_amdgpu_host_queue_buffer_release_scope(
    const iree_hal_buffer_t* target_buffer) {
  return iree_any_bit_set(iree_hal_buffer_memory_type(target_buffer),
                          IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)
             ? IREE_HSA_FENCE_SCOPE_SYSTEM
             : IREE_HSA_FENCE_SCOPE_NONE;
}
