// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/cts/util/atomic_test_util.h"

namespace iree::hal::cts {
namespace {

iree_hal_atomic_operation_flags_t AtomicOperationCapabilitiesForCell(
    const iree_hal_atomic_operation_capabilities_t& capabilities,
    iree_hal_atomic_width_t width, iree_hal_atomic_flags_t atomic_flags) {
  const bool system_scope =
      iree_any_bit_set(atomic_flags, IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE);
  switch (width) {
    case IREE_HAL_ATOMIC_WIDTH_32:
      return system_scope ? capabilities.system_scope_32
                          : capabilities.device_scope_32;
    case IREE_HAL_ATOMIC_WIDTH_64:
      return system_scope ? capabilities.system_scope_64
                          : capabilities.device_scope_64;
    default:
      return IREE_HAL_ATOMIC_OPERATION_FLAG_NONE;
  }
}

iree_hal_atomic_wait_condition_flags_t AtomicWaitConditionsForCell(
    const iree_hal_atomic_wait_condition_capabilities_t& capabilities,
    iree_hal_atomic_width_t width, iree_hal_atomic_flags_t atomic_flags) {
  const bool system_scope =
      iree_any_bit_set(atomic_flags, IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE);
  switch (width) {
    case IREE_HAL_ATOMIC_WIDTH_32:
      return system_scope ? capabilities.system_scope_32
                          : capabilities.device_scope_32;
    case IREE_HAL_ATOMIC_WIDTH_64:
      return system_scope ? capabilities.system_scope_64
                          : capabilities.device_scope_64;
    default:
      return IREE_HAL_ATOMIC_WAIT_CONDITION_FLAG_NONE;
  }
}

}  // namespace

bool SelectAtomicTestConfiguration(const iree_hal_device_spec_t* device_spec,
                                   const AtomicTestRequirements& requirements,
                                   AtomicTestConfiguration* out_configuration) {
  *out_configuration = {};
  const iree_hal_device_queue_spec_t* queues =
      iree_hal_device_spec_queues(device_spec);
  const iree_hal_device_memory_spec_t* memory =
      iree_hal_device_spec_memory(device_spec);
  if (!queues || !memory) return false;

  for (iree_host_size_t queue_index = 0; queue_index < queues->family_count;
       ++queue_index) {
    const iree_hal_queue_family_spec_t& queue_family =
        queues->families[queue_index];
    if (!iree_all_bits_set(queue_family.role_flags,
                           IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_ATOMIC) ||
        queue_family.queue_affinity == 0) {
      continue;
    }
    const iree_hal_atomic_operation_flags_t queue_operations =
        AtomicOperationCapabilitiesForCell(
            queue_family.atomic_capabilities.operations, requirements.width,
            requirements.atomic_flags);
    if (!iree_all_bits_set(queue_operations, requirements.operation_flags)) {
      continue;
    }
    const iree_hal_atomic_wait_condition_flags_t queue_wait_conditions =
        AtomicWaitConditionsForCell(
            queue_family.atomic_capabilities.wait_conditions,
            requirements.width, requirements.atomic_flags);
    if (!iree_all_bits_set(queue_wait_conditions,
                           requirements.wait_condition_flags)) {
      continue;
    }

    for (iree_host_size_t memory_index = 0;
         memory_index < memory->memory_type_count; ++memory_index) {
      const iree_hal_memory_type_spec_t& memory_type =
          memory->memory_types[memory_index];
      const iree_hal_memory_heap_spec_t& memory_heap =
          memory->heaps[memory_type.heap_index];
      if ((memory_heap.physical_device_affinity &
           queue_family.physical_device_affinity) == 0 ||
          !iree_all_bits_set(memory_type.memory_type,
                             requirements.memory_type) ||
          !iree_all_bits_set(memory_type.allowed_buffer_usage,
                             requirements.buffer_usage) ||
          !iree_all_bits_set(memory_type.allowed_memory_access,
                             requirements.memory_access)) {
        continue;
      }
      const iree_hal_atomic_operation_flags_t memory_operations =
          AtomicOperationCapabilitiesForCell(memory_type.atomic_operations,
                                             requirements.width,
                                             requirements.atomic_flags);
      if (!iree_all_bits_set(memory_operations, requirements.operation_flags)) {
        continue;
      }

      const iree_hal_queue_affinity_t family_affinity =
          queue_family.queue_affinity;
      out_configuration->queue_affinity =
          family_affinity & (~family_affinity + 1);
      out_configuration->buffer_params = {
          /*.usage=*/requirements.buffer_usage,
          /*.access=*/requirements.memory_access,
          /*.type=*/memory_type.memory_type,
          /*.queue_affinity=*/out_configuration->queue_affinity,
          /*.min_alignment=*/
          iree_hal_atomic_width_byte_count(requirements.width),
      };
      return true;
    }
  }
  return false;
}

}  // namespace iree::hal::cts
