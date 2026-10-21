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

bool SelectAtomicTestConfiguration(iree_hal_device_t* device,
                                   const AtomicTestRequirements& requirements,
                                   AtomicTestConfiguration* out_configuration) {
  *out_configuration = {};
  const iree_hal_device_spec_t* device_spec = iree_hal_device_spec(device);
  const iree_hal_device_queue_spec_t* queues =
      iree_hal_device_spec_queues(device_spec);
  if (!queues) {
    return false;
  }

  for (iree_host_size_t family_index = 0; family_index < queues->family_count;
       ++family_index) {
    const iree_hal_queue_family_spec_t& queue_family =
        queues->families[family_index];
    if (!iree_all_bits_set(queue_family.role_flags,
                           IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_ATOMIC) ||
        queue_family.provisioned_queue_count == 0) {
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

    iree_hal_atomic_operation_capabilities_t required_atomics = {};
    const bool system_scope = iree_any_bit_set(
        requirements.atomic_flags, IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE);
    switch (requirements.width) {
      case IREE_HAL_ATOMIC_WIDTH_32:
        if (system_scope) {
          required_atomics.system_scope_32 = requirements.operation_flags;
        } else {
          required_atomics.device_scope_32 = requirements.operation_flags;
        }
        break;
      case IREE_HAL_ATOMIC_WIDTH_64:
        if (system_scope) {
          required_atomics.system_scope_64 = requirements.operation_flags;
        } else {
          required_atomics.device_scope_64 = requirements.operation_flags;
        }
        break;
      default:
        return false;
    }

    out_configuration->queue_family_ordinal =
        (iree_hal_queue_family_ordinal_t)family_index;
    out_configuration->family_access.family = iree_hal_device_queue_family(
        device, out_configuration->queue_family_ordinal);
    out_configuration->family_access.usage = requirements.buffer_usage;
    out_configuration->family_access.atomics.queue = required_atomics;
    out_configuration->min_alignment =
        iree_hal_atomic_width_byte_count(requirements.width);
    return true;
  }
  return false;
}

iree_status_t CreateAtomicTestPool(iree_hal_device_group_t* device_group,
                                   const AtomicTestConfiguration& configuration,
                                   iree_hal_pool_t** out_pool) {
  const iree_hal_pool_scope_t scope = {
      .family_count = 1,
      .families = &configuration.family_access,
  };
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  options.trace_name = IREE_SV("cts.atomic");
  return iree_hal_slab_pool_create(device_group, scope, &options,
                                   iree_allocator_system(), out_pool);
}

iree_status_t CreateAtomicTestHostImportPool(
    iree_hal_device_group_t* device_group,
    const AtomicTestConfiguration& configuration, iree_hal_pool_t** out_pool) {
  const iree_hal_pool_scope_t scope = {
      .family_count = 1,
      .families = &configuration.family_access,
  };
  const iree_hal_pool_import_class_t import_class = {
      .storage_class = iree_hal_make_external_storage_class(
          IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
  };
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  options.owned.mode = IREE_HAL_SLAB_POOL_IMPORT_ONLY;
  options.imports.class_count = 1;
  options.imports.classes = &import_class;
  options.trace_name = IREE_SV("cts.atomic.import");
  return iree_hal_slab_pool_create(device_group, scope, &options,
                                   iree_allocator_system(), out_pool);
}

}  // namespace iree::hal::cts
