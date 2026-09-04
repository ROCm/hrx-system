// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/device_plan.h"

#include <inttypes.h>
#include <string.h>

bool iree_hal_vulkan_queue_selection_is_same(
    const iree_hal_vulkan_queue_selection_t* lhs,
    const iree_hal_vulkan_queue_selection_t* rhs) {
  return lhs->family_index == rhs->family_index &&
         lhs->queue_index == rhs->queue_index;
}

bool iree_hal_vulkan_queue_assignment_has_sparse_binding(
    const iree_hal_vulkan_queue_assignment_t* queue_assignment) {
  return queue_assignment->sparse_binding.family_index !=
         IREE_HAL_VULKAN_QUEUE_FAMILY_INVALID;
}

iree_status_t iree_hal_vulkan_queue_affinity_normalize(
    iree_hal_queue_affinity_t supported_affinity,
    iree_hal_queue_affinity_t requested_affinity,
    iree_hal_queue_affinity_t* out_normalized_affinity) {
  iree_hal_queue_affinity_t normalized_affinity =
      iree_hal_queue_affinity_is_any(requested_affinity) ? supported_affinity
                                                         : requested_affinity;
  iree_hal_queue_affinity_and_into(normalized_affinity, supported_affinity);
  if (iree_hal_queue_affinity_is_empty(normalized_affinity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no valid Vulkan queue affinity bits specified");
  }
  *out_normalized_affinity = normalized_affinity;
  return iree_ok_status();
}

static void iree_hal_vulkan_device_plan_initialize_empty(
    iree_allocator_t host_allocator, iree_hal_vulkan_device_plan_t* out_plan) {
  memset(out_plan, 0, sizeof(*out_plan));
  out_plan->host_allocator = host_allocator;
  out_plan->queue_assignment.sparse_binding.family_index =
      IREE_HAL_VULKAN_QUEUE_FAMILY_INVALID;
}

void iree_hal_vulkan_device_plan_deinitialize(
    iree_hal_vulkan_device_plan_t* plan) {
  IREE_ASSERT_ARGUMENT(plan);
  iree_allocator_free(plan->host_allocator, plan->queue_create_infos);
  iree_allocator_free(plan->host_allocator, plan->queue_priorities);
  iree_allocator_free(plan->host_allocator,
                      plan->queue_inventory.queue_indices);
  iree_allocator_free(plan->host_allocator, plan->queue_inventory.families);
  memset(plan, 0, sizeof(*plan));
}

static VkQueueFlags iree_hal_vulkan_effective_queue_flags(
    VkQueueFlags queue_flags) {
  if (iree_any_bit_set(queue_flags,
                       VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) {
    queue_flags |= VK_QUEUE_TRANSFER_BIT;
  }
  return queue_flags;
}

static bool iree_hal_vulkan_queue_family_has(
    const VkQueueFamilyProperties* queue_family, VkQueueFlags required_flags) {
  return queue_family->queueCount > 0 &&
         iree_all_bits_set(
             iree_hal_vulkan_effective_queue_flags(queue_family->queueFlags),
             required_flags);
}

static uint32_t iree_hal_vulkan_select_compute_queue_family(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    const iree_hal_vulkan_device_options_t* options) {
  const bool prefer_dedicated = iree_any_bit_set(
      options->flags, IREE_HAL_VULKAN_DEVICE_FLAG_DEDICATED_COMPUTE_QUEUE);
  uint32_t candidate_family_index = IREE_HAL_VULKAN_QUEUE_FAMILY_INVALID;
  for (uint32_t i = 0; i < snapshot->queue_family_count; ++i) {
    const VkQueueFamilyProperties* queue_family =
        &snapshot->queue_families[i].queueFamilyProperties;
    if (!iree_hal_vulkan_queue_family_has(queue_family, VK_QUEUE_COMPUTE_BIT)) {
      continue;
    }
    const bool has_graphics =
        iree_any_bit_set(queue_family->queueFlags, VK_QUEUE_GRAPHICS_BIT);
    if (!prefer_dedicated || !has_graphics) return i;
    if (candidate_family_index == IREE_HAL_VULKAN_QUEUE_FAMILY_INVALID) {
      candidate_family_index = i;
    }
  }
  return candidate_family_index;
}

static uint32_t iree_hal_vulkan_select_transfer_queue_family(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    uint32_t compute_family_index) {
  uint32_t candidate_family_index = IREE_HAL_VULKAN_QUEUE_FAMILY_INVALID;
  uint32_t same_family_index = IREE_HAL_VULKAN_QUEUE_FAMILY_INVALID;
  uint32_t non_graphics_family_index = IREE_HAL_VULKAN_QUEUE_FAMILY_INVALID;
  for (uint32_t i = 0; i < snapshot->queue_family_count; ++i) {
    const VkQueueFamilyProperties* queue_family =
        &snapshot->queue_families[i].queueFamilyProperties;
    if (!iree_hal_vulkan_queue_family_has(queue_family,
                                          VK_QUEUE_TRANSFER_BIT)) {
      continue;
    }
    const bool has_graphics =
        iree_any_bit_set(queue_family->queueFlags, VK_QUEUE_GRAPHICS_BIT);
    const bool has_compute =
        iree_any_bit_set(queue_family->queueFlags, VK_QUEUE_COMPUTE_BIT);
    if (!has_graphics && !has_compute) return i;
    if (i == compute_family_index) {
      same_family_index = i;
      continue;
    }
    if (!has_graphics &&
        non_graphics_family_index == IREE_HAL_VULKAN_QUEUE_FAMILY_INVALID) {
      non_graphics_family_index = i;
    }
    if (candidate_family_index == IREE_HAL_VULKAN_QUEUE_FAMILY_INVALID) {
      candidate_family_index = i;
    }
  }
  if (same_family_index != IREE_HAL_VULKAN_QUEUE_FAMILY_INVALID) {
    const VkQueueFamilyProperties* compute_family =
        &snapshot->queue_families[same_family_index].queueFamilyProperties;
    if (compute_family->queueCount >= 2) {
      return same_family_index;
    }
  }
  if (non_graphics_family_index != IREE_HAL_VULKAN_QUEUE_FAMILY_INVALID) {
    return non_graphics_family_index;
  }
  if (candidate_family_index != IREE_HAL_VULKAN_QUEUE_FAMILY_INVALID) {
    return candidate_family_index;
  }
  return same_family_index == IREE_HAL_VULKAN_QUEUE_FAMILY_INVALID
             ? compute_family_index
             : same_family_index;
}

static uint32_t iree_hal_vulkan_select_sparse_binding_queue_family(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    uint32_t compute_family_index, uint32_t transfer_family_index) {
  const VkQueueFamilyProperties* compute_family =
      &snapshot->queue_families[compute_family_index].queueFamilyProperties;
  if (iree_all_bits_set(compute_family->queueFlags,
                        VK_QUEUE_SPARSE_BINDING_BIT)) {
    return compute_family_index;
  }

  const VkQueueFamilyProperties* transfer_family =
      &snapshot->queue_families[transfer_family_index].queueFamilyProperties;
  if (iree_all_bits_set(transfer_family->queueFlags,
                        VK_QUEUE_SPARSE_BINDING_BIT)) {
    return transfer_family_index;
  }

  for (uint32_t i = 0; i < snapshot->queue_family_count; ++i) {
    const VkQueueFamilyProperties* queue_family =
        &snapshot->queue_families[i].queueFamilyProperties;
    if (iree_hal_vulkan_queue_family_has(queue_family,
                                         VK_QUEUE_SPARSE_BINDING_BIT)) {
      return i;
    }
  }
  return IREE_HAL_VULKAN_QUEUE_FAMILY_INVALID;
}

static iree_status_t iree_hal_vulkan_select_queue_assignment(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    const iree_hal_vulkan_device_options_t* options,
    iree_hal_vulkan_queue_assignment_t* out_queue_assignment) {
  memset(out_queue_assignment, 0, sizeof(*out_queue_assignment));
  out_queue_assignment->sparse_binding.family_index =
      IREE_HAL_VULKAN_QUEUE_FAMILY_INVALID;

  const uint32_t compute_family_index =
      iree_hal_vulkan_select_compute_queue_family(snapshot, options);
  if (compute_family_index == IREE_HAL_VULKAN_QUEUE_FAMILY_INVALID) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "Vulkan physical device has no compute queue");
  }
  const uint32_t transfer_family_index =
      iree_hal_vulkan_select_transfer_queue_family(snapshot,
                                                   compute_family_index);
  const uint32_t sparse_family_index =
      iree_hal_vulkan_select_sparse_binding_queue_family(
          snapshot, compute_family_index, transfer_family_index);

  const VkQueueFamilyProperties* compute_family =
      &snapshot->queue_families[compute_family_index].queueFamilyProperties;
  const VkQueueFamilyProperties* transfer_family =
      &snapshot->queue_families[transfer_family_index].queueFamilyProperties;
  uint32_t transfer_queue_index = 0;
  if (transfer_family_index == compute_family_index &&
      transfer_family->queueCount > 1) {
    transfer_queue_index = 1;
  }

  // Route the legacy device facade through compute and transfer affinity bits.
  // Exact queue identity is assigned after the canonical inventory is built.
  out_queue_assignment->compute = (iree_hal_vulkan_queue_selection_t){
      .family_index = compute_family_index,
      .queue_index = 0,
      .flags =
          iree_hal_vulkan_effective_queue_flags(compute_family->queueFlags),
      .timestamp_valid_bits = compute_family->timestampValidBits,
      .affinity = 1ull << 0,
  };
  out_queue_assignment->transfer = (iree_hal_vulkan_queue_selection_t){
      .family_index = transfer_family_index,
      .queue_index = transfer_queue_index,
      .flags =
          iree_hal_vulkan_effective_queue_flags(transfer_family->queueFlags),
      .timestamp_valid_bits = transfer_family->timestampValidBits,
      .affinity = 1ull << 1,
  };
  if (sparse_family_index != IREE_HAL_VULKAN_QUEUE_FAMILY_INVALID) {
    const VkQueueFamilyProperties* sparse_family =
        &snapshot->queue_families[sparse_family_index].queueFamilyProperties;
    out_queue_assignment->sparse_binding = (iree_hal_vulkan_queue_selection_t){
        .family_index = sparse_family_index,
        .queue_index = 0,
        .flags =
            iree_hal_vulkan_effective_queue_flags(sparse_family->queueFlags),
        .timestamp_valid_bits = sparse_family->timestampValidBits,
        .affinity = 0,
    };
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_device_plan_enable_extension_if_available(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    iree_hal_vulkan_device_extensions_t extension_bit,
    const char* extension_name, iree_hal_vulkan_device_plan_t* plan) {
  if (!iree_hal_vulkan_physical_device_has_extension(snapshot, extension_bit)) {
    return iree_ok_status();
  }
  if (plan->enabled_extension_count >=
      IREE_HAL_VULKAN_MAX_DEVICE_EXTENSION_NAMES) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "Vulkan device plan extension name storage is full");
  }
  plan->enabled_extension_names[plan->enabled_extension_count++] =
      extension_name;
  plan->enabled_extensions |= extension_bit;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_device_plan_select_extensions(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    iree_hal_vulkan_device_plan_t* plan) {
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_enable_extension_if_available(
          snapshot, IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_PORTABILITY_SUBSET,
          IREE_HAL_VULKAN_KHR_PORTABILITY_SUBSET_EXTENSION_NAME, plan));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_enable_extension_if_available(
          snapshot, IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_EXTERNAL_MEMORY,
          VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, plan));
#if defined(VK_USE_PLATFORM_WIN32_KHR)
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_enable_extension_if_available(
          snapshot, IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_EXTERNAL_MEMORY_WIN32,
          VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME, plan));
#else
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_enable_extension_if_available(
          snapshot, IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_EXTERNAL_MEMORY_FD,
          VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, plan));
#endif  // defined(VK_USE_PLATFORM_WIN32_KHR)
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_enable_extension_if_available(
          snapshot, IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_EXTERNAL_MEMORY_HOST,
          VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME, plan));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_enable_extension_if_available(
          snapshot, IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_CALIBRATED_TIMESTAMPS,
          VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME, plan));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_enable_extension_if_available(
          snapshot, IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_MEMORY_BUDGET,
          IREE_HAL_VULKAN_EXT_MEMORY_BUDGET_EXTENSION_NAME, plan));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_enable_extension_if_available(
          snapshot, IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_PUSH_DESCRIPTOR,
          VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, plan));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_enable_extension_if_available(
          snapshot, IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_COOPERATIVE_MATRIX,
          VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, plan));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_enable_extension_if_available(
          snapshot, IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_SHADER_BFLOAT16,
          IREE_HAL_VULKAN_KHR_SHADER_BFLOAT16_EXTENSION_NAME, plan));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_enable_extension_if_available(
          snapshot, IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_SHADER_ATOMIC_FLOAT,
          VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, plan));
  if (!iree_all_bits_set(
          plan->enabled_extensions,
          IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_SHADER_ATOMIC_FLOAT)) {
    return iree_ok_status();
  }
  return iree_hal_vulkan_device_plan_enable_extension_if_available(
      snapshot, IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_SHADER_ATOMIC_FLOAT2,
      VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME, plan);
}

static void iree_hal_vulkan_insert_queue_family_index(
    uint32_t family_index, uint32_t* family_indices,
    iree_host_size_t* family_count) {
  iree_host_size_t insert_index = 0;
  while (insert_index < *family_count &&
         family_indices[insert_index] < family_index) {
    ++insert_index;
  }
  if (insert_index < *family_count &&
      family_indices[insert_index] == family_index) {
    return;
  }
  for (iree_host_size_t i = *family_count; i > insert_index; --i) {
    family_indices[i] = family_indices[i - 1];
  }
  family_indices[insert_index] = family_index;
  ++*family_count;
}

static iree_status_t iree_hal_vulkan_device_plan_allocate_queue_inventory(
    iree_host_size_t family_count, iree_host_size_t queue_count,
    bool allocate_create_infos, iree_hal_vulkan_device_plan_t* plan) {
  if (family_count == 0 || family_count > IREE_HAL_MAX_QUEUE_FAMILIES) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "Vulkan logical device requires %" PRIhsz
        " queue families; HAL family affinities support [1, %" PRIhsz "]",
        family_count, IREE_HAL_MAX_QUEUE_FAMILIES);
  }

  iree_status_t status =
      iree_allocator_malloc_array(plan->host_allocator, family_count,
                                  sizeof(*plan->queue_inventory.families),
                                  (void**)&plan->queue_inventory.families);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        plan->host_allocator, queue_count,
        sizeof(*plan->queue_inventory.queue_indices),
        (void**)&plan->queue_inventory.queue_indices);
  }
  if (allocate_create_infos && iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(plan->host_allocator, queue_count,
                                         sizeof(*plan->queue_priorities),
                                         (void**)&plan->queue_priorities);
  }
  if (allocate_create_infos && iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(plan->host_allocator, family_count,
                                         sizeof(*plan->queue_create_infos),
                                         (void**)&plan->queue_create_infos);
  }
  if (iree_status_is_ok(status)) {
    plan->queue_inventory.family_count = family_count;
    plan->queue_inventory.queue_count = queue_count;
    plan->queue_create_info_count =
        allocate_create_infos ? (uint32_t)family_count : 0;
  } else {
    iree_hal_vulkan_device_plan_deinitialize(plan);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_device_plan_assign_queue_coordinate(
    const iree_hal_vulkan_queue_inventory_t* queue_inventory,
    iree_hal_vulkan_queue_selection_t* selection) {
  for (iree_host_size_t i = 0; i < queue_inventory->family_count; ++i) {
    const iree_hal_vulkan_queue_family_plan_t* family =
        &queue_inventory->families[i];
    if (family->native_family_index != selection->family_index) continue;
    for (uint32_t j = 0; j < family->queue_count; ++j) {
      if (queue_inventory->queue_indices[family->queue_offset + j] !=
          selection->queue_index) {
        continue;
      }
      selection->family_ordinal = (iree_hal_queue_family_ordinal_t)i;
      selection->queue_ordinal = j;
      return iree_ok_status();
    }
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "selected Vulkan queue %u in family %u is absent from the "
        "provisioned inventory",
        selection->queue_index, selection->family_index);
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "selected Vulkan queue family %u is absent from the provisioned "
      "inventory",
      selection->family_index);
}

static iree_status_t iree_hal_vulkan_device_plan_resolve_queue_coordinates(
    iree_hal_vulkan_device_plan_t* plan) {
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_device_plan_assign_queue_coordinate(
      &plan->queue_inventory, &plan->queue_assignment.compute));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_device_plan_assign_queue_coordinate(
      &plan->queue_inventory, &plan->queue_assignment.transfer));
  if (iree_hal_vulkan_queue_assignment_has_sparse_binding(
          &plan->queue_assignment)) {
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_device_plan_assign_queue_coordinate(
        &plan->queue_inventory, &plan->queue_assignment.sparse_binding));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_device_plan_initialize_owned_queues(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    iree_hal_vulkan_device_plan_t* plan) {
  uint32_t family_indices[IREE_HAL_MAX_QUEUE_FAMILIES];
  iree_host_size_t family_count = 0;
  iree_hal_vulkan_insert_queue_family_index(
      plan->queue_assignment.compute.family_index, family_indices,
      &family_count);
  iree_hal_vulkan_insert_queue_family_index(
      plan->queue_assignment.transfer.family_index, family_indices,
      &family_count);
  if (iree_hal_vulkan_queue_assignment_has_sparse_binding(
          &plan->queue_assignment)) {
    iree_hal_vulkan_insert_queue_family_index(
        plan->queue_assignment.sparse_binding.family_index, family_indices,
        &family_count);
  }

  iree_host_size_t queue_count = 0;
  for (iree_host_size_t i = 0; i < family_count; ++i) {
    const uint32_t native_queue_count =
        snapshot->queue_families[family_indices[i]]
            .queueFamilyProperties.queueCount;
    if (!iree_host_size_checked_add(queue_count, native_queue_count,
                                    &queue_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Vulkan queue inventory size overflow");
    }
  }
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_device_plan_allocate_queue_inventory(
      family_count, queue_count, /*allocate_create_infos=*/true, plan));

  iree_host_size_t queue_offset = 0;
  for (iree_host_size_t i = 0; i < family_count; ++i) {
    const uint32_t family_index = family_indices[i];
    const VkQueueFamilyProperties* properties =
        &snapshot->queue_families[family_index].queueFamilyProperties;
    plan->queue_inventory.families[i] = (iree_hal_vulkan_queue_family_plan_t){
        .native_family_index = family_index,
        .flags = iree_hal_vulkan_effective_queue_flags(properties->queueFlags),
        .timestamp_valid_bits = properties->timestampValidBits,
        .queue_count = properties->queueCount,
        .queue_offset = queue_offset,
    };
    for (uint32_t queue_index = 0; queue_index < properties->queueCount;
         ++queue_index) {
      plan->queue_inventory.queue_indices[queue_offset + queue_index] =
          queue_index;
      plan->queue_priorities[queue_offset + queue_index] = 1.0f;
    }
    plan->queue_create_infos[i] = (VkDeviceQueueCreateInfo){
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = family_index,
        .queueCount = properties->queueCount,
        .pQueuePriorities = &plan->queue_priorities[queue_offset],
    };
    queue_offset += properties->queueCount;
  }
  iree_status_t status =
      iree_hal_vulkan_device_plan_resolve_queue_coordinates(plan);
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_device_plan_deinitialize(plan);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_device_plan_verify_device_options(
    const iree_hal_vulkan_device_options_t* device_options) {
  const iree_hal_vulkan_device_flags_t recognized_device_flags =
      IREE_HAL_VULKAN_DEVICE_FLAG_DEDICATED_COMPUTE_QUEUE;
  if (iree_any_bit_set(device_options->flags, ~recognized_device_flags)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unrecognized Vulkan device option flag bits 0x%08x",
        device_options->flags & ~recognized_device_flags);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_device_plan_verify_request_flags(
    iree_hal_vulkan_request_flags_t request_flags) {
  const iree_hal_vulkan_request_flags_t unknown_request_flags =
      request_flags & ~IREE_HAL_VULKAN_REQUEST_FLAG_ALL_RECOGNIZED;
  if (unknown_request_flags) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unrecognized Vulkan request flag bits 0x%08x",
                            unknown_request_flags);
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_vulkan_device_plan_select_reported_general_feature(
    iree_hal_vulkan_general_features_t required_features,
    iree_hal_vulkan_general_features_t feature_bit, bool feature_available,
    const char* vulkan_feature_name,
    iree_hal_vulkan_general_features_t* enabled_features) {
  if (iree_any_bit_set(required_features, feature_bit) && !feature_available) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "required Vulkan %s is not available",
                            vulkan_feature_name);
  }
  if (feature_available) {
    *enabled_features |= feature_bit;
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_vulkan_device_plan_verify_external_reported_general_feature(
    iree_hal_vulkan_general_features_t enabled_features,
    iree_hal_vulkan_general_features_t feature_bit, bool feature_available,
    const char* vulkan_feature_name) {
  if (!iree_all_bits_set(enabled_features, feature_bit) || feature_available) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "external Vulkan VkDevice enabled %s but the physical device did not "
      "report it",
      vulkan_feature_name);
}

#define IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_COUNT 14

// Features enabled through VK_EXT_shader_atomic_float.
static const iree_hal_vulkan_shader_atomic_features_t
    kShaderAtomicFloatFeatureBits =
        IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT32 |
        IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT32_ADD |
        IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT32 |
        IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT32_ADD |
        IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT64 |
        IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT64_ADD |
        IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT64 |
        IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT64_ADD;

// Features enabled through VK_EXT_shader_atomic_float2.
static const iree_hal_vulkan_shader_atomic_features_t
    kShaderAtomicFloat2FeatureBits =
        IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT16 |
        IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT16_ADD |
        IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT16 |
        IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT16_ADD;

// Maps a public HAL feature bit to its Vulkan feature-struct member.
typedef struct iree_hal_vulkan_shader_atomic_feature_row_t {
  // HAL feature bit reported in device specifications.
  iree_hal_vulkan_shader_atomic_features_t feature_bit;
  // Whether the physical device reports the corresponding Vulkan feature.
  bool available;
  // Vulkan feature-struct member name used in diagnostics.
  const char* vulkan_feature_name;
} iree_hal_vulkan_shader_atomic_feature_row_t;

// Complete set of shader atomic feature rows queried from a device snapshot.
typedef struct iree_hal_vulkan_shader_atomic_feature_rows_t {
  // One row for each atomic capability represented by the Vulkan HAL.
  iree_hal_vulkan_shader_atomic_feature_row_t
      values[IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_COUNT];
} iree_hal_vulkan_shader_atomic_feature_rows_t;

static iree_hal_vulkan_shader_atomic_feature_rows_t
iree_hal_vulkan_device_plan_query_shader_atomic_features(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot) {
  const bool shader_atomic_float_available =
      iree_hal_vulkan_physical_device_has_extension(
          snapshot, IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_SHADER_ATOMIC_FLOAT);
  const bool shader_atomic_float2_available =
      shader_atomic_float_available &&
      iree_hal_vulkan_physical_device_has_extension(
          snapshot, IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_SHADER_ATOMIC_FLOAT2);
  return (iree_hal_vulkan_shader_atomic_feature_rows_t){
      .values =
          {
              {IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_INT64,
               snapshot->features12.shaderBufferInt64Atomics,
               "shaderBufferInt64Atomics"},
              {IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_INT64,
               snapshot->features12.shaderSharedInt64Atomics,
               "shaderSharedInt64Atomics"},
              {IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT16,
               shader_atomic_float2_available &&
                   snapshot->shader_atomic_float2_features
                       .shaderBufferFloat16Atomics,
               "shaderBufferFloat16Atomics"},
              {IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT16_ADD,
               shader_atomic_float2_available &&
                   snapshot->shader_atomic_float2_features
                       .shaderBufferFloat16AtomicAdd,
               "shaderBufferFloat16AtomicAdd"},
              {IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT16,
               shader_atomic_float2_available &&
                   snapshot->shader_atomic_float2_features
                       .shaderSharedFloat16Atomics,
               "shaderSharedFloat16Atomics"},
              {IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT16_ADD,
               shader_atomic_float2_available &&
                   snapshot->shader_atomic_float2_features
                       .shaderSharedFloat16AtomicAdd,
               "shaderSharedFloat16AtomicAdd"},
              {IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT32,
               shader_atomic_float_available &&
                   snapshot->shader_atomic_float_features
                       .shaderBufferFloat32Atomics,
               "shaderBufferFloat32Atomics"},
              {IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT32_ADD,
               shader_atomic_float_available &&
                   snapshot->shader_atomic_float_features
                       .shaderBufferFloat32AtomicAdd,
               "shaderBufferFloat32AtomicAdd"},
              {IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT32,
               shader_atomic_float_available &&
                   snapshot->shader_atomic_float_features
                       .shaderSharedFloat32Atomics,
               "shaderSharedFloat32Atomics"},
              {IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT32_ADD,
               shader_atomic_float_available &&
                   snapshot->shader_atomic_float_features
                       .shaderSharedFloat32AtomicAdd,
               "shaderSharedFloat32AtomicAdd"},
              {IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT64,
               shader_atomic_float_available &&
                   snapshot->shader_atomic_float_features
                       .shaderBufferFloat64Atomics,
               "shaderBufferFloat64Atomics"},
              {IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT64_ADD,
               shader_atomic_float_available &&
                   snapshot->shader_atomic_float_features
                       .shaderBufferFloat64AtomicAdd,
               "shaderBufferFloat64AtomicAdd"},
              {IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT64,
               shader_atomic_float_available &&
                   snapshot->shader_atomic_float_features
                       .shaderSharedFloat64Atomics,
               "shaderSharedFloat64Atomics"},
              {IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT64_ADD,
               shader_atomic_float_available &&
                   snapshot->shader_atomic_float_features
                       .shaderSharedFloat64AtomicAdd,
               "shaderSharedFloat64AtomicAdd"},
          },
  };
}

static iree_status_t iree_hal_vulkan_device_plan_select_shader_atomic_features(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    iree_hal_vulkan_shader_atomic_features_t required_features,
    iree_hal_vulkan_shader_atomic_features_t* enabled_features) {
  const iree_hal_vulkan_shader_atomic_feature_rows_t rows =
      iree_hal_vulkan_device_plan_query_shader_atomic_features(snapshot);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(rows.values); ++i) {
    const iree_hal_vulkan_shader_atomic_feature_row_t* row = &rows.values[i];
    if (iree_any_bit_set(required_features, row->feature_bit) &&
        !row->available) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "required Vulkan %s is not available",
                              row->vulkan_feature_name);
    }
    if (row->available) {
      *enabled_features |= row->feature_bit;
    }
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_vulkan_device_plan_verify_external_shader_atomic_features(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    iree_hal_vulkan_shader_atomic_features_t enabled_features) {
  const iree_hal_vulkan_shader_atomic_feature_rows_t rows =
      iree_hal_vulkan_device_plan_query_shader_atomic_features(snapshot);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(rows.values); ++i) {
    const iree_hal_vulkan_shader_atomic_feature_row_t* row = &rows.values[i];
    if (iree_all_bits_set(enabled_features, row->feature_bit) &&
        !row->available) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "external Vulkan VkDevice enabled %s but the physical device did "
          "not report it",
          row->vulkan_feature_name);
    }
  }
  return iree_ok_status();
}

static void iree_hal_vulkan_device_plan_initialize_shader_atomic_features(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    iree_hal_vulkan_device_plan_t* out_plan) {
  out_plan->enabled_shader_atomic_float_features =
      (VkPhysicalDeviceShaderAtomicFloatFeaturesEXT){
          .sType =
              VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
          .shaderBufferFloat32Atomics =
              snapshot->shader_atomic_float_features.shaderBufferFloat32Atomics,
          .shaderBufferFloat32AtomicAdd = snapshot->shader_atomic_float_features
                                              .shaderBufferFloat32AtomicAdd,
          .shaderBufferFloat64Atomics =
              snapshot->shader_atomic_float_features.shaderBufferFloat64Atomics,
          .shaderBufferFloat64AtomicAdd = snapshot->shader_atomic_float_features
                                              .shaderBufferFloat64AtomicAdd,
          .shaderSharedFloat32Atomics =
              snapshot->shader_atomic_float_features.shaderSharedFloat32Atomics,
          .shaderSharedFloat32AtomicAdd = snapshot->shader_atomic_float_features
                                              .shaderSharedFloat32AtomicAdd,
          .shaderSharedFloat64Atomics =
              snapshot->shader_atomic_float_features.shaderSharedFloat64Atomics,
          .shaderSharedFloat64AtomicAdd = snapshot->shader_atomic_float_features
                                              .shaderSharedFloat64AtomicAdd,
      };
  out_plan->enabled_shader_atomic_float2_features =
      (VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT){
          .sType =
              VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT,
          .shaderBufferFloat16Atomics = snapshot->shader_atomic_float2_features
                                            .shaderBufferFloat16Atomics,
          .shaderBufferFloat16AtomicAdd =
              snapshot->shader_atomic_float2_features
                  .shaderBufferFloat16AtomicAdd,
          .shaderSharedFloat16Atomics = snapshot->shader_atomic_float2_features
                                            .shaderSharedFloat16Atomics,
          .shaderSharedFloat16AtomicAdd =
              snapshot->shader_atomic_float2_features
                  .shaderSharedFloat16AtomicAdd,
      };
}

iree_status_t iree_hal_vulkan_device_plan_initialize_for_create(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    const iree_hal_vulkan_device_options_t* device_options,
    iree_hal_vulkan_request_flags_t request_flags,
    iree_hal_vulkan_features_t required_features,
    iree_allocator_t host_allocator, iree_hal_vulkan_device_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(snapshot);
  IREE_ASSERT_ARGUMENT(device_options);
  IREE_ASSERT_ARGUMENT(out_plan);
  iree_hal_vulkan_device_plan_initialize_empty(host_allocator, out_plan);

  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_verify_device_options(device_options));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_verify_request_flags(request_flags));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_features_validate(required_features));
  if (!iree_hal_vulkan_physical_device_supports_baseline(snapshot)) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "Vulkan physical device does not satisfy the Vulkan 1.3 baseline");
  }
  out_plan->request_flags = request_flags;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_select_queue_assignment(
      snapshot, device_options, &out_plan->queue_assignment));
  if (!iree_any_bit_set(required_features.general,
                        IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_BINDING)) {
    out_plan->queue_assignment.sparse_binding.family_index =
        IREE_HAL_VULKAN_QUEUE_FAMILY_INVALID;
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_select_extensions(snapshot, out_plan));

  out_plan->enabled_features13 = (VkPhysicalDeviceVulkan13Features){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .subgroupSizeControl = snapshot->features13.subgroupSizeControl,
      .computeFullSubgroups = snapshot->features13.computeFullSubgroups,
      .synchronization2 = VK_TRUE,
      .shaderIntegerDotProduct = snapshot->features13.shaderIntegerDotProduct,
  };
  out_plan->enabled_features12 = (VkPhysicalDeviceVulkan12Features){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &out_plan->enabled_features13,
      .storageBuffer8BitAccess = snapshot->features12.storageBuffer8BitAccess,
      .uniformAndStorageBuffer8BitAccess =
          snapshot->features12.uniformAndStorageBuffer8BitAccess,
      .storagePushConstant8 = snapshot->features12.storagePushConstant8,
      .shaderBufferInt64Atomics = snapshot->features12.shaderBufferInt64Atomics,
      .shaderSharedInt64Atomics = snapshot->features12.shaderSharedInt64Atomics,
      .shaderFloat16 = snapshot->features12.shaderFloat16,
      .shaderInt8 = snapshot->features12.shaderInt8,
      .scalarBlockLayout = VK_TRUE,
      .timelineSemaphore = VK_TRUE,
      .bufferDeviceAddress = VK_TRUE,
      .vulkanMemoryModel = snapshot->features12.vulkanMemoryModel,
      .vulkanMemoryModelDeviceScope =
          snapshot->features12.vulkanMemoryModel &&
          snapshot->features12.vulkanMemoryModelDeviceScope,
  };
  out_plan->enabled_features11 = (VkPhysicalDeviceVulkan11Features){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
      .pNext = &out_plan->enabled_features12,
      .storageBuffer16BitAccess = snapshot->features11.storageBuffer16BitAccess,
      .uniformAndStorageBuffer16BitAccess =
          snapshot->features11.uniformAndStorageBuffer16BitAccess,
      .storagePushConstant16 = snapshot->features11.storagePushConstant16,
      .storageInputOutput16 = snapshot->features11.storageInputOutput16,
  };
  out_plan->enabled_cooperative_matrix_features =
      (VkPhysicalDeviceCooperativeMatrixFeaturesKHR){
          .sType =
              VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
      };
  out_plan->enabled_shader_bfloat16_features =
      (VkPhysicalDeviceShaderBfloat16FeaturesKHR){
          .sType =
              VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
      };
  iree_hal_vulkan_device_plan_initialize_shader_atomic_features(snapshot,
                                                                out_plan);
  out_plan->enabled_features2 = (VkPhysicalDeviceFeatures2){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &out_plan->enabled_features11,
      .features =
          {
              .robustBufferAccess = VK_FALSE,
              .shaderFloat64 = snapshot->features2.features.shaderFloat64,
              .shaderInt64 = snapshot->features2.features.shaderInt64,
              .shaderInt16 = snapshot->features2.features.shaderInt16,
              .sparseBinding = VK_FALSE,
              .sparseResidencyBuffer = VK_FALSE,
              .sparseResidencyAliased = VK_FALSE,
          },
  };

  out_plan->enabled_features = (iree_hal_vulkan_features_t){
      .general = IREE_HAL_VULKAN_FEATURE_REQUIRED_BASELINE,
  };
  if (iree_any_bit_set(required_features.general,
                       IREE_HAL_VULKAN_FEATURE_ENABLE_ROBUST_BUFFER_ACCESS)) {
    if (!snapshot->features2.features.robustBufferAccess) {
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "required Vulkan robustBufferAccess is not available");
    }
    out_plan->enabled_features2.features.robustBufferAccess = VK_TRUE;
    out_plan->enabled_features.general |=
        IREE_HAL_VULKAN_FEATURE_ENABLE_ROBUST_BUFFER_ACCESS;
  }
  if (iree_any_bit_set(required_features.general,
                       IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_BINDING)) {
    if (!snapshot->features2.features.sparseBinding) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "required Vulkan sparseBinding is not available");
    }
    if (!iree_hal_vulkan_queue_assignment_has_sparse_binding(
            &out_plan->queue_assignment)) {
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "required Vulkan sparseBinding but no queue family reports "
          "VK_QUEUE_SPARSE_BINDING_BIT");
    }
    out_plan->enabled_features2.features.sparseBinding = VK_TRUE;
    out_plan->enabled_features.general |=
        IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_BINDING;
  }
  if (iree_all_bits_set(
          required_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_RESIDENCY_ALIASED)) {
    if (!snapshot->features2.features.sparseResidencyBuffer ||
        !snapshot->features2.features.sparseResidencyAliased) {
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "required Vulkan sparse residency aliasing is not available");
    }
    out_plan->enabled_features2.features.sparseResidencyBuffer = VK_TRUE;
    out_plan->enabled_features2.features.sparseResidencyAliased = VK_TRUE;
    out_plan->enabled_features.general |=
        IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_RESIDENCY_ALIASED;
  }
  if (iree_any_bit_set(required_features.general,
                       IREE_HAL_VULKAN_FEATURE_ENABLE_SUBGROUP_SIZE_CONTROL) &&
      !snapshot->features13.subgroupSizeControl) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "required Vulkan subgroupSizeControl is not available");
  }
  if (snapshot->features13.subgroupSizeControl) {
    out_plan->enabled_features.general |=
        IREE_HAL_VULKAN_FEATURE_ENABLE_SUBGROUP_SIZE_CONTROL;
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_select_reported_general_feature(
          required_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_8BIT_ACCESS,
          snapshot->features12.storageBuffer8BitAccess,
          "storageBuffer8BitAccess", &out_plan->enabled_features.general));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_select_reported_general_feature(
          required_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_16BIT_ACCESS,
          snapshot->features11.storageBuffer16BitAccess,
          "storageBuffer16BitAccess", &out_plan->enabled_features.general));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_select_reported_general_feature(
          required_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16,
          snapshot->features12.shaderFloat16, "shaderFloat16",
          &out_plan->enabled_features.general));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_select_reported_general_feature(
          required_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT64,
          snapshot->features2.features.shaderFloat64, "shaderFloat64",
          &out_plan->enabled_features.general));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_select_reported_general_feature(
          required_features.general, IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT8,
          snapshot->features12.shaderInt8, "shaderInt8",
          &out_plan->enabled_features.general));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_select_reported_general_feature(
          required_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT16,
          snapshot->features2.features.shaderInt16, "shaderInt16",
          &out_plan->enabled_features.general));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_select_reported_general_feature(
          required_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT64,
          snapshot->features2.features.shaderInt64, "shaderInt64",
          &out_plan->enabled_features.general));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_select_shader_atomic_features(
          snapshot, required_features.atomics,
          &out_plan->enabled_features.atomics));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_select_reported_general_feature(
          required_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INTEGER_DOT_PRODUCT,
          snapshot->features13.shaderIntegerDotProduct,
          "shaderIntegerDotProduct", &out_plan->enabled_features.general));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_select_reported_general_feature(
          required_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL,
          snapshot->features12.vulkanMemoryModel, "vulkanMemoryModel",
          &out_plan->enabled_features.general));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_select_reported_general_feature(
          required_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE,
          out_plan->enabled_features12.vulkanMemoryModelDeviceScope,
          "vulkanMemoryModelDeviceScope", &out_plan->enabled_features.general));
  const bool cooperative_matrix_available =
      iree_hal_vulkan_physical_device_has_extension(
          snapshot, IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_COOPERATIVE_MATRIX) &&
      snapshot->cooperative_matrix_features.cooperativeMatrix;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_select_reported_general_feature(
          required_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_COOPERATIVE_MATRIX,
          cooperative_matrix_available, "cooperativeMatrix",
          &out_plan->enabled_features.general));
  if (cooperative_matrix_available) {
    out_plan->enabled_cooperative_matrix_features.cooperativeMatrix = VK_TRUE;
  }
  const bool shader_bfloat16_extension_available =
      iree_hal_vulkan_physical_device_has_extension(
          snapshot, IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_SHADER_BFLOAT16);
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_select_reported_general_feature(
          required_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_TYPE,
          shader_bfloat16_extension_available &&
              snapshot->shader_bfloat16_features.shaderBFloat16Type,
          "shaderBFloat16Type", &out_plan->enabled_features.general));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_select_reported_general_feature(
          required_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_DOT_PRODUCT,
          shader_bfloat16_extension_available &&
              snapshot->shader_bfloat16_features.shaderBFloat16DotProduct,
          "shaderBFloat16DotProduct", &out_plan->enabled_features.general));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_select_reported_general_feature(
          required_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_COOPERATIVE_MATRIX,
          shader_bfloat16_extension_available &&
              snapshot->shader_bfloat16_features
                  .shaderBFloat16CooperativeMatrix,
          "shaderBFloat16CooperativeMatrix",
          &out_plan->enabled_features.general));
  out_plan->enabled_shader_bfloat16_features.shaderBFloat16Type =
      iree_all_bits_set(out_plan->enabled_features.general,
                        IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_TYPE)
          ? VK_TRUE
          : VK_FALSE;
  out_plan->enabled_shader_bfloat16_features.shaderBFloat16DotProduct =
      iree_all_bits_set(
          out_plan->enabled_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_DOT_PRODUCT)
          ? VK_TRUE
          : VK_FALSE;
  out_plan->enabled_shader_bfloat16_features.shaderBFloat16CooperativeMatrix =
      iree_all_bits_set(
          out_plan->enabled_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_COOPERATIVE_MATRIX)
          ? VK_TRUE
          : VK_FALSE;

  return iree_hal_vulkan_device_plan_initialize_owned_queues(snapshot,
                                                             out_plan);
}

static iree_status_t iree_hal_vulkan_verify_external_enabled_features(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    iree_hal_vulkan_features_t enabled_features) {
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_features_validate(enabled_features));

#define IREE_HAL_VULKAN_REQUIRE_ENABLED_FEATURE(bit, name)                \
  if (!iree_all_bits_set(enabled_features.general, (bit))) {              \
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,              \
                            "external Vulkan VkDevice did not enable %s", \
                            (name));                                      \
  }
  IREE_HAL_VULKAN_REQUIRE_ENABLED_FEATURE(
      IREE_HAL_VULKAN_FEATURE_ENABLE_TIMELINE_SEMAPHORES, "timelineSemaphore");
  IREE_HAL_VULKAN_REQUIRE_ENABLED_FEATURE(
      IREE_HAL_VULKAN_FEATURE_ENABLE_SYNCHRONIZATION2, "synchronization2");
  IREE_HAL_VULKAN_REQUIRE_ENABLED_FEATURE(
      IREE_HAL_VULKAN_FEATURE_ENABLE_SCALAR_BLOCK_LAYOUT, "scalarBlockLayout");
  IREE_HAL_VULKAN_REQUIRE_ENABLED_FEATURE(
      IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES,
      "bufferDeviceAddress");
#undef IREE_HAL_VULKAN_REQUIRE_ENABLED_FEATURE

  if (!iree_hal_vulkan_physical_device_supports_baseline(snapshot)) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "external Vulkan physical device does not satisfy the Vulkan 1.3 "
        "baseline");
  }
  if (iree_all_bits_set(enabled_features.general,
                        IREE_HAL_VULKAN_FEATURE_ENABLE_ROBUST_BUFFER_ACCESS) &&
      !snapshot->features2.features.robustBufferAccess) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "external Vulkan VkDevice enabled robustBufferAccess but the physical "
        "device did not report it");
  }
  if (iree_all_bits_set(
          enabled_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES) &&
      !snapshot->features12.bufferDeviceAddress) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "external Vulkan VkDevice enabled bufferDeviceAddress but the physical "
        "device did not report it");
  }
  if (iree_all_bits_set(enabled_features.general,
                        IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_BINDING) &&
      !snapshot->features2.features.sparseBinding) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "external Vulkan VkDevice enabled sparseBinding but the physical "
        "device did not report it");
  }
  if (iree_all_bits_set(
          enabled_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_RESIDENCY_ALIASED) &&
      (!snapshot->features2.features.sparseResidencyBuffer ||
       !snapshot->features2.features.sparseResidencyAliased)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "external Vulkan VkDevice enabled sparse residency aliasing but the "
        "physical device did not report it");
  }
  if (iree_all_bits_set(enabled_features.general,
                        IREE_HAL_VULKAN_FEATURE_ENABLE_SUBGROUP_SIZE_CONTROL) &&
      !snapshot->features13.subgroupSizeControl) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "external Vulkan VkDevice enabled subgroupSizeControl but the physical "
        "device did not report it");
  }
  if (iree_all_bits_set(enabled_features.general,
                        IREE_HAL_VULKAN_FEATURE_ENABLE_COOPERATIVE_MATRIX) &&
      (!iree_hal_vulkan_physical_device_has_extension(
           snapshot, IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_COOPERATIVE_MATRIX) ||
       !snapshot->cooperative_matrix_features.cooperativeMatrix)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "external Vulkan VkDevice enabled cooperativeMatrix but the physical "
        "device did not report it");
  }
  const bool shader_bfloat16_extension_available =
      iree_hal_vulkan_physical_device_has_extension(
          snapshot, IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_SHADER_BFLOAT16);
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_verify_external_reported_general_feature(
          enabled_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_TYPE,
          shader_bfloat16_extension_available &&
              snapshot->shader_bfloat16_features.shaderBFloat16Type,
          "shaderBFloat16Type"));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_verify_external_reported_general_feature(
          enabled_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_DOT_PRODUCT,
          shader_bfloat16_extension_available &&
              snapshot->shader_bfloat16_features.shaderBFloat16DotProduct,
          "shaderBFloat16DotProduct"));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_verify_external_reported_general_feature(
          enabled_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_COOPERATIVE_MATRIX,
          shader_bfloat16_extension_available &&
              snapshot->shader_bfloat16_features
                  .shaderBFloat16CooperativeMatrix,
          "shaderBFloat16CooperativeMatrix"));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_verify_external_reported_general_feature(
          enabled_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_8BIT_ACCESS,
          snapshot->features12.storageBuffer8BitAccess,
          "storageBuffer8BitAccess"));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_verify_external_reported_general_feature(
          enabled_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_16BIT_ACCESS,
          snapshot->features11.storageBuffer16BitAccess,
          "storageBuffer16BitAccess"));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_verify_external_reported_general_feature(
          enabled_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16,
          snapshot->features12.shaderFloat16, "shaderFloat16"));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_verify_external_reported_general_feature(
          enabled_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT64,
          snapshot->features2.features.shaderFloat64, "shaderFloat64"));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_verify_external_reported_general_feature(
          enabled_features.general, IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT8,
          snapshot->features12.shaderInt8, "shaderInt8"));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_verify_external_reported_general_feature(
          enabled_features.general, IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT16,
          snapshot->features2.features.shaderInt16, "shaderInt16"));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_verify_external_reported_general_feature(
          enabled_features.general, IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT64,
          snapshot->features2.features.shaderInt64, "shaderInt64"));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_verify_external_shader_atomic_features(
          snapshot, enabled_features.atomics));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_verify_external_reported_general_feature(
          enabled_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INTEGER_DOT_PRODUCT,
          snapshot->features13.shaderIntegerDotProduct,
          "shaderIntegerDotProduct"));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_plan_verify_external_reported_general_feature(
          enabled_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL,
          snapshot->features12.vulkanMemoryModel, "vulkanMemoryModel"));
  if (iree_all_bits_set(
          enabled_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE) &&
      (!iree_all_bits_set(enabled_features.general,
                          IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL) ||
       !snapshot->features12.vulkanMemoryModel ||
       !snapshot->features12.vulkanMemoryModelDeviceScope)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "external Vulkan VkDevice enabled vulkanMemoryModelDeviceScope without "
        "a reported Vulkan device-scope memory model");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_verify_external_enabled_extensions(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    iree_hal_vulkan_device_extensions_t enabled_extensions) {
  const iree_hal_vulkan_device_extensions_t unknown_extensions =
      enabled_extensions & ~IREE_HAL_VULKAN_DEVICE_EXTENSION_ALL_RECOGNIZED;
  if (unknown_extensions) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unrecognized Vulkan enabled extension bits "
                            "0x%08x",
                            unknown_extensions);
  }
  const iree_hal_vulkan_device_extensions_t unavailable_extensions =
      enabled_extensions & ~snapshot->available_extensions;
  if (unavailable_extensions) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "external Vulkan VkDevice enabled extension bits 0x%08x that the "
        "physical device did not report",
        unavailable_extensions);
  }
  if (iree_all_bits_set(
          enabled_extensions,
          IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_SHADER_ATOMIC_FLOAT2) &&
      !iree_all_bits_set(
          enabled_extensions,
          IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_SHADER_ATOMIC_FLOAT)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "external Vulkan VkDevice enabled VK_EXT_shader_atomic_float2 "
        "without its required VK_EXT_shader_atomic_float dependency");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_verify_external_device_contract(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    const iree_hal_vulkan_external_device_params_t* external_device_params) {
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_device_plan_verify_request_flags(
      external_device_params->request_flags));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_verify_external_enabled_features(
      snapshot, external_device_params->enabled_features));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_verify_external_enabled_extensions(
      snapshot, external_device_params->enabled_extensions));
  if (iree_all_bits_set(external_device_params->enabled_features.general,
                        IREE_HAL_VULKAN_FEATURE_ENABLE_COOPERATIVE_MATRIX) &&
      !iree_all_bits_set(
          external_device_params->enabled_extensions,
          IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_COOPERATIVE_MATRIX)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "external Vulkan VkDevice enabled cooperativeMatrix without reporting "
        "VK_KHR_cooperative_matrix as enabled");
  }
  const iree_hal_vulkan_general_features_t shader_bfloat16_features =
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_TYPE |
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_DOT_PRODUCT |
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_COOPERATIVE_MATRIX;
  if (iree_any_bit_set(external_device_params->enabled_features.general,
                       shader_bfloat16_features) &&
      !iree_all_bits_set(
          external_device_params->enabled_extensions,
          IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_SHADER_BFLOAT16)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "external Vulkan VkDevice enabled shaderBFloat16 features without "
        "reporting VK_KHR_shader_bfloat16 as enabled");
  }
  if (iree_any_bit_set(external_device_params->enabled_features.atomics,
                       kShaderAtomicFloatFeatureBits) &&
      !iree_all_bits_set(
          external_device_params->enabled_extensions,
          IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_SHADER_ATOMIC_FLOAT)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "external Vulkan VkDevice enabled shader atomic float features "
        "without reporting VK_EXT_shader_atomic_float as enabled");
  }
  if (iree_any_bit_set(external_device_params->enabled_features.atomics,
                       kShaderAtomicFloat2FeatureBits) &&
      !iree_all_bits_set(
          external_device_params->enabled_extensions,
          IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_SHADER_ATOMIC_FLOAT2)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "external Vulkan VkDevice enabled shader atomic float2 features "
        "without reporting VK_EXT_shader_atomic_float2 as enabled");
  }
  return iree_ok_status();
}

static uint64_t iree_hal_vulkan_queue_mask_for_count(uint32_t queue_count) {
  return queue_count >= 64 ? UINT64_MAX : ((1ull << queue_count) - 1ull);
}

static uint32_t iree_hal_vulkan_first_queue_index(uint64_t queue_indices) {
  for (uint32_t i = 0; i < 64; ++i) {
    if (iree_any_bit_set(queue_indices, 1ull << i)) return i;
  }
  return 0;
}

static iree_status_t iree_hal_vulkan_select_external_queue(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    const iree_hal_vulkan_queue_set_t* queue_set, VkQueueFlags required_flags,
    iree_hal_queue_affinity_t affinity, iree_string_view_t role,
    iree_hal_vulkan_queue_selection_t* out_queue) {
  memset(out_queue, 0, sizeof(*out_queue));
  if (!queue_set->queue_indices) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "external Vulkan %.*s queue set has no queue indices", (int)role.size,
        role.data);
  }
  if (queue_set->queue_family_index >= snapshot->queue_family_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "external Vulkan %.*s queue family %u is out of range; physical "
        "device has %u queue families",
        (int)role.size, role.data, queue_set->queue_family_index,
        snapshot->queue_family_count);
  }
  const VkQueueFamilyProperties* queue_family =
      &snapshot->queue_families[queue_set->queue_family_index]
           .queueFamilyProperties;
  if (!iree_all_bits_set(
          iree_hal_vulkan_effective_queue_flags(queue_family->queueFlags),
          required_flags)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "external Vulkan %.*s queue family %u flags 0x%08x do not include "
        "required flags 0x%08x",
        (int)role.size, role.data, queue_set->queue_family_index,
        queue_family->queueFlags, required_flags);
  }
  const uint64_t queue_mask =
      iree_hal_vulkan_queue_mask_for_count(queue_family->queueCount);
  if (iree_any_bit_set(queue_set->queue_indices, ~queue_mask)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "external Vulkan %.*s queue set 0x%016" PRIx64
                            " selects queues outside family %u queue count %u",
                            (int)role.size, role.data, queue_set->queue_indices,
                            queue_set->queue_family_index,
                            queue_family->queueCount);
  }

  *out_queue = (iree_hal_vulkan_queue_selection_t){
      .family_index = queue_set->queue_family_index,
      .queue_index =
          iree_hal_vulkan_first_queue_index(queue_set->queue_indices),
      .flags = iree_hal_vulkan_effective_queue_flags(queue_family->queueFlags),
      .timestamp_valid_bits = queue_family->timestampValidBits,
      .affinity = affinity,
  };
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_select_external_queue_assignment(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    const iree_hal_vulkan_external_device_params_t* external_device_params,
    iree_hal_vulkan_queue_assignment_t* out_queue_assignment) {
  memset(out_queue_assignment, 0, sizeof(*out_queue_assignment));
  out_queue_assignment->sparse_binding.family_index =
      IREE_HAL_VULKAN_QUEUE_FAMILY_INVALID;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_select_external_queue(
      snapshot, &external_device_params->compute_queue_set,
      VK_QUEUE_COMPUTE_BIT, 1ull << 0, IREE_SV("compute"),
      &out_queue_assignment->compute));

  if (!external_device_params->transfer_queue_set.queue_indices) {
    if (!iree_all_bits_set(out_queue_assignment->compute.flags,
                           VK_QUEUE_TRANSFER_BIT)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "external Vulkan VkDevice did not provide a transfer queue set and "
          "the selected compute queue family does not report transfer support");
    }
    out_queue_assignment->transfer = out_queue_assignment->compute;
    out_queue_assignment->transfer.affinity = 1ull << 1;
  } else {
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_select_external_queue(
        snapshot, &external_device_params->transfer_queue_set,
        VK_QUEUE_TRANSFER_BIT, 1ull << 1, IREE_SV("transfer"),
        &out_queue_assignment->transfer));
  }
  if (external_device_params->sparse_binding_queue_set.queue_indices) {
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_select_external_queue(
        snapshot, &external_device_params->sparse_binding_queue_set,
        VK_QUEUE_SPARSE_BINDING_BIT, 0, IREE_SV("sparse binding"),
        &out_queue_assignment->sparse_binding));
  } else if (iree_all_bits_set(out_queue_assignment->compute.flags,
                               VK_QUEUE_SPARSE_BINDING_BIT)) {
    out_queue_assignment->sparse_binding = out_queue_assignment->compute;
    out_queue_assignment->sparse_binding.affinity = 0;
  } else if (iree_all_bits_set(out_queue_assignment->transfer.flags,
                               VK_QUEUE_SPARSE_BINDING_BIT)) {
    out_queue_assignment->sparse_binding = out_queue_assignment->transfer;
    out_queue_assignment->sparse_binding.affinity = 0;
  }
  return iree_ok_status();
}

static void iree_hal_vulkan_merge_external_queue_set(
    const iree_hal_vulkan_queue_set_t* queue_set, uint32_t* family_indices,
    uint64_t* family_queue_masks, iree_host_size_t* family_count) {
  if (!queue_set->queue_indices) return;
  iree_host_size_t insert_index = 0;
  while (insert_index < *family_count &&
         family_indices[insert_index] < queue_set->queue_family_index) {
    ++insert_index;
  }
  if (insert_index < *family_count &&
      family_indices[insert_index] == queue_set->queue_family_index) {
    family_queue_masks[insert_index] |= queue_set->queue_indices;
    return;
  }
  for (iree_host_size_t i = *family_count; i > insert_index; --i) {
    family_indices[i] = family_indices[i - 1];
    family_queue_masks[i] = family_queue_masks[i - 1];
  }
  family_indices[insert_index] = queue_set->queue_family_index;
  family_queue_masks[insert_index] = queue_set->queue_indices;
  ++*family_count;
}

static iree_status_t iree_hal_vulkan_device_plan_initialize_wrapped_queues(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    const iree_hal_vulkan_external_device_params_t* external_device_params,
    iree_hal_vulkan_device_plan_t* plan) {
  uint32_t family_indices[IREE_HAL_MAX_QUEUE_FAMILIES];
  uint64_t family_queue_masks[IREE_HAL_MAX_QUEUE_FAMILIES];
  iree_host_size_t family_count = 0;
  iree_hal_vulkan_merge_external_queue_set(
      &external_device_params->compute_queue_set, family_indices,
      family_queue_masks, &family_count);
  iree_hal_vulkan_merge_external_queue_set(
      &external_device_params->transfer_queue_set, family_indices,
      family_queue_masks, &family_count);
  iree_hal_vulkan_merge_external_queue_set(
      &external_device_params->sparse_binding_queue_set, family_indices,
      family_queue_masks, &family_count);

  iree_host_size_t queue_count = 0;
  for (iree_host_size_t i = 0; i < family_count; ++i) {
    queue_count += iree_math_count_ones_u64(family_queue_masks[i]);
  }
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_device_plan_allocate_queue_inventory(
      family_count, queue_count, /*allocate_create_infos=*/false, plan));

  iree_host_size_t queue_offset = 0;
  for (iree_host_size_t i = 0; i < family_count; ++i) {
    const uint32_t family_index = family_indices[i];
    const VkQueueFamilyProperties* properties =
        &snapshot->queue_families[family_index].queueFamilyProperties;
    const uint32_t family_queue_count =
        iree_math_count_ones_u64(family_queue_masks[i]);
    plan->queue_inventory.families[i] = (iree_hal_vulkan_queue_family_plan_t){
        .native_family_index = family_index,
        .flags = iree_hal_vulkan_effective_queue_flags(properties->queueFlags),
        .timestamp_valid_bits = properties->timestampValidBits,
        .queue_count = family_queue_count,
        .queue_offset = queue_offset,
    };
    for (uint32_t queue_index = 0; queue_index < 64; ++queue_index) {
      if (!iree_any_bit_set(family_queue_masks[i], 1ull << queue_index)) {
        continue;
      }
      plan->queue_inventory.queue_indices[queue_offset++] = queue_index;
    }
  }
  iree_status_t status =
      iree_hal_vulkan_device_plan_resolve_queue_coordinates(plan);
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_device_plan_deinitialize(plan);
  }
  return status;
}

iree_status_t iree_hal_vulkan_device_plan_initialize_for_wrap(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    const iree_hal_vulkan_device_options_t* device_options,
    const iree_hal_vulkan_external_device_params_t* external_device_params,
    iree_allocator_t host_allocator, iree_hal_vulkan_device_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(snapshot);
  IREE_ASSERT_ARGUMENT(device_options);
  IREE_ASSERT_ARGUMENT(external_device_params);
  IREE_ASSERT_ARGUMENT(out_plan);
  iree_hal_vulkan_device_plan_initialize_empty(host_allocator, out_plan);

  if (device_options->flags != IREE_HAL_VULKAN_DEVICE_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "external Vulkan wrapping uses explicit queue sets and does not "
        "accept device option flags 0x%08x",
        device_options->flags);
  }
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_verify_external_device_contract(
      snapshot, external_device_params));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_select_external_queue_assignment(
      snapshot, external_device_params, &out_plan->queue_assignment));
  if (iree_all_bits_set(external_device_params->enabled_features.general,
                        IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_BINDING) &&
      !iree_hal_vulkan_queue_assignment_has_sparse_binding(
          &out_plan->queue_assignment)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "external Vulkan VkDevice enabled sparseBinding but no provided queue "
        "set reports VK_QUEUE_SPARSE_BINDING_BIT");
  }

  out_plan->enabled_features = external_device_params->enabled_features;
  out_plan->enabled_extensions = external_device_params->enabled_extensions;
  out_plan->request_flags = external_device_params->request_flags;
  return iree_hal_vulkan_device_plan_initialize_wrapped_queues(
      snapshot, external_device_params, out_plan);
}

static void iree_hal_vulkan_device_plan_refresh_feature_chain(
    iree_hal_vulkan_device_plan_t* plan) {
  void* feature_chain = NULL;
  if (plan->enabled_cooperative_matrix_features.cooperativeMatrix) {
    plan->enabled_cooperative_matrix_features.pNext = feature_chain;
    feature_chain = &plan->enabled_cooperative_matrix_features;
  }
  if (plan->enabled_shader_bfloat16_features.shaderBFloat16Type ||
      plan->enabled_shader_bfloat16_features.shaderBFloat16DotProduct ||
      plan->enabled_shader_bfloat16_features.shaderBFloat16CooperativeMatrix) {
    plan->enabled_shader_bfloat16_features.pNext = feature_chain;
    feature_chain = &plan->enabled_shader_bfloat16_features;
  }
  if (iree_any_bit_set(plan->enabled_features.atomics,
                       kShaderAtomicFloat2FeatureBits)) {
    plan->enabled_shader_atomic_float2_features.pNext = feature_chain;
    feature_chain = &plan->enabled_shader_atomic_float2_features;
  }
  if (iree_any_bit_set(plan->enabled_features.atomics,
                       kShaderAtomicFloatFeatureBits)) {
    plan->enabled_shader_atomic_float_features.pNext = feature_chain;
    feature_chain = &plan->enabled_shader_atomic_float_features;
  }
  plan->enabled_features13.pNext = feature_chain;
  plan->enabled_features12.pNext = &plan->enabled_features13;
  plan->enabled_features11.pNext = &plan->enabled_features12;
  plan->enabled_features2.pNext = &plan->enabled_features11;
}

void iree_hal_vulkan_device_plan_make_create_info(
    iree_hal_vulkan_device_plan_t* plan, VkDeviceCreateInfo* out_create_info) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(out_create_info);

  iree_hal_vulkan_device_plan_refresh_feature_chain(plan);
  for (uint32_t i = 0; i < plan->queue_create_info_count; ++i) {
    plan->queue_create_infos[i].pQueuePriorities =
        &plan->queue_priorities[plan->queue_inventory.families[i].queue_offset];
  }

  *out_create_info = (VkDeviceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &plan->enabled_features2,
      .queueCreateInfoCount = plan->queue_create_info_count,
      .pQueueCreateInfos = plan->queue_create_infos,
      .enabledExtensionCount = plan->enabled_extension_count,
      .ppEnabledExtensionNames = plan->enabled_extension_names,
  };
}
