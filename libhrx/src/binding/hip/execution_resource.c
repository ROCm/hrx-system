// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/execution_resource.h"

#include <string.h>

#include "common/device.h"

// "HRXR" identifies an HRX execution-resource token in the opaque HIP bytes.
#define IREE_HIP_EXECUTION_RESOURCE_TOKEN_MAGIC 0x52585248u
#define IREE_HIP_EXECUTION_RESOURCE_TOKEN_VERSION 1u

typedef enum iree_hip_execution_resource_token_kind_e {
  IREE_HIP_EXECUTION_RESOURCE_TOKEN_KIND_SM = 1,
} iree_hip_execution_resource_token_kind_t;

// Pointer-free token stored in hipDevResource::_internal_padding.
typedef struct iree_hip_execution_resource_token_t {
  // Format discriminator identifying an HRX execution-resource token.
  uint32_t magic;

  // Token format version.
  uint16_t version;

  // Resource payload kind from iree_hip_execution_resource_token_kind_t.
  uint16_t kind;

  // Streaming device ordinal that owned the resource when it was created.
  uint32_t device_ordinal;

  // Exact HIP SM resource flags copied into the public payload.
  uint32_t resource_flags;

  // Exact device-table incarnation that owns |set_id|.
  uint64_t table_incarnation;

  // Immutable execution-resource set identity within the device table.
  iree_hal_streaming_execution_resource_set_id_t set_id;
} iree_hip_execution_resource_token_t;

_Static_assert(
    sizeof(iree_hip_execution_resource_token_t) <=
        sizeof(((hipDevResource*)0)->_internal_padding),
    "execution-resource token must fit in the HIP opaque ABI storage");

static uint32_t iree_hip_execution_resource_gcd(uint32_t lhs, uint32_t rhs) {
  while (rhs != 0) {
    const uint32_t remainder = lhs % rhs;
    lhs = rhs;
    rhs = remainder;
  }
  return lhs;
}

// Derives the smallest valid raw-unit partition and common raw-unit alignment
// from one exact set. This remains generic over families whose selectable
// resources cover different numbers of raw execution units.
static bool iree_hip_execution_resource_measure_sm_constraints(
    const iree_hal_queue_family_spec_t* family_spec,
    const iree_hal_streaming_execution_resource_set_t* set,
    uint32_t* out_minimum_partition_size, uint32_t* out_alignment) {
  if (set->resources.count == 0 || !set->resources.ordinals) {
    return false;
  }

  uint32_t alignment = 0;
  for (iree_host_size_t i = 0; i < set->resources.count; ++i) {
    const iree_hal_queue_execution_resource_ordinal_t resource_ordinal =
        set->resources.ordinals[i];
    if (resource_ordinal >= family_spec->execution_resource_count) {
      return false;
    }
    const uint32_t execution_unit_count =
        family_spec->execution_resources[resource_ordinal].execution_unit_count;
    if (execution_unit_count == 0) {
      return false;
    }
    alignment = alignment == 0 ? execution_unit_count
                               : iree_hip_execution_resource_gcd(
                                     alignment, execution_unit_count);
  }

  uint64_t minimum_partition_size = 0;
  iree_host_size_t required_resource_count = 0;
  for (iree_host_size_t group_ordinal = 0;
       group_ordinal < family_spec->execution_resource_group_count;
       ++group_ordinal) {
    const uint32_t minimum_selected_resource_count =
        family_spec->execution_resource_groups[group_ordinal]
            .minimum_selected_resource_count;
    uint32_t previous_execution_unit_count = 0;
    iree_hal_queue_execution_resource_ordinal_t previous_resource_ordinal = 0;
    bool has_previous_resource = false;
    for (uint32_t selection_index = 0;
         selection_index < minimum_selected_resource_count; ++selection_index) {
      uint32_t selected_execution_unit_count = UINT32_MAX;
      iree_hal_queue_execution_resource_ordinal_t selected_resource_ordinal =
          UINT32_MAX;
      for (iree_host_size_t i = 0; i < set->resources.count; ++i) {
        const iree_hal_queue_execution_resource_ordinal_t resource_ordinal =
            set->resources.ordinals[i];
        const iree_hal_queue_execution_resource_spec_t* resource_spec =
            &family_spec->execution_resources[resource_ordinal];
        if (resource_spec->group_ordinal != group_ordinal) {
          continue;
        }
        if (has_previous_resource &&
            (resource_spec->execution_unit_count <
                 previous_execution_unit_count ||
             (resource_spec->execution_unit_count ==
                  previous_execution_unit_count &&
              resource_ordinal <= previous_resource_ordinal))) {
          continue;
        }
        if (resource_spec->execution_unit_count <
                selected_execution_unit_count ||
            (resource_spec->execution_unit_count ==
                 selected_execution_unit_count &&
             resource_ordinal < selected_resource_ordinal)) {
          selected_execution_unit_count = resource_spec->execution_unit_count;
          selected_resource_ordinal = resource_ordinal;
        }
      }
      if (selected_resource_ordinal == UINT32_MAX) {
        return false;
      }
      minimum_partition_size += selected_execution_unit_count;
      ++required_resource_count;
      previous_execution_unit_count = selected_execution_unit_count;
      previous_resource_ordinal = selected_resource_ordinal;
      has_previous_resource = true;
    }
  }

  if (required_resource_count == 0) {
    uint32_t smallest_execution_unit_count = UINT32_MAX;
    for (iree_host_size_t i = 0; i < set->resources.count; ++i) {
      const iree_hal_queue_execution_resource_ordinal_t resource_ordinal =
          set->resources.ordinals[i];
      smallest_execution_unit_count =
          iree_min(smallest_execution_unit_count,
                   family_spec->execution_resources[resource_ordinal]
                       .execution_unit_count);
    }
    minimum_partition_size = smallest_execution_unit_count;
  }
  if (minimum_partition_size == 0 || minimum_partition_size > UINT32_MAX ||
      alignment == 0) {
    return false;
  }

  *out_minimum_partition_size = (uint32_t)minimum_partition_size;
  *out_alignment = alignment;
  return true;
}

static hipError_t iree_hip_execution_resource_decode_sm_token(
    const hipDevResource* resource,
    iree_hip_execution_resource_token_t* out_token) {
  if (!resource) {
    return hipErrorInvalidValue;
  }
  if (resource->type != hipDevResourceTypeSm) {
    return hipErrorInvalidResourceType;
  }

  iree_hip_execution_resource_token_t token = {0};
  memcpy(&token, resource->_internal_padding, sizeof(token));
  if (token.magic != IREE_HIP_EXECUTION_RESOURCE_TOKEN_MAGIC ||
      token.version != IREE_HIP_EXECUTION_RESOURCE_TOKEN_VERSION ||
      token.kind != IREE_HIP_EXECUTION_RESOURCE_TOKEN_KIND_SM ||
      (token.resource_flags & ~hipDevSmResourceGroupBackfill) ||
      token.set_id == IREE_HAL_STREAMING_EXECUTION_RESOURCE_SET_ID_INVALID) {
    return hipErrorInvalidResourceConfiguration;
  }
  *out_token = token;
  return hipSuccess;
}

iree_status_t iree_hip_execution_resource_create_sm(
    iree_hal_streaming_device_t* device,
    const iree_hal_queue_family_t* queue_family,
    iree_hal_queue_execution_resource_list_t resources, unsigned int flags,
    hipDevResource* out_resource) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(queue_family);
  IREE_ASSERT_ARGUMENT(out_resource);

  if (IREE_UNLIKELY(flags & ~hipDevSmResourceGroupBackfill)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported HIP SM resource flags 0x%08X", flags);
  }
  if (IREE_UNLIKELY(device->ordinal > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HIP device ordinal exceeds token capacity");
  }

  const iree_hal_queue_family_spec_t* family_spec =
      iree_hal_queue_family_spec(queue_family);
  if (IREE_UNLIKELY(family_spec->execution_resource_count == 0)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "queue family exposes no selectable HIP execution resources");
  }

  iree_hal_streaming_execution_resource_set_id_t set_id =
      IREE_HAL_STREAMING_EXECUTION_RESOURCE_SET_ID_INVALID;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_execution_resource_table_intern(
      &device->execution_resource_table, queue_family, resources, &set_id));
  const iree_hal_streaming_execution_resource_set_t* set =
      iree_hal_streaming_execution_resource_table_resolve(
          &device->execution_resource_table, set_id);
  if (IREE_UNLIKELY(!set)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "newly interned HIP execution-resource set cannot be resolved");
  }

  uint32_t minimum_partition_size = 0;
  uint32_t alignment = 0;
  if (IREE_UNLIKELY(!iree_hip_execution_resource_measure_sm_constraints(
          family_spec, set, &minimum_partition_size, &alignment))) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "interned execution-resource set has inconsistent family constraints");
  }

  const iree_hip_execution_resource_token_t token = {
      .magic = IREE_HIP_EXECUTION_RESOURCE_TOKEN_MAGIC,
      .version = IREE_HIP_EXECUTION_RESOURCE_TOKEN_VERSION,
      .kind = IREE_HIP_EXECUTION_RESOURCE_TOKEN_KIND_SM,
      .device_ordinal = (uint32_t)device->ordinal,
      .resource_flags = flags,
      .table_incarnation =
          iree_hal_streaming_execution_resource_table_incarnation(
              &device->execution_resource_table),
      .set_id = set_id,
  };
  hipDevResource resource = {0};
  resource.type = hipDevResourceTypeSm;
  memcpy(resource._internal_padding, &token, sizeof(token));
  resource.sm.smCount = set->execution_unit_count;
  resource.sm.minSmPartitionSize = minimum_partition_size;
  resource.sm.smCoscheduledAlignment = alignment;
  resource.sm.flags = flags;
  resource.nextResource = NULL;
  *out_resource = resource;
  return iree_ok_status();
}

static bool iree_hip_execution_resource_cu_mask_bit_is_set(
    iree_host_size_t mask_word_count, const uint32_t* mask,
    uint32_t execution_unit_ordinal) {
  const iree_host_size_t word_ordinal = execution_unit_ordinal / 32u;
  return word_ordinal < mask_word_count &&
         (mask[word_ordinal] & (1u << (execution_unit_ordinal % 32u))) != 0;
}

iree_status_t iree_hip_execution_resource_intern_sm_cu_mask(
    iree_hal_streaming_device_t* device,
    const iree_hal_queue_family_t* queue_family,
    iree_host_size_t mask_word_count, const uint32_t* mask,
    const iree_hal_streaming_execution_resource_set_t** out_set) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(queue_family);
  IREE_ASSERT_ARGUMENT(mask);
  IREE_ASSERT_ARGUMENT(out_set);

  const iree_hal_queue_family_spec_t* family_spec =
      iree_hal_queue_family_spec(queue_family);
  if (IREE_UNLIKELY(family_spec->execution_resource_count == 0)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "queue family exposes no selectable HIP execution resources");
  }

  uint32_t selected_execution_unit_count = 0;
  for (uint32_t execution_unit_ordinal = 0;
       execution_unit_ordinal < family_spec->execution_unit_count;
       ++execution_unit_ordinal) {
    selected_execution_unit_count +=
        iree_hip_execution_resource_cu_mask_bit_is_set(mask_word_count, mask,
                                                       execution_unit_ordinal);
  }

  iree_hal_queue_execution_resource_ordinal_t* selected_resource_ordinals =
      NULL;
  iree_status_t status = iree_allocator_malloc_array(
      device->execution_resource_table.host_allocator,
      family_spec->execution_resource_count,
      sizeof(*selected_resource_ordinals), (void**)&selected_resource_ordinals);
  iree_host_size_t selected_resource_count = 0;
  uint32_t covered_execution_unit_count = 0;
  for (iree_host_size_t resource_ordinal = 0;
       resource_ordinal < family_spec->execution_resource_count &&
       iree_status_is_ok(status);
       ++resource_ordinal) {
    const iree_hal_queue_execution_resource_spec_t* resource_spec =
        &family_spec->execution_resources[resource_ordinal];
    uint32_t selected_resource_execution_unit_count = 0;
    for (uint32_t resource_execution_unit_ordinal = 0;
         resource_execution_unit_ordinal < resource_spec->execution_unit_count;
         ++resource_execution_unit_ordinal) {
      selected_resource_execution_unit_count +=
          iree_hip_execution_resource_cu_mask_bit_is_set(
              mask_word_count, mask,
              resource_spec->first_execution_unit_ordinal +
                  resource_execution_unit_ordinal);
    }
    covered_execution_unit_count += selected_resource_execution_unit_count;
    if (selected_resource_execution_unit_count == 0) {
      continue;
    }
    if (IREE_UNLIKELY(selected_resource_execution_unit_count !=
                      resource_spec->execution_unit_count)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HIP CU mask partially selects indivisible execution resource "
          "%" PRIhsz " covering raw units [%u, %u)",
          resource_ordinal, resource_spec->first_execution_unit_ordinal,
          resource_spec->first_execution_unit_ordinal +
              resource_spec->execution_unit_count);
      continue;
    }
    selected_resource_ordinals[selected_resource_count++] =
        (iree_hal_queue_execution_resource_ordinal_t)resource_ordinal;
  }
  if (iree_status_is_ok(status) &&
      IREE_UNLIKELY(covered_execution_unit_count !=
                    selected_execution_unit_count)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HIP CU mask selects raw execution units outside the queue family's "
        "canonical resources");
  }

  iree_hal_streaming_execution_resource_set_id_t set_id =
      IREE_HAL_STREAMING_EXECUTION_RESOURCE_SET_ID_INVALID;
  if (iree_status_is_ok(status)) {
    const iree_hal_queue_execution_resource_list_t selected_resources = {
        .count = selected_resource_count,
        .ordinals = selected_resource_count ? selected_resource_ordinals : NULL,
    };
    status = iree_hal_streaming_execution_resource_table_intern(
        &device->execution_resource_table, queue_family, selected_resources,
        &set_id);
  }
  const iree_hal_streaming_execution_resource_set_t* set = NULL;
  if (iree_status_is_ok(status)) {
    set = iree_hal_streaming_execution_resource_table_resolve(
        &device->execution_resource_table, set_id);
    if (IREE_UNLIKELY(!set)) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "interned HIP CU-mask execution-resource set cannot be resolved");
    }
  }
  iree_allocator_free(device->execution_resource_table.host_allocator,
                      selected_resource_ordinals);
  if (iree_status_is_ok(status)) {
    *out_set = set;
  }
  return status;
}

iree_status_t iree_hip_execution_resource_write_sm_cu_mask(
    const iree_hal_queue_family_t* queue_family,
    iree_hal_queue_execution_resource_list_t resources,
    iree_host_size_t mask_word_count, uint32_t* out_mask) {
  IREE_ASSERT_ARGUMENT(queue_family);
  IREE_ASSERT_ARGUMENT(out_mask);

  const iree_hal_queue_family_spec_t* family_spec =
      iree_hal_queue_family_spec(queue_family);
  if (IREE_UNLIKELY(family_spec->execution_resource_count == 0)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "queue family exposes no selectable HIP execution resources");
  }
  const iree_host_size_t required_mask_word_count =
      ((uint64_t)family_spec->execution_unit_count + 31u) / 32u;
  if (IREE_UNLIKELY(mask_word_count < required_mask_word_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP CU mask has %" PRIhsz
                            " words but queue family requires %" PRIhsz,
                            mask_word_count, required_mask_word_count);
  }
  if (IREE_UNLIKELY(resources.count && !resources.ordinals)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "execution-resource list has no ordinal storage");
  }

  const iree_host_size_t resource_count =
      resources.count ? resources.count : family_spec->execution_resource_count;
  for (iree_host_size_t i = 0; i < resource_count; ++i) {
    const iree_hal_queue_execution_resource_ordinal_t resource_ordinal =
        resources.count ? resources.ordinals[i]
                        : (iree_hal_queue_execution_resource_ordinal_t)i;
    if (IREE_UNLIKELY(resource_ordinal >=
                      family_spec->execution_resource_count)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "execution-resource ordinal %u is outside "
                              "family resource count %" PRIhsz,
                              resource_ordinal,
                              family_spec->execution_resource_count);
    }
    if (IREE_UNLIKELY(i > 0 && resources.count &&
                      resource_ordinal <= resources.ordinals[i - 1])) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "execution-resource ordinals must be sorted and unique");
    }
  }

  for (iree_host_size_t word_ordinal = 0; word_ordinal < mask_word_count;
       ++word_ordinal) {
    out_mask[word_ordinal] = 0;
  }
  for (iree_host_size_t i = 0; i < resource_count; ++i) {
    const iree_hal_queue_execution_resource_ordinal_t resource_ordinal =
        resources.count ? resources.ordinals[i]
                        : (iree_hal_queue_execution_resource_ordinal_t)i;
    const iree_hal_queue_execution_resource_spec_t* resource_spec =
        &family_spec->execution_resources[resource_ordinal];
    for (uint32_t resource_execution_unit_ordinal = 0;
         resource_execution_unit_ordinal < resource_spec->execution_unit_count;
         ++resource_execution_unit_ordinal) {
      const uint32_t execution_unit_ordinal =
          resource_spec->first_execution_unit_ordinal +
          resource_execution_unit_ordinal;
      out_mask[execution_unit_ordinal / 32u] |=
          1u << (execution_unit_ordinal % 32u);
    }
  }
  return iree_ok_status();
}

hipError_t iree_hip_execution_resource_resolve_sm_for_device(
    const hipDevResource* resource, iree_hal_streaming_device_t* device,
    const iree_hal_streaming_execution_resource_set_t** out_set) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_set);

  iree_hip_execution_resource_token_t token = {0};
  hipError_t result =
      iree_hip_execution_resource_decode_sm_token(resource, &token);
  if (result != hipSuccess) {
    return result;
  }
  if (token.device_ordinal != device->ordinal ||
      token.table_incarnation !=
          iree_hal_streaming_execution_resource_table_incarnation(
              &device->execution_resource_table)) {
    return hipErrorInvalidResourceConfiguration;
  }

  const iree_hal_streaming_execution_resource_set_t* set =
      iree_hal_streaming_execution_resource_table_resolve(
          &device->execution_resource_table, token.set_id);
  if (!set) {
    return hipErrorInvalidResourceConfiguration;
  }
  const iree_hal_queue_family_t* queue_family = iree_hal_device_queue_family(
      device->hal_device, set->queue_family_ordinal);
  if (!queue_family) {
    return hipErrorInvalidResourceConfiguration;
  }

  uint32_t minimum_partition_size = 0;
  uint32_t alignment = 0;
  if (!iree_hip_execution_resource_measure_sm_constraints(
          iree_hal_queue_family_spec(queue_family), set,
          &minimum_partition_size, &alignment) ||
      resource->sm.smCount != set->execution_unit_count ||
      resource->sm.minSmPartitionSize != minimum_partition_size ||
      resource->sm.smCoscheduledAlignment != alignment ||
      resource->sm.flags != token.resource_flags) {
    return hipErrorInvalidResourceConfiguration;
  }

  *out_set = set;
  return hipSuccess;
}

hipError_t iree_hip_execution_resource_resolve_sm(
    const hipDevResource* resource, iree_hal_streaming_device_t** out_device,
    const iree_hal_streaming_execution_resource_set_t** out_set) {
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_ASSERT_ARGUMENT(out_set);

  iree_hip_execution_resource_token_t token = {0};
  hipError_t result =
      iree_hip_execution_resource_decode_sm_token(resource, &token);
  if (result != hipSuccess) {
    return result;
  }
  iree_hal_streaming_device_t* device =
      iree_hal_streaming_device_entry(token.device_ordinal);
  if (!device) {
    return hipErrorInvalidResourceConfiguration;
  }

  const iree_hal_streaming_execution_resource_set_t* set = NULL;
  result =
      iree_hip_execution_resource_resolve_sm_for_device(resource, device, &set);
  if (result != hipSuccess) {
    return result;
  }
  *out_device = device;
  *out_set = set;
  return hipSuccess;
}

static hipError_t iree_hip_execution_resource_consume_status(
    iree_status_t status) {
  const iree_status_code_t status_code = iree_status_code(status);
  iree_status_free(status);
  switch (status_code) {
    case IREE_STATUS_RESOURCE_EXHAUSTED:
      return hipErrorOutOfMemory;
    case IREE_STATUS_INVALID_ARGUMENT:
    case IREE_STATUS_OUT_OF_RANGE:
      return hipErrorInvalidResourceConfiguration;
    case IREE_STATUS_UNIMPLEMENTED:
      return hipErrorNotSupported;
    default:
      return hipErrorUnknown;
  }
}

static iree_host_size_t iree_hip_execution_resource_count_group_resources(
    const iree_hal_queue_family_spec_t* family_spec,
    const iree_hal_streaming_execution_resource_set_t* set,
    iree_host_size_t group_ordinal) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < set->resources.count; ++i) {
    const iree_hal_queue_execution_resource_ordinal_t resource_ordinal =
        set->resources.ordinals[i];
    count += family_spec->execution_resources[resource_ordinal].group_ordinal ==
             group_ordinal;
  }
  return count;
}

// Tracks one input resource while planning disjoint output partitions.
typedef enum iree_hip_execution_resource_plan_state_e {
  // The resource has not been assigned to any partition.
  IREE_HIP_EXECUTION_RESOURCE_PLAN_STATE_AVAILABLE = 0,
  // The resource is assigned to the partition currently being planned.
  IREE_HIP_EXECUTION_RESOURCE_PLAN_STATE_SELECTED = 1,
  // The resource has been emitted into a completed partition.
  IREE_HIP_EXECUTION_RESOURCE_PLAN_STATE_CONSUMED = 2,
} iree_hip_execution_resource_plan_state_t;

// Exact partition plan over the canonical ordinals of one input resource.
typedef struct iree_hip_execution_resource_partition_plan_t {
  // Host allocator owning |ordinals|.
  iree_allocator_t host_allocator;
  // Packed partition ordinals followed by all remainder ordinals.
  iree_hal_queue_execution_resource_ordinal_t* ordinals;
  // Number of ordinals in the remainder suffix.
  iree_host_size_t remainder_resource_count;
} iree_hip_execution_resource_partition_plan_t;

static void iree_hip_execution_resource_partition_plan_deinitialize(
    iree_hip_execution_resource_partition_plan_t* plan) {
  iree_allocator_free(plan->host_allocator, plan->ordinals);
  *plan = (iree_hip_execution_resource_partition_plan_t){0};
}

// Plans variable-size exact partitions while preserving every queue-family
// resource-group constraint. |partition_resource_counts| contains selectable
// HAL resource counts rather than raw HIP SM counts.
static hipError_t iree_hip_execution_resource_plan_partitions(
    iree_allocator_t host_allocator,
    const iree_hal_queue_family_spec_t* family_spec,
    const iree_hal_streaming_execution_resource_set_t* input_set,
    iree_host_size_t partition_count,
    const iree_host_size_t* partition_resource_counts, bool preserve_remainder,
    iree_hip_execution_resource_partition_plan_t* out_plan) {
  const iree_host_size_t input_resource_count = input_set->resources.count;

  iree_host_size_t assigned_resource_count = 0;
  iree_host_size_t active_partition_count = 0;
  iree_host_size_t minimum_resources_per_partition = 0;
  for (iree_host_size_t group_ordinal = 0;
       group_ordinal < family_spec->execution_resource_group_count;
       ++group_ordinal) {
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            minimum_resources_per_partition,
            family_spec->execution_resource_groups[group_ordinal]
                .minimum_selected_resource_count,
            &minimum_resources_per_partition))) {
      return hipErrorInvalidResourceConfiguration;
    }
  }
  minimum_resources_per_partition =
      iree_max(minimum_resources_per_partition, 1u);
  for (iree_host_size_t i = 0; i < partition_count; ++i) {
    if (partition_resource_counts[i] == 0) {
      continue;
    }
    if (partition_resource_counts[i] < minimum_resources_per_partition ||
        IREE_UNLIKELY(!iree_host_size_checked_add(assigned_resource_count,
                                                  partition_resource_counts[i],
                                                  &assigned_resource_count)) ||
        IREE_UNLIKELY(!iree_host_size_checked_add(active_partition_count, 1,
                                                  &active_partition_count))) {
      return hipErrorInvalidResourceConfiguration;
    }
  }
  if (assigned_resource_count > input_resource_count) {
    return hipErrorInvalidResourceConfiguration;
  }

  const iree_host_size_t remainder_resource_count =
      input_resource_count - assigned_resource_count;
  const bool require_remainder_constraints =
      preserve_remainder && remainder_resource_count != 0;
  if (require_remainder_constraints &&
      remainder_resource_count < minimum_resources_per_partition) {
    return hipErrorInvalidResourceConfiguration;
  }

  iree_host_size_t constrained_partition_count = active_partition_count;
  if (require_remainder_constraints &&
      IREE_UNLIKELY(!iree_host_size_checked_add(
          constrained_partition_count, 1, &constrained_partition_count))) {
    return hipErrorInvalidResourceConfiguration;
  }
  for (iree_host_size_t group_ordinal = 0;
       group_ordinal < family_spec->execution_resource_group_count;
       ++group_ordinal) {
    iree_host_size_t required_resource_count = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            constrained_partition_count,
            family_spec->execution_resource_groups[group_ordinal]
                .minimum_selected_resource_count,
            &required_resource_count)) ||
        iree_hip_execution_resource_count_group_resources(
            family_spec, input_set, group_ordinal) < required_resource_count) {
      return hipErrorInvalidResourceConfiguration;
    }
  }

  iree_host_size_t* available_resource_counts_by_group = NULL;
  iree_hip_execution_resource_plan_state_t* resource_states = NULL;
  iree_hal_queue_execution_resource_ordinal_t* planned_ordinals = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, family_spec->execution_resource_group_count,
      sizeof(*available_resource_counts_by_group),
      (void**)&available_resource_counts_by_group);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, input_resource_count,
                                         sizeof(*resource_states),
                                         (void**)&resource_states);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, input_resource_count,
                                         sizeof(*planned_ordinals),
                                         (void**)&planned_ordinals);
  }
  hipError_t result = hipSuccess;
  if (!iree_status_is_ok(status)) {
    result = iree_hip_execution_resource_consume_status(status);
  }

  if (result == hipSuccess) {
    memset(resource_states, 0, input_resource_count * sizeof(*resource_states));
    for (iree_host_size_t group_ordinal = 0;
         group_ordinal < family_spec->execution_resource_group_count;
         ++group_ordinal) {
      available_resource_counts_by_group[group_ordinal] =
          iree_hip_execution_resource_count_group_resources(
              family_spec, input_set, group_ordinal);
    }
  }

  iree_host_size_t remaining_constrained_partition_count =
      constrained_partition_count;
  iree_host_size_t partition_offset = 0;
  for (iree_host_size_t output_index = 0;
       output_index < partition_count && result == hipSuccess; ++output_index) {
    const iree_host_size_t target_resource_count =
        partition_resource_counts[output_index];
    if (target_resource_count == 0) {
      continue;
    }
    --remaining_constrained_partition_count;
    iree_host_size_t selected_count = 0;
    for (iree_host_size_t group_ordinal = 0;
         group_ordinal < family_spec->execution_resource_group_count &&
         result == hipSuccess;
         ++group_ordinal) {
      const uint32_t minimum_selected_resource_count =
          family_spec->execution_resource_groups[group_ordinal]
              .minimum_selected_resource_count;
      for (uint32_t selection_index = 0;
           selection_index < minimum_selected_resource_count;
           ++selection_index) {
        iree_host_size_t selected_index = input_resource_count;
        for (iree_host_size_t i = 0; i < input_resource_count; ++i) {
          const iree_hal_queue_execution_resource_ordinal_t resource_ordinal =
              input_set->resources.ordinals[i];
          if (resource_states[i] ==
                  IREE_HIP_EXECUTION_RESOURCE_PLAN_STATE_AVAILABLE &&
              family_spec->execution_resources[resource_ordinal]
                      .group_ordinal == group_ordinal) {
            selected_index = i;
            break;
          }
        }
        if (selected_index == input_resource_count) {
          result = hipErrorInvalidResourceConfiguration;
          break;
        }
        resource_states[selected_index] =
            IREE_HIP_EXECUTION_RESOURCE_PLAN_STATE_SELECTED;
        --available_resource_counts_by_group[group_ordinal];
        ++selected_count;
      }
    }

    while (selected_count < target_resource_count && result == hipSuccess) {
      iree_host_size_t selected_index = input_resource_count;
      for (iree_host_size_t i = 0; i < input_resource_count; ++i) {
        if (resource_states[i] !=
            IREE_HIP_EXECUTION_RESOURCE_PLAN_STATE_AVAILABLE) {
          continue;
        }
        const iree_hal_queue_execution_resource_ordinal_t resource_ordinal =
            input_set->resources.ordinals[i];
        const iree_host_size_t group_ordinal =
            family_spec->execution_resources[resource_ordinal].group_ordinal;
        iree_host_size_t reserved_resource_count = 0;
        if (IREE_UNLIKELY(!iree_host_size_checked_mul(
                remaining_constrained_partition_count,
                family_spec->execution_resource_groups[group_ordinal]
                    .minimum_selected_resource_count,
                &reserved_resource_count))) {
          result = hipErrorInvalidResourceConfiguration;
          break;
        }
        if (available_resource_counts_by_group[group_ordinal] <=
            reserved_resource_count) {
          continue;
        }
        selected_index = i;
        break;
      }
      if (selected_index == input_resource_count) {
        result = hipErrorInvalidResourceConfiguration;
        break;
      }
      const iree_hal_queue_execution_resource_ordinal_t resource_ordinal =
          input_set->resources.ordinals[selected_index];
      resource_states[selected_index] =
          IREE_HIP_EXECUTION_RESOURCE_PLAN_STATE_SELECTED;
      --available_resource_counts_by_group
          [family_spec->execution_resources[resource_ordinal].group_ordinal];
      ++selected_count;
    }

    iree_host_size_t emitted_count = 0;
    for (iree_host_size_t i = 0;
         i < input_resource_count && result == hipSuccess; ++i) {
      if (resource_states[i] !=
          IREE_HIP_EXECUTION_RESOURCE_PLAN_STATE_SELECTED) {
        continue;
      }
      planned_ordinals[partition_offset + emitted_count++] =
          input_set->resources.ordinals[i];
      resource_states[i] = IREE_HIP_EXECUTION_RESOURCE_PLAN_STATE_CONSUMED;
    }
    if (emitted_count != target_resource_count) {
      result = hipErrorInvalidResourceConfiguration;
    }
    partition_offset += target_resource_count;
  }

  iree_host_size_t emitted_remainder_count = 0;
  if (result == hipSuccess) {
    for (iree_host_size_t i = 0; i < input_resource_count; ++i) {
      if (resource_states[i] !=
          IREE_HIP_EXECUTION_RESOURCE_PLAN_STATE_AVAILABLE) {
        continue;
      }
      planned_ordinals[partition_offset + emitted_remainder_count++] =
          input_set->resources.ordinals[i];
    }
    if (emitted_remainder_count != remainder_resource_count) {
      result = hipErrorInvalidResourceConfiguration;
    }
  }

  if (result == hipSuccess) {
    *out_plan = (iree_hip_execution_resource_partition_plan_t){
        .host_allocator = host_allocator,
        .ordinals = planned_ordinals,
        .remainder_resource_count = remainder_resource_count,
    };
    planned_ordinals = NULL;
  }
  iree_allocator_free(host_allocator, planned_ordinals);
  iree_allocator_free(host_allocator, resource_states);
  iree_allocator_free(host_allocator, available_resource_counts_by_group);
  return result;
}

static hipError_t iree_hip_execution_resource_uniform_width(
    const iree_hal_queue_family_spec_t* family_spec,
    const iree_hal_streaming_execution_resource_set_t* set,
    uint32_t* out_execution_units_per_resource) {
  if (set->resources.count == 0) {
    return hipErrorInvalidResourceConfiguration;
  }
  const iree_hal_queue_execution_resource_ordinal_t first_resource_ordinal =
      set->resources.ordinals[0];
  const uint32_t execution_units_per_resource =
      family_spec->execution_resources[first_resource_ordinal]
          .execution_unit_count;
  if (execution_units_per_resource == 0) {
    return hipErrorInvalidResourceConfiguration;
  }
  for (iree_host_size_t i = 1; i < set->resources.count; ++i) {
    const iree_hal_queue_execution_resource_ordinal_t resource_ordinal =
        set->resources.ordinals[i];
    if (family_spec->execution_resources[resource_ordinal]
            .execution_unit_count != execution_units_per_resource) {
      return hipErrorNotSupported;
    }
  }
  *out_execution_units_per_resource = execution_units_per_resource;
  return hipSuccess;
}

hipError_t iree_hip_execution_resource_split_sm_by_count(
    iree_hal_streaming_device_t* device,
    const iree_hal_streaming_execution_resource_set_t* input_set,
    const hipDevResource* input, unsigned int flags, unsigned int minimum_count,
    unsigned int* inout_group_count, hipDevResource* out_resources,
    hipDevResource* out_remainder) {
  const unsigned int known_flags = hipDevSmResourceSplitIgnoreSmCoscheduling |
                                   hipDevSmResourceSplitMaxPotentialClusterSize;
  if (flags & ~known_flags) {
    return hipErrorInvalidValue;
  }
  if (flags != 0) {
    // Ignoring coscheduling changes the selectable granule, while maximizing
    // cluster size requires topology not represented by the current HAL
    // family facts. Neither flag may degrade into a successful no-op.
    return hipErrorNotSupported;
  }

  if (minimum_count > input->sm.smCount) {
    return hipErrorInvalidValue;
  }

  hipError_t result = hipSuccess;

  const iree_hal_queue_family_t* queue_family = iree_hal_device_queue_family(
      device->hal_device, input_set->queue_family_ordinal);
  if (!queue_family) {
    return hipErrorInvalidResourceConfiguration;
  }
  const iree_hal_queue_family_spec_t* family_spec =
      iree_hal_queue_family_spec(queue_family);
  const iree_host_size_t input_resource_count = input_set->resources.count;

  // HIP measures equal partitions in raw execution units while HAL families
  // select indivisible resources. Without additional topology facts, only
  // uniform resource widths can guarantee equal exact partitions.
  uint32_t execution_units_per_resource = 0;
  result = iree_hip_execution_resource_uniform_width(
      family_spec, input_set, &execution_units_per_resource);
  if (result != hipSuccess) {
    return result;
  }

  // Bound the partition count by both total capacity and the resources each
  // family constraint group requires in every exact set.
  iree_host_size_t minimum_resources_per_partition = 0;
  iree_host_size_t possible_partition_count = input_resource_count;
  for (iree_host_size_t group_ordinal = 0;
       group_ordinal < family_spec->execution_resource_group_count;
       ++group_ordinal) {
    const iree_host_size_t selected_resource_count =
        iree_hip_execution_resource_count_group_resources(
            family_spec, input_set, group_ordinal);
    const uint32_t minimum_selected_resource_count =
        family_spec->execution_resource_groups[group_ordinal]
            .minimum_selected_resource_count;
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            minimum_resources_per_partition, minimum_selected_resource_count,
            &minimum_resources_per_partition))) {
      return hipErrorInvalidResourceConfiguration;
    }
    if (minimum_selected_resource_count != 0) {
      possible_partition_count =
          iree_min(possible_partition_count,
                   selected_resource_count / minimum_selected_resource_count);
    }
  }

  const uint64_t requested_execution_unit_count =
      iree_max((uint64_t)minimum_count, (uint64_t)input->sm.minSmPartitionSize);
  iree_host_size_t resources_per_partition =
      (iree_host_size_t)((requested_execution_unit_count +
                          execution_units_per_resource - 1) /
                         execution_units_per_resource);
  resources_per_partition =
      iree_max(resources_per_partition, minimum_resources_per_partition);
  if (resources_per_partition == 0 ||
      resources_per_partition > input_resource_count) {
    return hipErrorInvalidResourceConfiguration;
  }
  possible_partition_count = iree_min(
      possible_partition_count, input_resource_count / resources_per_partition);

  iree_host_size_t actual_partition_count =
      out_resources ? iree_min((iree_host_size_t)*inout_group_count,
                               possible_partition_count)
                    : possible_partition_count;
  if (out_remainder) {
    // A returned remainder is itself usable for queue acquisition. Reduce the
    // output count until every group minimum can also be retained there.
    while (actual_partition_count > 0) {
      const iree_host_size_t remainder_resource_count =
          input_resource_count -
          actual_partition_count * resources_per_partition;
      if (remainder_resource_count == 0) {
        break;
      }

      bool has_valid_remainder = true;
      for (iree_host_size_t group_ordinal = 0;
           group_ordinal < family_spec->execution_resource_group_count;
           ++group_ordinal) {
        const iree_host_size_t selected_resource_count =
            iree_hip_execution_resource_count_group_resources(
                family_spec, input_set, group_ordinal);
        const uint32_t minimum_selected_resource_count =
            family_spec->execution_resource_groups[group_ordinal]
                .minimum_selected_resource_count;
        iree_host_size_t remainder_partition_count = 0;
        iree_host_size_t required_resource_count = 0;
        if (IREE_UNLIKELY(!iree_host_size_checked_add(
                actual_partition_count, 1, &remainder_partition_count)) ||
            IREE_UNLIKELY(!iree_host_size_checked_mul(
                remainder_partition_count, minimum_selected_resource_count,
                &required_resource_count)) ||
            selected_resource_count < required_resource_count) {
          has_valid_remainder = false;
          break;
        }
      }
      if (has_valid_remainder) {
        break;
      }
      --actual_partition_count;
    }
  }
  if (!out_resources) {
    *inout_group_count = (unsigned int)actual_partition_count;
    return hipSuccess;
  }

  const iree_allocator_t host_allocator =
      device->execution_resource_table.host_allocator;
  iree_host_size_t* partition_resource_counts = NULL;
  hipDevResource* planned_partitions = NULL;
  iree_status_t status = iree_ok_status();
  if (actual_partition_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, actual_partition_count,
                                         sizeof(*partition_resource_counts),
                                         (void**)&partition_resource_counts);
  }
  if (iree_status_is_ok(status) && actual_partition_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, actual_partition_count,
                                         sizeof(*planned_partitions),
                                         (void**)&planned_partitions);
  }
  if (!iree_status_is_ok(status)) {
    result = iree_hip_execution_resource_consume_status(status);
  }

  if (result == hipSuccess) {
    for (iree_host_size_t i = 0; i < actual_partition_count; ++i) {
      partition_resource_counts[i] = resources_per_partition;
    }
  }

  iree_hip_execution_resource_partition_plan_t plan = {0};
  if (result == hipSuccess) {
    result = iree_hip_execution_resource_plan_partitions(
        host_allocator, family_spec, input_set, actual_partition_count,
        partition_resource_counts, out_remainder != NULL, &plan);
  }

  iree_host_size_t partition_offset = 0;
  for (iree_host_size_t i = 0;
       i < actual_partition_count && result == hipSuccess; ++i) {
    status = iree_hip_execution_resource_create_sm(
        device, queue_family,
        (iree_hal_queue_execution_resource_list_t){
            .count = resources_per_partition,
            .ordinals = &plan.ordinals[partition_offset],
        },
        hipDevSmResourceGroupDefault, &planned_partitions[i]);
    if (!iree_status_is_ok(status)) {
      result = iree_hip_execution_resource_consume_status(status);
    }
    partition_offset += resources_per_partition;
  }

  hipDevResource planned_remainder = {0};
  planned_remainder.type = hipDevResourceTypeInvalid;
  if (result == hipSuccess && out_remainder &&
      plan.remainder_resource_count != 0) {
    status = iree_hip_execution_resource_create_sm(
        device, queue_family,
        (iree_hal_queue_execution_resource_list_t){
            .count = plan.remainder_resource_count,
            .ordinals = &plan.ordinals[partition_offset],
        },
        hipDevSmResourceGroupDefault, &planned_remainder);
    if (!iree_status_is_ok(status)) {
      result = iree_hip_execution_resource_consume_status(status);
    }
  }

  if (result == hipSuccess) {
    if (actual_partition_count != 0) {
      memcpy(out_resources, planned_partitions,
             actual_partition_count * sizeof(*out_resources));
    }
    if (out_remainder) {
      *out_remainder = planned_remainder;
    }
    *inout_group_count = (unsigned int)actual_partition_count;
  }

  iree_hip_execution_resource_partition_plan_deinitialize(&plan);
  iree_allocator_free(host_allocator, planned_partitions);
  iree_allocator_free(host_allocator, partition_resource_counts);
  return result;
}

hipError_t iree_hip_execution_resource_split_sm(
    iree_hal_streaming_device_t* device,
    const iree_hal_streaming_execution_resource_set_t* input_set,
    const hipDevResource* input, unsigned int group_count, unsigned int flags,
    hipDevSmResourceGroupParams* group_parameters,
    hipDevResource* out_resources, hipDevResource* out_remainder) {
  if (group_count == 0 || !group_parameters) {
    return hipErrorInvalidValue;
  }
  if (flags != 0) {
    return hipErrorInvalidValue;
  }

  const iree_hal_queue_family_t* queue_family = iree_hal_device_queue_family(
      device->hal_device, input_set->queue_family_ordinal);
  if (!queue_family) {
    return hipErrorInvalidResourceConfiguration;
  }
  const iree_hal_queue_family_spec_t* family_spec =
      iree_hal_queue_family_spec(queue_family);

  // Structured HIP partitions are measured in raw execution units. Current
  // AMDGPU families expose uniform CU or WGP resources, which lets every
  // requested raw-unit count map to an exact canonical HAL resource count.
  uint32_t execution_units_per_resource = 0;
  hipError_t result = iree_hip_execution_resource_uniform_width(
      family_spec, input_set, &execution_units_per_resource);
  if (result != hipSuccess) {
    return result;
  }

  const iree_allocator_t host_allocator =
      device->execution_resource_table.host_allocator;
  hipDevSmResourceGroupParams* normalized_parameters = NULL;
  iree_host_size_t* partition_resource_counts = NULL;
  hipDevResource* planned_partitions = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, group_count, sizeof(*normalized_parameters),
      (void**)&normalized_parameters);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, group_count,
                                         sizeof(*partition_resource_counts),
                                         (void**)&partition_resource_counts);
  }
  if (iree_status_is_ok(status) && out_resources) {
    status = iree_allocator_malloc_array(host_allocator, group_count,
                                         sizeof(*planned_partitions),
                                         (void**)&planned_partitions);
  }
  if (!iree_status_is_ok(status)) {
    result = iree_hip_execution_resource_consume_status(status);
  }

  const iree_host_size_t input_resource_count = input_set->resources.count;
  iree_host_size_t remaining_resource_count = input_resource_count;
  for (iree_host_size_t i = 0; i < group_count && result == hipSuccess; ++i) {
    normalized_parameters[i] = group_parameters[i];
    hipDevSmResourceGroupParams* parameter = &normalized_parameters[i];
    if (parameter->flags & ~hipDevSmResourceGroupBackfill) {
      result = hipErrorInvalidValue;
      break;
    }
    for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(parameter->reserved); ++j) {
      if (parameter->reserved[j] != 0) {
        result = hipErrorInvalidValue;
        break;
      }
    }
    if (result != hipSuccess) {
      break;
    }

    if (parameter->coscheduledSmCount == 0) {
      parameter->coscheduledSmCount = input->sm.smCoscheduledAlignment;
    }
    if (parameter->coscheduledSmCount != input->sm.smCoscheduledAlignment) {
      // The current HAL resource topology only proves the native selectable
      // granule. Other co-scheduling groups require additional locality facts
      // and must not be reported as a guarantee until those facts exist.
      result = hipErrorNotSupported;
      break;
    }
    if (parameter->preferredCoscheduledSmCount == 0) {
      parameter->preferredCoscheduledSmCount = parameter->coscheduledSmCount;
    }
    if (parameter->preferredCoscheduledSmCount <
            parameter->coscheduledSmCount ||
        parameter->preferredCoscheduledSmCount %
                parameter->coscheduledSmCount !=
            0) {
      result = hipErrorInvalidResourceConfiguration;
      break;
    }

    const uint64_t minimum_sm_count =
        iree_max((uint64_t)input->sm.minSmPartitionSize,
                 (uint64_t)parameter->coscheduledSmCount);
    uint64_t selected_sm_count = parameter->smCount;
    if (selected_sm_count == 0) {
      iree_host_size_t selected_resource_count = remaining_resource_count;
      if (!(parameter->flags & hipDevSmResourceGroupBackfill)) {
        const iree_host_size_t coscheduled_resource_count =
            parameter->coscheduledSmCount / execution_units_per_resource;
        selected_resource_count -=
            selected_resource_count % coscheduled_resource_count;
      }
      selected_sm_count =
          (uint64_t)selected_resource_count * execution_units_per_resource;
      if (selected_sm_count < minimum_sm_count) {
        selected_sm_count = 0;
      }
    } else if (selected_sm_count < minimum_sm_count ||
               selected_sm_count % execution_units_per_resource != 0 ||
               (!(parameter->flags & hipDevSmResourceGroupBackfill) &&
                selected_sm_count % parameter->coscheduledSmCount != 0)) {
      result = hipErrorInvalidResourceConfiguration;
      break;
    }
    if (selected_sm_count > input->sm.smCount) {
      result = hipErrorInvalidResourceConfiguration;
      break;
    }

    const iree_host_size_t selected_resource_count =
        (iree_host_size_t)(selected_sm_count / execution_units_per_resource);
    if ((out_resources && selected_resource_count == 0) ||
        selected_resource_count > remaining_resource_count) {
      result = hipErrorInvalidResourceConfiguration;
      break;
    }
    partition_resource_counts[i] = selected_resource_count;
    remaining_resource_count -= selected_resource_count;
    parameter->smCount = (unsigned int)selected_sm_count;
  }

  iree_hip_execution_resource_partition_plan_t plan = {0};
  if (result == hipSuccess) {
    result = iree_hip_execution_resource_plan_partitions(
        host_allocator, family_spec, input_set, group_count,
        partition_resource_counts, out_remainder != NULL, &plan);
  }

  iree_host_size_t partition_offset = 0;
  for (iree_host_size_t i = 0; i < group_count && result == hipSuccess; ++i) {
    if (out_resources) {
      status = iree_hip_execution_resource_create_sm(
          device, queue_family,
          (iree_hal_queue_execution_resource_list_t){
              .count = partition_resource_counts[i],
              .ordinals = &plan.ordinals[partition_offset],
          },
          normalized_parameters[i].flags, &planned_partitions[i]);
      if (!iree_status_is_ok(status)) {
        result = iree_hip_execution_resource_consume_status(status);
      }
    }
    partition_offset += partition_resource_counts[i];
  }

  hipDevResource planned_remainder = {0};
  planned_remainder.type = hipDevResourceTypeInvalid;
  if (result == hipSuccess && out_remainder &&
      plan.remainder_resource_count != 0) {
    status = iree_hip_execution_resource_create_sm(
        device, queue_family,
        (iree_hal_queue_execution_resource_list_t){
            .count = plan.remainder_resource_count,
            .ordinals = &plan.ordinals[partition_offset],
        },
        hipDevSmResourceGroupDefault, &planned_remainder);
    if (!iree_status_is_ok(status)) {
      result = iree_hip_execution_resource_consume_status(status);
    }
  }

  if (result == hipSuccess) {
    if (out_resources) {
      memcpy(out_resources, planned_partitions,
             group_count * sizeof(*out_resources));
    }
    if (out_remainder) {
      *out_remainder = planned_remainder;
    }
    memcpy(group_parameters, normalized_parameters,
           group_count * sizeof(*group_parameters));
  }

  iree_hip_execution_resource_partition_plan_deinitialize(&plan);
  iree_allocator_free(host_allocator, planned_partitions);
  iree_allocator_free(host_allocator, partition_resource_counts);
  iree_allocator_free(host_allocator, normalized_parameters);
  return result;
}
