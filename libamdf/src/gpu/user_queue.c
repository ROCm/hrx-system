// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/user_queue.h"

#include <stddef.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/device.h"
#include "libamdf/src/endpoint.h"
#include "libamdf/src/gpu/device.h"
#include "libamdf/src/gpu/umd/user_queue.h"
#include "libamdf/src/memory.h"
#include "libamdf/src/structure.h"
#include "libamdf/src/user_queue.h"
#include "libamdf/src/wait.h"

typedef struct amdf_gpu_user_queue_t {
  // Generic user-queue state shared by every engine implementation.
  amdf_user_queue_t base;
  // Exact native directly published GPU queue.
  amdf_gpu_umd_user_queue_t* umd;
  // Optional scratch attachment borrowed until native teardown succeeds.
  amdf_memory_t* scratch_memory;
} amdf_gpu_user_queue_t;

_Static_assert(offsetof(amdf_gpu_user_queue_t, base) == 0,
               "GPU user queue base must be the first field");

typedef struct amdf_gpu_user_queue_mapping_t {
  // Generic mapping state and producer lifetime borrows.
  amdf_user_queue_mapping_t base;
  // Exact native producer-local queue mapping.
  amdf_gpu_umd_user_queue_mapping_t* umd;
} amdf_gpu_user_queue_mapping_t;

_Static_assert(offsetof(amdf_gpu_user_queue_mapping_t, base) == 0,
               "GPU user queue mapping base must be the first field");

static amdf_status_t amdf_gpu_user_queue_mapping_destroy_native(
    amdf_user_queue_mapping_t* base_mapping) {
  amdf_gpu_user_queue_mapping_t* mapping =
      (amdf_gpu_user_queue_mapping_t*)base_mapping;
  const amdf_status_t status =
      amdf_gpu_umd_user_queue_mapping_destroy(mapping->umd);
  if (amdf_status_is_ok(status)) mapping->umd = NULL;
  return status;
}

static const amdf_user_queue_mapping_vtable_t
    amdf_gpu_user_queue_mapping_vtable = {
        .destroy_native = amdf_gpu_user_queue_mapping_destroy_native,
};

static amdf_status_t amdf_gpu_user_queue_map(
    amdf_user_queue_t* base_queue, amdf_device_t* producer_device,
    amdf_user_queue_mapping_t** out_mapping) {
  amdf_gpu_user_queue_t* queue = (amdf_gpu_user_queue_t*)base_queue;
  amdf_gpu_umd_device_t* producer_umd = NULL;
  amdf_device_id_t producer_device_id = {0};
  uint64_t producer_reset_epoch = 0;
  if (producer_device == NULL) {
    if ((base_queue->info.capabilities &
         AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER) == 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
  } else {
    if (!amdf_device_is_engine(producer_device, AMDF_ENGINE_KIND_GPU)) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    if ((base_queue->info.capabilities &
         AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER) == 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    const amdf_gpu_device_info_t* producer_info =
        amdf_gpu_device_get_info(producer_device);
    producer_device_id = producer_info->id;
    producer_reset_epoch = producer_info->reset_epoch;
    producer_umd = amdf_gpu_device_get_umd(producer_device);
  }

  const amdf_allocator_t host_allocator = base_queue->host_allocator;
  amdf_gpu_user_queue_mapping_t* mapping = NULL;
  amdf_status_t status = amdf_calloc(
      host_allocator, sizeof(*mapping),
      amdf_alignof(amdf_gpu_user_queue_mapping_t), (void**)&mapping);
  if (!amdf_status_is_ok(status)) return status;
  const amdf_user_queue_mapping_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO,
      .structure_size = sizeof(info),
      .producer_device_id = producer_device_id,
      .queue_id = base_queue->info.queue_id,
      .queue_reset_epoch = base_queue->info.reset_epoch,
      .producer_reset_epoch = producer_reset_epoch,
      .command_type = base_queue->info.command_type,
      .format_version = base_queue->info.format_version,
      .format_features = base_queue->info.format_features,
      .ring_byte_length = base_queue->info.ring_byte_length,
      .metadata = base_queue->info.metadata,
      .metadata_ring_byte_length = base_queue->info.metadata_ring_byte_length,
  };
  status = amdf_user_queue_mapping_initialize(
      &mapping->base, &amdf_gpu_user_queue_mapping_vtable, base_queue,
      producer_device, &info);
  amdf_gpu_umd_user_queue_mapping_result_t result = {0};
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_umd_user_queue_map(queue->umd, producer_umd,
                                         &mapping->umd, &result);
  }
  if (amdf_status_is_ok(status)) {
    mapping->base.info.ring_address = result.ring_address;
    mapping->base.info.read_index_address = result.read_index_address;
    mapping->base.info.write_index_address = result.write_index_address;
    mapping->base.info.doorbell_address = result.doorbell_address;
    mapping->base.info.index_bits = result.index_bits;
    mapping->base.info.doorbell_bits = result.doorbell_bits;
    mapping->base.info.metadata_ring_address = result.metadata_ring_address;
    *out_mapping = &mapping->base;
  } else {
    if (mapping->base.queue != NULL) {
      amdf_user_queue_mapping_deinitialize(&mapping->base);
    }
    amdf_free(host_allocator, mapping);
  }
  return status;
}

static amdf_status_t amdf_gpu_user_queue_query_status(
    amdf_user_queue_t* base_queue, amdf_user_queue_status_t* out_status) {
  amdf_gpu_user_queue_t* queue = (amdf_gpu_user_queue_t*)base_queue;
  return amdf_gpu_umd_user_queue_query_status(queue->umd, out_status);
}

static amdf_status_t amdf_gpu_user_queue_wait_consumed(
    amdf_user_queue_t* base_queue, uint64_t published_index,
    uint64_t timeout_nanoseconds, uint64_t poll_duration_nanoseconds) {
  amdf_wait_deadline_t deadline;
  const amdf_status_t status = amdf_wait_deadline_initialize(
      timeout_nanoseconds, poll_duration_nanoseconds, &deadline);
  if (!amdf_status_is_ok(status)) return status;
  amdf_gpu_user_queue_t* queue = (amdf_gpu_user_queue_t*)base_queue;
  return amdf_gpu_umd_user_queue_wait_consumed(queue->umd, published_index,
                                               &deadline);
}

static amdf_status_t amdf_gpu_user_queue_destroy_native(
    amdf_user_queue_t* base_queue) {
  amdf_gpu_user_queue_t* queue = (amdf_gpu_user_queue_t*)base_queue;
  const amdf_status_t status = amdf_gpu_umd_user_queue_destroy(queue->umd);
  if (amdf_status_is_ok(status)) {
    queue->umd = NULL;
    if (queue->scratch_memory != NULL) {
      amdf_memory_unregister_child(queue->scratch_memory);
      queue->scratch_memory = NULL;
    }
  }
  return status;
}

static const amdf_user_queue_vtable_t amdf_gpu_user_queue_vtable = {
    .map = amdf_gpu_user_queue_map,
    .query_status = amdf_gpu_user_queue_query_status,
    .wait_consumed = amdf_gpu_user_queue_wait_consumed,
    .destroy_native = amdf_gpu_user_queue_destroy_native,
};

static amdf_status_t amdf_gpu_user_queue_validate_scratch(
    amdf_device_t* device, const amdf_queue_family_info_t* family_info,
    const amdf_gpu_queue_scratch_t* scratch,
    amdf_gpu_umd_queue_scratch_t* out_scratch) {
  const bool disabled = scratch->memory == NULL &&
                        scratch->access_ordinal == 0 &&
                        scratch->reserved == 0 && scratch->byte_offset == 0 &&
                        scratch->byte_length == 0 &&
                        scratch->maximum_private_segment_byte_length == 0 &&
                        scratch->maximum_wave_count == 0;
  if (disabled) return AMDF_STATUS_OK;
  if (scratch->memory == NULL || scratch->reserved != 0 ||
      scratch->byte_length == 0 ||
      scratch->maximum_private_segment_byte_length == 0 ||
      scratch->maximum_wave_count == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if ((family_info->roles & AMDF_QUEUE_ROLE_COMPUTE) == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const amdf_memory_t* memory = scratch->memory;
  if (scratch->access_ordinal >= memory->info.access_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const amdf_memory_access_state_t* access =
      &memory->accesses[scratch->access_ordinal];
  if (access->device != device) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if ((access->info.address_kinds & (UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU)) ==
          0 ||
      (access->info.access &
       (AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE)) !=
          (AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (scratch->byte_offset > memory->info.byte_length ||
      scratch->byte_length > memory->info.byte_length - scratch->byte_offset ||
      access->addresses[AMDF_MEMORY_ADDRESS_GPU] >
          UINT64_MAX - scratch->byte_offset) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const amdf_gpu_device_info_t* device_info = amdf_gpu_device_get_info(device);
  if (access->info.reset_epoch != device_info->reset_epoch) {
    return amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION);
  }
  *out_scratch = (amdf_gpu_umd_queue_scratch_t){
      .device_address =
          access->addresses[AMDF_MEMORY_ADDRESS_GPU] + scratch->byte_offset,
      .byte_length = scratch->byte_length,
      .maximum_private_segment_byte_length =
          scratch->maximum_private_segment_byte_length,
      .maximum_wave_count = scratch->maximum_wave_count,
  };
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_gpu_user_queue_create(
    amdf_device_t* device, const amdf_gpu_user_queue_create_info_t* create_info,
    amdf_user_queue_t** out_queue) {
  if (out_queue == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (!amdf_device_is_engine(device, AMDF_ENGINE_KIND_GPU)) {
    return device == NULL
               ? amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT)
               : amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_status_t status = amdf_structure_validate_input(
      create_info, AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO,
      (uint32_t)sizeof(amdf_gpu_user_queue_create_info_t));
  if (!amdf_status_is_ok(status)) return status;
  if (create_info->reserved != 0 ||
      (create_info->priority != AMDF_QUEUE_PRIORITY_LOW &&
       create_info->priority != AMDF_QUEUE_PRIORITY_NORMAL &&
       create_info->priority != AMDF_QUEUE_PRIORITY_HIGH) ||
      (create_info->producer_mode != AMDF_QUEUE_PRODUCER_MODE_SINGLE &&
       create_info->producer_mode != AMDF_QUEUE_PRODUCER_MODE_MULTI)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }

  amdf_queue_family_info_t family_info = {
      .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
      .structure_size = sizeof(family_info),
  };
  status = amdf_endpoint_query_queue_family_info(
      device->endpoint, create_info->queue_family_ordinal, &family_info);
  if (!amdf_status_is_ok(status)) return status;
  if ((family_info.command_type != AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 &&
       family_info.command_type != AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA &&
       family_info.command_type != AMDF_QUEUE_COMMAND_TYPE_GPU_AQL) ||
      (family_info.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_USER) == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if ((create_info->required_capabilities &
       ~family_info.user_queue_capabilities) != 0 ||
      (family_info.producer_modes & (1u << create_info->producer_mode)) == 0 ||
      (family_info.priority_capabilities & (1u << create_info->priority)) ==
          0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (create_info->ring_byte_length != 0 &&
      ((create_info->ring_byte_length & (create_info->ring_byte_length - 1)) !=
           0 ||
       create_info->ring_byte_length < family_info.minimum_ring_byte_length ||
       create_info->ring_byte_length > family_info.maximum_ring_byte_length ||
       create_info->ring_byte_length % family_info.ring_byte_length_alignment !=
           0)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }

  amdf_gpu_umd_queue_scratch_t scratch = {0};
  status = amdf_gpu_user_queue_validate_scratch(
      device, &family_info, &create_info->scratch, &scratch);
  if (!amdf_status_is_ok(status)) return status;

  const amdf_gpu_device_info_t* device_info = amdf_gpu_device_get_info(device);
  const amdf_user_queue_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_USER_QUEUE_INFO,
      .structure_size = sizeof(info),
      .device_id = device_info->id,
      .reset_epoch = device_info->reset_epoch,
      .queue_family_ordinal = family_info.ordinal,
      .command_type = family_info.command_type,
      .format_version = family_info.format_version,
      .format_features = family_info.format_features,
      .producer_mode = create_info->producer_mode,
      .priority = create_info->priority,
      .roles = family_info.roles,
      .metadata = family_info.metadata,
  };
  const amdf_gpu_umd_user_queue_create_info_t umd_create_info = {
      .command_type = family_info.command_type,
      .format_version = family_info.format_version,
      .priority = create_info->priority,
      .producer_mode = create_info->producer_mode,
      .required_capabilities = create_info->required_capabilities,
      .roles = family_info.roles,
      .ring_byte_length = create_info->ring_byte_length,
      .scratch = scratch,
  };
  const amdf_allocator_t host_allocator = amdf_device_host_allocator(device);
  amdf_gpu_user_queue_t* queue = NULL;
  status = amdf_calloc(host_allocator, sizeof(*queue),
                       amdf_alignof(amdf_gpu_user_queue_t), (void**)&queue);
  if (!amdf_status_is_ok(status)) return status;
  status = amdf_user_queue_initialize(&queue->base, &amdf_gpu_user_queue_vtable,
                                      device, &info);
  if (amdf_status_is_ok(status) && create_info->scratch.memory != NULL) {
    status = amdf_memory_register_child(create_info->scratch.memory);
    if (amdf_status_is_ok(status)) {
      queue->scratch_memory = create_info->scratch.memory;
    }
  }
  amdf_gpu_umd_user_queue_result_t result = {0};
  if (amdf_status_is_ok(status)) {
    status =
        amdf_gpu_umd_user_queue_create(amdf_gpu_device_get_umd(device),
                                       &umd_create_info, &queue->umd, &result);
  }
  if (amdf_status_is_ok(status)) {
    queue->base.info.queue_id = result.queue_id;
    queue->base.info.capabilities = result.capabilities;
    queue->base.info.ring_byte_length = result.ring_byte_length;
    queue->base.info.metadata_ring_byte_length =
        result.metadata_ring_byte_length;
    *out_queue = &queue->base;
  } else {
    if (queue->scratch_memory != NULL) {
      amdf_memory_unregister_child(queue->scratch_memory);
    }
    if (queue->base.device != NULL) {
      amdf_user_queue_deinitialize(&queue->base);
    }
    amdf_free(host_allocator, queue);
  }
  return status;
}
