// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/user_queue.h"

#include <errno.h>
#include <linux/kfd_ioctl.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/atomics.h"
#include "libamdf/src/gpu/umd/kfd/buffer.h"
#include "libamdf/src/gpu/umd/kfd/device.h"
#include "libamdf/src/gpu/umd/kfd/target/user_queue.h"
#include "libamdf/src/gpu/umd/kfd/user_queue_native.h"
#include "libamdf/src/platform/linux/file.h"
#include "libamdf/src/platform/wait.h"

// One KFD buffer and the addresses established with it.
typedef struct amdf_gpu_kfd_user_queue_buffer_t {
  // Native allocation owner, or NULL after release.
  amdf_gpu_kfd_buffer_t* native;
  // Stable GPU address of the first byte.
  uint64_t device_address;
  // Stable host address of the first byte.
  void* host_pointer;
} amdf_gpu_kfd_user_queue_buffer_t;

// Native proof required before queue-reachable mappings may be released.
typedef enum amdf_gpu_kfd_user_queue_retirement_state_e {
  // No native queue can reach owned storage.
  AMDF_GPU_KFD_USER_QUEUE_RETIREMENT_RELEASABLE = 0,
  // The native queue identifier remains live and must be destroyed.
  AMDF_GPU_KFD_USER_QUEUE_RETIREMENT_ACTIVE = 1,
  // Native removal succeeded and requires a heavyweight-flush trigger.
  AMDF_GPU_KFD_USER_QUEUE_RETIREMENT_FLUSH_REQUIRED = 2,
  // Native removal consumed the identifier without proving quiescence.
  AMDF_GPU_KFD_USER_QUEUE_RETIREMENT_RESET_REQUIRED = 3,
} amdf_gpu_kfd_user_queue_retirement_state_t;

// One directly published KFD queue and all native-reachable storage.
struct amdf_gpu_umd_user_queue_t {
  // Owning device borrowed through final queue release.
  amdf_gpu_umd_device_t* device;
  // Native operation table borrowed through final queue release.
  const amdf_gpu_kfd_user_queue_native_api_t* native_api;
  // Qualified target-specific family and native construction plan.
  amdf_gpu_kfd_user_queue_plan_t plan;
  // Combined primary and optional metadata command-ring allocation.
  amdf_gpu_kfd_user_queue_buffer_t ring;
  // Read index, write index, and exception payload page.
  amdf_gpu_kfd_user_queue_buffer_t control;
  // End-of-pipe ring required by KFD compute queues.
  amdf_gpu_kfd_user_queue_buffer_t end_of_pipe;
  // Context-save, control-stack, and debug storage required by KFD.
  amdf_gpu_kfd_user_queue_buffer_t context;
  // Queue-inaccessible mapping whose invalidation triggers retirement flush.
  amdf_gpu_kfd_user_queue_buffer_t retirement_flush_trigger;
  // Base of the complete mapped KFD doorbell aperture.
  void* doorbell_mapping;
  // Exact 64-bit doorbell selected within `doorbell_mapping`.
  volatile uint64_t* doorbell;
  // KFD queue identifier, valid while retirement state is ACTIVE.
  uint32_t queue_identifier;
  // Proof still required before releasing queue-reachable storage.
  amdf_gpu_kfd_user_queue_retirement_state_t retirement_state;
  // Sticky native VM, provider, or firmware failure.
  amdf_atomic_uint64_t terminal_status;
};

// One host view of queue-owned storage.
struct amdf_gpu_umd_user_queue_mapping_t {
  // Host allocator copied for mapping destruction.
  amdf_allocator_t host_allocator;
};

static amdf_atomic_uint64_t* amdf_gpu_kfd_user_queue_control_value(
    const amdf_gpu_umd_user_queue_t* queue, size_t byte_offset) {
  return (amdf_atomic_uint64_t*)((uint8_t*)queue->control.host_pointer +
                                 byte_offset);
}

static amdf_atomic_uint64_t* amdf_gpu_kfd_user_queue_read_index(
    const amdf_gpu_umd_user_queue_t* queue) {
  return amdf_gpu_kfd_user_queue_control_value(
      queue, queue->plan.control.read_index_byte_offset);
}

static amdf_atomic_uint64_t* amdf_gpu_kfd_user_queue_write_index(
    const amdf_gpu_umd_user_queue_t* queue) {
  return amdf_gpu_kfd_user_queue_control_value(
      queue, queue->plan.control.write_index_byte_offset);
}

static amdf_atomic_uint64_t* amdf_gpu_kfd_user_queue_error_payload(
    const amdf_gpu_umd_user_queue_t* queue) {
  return amdf_gpu_kfd_user_queue_control_value(
      queue, queue->plan.control.error_payload_byte_offset);
}

static amdf_status_t amdf_gpu_kfd_user_queue_buffer_create(
    amdf_gpu_umd_user_queue_t* queue,
    const amdf_gpu_kfd_buffer_create_info_t* create_info,
    amdf_gpu_kfd_user_queue_buffer_t* out_buffer) {
  if (create_info->byte_length == 0) {
    *out_buffer = (amdf_gpu_kfd_user_queue_buffer_t){0};
    return AMDF_STATUS_OK;
  }
  amdf_gpu_kfd_buffer_t* native = NULL;
  amdf_gpu_kfd_buffer_result_t result = {0};
  const amdf_status_t status = queue->native_api->buffer_create(
      queue->native_api->user_data, queue->device, create_info, &native,
      &result);
  if (!amdf_status_is_ok(status)) return status;
  if (create_info->host_access != AMDF_GPU_KFD_BUFFER_HOST_ACCESS_NONE) {
    memset(result.host_pointer, 0, create_info->byte_length);
  }
  *out_buffer = (amdf_gpu_kfd_user_queue_buffer_t){
      .native = native,
      .device_address = result.device_address,
      .host_pointer = result.host_pointer,
  };
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_gpu_kfd_user_queue_buffer_destroy(
    amdf_gpu_umd_user_queue_t* queue,
    amdf_gpu_kfd_user_queue_buffer_t* buffer) {
  if (buffer->native == NULL) return AMDF_STATUS_OK;
  const amdf_status_t status = queue->native_api->buffer_destroy(
      queue->native_api->user_data, buffer->native);
  if (amdf_status_is_ok(status)) {
    *buffer = (amdf_gpu_kfd_user_queue_buffer_t){0};
  }
  return status;
}

static void amdf_gpu_kfd_user_queue_record_failure(
    amdf_gpu_umd_user_queue_t* queue, amdf_status_t failure) {
  uint64_t expected = AMDF_STATUS_OK;
  amdf_atomic_uint64_compare_exchange_acq_rel(&queue->terminal_status,
                                              &expected, failure);
}

static amdf_status_t amdf_gpu_kfd_user_queue_classify_error(
    uint64_t error_payload) {
  if (error_payload == 0) return AMDF_STATUS_OK;
  if ((error_payload & (KFD_EC_MASK_DEVICE | KFD_EC_MASK_PROCESS)) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  }
  const uint64_t queue_error = error_payload & KFD_EC_MASK_QUEUE;
  if (queue_error != 0 &&
      (error_payload &
       ~(KFD_EC_MASK_QUEUE | KFD_EC_MASK_DEVICE | KFD_EC_MASK_PROCESS)) == 0) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE, (uint32_t)queue_error);
  }
  return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
}

static amdf_status_t amdf_gpu_kfd_user_queue_release_storage(
    amdf_gpu_umd_user_queue_t* queue) {
  amdf_status_t status = AMDF_STATUS_OK;
  if (queue->retirement_flush_trigger.native != NULL) {
    status = amdf_gpu_kfd_user_queue_buffer_destroy(
        queue, &queue->retirement_flush_trigger);
  }
  if (queue->doorbell_mapping != NULL) {
    if (amdf_status_is_ok(status)) {
      status = queue->native_api->doorbell_unmap(
          queue->native_api->user_data, queue->doorbell_mapping,
          queue->plan.doorbell.mapping_byte_length);
    }
    if (amdf_status_is_ok(status)) {
      queue->doorbell_mapping = NULL;
      queue->doorbell = NULL;
    }
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_kfd_user_queue_buffer_destroy(queue, &queue->context);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_kfd_user_queue_buffer_destroy(queue, &queue->end_of_pipe);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_kfd_user_queue_buffer_destroy(queue, &queue->control);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_kfd_user_queue_buffer_destroy(queue, &queue->ring);
  }
  if (amdf_status_is_ok(status)) {
    amdf_free(queue->device->host_allocator, queue);
  }
  return status;
}

amdf_status_t amdf_gpu_umd_user_queue_destroy(
    amdf_gpu_umd_user_queue_t* queue) {
  if (queue == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (queue->retirement_state == AMDF_GPU_KFD_USER_QUEUE_RETIREMENT_ACTIVE) {
    const uint64_t consumed_index = amdf_atomic_uint64_load_acquire(
        amdf_gpu_kfd_user_queue_read_index(queue));
    const uint64_t producer_index = amdf_atomic_uint64_load_acquire(
        amdf_gpu_kfd_user_queue_write_index(queue));
    if (consumed_index < producer_index) {
      return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
    }
    if (consumed_index > producer_index) {
      const amdf_status_t failure =
          amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
      amdf_gpu_kfd_user_queue_record_failure(queue, failure);
      return failure;
    }

    const amdf_gpu_kfd_user_queue_destroy_result_t result =
        queue->native_api->queue_destroy(queue->native_api->user_data,
                                         queue->device,
                                         queue->queue_identifier);
    if (amdf_status_is_ok(result.status) || result.identifier_consumed) {
      queue->queue_identifier = 0;
      queue->retirement_state =
          amdf_status_is_ok(result.status)
              ? AMDF_GPU_KFD_USER_QUEUE_RETIREMENT_FLUSH_REQUIRED
              : AMDF_GPU_KFD_USER_QUEUE_RETIREMENT_RESET_REQUIRED;
    }
    if (!amdf_status_is_ok(result.status)) return result.status;
  }

  if (queue->retirement_state ==
      AMDF_GPU_KFD_USER_QUEUE_RETIREMENT_FLUSH_REQUIRED) {
    const amdf_status_t status = amdf_gpu_kfd_user_queue_buffer_destroy(
        queue, &queue->retirement_flush_trigger);
    if (!amdf_status_is_ok(status)) return status;
    queue->retirement_state = AMDF_GPU_KFD_USER_QUEUE_RETIREMENT_RELEASABLE;
  } else if (queue->retirement_state ==
             AMDF_GPU_KFD_USER_QUEUE_RETIREMENT_RESET_REQUIRED) {
    amdf_gpu_kfd_reset_state_t reset_state = {0};
    const amdf_status_t status = queue->native_api->reset_query(
        queue->native_api->user_data, queue->device, &reset_state);
    if (!amdf_status_is_ok(status)) return status;
    if (!reset_state.reset_observed || reset_state.reset_in_progress) {
      return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
    }
    queue->retirement_state = AMDF_GPU_KFD_USER_QUEUE_RETIREMENT_RELEASABLE;
  }
  return amdf_gpu_kfd_user_queue_release_storage(queue);
}

static void amdf_gpu_kfd_user_queue_abandon(amdf_gpu_umd_user_queue_t* queue) {
  // Native queue addresses refer only to separate backing, never this metadata.
  // Preserve all remaining mappings when rollback cannot prove retirement.
  amdf_gpu_kfd_buffer_t* buffers[] = {
      queue->ring.native,
      queue->control.native,
      queue->end_of_pipe.native,
      queue->context.native,
      queue->retirement_flush_trigger.native,
  };
  for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); ++i) {
    if (buffers[i] != NULL) {
      queue->native_api->buffer_abandon(queue->native_api->user_data,
                                        buffers[i]);
    }
  }
  amdf_free(queue->device->host_allocator, queue);
}

static bool amdf_gpu_kfd_user_queue_select_plan(
    const amdf_gpu_umd_device_t* device,
    const amdf_gpu_umd_user_queue_create_info_t* create_info,
    amdf_gpu_kfd_user_queue_plan_t* out_plan) {
  if (!device->reset_monitor.context_owned) return false;
  amdf_gpu_kfd_user_queue_plans_t plans;
  amdf_gpu_kfd_target_user_queue_plans_initialize(
      &device->topology, device->page_size, device->cache_line_size, &plans);
  for (uint32_t i = 0; i < plans.count; ++i) {
    const amdf_gpu_kfd_user_queue_plan_t* plan = &plans.values[i];
    const amdf_gpu_queue_family_properties_t* family = &plan->family;
    if (create_info->command_type == family->command_type &&
        create_info->format_version == family->format_version &&
        create_info->priority == AMDF_QUEUE_PRIORITY_NORMAL &&
        create_info->producer_mode == AMDF_QUEUE_PRODUCER_MODE_SINGLE &&
        (create_info->required_capabilities &
         ~family->user_queue_capabilities) == 0 &&
        create_info->roles == family->roles &&
        (create_info->ring_byte_length == 0 ||
         create_info->ring_byte_length == family->minimum_ring_byte_length)) {
      *out_plan = *plan;
      return true;
    }
  }
  return false;
}

amdf_status_t amdf_gpu_umd_user_queue_create(
    amdf_gpu_umd_device_t* device,
    const amdf_gpu_umd_user_queue_create_info_t* create_info,
    amdf_gpu_umd_user_queue_t** out_queue,
    amdf_gpu_umd_user_queue_result_t* out_result) {
  if (device == NULL || create_info == NULL || out_queue == NULL ||
      out_result == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_gpu_kfd_user_queue_plan_t plan;
  if (device->user_queue_native_api == NULL ||
      !amdf_gpu_kfd_user_queue_select_plan(device, create_info, &plan)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }

  amdf_gpu_umd_user_queue_t* queue = NULL;
  amdf_status_t status =
      amdf_calloc(device->host_allocator, sizeof(*queue),
                  amdf_alignof(amdf_gpu_umd_user_queue_t), (void**)&queue);
  if (!amdf_status_is_ok(status)) return status;
  queue->device = device;
  queue->native_api = device->user_queue_native_api;
  queue->plan = plan;
  amdf_atomic_uint64_initialize(&queue->terminal_status, AMDF_STATUS_OK);

  status = amdf_gpu_kfd_user_queue_buffer_create(
      queue, &queue->plan.ring.storage, &queue->ring);
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_kfd_user_queue_buffer_create(
        queue, &queue->plan.control.storage, &queue->control);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_kfd_user_queue_buffer_create(
        queue, &queue->plan.compute.end_of_pipe_storage, &queue->end_of_pipe);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_kfd_user_queue_buffer_create(
        queue, &queue->plan.compute.context_storage, &queue->context);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_kfd_user_queue_buffer_create(
        queue, &queue->plan.retirement.flush_trigger_storage,
        &queue->retirement_flush_trigger);
  }
  if (amdf_status_is_ok(status) && queue->context.host_pointer != NULL) {
    struct kfd_context_save_area_header* header =
        (struct kfd_context_save_area_header*)queue->context.host_pointer;
    header->debug_offset = queue->plan.compute.debug_byte_offset;
    header->debug_size = queue->plan.compute.debug_byte_length;
    if (queue->plan.control.error_payload_byte_length != 0) {
      header->err_payload_addr = queue->control.device_address +
                                 queue->plan.control.error_payload_byte_offset;
    }
  }

  struct kfd_ioctl_create_queue_args arguments = {0};
  if (amdf_status_is_ok(status)) {
    arguments = (struct kfd_ioctl_create_queue_args){
        .ring_base_address =
            queue->ring.device_address + queue->plan.ring.primary_byte_offset,
        .write_pointer_address = queue->control.device_address +
                                 queue->plan.control.write_index_byte_offset,
        .read_pointer_address = queue->control.device_address +
                                queue->plan.control.read_index_byte_offset,
        .ring_size = (uint32_t)queue->plan.ring.primary_byte_length,
        .gpu_id = device->topology.gpu_id,
        .queue_type = queue->plan.native_queue_type,
        .queue_percentage = KFD_MAX_QUEUE_PERCENTAGE,
        .queue_priority = 7,
        .eop_buffer_address = queue->end_of_pipe.device_address,
        .eop_buffer_size = queue->plan.compute.end_of_pipe_storage.byte_length,
        .ctx_save_restore_address = queue->context.device_address,
        .ctx_save_restore_size =
            queue->plan.compute.context_save_restore_byte_length,
        .ctl_stack_size = queue->plan.compute.control_stack_byte_length,
    };
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    status = queue->native_api->queue_create(queue->native_api->user_data,
                                             device, &arguments);
    if (amdf_status_is_ok(status)) {
      queue->queue_identifier = arguments.queue_id;
      queue->retirement_state = AMDF_GPU_KFD_USER_QUEUE_RETIREMENT_ACTIVE;
    }
  }

  size_t doorbell_byte_offset = 0;
  if (amdf_status_is_ok(status)) {
    const size_t mapping_byte_length = queue->plan.doorbell.mapping_byte_length;
    const uint64_t mapping_mask = mapping_byte_length - 1;
    doorbell_byte_offset = (size_t)(arguments.doorbell_offset & mapping_mask);
    if ((mapping_byte_length & mapping_mask) != 0 ||
        doorbell_byte_offset % sizeof(uint64_t) != 0 ||
        doorbell_byte_offset > mapping_byte_length - sizeof(uint64_t)) {
      status = amdf_linux_error(EPROTO);
    } else {
      void* mapping = NULL;
      status = queue->native_api->doorbell_map(
          queue->native_api->user_data, device,
          arguments.doorbell_offset & ~mapping_mask, mapping_byte_length,
          &mapping);
      if (amdf_status_is_ok(status)) queue->doorbell_mapping = mapping;
    }
  }

  if (amdf_status_is_ok(status)) {
    queue->doorbell = (volatile uint64_t*)((uint8_t*)queue->doorbell_mapping +
                                           doorbell_byte_offset);
    const amdf_gpu_umd_user_queue_result_t result = {
        .queue_id =
            {
                .words = {device->topology.gpu_id, (uintptr_t)queue},
            },
        .capabilities = queue->plan.family.user_queue_capabilities,
        .ring_byte_length = queue->plan.ring.primary_byte_length,
        .metadata_ring_byte_length = queue->plan.ring.metadata_byte_length,
    };
    *out_queue = queue;
    *out_result = result;
  } else {
    const amdf_status_t release_status = amdf_gpu_umd_user_queue_destroy(queue);
    if (!amdf_status_is_ok(release_status)) {
      amdf_gpu_kfd_user_queue_abandon(queue);
      status = release_status;
    }
  }
  return status;
}

amdf_status_t amdf_gpu_umd_user_queue_map(
    amdf_gpu_umd_user_queue_t* queue, amdf_gpu_umd_device_t* producer_device,
    amdf_gpu_umd_user_queue_mapping_t** out_mapping,
    amdf_gpu_umd_user_queue_mapping_result_t* out_result) {
  if (queue == NULL || out_mapping == NULL || out_result == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (producer_device != NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_gpu_umd_user_queue_mapping_t* mapping = NULL;
  const amdf_status_t status = amdf_calloc(
      queue->device->host_allocator, sizeof(*mapping),
      amdf_alignof(amdf_gpu_umd_user_queue_mapping_t), (void**)&mapping);
  if (!amdf_status_is_ok(status)) return status;
  mapping->host_allocator = queue->device->host_allocator;
  const amdf_gpu_umd_user_queue_mapping_result_t result = {
      .ring_address = (uintptr_t)((uint8_t*)queue->ring.host_pointer +
                                  queue->plan.ring.primary_byte_offset),
      .read_index_address =
          (uintptr_t)amdf_gpu_kfd_user_queue_read_index(queue),
      .write_index_address =
          (uintptr_t)amdf_gpu_kfd_user_queue_write_index(queue),
      .doorbell_address = (uintptr_t)queue->doorbell,
      .index_bits = queue->plan.control.index_bit_count,
      .doorbell_bits = queue->plan.doorbell.bit_count,
      .metadata_ring_address =
          queue->plan.ring.metadata_byte_length == 0
              ? 0
              : (uintptr_t)((uint8_t*)queue->ring.host_pointer +
                            queue->plan.ring.metadata_byte_offset),
  };
  *out_mapping = mapping;
  *out_result = result;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_umd_user_queue_mapping_destroy(
    amdf_gpu_umd_user_queue_mapping_t* mapping) {
  if (mapping == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_allocator_t host_allocator = mapping->host_allocator;
  amdf_free(host_allocator, mapping);
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_umd_user_queue_query_status(
    amdf_gpu_umd_user_queue_t* queue, amdf_user_queue_status_t* out_status) {
  if (queue == NULL || out_status == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  // KFD reports memory violations through device events, not the queue's
  // context-save error payload. The render-file cache observes the same VM
  // without allocating an event or consulting another process's faults.
  struct drm_amdgpu_info_gpuvm_fault fault = {0};
  if (amdf_status_is_ok(
          amdf_atomic_uint64_load_acquire(&queue->terminal_status))) {
    const amdf_status_t status = queue->native_api->vm_fault_query(
        queue->native_api->user_data, queue->device, &fault);
    if (!amdf_status_is_ok(status)) return status;
  }
  const uint64_t consumed_index = amdf_atomic_uint64_load_acquire(
      amdf_gpu_kfd_user_queue_read_index(queue));
  const uint64_t producer_index = amdf_atomic_uint64_load_acquire(
      amdf_gpu_kfd_user_queue_write_index(queue));
  amdf_status_t terminal_status = AMDF_STATUS_OK;
  if (queue->plan.control.error_payload_byte_length != 0) {
    const uint64_t error_payload = amdf_atomic_uint64_load_acquire(
        amdf_gpu_kfd_user_queue_error_payload(queue));
    terminal_status = amdf_gpu_kfd_user_queue_classify_error(error_payload);
  }
  if (consumed_index > producer_index) {
    terminal_status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  if (fault.status != 0) {
    terminal_status = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  }
  if (!amdf_status_is_ok(terminal_status)) {
    amdf_gpu_kfd_user_queue_record_failure(queue, terminal_status);
  }
  terminal_status = amdf_atomic_uint64_load_acquire(&queue->terminal_status);
  *out_status = (amdf_user_queue_status_t){
      .type = AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS,
      .structure_size = sizeof(amdf_user_queue_status_t),
      .state =
          terminal_status == amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST)
              ? AMDF_QUEUE_STATE_DEVICE_LOST
          : amdf_status_is_ok(terminal_status) ? AMDF_QUEUE_STATE_ACTIVE
                                               : AMDF_QUEUE_STATE_FAILED,
      .reset_epoch = 1,
      .producer_index = producer_index,
      .consumed_index = consumed_index,
      .terminal_status = terminal_status,
  };
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_umd_user_queue_wait_consumed(
    amdf_gpu_umd_user_queue_t* queue, uint64_t published_index,
    const amdf_wait_deadline_t* deadline) {
  if (queue == NULL || deadline == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_user_queue_status_t queue_status = {0};
  amdf_status_t status =
      amdf_gpu_umd_user_queue_query_status(queue, &queue_status);
  if (!amdf_status_is_ok(status)) return status;
  if (published_index > queue_status.producer_index) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  for (;;) {
    if (!amdf_status_is_ok(queue_status.terminal_status)) {
      return queue_status.terminal_status;
    }
    if (queue_status.consumed_index >= published_index) return AMDF_STATUS_OK;

    amdf_wait_budget_t remaining = {0};
    status = amdf_wait_deadline_query_remaining(deadline, &remaining);
    if (!amdf_status_is_ok(status)) return status;
    if (remaining.timeout == 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
    }
    if (remaining.poll == 0) amdf_platform_wait_yield();
    status = amdf_gpu_umd_user_queue_query_status(queue, &queue_status);
    if (!amdf_status_is_ok(status)) return status;
  }
}
