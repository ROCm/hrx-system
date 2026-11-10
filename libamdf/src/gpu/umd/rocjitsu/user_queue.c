// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/rocjitsu/user_queue.h"

#include <linux/kfd_ioctl.h>
#include <stddef.h>
#include <string.h>
#include <sys/mman.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/rocjitsu/buffer.h"
#include "libamdf/src/gpu/umd/rocjitsu/device.h"
#include "libamdf/src/platform/linux/file.h"
#include "libamdf/src/platform/wait.h"

struct amdf_gpu_umd_user_queue_t {
  // Common device retains the source and allocator until queue release.
  amdf_gpu_umd_device_t* device;
  // Owned native command ring, attached to the selected GPU.
  amdf_rocjitsu_buffer_t* ring;
  // Owned native indices and optional AMD AQL queue/signal descriptors.
  amdf_rocjitsu_buffer_t* control;
  // Selected native packet and control-storage ABI.
  amdf_queue_command_type_t command_type;
  // Consumer frontier byte offset in control storage.
  size_t read_index_byte_offset;
  // Producer frontier byte offset in control storage.
  size_t write_index_byte_offset;
  // Native queue identifier, meaningful when created is true.
  uint32_t identifier;
  // Whether native queue destruction is required.
  bool created;
  // Whether explicit terminal removal has already consumed the native ID.
  bool inactivation_attempted;
  // First removal result; a failure preserves all reachable backing.
  amdf_status_t inactivation_status;
  // Whether firmware scratch requests pause for caller-owned service.
  bool scratch_service;
  // First native, VM, or firmware failure observed by this queue.
  amdf_atomic_uint64_t terminal_status;
  // Complete mapped doorbell page, or zero before mapping succeeds.
  uint64_t doorbell_mapping;
  // Exact notification word within the process doorbell page.
  uint64_t doorbell_address;
};

static void amdf_rocjitsu_user_queue_record_failure(
    amdf_gpu_umd_user_queue_t* queue, amdf_status_t status) {
  uint64_t expected = AMDF_STATUS_OK;
  amdf_atomic_uint64_compare_exchange_acq_rel(&queue->terminal_status,
                                              &expected, status);
}

static amdf_atomic_uint64_t* amdf_rocjitsu_user_queue_inactive_value(
    const amdf_gpu_umd_user_queue_t* queue) {
  return (amdf_atomic_uint64_t*)((uint8_t*)queue->control->pointer +
                                 queue->device->gpu->aql
                                     .inactive_signal_byte_offset +
                                 offsetof(amdf_gpu_aql_signal_t, value));
}

static bool amdf_rocjitsu_user_queue_is_scratch_request(uint64_t value) {
  return value != 0 && (value & ~UINT64_C(0x401)) == 0;
}

static amdf_status_t amdf_rocjitsu_user_queue_release(
    amdf_gpu_umd_user_queue_t* queue) {
  amdf_rocjitsu_source_t* source = queue->device->gpu->source;
  amdf_status_t status = queue->inactivation_status;
  if (queue->created) {
    struct kfd_ioctl_destroy_queue_args destroy = {.queue_id =
                                                       queue->identifier};
    queue->created = false;
    status = amdf_rocjitsu_source_command(source, AMDKFD_IOC_DESTROY_QUEUE,
                                          &destroy, sizeof(destroy));
  }
  if (amdf_status_is_ok(status) && queue->doorbell_mapping != 0) {
    rj_vm_unmap_t unmap = {.addr = queue->doorbell_mapping, .length = 4096};
    status = amdf_rocjitsu_status(source->library.rj_vm_device_unmap_as(
        source->machine, source->process, &unmap));
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_rocjitsu_buffer_destroy(queue->control);
    if (amdf_status_is_ok(status)) {
      queue->control = NULL;
    }
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_rocjitsu_buffer_destroy(queue->ring);
    if (amdf_status_is_ok(status)) {
      queue->ring = NULL;
    }
  }
  // A failed native teardown cannot establish that the engine stopped reaching
  // this storage. Metadata is consumed, while the reachable mappings survive.
  amdf_rocjitsu_buffer_abandon(queue->control);
  amdf_rocjitsu_buffer_abandon(queue->ring);
  amdf_free(queue->device->host_allocator, queue);
  return status;
}

amdf_status_t amdf_rocjitsu_user_queue_create(
    amdf_gpu_umd_device_t* device,
    const amdf_gpu_umd_user_queue_create_info_t* create_info,
    amdf_gpu_umd_user_queue_t** out_queue,
    amdf_gpu_umd_user_queue_result_t* out_result) {
  const bool aql = create_info->command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL;
  if ((!aql && create_info->scratch.byte_length != 0) ||
      create_info->dispatch_timestamps) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const uint64_t ring_length =
      create_info->ring_byte_length != 0 ? create_info->ring_byte_length : 4096;
  amdf_gpu_aql_descriptor_t descriptor = {0};
  if (aql) {
    const amdf_status_t status = amdf_gpu_aql_descriptor_initialize(
        &device->gpu->aql, ring_length, create_info, &descriptor);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  amdf_gpu_umd_user_queue_t* queue = NULL;
  amdf_status_t status =
      amdf_calloc(device->host_allocator, sizeof(*queue),
                  amdf_alignof(amdf_gpu_umd_user_queue_t), (void**)&queue);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  queue->device = device;
  queue->command_type = create_info->command_type;
  queue->scratch_service = create_info->scratch_service;
  queue->read_index_byte_offset =
      aql ? offsetof(amdf_gpu_aql_descriptor_t, read_dispatch_id) : 0;
  queue->write_index_byte_offset =
      aql ? offsetof(amdf_gpu_aql_descriptor_t, write_dispatch_id) : 64;
  const amdf_memory_access_t access =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  status = amdf_rocjitsu_buffer_prepare(
      device->gpu->source, 1, &device->gpu->native.gpu_id,
      AMDF_MEMORY_CLASS_SYSTEM, ring_length, 4096, access, &queue->ring);
  if (amdf_status_is_ok(status)) {
    status = amdf_rocjitsu_buffer_prepare(
        device->gpu->source, 1, &device->gpu->native.gpu_id,
        AMDF_MEMORY_CLASS_SYSTEM, 4096, 4096, access, &queue->control);
  }
  struct kfd_ioctl_create_queue_args create = {0};
  if (amdf_status_is_ok(status)) {
    memset(queue->ring->pointer, 0, queue->ring->byte_length);
    memset(queue->control->pointer, 0, queue->control->byte_length);
    if (aql) {
      descriptor.ring_address = (uintptr_t)queue->ring->pointer;
      descriptor.id = (uintptr_t)queue->control->pointer;
      descriptor.inactive_signal_address =
          (uintptr_t)queue->control->pointer + sizeof(descriptor);
      memcpy(queue->control->pointer, &descriptor, sizeof(descriptor));
      amdf_gpu_aql_signal_t* inactive_signal =
          (void*)((uint8_t*)queue->control->pointer + sizeof(descriptor));
      inactive_signal->kind = 1;
      inactive_signal->event_mailbox_address =
          create_info->inactive_event.mailbox_address;
      inactive_signal->event_id = create_info->inactive_event.interrupt_context;
      for (size_t offset = 0; offset < ring_length; offset += 64) {
        *(uint32_t*)((uint8_t*)queue->ring->pointer + offset) = 1;
      }
    }
    create.gpu_id = device->gpu->native.gpu_id;
    create.queue_type =
        aql ? KFD_IOC_QUEUE_TYPE_COMPUTE_AQL : KFD_IOC_QUEUE_TYPE_COMPUTE;
    create.ring_base_address = (uintptr_t)queue->ring->pointer;
    create.ring_size = ring_length;
    create.read_pointer_address =
        (uintptr_t)queue->control->pointer + queue->read_index_byte_offset;
    create.write_pointer_address =
        (uintptr_t)queue->control->pointer + queue->write_index_byte_offset;
    create.queue_percentage = 100;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    status = amdf_rocjitsu_source_command(
        device->gpu->source, AMDKFD_IOC_CREATE_QUEUE, &create, sizeof(create));
    if (amdf_status_is_ok(status)) {
      queue->created = true;
      queue->identifier = create.queue_id;
    }
  }
  if (amdf_status_is_ok(status)) {
    rj_vm_map_t map = {
        .length = 4096,
        .offset = (int64_t)(create.doorbell_offset & ~UINT64_C(4095)),
        .prot = PROT_READ | PROT_WRITE,
        .flags = MAP_SHARED,
    };
    status = amdf_rocjitsu_source_map(device->gpu->source, &map);
    if (amdf_status_is_ok(status)) {
      queue->doorbell_mapping = map.mapped_addr;
      queue->doorbell_address =
          map.mapped_addr + (create.doorbell_offset & 4095);
    }
  }
  if (amdf_status_is_ok(status)) {
    *out_result = (amdf_gpu_umd_user_queue_result_t){
        .queue_id = {{device->gpu->source->generation, queue->identifier}},
        .capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER |
                        AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER,
        .placement = AMDF_QUEUE_PLACEMENT_HOST_LOCAL,
        .ring_byte_length = ring_length,
    };
    *out_queue = queue;
  } else {
    const amdf_status_t release_status =
        amdf_rocjitsu_user_queue_release(queue);
    if (!amdf_status_is_ok(release_status)) {
      status = release_status;
    }
  }
  return status;
}

amdf_status_t amdf_rocjitsu_user_queue_map(
    amdf_gpu_umd_user_queue_t* queue, amdf_gpu_umd_device_t* producer_device,
    amdf_gpu_umd_user_queue_mapping_t** out_mapping,
    amdf_gpu_umd_user_queue_mapping_result_t* out_result) {
  if (producer_device != NULL && producer_device != queue->device) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  *out_result = (amdf_gpu_umd_user_queue_mapping_result_t){
      .ring_address = (uintptr_t)queue->ring->pointer,
      .read_index_address =
          (uintptr_t)queue->control->pointer + queue->read_index_byte_offset,
      .write_index_address =
          (uintptr_t)queue->control->pointer + queue->write_index_byte_offset,
      .doorbell_address = queue->doorbell_address,
      .index_bits = 64,
      .doorbell_bits = 64,
  };
  // Explicit doorbell mmap registers this VA in the simulator's GPU page
  // tables. Ring and control allocations already share their host/GPU VA.
  *out_mapping = (amdf_gpu_umd_user_queue_mapping_t*)queue;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_rocjitsu_user_queue_mapping_flush(
    amdf_gpu_umd_user_queue_mapping_t* mapping) {
  (void)mapping;
  // Coherent queue mappings advertise no HOST_API publication operation.
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

amdf_status_t amdf_rocjitsu_user_queue_mapping_destroy(
    amdf_gpu_umd_user_queue_mapping_t* mapping) {
  (void)mapping;
  return AMDF_STATUS_OK;
}

static void amdf_rocjitsu_user_queue_sample_progress(
    amdf_gpu_umd_user_queue_t* queue, amdf_user_queue_status_t* out_status) {
  const amdf_atomic_uint64_t* write_index =
      (const void*)((const uint8_t*)queue->control->pointer +
                    queue->write_index_byte_offset);
  const void* read_index =
      (const uint8_t*)queue->control->pointer + queue->read_index_byte_offset;
  const bool aql = queue->command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL;
  uint64_t producer;
  uint64_t consumer;
  do {
    producer = amdf_atomic_uint64_load_acquire(write_index);
    // AQL writes a monotonic packet index. PM4 follows MEC's ring-relative
    // DWORD RPTR writeback, leaving the upper storage word zero.
    consumer = aql ? amdf_atomic_uint64_load_acquire(read_index)
                   : amdf_atomic_uint32_load_acquire(read_index);
  } while (producer != amdf_atomic_uint64_load_acquire(write_index));
  const uint64_t mask = aql ? UINT64_MAX : queue->ring->byte_length / 4 - 1;
  amdf_status_t terminal =
      amdf_atomic_uint64_load_acquire(&queue->terminal_status);
  if (amdf_status_is_ok(terminal)) {
    terminal = amdf_rocjitsu_source_status(queue->device->gpu->source);
    if (!amdf_status_is_ok(terminal)) {
      amdf_rocjitsu_user_queue_record_failure(queue, terminal);
    }
  }
  amdf_queue_state_t state = amdf_status_is_ok(terminal)
                                 ? AMDF_QUEUE_STATE_ACTIVE
                                 : AMDF_QUEUE_STATE_DEVICE_LOST;
  if (aql && amdf_status_is_ok(terminal)) {
    const uint64_t reason = amdf_atomic_uint64_load_acquire(
        amdf_rocjitsu_user_queue_inactive_value(queue));
    if (reason != 0 && !(queue->scratch_service &&
                         amdf_rocjitsu_user_queue_is_scratch_request(reason))) {
      state = AMDF_QUEUE_STATE_FAILED;
      terminal =
          reason > UINT32_MAX
              ? amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL)
              : amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE, (uint32_t)reason);
      amdf_rocjitsu_user_queue_record_failure(queue, terminal);
    }
  }
  *out_status = (amdf_user_queue_status_t){
      .type = AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS,
      .structure_size = sizeof(*out_status),
      .state = state,
      .producer_index = producer,
      .consumed_index = producer - ((producer - consumer) & mask),
      .terminal_status = terminal,
  };
}

amdf_status_t amdf_rocjitsu_user_queue_query_status(
    amdf_gpu_umd_user_queue_t* queue, amdf_user_queue_status_t* out_status) {
  amdf_rocjitsu_user_queue_sample_progress(queue, out_status);
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_rocjitsu_user_queue_query_scratch(
    amdf_gpu_umd_user_queue_t* queue,
    amdf_gpu_aql_scratch_request_t* out_request) {
  if (!queue->scratch_service) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_user_queue_status_t progress;
  amdf_rocjitsu_user_queue_sample_progress(queue, &progress);
  if (!amdf_status_is_ok(progress.terminal_status)) {
    return progress.terminal_status;
  }
  return amdf_gpu_aql_scratch_query(&queue->device->gpu->aql,
                                    queue->control->pointer,
                                    queue->ring->pointer, out_request);
}

amdf_status_t amdf_rocjitsu_user_queue_resume_scratch(
    amdf_gpu_umd_user_queue_t* queue,
    const amdf_gpu_umd_queue_scratch_t* scratch) {
  amdf_gpu_aql_scratch_request_t request;
  amdf_status_t status =
      amdf_rocjitsu_user_queue_query_scratch(queue, &request);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (request.byte_length == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION);
  }
  if (scratch->maximum_private_segment_byte_length <
      request.private_segment_byte_length) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const amdf_gpu_aql_descriptor_profile_t* profile = &queue->device->gpu->aql;
  const uint64_t alignment = UINT64_C(1) << profile->scratch.wave_size_shift;
  const uint64_t wave_byte_length =
      ((uint64_t)scratch->maximum_private_segment_byte_length *
           request.wavefront_size +
       alignment - 1) &
      ~(alignment - 1);
  status = amdf_gpu_aql_scratch_initialize(profile, scratch, wave_byte_length,
                                           queue->control->pointer);
  if (amdf_status_is_ok(status)) {
    amdf_atomic_uint64_store_release(
        amdf_rocjitsu_user_queue_inactive_value(queue), 0);
  }
  return status;
}

// A successful pause joins CP access before descriptor fields are changed.
// Failure cannot authorize scratch release and becomes a sticky queue error.
static amdf_status_t amdf_rocjitsu_user_queue_set_active(
    amdf_gpu_umd_user_queue_t* queue, bool active) {
  struct kfd_ioctl_update_queue_args update = {
      .ring_base_address = (uintptr_t)queue->ring->pointer,
      .queue_id = queue->identifier,
      .ring_size = (uint32_t)queue->ring->byte_length,
      .queue_percentage = active ? 100 : 0,
  };
  const amdf_status_t status = amdf_rocjitsu_source_command(
      queue->device->gpu->source, AMDKFD_IOC_UPDATE_QUEUE, &update,
      sizeof(update));
  if (!amdf_status_is_ok(status)) {
    amdf_rocjitsu_user_queue_record_failure(queue, status);
  }
  return status;
}

amdf_status_t amdf_rocjitsu_user_queue_trim_scratch(
    amdf_gpu_umd_user_queue_t* queue) {
  if (!queue->scratch_service) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_user_queue_status_t progress;
  amdf_rocjitsu_user_queue_sample_progress(queue, &progress);
  if (!amdf_status_is_ok(progress.terminal_status)) {
    return progress.terminal_status;
  }
  amdf_gpu_aql_descriptor_t* descriptor = queue->control->pointer;
  if (descriptor->scratch_wave64_lane_byte_length == 0) {
    return AMDF_STATUS_OK;
  }
  amdf_status_t status = amdf_rocjitsu_user_queue_set_active(queue, false);
  if (amdf_status_is_ok(status)) {
    descriptor->compute_temporary_ring_size = 0;
    memcpy(descriptor->scratch_resource_descriptor,
           queue->device->gpu->aql.scratch.resource_descriptor,
           sizeof(descriptor->scratch_resource_descriptor));
    descriptor->scratch_backing_address = 0;
    descriptor->scratch_wave64_lane_byte_length = 0;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    status = amdf_rocjitsu_user_queue_set_active(queue, true);
  }
  return status;
}

amdf_status_t amdf_rocjitsu_user_queue_set_dispatch_timestamps(
    amdf_gpu_umd_user_queue_t* queue, bool enabled) {
  (void)queue;
  (void)enabled;
  // No clock domain or timestamp capability is advertised by this backend.
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

amdf_status_t amdf_rocjitsu_user_queue_wait_consumed(
    amdf_gpu_umd_user_queue_t* queue, uint64_t published_index,
    const amdf_wait_deadline_t* deadline) {
  amdf_user_queue_status_t progress;
  amdf_rocjitsu_user_queue_sample_progress(queue, &progress);
  if (published_index > progress.producer_index) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  for (;;) {
    if (!amdf_status_is_ok(progress.terminal_status)) {
      return progress.terminal_status;
    }
    if (progress.consumed_index >= published_index) {
      return AMDF_STATUS_OK;
    }
    amdf_wait_budget_t remaining;
    const amdf_status_t status =
        amdf_wait_deadline_query_remaining(deadline, &remaining);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (remaining.timeout == 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
    }
    if (remaining.poll == 0) {
      amdf_platform_wait_yield();
    }
    amdf_rocjitsu_user_queue_sample_progress(queue, &progress);
  }
}

amdf_status_t amdf_rocjitsu_user_queue_destroy(
    amdf_gpu_umd_user_queue_t* queue) {
  if (!queue->inactivation_attempted && queue->created) {
    amdf_user_queue_status_t progress;
    amdf_rocjitsu_user_queue_sample_progress(queue, &progress);
    if (progress.consumed_index < progress.producer_index) {
      return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
    }
    if (progress.consumed_index > progress.producer_index) {
      queue->created = false;
      queue->inactivation_status =
          amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
  }
  return amdf_rocjitsu_user_queue_release(queue);
}

amdf_status_t amdf_rocjitsu_user_queue_inactivate(
    amdf_gpu_umd_user_queue_t* queue) {
  if (queue->inactivation_attempted) {
    return queue->inactivation_status;
  }
  queue->inactivation_attempted = true;
  struct kfd_ioctl_destroy_queue_args destroy = {.queue_id = queue->identifier};
  const amdf_status_t status = amdf_rocjitsu_source_command(
      queue->device->gpu->source, AMDKFD_IOC_DESTROY_QUEUE, &destroy,
      sizeof(destroy));
  queue->created = false;
  queue->inactivation_status = status;
  amdf_rocjitsu_user_queue_record_failure(
      queue, amdf_status_is_ok(status)
                 ? amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION)
                 : status);
  return status;
}
