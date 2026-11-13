// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/code_queue.h"

#include "iree/hal/driver/amd/fabric/event.h"
#include "iree/hal/driver/amd/gpu/queue/aql/ring.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/barrier.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/commands.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/kmq/storage.h"
#include "iree/hal/driver/amd/gpu/queue/user_queue.h"
#include "iree/hal/driver/amd/status.h"

enum {
  IREE_HAL_AMD_GPU_CODE_QUEUE_CAPACITY = 65536,
  IREE_HAL_AMD_GPU_CODE_QUEUE_COMMAND_OFFSET = 64,
  // Maximum native wait between authoritative publication-status queries.
  IREE_HAL_AMD_GPU_CODE_QUEUE_RECHECK_MILLISECONDS = 5,
  // Leaves room for the single-XCC prefix, native write header, cache action,
  // completion, and KERNEL alignment. The write body also fits PRED_EXEC.
  IREE_HAL_AMD_GPU_CODE_QUEUE_DATA_WORDS = 16320,
};

static uint64_t iree_hal_amd_gpu_code_queue_timeout_nanoseconds(
    iree_timeout_t timeout) {
  return iree_timeout_is_infinite(timeout)
             ? AMDF_TIMEOUT_INFINITE
             : (uint64_t)iree_timeout_as_duration_ns(timeout);
}

struct iree_hal_amd_gpu_code_queue_t {
  // Host metadata allocator; publication allocates no host storage.
  iree_allocator_t host_allocator;
  // Borrowed native lifetime, including on failed cleanup.
  iree_hal_amd_gpu_context_t* context;
  // Serializes the sole command buffer and completion between loading callers.
  iree_slim_mutex_t mutex;
  // Selected compute protocol, independent of the target's instruction set.
  bool aql;
  // USER native queue and mapping, absent for KERNEL publication.
  iree_hal_amd_gpu_user_queue_t user;
  // KERNEL command storage and checked native retirement, absent for USER.
  iree_hal_amd_gpu_pm4_kmq_storage_t* kernel;
  // USER completion signal and indirect command buffer.
  iree_hal_amd_fabric_memory_t control;
  // Exact storage used by this publisher, borrowed from control or kernel.
  iree_hal_amd_fabric_memory_t* command_memory;
  // GPU address of command_memory byte zero.
  uint64_t command_address;
  // Directly published AQL operands, including the monotonic packet position.
  iree_hal_amd_gpu_aql_ring_t aql_ring;
  // Directly published PM4 operands; each publication occupies sixteen DWORDs.
  struct {
    // Borrowed primary ring, used only for IB entry and event notification.
    uint32_t* words;
    // Native ring capacity in DWORDs, a power of two.
    uint32_t capacity;
    // Borrowed monotonic host producer index.
    iree_atomic_int64_t* write_index;
    // Borrowed write-only 64-bit MMIO doorbell.
    volatile uint64_t* doorbell;
    // End of the last accepted primary-ring publication.
    uint64_t published;
  } pm4_ring;
  // Private USER wake: ordinary retirement never consumes its notifications.
  amdf_gpu_event_t* event;
  // Immutable USER notification operands.
  amdf_gpu_event_info_t event_info;
  // Private KERNEL wake, also signaled by terminal closure.
  iree_async_event_native_t kernel_event;
  // Complete target-specific instruction and data cache action.
  uint32_t cache_control;
  // First native failure, protected by mutex until loaders have joined.
  amdf_status_t failure;
  // Parent closes publication before joining a blocked loading caller.
  iree_atomic_int64_t closing_failure;
  // Failed publication's existing target, retained through native stop.
  iree_hal_amd_fabric_memory_t unretired;
};

iree_status_t iree_hal_amd_gpu_code_queue_allocate(
    iree_hal_amd_gpu_context_t* context, uint32_t native_family_ordinal,
    amdf_queue_publication_modes_t publication_mode,
    const iree_hal_amd_fabric_memory_options_t* control,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_code_queue_t** out_queue) {
  *out_queue = NULL;
  iree_hal_amd_gpu_code_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*queue), (void**)&queue));
  queue->host_allocator = host_allocator;
  queue->context = context;
  queue->aql = context->families[native_family_ordinal].command_type ==
               AMDF_QUEUE_COMMAND_TYPE_GPU_AQL;
  iree_slim_mutex_initialize(&queue->mutex);
  *out_queue = queue;
  if (context->target_info.gfx_ip.major != 9) {
    iree_hal_amd_gpu_pm4_barrier_profile_t profile;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_barrier_profile_initialize(
        (iree_hal_amd_gpu_gfxip_version_t){
            context->target_info.gfx_ip.major,
            context->target_info.gfx_ip.minor,
            context->target_info.gfx_ip.stepping},
        &profile));
    queue->cache_control = profile.acquire | 1u;  // GLI_INV.
  }
  if (publication_mode == AMDF_QUEUE_PUBLICATION_MODE_KERNEL) {
    IREE_RETURN_IF_ERROR(
        iree_async_event_native_initialize(&queue->kernel_event));
    iree_hal_amd_gpu_pm4_ring_t ring;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_kmq_storage_allocate(
        context, native_family_ordinal, IREE_HAL_AMD_GPU_CODE_QUEUE_CAPACITY, 1,
        control, host_allocator, &queue->kernel, &ring));
    queue->command_memory = &queue->kernel->memory;
    queue->command_address = queue->kernel->address;
    return iree_ok_status();
  }

  const amdf_gpu_event_create_info_t event_create = {
      .type = AMDF_STRUCTURE_TYPE_GPU_EVENT_CREATE_INFO,
      .structure_size = sizeof(event_create),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->gpu->event_create(context->device, &event_create, &queue->event),
      "create code publication event"));
  queue->event_info = (amdf_gpu_event_info_t){
      .type = AMDF_STRUCTURE_TYPE_GPU_EVENT_INFO,
      .structure_size = sizeof(queue->event_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->gpu->event_query_info(queue->event, &queue->event_info),
      "query code publication event"));
  if (queue->event_info.format != AMDF_GPU_EVENT_FORMAT_KFD_SIGNAL ||
      queue->event_info.format_version != 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "compute code publication requires native USER signals");
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_memory_allocate(
      context->api, control, IREE_HAL_AMD_GPU_CODE_QUEUE_CAPACITY, 64,
      context->event_sink, &queue->control, NULL));
  queue->command_memory = &queue->control;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->memory_query_address(queue->control.handle, 0,
                                         AMDF_MEMORY_ADDRESS_GPU,
                                         &queue->command_address),
      "query code publication control address"));
  if (queue->command_address >=
          (UINT64_C(1) << 48) - IREE_HAL_AMD_GPU_CODE_QUEUE_CAPACITY ||
      (queue->command_address & 63)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "code publication control address is invalid");
  }
  iree_hal_amd_gpu_aql_signal_t* signal = (void*)queue->control.contents.data;
  *signal = (iree_hal_amd_gpu_aql_signal_t){
      .kind = 1,
      .event_mailbox = queue->event_info.mailbox_address,
      .event_id = queue->event_info.interrupt_context,
  };
  const amdf_gpu_user_queue_create_info_t queue_create = {
      .type = AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO,
      .structure_size = sizeof(queue_create),
      .queue_family_ordinal = native_family_ordinal,
      .priority = AMDF_QUEUE_PRIORITY_NORMAL,
      .placement = AMDF_QUEUE_PLACEMENT_HOST_LOCAL,
      .producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE,
      .required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
      .ring_byte_length = 4096,
  };
  amdf_user_queue_mapping_info_t mapping;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_user_queue_initialize(
      context, &queue_create, &queue->user, &mapping, NULL));
  if (mapping.index_bits != 64 || mapping.doorbell_bits != 64 ||
      !mapping.read_index_address || !mapping.write_index_address ||
      !mapping.doorbell_address) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "code publication requires 64-bit queue operands");
  }
  if (queue->aql) {
    iree_hal_amd_gpu_aql_ring_initialize(
        (void*)(uintptr_t)mapping.ring_address, 0,
        (uint32_t)(mapping.ring_byte_length / 64),
        (void*)(uintptr_t)mapping.read_index_address,
        (void*)(uintptr_t)mapping.write_index_address,
        (void*)(uintptr_t)mapping.doorbell_address,
        queue->user.host_publish_executor, &queue->aql_ring);
  } else {
    queue->pm4_ring.words = (void*)(uintptr_t)mapping.ring_address;
    queue->pm4_ring.capacity = (uint32_t)(mapping.ring_byte_length / 4);
    queue->pm4_ring.write_index = (void*)(uintptr_t)mapping.write_index_address;
    queue->pm4_ring.doorbell = (void*)(uintptr_t)mapping.doorbell_address;
  }
  return iree_ok_status();
}

static uint32_t* iree_hal_amd_gpu_code_queue_emit_cache(
    const iree_hal_amd_gpu_code_queue_t* queue, uint64_t address,
    uint64_t byte_length, uint32_t* words) {
  if (queue->context->target_info.gfx_ip.major != 9) {
    return iree_hal_amd_gpu_pm4_emit_acquire(words, queue->cache_control);
  }
  // GFX9 uses the coherency-control form. Every XCC executes this action,
  // independently of the single executor used for the preceding writes.
  const uint64_t granules = (byte_length + 255) >> 8;
  words[0] = UINT32_C(0xc0055800);
  words[1] = UINT32_C(0x28840000);  // SH_ICACHE, SH_KCACHE, TC inv/writeback.
  words[2] = (uint32_t)granules;
  words[3] = (uint32_t)(granules >> 32);
  words[4] = (uint32_t)(address >> 8);
  words[5] = (uint32_t)(address >> 40);
  words[6] = 0x0a;
  return words + 7;
}

static amdf_status_t iree_hal_amd_gpu_code_queue_submit_user(
    iree_hal_amd_gpu_code_queue_t* queue, uint32_t word_count) {
  iree_hal_amd_gpu_aql_signal_t* signal = (void*)queue->control.contents.data;
  iree_atomic_store((iree_atomic_int64_t*)&signal->value, 1,
                    iree_memory_order_release);
  const amdf_status_t flush_status =
      queue->context->api->host_mapping_cache_control(
          queue->control.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
          IREE_HAL_AMD_GPU_CODE_QUEUE_COMMAND_OFFSET + word_count * 4);
  if (!amdf_status_is_ok(flush_status)) {
    return flush_status;
  }
  if (queue->aql) {
    const iree_hal_amd_gpu_aql_packet_t packet =
        iree_hal_amd_gpu_aql_packet_indirect_buffer(
            IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM |
                IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM,
            queue->command_address + IREE_HAL_AMD_GPU_CODE_QUEUE_COMMAND_OFFSET,
            word_count, queue->command_address);
    uint64_t position = 0;
    // Each call joins its packet before another publication. That execution
    // proof also returns its primary-ring slot; no native read-index wait is
    // needed even if the asynchronous read-pointer writeback has not arrived.
    queue->aql_ring.consumed = queue->aql_ring.reserved;
    const bool reserved =
        iree_hal_amd_gpu_aql_ring_try_reserve(&queue->aql_ring, 1, &position);
    IREE_ASSERT(reserved);
    (void)reserved;
    iree_hal_amd_gpu_aql_ring_write_body(&queue->aql_ring, position, &packet);
    const amdf_status_t status = iree_hal_amd_gpu_aql_ring_publish_header(
        &queue->aql_ring, &queue->user, position, packet.words[0]);
    if (amdf_status_is_ok(status)) {
      iree_hal_amd_gpu_aql_ring_notify(&queue->aql_ring,
                                       queue->aql_ring.reserved - 1);
    }
    return status;
  }
  // One completed IB leaves at most its notification/padding tail unproven.
  // Sixteen-DWORD publications therefore never exhaust this larger ring.
  uint32_t* words = queue->pm4_ring.words + (queue->pm4_ring.published &
                                             (queue->pm4_ring.capacity - 1));
  words = iree_hal_amd_gpu_pm4_emit_indirect_buffer(
      words,
      queue->command_address + IREE_HAL_AMD_GPU_CODE_QUEUE_COMMAND_OFFSET,
      word_count, false);
  words = iree_hal_amd_gpu_pm4_emit_notify32(
      words, 0, queue->event_info.mailbox_address,
      queue->event_info.trigger_data, queue->event_info.interrupt_context);
  iree_hal_amd_gpu_pm4_emit_nop(words, 4);
  if (queue->user.host_publish_executor !=
      AMDF_CACHE_TRANSITION_EXECUTOR_NONE) {
    const amdf_status_t status =
        iree_hal_amd_gpu_user_queue_flush(&queue->user);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  queue->pm4_ring.published += 16;
  iree_atomic_store(queue->pm4_ring.write_index, queue->pm4_ring.published,
                    iree_memory_order_release);
  const iree_hal_memory_transition_recipe_info_t fence = {
      .host = {.fence_after = IREE_HAL_HOST_CACHE_FENCE_X86_SFENCE},
  };
  iree_hal_memory_transition_execute_host_direct(&fence, NULL, 1);
  *queue->pm4_ring.doorbell = queue->pm4_ring.published;
  return AMDF_STATUS_OK;
}

static iree_status_t iree_hal_amd_gpu_code_queue_wait(
    iree_hal_amd_gpu_code_queue_t* queue, iree_timeout_t timeout) {
  const amdf_native_event_t notification =
      iree_hal_amd_fabric_event_native_handle(&queue->kernel_event);
  if (queue->kernel && amdf_status_is_ok(queue->failure)) {
    queue->failure = queue->context->api->kernel_queue_request_notification(
        queue->kernel->queue, queue->kernel->last_submission, &notification);
  }
  while (amdf_status_is_ok(queue->failure)) {
    queue->failure = (amdf_status_t)iree_atomic_load(&queue->closing_failure,
                                                     iree_memory_order_acquire);
    if (!amdf_status_is_ok(queue->failure)) {
      break;
    }
    if (queue->kernel) {
      uint64_t retired = 0;
      queue->failure =
          iree_hal_amd_gpu_pm4_kmq_storage_refresh(queue->kernel, &retired);
      if (!amdf_status_is_ok(queue->failure) ||
          retired >= queue->kernel->last_submission) {
        break;
      }
    } else {
      iree_hal_amd_gpu_aql_signal_t* signal =
          (void*)queue->control.contents.data;
      const bool completed =
          queue->aql ? iree_atomic_load((iree_atomic_int64_t*)&signal->value,
                                        iree_memory_order_acquire) == 0
                     : iree_atomic_load((iree_atomic_int32_t*)&signal->value,
                                        iree_memory_order_acquire) == 0;
      if (completed) {
        break;
      }
    }
    if (iree_timeout_as_duration_ns(timeout) == 0) {
      queue->failure = amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
      break;
    }
    const iree_timeout_t wait_timeout = iree_timeout_min(
        timeout,
        iree_make_timeout_ms(IREE_HAL_AMD_GPU_CODE_QUEUE_RECHECK_MILLISECONDS));
    if (queue->kernel) {
      // The private event is wakeable by close. Each expiry returns to the
      // authoritative native retirement query without resubmitting work.
      iree_status_t wait_status =
          iree_async_event_native_wait(&queue->kernel_event, wait_timeout);
      if (iree_status_is_deadline_exceeded(wait_status)) {
        iree_status_free(wait_status);
        continue;
      }
      if (!iree_status_is_ok(wait_status)) {
        return wait_status;
      }
    } else {
      queue->failure = queue->context->gpu->event_wait(
          queue->event,
          iree_hal_amd_gpu_code_queue_timeout_nanoseconds(wait_timeout));
      if (queue->failure ==
          amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED)) {
        queue->failure = AMDF_STATUS_OK;
      }
    }
  }
  return IREE_HAL_AMD_STATUS_FROM_AMDF(queue->failure,
                                       "wait for compute code publication");
}

iree_status_t iree_hal_amd_gpu_code_queue_publish(
    iree_hal_amd_gpu_code_queue_t* queue, iree_timeout_t timeout,
    iree_hal_amd_fabric_memory_t* memory, iree_const_byte_span_t contents) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_convert_timeout_to_absolute(&timeout);
  uint64_t address = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_HAL_AMD_STATUS_FROM_AMDF(
              queue->context->api->memory_query_address(
                  memory->handle, 0, AMDF_MEMORY_ADDRESS_GPU, &address),
              "query code publication address"));
  iree_slim_mutex_lock(&queue->mutex);
  const amdf_status_t closing = (amdf_status_t)iree_atomic_load(
      &queue->closing_failure, iree_memory_order_acquire);
  if (!amdf_status_is_ok(closing) && amdf_status_is_ok(queue->failure)) {
    queue->failure = closing;
  }
  iree_status_t status = IREE_HAL_AMD_STATUS_FROM_AMDF(
      queue->failure, "compute code publisher is closed");
  bool accepted = false;
  bool final = false;
  for (iree_host_size_t offset = 0; !final && iree_status_is_ok(status);) {
    const uint32_t count =
        (uint32_t)iree_min((contents.data_length - offset) / 4,
                           IREE_HAL_AMD_GPU_CODE_QUEUE_DATA_WORDS);
    final = offset + count * 4 == contents.data_length;
    uint32_t* begin = (void*)(queue->command_memory->contents.data +
                              IREE_HAL_AMD_GPU_CODE_QUEUE_COMMAND_OFFSET);
    uint32_t* words = begin;
    if (count) {
      if (queue->aql) {
        words = iree_hal_amd_gpu_aql_emit_single_executor(
            words, queue->context->target_info.topology.xcc_count, count + 4);
      }
      words = iree_hal_amd_gpu_pm4_emit_write(
          words, address + offset, count,
          (const uint32_t*)(contents.data + offset));
    }
    if (final) {
      words = iree_hal_amd_gpu_code_queue_emit_cache(
          queue, address, memory->info.byte_length, words);
    }
    if (queue->kernel) {
      uint32_t padding = (uint32_t)(-(words - begin)) & 7u;
      if (padding == 1) {
        padding += 8;
      }
      if (padding) {
        words = iree_hal_amd_gpu_pm4_emit_nop(words, padding);
      }
      queue->kernel->pending = (amdf_gpu_kernel_command_t){
          .memory = queue->command_memory->handle,
          .byte_offset = IREE_HAL_AMD_GPU_CODE_QUEUE_COMMAND_OFFSET,
          .byte_length = (uint64_t)(words - begin) * 4,
      };
      queue->failure = queue->context->api->host_mapping_cache_control(
          queue->command_memory->mapping, AMDF_HOST_CACHE_OPERATION_FLUSH,
          IREE_HAL_AMD_GPU_CODE_QUEUE_COMMAND_OFFSET,
          queue->kernel->pending.byte_length);
      if (amdf_status_is_ok(queue->failure)) {
        queue->failure =
            iree_hal_amd_gpu_pm4_kmq_storage_submit(queue->kernel, 0);
      }
    } else {
      if (!queue->aql) {
        words = iree_hal_amd_gpu_pm4_emit_release32(
            words, 0,
            queue->command_address +
                offsetof(iree_hal_amd_gpu_aql_signal_t, value),
            0);
      }
      queue->failure = iree_hal_amd_gpu_code_queue_submit_user(
          queue, (uint32_t)(words - begin));
    }
    accepted = amdf_status_is_ok(queue->failure);
    status = iree_hal_amd_gpu_code_queue_wait(queue, timeout);
    if (iree_status_is_ok(status)) {
      accepted = false;
      offset += count * 4;
    }
  }
  if (accepted) {
    queue->unretired = *memory;
    *memory = (iree_hal_amd_fabric_memory_t){.info = memory->info};
  }
  // An OS event failure has an IREE diagnostic, not a native status. Preserve
  // that original result while closing the native ownership domain as well.
  if (!iree_status_is_ok(status) && amdf_status_is_ok(queue->failure)) {
    queue->failure = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  const amdf_status_t failure = queue->failure;
  iree_slim_mutex_unlock(&queue->mutex);
  if (!amdf_status_is_ok(failure)) {
    iree_hal_amd_gpu_context_fail_execution(queue->context, failure,
                                            "compute code publication");
    iree_hal_amd_gpu_context_wake_failure(queue->context);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

amdf_status_t iree_hal_amd_gpu_code_queue_close(
    iree_hal_amd_gpu_code_queue_t* queue, amdf_status_t failure) {
  if (!queue) {
    return AMDF_STATUS_OK;
  }
  int64_t expected = 0;
  iree_atomic_compare_exchange_strong(&queue->closing_failure, &expected,
                                      failure, iree_memory_order_release,
                                      iree_memory_order_relaxed);
  if (queue->event) {
    return queue->context->gpu->event_signal(queue->event);
  }
  if (queue->kernel) {
    iree_async_event_native_set(&queue->kernel_event);
  }
  return AMDF_STATUS_OK;
}

bool iree_hal_amd_gpu_code_queue_is_host_idle(
    iree_hal_amd_gpu_code_queue_t* queue) {
  if (!queue) {
    return true;
  }
  if (!iree_slim_mutex_try_lock(&queue->mutex)) {
    return false;
  }
  iree_slim_mutex_unlock(&queue->mutex);
  return true;
}

bool iree_hal_amd_gpu_code_queue_try_inactivate(
    iree_hal_amd_gpu_code_queue_t* queue, amdf_status_t* out_status) {
  if (!queue) {
    *out_status = AMDF_STATUS_OK;
    return true;
  }
  if (!iree_slim_mutex_try_lock(&queue->mutex)) {
    return false;
  }
  *out_status = queue->kernel
                    ? iree_hal_amd_gpu_pm4_kmq_storage_inactivate(queue->kernel)
                    : iree_hal_amd_gpu_user_queue_inactivate(&queue->user);
  iree_slim_mutex_unlock(&queue->mutex);
  return true;
}

amdf_status_t iree_hal_amd_gpu_code_queue_free(
    iree_hal_amd_gpu_code_queue_t* queue) {
  if (!queue) {
    return AMDF_STATUS_OK;
  }
  const bool inactivated =
      queue->kernel ? queue->kernel->inactivated : queue->user.inactivated;
  if (!amdf_status_is_ok(queue->failure) && !inactivated) {
    return queue->failure;
  }
  if (queue->kernel) {
    const amdf_status_t status =
        iree_hal_amd_gpu_pm4_kmq_storage_free(queue->kernel);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    queue->kernel = NULL;
  }
  if (queue->user.context) {
    const amdf_status_t status = iree_hal_amd_gpu_user_queue_deinitialize(
        &queue->user,
        queue->aql ? queue->aql_ring.reserved : queue->pm4_ring.published);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  if (queue->event) {
    const amdf_status_t status =
        queue->context->gpu->event_destroy(queue->event);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  iree_async_event_native_deinitialize(&queue->kernel_event);
  iree_hal_amd_fabric_memory_deinitialize(
      queue->context->api, queue->context->event_sink, &queue->control);
  iree_hal_amd_fabric_memory_deinitialize(
      queue->context->api, queue->context->event_sink, &queue->unretired);
  iree_slim_mutex_deinitialize(&queue->mutex);
  iree_allocator_free(queue->host_allocator, queue);
  return AMDF_STATUS_OK;
}
