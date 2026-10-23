// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/code.h"

#include "iree/hal/drivers/amd/gpu/queue/aql/ring.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/barrier.h"
#include "iree/hal/drivers/amd/gpu/queue/user_queue.h"
#include "iree/hal/drivers/amd/status.h"

struct iree_hal_amd_gpu_aql_code_t {
  // Host metadata allocator; no publication allocates host storage.
  iree_allocator_t host_allocator;
  // Borrowed native lifetime, including on failed cleanup.
  iree_hal_amd_gpu_context_t* context;
  // Serializes the sole cache IB and completion signal between loading callers.
  iree_slim_mutex_t mutex;
  // Cache-only queue, independent of all user work and native allocations.
  iree_hal_amd_gpu_user_queue_t queue;
  // Single-producer native packet operands.
  iree_hal_amd_gpu_aql_ring_t ring;
  // Coherent control page containing a signal followed by the cache IB.
  iree_hal_amd_fabric_memory_t control;
  // Private wake source: ordinary retirement never consumes its notifications.
  amdf_gpu_event_t* event;
  // Immutable vendor packet; only its pointed-to IB and signal are reused.
  iree_hal_amd_gpu_aql_packet_t packet;
  // Terminal native failure, protected by mutex until loaders have joined.
  amdf_status_t failure;
};

iree_status_t iree_hal_amd_gpu_aql_code_allocate(
    iree_hal_amd_gpu_context_t* context, uint32_t native_family_ordinal,
    const iree_hal_amd_fabric_memory_options_t* control,
    iree_allocator_t host_allocator, iree_hal_amd_gpu_aql_code_t** out_code) {
  *out_code = NULL;
  iree_hal_amd_gpu_aql_code_t* code = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*code), (void**)&code));
  code->host_allocator = host_allocator;
  code->context = context;
  iree_slim_mutex_initialize(&code->mutex);
  *out_code = code;
  const amdf_gpu_event_create_info_t event_create = {
      .type = AMDF_STRUCTURE_TYPE_GPU_EVENT_CREATE_INFO,
      .structure_size = sizeof(event_create),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->gpu->event_create(context->device, &event_create, &code->event),
      "create code publication event"));
  amdf_gpu_event_info_t event_info = {
      .type = AMDF_STRUCTURE_TYPE_GPU_EVENT_INFO,
      .structure_size = sizeof(event_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->gpu->event_query_info(code->event, &event_info),
      "query code publication event"));
  if (event_info.format != AMDF_GPU_EVENT_FORMAT_KFD_SIGNAL ||
      event_info.format_version != 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AQL code publication requires native USER signals");
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_memory_allocate(
      context->api, control, 4096, 64, context->event_sink, &code->control,
      NULL));
  uint64_t address = 0;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->memory_query_address(code->control.handle, 0,
                                         AMDF_MEMORY_ADDRESS_GPU, &address),
      "query code publication control address"));
  if (address >= (UINT64_C(1) << 48) - 4096 || (address & 63)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AQL code publication control address is invalid");
  }
  memset(code->control.contents.data, 0, 4096);
  iree_hal_amd_gpu_aql_signal_t* signal = (void*)code->control.contents.data;
  *signal = (iree_hal_amd_gpu_aql_signal_t){
      .kind = 1,
      .event_mailbox = event_info.mailbox_address,
      .event_id = event_info.interrupt_context,
  };
  uint32_t* words = (void*)(signal + 1);
  uint32_t word_count = 7;
  if (context->target_info.gfx_ip.major == 9) {
    // SH_ICACHE, SH_KCACHE, TC invalidate and writeback. The cache action is
    // unpredicated: every participating XCC must publish its code caches.
    words[0] = UINT32_C(0xc0055800);
    words[1] = UINT32_C(0x28840000);
  } else {
    iree_hal_amd_gpu_pm4_barrier_profile_t profile;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_barrier_profile_initialize(
        (iree_hal_amd_gpu_gfxip_version_t){
            context->target_info.gfx_ip.major,
            context->target_info.gfx_ip.minor,
            context->target_info.gfx_ip.stepping},
        &profile));
    words[0] = UINT32_C(0xc0065800);
    words[7] = profile.acquire | 1u;  // GLI_INV plus data publication.
    word_count = 8;
  }
  code->packet = iree_hal_amd_gpu_aql_packet_indirect_buffer(
      0, address + sizeof(*signal), word_count, address);
  const amdf_gpu_user_queue_create_info_t queue_create = {
      .type = AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO,
      .structure_size = sizeof(queue_create),
      .queue_family_ordinal = native_family_ordinal,
      .priority = AMDF_QUEUE_PRIORITY_NORMAL,
      .producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE,
      .required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
      .ring_byte_length = 4096,
  };
  amdf_user_queue_mapping_info_t mapping;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_user_queue_initialize(
      context, &queue_create, &code->queue, &mapping, NULL));
  if (mapping.index_bits != 64 || mapping.doorbell_bits != 64 ||
      !mapping.read_index_address || !mapping.write_index_address ||
      !mapping.doorbell_address) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AQL code publication requires 64-bit queue operands");
  }
  iree_hal_amd_gpu_aql_ring_initialize(
      (void*)(uintptr_t)mapping.ring_address, 0,
      (uint32_t)(mapping.ring_byte_length / 64),
      (void*)(uintptr_t)mapping.read_index_address,
      (void*)(uintptr_t)mapping.write_index_address,
      (void*)(uintptr_t)mapping.doorbell_address,
      code->queue.host_publish_executor, &code->ring);
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_aql_code_publish(
    iree_hal_amd_gpu_aql_code_t* code, iree_hal_amd_fabric_memory_t* memory) {
  IREE_TRACE_ZONE_BEGIN(z0);
  uint64_t address = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_HAL_AMD_STATUS_FROM_AMDF(
              code->context->api->memory_query_address(
                  memory->handle, 0, AMDF_MEMORY_ADDRESS_GPU, &address),
              "query code publication address"));
  iree_slim_mutex_lock(&code->mutex);
  bool published = false;
  if (amdf_status_is_ok(code->failure)) {
    iree_hal_amd_gpu_aql_signal_t* signal = (void*)code->control.contents.data;
    uint32_t* words = (void*)(signal + 1);
    const uint64_t granules = (memory->info.byte_length + 255) >> 8;
    words[2] = (uint32_t)granules;
    words[3] = (uint32_t)(granules >> 32);
    words[4] = (uint32_t)(address >> 8);
    words[5] = (uint32_t)(address >> 40);
    iree_atomic_store((iree_atomic_int64_t*)&signal->value, 1,
                      iree_memory_order_release);
    uint64_t position = 0;
    if (!iree_hal_amd_gpu_aql_ring_try_reserve(&code->ring, 1, &position)) {
      code->failure = code->context->api->user_queue_wait_consumed(
          code->queue.queue, code->ring.reserved, AMDF_TIMEOUT_INFINITE, 0);
      if (amdf_status_is_ok(code->failure)) {
        const bool reserved =
            iree_hal_amd_gpu_aql_ring_try_reserve(&code->ring, 1, &position);
        IREE_ASSERT(reserved);
        (void)reserved;
      }
    }
    if (amdf_status_is_ok(code->failure)) {
      iree_hal_amd_gpu_aql_ring_write_body(&code->ring, position,
                                           &code->packet);
      code->failure = iree_hal_amd_gpu_aql_ring_publish_header(
          &code->ring, &code->queue, position, code->packet.words[0]);
      if (amdf_status_is_ok(code->failure)) {
        iree_hal_amd_gpu_aql_ring_notify(&code->ring);
        published = true;
      }
      while (amdf_status_is_ok(code->failure) &&
             iree_atomic_load((iree_atomic_int64_t*)&signal->value,
                              iree_memory_order_acquire) != 0) {
        code->failure =
            code->context->gpu->event_wait(code->event, AMDF_TIMEOUT_INFINITE);
      }
    }
  }
  const amdf_status_t failure = code->failure;
  if (published && !amdf_status_is_ok(failure)) {
    // An incomplete cache writeback may still refer to this address. Keep the
    // native backing through failed-owner teardown, as for unretired uploads.
    memory->handle = NULL;
    memory->mapping = NULL;
    memory->contents = iree_byte_span_empty();
    iree_hal_amd_gpu_context_report(
        code->context, failure,
        "code publication retirement unproven; preserving native backing");
  }
  iree_slim_mutex_unlock(&code->mutex);
  IREE_TRACE_ZONE_END(z0);
  return IREE_HAL_AMD_STATUS_FROM_AMDF(failure,
                                       "publish AQL instruction caches");
}

amdf_status_t iree_hal_amd_gpu_aql_code_free(
    iree_hal_amd_gpu_aql_code_t* code) {
  if (!code) {
    return AMDF_STATUS_OK;
  }
  if (!amdf_status_is_ok(code->failure)) {
    return code->failure;
  }
  const iree_hal_amd_gpu_context_t* context = code->context;
  if (code->queue.context) {
    amdf_status_t status = iree_hal_amd_gpu_user_queue_deinitialize(
        &code->queue, code->ring.reserved);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  if (code->event) {
    amdf_status_t status = context->gpu->event_destroy(code->event);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  iree_hal_amd_fabric_memory_deinitialize(context->api, context->event_sink,
                                          &code->control);
  iree_slim_mutex_deinitialize(&code->mutex);
  iree_allocator_free(code->host_allocator, code);
  return AMDF_STATUS_OK;
}
