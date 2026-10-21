// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/code.h"

#include "iree/base/threading/notification.h"
#include "iree/hal/drivers/amd/fabric/memory_transition.h"
#include "iree/hal/drivers/amd/gpu/queue/sdma/queue.h"
#include "iree/hal/drivers/amd/status.h"

struct iree_hal_amd_gpu_code_upload_t {
  // Temporary native storage, released as soon as its last copy retires.
  iree_hal_amd_fabric_memory_t memory;
  // Native release never depends on allocating more host storage.
  iree_hal_amd_gpu_memory_release_t release;
  // Allocator owning this load state through native release completion.
  iree_allocator_t host_allocator;
  // Logical initialized extent, independent of native allocation granularity.
  uint64_t byte_length;
  // Reused copy record, owned through its single terminal callback.
  iree_hal_amd_gpu_sdma_copy_t copy;
  // Joins result publication and the callback's final notification access.
  iree_slim_mutex_t mutex;
  // Parks only the loading caller; the device's observer retires the copy.
  iree_notification_t notification;
  // Native result of the current bounded copy.
  amdf_status_t status;
  // Whether native consumers of the current copy have retired.
  iree_hal_amd_gpu_sdma_retirement_t retirement;
  // Callback result is ready, protected by mutex.
  bool completed;
};

static void iree_hal_amd_gpu_code_upload_released(void* user_data) {
  iree_hal_amd_gpu_code_upload_t* upload = user_data;
  iree_notification_deinitialize(&upload->notification);
  iree_slim_mutex_deinitialize(&upload->mutex);
  iree_allocator_free(upload->host_allocator, upload);
}

void iree_hal_amd_gpu_code_upload_free(iree_hal_amd_gpu_context_t* context,
                                       iree_hal_amd_gpu_code_upload_t* upload) {
  if (!upload) {
    return;
  }
  iree_hal_amd_gpu_context_release_memory(
      context, 1, &upload->memory,
      (iree_hal_amd_fabric_memory_release_callback_t){
          .fn = iree_hal_amd_gpu_code_upload_released, .user_data = upload},
      &upload->release);
}

iree_status_t iree_hal_amd_gpu_code_allocate(
    iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* resident,
    const iree_hal_amd_fabric_memory_options_t* staging, uint64_t byte_length,
    uint64_t alignment, iree_allocator_t host_allocator,
    iree_hal_amd_fabric_memory_t* out_memory,
    iree_hal_amd_gpu_code_upload_t** out_upload,
    iree_byte_span_t* out_contents) {
  *out_memory = (iree_hal_amd_fabric_memory_t){0};
  *out_upload = NULL;
  *out_contents = iree_byte_span_empty();
  iree_hal_amd_gpu_code_upload_t* upload = NULL;
  if (!resident->host_access) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*upload),
                                               (void**)&upload));
    upload->host_allocator = host_allocator;
    upload->byte_length = byte_length;
    iree_slim_mutex_initialize(&upload->mutex);
    iree_notification_initialize(&upload->notification);
  }
  const iree_hal_amd_fabric_memory_request_t requests[2] = {
      {.options = resident, .byte_length = byte_length, .alignment = alignment},
      {.options = staging, .byte_length = byte_length, .alignment = alignment},
  };
  iree_hal_amd_fabric_memory_t memories[2] = {0};
  // Loading runs on its synchronous caller. Native serialization orders
  // individual driver calls without waiting behind a whole slab batch.
  const uint32_t count = upload ? 2 : 1;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    status = iree_hal_amd_fabric_memory_allocate(
        context->api, requests[i].options, byte_length, alignment,
        context->event_sink, &memories[i], NULL);
  }
  if (iree_status_is_ok(status)) {
    *out_memory = memories[0];
    if (upload) {
      upload->memory = memories[1];
    }
    *out_upload = upload;
    *out_contents = iree_make_byte_span(memories[upload ? 1 : 0].contents.data,
                                        (iree_host_size_t)byte_length);
  } else {
    for (uint32_t i = 0; i < count; ++i) {
      iree_hal_amd_fabric_memory_deinitialize(context->api, context->event_sink,
                                              &memories[i]);
    }
    if (upload) {
      iree_hal_amd_gpu_code_upload_released(upload);
    }
  }
  return status;
}

static iree_status_t iree_hal_amd_gpu_code_queue_effect(
    const iree_hal_memory_transition_recipe_info_t* recipe, bool release,
    iree_hal_memory_effects_t* effects) {
  if (recipe->kind == IREE_HAL_MEMORY_TRANSITION_KIND_NONE) {
    return iree_ok_status();
  }
  if (recipe->kind != IREE_HAL_MEMORY_TRANSITION_KIND_GLOBAL ||
      recipe->executor != IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "code upload requires unsupported queue visibility");
  }
  effects->bits |= release ? IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM
                           : IREE_HAL_MEMORY_EFFECT_GLOBAL_ACQUIRE_FROM_SYSTEM;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_code_prepare_copy(
    iree_hal_amd_gpu_context_t* context, iree_hal_amd_gpu_sdma_queue_t* queue,
    uint32_t transfer_family_ordinal, uint32_t compute_family_ordinal,
    iree_hal_amd_fabric_memory_t* memory,
    iree_hal_amd_gpu_code_upload_t* upload) {
  if (!queue) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "unmapped local code requires a native transfer queue");
  }
  amdf_memory_site_t producer = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
      .structure_size = sizeof(producer),
      .kind = AMDF_MEMORY_SITE_KIND_HOST,
      .value.host.mapping = upload->memory.mapping,
  };
  amdf_memory_site_t consumer = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
      .structure_size = sizeof(consumer),
      .kind = AMDF_MEMORY_SITE_KIND_DEVICE,
      .value.device = {.memory = upload->memory.handle,
                       .queue_family_ordinal = transfer_family_ordinal},
  };
  amdf_memory_pair_info_t native = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = sizeof(native),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->memory_query_pair_info(&producer, &consumer, &native),
      "qualify code staging publication"));
  iree_hal_memory_pair_info_t pair;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_fabric_memory_translate_pair(&native, &pair));
  if (pair.release.kind != IREE_HAL_MEMORY_TRANSITION_KIND_NONE &&
      pair.release.executor !=
          IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_DIRECT &&
      pair.release.executor != IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "code staging requires unsupported host publication");
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_code_queue_effect(
      &pair.acquire, false, &upload->copy.effects));
  producer = consumer;
  producer.value.device.memory = memory->handle;
  consumer = producer;
  consumer.value.device.queue_family_ordinal = compute_family_ordinal;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->memory_query_pair_info(&producer, &consumer, &native),
      "qualify local code publication"));
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_fabric_memory_translate_pair(&native, &pair));
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_code_queue_effect(
      &pair.release, true, &upload->copy.effects));
  // Execution boundaries acquire uploaded data. AQL loading separately
  // invalidates instruction caches before exposing the executable.
  iree_hal_memory_effects_t compute = {0};
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_code_queue_effect(&pair.acquire, false, &compute));
  if (upload->copy.effects.bits &&
      !iree_hal_amd_gpu_sdma_queue_supports_cache_control(queue)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "code transfer queue lacks required cache control");
  }
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->memory_query_address(upload->memory.handle, 0,
                                         AMDF_MEMORY_ADDRESS_GPU,
                                         &upload->copy.source),
      "query code staging address"));
  return IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->memory_query_address(
          memory->handle, 0, AMDF_MEMORY_ADDRESS_GPU, &upload->copy.target),
      "query resident code address");
}

static void iree_hal_amd_gpu_code_copy_complete(
    void* user_data, amdf_status_t status,
    iree_hal_amd_gpu_sdma_retirement_t retirement) {
  iree_hal_amd_gpu_code_upload_t* upload = user_data;
  iree_slim_mutex_lock(&upload->mutex);
  upload->status = status;
  upload->retirement = retirement;
  upload->completed = true;
  iree_notification_post(&upload->notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&upload->mutex);
}

static bool iree_hal_amd_gpu_code_copy_is_complete(void* user_data) {
  iree_hal_amd_gpu_code_upload_t* upload = user_data;
  iree_slim_mutex_lock(&upload->mutex);
  bool completed = upload->completed;
  iree_slim_mutex_unlock(&upload->mutex);
  return completed;
}

iree_status_t iree_hal_amd_gpu_code_publish(
    iree_hal_amd_gpu_context_t* context, iree_hal_amd_gpu_sdma_queue_t* queue,
    uint32_t transfer_family_ordinal, uint32_t compute_family_ordinal,
    iree_hal_amd_fabric_memory_t* memory,
    iree_hal_amd_gpu_code_upload_t** upload_ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_amd_gpu_code_upload_t* upload = *upload_ptr;
  *upload_ptr = NULL;
  iree_hal_amd_fabric_memory_t* source = upload ? &upload->memory : memory;
  iree_status_t status = IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->host_mapping_cache_control(source->mapping,
                                               AMDF_HOST_CACHE_OPERATION_FLUSH,
                                               0, source->contents.data_length),
      "publish code host writes");
  if (upload && iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_code_prepare_copy(
        context, queue, transfer_family_ordinal, compute_family_ordinal, memory,
        upload);
  }
  const uint64_t length = upload ? upload->byte_length : 0;
  if (upload) {
    upload->copy.notify = iree_hal_amd_gpu_code_copy_complete;
    upload->copy.user_data = upload;
    upload->retirement = IREE_HAL_AMD_GPU_SDMA_RETIRED;
  }
  for (uint64_t offset = 0; offset < length && iree_status_is_ok(status);) {
    upload->copy.length = (uint32_t)iree_min(
        length - offset, IREE_HAL_AMD_GPU_SDMA_MAX_COPY_LENGTH);
    upload->completed = false;
    iree_hal_amd_gpu_sdma_queue_submit(queue, &upload->copy);
    iree_notification_await(&upload->notification,
                            iree_hal_amd_gpu_code_copy_is_complete, upload,
                            iree_infinite_timeout());
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(upload->status, "upload local code");
    offset += upload->copy.length;
    upload->copy.source += upload->copy.length;
    upload->copy.target += upload->copy.length;
  }
  if (upload && upload->retirement == IREE_HAL_AMD_GPU_SDMA_UNRETIRED) {
    // Keep geometry for sanitizer rollback, but no failed executable may free
    // native operands whose accepted copy has no retirement proof. The stopped
    // SDMA owner keeps its context alive. Native memory/mapping handles remain
    // allocated; the completed callback has returned host metadata ownership.
    memory->handle = NULL;
    memory->mapping = NULL;
    memory->contents = iree_byte_span_empty();
    iree_hal_amd_gpu_context_report(
        context, upload->status,
        "code upload retirement unproven; preserving native backing");
    iree_hal_amd_gpu_code_upload_released(upload);
  } else {
    iree_hal_amd_gpu_code_upload_free(context, upload);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}
