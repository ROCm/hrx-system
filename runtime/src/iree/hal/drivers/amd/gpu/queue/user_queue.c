// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/user_queue.h"

#include "iree/hal/drivers/amd/fabric/memory_transition.h"
#include "iree/hal/drivers/amd/status.h"

static iree_status_t iree_hal_amd_gpu_user_queue_map(
    const iree_hal_amd_gpu_context_t* context, amdf_user_queue_t* queue,
    amdf_device_t* producer, amdf_user_queue_mapping_t** out_mapping,
    amdf_user_queue_mapping_info_t* out_info) {
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->user_queue_map(queue, producer, out_mapping),
      "user_queue_map"));
  *out_info = (amdf_user_queue_mapping_info_t){
      .type = AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO,
      .structure_size = sizeof(*out_info),
  };
  return IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->user_queue_mapping_query_info(*out_mapping, out_info),
      "user_queue_mapping_query_info");
}

iree_status_t iree_hal_amd_gpu_user_queue_initialize(
    const iree_hal_amd_gpu_context_t* context,
    const amdf_gpu_user_queue_create_info_t* create_info,
    iree_hal_amd_gpu_user_queue_t* out_storage,
    amdf_user_queue_mapping_info_t* out_mapping_info,
    amdf_user_queue_mapping_info_t* out_device_mapping_info) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_storage = (iree_hal_amd_gpu_user_queue_t){.context = context};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_HAL_AMD_STATUS_FROM_AMDF(
              context->gpu->user_queue_create(context->device, create_info,
                                              &out_storage->queue),
              "gpu.user_queue_create"));
  amdf_user_queue_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_USER_QUEUE_INFO,
      .structure_size = sizeof(info),
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_HAL_AMD_STATUS_FROM_AMDF(
              context->api->user_queue_query_info(out_storage->queue, &info),
              "user_queue_query_info"));
  // These ABI vocabularies intentionally use identical locality values.
  static_assert(
      (iree_hal_queue_placement_t)AMDF_QUEUE_PLACEMENT_DEFAULT ==
              IREE_HAL_QUEUE_PLACEMENT_DEFAULT &&
          (iree_hal_queue_placement_t)AMDF_QUEUE_PLACEMENT_HOST_LOCAL ==
              IREE_HAL_QUEUE_PLACEMENT_HOST_LOCAL &&
          (iree_hal_queue_placement_t)AMDF_QUEUE_PLACEMENT_DEVICE_LOCAL ==
              IREE_HAL_QUEUE_PLACEMENT_DEVICE_LOCAL,
      "native and HAL queue locality values agree");
  out_storage->placement = (iree_hal_queue_placement_t)info.placement;
  amdf_user_queue_mapping_info_t mapping_info;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_amd_gpu_user_queue_map(context, out_storage->queue, NULL,
                                      &out_storage->mapping, &mapping_info));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_amd_fabric_memory_translate_host_fence(
          mapping_info.host_publish_fence, &out_storage->host_publish_fence));
  out_storage->host_publish_executor = mapping_info.host_publish_executor;
  amdf_user_queue_mapping_info_t device_mapping_info;
  iree_status_t status = iree_ok_status();
  if (out_device_mapping_info) {
    status = iree_hal_amd_gpu_user_queue_map(
        context, out_storage->queue, context->device,
        &out_storage->device_mapping, &device_mapping_info);
  }
  if (iree_status_is_ok(status)) {
    *out_mapping_info = mapping_info;
    if (out_device_mapping_info) {
      *out_device_mapping_info = device_mapping_info;
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

amdf_status_t iree_hal_amd_gpu_user_queue_flush(
    const iree_hal_amd_gpu_user_queue_t* storage) {
  if (storage->host_publish_executor ==
      AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API) {
    return storage->context->api->user_queue_mapping_flush(storage->mapping);
  }
  const iree_hal_memory_transition_recipe_info_t fence = {
      .host = {.fence_after = storage->host_publish_fence},
  };
  iree_hal_memory_transition_execute_host_direct(&fence, NULL, 1);
  return AMDF_STATUS_OK;
}

amdf_status_t iree_hal_amd_gpu_user_queue_deinitialize(
    iree_hal_amd_gpu_user_queue_t* storage, uint64_t published_index) {
  if (!amdf_status_is_ok(storage->removal_status)) {
    return storage->removal_status;
  }
  const amdf_api_t* api = storage->context->api;
  if (storage->queue && published_index) {
    amdf_status_t status = api->user_queue_wait_consumed(
        storage->queue, published_index, AMDF_TIMEOUT_INFINITE, 0);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  if (storage->device_mapping) {
    amdf_status_t status =
        api->user_queue_mapping_destroy(storage->device_mapping);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    storage->device_mapping = NULL;
  }
  if (storage->mapping) {
    amdf_status_t status = api->user_queue_mapping_destroy(storage->mapping);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    storage->mapping = NULL;
  }
  if (storage->queue) {
    amdf_status_t status = api->user_queue_destroy(storage->queue);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
      return status;
    }
    storage->queue = NULL;
    storage->removal_status = status;
    return status;
  }
  return AMDF_STATUS_OK;
}
