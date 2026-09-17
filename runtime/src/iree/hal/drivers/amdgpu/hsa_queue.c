// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/hsa_queue.h"

iree_status_t iree_hal_amdgpu_hsa_queue_create(
    const iree_hal_amdgpu_hsa_queue_params_t* params, hsa_queue_t** out_queue) {
  if (IREE_UNLIKELY((params->compute_unit_mask_bit_count == 0) !=
                    (params->compute_unit_mask == NULL))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "native compute-unit mask count and storage must both be empty or "
        "both be present");
  }
  if (IREE_UNLIKELY(params->compute_unit_mask_bit_count % 32u != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "native compute-unit mask bit count %u is not a multiple of 32",
        params->compute_unit_mask_bit_count);
  }
#if IREE_HAL_AMDGPU_HAVE_HSA_AMD_QUEUE_CREATE
  const bool use_descriptor_create =
      iree_hal_amdgpu_libhsa_has_hsa_amd_queue_create(params->libhsa);
  if (IREE_UNLIKELY(use_descriptor_create &&
                    params->packet_count >
                        UINT32_MAX / sizeof(hsa_kernel_dispatch_packet_t))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "HSA queue packet capacity %u exceeds the native queue byte-length "
        "limit",
        params->packet_count);
  }
#endif  // IREE_HAL_AMDGPU_HAVE_HSA_AMD_QUEUE_CREATE
  if (IREE_UNLIKELY(params->type != HSA_QUEUE_TYPE_MULTI &&
                    params->type != HSA_QUEUE_TYPE_COOPERATIVE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported native HSA queue type %u",
                            params->type);
  }
  const bool is_cooperative = params->type == HSA_QUEUE_TYPE_COOPERATIVE;
  if (IREE_UNLIKELY(is_cooperative &&
                    params->priority != HSA_AMD_QUEUE_PRIORITY_NORMAL)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "native cooperative queues require normal scheduling priority");
  }

  hsa_queue_t* queue = NULL;
  iree_status_t status;
#if IREE_HAL_AMDGPU_HAVE_HSA_AMD_QUEUE_CREATE
  if (use_descriptor_create) {
    hsa_amd_queue_create_desc_t descriptor = {
        .version = HSA_AMD_QUEUE_CREATE_DESC_VERSION,
        .flags = HSA_AMD_QUEUE_CREATE_SYSTEM_MEM,
        .engine_type = HSA_AMD_QUEUE_ENGINE_COMPUTE,
        .queue_size_bytes = (uint32_t)(params->packet_count *
                                       sizeof(hsa_kernel_dispatch_packet_t)),
        .priority = params->priority,
        .callback = params->error_callback,
        .callback_data = params->error_callback_data,
        .engine.compute =
            {
                .cu_mask = NULL,
                .type = params->type,
                .private_segment_size = HSA_AMD_PRIVATE_SEGMENT_SIZE_DEFAULT,
                .cu_mask_count = 0,
            },
    };
    status = iree_hsa_amd_queue_create(IREE_LIBHSA(params->libhsa),
                                       params->agent, &descriptor,
                                       /*num_descs=*/1);
    queue = descriptor.queue;
  } else
#endif  // IREE_HAL_AMDGPU_HAVE_HSA_AMD_QUEUE_CREATE
  {
    // Extension revisions before 1.26 do not expose exact queue descriptors,
    // and a dynamically loaded runtime may omit the optional entry point even
    // when the build headers declare it.
    // Create through the stable API, then apply the requested priority while
    // the queue is still private.
    status = iree_hsa_queue_create(
        IREE_LIBHSA(params->libhsa), params->agent, params->packet_count,
        params->type, params->error_callback, params->error_callback_data,
        /*private_segment_size=*/UINT32_MAX,
        /*group_segment_size=*/UINT32_MAX, &queue);
    if (iree_status_is_ok(status) && IREE_UNLIKELY(!queue)) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "HSA reported successful queue creation without returning a queue");
    }
    if (iree_status_is_ok(status) && !is_cooperative) {
      status = iree_hsa_amd_queue_set_priority(IREE_LIBHSA(params->libhsa),
                                               queue, params->priority);
    }
  }
  if (iree_status_is_ok(status) && IREE_UNLIKELY(!queue)) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "HSA reported successful queue creation without returning a queue");
  }
  // Older queue creation has no mask parameter, and the descriptor mask is not
  // honored by all ROCr implementations exposing hsa_amd_queue_create. Apply
  // the mask through the dedicated queue operation while the queue remains
  // private. Any native rejection prevents publication through the HAL.
  // ROCr exposes one shared cooperative queue per agent, so never mutate its
  // mask; cooperative callers are restricted to the complete resource set.
  if (iree_status_is_ok(status) && !is_cooperative &&
      params->compute_unit_mask_bit_count) {
    status = iree_hsa_amd_queue_cu_set_mask(IREE_LIBHSA(params->libhsa), queue,
                                            params->compute_unit_mask_bit_count,
                                            params->compute_unit_mask);
  }

  if (iree_status_is_ok(status)) {
    *out_queue = queue;
  } else if (queue) {
    iree_hal_amdgpu_hsa_queue_destroy(params->libhsa, queue);
  }
  return status;
}

void iree_hal_amdgpu_hsa_queue_destroy(const iree_hal_amdgpu_libhsa_t* libhsa,
                                       hsa_queue_t* queue) {
  iree_hal_amdgpu_hsa_cleanup_assert_success(
      iree_hsa_queue_destroy_raw(libhsa, queue));
}
