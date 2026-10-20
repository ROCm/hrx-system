// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/event.h"

#include "libamdf/src/allocator.h"
#include "libamdf/src/device.h"
#include "libamdf/src/gpu/device.h"
#include "libamdf/src/gpu/umd/event.h"
#include "libamdf/src/structure.h"

struct amdf_gpu_event_t {
  // Allocator copied for this resource's metadata.
  amdf_allocator_t host_allocator;
  // Native parent guarded until event destruction succeeds.
  amdf_device_t* device;
  // Native resource borrowing the creating device through destruction.
  amdf_gpu_umd_event_t* umd;
  // Immutable producer facts established before publication.
  amdf_gpu_event_info_t info;
};

amdf_status_t AMDF_CALL amdf_gpu_event_create(
    amdf_device_t* device, const amdf_gpu_event_create_info_t* create_info,
    amdf_gpu_event_t** out_event) {
  if (out_event == NULL || device == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t validation_status = amdf_structure_validate_input(
      create_info, AMDF_STRUCTURE_TYPE_GPU_EVENT_CREATE_INFO,
      (uint32_t)sizeof(amdf_gpu_event_create_info_t));
  if (!amdf_status_is_ok(validation_status)) {
    return validation_status;
  }
  if (create_info->reserved != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (!amdf_device_is_engine(device, AMDF_ENGINE_KIND_GPU)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const amdf_gpu_device_info_t* device_info = amdf_gpu_device_get_info(device);
  if ((device_info->features & AMDF_GPU_DEVICE_FEATURE_NATIVE_EVENTS) == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }

  const amdf_allocator_t host_allocator = device->host_allocator;
  amdf_gpu_event_t* event = NULL;
  amdf_status_t status =
      amdf_calloc(host_allocator, sizeof(*event),
                  amdf_alignof(amdf_gpu_event_t), (void**)&event);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  event->host_allocator = host_allocator;
  status = amdf_device_register_child(device);
  if (!amdf_status_is_ok(status)) {
    amdf_free(host_allocator, event);
    return status;
  }
  event->device = device;
  amdf_gpu_umd_event_result_t result = {0};
  status = amdf_gpu_umd_event_create(amdf_gpu_device_get_umd(device),
                                     &event->umd, &result);
  if (amdf_status_is_ok(status)) {
    event->info = (amdf_gpu_event_info_t){
        .type = AMDF_STRUCTURE_TYPE_GPU_EVENT_INFO,
        .structure_size = sizeof(event->info),
        .device_id = device_info->id,
        .reset_epoch = device_info->reset_epoch,
        .format = result.format,
        .format_version = result.format_version,
        .publication_modes = result.publication_modes,
        .mailbox_address = result.mailbox_address,
        .trigger_data = result.trigger_data,
        .interrupt_context = result.interrupt_context,
    };
    *out_event = event;
  } else {
    amdf_device_unregister_child(device);
    amdf_free(host_allocator, event);
  }
  return status;
}

amdf_status_t AMDF_CALL amdf_gpu_event_query_info(
    amdf_gpu_event_t* event, amdf_gpu_event_info_t* out_info) {
  if (event == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_GPU_EVENT_INFO,
      (uint32_t)sizeof(amdf_gpu_event_info_t));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const uint32_t structure_size = out_info->structure_size;
  void* const next = out_info->next;
  *out_info = event->info;
  out_info->structure_size = structure_size;
  out_info->next = next;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_gpu_event_signal(amdf_gpu_event_t* event) {
  if (event == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  return amdf_gpu_umd_event_signal(event->umd);
}

amdf_device_t* amdf_gpu_event_get_device(const amdf_gpu_event_t* event) {
  return event->device;
}

amdf_status_t AMDF_CALL amdf_gpu_event_wait(amdf_gpu_event_t* event,
                                            uint64_t timeout_nanoseconds) {
  if (event == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  return amdf_gpu_umd_event_wait(event->umd, timeout_nanoseconds);
}

amdf_status_t AMDF_CALL amdf_gpu_event_destroy(amdf_gpu_event_t* event) {
  if (event == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = amdf_gpu_umd_event_destroy(event->umd);
  if (amdf_status_is_ok(status)) {
    amdf_device_unregister_child(event->device);
    amdf_free(event->host_allocator, event);
  }
  return status;
}
