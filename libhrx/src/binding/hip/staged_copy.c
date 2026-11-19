// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/staged_copy.h"

#include "common/direct_transfer.h"
#include "common/stream.h"

typedef struct iree_hip_staged_copy_t {
  // Destination descriptor borrowed for this blocking operation.
  const iree_hip_staged_copy_endpoint_t* destination;
  // Source descriptor borrowed for this blocking operation.
  const iree_hip_staged_copy_endpoint_t* source;
  // Bytes copied from each source row.
  iree_device_size_t width;
  // Number of rows in each depth slice.
  iree_host_size_t height;
  // Number of depth slices.
  iree_host_size_t depth;
  // Scratch row used to bridge device-to-device copies.
  uint8_t* staging;
} iree_hip_staged_copy_t;

static bool iree_hip_staged_copy_endpoint_is_device(
    const iree_hip_staged_copy_endpoint_t* endpoint) {
  return endpoint->context != NULL && endpoint->buffer != NULL &&
         endpoint->host_pointer == NULL;
}

static bool iree_hip_staged_copy_endpoint_is_host(
    const iree_hip_staged_copy_endpoint_t* endpoint) {
  return endpoint->context == NULL && endpoint->buffer == NULL &&
         endpoint->offset == 0 && endpoint->host_pointer != NULL;
}

static bool iree_hip_staged_copy_calculate_span(
    const iree_hip_staged_copy_endpoint_t* endpoint, iree_device_size_t width,
    iree_host_size_t height, iree_host_size_t depth,
    iree_device_size_t* out_span) {
  iree_device_size_t slice_offset = 0;
  iree_device_size_t row_offset = 0;
  return iree_device_size_checked_mul((iree_device_size_t)(depth - 1),
                                      endpoint->slice_pitch, &slice_offset) &&
         iree_device_size_checked_mul((iree_device_size_t)(height - 1),
                                      endpoint->row_pitch, &row_offset) &&
         iree_device_size_checked_add(slice_offset, row_offset,
                                      &slice_offset) &&
         iree_device_size_checked_add(slice_offset, width, out_span);
}

static iree_status_t iree_hip_staged_copy_execute(void* user_data) {
  iree_hip_staged_copy_t* copy = (iree_hip_staged_copy_t*)user_data;
  const bool source_is_device = copy->source->buffer != NULL;
  const bool destination_is_device = copy->destination->buffer != NULL;
  if (!copy->staging) {
    const iree_hal_transfer_region_t region = {
        .row_length = copy->width,
        .row_count = copy->height,
        .slice_count = copy->depth,
        .source_strides = {copy->source->row_pitch, copy->source->slice_pitch},
        .target_strides = {copy->destination->row_pitch,
                           copy->destination->slice_pitch},
    };
    iree_hal_transfer_operation_t operation = {0};
    iree_hal_streaming_context_t* context = NULL;
    if (source_is_device && destination_is_device) {
      context = copy->source->context;
      operation.type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY;
      operation.copy.source_buffer = copy->source->buffer;
      operation.copy.source_offset = copy->source->offset;
      operation.copy.target_buffer = copy->destination->buffer;
      operation.copy.target_offset = copy->destination->offset;
      operation.copy.region = &region;
    } else if (source_is_device) {
      context = copy->source->context;
      operation.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
      operation.download.source_buffer = copy->source->buffer;
      operation.download.source_offset = copy->source->offset;
      operation.download.target = copy->destination->host_pointer;
      operation.download.region = &region;
    } else {
      context = copy->destination->context;
      operation.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
      operation.upload.source = copy->source->host_pointer;
      operation.upload.target_buffer = copy->destination->buffer;
      operation.upload.target_offset = copy->destination->offset;
      operation.upload.region = &region;
    }
    return iree_hal_streaming_direct_transfer_context_and_wait(context, 1,
                                                               &operation);
  }

  // Independently owned device contexts cannot share a transfer queue. Their
  // compatibility bridge reuses one host row through both completed transfers.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t z = 0; z < copy->depth && iree_status_is_ok(status);
       ++z) {
    for (iree_host_size_t y = 0; y < copy->height && iree_status_is_ok(status);
         ++y) {
      const iree_device_size_t source_offset = copy->source->offset +
                                               z * copy->source->slice_pitch +
                                               y * copy->source->row_pitch;
      const iree_device_size_t destination_offset =
          copy->destination->offset + z * copy->destination->slice_pitch +
          y * copy->destination->row_pitch;
      status = iree_hal_streaming_direct_transfer_d2h(
          copy->source->context, copy->source->buffer, source_offset,
          copy->staging, copy->width);
      if (iree_status_is_ok(status)) {
        status = iree_hal_streaming_direct_transfer_h2d(
            copy->destination->context, copy->staging,
            copy->destination->buffer, destination_offset, copy->width);
      }
    }
  }
  return status;
}

iree_status_t iree_hip_staged_copy_3d(
    iree_hal_streaming_stream_t* stream,
    const iree_hip_staged_copy_endpoint_t* destination,
    const iree_hip_staged_copy_endpoint_t* source, iree_device_size_t width,
    iree_host_size_t height, iree_host_size_t depth) {
  IREE_ASSERT_ARGUMENT(destination);
  IREE_ASSERT_ARGUMENT(source);
  if (IREE_UNLIKELY(width == 0 || height == 0 || depth == 0)) {
    return iree_ok_status();
  }

  const bool source_is_device = iree_hip_staged_copy_endpoint_is_device(source);
  const bool source_is_host = iree_hip_staged_copy_endpoint_is_host(source);
  const bool destination_is_device =
      iree_hip_staged_copy_endpoint_is_device(destination);
  const bool destination_is_host =
      iree_hip_staged_copy_endpoint_is_host(destination);
  if (IREE_UNLIKELY((!source_is_device && !source_is_host) ||
                    (!destination_is_device && !destination_is_host) ||
                    (source_is_host && destination_is_host) ||
                    (height > 1 && (width > source->row_pitch ||
                                    width > destination->row_pitch)) ||
                    width > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid staged copy endpoints or shape");
  }

  iree_device_size_t source_span = 0;
  iree_device_size_t destination_span = 0;
  if (IREE_UNLIKELY(!iree_hip_staged_copy_calculate_span(source, width, height,
                                                         depth, &source_span) ||
                    !iree_hip_staged_copy_calculate_span(destination, width,
                                                         height, depth,
                                                         &destination_span))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "staged copy range overflow");
  }
  if (source_is_device) {
    IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_range(
        source->buffer, source->offset, source_span));
  }
  if (destination_is_device) {
    IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_range(
        destination->buffer, destination->offset, destination_span));
  }
  if (IREE_UNLIKELY(
          (source_is_host && source_span > IREE_HOST_SIZE_MAX) ||
          (destination_is_host && destination_span > IREE_HOST_SIZE_MAX))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "host staged copy range overflow");
  }

  iree_hip_staged_copy_t copy = {
      .destination = destination,
      .source = source,
      .width = width,
      .height = height,
      .depth = depth,
      .staging = NULL,
  };
  iree_status_t status = iree_ok_status();
  if (source_is_device && destination_is_device &&
      source->context != destination->context) {
    status =
        iree_allocator_malloc(iree_allocator_system(), (iree_host_size_t)width,
                              (void**)&copy.staging);
  }
  if (iree_status_is_ok(status)) {
    status = stream ? iree_hal_streaming_execute_host_operation(
                          stream, iree_hip_staged_copy_execute, &copy)
                    : iree_hip_staged_copy_execute(&copy);
  }
  iree_allocator_free(iree_allocator_system(), copy.staging);
  return status;
}
