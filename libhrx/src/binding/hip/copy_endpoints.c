// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/copy_endpoints.h"

#include "common/context.h"

iree_hip_copy_endpoint_t iree_hip_copy_endpoint_resolve(
    iree_hal_streaming_context_t* context, const void* pointer,
    bool search_foreign) {
  iree_hip_copy_endpoint_t endpoint = {0};
  if (iree_hal_streaming_memory_try_lookup(
          context, (iree_hal_streaming_deviceptr_t)pointer,
          &endpoint.reference)) {
    endpoint.context = context;
  } else if (search_foreign) {
    iree_hal_streaming_memory_try_lookup_across_contexts(
        context, (iree_hal_streaming_deviceptr_t)pointer, &endpoint.context,
        &endpoint.reference);
  }
  return endpoint;
}

void iree_hip_copy_endpoint_deinitialize(iree_hal_streaming_context_t* context,
                                         iree_hip_copy_endpoint_t* endpoint) {
  if (endpoint->context && endpoint->context != context) {
    iree_hal_streaming_context_release(endpoint->context);
  }
}

static bool iree_hip_copy_endpoint_is_device(
    const iree_hip_copy_endpoint_t* endpoint) {
  return endpoint->reference.buffer &&
         endpoint->reference.buffer->memory_kind !=
             IREE_HAL_STREAMING_MEMORY_KIND_HOST;
}

bool iree_hip_copy_endpoint_contains(const iree_hip_copy_endpoint_t* endpoint,
                                     iree_host_size_t span,
                                     bool device_operand) {
  if (!endpoint->reference.buffer) {
    return !device_operand;
  }
  const iree_device_size_t limit =
      device_operand ? endpoint->reference.buffer->logical_size
                     : endpoint->reference.buffer->size;
  return endpoint->reference.offset <= limit &&
         span <= limit - endpoint->reference.offset;
}

hipError_t iree_hip_copy_endpoints_initialize(
    iree_hal_streaming_context_t* context, hipMemcpyKind kind,
    const void* target, iree_host_size_t target_span, const void* source,
    iree_host_size_t source_span, iree_hip_copy_endpoints_t* out_endpoints) {
  *out_endpoints = (iree_hip_copy_endpoints_t){.kind = kind};
  if (kind == hipMemcpyDeviceToDeviceNoCU) {
    kind = hipMemcpyDeviceToDevice;
  }
  if (kind == hipMemcpyHostToHost) {
    return hipSuccess;
  }
  if (kind != hipMemcpyHostToDevice && kind != hipMemcpyDeviceToHost &&
      kind != hipMemcpyDeviceToDevice && kind != hipMemcpyDefault) {
    return hipErrorInvalidMemcpyDirection;
  }

  iree_hip_copy_endpoints_t endpoints = {
      .source = iree_hip_copy_endpoint_resolve(context, source,
                                               kind != hipMemcpyHostToDevice),
      .target = iree_hip_copy_endpoint_resolve(context, target,
                                               kind != hipMemcpyDeviceToHost),
  };
  if (kind == hipMemcpyDefault) {
    const bool device_source =
        iree_hip_copy_endpoint_is_device(&endpoints.source);
    const bool device_target =
        iree_hip_copy_endpoint_is_device(&endpoints.target);
    kind =
        device_source
            ? (device_target ? hipMemcpyDeviceToDevice : hipMemcpyDeviceToHost)
            : (device_target ? hipMemcpyHostToDevice : hipMemcpyHostToHost);
  }
  endpoints.kind = kind;

  // Host registrations belong to their context. A foreign host allocation is
  // still an ordinary host pointer to this graph, not a local device mapping.
  if ((kind == hipMemcpyHostToDevice || kind == hipMemcpyHostToHost) &&
      endpoints.source.context != context) {
    iree_hip_copy_endpoint_deinitialize(context, &endpoints.source);
    endpoints.source = (iree_hip_copy_endpoint_t){0};
  }
  if ((kind == hipMemcpyDeviceToHost || kind == hipMemcpyHostToHost) &&
      endpoints.target.context != context) {
    iree_hip_copy_endpoint_deinitialize(context, &endpoints.target);
    endpoints.target = (iree_hip_copy_endpoint_t){0};
  }
  const bool valid =
      kind == hipMemcpyHostToHost ||
      (iree_hip_copy_endpoint_contains(&endpoints.source, source_span,
                                       kind != hipMemcpyHostToDevice) &&
       iree_hip_copy_endpoint_contains(&endpoints.target, target_span,
                                       kind != hipMemcpyDeviceToHost));
  if (!valid) {
    iree_hip_copy_endpoints_deinitialize(context, &endpoints);
    return hipErrorInvalidValue;
  }
  *out_endpoints = endpoints;
  return hipSuccess;
}

void iree_hip_copy_endpoints_deinitialize(
    iree_hal_streaming_context_t* context,
    iree_hip_copy_endpoints_t* endpoints) {
  iree_hip_copy_endpoint_deinitialize(context, &endpoints->source);
  iree_hip_copy_endpoint_deinitialize(context, &endpoints->target);
}
