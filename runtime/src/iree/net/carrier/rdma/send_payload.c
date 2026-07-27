// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/send_payload.h"

#include <string.h>

static bool iree_net_rdma_send_payload_span_is_registered(
    iree_async_span_t span) {
  return span.region && span.region->type == IREE_ASYNC_REGION_TYPE_RDMA;
}

static iree_status_t iree_net_rdma_send_payload_validate_staging_source(
    iree_async_span_t span, iree_host_size_t span_index) {
  if (!iree_async_span_is_cpu_accessible(span)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "RDMA send staging requires CPU-accessible span %" PRIhsz, span_index);
  }
  if (!iree_async_span_ptr(span)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "send span %" PRIhsz " has a NULL source pointer",
                            span_index);
  }
  if (span.region) {
    bool in_range = span.offset <= span.region->length &&
                    span.length <= span.region->length - span.offset;
    if (!in_range) {
      iree_host_size_t span_end = span.length > SIZE_MAX - span.offset
                                      ? SIZE_MAX
                                      : span.offset + span.length;
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "send span %" PRIhsz " [%" PRIhsz ", %" PRIhsz
                              ") exceeds region length %" PRIhsz,
                              span_index, span.offset, span_end,
                              span.region->length);
    }
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_net_rdma_send_payload_prepare(
    iree_async_span_list_t source_spans, iree_net_send_flags_t send_flags,
    iree_async_buffer_pool_t* staging_pool,
    iree_net_rdma_send_payload_t* out_payload) {
  if (!out_payload) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_payload must not be NULL");
  }
  memset(out_payload, 0, sizeof(*out_payload));
  if (source_spans.count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source_spans must not be empty");
  }
  if (!source_spans.values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source span values must not be NULL");
  }
  if (source_spans.count > IREE_NET_RDMA_CONNECTION_DATA_MAX_SEND_SGE) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "source span count %" PRIhsz " exceeds RDMA send limit %u",
        source_spans.count, IREE_NET_RDMA_CONNECTION_DATA_MAX_SEND_SGE);
  }

  iree_host_size_t byte_length = 0;
  iree_host_size_t staging_byte_length = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < source_spans.count && iree_status_is_ok(status); ++i) {
    iree_async_span_t span = source_spans.values[i];
    iree_host_size_t new_byte_length = 0;
    if (!iree_host_size_checked_add(byte_length, span.length,
                                    &new_byte_length)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "send span length total overflows host size");
      continue;
    }
    byte_length = new_byte_length;
    if (span.length == 0 ||
        iree_net_rdma_send_payload_span_is_registered(span)) {
      continue;
    }
    status = iree_net_rdma_send_payload_validate_staging_source(span, i);
    if (iree_status_is_ok(status) &&
        !iree_host_size_checked_add(staging_byte_length, span.length,
                                    &staging_byte_length)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "unregistered send span length total overflows host size");
    }
  }
  if (iree_status_is_ok(status) && byte_length == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "send payload must not be empty");
  }
  if (iree_status_is_ok(status) && staging_byte_length > 0 &&
      iree_any_bit_set(send_flags, IREE_NET_SEND_FLAG_ZERO_COPY)) {
    status =
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "RDMA ZERO_COPY send requires RDMA-registered spans");
  }
  if (iree_status_is_ok(status) && staging_byte_length > 0 && !staging_pool) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "RDMA send staging pool is not available");
  }
  if (iree_status_is_ok(status) && staging_byte_length > 0) {
    iree_host_size_t buffer_size =
        iree_async_buffer_pool_buffer_size(staging_pool);
    if (staging_byte_length > buffer_size) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "unregistered send data size %" PRIhsz
                                " exceeds RDMA staging buffer size %" PRIhsz,
                                staging_byte_length, buffer_size);
    }
  }

  iree_async_buffer_lease_t staging_buffer_lease;
  memset(&staging_buffer_lease, 0, sizeof(staging_buffer_lease));
  if (iree_status_is_ok(status) && staging_byte_length > 0) {
    status =
        iree_async_buffer_pool_acquire(staging_pool, &staging_buffer_lease);
  }

  if (iree_status_is_ok(status)) {
    iree_host_size_t output_count = 0;
    iree_host_size_t staging_offset = 0;
    bool previous_span_is_staged = false;
    for (iree_host_size_t i = 0; i < source_spans.count; ++i) {
      iree_async_span_t source_span = source_spans.values[i];
      if (source_span.length == 0) continue;

      if (iree_net_rdma_send_payload_span_is_registered(source_span)) {
        out_payload->spans[output_count++] = source_span;
        previous_span_is_staged = false;
        continue;
      }

      memcpy(iree_async_span_ptr(staging_buffer_lease.span) + staging_offset,
             iree_async_span_ptr(source_span), source_span.length);
      iree_host_size_t span_offset =
          staging_buffer_lease.span.offset + staging_offset;
      if (previous_span_is_staged) {
        iree_async_span_t* previous_span =
            &out_payload->spans[output_count - 1];
        previous_span->length += source_span.length;
        staging_offset += source_span.length;
        continue;
      }
      out_payload->spans[output_count++] = iree_async_span_make(
          staging_buffer_lease.span.region, span_offset, source_span.length);
      previous_span_is_staged = true;
      staging_offset += source_span.length;
    }

    out_payload->span_count = output_count;
    out_payload->byte_length = byte_length;
    out_payload->staging_buffer_lease = staging_buffer_lease;
    memset(&staging_buffer_lease, 0, sizeof(staging_buffer_lease));
  }

  iree_async_buffer_lease_release(&staging_buffer_lease);
  return status;
}

IREE_API_EXPORT void iree_net_rdma_send_payload_deinitialize(
    iree_net_rdma_send_payload_t* payload) {
  if (!payload) return;
  iree_async_buffer_lease_release(&payload->staging_buffer_lease);
  memset(payload, 0, sizeof(*payload));
}
