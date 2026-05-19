// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/sge.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>

static iree_status_t iree_net_rdma_sge_validate_span(iree_async_span_t span) {
  if (!span.region) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "RDMA span region must not be NULL");
  }
  if (span.region->type != IREE_ASYNC_REGION_TYPE_RDMA) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "span region type %u is not RDMA",
                            (unsigned)span.region->type);
  }
  if (!span.region->base_ptr) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "CPU-inaccessible RDMA regions need an explicit device IOVA");
  }
  bool in_range = span.offset <= span.region->length &&
                  span.length <= span.region->length - span.offset;
  if (!in_range) {
    iree_host_size_t span_end = span.length > SIZE_MAX - span.offset
                                    ? SIZE_MAX
                                    : span.offset + span.length;
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "RDMA span [%" PRIhsz ", %" PRIhsz
                            ") exceeds region length %" PRIhsz,
                            span.offset, span_end, span.region->length);
  }
  if (span.length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "RDMA SGE length too large: %" PRIhsz
                            " bytes (max %" PRIu32 ")",
                            span.length, UINT32_MAX);
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_net_rdma_sge_from_span(iree_async_span_t span, struct ibv_sge* out_sge) {
  IREE_ASSERT_ARGUMENT(out_sge);
  memset(out_sge, 0, sizeof(*out_sge));

  iree_status_t status = iree_net_rdma_sge_validate_span(span);
  if (iree_status_is_ok(status)) {
    out_sge->addr = (uint64_t)(uintptr_t)iree_async_span_ptr(span);
    out_sge->length = (uint32_t)span.length;
    out_sge->lkey = span.region->handles.rdma.lkey;
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_net_rdma_sge_list_from_span_list(
    iree_async_span_list_t spans, iree_host_size_t sge_capacity,
    struct ibv_sge* out_sges, int* out_sge_count) {
  IREE_ASSERT_ARGUMENT(out_sge_count);
  *out_sge_count = 0;

  iree_status_t status = iree_ok_status();
  if (spans.count > 0 && !spans.values) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "RDMA span list values must not be NULL");
  }
  if (iree_status_is_ok(status) && spans.count > sge_capacity) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "RDMA span count %" PRIhsz
                              " exceeds SGE capacity %" PRIhsz,
                              spans.count, sge_capacity);
  }
  if (iree_status_is_ok(status) && spans.count > INT_MAX) {
    status =
        iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                         "RDMA span count too large: %" PRIhsz, spans.count);
  }
  if (iree_status_is_ok(status) && spans.count > 0 && !out_sges) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "out_sges must not be NULL");
  }

  iree_host_size_t initialized_count = 0;
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < spans.count;
       ++i) {
    status = iree_net_rdma_sge_from_span(spans.values[i], &out_sges[i]);
    if (iree_status_is_ok(status)) {
      initialized_count = i + 1;
    }
  }

  if (iree_status_is_ok(status)) {
    *out_sge_count = (int)initialized_count;
  } else if (out_sges && initialized_count > 0) {
    memset(out_sges, 0, initialized_count * sizeof(*out_sges));
  }
  return status;
}
