// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_PREPARED_REQUEST_H_
#define LOOM_SERVICE_PREPARED_REQUEST_H_

#include "iree/vm/buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Source-prepared input independent of a captured execution or checkpoint bank.
// The request owner retains the source input through cohort completion when
// these bytes borrow it. Equal tags permit batching within the source's limit.
typedef struct loom_service_prepared_request_t {
  // Owned opaque source bytes, released by the request owner.
  iree_vm_buffer_t* bytes;
  // Positive additive source cost, at most the declared batch limit.
  uint64_t units;
  // Source-defined compatibility class; no native geometry interpretation.
  uint64_t tag;
} loom_service_prepared_request_t;

// Packs a little-endian [byte_offset, byte_length] i64 directory followed by
// prepared bytes in stable request order. Offsets are relative to packed;
// its first count * 16 bytes equal ranges. Source may forward packed directly
// as a device upload, including the directory, or produce another layout.
// Each request's bytes are copied once. Success transfers both buffers;
// failure leaves both NULL.
iree_status_t loom_service_prepared_requests_pack(
    iree_host_size_t count, const loom_service_prepared_request_t* requests,
    iree_allocator_t host_allocator, iree_vm_buffer_t** out_ranges,
    iree_vm_buffer_t** out_packed);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVICE_PREPARED_REQUEST_H_
