// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_MEDIA_DIGEST_H_
#define LOOM_SERVICE_MEDIA_DIGEST_H_

#include "loom/util/digest.h"

// Media cache identities use the shared SHA-256 representation.
typedef loom_sha256_t loom_service_media_digest_t;

static inline void loom_service_media_digest_compute(
    iree_const_byte_span_t data, loom_service_media_digest_t* out_digest) {
  loom_sha256_compute(data, out_digest);
}

static inline bool loom_service_media_digest_equal(
    const loom_service_media_digest_t* lhs,
    const loom_service_media_digest_t* rhs) {
  return loom_sha256_equal(lhs, rhs);
}

#endif  // LOOM_SERVICE_MEDIA_DIGEST_H_
