// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_MEDIA_DIGEST_H_
#define LOOM_SERVE_MEDIA_DIGEST_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// SHA-256 content identity used for attacker-controlled media cache keys.
typedef struct loom_serve_media_digest_t {
  // Big-endian SHA-256 digest bytes.
  uint8_t bytes[32];
} loom_serve_media_digest_t;

// Computes the digest of |data|. Serving input bounds keep data below the
// SHA-256 2^61-byte bit-length limit.
void loom_serve_media_digest_compute(iree_const_byte_span_t data,
                                     loom_serve_media_digest_t* out_digest);

// Returns true when both complete digests are equal.
bool loom_serve_media_digest_equal(const loom_serve_media_digest_t* lhs,
                                   const loom_serve_media_digest_t* rhs);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_MEDIA_DIGEST_H_
