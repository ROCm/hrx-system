// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_MEDIA_INPUT_H_
#define LOOM_SERVE_SERVICE_TEXT_MEDIA_INPUT_H_

#include "loom/service/finite/result.h"

// One projected image in checked, source-tokenized request input. The result
// was prepared with input_count expected rows for request_index. Several images
// and several text requests may share the same immutable finite cohort.
typedef struct loom_serve_text_media_input_span_t {
  // First logical input in the request's prepared token array.
  uint32_t input_offset;
  // Number of projected rows replacing logical input IDs.
  uint32_t input_count;
  // Original request ordinal in result, unaffected by peer cancellation.
  uint32_t request_index;
  // Borrowed cohort owning device rows and their producer readiness.
  const loom_service_finite_result_t* result;
  // Borrowed source-private image metadata, already checked at admission.
  iree_const_byte_span_t metadata;
} loom_serve_text_media_input_span_t;

// Immutable request media borrowed only through epoch submission. Spans are
// sorted, nonoverlapping, and in prepared-input coordinates. Submission copies
// metadata and queue-captures device views before returning, even on failure;
// it never borrows a host feature payload. The request owner retains results
// only while later chunks still need their rows.
typedef struct loom_serve_text_media_input_t {
  // Borrowed ordered image spans.
  const loom_serve_text_media_input_span_t* spans;
  // Number of populated image spans.
  iree_host_size_t span_count;
} loom_serve_text_media_input_t;

#endif  // LOOM_SERVE_SERVICE_TEXT_MEDIA_INPUT_H_
