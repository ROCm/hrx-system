// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_MEDIA_IMAGE_H_
#define LOOM_SERVE_MEDIA_IMAGE_H_

#include "iree/base/api.h"
#include "loom/serve/media/image_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Borrowed transport input. A video sequence supplies individual base64 slices
// with a known format; ordinary image inputs supply complete data URLs. Neither
// form requires copying or synthesizing a URL before bounded inspection.
typedef struct loom_serve_media_image_source_t {
  // Raw JSON string contents, excluding quotes and retaining JSON escapes.
  iree_string_view_t value;
  // Declared format for bare base64, or zero to read it from a data URL.
  loom_serve_media_image_format_t format;
} loom_serve_media_image_source_t;

static inline loom_serve_media_image_source_t
loom_serve_media_image_source_from_data_url(iree_string_view_t value) {
  loom_serve_media_image_source_t source = {value,
                                            (loom_serve_media_image_format_t)0};
  return source;
}

// Immutable inspected compressed input. The owner can collect a cohort before
// decoding and release each compressed payload immediately after realization.
// No cache is created here; callers account these bytes in their request
// budget.
typedef struct loom_serve_media_image_input_t {
  // Validated geometry, pixel layout, and identity of encoded bytes.
  loom_serve_media_image_info_t info;
  // Immutable compressed payload, borrowed from allocation.
  iree_const_byte_span_t encoded;
  // Allocator owning allocation.
  iree_allocator_t host_allocator;
  // Owned transport-decoding storage.
  uint8_t* allocation;
  // Retained allocation size, including transport decoding slack, for budgets.
  iree_host_size_t allocation_size;
} loom_serve_media_image_input_t;

// Opens one transport input without realizing pixels. Failure zeros out_input.
iree_status_t loom_serve_media_image_input_initialize(
    loom_serve_media_image_source_t source,
    const loom_serve_media_image_limits_t* limits,
    iree_allocator_t host_allocator, loom_serve_media_image_input_t* out_input);

// Releases compressed storage; zero-initialized inputs are accepted.
void loom_serve_media_image_input_deinitialize(
    loom_serve_media_image_input_t* input);

// Decodes an immutable input cohort, grouping work by its selected provider.
// Each exact-size destination and terminal status belongs to the corresponding
// input ordinal. Per-image failures do not cancel other cohort members. The
// return status reports only shared scratch-allocation failure; in that case
// every out_statuses entry is OK and no destination was written. Callers
// consume every per-image status. Scratch is O(count), independent of pixel
// dimensions.
iree_status_t loom_serve_media_image_decode_batch(
    iree_host_size_t count, const loom_serve_media_image_input_t* const* inputs,
    const iree_byte_span_t* pixels, iree_allocator_t host_allocator,
    iree_status_t* out_statuses);

// Inspects one borrowed transport input. The temporary compressed
// representation is released before return.
// Animated inputs and MIME/signature mismatches are rejected. No pixel plane is
// allocated. Failure zeros |out_info|.
iree_status_t loom_serve_media_image_inspect(
    loom_serve_media_image_source_t source,
    const loom_serve_media_image_limits_t* limits,
    iree_allocator_t host_allocator, loom_serve_media_image_info_t* out_info);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_MEDIA_IMAGE_H_
