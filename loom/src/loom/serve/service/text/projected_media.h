// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_PROJECTED_MEDIA_H_
#define LOOM_SERVE_SERVICE_TEXT_PROJECTED_MEDIA_H_

#include "loom/serve/service/finite/model.h"
#include "loom/serve/service/text/input.h"

#ifdef __cplusplus
extern "C" {
#endif

// Request-owned projected image rows. Each image occurrence owns one detached
// finite result; row views and source metadata are exposed through |input| for
// mixed text epochs. Metadata borrows the request chat until deinitialization.
typedef struct loom_serve_text_projected_media_t {
  // Allocator owning every exact table below.
  iree_allocator_t host_allocator;
  // Owned detached result per projected image occurrence.
  loom_serve_finite_result_t* results;
  // Owned request image index per detached result.
  uint32_t* image_indices;
  // Owned complete span table; input.spans may advance within this table.
  loom_serve_text_media_input_span_t* spans;
  // Owned complete row-view table borrowed by input.
  iree_const_byte_span_t* rows;
  // Number of image results and complete spans.
  iree_host_size_t result_count;
  // Number of leading results already released after epoch retirement.
  iree_host_size_t released_count;
  // Borrowed immutable mixed-epoch view over the surviving spans.
  loom_serve_text_media_input_t input;
} loom_serve_text_projected_media_t;

// Allocates exact ownership tables for every complete media span at or after
// |input_origin|. A prefix boundary inside an image span violates the prefix
// producer contract. Image indices and source metadata offsets are copied from
// already checked source input; metadata bytes continue to borrow |chat|.
iree_status_t loom_serve_text_projected_media_initialize(
    const loom_serve_text_chat_t* chat, const loom_serve_text_input_t* input,
    iree_host_size_t input_origin, iree_allocator_t host_allocator,
    loom_serve_text_projected_media_t* out_media);

// Transfers one successfully detached finite result into its image slot and
// publishes its checked row views. The result must contain exactly one request
// with the source-declared number of rows.
void loom_serve_text_projected_media_set_result(
    loom_serve_text_projected_media_t* media, iree_host_size_t image,
    loom_serve_finite_result_t* result);

// Releases complete leading image results whose logical input ranges end at or
// before |input_frontier|. Epochs borrowing those rows have already retired.
void loom_serve_text_projected_media_release_through(
    loom_serve_text_projected_media_t* media, iree_host_size_t input_frontier);

// Returns the immutable mixed-epoch view, or NULL when no image rows survive.
const loom_serve_text_media_input_t* loom_serve_text_projected_media_input(
    const loom_serve_text_projected_media_t* media);

// Releases every remaining detached result and exact ownership table. The
// borrowed request chat may be released afterward. Zero state is accepted.
void loom_serve_text_projected_media_deinitialize(
    loom_serve_text_projected_media_t* media);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_PROJECTED_MEDIA_H_
