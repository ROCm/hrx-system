// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/projected_media.h"

#include <string.h>

iree_status_t loom_serve_text_projected_media_initialize(
    const loom_serve_text_chat_t* chat, const loom_serve_text_input_t* input,
    iree_host_size_t input_origin, iree_allocator_t host_allocator,
    loom_serve_text_projected_media_t* out_media) {
  memset(out_media, 0, sizeof(*out_media));
  out_media->host_allocator = host_allocator;
  iree_host_size_t first = 0;
  while (first < input->media_span_count) {
    const loom_serve_text_media_span_t* span = &input->media_spans[first];
    if ((iree_host_size_t)span->input_offset + span->input_count >
        input_origin) {
      break;
    }
    ++first;
  }
  IREE_ASSERT(first == input->media_span_count ||
              input->media_spans[first].input_offset >= input_origin);
  const iree_host_size_t result_count = input->media_span_count - first;
  iree_host_size_t row_count = 0;
  for (iree_host_size_t i = first; i < input->media_span_count; ++i) {
    if (!iree_host_size_checked_add(
            row_count, input->media_spans[i].input_count, &row_count)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "projected row table exceeds host capacity");
    }
  }
  if (!result_count) {
    return iree_ok_status();
  }
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, result_count, sizeof(*out_media->results),
      (void**)&out_media->results);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, result_count,
                                         sizeof(*out_media->image_indices),
                                         (void**)&out_media->image_indices);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, result_count,
                                         sizeof(*out_media->spans),
                                         (void**)&out_media->spans);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, row_count,
                                         sizeof(*out_media->rows),
                                         (void**)&out_media->rows);
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_text_projected_media_deinitialize(out_media);
    return status;
  }
  out_media->result_count = result_count;
  iree_host_size_t row_offset = 0;
  for (iree_host_size_t i = 0; i < result_count; ++i) {
    const loom_serve_text_media_span_t* source = &input->media_spans[first + i];
    const loom_serve_text_chat_image_t* image =
        &chat->images.values[source->image_index];
    IREE_ASSERT(row_offset <= UINT32_MAX);
    out_media->image_indices[i] = source->image_index;
    out_media->spans[i] = (loom_serve_text_media_input_span_t){
        .input_offset = source->input_offset,
        .input_count = source->input_count,
        .row_offset = (uint32_t)row_offset,
        .metadata_offset = image->metadata_offset,
        .metadata_length = image->metadata_length,
    };
    row_offset += source->input_count;
  }
  out_media->input = (loom_serve_text_media_input_t){
      .spans = out_media->spans,
      .span_count = result_count,
      .rows = out_media->rows,
      .row_count = row_count,
      .metadata = iree_make_const_byte_span(
          (const uint8_t*)iree_string_builder_buffer(&chat->images.metadata),
          iree_string_builder_size(&chat->images.metadata)),
  };
  return iree_ok_status();
}

void loom_serve_text_projected_media_set_result(
    loom_serve_text_projected_media_t* media, iree_host_size_t image,
    loom_serve_finite_result_t* result) {
  media->results[image] = *result;
  memset(result, 0, sizeof(*result));
  const loom_serve_text_media_input_span_t* span = &media->spans[image];
  for (iree_host_size_t row = 0; row < span->input_count; ++row) {
    media->rows[span->row_offset + row] =
        loom_serve_finite_result_row(&media->results[image], 0, row);
  }
}

void loom_serve_text_projected_media_release_through(
    loom_serve_text_projected_media_t* media, iree_host_size_t input_frontier) {
  while (media->released_count < media->result_count) {
    const loom_serve_text_media_input_span_t* span =
        &media->spans[media->released_count];
    if ((iree_host_size_t)span->input_offset + span->input_count >
        input_frontier) {
      break;
    }
    loom_serve_finite_result_deinitialize(
        &media->results[media->released_count]);
    ++media->released_count;
  }
  media->input.spans = media->spans + media->released_count;
  media->input.span_count = media->result_count - media->released_count;
}

const loom_serve_text_media_input_t* loom_serve_text_projected_media_input(
    const loom_serve_text_projected_media_t* media) {
  return media->input.span_count ? &media->input : NULL;
}

void loom_serve_text_projected_media_deinitialize(
    loom_serve_text_projected_media_t* media) {
  for (iree_host_size_t i = 0; i < media->result_count; ++i) {
    loom_serve_finite_result_deinitialize(&media->results[i]);
  }
  iree_allocator_free(media->host_allocator, media->rows);
  iree_allocator_free(media->host_allocator, media->spans);
  iree_allocator_free(media->host_allocator, media->image_indices);
  iree_allocator_free(media->host_allocator, media->results);
  memset(media, 0, sizeof(*media));
}
