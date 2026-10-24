// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/recognition/transcription.h"

#include <math.h>
#include <stdlib.h>

#include "iree/base/internal/unicode.h"
#include "loom/util/json.h"

iree_status_t loom_serve_transcription_parse(
    loom_serve_http_part_list_t parts, iree_string_view_t model,
    loom_serve_recognition_format_t format,
    loom_serve_transcription_t* out_request) {
  *out_request = (loom_serve_transcription_t){0};
  static const char* const names[] = {
      "model",           "file",  "language", "prompt", "temperature",
      "response_format", "stream"};
  iree_string_view_t values[IREE_ARRAYSIZE(names)] = {0};
  iree_string_view_t content_type = iree_string_view_empty();
  uint32_t present = 0;
  for (iree_host_size_t i = 0; i < parts.count; ++i) {
    const loom_serve_http_part_t* part = &parts.values[i];
    iree_host_size_t field = 0;
    for (; field < IREE_ARRAYSIZE(names); ++field) {
      if (iree_string_view_equal(part->name,
                                 iree_make_cstring_view(names[field]))) {
        break;
      }
    }
    if (field == IREE_ARRAYSIZE(names)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported transcription field '%.*s'",
                              (int)part->name.size, part->name.data);
    }
    if ((present & (1u << field)) || (field != 1 && part->filename.data)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "duplicate or file-valued transcription field '%s'", names[field]);
    }
    if (field != 1 && !iree_unicode_utf8_validate(part->contents)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "transcription field '%s' must be UTF-8",
                              names[field]);
    }
    values[field] = part->contents;
    if (field == 1) {
      content_type = part->content_type;
    }
    present |= 1u << field;
  }
  if (!iree_string_view_equal(values[0], model) || !values[1].size) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "transcription requires the selected model and a nonempty file");
  }
  float temperature = 0;
  if ((present & (1u << 4)) &&
      (!iree_string_view_atof(values[4], &temperature) ||
       !isfinite(temperature) || temperature < 0 || temperature > 1)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "temperature must be a finite number from zero to one");
  }
  const bool text_response = iree_string_view_equal(values[5], IREE_SV("text"));
  if ((present & (1u << 5)) && !text_response &&
      !iree_string_view_equal(values[5], IREE_SV("json"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "select response_format=json or text");
  }
  if ((present & (1u << 6)) &&
      !iree_string_view_equal(values[6], IREE_SV("false"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "this transcription profile requires stream=false");
  }
  const iree_const_byte_span_t audio =
      iree_make_const_byte_span(values[1].data, values[1].size);
  loom_serve_audio_source_t source;
  IREE_RETURN_IF_ERROR(
      loom_serve_audio_source_parse(content_type, audio, &source));
  uint64_t normalized_frame_count = 0;
  if (source.encoding != LOOM_SERVE_AUDIO_ENCODING_CONTAINER) {
    if (source.channels != format.channels) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "audio/pcm requires channels=%u",
                              format.channels);
    }
    IREE_RETURN_IF_ERROR(loom_serve_audio_frame_count(
        source.frame_count, source.sample_rate, format.sample_rate,
        &normalized_frame_count));
  }
  *out_request = (loom_serve_transcription_t){
      .request = {.language = values[2],
                  .prompt = values[3],
                  .temperature = temperature,
                  .outputs = LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT},
      .audio = source,
      .normalized_frame_count = normalized_frame_count,
      .text_response = text_response};
  return iree_ok_status();
}

static int loom_serve_transcription_compare(const void* lhs, const void* rhs) {
  const loom_serve_recognition_transcript_t* a =
      *(const loom_serve_recognition_transcript_t* const*)lhs;
  const loom_serve_recognition_transcript_t* b =
      *(const loom_serve_recognition_transcript_t* const*)rhs;
  if (a->begin != b->begin) {
    return a->begin < b->begin ? -1 : 1;
  }
  return (a->segment > b->segment) - (a->segment < b->segment);
}

iree_status_t loom_serve_transcription_format(
    const loom_serve_recognition_document_t* document, bool text_response,
    iree_string_builder_t* output) {
  const loom_serve_recognition_transcript_t** ordered = NULL;
  if (document->count) {
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc_array(output->allocator, document->count,
                                    sizeof(*ordered), (void**)&ordered));
  }
  for (iree_host_size_t i = 0; i < document->count; ++i) {
    ordered[i] = loom_serve_recognition_document_at(document, i);
  }
  if (document->count > 1) {
    qsort(ordered, document->count, sizeof(*ordered),
          loom_serve_transcription_compare);
  }
  iree_string_builder_t text;
  iree_string_builder_initialize(output->allocator, &text);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < document->count && iree_status_is_ok(status);
       ++i) {
    if (!ordered[i]->text.size) {
      continue;
    }
    if (iree_string_builder_size(&text)) {
      status = iree_string_builder_append_cstring(&text, "\n");
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_string(&text, ordered[i]->text);
    }
  }
  if (iree_status_is_ok(status) && text_response) {
    status = iree_string_builder_append_string(output,
                                               iree_string_builder_view(&text));
  } else if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(output, "{\"text\":");
    if (iree_status_is_ok(status)) {
      loom_output_stream_t stream;
      loom_output_stream_for_builder(output, &stream);
      status = loom_json_write_escaped_string(&stream,
                                              iree_string_builder_view(&text));
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_cstring(output, "}");
    }
  }
  iree_string_builder_deinitialize(&text);
  iree_allocator_free(output->allocator, ordered);
  return status;
}
