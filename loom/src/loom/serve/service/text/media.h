// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_MEDIA_H_
#define LOOM_SERVE_SERVICE_TEXT_MEDIA_H_

#include "iree/base/api.h"
#include "iree/vm/buffer.h"
#include "loom/serve/media/image.h"

#ifdef __cplusplus
extern "C" {
#endif

// Model interpretation of the OpenAI image_url detail hint.
typedef enum loom_serve_text_chat_image_detail_e {
  LOOM_SERVE_TEXT_CHAT_IMAGE_DETAIL_AUTO = 0,
  LOOM_SERVE_TEXT_CHAT_IMAGE_DETAIL_LOW = 1,
  LOOM_SERVE_TEXT_CHAT_IMAGE_DETAIL_HIGH = 2,
} loom_serve_text_chat_image_detail_t;

// Source-visible ordered content record fields. Each record is serialized as
// this many little-endian u32 words. Text records name a span in the decoded
// message content. Image records name a request image and its display geometry;
// codec and encoded-orientation facts remain native preprocessing concerns.
typedef enum loom_serve_text_chat_content_field_e {
  LOOM_SERVE_TEXT_CHAT_CONTENT_FIELD_KIND = 0,
  LOOM_SERVE_TEXT_CHAT_CONTENT_FIELD_TEXT_OFFSET = 1,
  LOOM_SERVE_TEXT_CHAT_CONTENT_FIELD_TEXT_LENGTH = 2,
  LOOM_SERVE_TEXT_CHAT_CONTENT_FIELD_IMAGE_INDEX = 3,
  LOOM_SERVE_TEXT_CHAT_CONTENT_FIELD_DISPLAY_WIDTH = 4,
  LOOM_SERVE_TEXT_CHAT_CONTENT_FIELD_DISPLAY_HEIGHT = 5,
  LOOM_SERVE_TEXT_CHAT_CONTENT_FIELD_IMAGE_DETAIL = 6,
  LOOM_SERVE_TEXT_CHAT_CONTENT_FIELD_COUNT = 7,
} loom_serve_text_chat_content_field_t;

typedef enum loom_serve_text_chat_content_kind_e {
  LOOM_SERVE_TEXT_CHAT_CONTENT_TEXT = 0,
  LOOM_SERVE_TEXT_CHAT_CONTENT_IMAGE = 1,
} loom_serve_text_chat_content_kind_t;

// Source chat-message image plan fields. One record for each image occurrence
// precedes source-private metadata in the returned buffer. Marker byte ranges
// are relative to the returned message fragment. Metadata offsets are relative
// to the plan buffer and must follow the complete record array.
typedef enum loom_serve_text_media_marker_field_e {
  LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_IMAGE_INDEX = 0,
  LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_BYTE_OFFSET = 1,
  LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_BYTE_LENGTH = 2,
  LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_FEATURE_COUNT = 3,
  LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_METADATA_OFFSET = 4,
  LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_METADATA_LENGTH = 5,
  LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_COUNT = 6,
} loom_serve_text_media_marker_field_t;

// Source prepare-input result fields. Records are ordered like the input marker
// plan and map each image occurrence to its exact logical model-input rows.
typedef enum loom_serve_text_media_span_field_e {
  LOOM_SERVE_TEXT_MEDIA_SPAN_FIELD_IMAGE_INDEX = 0,
  LOOM_SERVE_TEXT_MEDIA_SPAN_FIELD_INPUT_OFFSET = 1,
  LOOM_SERVE_TEXT_MEDIA_SPAN_FIELD_INPUT_COUNT = 2,
  LOOM_SERVE_TEXT_MEDIA_SPAN_FIELD_COUNT = 3,
} loom_serve_text_media_span_field_t;

// One inspected image occurrence borrowing its escaped data URL from the HTTP
// body. Pixel storage is deliberately absent until admission establishes that
// the image-conditioned prefix is not already resident.
typedef struct loom_serve_text_chat_image_t {
  // Raw escaped JSON string contents, excluding surrounding quotes.
  iree_string_view_t data_url;
  // Source-visible processing hint participating in semantic identity.
  loom_serve_text_chat_image_detail_t detail;
  // Message ordinal containing this occurrence.
  uint32_t message_ordinal;
  // Ordered content-part ordinal within the message.
  uint32_t content_ordinal;
  // Decoded text bytes preceding this occurrence within the message.
  uint32_t text_offset;
  // Global byte offset of the source-rendered image marker in the prompt.
  uint32_t marker_offset;
  // Nonzero byte length of the source-rendered image marker.
  uint32_t marker_length;
  // Number of projected rows replacing this marker in model input.
  uint32_t feature_count;
  // Byte offset of source-private preprocessing metadata in images.metadata.
  uint32_t metadata_offset;
  // Byte length of source-private preprocessing metadata.
  uint32_t metadata_length;
  // Digest of the exact source-private preprocessing metadata.
  loom_serve_media_digest_t metadata_digest;
  // Exact compressed identity and codec/display geometry.
  loom_serve_media_image_info_t info;
} loom_serve_text_chat_image_t;

// One checked image range in source-tokenized model input. Ranges are sorted,
// nonoverlapping and refer to image occurrences in the owning chat request.
typedef struct loom_serve_text_media_span_t {
  // Request image occurrence supplying projected features.
  uint32_t image_index;
  // First logical model-input row replaced by projected features.
  uint32_t input_offset;
  // Number of contiguous projected feature rows.
  uint32_t input_count;
} loom_serve_text_media_span_t;

// Source-visible fields describing one projected row in a media-bearing
// epoch. Records are serialized as little-endian u32 words and ordered by the
// first epoch followed by its optional continuation. The source maps the plan
// record and relative input offset into its model-specific packet record.
typedef enum loom_serve_text_media_epoch_field_e {
  LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_PLAN_INDEX = 0,
  LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_SPAN_INPUT_OFFSET = 1,
  LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_FEATURE_OFFSET = 2,
  LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_FEATURE_LENGTH = 3,
  LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_METADATA_OFFSET = 4,
  LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_METADATA_LENGTH = 5,
  LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_COUNT = 6,
} loom_serve_text_media_epoch_field_t;

// Source-written device packet geometry for a media-bearing epoch. Every
// logical packet slot has this fixed record extent; projected row payloads
// begin after the complete first/continuation packet capacities.
enum {
  LOOM_SERVE_TEXT_MEDIA_PACKET_RECORD_BYTES = 32,
  LOOM_SERVE_TEXT_MEDIA_PACKET_FEATURE_ALIGNMENT = 16,
};

// One projected image range in a prepared request. The logical input range is
// absolute within the request's complete token array. row_offset selects the
// first corresponding entry from the owning media input's row table.
typedef struct loom_serve_text_media_input_span_t {
  // First logical request input replaced by a projected row.
  uint32_t input_offset;
  // Number of contiguous projected rows in this image occurrence.
  uint32_t input_count;
  // First projected row view in the owning media input.
  uint32_t row_offset;
  // Source-private metadata byte offset in the owning media input.
  uint32_t metadata_offset;
  // Source-private metadata byte length for this image occurrence.
  uint32_t metadata_length;
} loom_serve_text_media_input_span_t;

// Immutable projected media borrowed by scheduled epochs. Spans are sorted,
// nonoverlapping request ranges. Each row view and the metadata bytes remain
// stable until every epoch that borrowed them completes or terminal model
// retirement proves its transfers no longer retain host payloads.
typedef struct loom_serve_text_media_input_t {
  // Borrowed sorted image ranges in absolute request input coordinates.
  const loom_serve_text_media_input_span_t* spans;
  // Number of populated image ranges.
  iree_host_size_t span_count;
  // Borrowed projected row byte views in span order.
  const iree_const_byte_span_t* rows;
  // Number of populated projected row views.
  iree_host_size_t row_count;
  // Borrowed concatenated source-private image metadata.
  iree_const_byte_span_t metadata;
} loom_serve_text_media_input_t;

// Immutable ordered image identity independent of request/body storage. The
// compact reference-counted storage contains content and source-plan digests,
// exact placement, detail, marker and feature-count semantics. An empty
// identity owns no storage.
typedef struct loom_serve_text_chat_media_identity_t {
  // Owned immutable packed identity storage.
  iree_vm_buffer_t* storage;
  // Number of ordered image occurrences in storage.
  iree_host_size_t count;
} loom_serve_text_chat_media_identity_t;

typedef struct loom_serve_text_chat_t loom_serve_text_chat_t;

// Validates one source-returned message plan and commits its exact marker and
// metadata semantics to the already inspected image occurrences. No prompt
// bytes are scanned. The caller appends the matching fragment only on success.
iree_status_t loom_serve_text_media_apply_message_plan(
    loom_serve_text_chat_t* chat, iree_host_size_t first_image,
    iree_host_size_t prompt_offset, iree_host_size_t fragment_length,
    iree_const_byte_span_t plan);

// Builds the source prepare-input marker plan for [prompt_offset, prompt_end).
// Prior image occurrences are omitted; a boundary through a marker is invalid.
// The returned VM buffer is owned and always non-null on success, including
// when marker_count is zero.
iree_status_t loom_serve_text_media_build_input_plan(
    const loom_serve_text_chat_t* chat, iree_host_size_t prompt_offset,
    iree_allocator_t host_allocator, iree_vm_buffer_t** out_plan,
    iree_host_size_t* out_marker_count);

// Checks the source prepare-input span result against the marker plan and
// token count. input_shift accounts for a native retained prediction prepended
// to the source token stream. Successful nonempty results own *out_spans.
iree_status_t loom_serve_text_media_parse_spans(
    iree_const_byte_span_t marker_plan, iree_host_size_t marker_count,
    iree_const_byte_span_t source_spans, iree_host_size_t token_count,
    iree_host_size_t input_shift, iree_allocator_t host_allocator,
    loom_serve_text_media_span_t** out_spans, iree_host_size_t* out_span_count);

// Captures the request's ordered image semantics without retaining encoded
// inputs. Failure zeros |out_identity|.
iree_status_t loom_serve_text_chat_media_identity_initialize(
    const loom_serve_text_chat_t* chat, iree_allocator_t host_allocator,
    loom_serve_text_chat_media_identity_t* out_identity);

// Copies an identity reference. Source and target may be released
// independently.
void loom_serve_text_chat_media_identity_retain(
    const loom_serve_text_chat_media_identity_t* source,
    loom_serve_text_chat_media_identity_t* out_identity);

// Releases one identity reference; zero-initialized values are accepted.
void loom_serve_text_chat_media_identity_deinitialize(
    loom_serve_text_chat_media_identity_t* identity);

// Whether the complete request image sequence has this exact identity.
bool loom_serve_text_chat_media_identity_equal_chat(
    const loom_serve_text_chat_media_identity_t* identity,
    const loom_serve_text_chat_t* chat);

// Whether the identity is an exact ordered prefix of the request images.
bool loom_serve_text_chat_media_identity_is_prefix(
    const loom_serve_text_chat_media_identity_t* identity,
    const loom_serve_text_chat_t* chat);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_MEDIA_H_
