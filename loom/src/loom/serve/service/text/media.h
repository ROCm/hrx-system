// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_MEDIA_H_
#define LOOM_SERVE_SERVICE_TEXT_MEDIA_H_

#include "iree/base/api.h"
#include "iree/vm/buffer.h"
#include "loom/service/media/image.h"

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
// message content. Media records name a nonempty range in the message-local
// frame table. Codec and encoded-orientation facts remain native concerns.
typedef enum loom_serve_text_chat_content_field_e {
  LOOM_SERVE_TEXT_CHAT_CONTENT_FIELD_KIND = 0,
  LOOM_SERVE_TEXT_CHAT_CONTENT_FIELD_TEXT_OFFSET = 1,
  LOOM_SERVE_TEXT_CHAT_CONTENT_FIELD_TEXT_LENGTH = 2,
  LOOM_SERVE_TEXT_CHAT_CONTENT_FIELD_IMAGE_BEGIN = 3,
  LOOM_SERVE_TEXT_CHAT_CONTENT_FIELD_IMAGE_COUNT = 4,
  LOOM_SERVE_TEXT_CHAT_CONTENT_FIELD_IMAGE_DETAIL = 5,
  LOOM_SERVE_TEXT_CHAT_CONTENT_FIELD_COUNT = 6,
} loom_serve_text_chat_content_field_t;

typedef enum loom_serve_text_chat_content_kind_e {
  LOOM_SERVE_TEXT_CHAT_CONTENT_TEXT = 0,
  LOOM_SERVE_TEXT_CHAT_CONTENT_IMAGE = 1,
  LOOM_SERVE_TEXT_CHAT_CONTENT_VIDEO = 2,
} loom_serve_text_chat_content_kind_t;

// Source-visible message-local frame facts. Width and height are little-endian
// u32 display dimensions. Timestamp is little-endian f64 seconds, finite and
// nonnegative for video frames and zero for ordinary images.
enum {
  LOOM_SERVE_TEXT_CHAT_FRAME_WIDTH_OFFSET = 0,
  LOOM_SERVE_TEXT_CHAT_FRAME_HEIGHT_OFFSET = 4,
  LOOM_SERVE_TEXT_CHAT_FRAME_TIMESTAMP_OFFSET = 8,
  LOOM_SERVE_TEXT_CHAT_FRAME_BYTES = 16,
};

// Source chat-message plan. The explicit returned count names records before
// source-private metadata. Input ranges are message-local, ordered, nonempty,
// and cover all inspected frames without crossing content parts. Marker bytes
// are relative to the returned fragment; metadata offsets are plan-relative.
typedef enum loom_serve_text_media_projection_field_e {
  LOOM_SERVE_TEXT_MEDIA_PROJECTION_FIELD_IMAGE_BEGIN = 0,
  LOOM_SERVE_TEXT_MEDIA_PROJECTION_FIELD_IMAGE_COUNT = 1,
  LOOM_SERVE_TEXT_MEDIA_PROJECTION_FIELD_BYTE_OFFSET = 2,
  LOOM_SERVE_TEXT_MEDIA_PROJECTION_FIELD_BYTE_LENGTH = 3,
  LOOM_SERVE_TEXT_MEDIA_PROJECTION_FIELD_FEATURE_COUNT = 4,
  LOOM_SERVE_TEXT_MEDIA_PROJECTION_FIELD_METADATA_OFFSET = 5,
  LOOM_SERVE_TEXT_MEDIA_PROJECTION_FIELD_METADATA_LENGTH = 6,
  LOOM_SERVE_TEXT_MEDIA_PROJECTION_FIELD_COUNT = 7,
} loom_serve_text_media_projection_field_t;

// Native-normalized prepare-input marker plan. These records name checked
// request projection ordinals, not input frames. Metadata follows the records;
// marker offsets are relative to the input text suffix.
typedef enum loom_serve_text_media_marker_field_e {
  LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_PROJECTION_INDEX = 0,
  LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_BYTE_OFFSET = 1,
  LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_BYTE_LENGTH = 2,
  LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_FEATURE_COUNT = 3,
  LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_METADATA_OFFSET = 4,
  LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_METADATA_LENGTH = 5,
  LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_COUNT = 6,
} loom_serve_text_media_marker_field_t;

// Source prepare-input result fields. Records are ordered like the input marker
// plan and map each projection to its exact logical model-input rows.
typedef enum loom_serve_text_media_span_field_e {
  LOOM_SERVE_TEXT_MEDIA_SPAN_FIELD_PROJECTION_INDEX = 0,
  LOOM_SERVE_TEXT_MEDIA_SPAN_FIELD_INPUT_OFFSET = 1,
  LOOM_SERVE_TEXT_MEDIA_SPAN_FIELD_INPUT_COUNT = 2,
  LOOM_SERVE_TEXT_MEDIA_SPAN_FIELD_COUNT = 3,
} loom_serve_text_media_span_field_t;

// One inspected image occurrence borrowing its encoded source from the HTTP
// body. Pixel storage is deliberately absent until admission establishes that
// the image-conditioned prefix is not already resident.
typedef struct loom_serve_text_chat_image_t {
  // Encoded image bytes borrowed from the request until bounded realization.
  loom_service_media_image_source_t source;
  // Source-visible processing hint participating in semantic identity.
  loom_serve_text_chat_image_detail_t detail;
  // Image or video content kind; all frames in one part have the same kind.
  loom_serve_text_chat_content_kind_t kind;
  // Supplied presentation time in seconds, or zero for an ordinary image.
  double timestamp;
  // Message ordinal containing this occurrence.
  uint32_t message_ordinal;
  // Ordered content-part ordinal within the message.
  uint32_t content_ordinal;
  // Decoded text bytes preceding this occurrence within the message.
  uint32_t text_offset;
  // Exact compressed identity and codec/display geometry.
  loom_service_media_image_info_t info;
} loom_serve_text_chat_image_t;

// One checked source projection, independent of inspected input storage and
// of the finite result that will eventually supply its device features.
typedef struct loom_serve_text_media_projection_t {
  // First inspected input in the owning request's images array.
  uint32_t image_begin;
  // Nonempty contiguous input range, wholly within one content part.
  uint32_t image_count;
  // Global byte offset of the source-rendered image marker in the prompt.
  uint32_t marker_offset;
  // Nonzero byte length of the source-rendered image marker.
  uint32_t marker_length;
  // Number of projected rows replacing this marker in model input.
  uint32_t feature_count;
  // Byte offset of source-private preprocessing metadata in
  // projections.metadata.
  uint32_t metadata_offset;
  // Byte length of source-private preprocessing metadata.
  uint32_t metadata_length;
  // Immutable identity of every input, timestamp, placement, and source fact.
  loom_service_media_digest_t identity_digest;
} loom_serve_text_media_projection_t;

// One checked projection range in source-tokenized model input. Ranges are
// sorted, nonoverlapping, and refer to the owning request's source projections.
typedef struct loom_serve_text_media_span_t {
  // Source projection supplying these features.
  uint32_t projection_index;
  // First logical model-input row replaced by projected features.
  uint32_t input_offset;
  // Number of contiguous projected feature rows.
  uint32_t input_count;
} loom_serve_text_media_span_t;

// Immutable ordered image identity independent of request/body storage. The
// compact reference-counted storage contains the checked projection digests.
// An empty identity owns no storage.
typedef struct loom_serve_text_chat_media_identity_t {
  // Owned immutable packed identity storage.
  iree_vm_buffer_t* storage;
  // Number of ordered projections in storage.
  iree_host_size_t count;
} loom_serve_text_chat_media_identity_t;

typedef struct loom_serve_text_chat_t loom_serve_text_chat_t;

// Validates one source-returned message plan and commits its exact marker and
// metadata semantics as projections of the inspected image occurrences. No
// prompt bytes are scanned. The caller appends the matching fragment only on
// success.
iree_status_t loom_serve_text_media_apply_message_plan(
    loom_serve_text_chat_t* chat, iree_host_size_t first_image,
    iree_host_size_t prompt_offset, iree_host_size_t fragment_length,
    iree_host_size_t projection_count, iree_const_byte_span_t plan);

// Builds the source prepare-input marker plan for [prompt_offset, prompt_end).
// prompt_offset is within the rendered prompt. Prior image occurrences are
// omitted; a boundary through a marker is invalid.
// The returned VM buffer is owned and always non-null on success, including
// when marker_count is zero.
iree_status_t loom_serve_text_media_build_input_plan(
    const loom_serve_text_chat_t* chat, iree_host_size_t prompt_offset,
    iree_allocator_t host_allocator, iree_vm_buffer_t** out_plan,
    iree_host_size_t* out_marker_count);

// Checks the source prepare-input span result against the checked projections
// for [first_projection, projections.count), not the plan exposed to source.
// input_shift accounts for a native retained prediction prepended to the
// source token stream. Successful nonempty results own *out_spans.
iree_status_t loom_serve_text_media_parse_spans(
    const loom_serve_text_chat_t* chat, iree_host_size_t first_projection,
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
