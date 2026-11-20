// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/media.h"

#include <string.h>

#include "loom/serve/service/text/chat.h"
#include "loom/util/digest.h"

enum {
  MEDIA_PROJECTION_RECORD_BYTES =
      LOOM_SERVE_TEXT_MEDIA_PROJECTION_FIELD_COUNT * sizeof(uint32_t),
  MEDIA_MARKER_RECORD_BYTES =
      LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_COUNT * sizeof(uint32_t),
  MEDIA_SPAN_RECORD_BYTES =
      LOOM_SERVE_TEXT_MEDIA_SPAN_FIELD_COUNT * sizeof(uint32_t),
};

static uint32_t loom_serve_text_media_load_field(const uint8_t* record,
                                                 iree_host_size_t field) {
  return iree_unaligned_load_le_u32(record + field * sizeof(uint32_t));
}

static void loom_serve_text_media_store_field(uint8_t* record,
                                              iree_host_size_t field,
                                              uint32_t value) {
  iree_unaligned_store_le_u32(record + field * sizeof(uint32_t), value);
}

// Establish retention identity once, while all source semantics and inspected
// inputs are available. Later queries compare fixed digests without rescanning
// frames, borrowing request storage, or rehashing preprocessing metadata.
static void loom_serve_text_media_projection_identify(
    const loom_serve_text_chat_t* chat, iree_const_byte_span_t metadata,
    loom_serve_text_media_projection_t* projection) {
  const loom_serve_text_chat_image_t* first =
      &chat->images.values[projection->image_begin];
  const uint32_t fields[] = {
      projection->image_count,   (uint32_t)first->kind,
      (uint32_t)first->detail,   first->message_ordinal,
      first->content_ordinal,    first->text_offset,
      projection->marker_offset, projection->marker_length,
      projection->feature_count, projection->metadata_length,
  };
  uint8_t header[sizeof(fields)];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(fields); ++i) {
    iree_unaligned_store_le_u32(header + i * sizeof(uint32_t), fields[i]);
  }
  loom_sha256_state_t hash;
  loom_sha256_initialize(&hash);
  loom_sha256_update(&hash, iree_make_const_byte_span(header, sizeof(header)));
  for (iree_host_size_t i = 0; i < projection->image_count; ++i) {
    const loom_serve_text_chat_image_t* image =
        &chat->images.values[projection->image_begin + i];
    uint8_t timestamp[sizeof(double)];
    iree_unaligned_store_le_f64(timestamp, image->timestamp);
    loom_sha256_update(&hash,
                       iree_make_const_byte_span(timestamp, sizeof(timestamp)));
    loom_sha256_update(
        &hash, iree_make_const_byte_span(image->info.digest.bytes,
                                         sizeof(image->info.digest.bytes)));
  }
  loom_sha256_update(&hash, metadata);
  loom_sha256_finalize(&hash, &projection->identity_digest);
}

iree_status_t loom_serve_text_media_apply_message_plan(
    loom_serve_text_chat_t* chat, iree_host_size_t first_image,
    iree_host_size_t prompt_offset, iree_host_size_t fragment_length,
    iree_host_size_t projection_count, iree_const_byte_span_t plan) {
  const iree_host_size_t image_count = chat->images.count - first_image;
  if (projection_count > image_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source projection count exceeds inspected inputs");
  }
  const iree_host_size_t record_bytes =
      projection_count * MEDIA_PROJECTION_RECORD_BYTES;
  if (plan.data_length < record_bytes) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source image plan is shorter than its records");
  }
  if (!projection_count) {
    if (image_count || plan.data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source must project every inspected input");
    }
    return iree_ok_status();
  }
  uint64_t metadata_length = 0;
  iree_host_size_t previous_end = 0;
  iree_host_size_t input_cursor = 0;
  for (iree_host_size_t i = 0; i < projection_count; ++i) {
    const uint8_t* record = plan.data + i * MEDIA_PROJECTION_RECORD_BYTES;
    const uint32_t image_begin = loom_serve_text_media_load_field(
        record, LOOM_SERVE_TEXT_MEDIA_PROJECTION_FIELD_IMAGE_BEGIN);
    const uint32_t input_count = loom_serve_text_media_load_field(
        record, LOOM_SERVE_TEXT_MEDIA_PROJECTION_FIELD_IMAGE_COUNT);
    if (image_begin != input_cursor || !input_count ||
        input_count > image_count - input_cursor) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "source projection %zu has an invalid input range", i);
    }
    const loom_serve_text_chat_image_t* first =
        &chat->images.values[first_image + input_cursor];
    const loom_serve_text_chat_image_t* last =
        &chat->images.values[first_image + input_cursor + input_count - 1];
    // Inspection emits inputs in message/part order. Equal endpoints establish
    // that this entire contiguous range belongs to one content part.
    if (first->content_ordinal != last->content_ordinal ||
        (first->kind == LOOM_SERVE_TEXT_CHAT_CONTENT_IMAGE &&
         input_count != 1)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source projection %zu crosses content parts", i);
    }
    const uint32_t marker_offset = loom_serve_text_media_load_field(
        record, LOOM_SERVE_TEXT_MEDIA_PROJECTION_FIELD_BYTE_OFFSET);
    const uint32_t marker_length = loom_serve_text_media_load_field(
        record, LOOM_SERVE_TEXT_MEDIA_PROJECTION_FIELD_BYTE_LENGTH);
    const uint32_t feature_count = loom_serve_text_media_load_field(
        record, LOOM_SERVE_TEXT_MEDIA_PROJECTION_FIELD_FEATURE_COUNT);
    const uint32_t metadata_offset = loom_serve_text_media_load_field(
        record, LOOM_SERVE_TEXT_MEDIA_PROJECTION_FIELD_METADATA_OFFSET);
    const uint32_t length = loom_serve_text_media_load_field(
        record, LOOM_SERVE_TEXT_MEDIA_PROJECTION_FIELD_METADATA_LENGTH);
    const uint64_t marker_end = (uint64_t)marker_offset + marker_length;
    const uint64_t metadata_end = (uint64_t)metadata_offset + length;
    if (!marker_length || !feature_count || marker_offset < previous_end ||
        marker_end > fragment_length || metadata_offset < record_bytes ||
        metadata_end > plan.data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source returned invalid image plan record %zu",
                              i);
    }
    input_cursor += input_count;
    metadata_length += length;
    previous_end = (iree_host_size_t)marker_end;
  }
  if (input_cursor != image_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source must project every inspected input");
  }
  // The protocol limits image count to 4096. Bound the complete normalized
  // request plan here, including records, so all retained offsets, suffix
  // plans, and the metadata builder's terminator fit even on 32-bit hosts.
  const uint64_t metadata_end =
      iree_string_builder_size(&chat->projections.metadata) + metadata_length;
  const uint64_t plan_length =
      (chat->projections.count + projection_count) * MEDIA_MARKER_RECORD_BYTES +
      metadata_end;
  iree_host_size_t prompt_end = 0;
  if (!iree_host_size_checked_add(prompt_offset, fragment_length,
                                  &prompt_end) ||
      prompt_end > UINT32_MAX || plan_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "source image plan exceeds host representation");
  }
  const iree_host_size_t required_capacity =
      chat->projections.count + projection_count;
  if (required_capacity > chat->projections.capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        chat->host_allocator, required_capacity,
        sizeof(*chat->projections.values), &chat->projections.capacity,
        (void**)&chat->projections.values));
  }
  char* metadata_data = NULL;
  iree_host_size_t metadata_capacity = 0;
  IREE_RETURN_IF_ERROR(iree_string_builder_reserve_for_append(
      &chat->projections.metadata, (iree_host_size_t)metadata_length,
      &metadata_data, &metadata_capacity));
  const iree_host_size_t metadata_begin =
      iree_string_builder_size(&chat->projections.metadata);
  iree_host_size_t metadata_cursor = 0;
  for (iree_host_size_t i = 0; i < projection_count; ++i) {
    const uint8_t* record = plan.data + i * MEDIA_PROJECTION_RECORD_BYTES;
    loom_serve_text_media_projection_t* projection =
        &chat->projections.values[chat->projections.count + i];
    projection->image_begin =
        (uint32_t)first_image +
        loom_serve_text_media_load_field(
            record, LOOM_SERVE_TEXT_MEDIA_PROJECTION_FIELD_IMAGE_BEGIN);
    projection->image_count = loom_serve_text_media_load_field(
        record, LOOM_SERVE_TEXT_MEDIA_PROJECTION_FIELD_IMAGE_COUNT);
    const uint32_t marker_offset = loom_serve_text_media_load_field(
        record, LOOM_SERVE_TEXT_MEDIA_PROJECTION_FIELD_BYTE_OFFSET);
    const uint32_t metadata_offset = loom_serve_text_media_load_field(
        record, LOOM_SERVE_TEXT_MEDIA_PROJECTION_FIELD_METADATA_OFFSET);
    const uint32_t metadata_length = loom_serve_text_media_load_field(
        record, LOOM_SERVE_TEXT_MEDIA_PROJECTION_FIELD_METADATA_LENGTH);
    projection->marker_offset = (uint32_t)(prompt_offset + marker_offset);
    projection->marker_length = loom_serve_text_media_load_field(
        record, LOOM_SERVE_TEXT_MEDIA_PROJECTION_FIELD_BYTE_LENGTH);
    projection->feature_count = loom_serve_text_media_load_field(
        record, LOOM_SERVE_TEXT_MEDIA_PROJECTION_FIELD_FEATURE_COUNT);
    projection->metadata_offset = (uint32_t)(metadata_begin + metadata_cursor);
    projection->metadata_length = metadata_length;
    const iree_const_byte_span_t metadata =
        iree_make_const_byte_span(plan.data + metadata_offset, metadata_length);
    loom_serve_text_media_projection_identify(chat, metadata, projection);
    if (metadata_length) {
      memcpy(metadata_data + metadata_cursor, metadata.data, metadata_length);
    }
    metadata_cursor += metadata_length;
  }
  iree_string_builder_commit_append(&chat->projections.metadata,
                                    metadata_cursor);
  chat->projections.count = required_capacity;
  return iree_ok_status();
}

iree_status_t loom_serve_text_media_build_input_plan(
    const loom_serve_text_chat_t* chat, iree_host_size_t prompt_offset,
    iree_allocator_t host_allocator, iree_vm_buffer_t** out_plan,
    iree_host_size_t* out_marker_count) {
  *out_plan = NULL;
  *out_marker_count = 0;
  iree_host_size_t first = 0;
  while (first < chat->projections.count) {
    const loom_serve_text_media_projection_t* projection =
        &chat->projections.values[first];
    const iree_host_size_t marker_end =
        (iree_host_size_t)projection->marker_offset + projection->marker_length;
    if (marker_end > prompt_offset) {
      break;
    }
    ++first;
  }
  if (first < chat->projections.count &&
      chat->projections.values[first].marker_offset < prompt_offset) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "input boundary splits an image marker");
  }
  const iree_host_size_t count = chat->projections.count - first;
  const iree_host_size_t record_bytes = count * MEDIA_MARKER_RECORD_BYTES;
  iree_host_size_t metadata_bytes = 0;
  for (iree_host_size_t i = first; i < chat->projections.count; ++i) {
    metadata_bytes += chat->projections.values[i].metadata_length;
  }
  const iree_host_size_t plan_length = record_bytes + metadata_bytes;
  iree_vm_buffer_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_create(plan_length, 4, host_allocator, &plan));
  uint8_t* bytes = iree_vm_buffer_data(plan);
  iree_host_size_t metadata_offset = record_bytes;
  for (iree_host_size_t i = 0; i < count; ++i) {
    const iree_host_size_t projection_index = first + i;
    const loom_serve_text_media_projection_t* projection =
        &chat->projections.values[projection_index];
    uint8_t* record = bytes + i * MEDIA_MARKER_RECORD_BYTES;
    loom_serve_text_media_store_field(
        record, LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_PROJECTION_INDEX,
        (uint32_t)projection_index);
    loom_serve_text_media_store_field(
        record, LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_BYTE_OFFSET,
        projection->marker_offset - (uint32_t)prompt_offset);
    loom_serve_text_media_store_field(
        record, LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_BYTE_LENGTH,
        projection->marker_length);
    loom_serve_text_media_store_field(
        record, LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_FEATURE_COUNT,
        projection->feature_count);
    loom_serve_text_media_store_field(
        record, LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_METADATA_OFFSET,
        (uint32_t)metadata_offset);
    loom_serve_text_media_store_field(
        record, LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_METADATA_LENGTH,
        projection->metadata_length);
    if (projection->metadata_length) {
      memcpy(bytes + metadata_offset,
             iree_string_builder_buffer(&chat->projections.metadata) +
                 projection->metadata_offset,
             projection->metadata_length);
      metadata_offset += projection->metadata_length;
    }
  }
  *out_plan = plan;
  *out_marker_count = count;
  return iree_ok_status();
}

iree_status_t loom_serve_text_media_parse_spans(
    const loom_serve_text_chat_t* chat, iree_host_size_t first_projection,
    iree_const_byte_span_t source_spans, iree_host_size_t token_count,
    iree_host_size_t input_shift, iree_allocator_t host_allocator,
    loom_serve_text_media_span_t** out_spans,
    iree_host_size_t* out_span_count) {
  *out_spans = NULL;
  *out_span_count = 0;
  const iree_host_size_t marker_count =
      chat->projections.count - first_projection;
  const iree_host_size_t span_bytes = marker_count * MEDIA_SPAN_RECORD_BYTES;
  if (source_spans.data_length != span_bytes) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source input returned invalid image span storage");
  }
  if (!marker_count) {
    return iree_ok_status();
  }
  loom_serve_text_media_span_t* spans = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, marker_count, sizeof(*spans), (void**)&spans));
  iree_status_t status = iree_ok_status();
  iree_host_size_t previous_end = 0;
  for (iree_host_size_t i = 0; i < marker_count && iree_status_is_ok(status);
       ++i) {
    const uint8_t* record = source_spans.data + i * MEDIA_SPAN_RECORD_BYTES;
    const uint32_t projection_index = loom_serve_text_media_load_field(
        record, LOOM_SERVE_TEXT_MEDIA_SPAN_FIELD_PROJECTION_INDEX);
    const uint32_t input_offset = loom_serve_text_media_load_field(
        record, LOOM_SERVE_TEXT_MEDIA_SPAN_FIELD_INPUT_OFFSET);
    const uint32_t input_count = loom_serve_text_media_load_field(
        record, LOOM_SERVE_TEXT_MEDIA_SPAN_FIELD_INPUT_COUNT);
    const iree_host_size_t expected_projection = first_projection + i;
    const uint32_t expected_count =
        chat->projections.values[expected_projection].feature_count;
    const uint64_t input_end = (uint64_t)input_offset + input_count;
    if (projection_index != expected_projection ||
        input_count != expected_count || input_offset < previous_end ||
        input_end > token_count || input_end > UINT32_MAX ||
        input_shift > UINT32_MAX - input_end) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "source returned invalid image input span record %zu", i);
      break;
    }
    spans[i] = (loom_serve_text_media_span_t){
        .projection_index = projection_index,
        .input_offset = (uint32_t)(input_offset + input_shift),
        .input_count = input_count,
    };
    previous_end = (iree_host_size_t)input_end;
  }
  if (iree_status_is_ok(status)) {
    *out_spans = spans;
    *out_span_count = marker_count;
  } else {
    iree_allocator_free(host_allocator, spans);
  }
  return status;
}

iree_status_t loom_serve_text_chat_media_identity_initialize(
    const loom_serve_text_chat_t* chat, iree_allocator_t host_allocator,
    loom_serve_text_chat_media_identity_t* out_identity) {
  memset(out_identity, 0, sizeof(*out_identity));
  if (!chat->projections.count) {
    return iree_ok_status();
  }
  const iree_host_size_t length =
      chat->projections.count * sizeof(loom_service_media_digest_t);
  iree_vm_buffer_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_create(length, 1, host_allocator, &storage));
  uint8_t* bytes = iree_vm_buffer_data(storage);
  for (iree_host_size_t i = 0; i < chat->projections.count; ++i) {
    const loom_serve_text_media_projection_t* projection =
        &chat->projections.values[i];
    memcpy(bytes + i * sizeof(projection->identity_digest),
           projection->identity_digest.bytes,
           sizeof(projection->identity_digest));
  }
  *out_identity = (loom_serve_text_chat_media_identity_t){
      .storage = storage, .count = chat->projections.count};
  return iree_ok_status();
}

void loom_serve_text_chat_media_identity_retain(
    const loom_serve_text_chat_media_identity_t* source,
    loom_serve_text_chat_media_identity_t* out_identity) {
  *out_identity = *source;
  iree_vm_buffer_retain(out_identity->storage);
}

void loom_serve_text_chat_media_identity_deinitialize(
    loom_serve_text_chat_media_identity_t* identity) {
  iree_vm_buffer_release(identity->storage);
  memset(identity, 0, sizeof(*identity));
}

bool loom_serve_text_chat_media_identity_is_prefix(
    const loom_serve_text_chat_media_identity_t* identity,
    const loom_serve_text_chat_t* chat) {
  if (identity->count > chat->projections.count) {
    return false;
  }
  if (!identity->count) {
    return true;
  }
  const uint8_t* bytes = iree_vm_buffer_const_data(identity->storage);
  for (iree_host_size_t i = 0; i < identity->count; ++i) {
    const loom_serve_text_media_projection_t* projection =
        &chat->projections.values[i];
    if (memcmp(bytes + i * sizeof(projection->identity_digest),
               projection->identity_digest.bytes,
               sizeof(projection->identity_digest)) != 0) {
      return false;
    }
  }
  return true;
}

bool loom_serve_text_chat_media_identity_equal_chat(
    const loom_serve_text_chat_media_identity_t* identity,
    const loom_serve_text_chat_t* chat) {
  return identity->count == chat->projections.count &&
         loom_serve_text_chat_media_identity_is_prefix(identity, chat);
}
