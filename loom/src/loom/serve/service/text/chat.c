// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/chat.h"

#include <math.h>
#include <string.h>

#include "iree/base/internal/json.h"
#include "iree/base/internal/unicode.h"
#include "iree/vm/reflection.h"
#include "iree/vm/sync.h"
#include "loom/serve/service/text/chat_fields.h"
#include "loom/serve/service/text/tool_schemas.h"
#include "loom/util/json.h"

static iree_status_t loom_serve_text_chat_append_decoded(
    iree_string_builder_t* output, iree_string_view_t raw) {
  char* data = NULL;
  iree_host_size_t capacity = 0;
  IREE_RETURN_IF_ERROR(iree_string_builder_reserve_for_append(
      output, raw.size, &data, &capacity));
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_json_unescape_string(raw, capacity, data, &length));
  iree_string_builder_commit_append(output, length);
  return iree_ok_status();
}

static iree_status_t loom_serve_text_chat_quote(iree_string_builder_t* output,
                                                iree_string_view_t value) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  return loom_json_write_escaped_string(&stream, value);
}

static iree_status_t loom_serve_text_chat_text_part(void* user_data,
                                                    iree_host_size_t index,
                                                    iree_json_value_type_t type,
                                                    iree_string_view_t value) {
  if (type != IREE_JSON_VALUE_TYPE_OBJECT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "chat content parts must be text objects");
  }
  loom_serve_text_chat_field_t fields[] = {
      {"type", LOOM_SERVE_TEXT_CHAT_STRING},
      {"text", LOOM_SERVE_TEXT_CHAT_STRING}};
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_fields_read(
      value, IREE_ARRAYSIZE(fields), fields, 0));
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_required(&fields[1]));
  if (!iree_string_view_equal(fields[0].value, IREE_SV("text"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "only text content is supported");
  }
  return loom_serve_text_chat_append_decoded(user_data, fields[1].value);
}

static iree_status_t loom_serve_text_chat_text_content(
    const loom_serve_text_chat_field_t* field, iree_string_builder_t* output) {
  iree_string_builder_reset(output);
  if (!field->value.data || field->type == IREE_JSON_VALUE_TYPE_NULL) {
    return iree_ok_status();
  }
  if (field->type == IREE_JSON_VALUE_TYPE_STRING) {
    return loom_serve_text_chat_append_decoded(output, field->value);
  }
  return iree_json_enumerate_array_typed(
      field->value, loom_serve_text_chat_text_part, output);
}

// Request-wide metadata for client-selected JPEG frames. The timestamp table
// is built once, then shared by every video content part as in media_io_kwargs.
typedef struct loom_serve_text_chat_video_options_t {
  // Source frame rate used to turn frame indices into seconds.
  double fps;
  // Number of source frames, or zero when the client omitted it.
  uint64_t total_num_frames;
  // Per-clip transport bound; zero means all supplied frames were requested.
  iree_host_size_t maximum_frames;
  // Optional array of native doubles corresponding to frames_indices.
  iree_string_builder_t timestamps;
} loom_serve_text_chat_video_options_t;

typedef struct loom_serve_text_chat_video_indices_t {
  // Request-wide destination for the validated timestamps.
  loom_serve_text_chat_video_options_t* options;
  // Model input limit bounding timestamp storage before inspecting any image.
  iree_host_size_t maximum_count;
  // Last accepted source index, establishing presentation order.
  uint64_t last_index;
} loom_serve_text_chat_video_indices_t;

static iree_status_t loom_serve_text_chat_video_index(
    void* user_data, iree_host_size_t index, iree_json_value_type_t type,
    iree_string_view_t value) {
  loom_serve_text_chat_video_indices_t* indices = user_data;
  loom_serve_text_chat_video_options_t* options = indices->options;
  if (type != IREE_JSON_VALUE_TYPE_NUMBER || index >= indices->maximum_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "video frames_indices must contain integers within "
                            "the model input count bound");
  }
  uint64_t frame_index = 0;
  IREE_RETURN_IF_ERROR(iree_json_parse_uint64(value, &frame_index));
  const double timestamp = (double)frame_index / options->fps;
  if ((index && frame_index < indices->last_index) || !isfinite(timestamp)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "video frames_indices must be nondecreasing and "
                            "produce finite timestamps");
  }
  indices->last_index = frame_index;
  return iree_string_builder_append_string(
      &options->timestamps,
      iree_make_string_view((const char*)&timestamp, sizeof(timestamp)));
}

static iree_status_t loom_serve_text_chat_video_options_read(
    iree_string_view_t object, iree_host_size_t maximum_count,
    loom_serve_text_chat_video_options_t* options) {
  options->fps = 1.0;
  options->maximum_frames = 32;
  if (!object.data) {
    return iree_ok_status();
  }
  loom_serve_text_chat_field_t modalities[] = {
      {"video", LOOM_SERVE_TEXT_CHAT_OBJECT}};
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_fields_read(
      object, IREE_ARRAYSIZE(modalities), modalities, 0));
  if (!modalities[0].value.data) {
    return iree_ok_status();
  }
  loom_serve_text_chat_field_t fields[] = {
      {"fps", LOOM_SERVE_TEXT_CHAT_NUMBER},
      {"frames_indices", LOOM_SERVE_TEXT_CHAT_ARRAY},
      {"total_num_frames", LOOM_SERVE_TEXT_CHAT_NUMBER},
      {"duration", LOOM_SERVE_TEXT_CHAT_NUMBER},
      {"num_frames", LOOM_SERVE_TEXT_CHAT_NUMBER},
      {"do_sample_frames", LOOM_SERVE_TEXT_CHAT_BOOL}};
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_fields_read(
      modalities[0].value, IREE_ARRAYSIZE(fields), fields, 0));
  if (fields[0].value.data &&
      (!iree_string_view_atod(fields[0].value, &options->fps) ||
       !isfinite(options->fps) || options->fps <= 0.0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "video fps must be positive and finite");
  }
  if (fields[1].value.data) {
    loom_serve_text_chat_video_indices_t indices = {
        .options = options, .maximum_count = maximum_count};
    IREE_RETURN_IF_ERROR(iree_json_enumerate_array_typed(
        fields[1].value, loom_serve_text_chat_video_index, &indices));
    if (!options->timestamps.size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "video frames_indices must not be empty");
    }
  }
  if (fields[2].value.data) {
    IREE_RETURN_IF_ERROR(
        iree_json_parse_uint64(fields[2].value, &options->total_num_frames));
    if (!options->total_num_frames) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "video total_num_frames must be positive");
    }
  }
  if (fields[3].value.data) {
    double duration = 0.0;
    if (!iree_string_view_atod(fields[3].value, &duration) ||
        !isfinite(duration) || duration < 0.0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "video duration must be finite and nonnegative");
    }
    // Duration and total_num_frames describe the original clip. Client-selected
    // frames already establish this request's timeline without resampling.
  }
  if (fields[4].value.data) {
    int64_t num_frames = 0;
    IREE_RETURN_IF_ERROR(iree_json_parse_int64(fields[4].value, &num_frames));
    if (num_frames != -1) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "video num_frames supports only -1 (all supplied "
                              "frames); select frames in the client");
    }
    options->maximum_frames = 0;
  }
  if (fields[5].type == IREE_JSON_VALUE_TYPE_TRUE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "video requires do_sample_frames=false; select "
                            "frames in the client");
  }
  return iree_ok_status();
}

typedef struct loom_serve_text_chat_content_t {
  // Request receiving inspected image descriptors.
  loom_serve_text_chat_t* chat;
  // Validated request-wide video timing and frame-selection options.
  const loom_serve_text_chat_video_options_t* video;
  // Message ordinal containing these parts.
  iree_host_size_t message_ordinal;
  // Decoded text concatenated in content-part order.
  iree_string_builder_t* text;
  // Little-endian source-policy content records.
  iree_string_builder_t* records;
  // Message-local display geometry and presentation times.
  iree_string_builder_t* frames;
  // Request input base corresponding to frame zero in this message.
  iree_host_size_t first_image;
  // Number of complete records in records.
  iree_host_size_t count;
} loom_serve_text_chat_content_t;

static iree_status_t loom_serve_text_chat_content_record(
    loom_serve_text_chat_content_t* content,
    loom_serve_text_chat_content_kind_t kind, uint32_t text_offset,
    uint32_t text_length, uint32_t image_begin, uint32_t image_count,
    loom_serve_text_chat_image_detail_t detail) {
  if (content->count >= LOOM_SERVE_TEXT_CHAT_MAX_CONTENT_PART_COUNT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "chat messages support at most %d ordered content parts",
        LOOM_SERVE_TEXT_CHAT_MAX_CONTENT_PART_COUNT);
  }
  uint8_t record[LOOM_SERVE_TEXT_CHAT_CONTENT_FIELD_COUNT * sizeof(uint32_t)] =
      {0};
  iree_unaligned_store_le_u32(
      record + LOOM_SERVE_TEXT_CHAT_CONTENT_FIELD_KIND * sizeof(uint32_t),
      kind);
  iree_unaligned_store_le_u32(
      record +
          LOOM_SERVE_TEXT_CHAT_CONTENT_FIELD_TEXT_OFFSET * sizeof(uint32_t),
      text_offset);
  iree_unaligned_store_le_u32(
      record +
          LOOM_SERVE_TEXT_CHAT_CONTENT_FIELD_TEXT_LENGTH * sizeof(uint32_t),
      text_length);
  iree_unaligned_store_le_u32(
      record +
          LOOM_SERVE_TEXT_CHAT_CONTENT_FIELD_IMAGE_BEGIN * sizeof(uint32_t),
      image_begin);
  iree_unaligned_store_le_u32(
      record +
          LOOM_SERVE_TEXT_CHAT_CONTENT_FIELD_IMAGE_COUNT * sizeof(uint32_t),
      image_count);
  iree_unaligned_store_le_u32(
      record +
          LOOM_SERVE_TEXT_CHAT_CONTENT_FIELD_IMAGE_DETAIL * sizeof(uint32_t),
      detail);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      content->records,
      iree_make_string_view((const char*)record, sizeof(record))));
  ++content->count;
  return iree_ok_status();
}

static iree_status_t loom_serve_text_chat_media_detail(
    const loom_serve_text_chat_field_t* field,
    loom_serve_text_chat_image_detail_t* out_detail) {
  *out_detail = LOOM_SERVE_TEXT_CHAT_IMAGE_DETAIL_AUTO;
  if (!field->value.data ||
      iree_string_view_equal(field->value, IREE_SV("auto"))) {
    return iree_ok_status();
  }
  if (iree_string_view_equal(field->value, IREE_SV("low"))) {
    *out_detail = LOOM_SERVE_TEXT_CHAT_IMAGE_DETAIL_LOW;
  } else if (iree_string_view_equal(field->value, IREE_SV("high"))) {
    *out_detail = LOOM_SERVE_TEXT_CHAT_IMAGE_DETAIL_HIGH;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image detail must be auto, low, or high");
  }
  return iree_ok_status();
}

// Inspect one input and publish its established facts to both the native
// request and the message-local source frame table. Pixels remain deferred.
static iree_status_t loom_serve_text_chat_inspect_image(
    loom_serve_text_chat_content_t* content, iree_host_size_t content_ordinal,
    loom_serve_text_chat_content_kind_t kind,
    loom_serve_media_image_source_t source,
    loom_serve_text_chat_image_detail_t detail, double timestamp) {
  if (content->message_ordinal > UINT32_MAX || content_ordinal > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "chat content ordinal exceeds 32 bits");
  }
  if (!content->chat->policy->images.maximum_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "this model does not accept image input");
  }
  if (content->chat->images.count >=
      content->chat->policy->images.maximum_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "chat request supports at most %zu image occurrences",
        content->chat->policy->images.maximum_count);
  }
  loom_serve_media_image_info_t info;
  IREE_RETURN_IF_ERROR(loom_serve_media_image_inspect(
      source, &content->chat->policy->images.limits,
      content->chat->host_allocator, &info));
  const iree_host_size_t image_index = content->chat->images.count;
  const iree_host_size_t required_capacity = image_index + 1;
  if (required_capacity > content->chat->images.capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        content->chat->host_allocator, required_capacity,
        sizeof(*content->chat->images.values), &content->chat->images.capacity,
        (void**)&content->chat->images.values));
  }
  uint8_t frame[LOOM_SERVE_TEXT_CHAT_FRAME_BYTES];
  iree_unaligned_store_le_u32(frame + LOOM_SERVE_TEXT_CHAT_FRAME_WIDTH_OFFSET,
                              info.display_width);
  iree_unaligned_store_le_u32(frame + LOOM_SERVE_TEXT_CHAT_FRAME_HEIGHT_OFFSET,
                              info.display_height);
  iree_unaligned_store_le_f64(
      frame + LOOM_SERVE_TEXT_CHAT_FRAME_TIMESTAMP_OFFSET, timestamp);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      content->frames,
      iree_make_string_view((const char*)frame, sizeof(frame))));
  content->chat->images.values[image_index] = (loom_serve_text_chat_image_t){
      .source = source,
      .detail = detail,
      .kind = kind,
      .timestamp = timestamp,
      .message_ordinal = (uint32_t)content->message_ordinal,
      .content_ordinal = (uint32_t)content_ordinal,
      .text_offset = (uint32_t)content->text->size,
      .info = info,
  };
  ++content->chat->images.count;
  return iree_ok_status();
}

static iree_status_t loom_serve_text_chat_image_part(
    loom_serve_text_chat_content_t* content, iree_host_size_t content_ordinal,
    iree_string_view_t value) {
  loom_serve_text_chat_field_t fields[] = {
      {"url", LOOM_SERVE_TEXT_CHAT_STRING},
      {"detail", LOOM_SERVE_TEXT_CHAT_STRING}};
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_fields_read(
      value, IREE_ARRAYSIZE(fields), fields, 0));
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_required(&fields[0]));
  loom_serve_text_chat_image_detail_t detail;
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_media_detail(&fields[1], &detail));
  const uint32_t image_begin =
      (uint32_t)(content->chat->images.count - content->first_image);
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_inspect_image(
      content, content_ordinal, LOOM_SERVE_TEXT_CHAT_CONTENT_IMAGE,
      loom_serve_media_image_source_from_data_url(fields[0].value), detail,
      0.0));
  return loom_serve_text_chat_content_record(content,
                                             LOOM_SERVE_TEXT_CHAT_CONTENT_IMAGE,
                                             0, 0, image_begin, 1, detail);
}

// Split a validated JSON string without copying the clip. A comma may itself
// be escaped as \u002c; other complete escapes remain in the borrowed slice
// for the image transport decoder. JSON parsing already checked escape bounds.
static iree_string_view_t loom_serve_text_chat_video_component(
    iree_string_view_t* cursor) {
  iree_host_size_t offset = 0;
  iree_host_size_t delimiter_length = 0;
  while (offset < cursor->size) {
    const char* data = cursor->data + offset;
    if (*data == ',') {
      delimiter_length = 1;
      break;
    }
    if (*data == '\\') {
      if (data[1] == 'u' && data[2] == '0' && data[3] == '0' &&
          data[4] == '2' && (data[5] == 'c' || data[5] == 'C')) {
        delimiter_length = 6;
        break;
      }
      offset += data[1] == 'u' ? 6 : 2;
    } else {
      ++offset;
    }
  }
  const iree_string_view_t component =
      iree_string_view_substr(*cursor, 0, offset);
  *cursor = delimiter_length
                ? iree_string_view_substr(*cursor, offset + delimiter_length,
                                          IREE_HOST_SIZE_MAX)
                : iree_string_view_empty();
  return component;
}

static iree_status_t loom_serve_text_chat_video_part(
    loom_serve_text_chat_content_t* content, iree_host_size_t content_ordinal,
    iree_string_view_t value) {
  loom_serve_text_chat_field_t fields[] = {
      {"url", LOOM_SERVE_TEXT_CHAT_STRING}};
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_fields_read(
      value, IREE_ARRAYSIZE(fields), fields, 0));
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_required(&fields[0]));
  iree_string_view_t cursor = fields[0].value;
  const iree_string_view_t header =
      loom_serve_text_chat_video_component(&cursor);
  char prefix[sizeof("data:video/jpeg;base64") - 1];
  iree_host_size_t prefix_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_json_unescape_string(header, 0, NULL, &prefix_length));
  if (!cursor.data || prefix_length != sizeof(prefix)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "video_url requires a data:video/jpeg;base64 "
                            "frame sequence; containers and remote URLs are "
                            "not supported");
  }
  IREE_RETURN_IF_ERROR(iree_json_unescape_string(header, sizeof(prefix), prefix,
                                                 &prefix_length));
  if (!iree_string_view_equal_case(iree_make_string_view(prefix, prefix_length),
                                   IREE_SV("data:video/jpeg;base64"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "video_url requires data:video/jpeg;base64");
  }
  const loom_serve_text_chat_video_options_t* options = content->video;
  const iree_host_size_t timestamp_count =
      options->timestamps.size / sizeof(double);
  const char* timestamps = iree_string_builder_buffer(&options->timestamps);
  const iree_host_size_t first = content->chat->images.count;
  iree_host_size_t count = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) && cursor.data) {
    if (options->maximum_frames && count >= options->maximum_frames) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "video has more than 32 frames; set media_io_kwargs.video.num_frames "
          "to -1 to consume every supplied frame");
    } else if (timestamp_count && count >= timestamp_count) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "video frames_indices must match the supplied "
                                "frame count");
    } else {
      double timestamp = (double)count / options->fps;
      if (timestamp_count) {
        memcpy(&timestamp, timestamps + count * sizeof(double),
               sizeof(timestamp));
      }
      if (!isfinite(timestamp)) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "video fps produces a nonfinite timestamp");
      } else {
        const loom_serve_media_image_source_t source = {
            .value = loom_serve_text_chat_video_component(&cursor),
            .format = LOOM_SERVE_MEDIA_IMAGE_FORMAT_JPEG,
        };
        status = loom_serve_text_chat_inspect_image(
            content, content_ordinal, LOOM_SERVE_TEXT_CHAT_CONTENT_VIDEO,
            source, LOOM_SERVE_TEXT_CHAT_IMAGE_DETAIL_AUTO, timestamp);
        ++count;
      }
    }
  }
  IREE_RETURN_IF_ERROR(status);
  if (timestamp_count && count != timestamp_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "video frames_indices must match the supplied frame count");
  }
  if (options->total_num_frames && options->total_num_frames < count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "video total_num_frames must cover the supplied frames");
  }
  return loom_serve_text_chat_content_record(
      content, LOOM_SERVE_TEXT_CHAT_CONTENT_VIDEO, 0, 0,
      (uint32_t)(first - content->first_image), (uint32_t)count,
      LOOM_SERVE_TEXT_CHAT_IMAGE_DETAIL_AUTO);
}

static iree_status_t loom_serve_text_chat_ordered_content_part(
    void* user_data, iree_host_size_t index, iree_json_value_type_t type,
    iree_string_view_t value) {
  loom_serve_text_chat_content_t* content = user_data;
  if (type != IREE_JSON_VALUE_TYPE_OBJECT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "chat content parts must be objects");
  }
  loom_serve_text_chat_field_t fields[] = {
      {"type", LOOM_SERVE_TEXT_CHAT_STRING},
      {"text", LOOM_SERVE_TEXT_CHAT_STRING},
      {"image_url", LOOM_SERVE_TEXT_CHAT_OBJECT},
      {"video_url", LOOM_SERVE_TEXT_CHAT_OBJECT}};
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_fields_read(
      value, IREE_ARRAYSIZE(fields), fields, 0));
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_required(&fields[0]));
  if (iree_string_view_equal(fields[0].value, IREE_SV("text"))) {
    IREE_RETURN_IF_ERROR(loom_serve_text_chat_required(&fields[1]));
    if (fields[2].value.data || fields[3].value.data) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "text content parts cannot contain media");
    }
    if (content->text->size > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "decoded chat content exceeds 4 GiB");
    }
    const uint32_t offset = (uint32_t)content->text->size;
    IREE_RETURN_IF_ERROR(
        loom_serve_text_chat_append_decoded(content->text, fields[1].value));
    if (content->text->size > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "decoded chat content exceeds 4 GiB");
    }
    const iree_host_size_t length = content->text->size - offset;
    return loom_serve_text_chat_content_record(
        content, LOOM_SERVE_TEXT_CHAT_CONTENT_TEXT, offset, (uint32_t)length, 0,
        0, LOOM_SERVE_TEXT_CHAT_IMAGE_DETAIL_AUTO);
  }
  if (iree_string_view_equal(fields[0].value, IREE_SV("image_url"))) {
    IREE_RETURN_IF_ERROR(loom_serve_text_chat_required(&fields[2]));
    if (fields[1].value.data || fields[3].value.data) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "image content parts cannot contain text or video_url");
    }
    return loom_serve_text_chat_image_part(content, index, fields[2].value);
  }
  if (iree_string_view_equal(fields[0].value, IREE_SV("video_url"))) {
    IREE_RETURN_IF_ERROR(loom_serve_text_chat_required(&fields[3]));
    if (fields[1].value.data || fields[2].value.data) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "video content parts cannot contain text or image_url");
    }
    return loom_serve_text_chat_video_part(content, index, fields[3].value);
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "content part type must be text, image_url, or video_url");
}

static iree_status_t loom_serve_text_chat_ordered_content(
    const loom_serve_text_chat_field_t* field, loom_serve_text_chat_t* chat,
    iree_host_size_t message_ordinal,
    const loom_serve_text_chat_video_options_t* video,
    iree_string_builder_t* text, iree_string_builder_t* records,
    iree_string_builder_t* frames, iree_host_size_t* out_count) {
  iree_string_builder_reset(text);
  iree_string_builder_reset(records);
  iree_string_builder_reset(frames);
  *out_count = 0;
  if (!field->value.data || field->type == IREE_JSON_VALUE_TYPE_NULL) {
    return iree_ok_status();
  }
  if (field->type == IREE_JSON_VALUE_TYPE_STRING) {
    return loom_serve_text_chat_append_decoded(text, field->value);
  }
  loom_serve_text_chat_content_t content = {.chat = chat,
                                            .video = video,
                                            .message_ordinal = message_ordinal,
                                            .text = text,
                                            .records = records,
                                            .frames = frames,
                                            .first_image = chat->images.count,
                                            .count = 0};
  IREE_RETURN_IF_ERROR(iree_json_enumerate_array_typed(
      field->value, loom_serve_text_chat_ordered_content_part, &content));
  *out_count = content.count;
  return iree_ok_status();
}

void loom_serve_text_chat_policy_deinitialize(
    loom_serve_text_chat_policy_t* policy) {
  iree_vm_buffer_release(policy->name_storage);
  iree_vm_buffer_release(policy->defaults);
  memset(policy, 0, sizeof(*policy));
}

iree_status_t loom_serve_text_chat_policy_initialize(
    iree_vm_environment_t* environment, loom_serve_program_t* program,
    iree_vm_buffer_t* options, loom_serve_text_chat_policy_t* out_policy) {
  memset(out_policy, 0, sizeof(*out_policy));
  IREE_RETURN_IF_ERROR(iree_vm_ref_types_resolve(
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
      &out_policy->types));
  out_policy->invocation = loom_serve_program_invocation(program);
  iree_vm_function_t name_function = {0};
  iree_vm_function_t defaults_function = {0};
  iree_vm_function_t image_limits_function = {0};
  iree_vm_function_t reasoning_function = {0};
  const iree_vm_signature_type_t buffer = {
      .kind = IREE_VM_SIGNATURE_TYPE_KIND_REF,
      .value.ref = out_policy->types.buffer};
  const iree_vm_signature_type_t i32 = {
      .kind = IREE_VM_SIGNATURE_TYPE_KIND_SCALAR,
      .value.scalar = IREE_VM_SCALAR_TYPE_I32};
  const iree_vm_signature_type_t i64 = {
      .kind = IREE_VM_SIGNATURE_TYPE_KIND_SCALAR,
      .value.scalar = IREE_VM_SCALAR_TYPE_I64};
  const iree_vm_signature_type_t buffers[] = {buffer, buffer};
  const iree_vm_signature_type_t reasoning_results[] = {i32, i32, i32};
  const iree_vm_signature_type_t fragment_results[] = {buffer, i64};
  const iree_vm_signature_type_t media_results[] = {buffer, i64, buffer};
  const iree_vm_signature_type_t message_results[] = {buffer, i64, buffer, i64};
  const iree_vm_signature_type_t image_limit_results[] = {i64, i64, i64, i64,
                                                          i64};
  const iree_vm_signature_type_t message_arguments[] = {
      buffer, buffer, buffer, buffer, buffer, buffer,
      buffer, i64,    i64,    i64,    i64};
  const iree_vm_signature_type_t tool_arguments[] = {buffer, buffer, i64};
  const iree_vm_signature_type_t input_arguments[] = {buffer, buffer, i64,
                                                      i32,    i32,    i64};
  const iree_vm_signature_type_t output_arguments[] = {buffer, i64, i64, i32};
  const iree_vm_signature_type_t output_results[] = {i64, i64, i64, i64};
  const iree_vm_signature_type_t completion_arguments[] = {
      buffer, buffer, buffer, buffer, buffer, i64, i64};
  const iree_vm_signature_type_t completion_results[] = {buffer, i32};
  const loom_serve_program_binding_t bindings[] = {
      {IREE_SVL("model_name"), {NULL, 0}, {&buffer, 1}, &name_function},
      {IREE_SVL("chat_reasoning"),
       {NULL, 0},
       {reasoning_results, IREE_ARRAYSIZE(reasoning_results)},
       &reasoning_function},
      {IREE_SVL("chat_image_limits"),
       {NULL, 0},
       {image_limit_results, IREE_ARRAYSIZE(image_limit_results)},
       &image_limits_function},
      {IREE_SVL("chat_begin"),
       {buffers, IREE_ARRAYSIZE(buffers)},
       {fragment_results, IREE_ARRAYSIZE(fragment_results)},
       &out_policy->chat_begin},
      {IREE_SVL("chat_message"),
       {message_arguments, IREE_ARRAYSIZE(message_arguments)},
       {message_results, IREE_ARRAYSIZE(message_results)},
       &out_policy->chat_message},
      {IREE_SVL("chat_end"),
       {&i64, 1},
       {fragment_results, IREE_ARRAYSIZE(fragment_results)},
       &out_policy->chat_end},
      {IREE_SVL("parse_tools"),
       {tool_arguments, IREE_ARRAYSIZE(tool_arguments)},
       {&buffer, 1},
       &out_policy->parse_tools},
      {IREE_SVL("render_tool"),
       {tool_arguments, IREE_ARRAYSIZE(tool_arguments)},
       {fragment_results, IREE_ARRAYSIZE(fragment_results)},
       &out_policy->render_tool},
      {IREE_SVL("prepare_input"),
       {input_arguments, IREE_ARRAYSIZE(input_arguments)},
       {media_results, IREE_ARRAYSIZE(media_results)},
       &out_policy->prepare_input},
      {IREE_SVL("output_next"),
       {output_arguments, IREE_ARRAYSIZE(output_arguments)},
       {output_results, IREE_ARRAYSIZE(output_results)},
       &out_policy->output_next},
      {IREE_SVL("complete_text"),
       {completion_arguments, IREE_ARRAYSIZE(completion_arguments)},
       {completion_results, IREE_ARRAYSIZE(completion_results)},
       &out_policy->complete_text},
      {IREE_SVL("request_defaults"),
       {&buffer, 1},
       {&buffer, 1},
       &defaults_function},
      {IREE_SVL("request_options"),
       {tool_arguments, IREE_ARRAYSIZE(tool_arguments)},
       {&buffer, 1},
       &out_policy->request_options},
  };
  iree_status_t status = loom_serve_program_bind_service(
      program, IREE_ARRAYSIZE(bindings), bindings);
  iree_vm_variant_t result = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(out_policy->invocation, name_function,
                            iree_vm_variant_span_empty(),
                            iree_vm_variant_span_from_ptr(&result, 1));
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(&out_policy->types, &result,
                                                  &out_policy->name_storage);
  }
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  if (iree_status_is_ok(status)) {
    iree_vm_variant_t argument =
        iree_vm_buffer_variant_from_ptr_borrowed(&out_policy->types, options);
    status = iree_vm_invoke(out_policy->invocation, defaults_function,
                            iree_vm_variant_span_from_ptr(&argument, 1),
                            iree_vm_variant_span_from_ptr(&result, 1));
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(&out_policy->types, &result,
                                                  &out_policy->defaults);
  }
  if (iree_status_is_ok(status) && !out_policy->defaults) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source request defaults must not be null");
  }
  if (iree_status_is_ok(status) && !out_policy->name_storage) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source model name is null");
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_read(
        out_policy->name_storage, 0,
        iree_vm_buffer_length(out_policy->name_storage), &bytes);
  }
  if (iree_status_is_ok(status)) {
    out_policy->name =
        iree_make_string_view((const char*)bytes.data, bytes.data_length);
    if (!out_policy->name.size ||
        !iree_unicode_utf8_validate(out_policy->name)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "source model name must be nonempty UTF-8");
    }
  }
  iree_vm_variant_reset(&result);
  iree_vm_variant_t reasoning_values[3] = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(out_policy->invocation, reasoning_function,
                            iree_vm_variant_span_empty(),
                            iree_vm_variant_span_from_array(reasoning_values));
  }
  int32_t reasoning[IREE_ARRAYSIZE(reasoning_values)] = {0};
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(reasoning_values) && iree_status_is_ok(status); ++i) {
    status = iree_vm_i32_from_variant(reasoning_values[i], &reasoning[i]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_reasoning_initialize(
        reasoning[0], reasoning[1], reasoning[2], &out_policy->reasoning);
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(reasoning_values));
  iree_vm_variant_t image_limits[5] = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(out_policy->invocation, image_limits_function,
                            iree_vm_variant_span_empty(),
                            iree_vm_variant_span_from_array(image_limits));
  }
  int64_t limits[IREE_ARRAYSIZE(image_limits)] = {0};
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(image_limits) && iree_status_is_ok(status); ++i) {
    status = iree_vm_i64_from_variant(image_limits[i], &limits[i]);
  }
  if (iree_status_is_ok(status)) {
    const bool disabled = limits[0] == 0;
    const bool valid_disabled = disabled && limits[1] == 0 && limits[2] == 0 &&
                                limits[3] == 0 && limits[4] == 0;
    const bool valid_enabled =
        !disabled && limits[0] > 0 &&
        limits[0] <= LOOM_SERVE_TEXT_CHAT_MAX_CONTENT_PART_COUNT &&
        limits[1] > 0 && (uint64_t)limits[1] <= SIZE_MAX && limits[2] > 0 &&
        (uint64_t)limits[2] <= UINT32_MAX && limits[3] > 0 &&
        (uint64_t)limits[3] <= UINT32_MAX && limits[4] > 0 &&
        (uint64_t)limits[4] <= (uint64_t)limits[2] * (uint64_t)limits[3];
    if (!valid_disabled && !valid_enabled) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "source image limits must be all zero or positive bounded count, "
          "compressed bytes, width, height and pixels");
    } else if (valid_enabled) {
      out_policy->images.maximum_count = (iree_host_size_t)limits[0];
      out_policy->images.limits = (loom_serve_media_image_limits_t){
          .compressed_bytes = (iree_host_size_t)limits[1],
          .width = (uint32_t)limits[2],
          .height = (uint32_t)limits[3],
          .pixels = (uint64_t)limits[4],
      };
    }
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(image_limits));
  if (!iree_status_is_ok(status)) {
    loom_serve_text_chat_policy_deinitialize(out_policy);
  }
  return status;
}

// Request fragments never escape this synchronous call. The returned buffer
// may alias an input; it is appended before argument references are released.
static iree_status_t loom_serve_text_chat_render(
    const loom_serve_text_chat_policy_t* policy, iree_vm_function_t function,
    iree_host_size_t input_count, const iree_string_view_t* inputs,
    iree_host_size_t scalar_count, const int64_t* scalars,
    iree_string_builder_t* output, iree_allocator_t scratch_allocator,
    int64_t* out_state) {
  iree_vm_variant_t arguments[10] = {0};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < input_count && iree_status_is_ok(status);
       ++i) {
    iree_vm_buffer_t* buffer = NULL;
    status = iree_vm_buffer_wrap(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_byte_span((void*)inputs[i].data, inputs[i].size),
        iree_vm_buffer_release_callback_null(), scratch_allocator, &buffer);
    if (iree_status_is_ok(status)) {
      arguments[i] =
          iree_vm_buffer_variant_from_ptr_move(&policy->types, &buffer);
    }
  }
  for (iree_host_size_t i = 0; i < scalar_count; ++i) {
    arguments[input_count + i] = iree_vm_variant_from_i64(scalars[i]);
  }
  iree_vm_variant_t results[2] = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(
        policy->invocation, function,
        iree_vm_variant_span_from_ptr(arguments, input_count + scalar_count),
        iree_vm_variant_span_from_ptr(results, out_state ? 2 : 1));
  }
  iree_vm_buffer_t* rendered = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_borrowed(&policy->types,
                                                      results[0], &rendered);
  }
  if (iree_status_is_ok(status) && !rendered) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source returned a null chat fragment");
  }
  int64_t state = 0;
  if (iree_status_is_ok(status) && out_state) {
    status = iree_vm_i64_from_variant(results[1], &state);
  }
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_read(rendered, 0,
                                     iree_vm_buffer_length(rendered), &bytes);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(
        output,
        iree_make_string_view((const char*)bytes.data, bytes.data_length));
  }
  if (iree_status_is_ok(status) && out_state) {
    *out_state = state;
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  return status;
}

// Message rendering additionally establishes the exact source marker plan.
// The plan is checked before either the fragment or rendering state commits.
static iree_status_t loom_serve_text_chat_render_message(
    const loom_serve_text_chat_policy_t* policy, iree_host_size_t input_count,
    const iree_string_view_t* inputs, iree_host_size_t scalar_count,
    const int64_t* scalars, iree_host_size_t first_image,
    loom_serve_text_chat_t* chat, int64_t* out_state) {
  iree_vm_variant_t arguments[11] = {0};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < input_count && iree_status_is_ok(status);
       ++i) {
    iree_vm_buffer_t* buffer = NULL;
    status = iree_vm_buffer_wrap(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_byte_span((void*)inputs[i].data, inputs[i].size),
        iree_vm_buffer_release_callback_null(), chat->host_allocator, &buffer);
    if (iree_status_is_ok(status)) {
      arguments[i] =
          iree_vm_buffer_variant_from_ptr_move(&policy->types, &buffer);
    }
  }
  for (iree_host_size_t i = 0; i < scalar_count; ++i) {
    arguments[input_count + i] = iree_vm_variant_from_i64(scalars[i]);
  }
  iree_vm_variant_t results[4] = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(
        policy->invocation, policy->chat_message,
        iree_vm_variant_span_from_ptr(arguments, input_count + scalar_count),
        iree_vm_variant_span_from_array(results));
  }
  iree_vm_buffer_t* fragment = NULL;
  iree_vm_buffer_t* plan = NULL;
  int64_t state = 0;
  int64_t projection_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_borrowed(&policy->types,
                                                      results[0], &fragment);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_i64_from_variant(results[1], &state);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_borrowed(&policy->types,
                                                      results[2], &plan);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_i64_from_variant(results[3], &projection_count);
  }
  if (iree_status_is_ok(status) &&
      (projection_count < 0 ||
       (uint64_t)projection_count > chat->images.count - first_image)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source returned an invalid projection count");
  }
  if (iree_status_is_ok(status) && (!fragment || !plan)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source returned null message storage");
  }
  iree_const_byte_span_t fragment_bytes = iree_const_byte_span_empty();
  iree_const_byte_span_t plan_bytes = iree_const_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_read(
        fragment, 0, iree_vm_buffer_length(fragment), &fragment_bytes);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_read(plan, 0, iree_vm_buffer_length(plan),
                                     &plan_bytes);
  }
  const iree_host_size_t prompt_offset =
      iree_string_builder_size(&chat->prompt);
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_media_apply_message_plan(
        chat, first_image, prompt_offset, fragment_bytes.data_length,
        (iree_host_size_t)projection_count, plan_bytes);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(
        &chat->prompt, iree_make_string_view((const char*)fragment_bytes.data,
                                             fragment_bytes.data_length));
  }
  if (iree_status_is_ok(status)) {
    *out_state = state;
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  return status;
}

iree_status_t loom_serve_text_chat_prepare_input(
    const loom_serve_text_chat_t* chat, iree_host_size_t prompt_offset,
    loom_serve_text_chat_input_format_t format,
    loom_serve_text_chat_boundary_t boundary, iree_host_size_t capacity,
    iree_host_size_t input_shift, iree_allocator_t host_allocator,
    int32_t* out_tokens, iree_host_size_t* out_count,
    loom_serve_text_media_span_t** out_media_spans,
    iree_host_size_t* out_media_span_count) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_count = 0;
  *out_media_spans = NULL;
  *out_media_span_count = 0;
  const loom_serve_text_chat_policy_t* policy = chat->policy;
  const iree_string_view_t prompt = iree_string_builder_view(&chat->prompt);
  if (prompt_offset > prompt.size) {
    IREE_RETURN_AND_END_ZONE(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "input begins beyond the rendered prompt"));
  }
  const iree_string_view_t text =
      iree_string_view_substr(prompt, prompt_offset, IREE_HOST_SIZE_MAX);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, text.size);
  iree_vm_buffer_t* media_plan = NULL;
  iree_host_size_t marker_count = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, loom_serve_text_media_build_input_plan(
              chat, prompt_offset, host_allocator, &media_plan, &marker_count));
  iree_vm_buffer_t* input = NULL;
  iree_status_t status = iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_byte_span((void*)text.data, text.size),
      iree_vm_buffer_release_callback_null(), host_allocator, &input);
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_move(&policy->types, &input),
      iree_vm_buffer_variant_from_ptr_borrowed(&policy->types, media_plan),
      iree_vm_variant_from_i64((int64_t)marker_count),
      iree_vm_variant_from_i32(format),
      iree_vm_variant_from_i32(boundary),
      iree_vm_variant_from_i64(capacity)};
  iree_vm_variant_t results[3] = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(policy->invocation, policy->prepare_input,
                            iree_vm_variant_span_from_array(arguments),
                            iree_vm_variant_span_from_array(results));
  }
  iree_vm_buffer_t* output = NULL;
  iree_vm_buffer_t* media_spans = NULL;
  int64_t count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_borrowed(&policy->types,
                                                      results[0], &output);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_i64_from_variant(results[1], &count);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_borrowed(&policy->types,
                                                      results[2], &media_spans);
  }
  if (iree_status_is_ok(status) &&
      (!output || !media_spans || count < 0 || (uint64_t)count > capacity ||
       (uint64_t)count > SIZE_MAX / sizeof(*out_tokens))) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source input returned invalid token storage");
  }
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_buffer_map_read(output, 0, count * sizeof(*out_tokens), &bytes);
  }
  iree_const_byte_span_t span_bytes = iree_const_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_read(
        media_spans, 0, iree_vm_buffer_length(media_spans), &span_bytes);
  }
  loom_serve_text_media_span_t* parsed_spans = NULL;
  iree_host_size_t parsed_span_count = 0;
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_media_parse_spans(
        chat, chat->projections.count - marker_count, span_bytes,
        (iree_host_size_t)count, input_shift, host_allocator, &parsed_spans,
        &parsed_span_count);
  }
  if (iree_status_is_ok(status)) {
    if (out_tokens) {
      for (iree_host_size_t i = 0; i < (iree_host_size_t)count; ++i) {
        out_tokens[i] = (int32_t)iree_unaligned_load_le_u32(bytes.data + i * 4);
      }
    }
    *out_count = (iree_host_size_t)count;
    *out_media_spans = parsed_spans;
    *out_media_span_count = parsed_span_count;
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_buffer_release(media_plan);
  IREE_RETURN_AND_END_ZONE(z0, status);
}

// Function records have the same decoded name/argument semantics in client
// history and generated output. The scratch owner retains both through the
// synchronous source formatter call.
static iree_status_t loom_serve_text_chat_function_decode(
    iree_string_view_t value, iree_string_builder_t* scratch,
    iree_string_view_t* out_name, iree_string_view_t* out_arguments) {
  loom_serve_text_chat_field_t fields[] = {
      {"name", LOOM_SERVE_TEXT_CHAT_STRING},
      {"arguments", LOOM_SERVE_TEXT_CHAT_STRING}};
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_fields_read(
      value, IREE_ARRAYSIZE(fields), fields, 0));
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_required(&fields[0]));
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_required(&fields[1]));
  iree_string_builder_reset(scratch);
  IREE_RETURN_IF_ERROR(
      loom_serve_text_chat_append_decoded(scratch, fields[0].value));
  const iree_host_size_t name_length = iree_string_builder_size(scratch);
  IREE_RETURN_IF_ERROR(
      loom_serve_text_chat_append_decoded(scratch, fields[1].value));
  const iree_string_view_t decoded = iree_string_builder_view(scratch);
  *out_name = iree_string_view_substr(decoded, 0, name_length);
  *out_arguments =
      iree_string_view_substr(decoded, name_length, IREE_HOST_SIZE_MAX);
  return iree_ok_status();
}

// Input wrappers borrow only for this synchronous invocation. Source owns the
// returned bytes until they are appended; no references escape into a request.
static iree_status_t loom_serve_text_chat_function_render(
    const loom_serve_text_chat_policy_t* policy, iree_string_view_t name,
    iree_string_view_t arguments, iree_host_size_t ordinal,
    iree_string_builder_t* output, iree_allocator_t scratch_allocator) {
  const iree_string_view_t inputs[] = {name, arguments};
  iree_vm_variant_t values[3] = {0};
  values[2] = iree_vm_variant_from_i64(ordinal);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(inputs) && iree_status_is_ok(status); ++i) {
    iree_vm_buffer_t* buffer = NULL;
    status = iree_vm_buffer_wrap(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_byte_span((void*)inputs[i].data, inputs[i].size),
        iree_vm_buffer_release_callback_null(), scratch_allocator, &buffer);
    if (iree_status_is_ok(status)) {
      values[i] = iree_vm_buffer_variant_from_ptr_move(&policy->types, &buffer);
    }
  }
  iree_vm_variant_t results[2] = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(policy->invocation, policy->render_tool,
                            iree_vm_variant_span_from_array(values),
                            iree_vm_variant_span_from_array(results));
  }
  iree_vm_buffer_t* rendered = NULL;
  int64_t length = 0;
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_borrowed(&policy->types,
                                                      results[0], &rendered);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_i64_from_variant(results[1], &length);
  }
  if (iree_status_is_ok(status) &&
      (!rendered || length < 0 || (uint64_t)length > SIZE_MAX)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source tool formatter returned invalid storage");
  }
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_buffer_map_read(rendered, 0, (iree_host_size_t)length, &bytes);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(
        output,
        iree_make_string_view((const char*)bytes.data, bytes.data_length));
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(values));
  return status;
}

typedef struct loom_serve_text_chat_history_t {
  // Expected deployment identity, borrowed only while rendering the request.
  iree_string_view_t model_name;
  // Request being rendered.
  loom_serve_text_chat_t* chat;
  // Request-wide video timing, shared by every content part.
  loom_serve_text_chat_video_options_t video;
  // Decoded message content scratch.
  iree_string_builder_t scratch;
  // Ordered source-policy content records for one message.
  iree_string_builder_t content_records;
  // Message-local display geometry and presentation time for inspected inputs.
  iree_string_builder_t frame_records;
  // Decoded function name and argument JSON, stable through source rendering.
  iree_string_builder_t arguments;
  // Decoded reasoning content for one message.
  iree_string_builder_t reasoning;
  // Canonical tool calls for one message.
  iree_string_builder_t calls;
  // Source-owned rendering state, with no native interpretation.
  int64_t state;
} loom_serve_text_chat_history_t;

static iree_status_t loom_serve_text_chat_history_tool(
    void* user_data, iree_host_size_t index, iree_json_value_type_t type,
    iree_string_view_t value) {
  loom_serve_text_chat_history_t* history = user_data;
  if (type != IREE_JSON_VALUE_TYPE_OBJECT ||
      index >= LOOM_SERVE_TEXT_CHAT_MAX_TOOL_COUNT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "assistant tool_calls must contain at most %d objects",
        LOOM_SERVE_TEXT_CHAT_MAX_TOOL_COUNT);
  }
  loom_serve_text_chat_field_t fields[] = {
      {"id", LOOM_SERVE_TEXT_CHAT_STRING},
      {"type", LOOM_SERVE_TEXT_CHAT_STRING},
      {"function", LOOM_SERVE_TEXT_CHAT_OBJECT}};
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_fields_read(
      value, IREE_ARRAYSIZE(fields), fields, 0));
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_required(&fields[2]));
  if (!iree_string_view_equal(fields[1].value, IREE_SV("function"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "assistant tool_calls must be functions");
  }
  iree_string_view_t name, arguments;
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_function_decode(
      fields[2].value, &history->arguments, &name, &arguments));
  return loom_serve_text_chat_function_render(history->chat->policy, name,
                                              arguments, index, &history->calls,
                                              history->chat->host_allocator);
}

static iree_status_t loom_serve_text_chat_history_message(
    void* user_data, iree_host_size_t index, iree_json_value_type_t type,
    iree_string_view_t value) {
  loom_serve_text_chat_history_t* history = user_data;
  if (type != IREE_JSON_VALUE_TYPE_OBJECT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "messages must contain objects");
  }
  loom_serve_text_chat_field_t fields[] = {
      {"role", LOOM_SERVE_TEXT_CHAT_STRING},
      {"content", LOOM_SERVE_TEXT_CHAT_STRING | LOOM_SERVE_TEXT_CHAT_ARRAY |
                      LOOM_SERVE_TEXT_CHAT_NULL},
      {"reasoning_content", LOOM_SERVE_TEXT_CHAT_STRING},
      {"tool_calls", LOOM_SERVE_TEXT_CHAT_ARRAY},
      {"tool_call_id", LOOM_SERVE_TEXT_CHAT_STRING},
      {"name", LOOM_SERVE_TEXT_CHAT_STRING}};
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_fields_read(
      value, IREE_ARRAYSIZE(fields), fields,
      LOOM_SERVE_TEXT_CHAT_FIELDS_IGNORE_UNKNOWN));
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_required(&fields[0]));
  const iree_host_size_t first_image = history->chat->images.count;
  iree_host_size_t content_count = 0;
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_ordered_content(
      &fields[1], history->chat, index, &history->video, &history->scratch,
      &history->content_records, &history->frame_records, &content_count));
  IREE_RETURN_IF_ERROR(
      loom_serve_text_chat_text_content(&fields[2], &history->reasoning));
  iree_string_builder_reset(&history->calls);
  if (fields[3].value.data) {
    IREE_RETURN_IF_ERROR(iree_json_enumerate_array_typed(
        fields[3].value, loom_serve_text_chat_history_tool, history));
  }
  const iree_string_view_t inputs[] = {
      value,
      fields[0].value,
      iree_string_builder_view(&history->scratch),
      iree_string_builder_view(&history->content_records),
      iree_string_builder_view(&history->frame_records),
      iree_string_builder_view(&history->reasoning),
      iree_string_builder_view(&history->calls)};
  const int64_t scalars[] = {history->state, (int64_t)index,
                             (fields[2].value.data ? 1 : 0) |
                                 (fields[3].value.data ? 2 : 0) |
                                 (fields[4].value.data ? 4 : 0),
                             (int64_t)content_count};
  return loom_serve_text_chat_render_message(
      history->chat->policy, IREE_ARRAYSIZE(inputs), inputs,
      IREE_ARRAYSIZE(scalars), scalars, first_image, history->chat,
      &history->state);
}

static iree_status_t loom_serve_text_chat_tool_limit(
    void* user_data, iree_host_size_t index, iree_json_value_type_t type,
    iree_string_view_t value) {
  if (index >= LOOM_SERVE_TEXT_CHAT_MAX_TOOL_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tools must contain at most %d entries",
                            LOOM_SERVE_TEXT_CHAT_MAX_TOOL_COUNT);
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_text_chat_request_render(
    iree_string_view_t body, loom_serve_text_chat_history_t* history) {
  if (!iree_unicode_utf8_validate(body)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "chat JSON must be UTF-8");
  }
  iree_string_view_t cursor = body;
  iree_string_view_t object;
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&cursor, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  if (cursor.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "trailing chat JSON");
  }
  loom_serve_text_chat_field_t fields[] = {
      {"model", LOOM_SERVE_TEXT_CHAT_STRING},
      {"messages", LOOM_SERVE_TEXT_CHAT_ARRAY},
      {"stream", LOOM_SERVE_TEXT_CHAT_BOOL | LOOM_SERVE_TEXT_CHAT_NULL},
      {"tools", LOOM_SERVE_TEXT_CHAT_ARRAY},
      {"max_tokens", LOOM_SERVE_TEXT_CHAT_NUMBER},
      {"max_completion_tokens", LOOM_SERVE_TEXT_CHAT_NUMBER},
      {"stream_options",
       LOOM_SERVE_TEXT_CHAT_OBJECT | LOOM_SERVE_TEXT_CHAT_NULL},
      {"store", LOOM_SERVE_TEXT_CHAT_BOOL},
      {"media_io_kwargs", LOOM_SERVE_TEXT_CHAT_OBJECT}};
  // Only native transport and resource controls are decoded here. Source
  // receives the complete request and owns its model-specific options.
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_fields_read(
      object, IREE_ARRAYSIZE(fields), fields,
      LOOM_SERVE_TEXT_CHAT_FIELDS_IGNORE_UNKNOWN));
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_required(&fields[1]));
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_required(&fields[0]));
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_video_options_read(
      fields[8].value, history->chat->policy->images.maximum_count,
      &history->video));
  iree_string_builder_reset(&history->scratch);
  IREE_RETURN_IF_ERROR(
      loom_serve_text_chat_append_decoded(&history->scratch, fields[0].value));
  if (!iree_string_view_equal(iree_string_builder_view(&history->scratch),
                              history->model_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "select model %.*s",
                            (int)history->model_name.size,
                            history->model_name.data);
  }
  if (fields[2].value.data && fields[2].type == IREE_JSON_VALUE_TYPE_TRUE) {
    history->chat->flags |= LOOM_SERVE_TEXT_CHAT_FLAG_STREAM;
  }
  if (fields[7].value.data && fields[7].type != IREE_JSON_VALUE_TYPE_FALSE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "only store=false is supported");
  }
  if (fields[4].value.data && fields[5].value.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "specify one output token limit");
  }
  const iree_string_view_t limit =
      fields[4].value.data ? fields[4].value : fields[5].value;
  if (limit.data) {
    uint64_t max_tokens = 0;
    IREE_RETURN_IF_ERROR(iree_json_parse_uint64(limit, &max_tokens));
    if (!max_tokens || max_tokens > LOOM_SERVE_TEXT_CHAT_MAX_OUTPUT_TOKENS) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "output limit must be in [1, %d]",
                              LOOM_SERVE_TEXT_CHAT_MAX_OUTPUT_TOKENS);
    }
    history->chat->max_tokens = (iree_host_size_t)max_tokens;
  }
  if (fields[6].value.data && fields[6].type != IREE_JSON_VALUE_TYPE_NULL) {
    if (!iree_any_bit_set(history->chat->flags,
                          LOOM_SERVE_TEXT_CHAT_FLAG_STREAM)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "stream_options requires stream=true");
    }
    loom_serve_text_chat_field_t options[] = {
        {"include_usage", LOOM_SERVE_TEXT_CHAT_BOOL}};
    IREE_RETURN_IF_ERROR(loom_serve_text_chat_fields_read(
        fields[6].value, IREE_ARRAYSIZE(options), options, 0));
    if (options[0].value.data && options[0].type == IREE_JSON_VALUE_TYPE_TRUE) {
      history->chat->flags |= LOOM_SERVE_TEXT_CHAT_FLAG_INCLUDE_USAGE;
    }
  }
  history->chat->tool_schemas =
      fields[3].value.data ? fields[3].value : IREE_SV("[]");
  IREE_RETURN_IF_ERROR(iree_json_enumerate_array_typed(
      history->chat->tool_schemas, loom_serve_text_chat_tool_limit, NULL));
  const iree_string_view_t inputs[] = {body, history->chat->tool_schemas};
  IREE_TRACE_ZONE_BEGIN_NAMED(z_begin, "loom_serve_text_chat_begin");
  iree_status_t status = loom_serve_text_chat_render(
      history->chat->policy, history->chat->policy->chat_begin,
      IREE_ARRAYSIZE(inputs), inputs, 0, NULL, &history->chat->prompt,
      history->chat->host_allocator, &history->state);
  IREE_TRACE_ZONE_END(z_begin);
  IREE_RETURN_IF_ERROR(status);
  IREE_TRACE_ZONE_BEGIN_NAMED(z_messages, "loom_serve_text_chat_messages");
  status = iree_json_enumerate_array_typed(
      fields[1].value, loom_serve_text_chat_history_message, history);
  IREE_TRACE_ZONE_END(z_messages);
  IREE_RETURN_IF_ERROR(status);
  IREE_TRACE_ZONE_BEGIN_NAMED(z_end, "loom_serve_text_chat_end");
  status = loom_serve_text_chat_render(
      history->chat->policy, history->chat->policy->chat_end, 0, NULL, 1,
      &history->state, &history->chat->prompt, history->chat->host_allocator,
      &history->chat->output.initial_state);
  IREE_TRACE_ZONE_END(z_end);
  IREE_RETURN_IF_ERROR(status);
  history->chat->output.state = history->chat->output.initial_state;
  return iree_ok_status();
}

static iree_status_t loom_serve_text_chat_resolve_options(
    loom_serve_text_chat_t* chat, iree_string_view_t body,
    uint64_t request_id) {
  const loom_serve_text_chat_policy_t* policy = chat->policy;
  iree_vm_buffer_t* input = NULL;
  // Source may retain or return this input as its opaque request options.
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                           iree_make_const_byte_span(body.data, body.size), 1,
                           chat->prompt.allocator, &input));
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&policy->types,
                                               policy->defaults),
      iree_vm_buffer_variant_from_ptr_move(&policy->types, &input),
      iree_vm_variant_from_i64((int64_t)request_id)};
  iree_vm_variant_t result = {0};
  iree_status_t status =
      iree_vm_invoke(policy->invocation, policy->request_options,
                     iree_vm_variant_span_from_array(arguments),
                     iree_vm_variant_span_from_ptr(&result, 1));
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(&policy->types, &result,
                                                  &chat->options);
  }
  if (iree_status_is_ok(status) && !chat->options) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source request options must not be null");
  }
  iree_vm_variant_reset(&result);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  return status;
}

iree_status_t loom_serve_text_chat_initialize(
    const loom_serve_text_chat_policy_t* policy, iree_string_view_t model_name,
    iree_string_view_t body, uint64_t request_id,
    iree_host_size_t default_max_tokens, iree_allocator_t host_allocator,
    loom_serve_text_chat_t* out_chat) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, body.size);
  memset(out_chat, 0, sizeof(*out_chat));
  out_chat->policy = policy;
  out_chat->max_tokens = default_max_tokens;
  out_chat->host_allocator = host_allocator;
  iree_string_builder_initialize(host_allocator, &out_chat->prompt);
  iree_string_builder_initialize(host_allocator,
                                 &out_chat->projections.metadata);
  iree_string_builder_initialize(host_allocator, &out_chat->output.content);
  iree_string_builder_initialize(host_allocator, &out_chat->output.reasoning);
  loom_serve_text_chat_history_t history = {.model_name = model_name,
                                            .chat = out_chat};
  iree_string_builder_initialize(host_allocator, &history.video.timestamps);
  iree_string_builder_initialize(host_allocator, &history.scratch);
  iree_string_builder_initialize(host_allocator, &history.content_records);
  iree_string_builder_initialize(host_allocator, &history.frame_records);
  iree_string_builder_initialize(host_allocator, &history.arguments);
  iree_string_builder_initialize(host_allocator, &history.reasoning);
  iree_string_builder_initialize(host_allocator, &history.calls);
  IREE_TRACE_ZONE_BEGIN_NAMED(z_render, "loom_serve_text_chat_request_render");
  iree_status_t status = loom_serve_text_chat_request_render(body, &history);
  IREE_TRACE_ZONE_END(z_render);
  if (iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_BEGIN_NAMED(z_options,
                                "loom_serve_text_chat_resolve_options");
    status = loom_serve_text_chat_resolve_options(out_chat, body, request_id);
    IREE_TRACE_ZONE_END(z_options);
  }
  iree_string_builder_deinitialize(&history.calls);
  iree_string_builder_deinitialize(&history.reasoning);
  iree_string_builder_deinitialize(&history.arguments);
  iree_string_builder_deinitialize(&history.frame_records);
  iree_string_builder_deinitialize(&history.content_records);
  iree_string_builder_deinitialize(&history.scratch);
  iree_string_builder_deinitialize(&history.video.timestamps);
  if (!iree_status_is_ok(status)) {
    loom_serve_text_chat_deinitialize(out_chat);
  }
  IREE_RETURN_AND_END_ZONE(z0, status);
}

void loom_serve_text_chat_deinitialize(loom_serve_text_chat_t* chat) {
  iree_string_builder_deinitialize(&chat->projections.metadata);
  iree_allocator_free(chat->host_allocator, chat->images.values);
  iree_allocator_free(chat->host_allocator, chat->projections.values);
  iree_vm_buffer_release(chat->options);
  iree_string_builder_deinitialize(&chat->output.reasoning);
  iree_string_builder_deinitialize(&chat->output.content);
  iree_string_builder_deinitialize(&chat->prompt);
  memset(chat, 0, sizeof(*chat));
}

iree_status_t loom_serve_text_chat_output_next(
    loom_serve_text_chat_t* chat, iree_string_view_t response,
    loom_serve_text_chat_output_phase_t phase,
    iree_allocator_t scratch_allocator,
    loom_serve_text_chat_output_t* out_output) {
  memset(out_output, 0, sizeof(*out_output));
  const loom_serve_text_chat_policy_t* policy = chat->policy;
  iree_vm_buffer_t* input = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_byte_span((void*)response.data, response.size),
      iree_vm_buffer_release_callback_null(), scratch_allocator, &input));
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_move(&policy->types, &input),
      iree_vm_variant_from_i64(chat->output.cursor),
      iree_vm_variant_from_i64(chat->output.state),
      iree_vm_variant_from_i32(phase)};
  iree_vm_variant_t results[4] = {0};
  iree_status_t status =
      iree_vm_invoke(policy->invocation, policy->output_next,
                     iree_vm_variant_span_from_array(arguments),
                     iree_vm_variant_span_from_array(results));
  int64_t values[4] = {0};
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(values) && iree_status_is_ok(status); ++i) {
    status = iree_vm_i64_from_variant(results[i], &values[i]);
  }
  const int64_t kind = values[0];
  const int64_t begin = values[1];
  const int64_t end = values[2];
  if (iree_status_is_ok(status) &&
      (kind < LOOM_SERVE_TEXT_CHAT_OUTPUT_HIDDEN ||
       kind > LOOM_SERVE_TEXT_CHAT_OUTPUT_REASONING || begin < 0 ||
       (uint64_t)begin < chat->output.cursor || end < begin ||
       (uint64_t)end > response.size ||
       ((uint64_t)end == chat->output.cursor &&
        values[3] != chat->output.state))) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source returned an invalid output span");
  }
  iree_string_view_t text = iree_string_view_empty();
  if (iree_status_is_ok(status) && kind != LOOM_SERVE_TEXT_CHAT_OUTPUT_HIDDEN) {
    text = iree_string_view_substr(response, begin, end - begin);
    status = iree_string_builder_append_string(
        kind == LOOM_SERVE_TEXT_CHAT_OUTPUT_CONTENT ? &chat->output.content
                                                    : &chat->output.reasoning,
        text);
  }
  if (iree_status_is_ok(status)) {
    *out_output = (loom_serve_text_chat_output_t){
        .advanced = (uint64_t)end != chat->output.cursor,
        .kind = (loom_serve_text_chat_output_kind_t)kind,
        .text = text};
    chat->output.cursor = (iree_host_size_t)end;
    chat->output.state = values[3];
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  return status;
}

void loom_serve_text_chat_completion_deinitialize(
    loom_serve_text_chat_completion_t* completion) {
  loom_serve_text_chat_media_identity_deinitialize(&completion->media);
  iree_vm_buffer_release(completion->storage);
  memset(completion, 0, sizeof(*completion));
}

void loom_serve_text_chat_completion_retain(
    const loom_serve_text_chat_completion_t* source,
    loom_serve_text_chat_completion_t* out_completion) {
  *out_completion = *source;
  iree_vm_buffer_retain(out_completion->storage);
  loom_serve_text_chat_media_identity_retain(&source->media,
                                             &out_completion->media);
}

bool loom_serve_text_chat_completion_is_prefix(
    const loom_serve_text_chat_completion_t* completion,
    const loom_serve_text_chat_t* chat) {
  return iree_string_view_starts_with(iree_string_builder_view(&chat->prompt),
                                      completion->transcript) &&
         loom_serve_text_chat_media_identity_is_prefix(&completion->media,
                                                       chat);
}

// Returned buffers may alias any argument. Owned input clones make every such
// result independent of HTTP claims, mutable builders and invocation storage.
static iree_status_t loom_serve_text_chat_checkpoint(
    const loom_serve_text_chat_t* chat, iree_string_view_t response,
    iree_string_view_t tools, iree_allocator_t host_allocator,
    loom_serve_text_chat_completion_t* out_completion) {
  const loom_serve_text_chat_policy_t* policy = chat->policy;
  const iree_string_view_t inputs[] = {
      iree_string_builder_view(&chat->prompt), response,
      iree_string_builder_view(&chat->output.content),
      iree_string_builder_view(&chat->output.reasoning), tools};
  iree_vm_variant_t arguments[7] = {0};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(inputs) && iree_status_is_ok(status); ++i) {
    iree_vm_buffer_t* buffer = NULL;
    status = iree_vm_buffer_clone(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_const_byte_span(inputs[i].data, inputs[i].size), 1,
        host_allocator, &buffer);
    if (iree_status_is_ok(status)) {
      arguments[i] =
          iree_vm_buffer_variant_from_ptr_move(&policy->types, &buffer);
    }
  }
  arguments[5] = iree_vm_variant_from_i64(chat->output.initial_state);
  arguments[6] = iree_vm_variant_from_i64(chat->output.state);
  iree_vm_variant_t results[2] = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(policy->invocation, policy->complete_text,
                            iree_vm_variant_span_from_array(arguments),
                            iree_vm_variant_span_from_array(results));
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(&policy->types, &results[0],
                                                  &out_completion->storage);
  }
  int32_t reusable = 0;
  if (iree_status_is_ok(status)) {
    status = iree_vm_i32_from_variant(results[1], &reusable);
  }
  if (iree_status_is_ok(status) && reusable != 0 && reusable != 1) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source returned invalid checkpoint eligibility");
  }
  if (iree_status_is_ok(status) && !out_completion->storage) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source returned null checkpoint storage");
  }
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_read(
        out_completion->storage, 0,
        iree_vm_buffer_length(out_completion->storage), &bytes);
  }
  if (iree_status_is_ok(status)) {
    out_completion->transcript =
        iree_make_string_view((const char*)bytes.data, bytes.data_length);
    out_completion->reusable = reusable != 0;
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  return status;
}

typedef struct loom_serve_text_chat_completion_tools_t {
  // Batch allocator for synchronous VM argument wrappers.
  iree_allocator_t scratch_allocator;
  // Source policy used for canonical history formatting.
  const loom_serve_text_chat_policy_t* policy;
  // Request-unique namespace for transport tool IDs.
  uint64_t request_id;
  // Owned validated records, published only after canonical history succeeds.
  loom_serve_text_tool_calls_t* calls;
  // Exclusively borrowed formatting storage, reused across completions.
  loom_serve_text_chat_scratch_t* scratch;
} loom_serve_text_chat_completion_tools_t;

static iree_status_t loom_serve_text_chat_completion_tool(
    void* user_data, iree_host_size_t index, iree_json_value_type_t type,
    iree_string_view_t value) {
  loom_serve_text_chat_completion_tools_t* tools = user_data;
  if (type != IREE_JSON_VALUE_TYPE_OBJECT ||
      index >= LOOM_SERVE_TEXT_CHAT_MAX_TOOL_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source tool calls must contain at most %d objects",
                            LOOM_SERVE_TEXT_CHAT_MAX_TOOL_COUNT);
  }
  loom_serve_text_chat_field_t call[] = {
      {"type", LOOM_SERVE_TEXT_CHAT_STRING},
      {"function", LOOM_SERVE_TEXT_CHAT_OBJECT}};
  IREE_RETURN_IF_ERROR(
      loom_serve_text_chat_fields_read(value, IREE_ARRAYSIZE(call), call, 0));
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_required(&call[1]));
  if (!iree_string_view_equal(call[0].value, IREE_SV("function"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source tool call must be a function");
  }
  iree_string_view_t name, arguments;
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_function_decode(
      call[1].value, &tools->scratch->arguments, &name, &arguments));
  IREE_RETURN_IF_ERROR(loom_serve_text_tool_calls_append(
      tools->calls, tools->request_id, name, arguments));
  const loom_serve_text_tool_call_t decoded_call =
      loom_serve_text_tool_calls_at(tools->calls, index);
  return loom_serve_text_chat_function_render(
      tools->policy, decoded_call.name, decoded_call.arguments, index,
      &tools->scratch->rendered, tools->scratch_allocator);
}

void loom_serve_text_chat_scratch_initialize(
    iree_allocator_t host_allocator,
    loom_serve_text_chat_scratch_t* out_scratch) {
  iree_string_builder_initialize(host_allocator, &out_scratch->parsed);
  iree_string_builder_initialize(host_allocator, &out_scratch->rendered);
  iree_string_builder_initialize(host_allocator, &out_scratch->arguments);
}

void loom_serve_text_chat_scratch_deinitialize(
    loom_serve_text_chat_scratch_t* scratch) {
  iree_string_builder_deinitialize(&scratch->arguments);
  iree_string_builder_deinitialize(&scratch->rendered);
  iree_string_builder_deinitialize(&scratch->parsed);
}

iree_status_t loom_serve_text_chat_complete(
    loom_serve_text_chat_t* chat, iree_string_view_t response, bool truncated,
    uint64_t request_id, loom_serve_text_tool_calls_t* tool_calls,
    loom_serve_text_chat_scratch_t* scratch, iree_allocator_t scratch_allocator,
    loom_serve_text_chat_completion_t* completion) {
  loom_serve_text_tool_calls_reset(tool_calls);
  loom_serve_text_chat_completion_t result = {0};
  iree_string_builder_reset(&scratch->parsed);
  iree_string_builder_reset(&scratch->rendered);
  iree_string_builder_reset(&scratch->arguments);
  loom_serve_text_chat_output_t output;
  do {
    IREE_RETURN_IF_ERROR(loom_serve_text_chat_output_next(
        chat, response, LOOM_SERVE_TEXT_CHAT_OUTPUT_COMPLETE, scratch_allocator,
        &output));
  } while (output.advanced);
  const iree_string_view_t inputs[] = {
      chat->tool_schemas, iree_string_view_substr(response, chat->output.cursor,
                                                  IREE_HOST_SIZE_MAX)};
  loom_serve_text_chat_completion_tools_t tools = {
      .scratch_allocator = scratch_allocator,
      .policy = chat->policy,
      .request_id = request_id,
      .calls = tool_calls,
      .scratch = scratch};
  const int64_t truncated_argument = truncated;
  iree_status_t status = loom_serve_text_chat_render(
      chat->policy, chat->policy->parse_tools, IREE_ARRAYSIZE(inputs), inputs,
      1, &truncated_argument, &scratch->parsed, scratch_allocator, NULL);
  if (iree_status_is_ok(status)) {
    status = iree_json_enumerate_array_typed(
        iree_string_builder_view(&scratch->parsed),
        loom_serve_text_chat_completion_tool, &tools);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_chat_checkpoint(
        chat, response, iree_string_builder_view(&scratch->rendered),
        chat->host_allocator, &result);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_chat_media_identity_initialize(
        chat, chat->host_allocator, &result.media);
  }
  if (iree_status_is_ok(status)) {
    loom_serve_text_chat_completion_deinitialize(completion);
    *completion = result;
  } else {
    loom_serve_text_tool_calls_reset(tool_calls);
    loom_serve_text_chat_completion_deinitialize(&result);
  }
  return status;
}

static iree_status_t loom_serve_text_chat_response_begin(
    const loom_serve_text_chat_response_t* response, const char* object,
    iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      "{\"id\":\"loom-%" PRIu64 "\",\"object\":\"%s\",\"created\":%" PRId64
      ",\"model\":",
      response->request_id, object, response->created_unix_seconds));
  return loom_serve_text_chat_quote(output, response->model_name);
}

static iree_status_t loom_serve_text_chat_response_usage(
    const loom_serve_text_chat_response_t* response,
    iree_string_builder_t* output) {
  return iree_string_builder_append_format(
      output,
      "{\"prompt_tokens\":%zu,\"completion_tokens\":%zu,\"total_tokens\":%zu,"
      "\"prompt_tokens_details\":{\"cached_tokens\":%zu}}",
      response->usage.input_tokens, response->usage.output_tokens,
      response->usage.input_tokens + response->usage.output_tokens,
      response->usage.cached_tokens);
}

iree_status_t loom_serve_text_chat_event(
    const loom_serve_text_chat_response_t* response, iree_string_view_t delta,
    iree_string_view_t finish_reason, iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "data: "));
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_response_begin(
      response, "chat.completion.chunk", output));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      output, ",\"choices\":[{\"index\":0,\"delta\":"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(output, delta));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"finish_reason\":"));
  if (finish_reason.size) {
    IREE_RETURN_IF_ERROR(loom_serve_text_chat_quote(output, finish_reason));
  } else {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "null"));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"logprobs\":null}]"));
  if (iree_any_bit_set(response->flags,
                       LOOM_SERVE_TEXT_CHAT_FLAG_INCLUDE_USAGE)) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(output, ",\"usage\":null"));
  }
  return iree_string_builder_append_cstring(output, "}\n\n");
}

iree_status_t loom_serve_text_chat_usage(
    const loom_serve_text_chat_response_t* response,
    iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "data: "));
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_response_begin(
      response, "chat.completion.chunk", output));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"choices\":[],\"usage\":"));
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_response_usage(response, output));
  return iree_string_builder_append_cstring(output, "}\n\n");
}

iree_status_t loom_serve_text_chat_response(
    const loom_serve_text_chat_response_t* response, iree_string_view_t content,
    iree_string_view_t reasoning,
    const loom_serve_text_tool_calls_t* tool_calls,
    iree_string_view_t finish_reason, iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(
      loom_serve_text_chat_response_begin(response, "chat.completion", output));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      output,
      ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
      "\"content\":"));
  if (tool_calls && tool_calls->count && !content.size) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "null"));
  } else {
    IREE_RETURN_IF_ERROR(loom_serve_text_chat_quote(output, content));
  }
  if (reasoning.size) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(output, ",\"reasoning_content\":"));
    IREE_RETURN_IF_ERROR(loom_serve_text_chat_quote(output, reasoning));
  }
  if (tool_calls && tool_calls->count) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(output, ",\"tool_calls\":"));
    IREE_RETURN_IF_ERROR(
        loom_serve_text_tool_calls_write_json(tool_calls, false, output));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, "},\"finish_reason\":"));
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_quote(output, finish_reason));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      output, ",\"logprobs\":null}],\"usage\":"));
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_response_usage(response, output));
  return iree_string_builder_append_cstring(output, "}");
}
