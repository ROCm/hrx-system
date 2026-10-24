// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/media/audio_source.h"

#include <math.h>

// MIME quoted pairs are unescaped while comparing or accumulating digits.
static bool loom_serve_audio_value(iree_string_view_t value,
                                   iree_string_view_t expected,
                                   uint32_t* out_number) {
  const bool quoted = iree_string_view_consume_prefix(&value, IREE_SV("\""));
  if (quoted && !iree_string_view_consume_suffix(&value, IREE_SV("\""))) {
    return false;
  }
  uint32_t number = 0;
  iree_host_size_t position = 0;
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    if (quoted && value.data[i] == '\\' && ++i == value.size) {
      return false;
    }
    const uint8_t character = (uint8_t)value.data[i];
    if (out_number) {
      const uint32_t digit = character - (uint32_t)'0';
      if (digit > 9 || number > (UINT32_MAX - digit) / 10) {
        return false;
      }
      number = number * 10 + digit;
    } else if (position >= expected.size ||
               character != (uint8_t)expected.data[position]) {
      return false;
    }
    ++position;
  }
  if (out_number) {
    *out_number = number;
    return number != 0;
  }
  return position == expected.size;
}

iree_status_t loom_serve_audio_source_parse(
    iree_string_view_t content_type, iree_const_byte_span_t data,
    loom_serve_audio_source_t* out_source) {
  *out_source = (loom_serve_audio_source_t){0};
  if (!data.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "audio input must not be empty");
  }
  loom_serve_audio_source_t source = {.data = data};
  iree_string_view_t media_type;
  iree_string_view_t parameters;
  iree_string_view_split(content_type, ';', &media_type, &parameters);
  if (!iree_string_view_equal_case(iree_string_view_trim(media_type),
                                   IREE_SV("audio/pcm"))) {
    *out_source = source;
    return iree_ok_status();
  }
  source.encoding = LOOM_SERVE_AUDIO_ENCODING_S16LE;
  bool has_format = false;
  while (parameters.size) {
    iree_string_view_t parameter;
    iree_string_view_split(parameters, ';', &parameter, &parameters);
    iree_string_view_t name;
    iree_string_view_t value;
    iree_string_view_split(parameter, '=', &name, &value);
    name = iree_string_view_trim(name);
    value = iree_string_view_trim(value);
    if (iree_string_view_equal_case(name, IREE_SV("format"))) {
      if (has_format) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "audio/pcm format must occur once");
      }
      has_format = true;
      if (loom_serve_audio_value(value, IREE_SV("f32le"), NULL)) {
        source.encoding = LOOM_SERVE_AUDIO_ENCODING_F32LE;
      } else if (!loom_serve_audio_value(value, IREE_SV("s16le"), NULL)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "audio/pcm format must be s16le or f32le");
      }
      continue;
    }
    uint32_t* target = NULL;
    if (iree_string_view_equal_case(name, IREE_SV("rate"))) {
      target = &source.sample_rate;
    } else if (iree_string_view_equal_case(name, IREE_SV("channels"))) {
      target = &source.channels;
    }
    if (!target || *target ||
        !loom_serve_audio_value(value, iree_string_view_empty(), target)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "audio/pcm requires unique positive decimal rate and channels");
    }
  }
  if (!source.sample_rate || !source.channels) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "audio/pcm requires positive rate and channels");
  }
  const uint64_t frame_bytes =
      (uint64_t)source.channels *
      (source.encoding == LOOM_SERVE_AUDIO_ENCODING_F32LE ? sizeof(float)
                                                          : sizeof(int16_t));
  if (data.data_length % frame_bytes) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "audio/pcm requires complete interleaved frames");
  }
  source.frame_count = data.data_length / frame_bytes;
  *out_source = source;
  return iree_ok_status();
}

iree_status_t loom_serve_audio_frame_count(uint64_t input_frames,
                                           uint32_t input_rate,
                                           uint32_t output_rate,
                                           uint64_t* out_frame_count) {
  *out_frame_count = 0;
  const uint64_t remainder =
      ((input_frames % input_rate) * output_rate + input_rate - 1) / input_rate;
  uint64_t frame_count = 0;
  if (!iree_checked_mul_u64(input_frames / input_rate, output_rate,
                            &frame_count) ||
      !iree_checked_add_u64(frame_count, remainder, &frame_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PCM duration exceeds the normalized sample clock");
  }
  *out_frame_count = frame_count;
  return iree_ok_status();
}

iree_status_t loom_serve_audio_source_read(
    const loom_serve_audio_source_t* source, uint64_t frame_offset,
    iree_host_size_t frame_count, float* output) {
  const iree_host_size_t sample_count = frame_count * source->channels;
  if (source->encoding == LOOM_SERVE_AUDIO_ENCODING_S16LE) {
    const uint8_t* input =
        source->data.data + frame_offset * source->channels * sizeof(int16_t);
    for (iree_host_size_t i = 0; i < sample_count; ++i) {
      output[i] = (float)(int16_t)iree_unaligned_load_le_u16(
                      input + i * sizeof(int16_t)) /
                  32768.0f;
    }
    return iree_ok_status();
  }
  const uint8_t* input =
      source->data.data + frame_offset * source->channels * sizeof(float);
  for (iree_host_size_t i = 0; i < sample_count; ++i) {
    const float value = iree_unaligned_load_le_f32(input + i * sizeof(float));
    if (!isfinite(value) || value < -1.0f || value > 1.0f) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "F32 audio must contain finite samples in [-1, 1]");
    }
    output[i] = value;
  }
  return iree_ok_status();
}
