// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/drm/vcn_encode.h"

#include <string.h>

// VCN 4 encode tasks are flat little-endian DWORD packages. Length words count
// their complete package in bytes, including the length and operation words.
// The top-level unified-engine package contains one session package, one task
// package, and the operation-specific payload.
enum {
  VCN_ENCODE_INTERFACE_VERSION = 0x0001000f,
  VCN_ENCODE_ENGINE_IDENTIFIER = 0x30000001,
  VCN_ENCODE_ENGINE_TYPE = 2,
  VCN_ENCODE_SESSION_CONTEXT_BYTES = 131072,
  VCN_ENCODE_RECONSTRUCTION_LIMIT = 34,
  VCN_ENCODE_AV1_SDB_BYTES = 947200,
  VCN_ENCODE_AV1_CDF_BYTES = 22528,
  VCN_ENCODE_AV1_CDEF_BYTES = 1536,
};

typedef struct amdf_media_drm_vcn_encode_writer_t {
  // Borrowed host-visible command storage.
  uint32_t* words;
  // Number of complete words emitted so far.
  uint32_t count;
  // TASK_INFO payload word patched with the final task extent.
  uint32_t task_size_index;
  // First operation-specific payload word.
  uint32_t payload_index;
} amdf_media_drm_vcn_encode_writer_t;

static void amdf_media_drm_vcn_encode_word(
    amdf_media_drm_vcn_encode_writer_t* writer, uint32_t value) {
  writer->words[writer->count++] = value;
}

static void amdf_media_drm_vcn_encode_address(
    amdf_media_drm_vcn_encode_writer_t* writer, uint64_t address) {
  amdf_media_drm_vcn_encode_word(writer, (uint32_t)(address >> 32));
  amdf_media_drm_vcn_encode_word(writer, (uint32_t)address);
}

static uint32_t amdf_media_drm_vcn_encode_block_begin(
    amdf_media_drm_vcn_encode_writer_t* writer, uint32_t operation) {
  const uint32_t start = writer->count;
  amdf_media_drm_vcn_encode_word(writer, 0);
  amdf_media_drm_vcn_encode_word(writer, operation);
  return start;
}

static void amdf_media_drm_vcn_encode_block_end(
    amdf_media_drm_vcn_encode_writer_t* writer, uint32_t start) {
  writer->words[start] = (writer->count - start) * sizeof(uint32_t);
}

static uint32_t amdf_media_drm_vcn_encode_codec(amdf_media_codec_t codec) {
  return codec == AMDF_MEDIA_CODEC_H264  ? 1
         : codec == AMDF_MEDIA_CODEC_AV1 ? 2
                                         : 0;
}

static void amdf_media_drm_vcn_encode_task_initialize(
    amdf_media_drm_vcn_encode_writer_t* writer, uint32_t* words,
    const amdf_media_drm_session_t* session, uint32_t task_id, bool feedback) {
  *writer = (amdf_media_drm_vcn_encode_writer_t){.words = words, .count = 4};
  words[0] = 16;
  words[1] = VCN_ENCODE_ENGINE_IDENTIFIER;
  words[2] = VCN_ENCODE_ENGINE_TYPE;
  words[3] = 0;

  uint32_t block = amdf_media_drm_vcn_encode_block_begin(writer, 1);
  amdf_media_drm_vcn_encode_word(writer, VCN_ENCODE_INTERFACE_VERSION);
  amdf_media_drm_vcn_encode_address(writer, session->storage->address);
  amdf_media_drm_vcn_encode_word(writer, 0);
  amdf_media_drm_vcn_encode_block_end(writer, block);

  block = amdf_media_drm_vcn_encode_block_begin(writer, 2);
  writer->task_size_index = writer->count;
  amdf_media_drm_vcn_encode_word(writer, 0);
  amdf_media_drm_vcn_encode_word(writer, task_id);
  amdf_media_drm_vcn_encode_word(writer, feedback ? 1 : 0);
  amdf_media_drm_vcn_encode_block_end(writer, block);
  writer->payload_index = writer->count;
}

static uint32_t amdf_media_drm_vcn_encode_task_end(
    amdf_media_drm_vcn_encode_writer_t* writer) {
  writer->words[writer->task_size_index] =
      20 + (writer->count - writer->payload_index) * sizeof(uint32_t);
  writer->words[3] = writer->count * sizeof(uint32_t);
  return writer->words[3];
}

static void amdf_media_drm_vcn_encode_empty_block(
    amdf_media_drm_vcn_encode_writer_t* writer, uint32_t operation) {
  const uint32_t block =
      amdf_media_drm_vcn_encode_block_begin(writer, operation);
  amdf_media_drm_vcn_encode_block_end(writer, block);
}

static uint32_t amdf_media_drm_vcn_encode_rate_control_method(
    amdf_media_encode_rate_control_mode_t mode) {
  return mode == AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CBR   ? 3
         : mode == AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_VBR ? 2
                                                           : 0;
}

static uint32_t amdf_media_drm_vcn_encode_per_picture_integer(
    uint64_t rate, uint32_t numerator, uint32_t denominator) {
  return (uint32_t)(rate * denominator / numerator);
}

static uint32_t amdf_media_drm_vcn_encode_per_picture_fraction(
    uint64_t rate, uint32_t numerator, uint32_t denominator) {
  return (uint32_t)(((rate * denominator % numerator) << 32) / numerator);
}

static void amdf_media_drm_vcn_encode_rate_control(
    amdf_media_drm_vcn_encode_writer_t* writer,
    const amdf_media_drm_session_t* session) {
  const amdf_media_encode_rate_control_t* rate = &session->encode.rate_control;
  const uint32_t method =
      amdf_media_drm_vcn_encode_rate_control_method(rate->mode);
  const amdf_media_encode_bit_rate_t* bit_rate = &rate->control.bit_rate;
  // Disabled rate control still requires a complete layer record. Firmware
  // ignores its bitrate fields but the qualified native sequence uses one
  // deterministic nominal stream rather than leaving them undefined.
  uint64_t target = 1200000;
  uint64_t peak = 1200000;
  uint64_t virtual_buffer = 1200000;
  uint32_t virtual_buffer_level = 32;
  uint32_t intra_qp = rate->control.cqp.intra_quantization_parameter;
  uint32_t predicted_qp = rate->control.cqp.predicted_quantization_parameter;
  const bool av1 =
      session->requirements.configuration.codec == AMDF_MEDIA_CODEC_AV1;
  uint32_t minimum_qp = av1 ? 8 : 1;
  uint32_t maximum_qp = av1 ? 255 : 51;
  uint64_t maximum_frame_bits = 0;
  if (method != 0) {
    target = bit_rate->target_bit_rate;
    peak = bit_rate->peak_bit_rate;
    virtual_buffer = bit_rate->virtual_buffer_capacity;
    virtual_buffer_level =
        (uint32_t)((bit_rate->initial_virtual_buffer_fullness << 6) /
                   virtual_buffer);
    intra_qp = predicted_qp = 20;
    minimum_qp = bit_rate->minimum_quantization_parameter;
    maximum_qp = bit_rate->maximum_quantization_parameter;
    maximum_frame_bits = bit_rate->maximum_frame_bit_count;
  }

  uint32_t block = amdf_media_drm_vcn_encode_block_begin(writer, 4);
  amdf_media_drm_vcn_encode_word(writer, 1);
  amdf_media_drm_vcn_encode_word(writer, 1);
  amdf_media_drm_vcn_encode_block_end(writer, block);

  block = amdf_media_drm_vcn_encode_block_begin(writer, 6);
  amdf_media_drm_vcn_encode_word(writer, method);
  amdf_media_drm_vcn_encode_word(writer, virtual_buffer_level);
  amdf_media_drm_vcn_encode_block_end(writer, block);

  block = amdf_media_drm_vcn_encode_block_begin(writer, 0x18);
  amdf_media_drm_vcn_encode_word(writer, 0);
  amdf_media_drm_vcn_encode_block_end(writer, block);

  block = amdf_media_drm_vcn_encode_block_begin(writer, 5);
  amdf_media_drm_vcn_encode_word(writer, 0);
  amdf_media_drm_vcn_encode_block_end(writer, block);

  block = amdf_media_drm_vcn_encode_block_begin(writer, 7);
  amdf_media_drm_vcn_encode_word(writer, (uint32_t)target);
  amdf_media_drm_vcn_encode_word(writer, (uint32_t)peak);
  amdf_media_drm_vcn_encode_word(writer, rate->frame_rate_numerator);
  amdf_media_drm_vcn_encode_word(writer, rate->frame_rate_denominator);
  amdf_media_drm_vcn_encode_word(writer, (uint32_t)virtual_buffer);
  amdf_media_drm_vcn_encode_word(
      writer,
      amdf_media_drm_vcn_encode_per_picture_integer(
          target, rate->frame_rate_numerator, rate->frame_rate_denominator));
  amdf_media_drm_vcn_encode_word(
      writer,
      amdf_media_drm_vcn_encode_per_picture_integer(
          peak, rate->frame_rate_numerator, rate->frame_rate_denominator));
  amdf_media_drm_vcn_encode_word(
      writer,
      amdf_media_drm_vcn_encode_per_picture_fraction(
          peak, rate->frame_rate_numerator, rate->frame_rate_denominator));
  amdf_media_drm_vcn_encode_block_end(writer, block);

  block = amdf_media_drm_vcn_encode_block_begin(writer, 5);
  amdf_media_drm_vcn_encode_word(writer, 0);
  amdf_media_drm_vcn_encode_block_end(writer, block);

  block = amdf_media_drm_vcn_encode_block_begin(writer, 0x1d);
  amdf_media_drm_vcn_encode_word(writer, intra_qp);
  amdf_media_drm_vcn_encode_word(writer, predicted_qp);
  amdf_media_drm_vcn_encode_word(writer, predicted_qp);
  for (uint32_t i = 0; i < 3; ++i) {
    amdf_media_drm_vcn_encode_word(writer, minimum_qp);
    amdf_media_drm_vcn_encode_word(writer, maximum_qp);
  }
  for (uint32_t i = 0; i < 3; ++i) {
    amdf_media_drm_vcn_encode_word(writer, (uint32_t)maximum_frame_bits);
  }
  amdf_media_drm_vcn_encode_word(writer, 0);
  amdf_media_drm_vcn_encode_word(writer, 0);
  amdf_media_drm_vcn_encode_word(writer, method != 0);
  amdf_media_drm_vcn_encode_word(writer, 0);
  amdf_media_drm_vcn_encode_block_end(writer, block);

  amdf_media_drm_vcn_encode_empty_block(writer, 0x01000004);
  amdf_media_drm_vcn_encode_empty_block(writer, 0x01000005);
}

uint32_t amdf_media_drm_vcn_encode_control(
    amdf_media_drm_buffer_t* storage, const amdf_media_drm_session_t* session,
    amdf_media_command_operation_t operation) {
  amdf_media_drm_vcn_encode_writer_t writer;
  const bool begin = operation == AMDF_MEDIA_COMMAND_OPERATION_BEGIN;
  amdf_media_drm_vcn_encode_task_initialize(
      &writer, storage->host_pointer, session, begin ? 1 : UINT32_MAX, false);
  if (begin) {
    amdf_media_drm_vcn_encode_empty_block(&writer, 0x01000001);
    const amdf_media_configuration_t* configuration =
        &session->requirements.configuration;
    const bool h264 = configuration->codec == AMDF_MEDIA_CODEC_H264;
    const uint32_t width_alignment = h264 ? 16 : 64;
    const uint32_t aligned_width =
        (configuration->coded_width + width_alignment - 1) &
        ~(width_alignment - 1);
    const uint32_t aligned_height = (configuration->coded_height + 15) & ~15u;
    uint32_t block = amdf_media_drm_vcn_encode_block_begin(&writer, 3);
    amdf_media_drm_vcn_encode_word(
        &writer, amdf_media_drm_vcn_encode_codec(configuration->codec));
    amdf_media_drm_vcn_encode_word(&writer, aligned_width);
    amdf_media_drm_vcn_encode_word(&writer, aligned_height);
    amdf_media_drm_vcn_encode_word(&writer,
                                   aligned_width - configuration->coded_width);
    amdf_media_drm_vcn_encode_word(
        &writer, aligned_height - configuration->coded_height);
    for (uint32_t i = 0; i < 4; ++i) {
      amdf_media_drm_vcn_encode_word(&writer, 0);
    }
    const amdf_media_drm_device_t* device =
        (const amdf_media_drm_device_t*)session->base.queue->device;
    const uint32_t version =
        device->engines[session->base.queue->info.queue_family_ordinal]
            .information.ip_discovery_version;
    const bool av1_workaround =
        configuration->codec == AMDF_MEDIA_CODEC_AV1 &&
        (version == 0x040002 || version == 0x040005 || version == 0x040006);
    amdf_media_drm_vcn_encode_word(&writer, av1_workaround);
    amdf_media_drm_vcn_encode_word(&writer, 0);
    amdf_media_drm_vcn_encode_block_end(&writer, block);
    amdf_media_drm_vcn_encode_rate_control(&writer, session);
  } else {
    amdf_media_drm_vcn_encode_empty_block(&writer, 0x01000002);
  }
  return amdf_media_drm_vcn_encode_task_end(&writer);
}

typedef struct amdf_media_drm_vcn_encode_bits_t {
  // Big-endian codec bits packed into little-endian command DWORD storage.
  uint32_t words[16];
  // Number of meaningful bits across words.
  uint32_t count;
} amdf_media_drm_vcn_encode_bits_t;

static void amdf_media_drm_vcn_encode_bits_write(
    amdf_media_drm_vcn_encode_bits_t* bits, uint32_t value, uint32_t width) {
  for (uint32_t remaining = width; remaining != 0; --remaining) {
    if ((value & (1u << (remaining - 1))) != 0) {
      bits->words[bits->count / 32] |= 1u << (31 - bits->count % 32);
    }
    ++bits->count;
  }
}

static void amdf_media_drm_vcn_encode_bits_unsigned_exp_golomb(
    amdf_media_drm_vcn_encode_bits_t* bits, uint32_t value) {
  const uint32_t code = value + 1;
  const uint32_t width = 32 - (uint32_t)__builtin_clz(code);
  amdf_media_drm_vcn_encode_bits_write(bits, 0, width - 1);
  amdf_media_drm_vcn_encode_bits_write(bits, code, width);
}

static void amdf_media_drm_vcn_encode_bits_signed_exp_golomb(
    amdf_media_drm_vcn_encode_bits_t* bits, int32_t value) {
  const uint32_t magnitude = value < 0 ? 0u - (uint32_t)value : (uint32_t)value;
  const uint32_t code = magnitude * 2 - (value > 0 ? 1u : 0u);
  amdf_media_drm_vcn_encode_bits_unsigned_exp_golomb(bits, code);
}

static void amdf_media_drm_vcn_encode_slice_copy(
    uint32_t data[16], uint32_t* data_count, uint32_t instructions[32],
    uint32_t* instruction_count, const amdf_media_drm_vcn_encode_bits_t* bits) {
  const uint32_t word_count = (bits->count + 31) / 32;
  memcpy(data + *data_count, bits->words, word_count * sizeof(uint32_t));
  *data_count += word_count;
  instructions[(*instruction_count)++] = 1;
  instructions[(*instruction_count)++] = bits->count;
}

static void amdf_media_drm_vcn_encode_slice_instruction(
    uint32_t instructions[32], uint32_t* count, uint32_t operation) {
  instructions[(*count)++] = operation;
  instructions[(*count)++] = 0;
}

static void amdf_media_drm_vcn_encode_slice_block(
    amdf_media_drm_vcn_encode_writer_t* writer, const uint32_t data[16],
    const uint32_t instructions[32]) {
  const uint32_t block = amdf_media_drm_vcn_encode_block_begin(writer, 0x0b);
  for (uint32_t i = 0; i < 16; ++i) {
    amdf_media_drm_vcn_encode_word(writer, data[i]);
  }
  for (uint32_t i = 0; i < 32; ++i) {
    amdf_media_drm_vcn_encode_word(writer, instructions[i]);
  }
  amdf_media_drm_vcn_encode_block_end(writer, block);
}

static void amdf_media_drm_vcn_encode_h264_slice(
    amdf_media_drm_vcn_encode_writer_t* writer,
    const amdf_media_drm_session_t* session,
    const amdf_media_h264_encode_picture_info_t* picture) {
  uint32_t data[16] = {0};
  uint32_t instructions[32] = {0};
  uint32_t data_count = 0;
  uint32_t instruction_count = 0;
  amdf_media_drm_vcn_encode_bits_t bits = {0};
  const bool idr = picture->frame_type == AMDF_MEDIA_H264_ENCODE_FRAME_TYPE_IDR;
  amdf_media_drm_vcn_encode_bits_write(&bits, idr ? 0x65 : 0x41, 8);
  amdf_media_drm_vcn_encode_slice_copy(data, &data_count, instructions,
                                       &instruction_count, &bits);
  amdf_media_drm_vcn_encode_slice_instruction(instructions, &instruction_count,
                                              0x20000);

  bits = (amdf_media_drm_vcn_encode_bits_t){0};
  amdf_media_drm_vcn_encode_bits_unsigned_exp_golomb(&bits, idr ? 7 : 5);
  amdf_media_drm_vcn_encode_bits_unsigned_exp_golomb(&bits, 0);
  amdf_media_drm_vcn_encode_bits_write(
      &bits, picture->frame_number,
      session->encode_codec.h264.log2_max_frame_number_minus4 + 4);
  if (idr) {
    amdf_media_drm_vcn_encode_bits_unsigned_exp_golomb(&bits, 0);
    amdf_media_drm_vcn_encode_bits_write(&bits, 0, 2);
  } else {
    amdf_media_drm_vcn_encode_bits_write(&bits, 0, 1);
    amdf_media_drm_vcn_encode_bits_write(&bits, 0, 1);
    amdf_media_drm_vcn_encode_bits_write(&bits, 0, 1);
  }
  amdf_media_drm_vcn_encode_slice_copy(data, &data_count, instructions,
                                       &instruction_count, &bits);
  amdf_media_drm_vcn_encode_slice_instruction(instructions, &instruction_count,
                                              0x20001);

  bits = (amdf_media_drm_vcn_encode_bits_t){0};
  amdf_media_drm_vcn_encode_bits_unsigned_exp_golomb(&bits, 0);
  amdf_media_drm_vcn_encode_bits_signed_exp_golomb(&bits, 0);
  amdf_media_drm_vcn_encode_bits_signed_exp_golomb(&bits, 0);
  amdf_media_drm_vcn_encode_slice_copy(data, &data_count, instructions,
                                       &instruction_count, &bits);
  amdf_media_drm_vcn_encode_slice_instruction(instructions, &instruction_count,
                                              0);
  amdf_media_drm_vcn_encode_slice_block(writer, data, instructions);
}

static void amdf_media_drm_vcn_encode_hevc_slice(
    amdf_media_drm_vcn_encode_writer_t* writer,
    const amdf_media_drm_session_t* session,
    const amdf_media_hevc_encode_picture_info_t* picture) {
  uint32_t data[16] = {0};
  uint32_t instructions[32] = {0};
  uint32_t data_count = 0;
  uint32_t instruction_count = 0;
  amdf_media_drm_vcn_encode_bits_t bits = {0};
  const bool idr = picture->frame_type == AMDF_MEDIA_HEVC_ENCODE_FRAME_TYPE_IDR;
  amdf_media_drm_vcn_encode_bits_write(&bits, ((idr ? 19 : 1) << 9) | 1, 16);
  amdf_media_drm_vcn_encode_slice_copy(data, &data_count, instructions,
                                       &instruction_count, &bits);
  amdf_media_drm_vcn_encode_slice_instruction(instructions, &instruction_count,
                                              0x10001);

  bits = (amdf_media_drm_vcn_encode_bits_t){0};
  if (idr) {
    amdf_media_drm_vcn_encode_bits_write(&bits, 0, 1);
  }
  amdf_media_drm_vcn_encode_bits_unsigned_exp_golomb(&bits, 0);
  amdf_media_drm_vcn_encode_slice_copy(data, &data_count, instructions,
                                       &instruction_count, &bits);
  amdf_media_drm_vcn_encode_slice_instruction(instructions, &instruction_count,
                                              0x10002);
  amdf_media_drm_vcn_encode_slice_instruction(instructions, &instruction_count,
                                              0x10000);

  bits = (amdf_media_drm_vcn_encode_bits_t){0};
  amdf_media_drm_vcn_encode_bits_unsigned_exp_golomb(&bits, idr ? 2 : 1);
  if (!idr) {
    const uint32_t order_bits =
        session->encode_codec.hevc.log2_max_picture_order_count_lsb_minus4 + 4;
    const uint32_t order_mask = (1u << order_bits) - 1;
    amdf_media_drm_vcn_encode_bits_write(
        &bits, (uint32_t)picture->picture_order_count & order_mask, order_bits);
    amdf_media_drm_vcn_encode_bits_write(&bits, 0, 1);
    amdf_media_drm_vcn_encode_bits_unsigned_exp_golomb(&bits, 1);
    amdf_media_drm_vcn_encode_bits_unsigned_exp_golomb(&bits, 0);
    amdf_media_drm_vcn_encode_bits_unsigned_exp_golomb(&bits, 0);
    amdf_media_drm_vcn_encode_bits_write(&bits, 1, 1);
    amdf_media_drm_vcn_encode_bits_write(&bits, 0, 1);
    amdf_media_drm_vcn_encode_bits_write(&bits, 0, 1);
    amdf_media_drm_vcn_encode_bits_unsigned_exp_golomb(&bits, 0);
  }
  amdf_media_drm_vcn_encode_slice_copy(data, &data_count, instructions,
                                       &instruction_count, &bits);
  amdf_media_drm_vcn_encode_slice_instruction(instructions, &instruction_count,
                                              0x10003);

  bits = (amdf_media_drm_vcn_encode_bits_t){0};
  amdf_media_drm_vcn_encode_bits_signed_exp_golomb(&bits, 0);
  amdf_media_drm_vcn_encode_bits_signed_exp_golomb(&bits, 0);
  amdf_media_drm_vcn_encode_bits_write(&bits, 1, 1);
  amdf_media_drm_vcn_encode_slice_copy(data, &data_count, instructions,
                                       &instruction_count, &bits);
  amdf_media_drm_vcn_encode_slice_instruction(instructions, &instruction_count,
                                              0);
  amdf_media_drm_vcn_encode_slice_block(writer, data, instructions);
}

static void amdf_media_drm_vcn_encode_av1_instruction(
    amdf_media_drm_vcn_encode_writer_t* writer, uint32_t operation) {
  amdf_media_drm_vcn_encode_empty_block(writer, operation);
}

static void amdf_media_drm_vcn_encode_av1_copy(
    amdf_media_drm_vcn_encode_writer_t* writer,
    const amdf_media_drm_vcn_encode_bits_t* bits) {
  const uint32_t block = amdf_media_drm_vcn_encode_block_begin(writer, 1);
  amdf_media_drm_vcn_encode_word(writer, bits->count);
  const uint32_t word_count = (bits->count + 31) / 32;
  for (uint32_t i = 0; i < word_count; ++i) {
    amdf_media_drm_vcn_encode_word(writer, bits->words[i]);
  }
  amdf_media_drm_vcn_encode_block_end(writer, block);
}

static void amdf_media_drm_vcn_encode_av1_header(
    amdf_media_drm_vcn_encode_writer_t* writer,
    const amdf_media_drm_session_t* session,
    const amdf_media_av1_encode_picture_info_t* picture) {
  const uint32_t outer =
      amdf_media_drm_vcn_encode_block_begin(writer, 0x300002);
  uint32_t block = amdf_media_drm_vcn_encode_block_begin(writer, 2);
  amdf_media_drm_vcn_encode_word(writer, 1);
  amdf_media_drm_vcn_encode_block_end(writer, block);

  amdf_media_drm_vcn_encode_bits_t bits = {0};
  amdf_media_drm_vcn_encode_bits_write(&bits, 0x32, 8);
  amdf_media_drm_vcn_encode_av1_copy(writer, &bits);
  amdf_media_drm_vcn_encode_av1_instruction(writer, 3);

  const bool inter = picture->frame_type != 0;
  bits = (amdf_media_drm_vcn_encode_bits_t){0};
  amdf_media_drm_vcn_encode_bits_write(&bits, 0, 1);
  amdf_media_drm_vcn_encode_bits_write(&bits, picture->frame_type, 2);
  amdf_media_drm_vcn_encode_bits_write(&bits, 1, 1);
  if (inter) {
    amdf_media_drm_vcn_encode_bits_write(&bits, 0, 1);
  }
  amdf_media_drm_vcn_encode_bits_write(&bits, 0, 1);
  amdf_media_drm_vcn_encode_bits_write(&bits, 0, 1);
  amdf_media_drm_vcn_encode_bits_write(
      &bits, picture->order_hint,
      session->encode_codec.av1.order_hint_bit_count);
  if (inter) {
    amdf_media_drm_vcn_encode_bits_write(&bits, picture->primary_reference, 3);
    amdf_media_drm_vcn_encode_bits_write(&bits, picture->refresh_frame_flags,
                                         8);
    amdf_media_drm_vcn_encode_bits_write(&bits, 0, 1);
    for (uint32_t i = 0; i < 7; ++i) {
      amdf_media_drm_vcn_encode_bits_write(&bits, picture->reference_indices[i],
                                           3);
    }
  }
  amdf_media_drm_vcn_encode_bits_write(&bits, 0, 1);
  amdf_media_drm_vcn_encode_av1_copy(writer, &bits);

  if (inter) {
    amdf_media_drm_vcn_encode_av1_instruction(writer, 5);
    amdf_media_drm_vcn_encode_av1_instruction(writer, 7);
    bits = (amdf_media_drm_vcn_encode_bits_t){0};
    amdf_media_drm_vcn_encode_bits_write(&bits, 0, 1);
    amdf_media_drm_vcn_encode_bits_write(&bits, 0, 1);
  } else {
    bits = (amdf_media_drm_vcn_encode_bits_t){0};
    amdf_media_drm_vcn_encode_bits_write(&bits, 0, 1);
  }
  amdf_media_drm_vcn_encode_av1_copy(writer, &bits);
  amdf_media_drm_vcn_encode_av1_instruction(writer, 9);
  amdf_media_drm_vcn_encode_av1_instruction(writer, 10);
  bits = (amdf_media_drm_vcn_encode_bits_t){0};
  amdf_media_drm_vcn_encode_bits_write(&bits, 0, 1);
  amdf_media_drm_vcn_encode_av1_copy(writer, &bits);
  amdf_media_drm_vcn_encode_av1_instruction(writer, 11);
  amdf_media_drm_vcn_encode_av1_instruction(writer, 6);
  amdf_media_drm_vcn_encode_av1_instruction(writer, 8);
  amdf_media_drm_vcn_encode_av1_instruction(writer, 12);
  amdf_media_drm_vcn_encode_av1_instruction(writer, 13);
  bits = (amdf_media_drm_vcn_encode_bits_t){0};
  if (inter) {
    amdf_media_drm_vcn_encode_bits_write(&bits, 0, 1);
  }
  amdf_media_drm_vcn_encode_bits_write(&bits, 0, 1);
  if (inter) {
    for (uint32_t i = 0; i < 7; ++i) {
      amdf_media_drm_vcn_encode_bits_write(&bits, 0, 1);
    }
  }
  amdf_media_drm_vcn_encode_av1_copy(writer, &bits);
  amdf_media_drm_vcn_encode_av1_instruction(writer, 14);
  amdf_media_drm_vcn_encode_av1_instruction(writer, 4);
  amdf_media_drm_vcn_encode_av1_instruction(writer, 0);
  amdf_media_drm_vcn_encode_block_end(writer, outer);
}

static void amdf_media_drm_vcn_encode_layer_select(
    amdf_media_drm_vcn_encode_writer_t* writer) {
  const uint32_t block = amdf_media_drm_vcn_encode_block_begin(writer, 5);
  amdf_media_drm_vcn_encode_word(writer, 0);
  amdf_media_drm_vcn_encode_block_end(writer, block);
}

static void amdf_media_drm_vcn_encode_quality(
    amdf_media_drm_vcn_encode_writer_t* writer) {
  const uint32_t block = amdf_media_drm_vcn_encode_block_begin(writer, 9);
  for (uint32_t i = 0; i < 5; ++i) {
    amdf_media_drm_vcn_encode_word(writer, 0);
  }
  amdf_media_drm_vcn_encode_block_end(writer, block);
}

static void amdf_media_drm_vcn_encode_parameters(
    amdf_media_drm_vcn_encode_writer_t* writer,
    const amdf_media_drm_session_t* session,
    const amdf_media_drm_vcn_encode_resources_t* resources,
    uint32_t picture_type, uint32_t reference_slot,
    uint32_t reconstruction_slot) {
  const uint32_t sample_bytes =
      session->requirements.configuration.format == AMDF_MEDIA_PIXEL_FORMAT_P010
          ? 2
          : 1;
  const uint32_t block = amdf_media_drm_vcn_encode_block_begin(writer, 0x0f);
  amdf_media_drm_vcn_encode_word(writer, picture_type);
  amdf_media_drm_vcn_encode_word(writer, resources->compressed_byte_length);
  amdf_media_drm_vcn_encode_address(writer, resources->input.addresses[0]);
  amdf_media_drm_vcn_encode_address(writer, resources->input.addresses[1]);
  amdf_media_drm_vcn_encode_word(
      writer, resources->input.row_pitches[0] / sample_bytes);
  amdf_media_drm_vcn_encode_word(
      writer, resources->input.row_pitches[1] / (2 * sample_bytes));
  amdf_media_drm_vcn_encode_word(writer, 0);
  amdf_media_drm_vcn_encode_word(writer, reference_slot);
  amdf_media_drm_vcn_encode_word(writer, reconstruction_slot);
  amdf_media_drm_vcn_encode_block_end(writer, block);
}

static void amdf_media_drm_vcn_encode_context(
    amdf_media_drm_vcn_encode_writer_t* writer,
    const amdf_media_drm_session_t* session,
    const amdf_media_drm_vcn_encode_resources_t* resources) {
  const uint32_t sample_bytes =
      session->requirements.configuration.format == AMDF_MEDIA_PIXEL_FORMAT_P010
          ? 2
          : 1;
  const bool av1 =
      session->requirements.configuration.codec == AMDF_MEDIA_CODEC_AV1;
  const uint32_t pixel_byte_length =
      resources->reconstruction_layer_byte_length *
      resources->reconstruction_slot_count;
  const uint32_t codec_state_offset =
      pixel_byte_length + (av1 ? VCN_ENCODE_AV1_SDB_BYTES : 0);
  const uint32_t block = amdf_media_drm_vcn_encode_block_begin(writer, 0x11);
  amdf_media_drm_vcn_encode_address(writer, resources->reconstruction_address);
  amdf_media_drm_vcn_encode_word(writer, sample_bytes == 1 ? 2 : 0x10000000);
  amdf_media_drm_vcn_encode_word(
      writer, resources->reconstruction_row_pitch / sample_bytes);
  amdf_media_drm_vcn_encode_word(
      writer, resources->reconstruction_row_pitch / sample_bytes);
  amdf_media_drm_vcn_encode_word(writer, resources->reconstruction_slot_count);
  for (uint32_t i = 0; i < VCN_ENCODE_RECONSTRUCTION_LIMIT; ++i) {
    if (i < resources->reconstruction_slot_count) {
      const uint32_t layer = i * resources->reconstruction_layer_byte_length;
      amdf_media_drm_vcn_encode_word(writer, layer);
      amdf_media_drm_vcn_encode_word(
          writer, layer + resources->reconstruction_luma_byte_length);
      if (av1) {
        const uint32_t cdf =
            codec_state_offset +
            i * (VCN_ENCODE_AV1_CDF_BYTES + VCN_ENCODE_AV1_CDEF_BYTES);
        amdf_media_drm_vcn_encode_word(writer, cdf);
        amdf_media_drm_vcn_encode_word(writer, cdf + VCN_ENCODE_AV1_CDF_BYTES);
      } else {
        amdf_media_drm_vcn_encode_word(writer, 0);
        amdf_media_drm_vcn_encode_word(writer, 0);
      }
    } else {
      for (uint32_t j = 0; j < 4; ++j) {
        amdf_media_drm_vcn_encode_word(writer, 0);
      }
    }
  }
  amdf_media_drm_vcn_encode_word(
      writer, resources->reconstruction_row_pitch / sample_bytes);
  amdf_media_drm_vcn_encode_word(
      writer, resources->reconstruction_row_pitch / sample_bytes);
  for (uint32_t i = 0; i < VCN_ENCODE_RECONSTRUCTION_LIMIT * 4; ++i) {
    amdf_media_drm_vcn_encode_word(writer, 0);
  }
  for (uint32_t i = 0; i < 4; ++i) {
    amdf_media_drm_vcn_encode_word(writer, 0);
  }
  amdf_media_drm_vcn_encode_word(writer, av1 ? pixel_byte_length : 0);
  amdf_media_drm_vcn_encode_block_end(writer, block);
}

static void amdf_media_drm_vcn_encode_output(
    amdf_media_drm_vcn_encode_writer_t* writer,
    const amdf_media_drm_session_t* session,
    const amdf_media_drm_vcn_encode_resources_t* resources) {
  uint32_t block = amdf_media_drm_vcn_encode_block_begin(writer, 0x12);
  amdf_media_drm_vcn_encode_word(writer, 0);
  amdf_media_drm_vcn_encode_address(writer, resources->compressed_address);
  amdf_media_drm_vcn_encode_word(writer, resources->compressed_byte_length);
  amdf_media_drm_vcn_encode_word(writer, 0);
  amdf_media_drm_vcn_encode_block_end(writer, block);

  block = amdf_media_drm_vcn_encode_block_begin(writer, 0x15);
  amdf_media_drm_vcn_encode_word(writer, 0);
  amdf_media_drm_vcn_encode_address(writer, resources->feedback_address);
  amdf_media_drm_vcn_encode_word(writer, 16);
  amdf_media_drm_vcn_encode_word(writer,
                                 AMDF_MEDIA_DRM_VCN_ENCODE_FEEDBACK_BYTES);
  amdf_media_drm_vcn_encode_block_end(writer, block);

  block = amdf_media_drm_vcn_encode_block_begin(writer, 0x10);
  for (uint32_t i = 0; i < 3; ++i) {
    amdf_media_drm_vcn_encode_word(writer, 0);
  }
  amdf_media_drm_vcn_encode_block_end(writer, block);

  block = amdf_media_drm_vcn_encode_block_begin(writer, 0x14);
  for (uint32_t i = 0; i < 4; ++i) {
    amdf_media_drm_vcn_encode_word(writer, 0);
  }
  amdf_media_drm_vcn_encode_block_end(writer, block);

  const uint32_t ten_bit = session->requirements.configuration.format ==
                           AMDF_MEDIA_PIXEL_FORMAT_P010;
  block = amdf_media_drm_vcn_encode_block_begin(writer, 0x0c);
  for (uint32_t i = 0; i < 5; ++i) {
    amdf_media_drm_vcn_encode_word(writer, 0);
  }
  amdf_media_drm_vcn_encode_word(writer, ten_bit);
  amdf_media_drm_vcn_encode_word(writer, ten_bit);
  amdf_media_drm_vcn_encode_block_end(writer, block);

  block = amdf_media_drm_vcn_encode_block_begin(writer, 0x0d);
  for (uint32_t i = 0; i < 3; ++i) {
    amdf_media_drm_vcn_encode_word(writer, 0);
  }
  amdf_media_drm_vcn_encode_word(writer, ten_bit);
  amdf_media_drm_vcn_encode_block_end(writer, block);
  amdf_media_drm_vcn_encode_empty_block(writer, 0x01000006);
  amdf_media_drm_vcn_encode_empty_block(writer, 0x01000003);
}

static void amdf_media_drm_vcn_encode_picture_begin(
    amdf_media_drm_vcn_encode_writer_t* writer,
    amdf_media_drm_buffer_t* storage, const amdf_media_drm_session_t* session,
    uint32_t task_id) {
  amdf_media_drm_vcn_encode_task_initialize(writer, storage->host_pointer,
                                            session, task_id, true);
  amdf_media_drm_vcn_encode_layer_select(writer);
  amdf_media_drm_vcn_encode_quality(writer);
}

static void amdf_media_drm_vcn_encode_picture_end(
    amdf_media_drm_vcn_encode_writer_t* writer,
    const amdf_media_drm_session_t* session,
    const amdf_media_drm_vcn_encode_resources_t* resources) {
  amdf_media_drm_vcn_encode_context(writer, session, resources);
  amdf_media_drm_vcn_encode_output(writer, session, resources);
}

uint32_t amdf_media_drm_vcn_encode_h264(
    amdf_media_drm_buffer_t* storage, const amdf_media_drm_session_t* session,
    const amdf_media_drm_vcn_encode_resources_t* resources,
    uint32_t reconstruction_slot,
    const amdf_media_h264_encode_picture_info_t* picture) {
  amdf_media_drm_vcn_encode_writer_t writer;
  amdf_media_drm_vcn_encode_picture_begin(
      &writer, storage, session, picture->frame_decoding_order_number + 2);
  uint32_t block = amdf_media_drm_vcn_encode_block_begin(&writer, 0x200001);
  amdf_media_drm_vcn_encode_word(&writer, 0);
  amdf_media_drm_vcn_encode_word(
      &writer,
      ((session->requirements.configuration.coded_width + 15) / 16) *
          ((session->requirements.configuration.coded_height + 15) / 16));
  amdf_media_drm_vcn_encode_block_end(&writer, block);

  block = amdf_media_drm_vcn_encode_block_begin(&writer, 0x200002);
  amdf_media_drm_vcn_encode_word(&writer, 0);
  amdf_media_drm_vcn_encode_word(&writer, 0);
  amdf_media_drm_vcn_encode_word(&writer, 0);
  amdf_media_drm_vcn_encode_word(&writer, 1);
  amdf_media_drm_vcn_encode_word(&writer, 1);
  amdf_media_drm_vcn_encode_word(&writer, 77);
  amdf_media_drm_vcn_encode_word(&writer, session->codec.h264.level_idc);
  amdf_media_drm_vcn_encode_word(&writer, 0);
  amdf_media_drm_vcn_encode_word(&writer, 0);
  amdf_media_drm_vcn_encode_block_end(&writer, block);

  block = amdf_media_drm_vcn_encode_block_begin(&writer, 0x200004);
  for (uint32_t i = 0; i < 5; ++i) {
    amdf_media_drm_vcn_encode_word(&writer, 0);
  }
  amdf_media_drm_vcn_encode_block_end(&writer, block);
  amdf_media_drm_vcn_encode_h264_slice(&writer, session, picture);

  const bool idr = picture->frame_type == AMDF_MEDIA_H264_ENCODE_FRAME_TYPE_IDR;
  const amdf_media_h264_encode_reference_t* reference =
      idr ? NULL : &picture->references[0];
  const uint32_t reference_slot =
      reference != NULL ? reference->binding.slot_index : UINT32_MAX;
  amdf_media_drm_vcn_encode_parameters(&writer, session, resources, idr ? 2 : 1,
                                       reference_slot, reconstruction_slot);

  block = amdf_media_drm_vcn_encode_block_begin(&writer, 0x200003);
  amdf_media_drm_vcn_encode_word(&writer, 0);
  amdf_media_drm_vcn_encode_word(&writer,
                                 (uint32_t)picture->picture_order_count);
  amdf_media_drm_vcn_encode_word(&writer, 0);
  amdf_media_drm_vcn_encode_word(
      &writer, reference != NULL && reference->frame_type ==
                                        AMDF_MEDIA_H264_ENCODE_FRAME_TYPE_IDR
                   ? 2
                   : 1);
  amdf_media_drm_vcn_encode_word(&writer, 0);
  amdf_media_drm_vcn_encode_word(&writer, 0);
  amdf_media_drm_vcn_encode_word(
      &writer,
      reference != NULL ? (uint32_t)reference->picture_order_count : 0);
  amdf_media_drm_vcn_encode_word(&writer, UINT32_MAX);
  for (uint32_t i = 0; i < 4; ++i) {
    amdf_media_drm_vcn_encode_word(&writer, 0);
  }
  amdf_media_drm_vcn_encode_word(&writer, UINT32_MAX);
  for (uint32_t i = 0; i < 4; ++i) {
    amdf_media_drm_vcn_encode_word(&writer, 0);
  }
  amdf_media_drm_vcn_encode_word(&writer, picture->used_as_reference);
  amdf_media_drm_vcn_encode_block_end(&writer, block);

  amdf_media_drm_vcn_encode_picture_end(&writer, session, resources);
  return amdf_media_drm_vcn_encode_task_end(&writer);
}

uint32_t amdf_media_drm_vcn_encode_hevc(
    amdf_media_drm_buffer_t* storage, const amdf_media_drm_session_t* session,
    const amdf_media_drm_vcn_encode_resources_t* resources,
    uint32_t reconstruction_slot,
    const amdf_media_hevc_encode_picture_info_t* picture) {
  amdf_media_drm_vcn_encode_writer_t writer;
  amdf_media_drm_vcn_encode_picture_begin(
      &writer, storage, session, (uint32_t)picture->picture_order_count + 2);
  const uint32_t block_count =
      ((session->requirements.configuration.coded_width + 63) / 64) *
      ((session->requirements.configuration.coded_height + 63) / 64);
  uint32_t block = amdf_media_drm_vcn_encode_block_begin(&writer, 0x100001);
  amdf_media_drm_vcn_encode_word(&writer, 0);
  amdf_media_drm_vcn_encode_word(&writer, block_count);
  amdf_media_drm_vcn_encode_word(&writer, block_count);
  amdf_media_drm_vcn_encode_block_end(&writer, block);

  block = amdf_media_drm_vcn_encode_block_begin(&writer, 0x100002);
  amdf_media_drm_vcn_encode_word(
      &writer, session->encode_codec.hevc.minimum_luma_coding_unit_log2 - 3);
  amdf_media_drm_vcn_encode_word(&writer, 1);
  for (uint32_t i = 0; i < 3; ++i) {
    amdf_media_drm_vcn_encode_word(&writer, 0);
  }
  for (uint32_t i = 0; i < 4; ++i) {
    amdf_media_drm_vcn_encode_word(&writer, 1);
  }
  amdf_media_drm_vcn_encode_block_end(&writer, block);

  block = amdf_media_drm_vcn_encode_block_begin(&writer, 0x100003);
  amdf_media_drm_vcn_encode_word(&writer, 1);
  for (uint32_t i = 0; i < 5; ++i) {
    amdf_media_drm_vcn_encode_word(&writer, 0);
  }
  amdf_media_drm_vcn_encode_word(&writer, 1);
  amdf_media_drm_vcn_encode_block_end(&writer, block);
  amdf_media_drm_vcn_encode_hevc_slice(&writer, session, picture);

  const bool idr = picture->frame_type == AMDF_MEDIA_HEVC_ENCODE_FRAME_TYPE_IDR;
  amdf_media_drm_vcn_encode_parameters(
      &writer, session, resources, idr ? 2 : 1,
      idr ? UINT32_MAX : picture->references[0].binding.slot_index,
      reconstruction_slot);
  amdf_media_drm_vcn_encode_picture_end(&writer, session, resources);
  return amdf_media_drm_vcn_encode_task_end(&writer);
}

uint32_t amdf_media_drm_vcn_encode_av1(
    amdf_media_drm_buffer_t* storage, const amdf_media_drm_session_t* session,
    const amdf_media_drm_vcn_encode_resources_t* resources,
    uint32_t reconstruction_slot,
    const amdf_media_av1_encode_picture_info_t* picture) {
  amdf_media_drm_vcn_encode_writer_t writer;
  amdf_media_drm_vcn_encode_picture_begin(&writer, storage, session,
                                          picture->picture_index + 2);
  uint32_t block = amdf_media_drm_vcn_encode_block_begin(&writer, 0x300001);
  amdf_media_drm_vcn_encode_word(&writer, 0);
  amdf_media_drm_vcn_encode_word(&writer, 0x10);
  amdf_media_drm_vcn_encode_word(&writer, 1);
  amdf_media_drm_vcn_encode_word(&writer, 0);
  amdf_media_drm_vcn_encode_word(&writer, 0);
  amdf_media_drm_vcn_encode_word(&writer, 1);
  amdf_media_drm_vcn_encode_word(&writer, 0);
  amdf_media_drm_vcn_encode_word(&writer, 0);
  amdf_media_drm_vcn_encode_word(&writer, UINT32_MAX);
  amdf_media_drm_vcn_encode_word(&writer, UINT32_MAX);
  amdf_media_drm_vcn_encode_block_end(&writer, block);
  amdf_media_drm_vcn_encode_av1_header(&writer, session, picture);

  const bool key = picture->frame_type == 0;
  amdf_media_drm_vcn_encode_parameters(
      &writer, session, resources, key ? 2 : 1,
      key ? UINT32_MAX : picture->references[0].binding.slot_index,
      reconstruction_slot);

  block = amdf_media_drm_vcn_encode_block_begin(&writer, 0x19);
  amdf_media_drm_vcn_encode_word(&writer,
                                 key || picture->primary_reference == 7);
  const uint64_t cdf_address =
      session->storage->address + VCN_ENCODE_SESSION_CONTEXT_BYTES;
  // This one firmware package uniquely stores its address low half first.
  amdf_media_drm_vcn_encode_word(&writer, (uint32_t)cdf_address);
  amdf_media_drm_vcn_encode_word(&writer, (uint32_t)(cdf_address >> 32));
  amdf_media_drm_vcn_encode_block_end(&writer, block);

  amdf_media_drm_vcn_encode_picture_end(&writer, session, resources);
  return amdf_media_drm_vcn_encode_task_end(&writer);
}
