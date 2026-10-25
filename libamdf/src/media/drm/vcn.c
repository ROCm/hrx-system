// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/drm/vcn.h"

#include <string.h>

#include "libamdf/src/media/drm/av1_grain.h"
#include "libamdf/src/media/hevc.h"

// The firmware ABI consists of little-endian DWORD messages. These offsets
// are the VCN decode protocol, independent of host C layout and pointer width.
// Every reserved word remains zero. See ac_vcn_dec.h in the native reference.
enum {
  MESSAGE_HEADER_WORDS = 6,
  MESSAGE_INDEX_WORDS = 4,
  DECODE_WORDS = 47,
  DYNAMIC_WORDS = 42,
  AVC_BYTES = 1124,
  HEVC_BYTES = 296,
  VP9_BYTES = 120,
  ENGINE_PACKAGE_BYTES = 16,
  DECODE_PACKAGE_BYTES = 140,
  INDIRECT_BUFFER_BYTES = ENGINE_PACKAGE_BYTES + DECODE_PACKAGE_BYTES,
  SESSION_CONTEXT_BYTES = 131072,
  STREAM_H264 = 7,
  STREAM_HEVC = 16,
  STREAM_VP9 = 17,
  STREAM_AV1 = 19,
};

// DWORD offsets within the unified DECODE_BUFFER package. Each address pair
// stores its high half first; the dynamic-reference message uses low first.
enum {
  BINDING_FLAGS = 0,
  BINDING_MESSAGE = 1,
  BINDING_OUTPUT = 5,
  BINDING_SESSION = 7,
  BINDING_COMPRESSED = 9,
  BINDING_CONTEXT = 11,
  BINDING_PROBABILITIES = 17,
  BINDING_SCALING = 21,
};

// DWORD offsets in the generic decode and tier-2 reference messages. Reserved
// entries, tiled-surface controls, and unused secondary targets stay zero.
enum {
  DECODE_STREAM = 0,
  DECODE_FLAGS = 1,
  DECODE_WIDTH = 2,
  DECODE_HEIGHT = 3,
  DECODE_COMPRESSED_BYTES = 4,
  DECODE_OUTPUT_BYTES = 6,
  DECODE_HARDWARE_CONTEXT_BYTES = 9,
  DECODE_SESSION_CONTEXT_BYTES = 10,
  DECODE_BUFFER_FLAGS = 17,
  DECODE_REFERENCE_PITCH = 18,
  DECODE_REFERENCE_ROWS = 19,
  DECODE_REFERENCE_SWIZZLE = 21,
  DECODE_REFERENCE_ARRAY_MODE = 22,
  DECODE_OUTPUT_PITCH = 25,
  DECODE_OUTPUT_CHROMA_PITCH = 26,
  DECODE_OUTPUT_ARRAY_MODE = 29,
  DECODE_OUTPUT_CHROMA_OFFSET = 36,
  DECODE_WRITE_COHERENCY = 40,
  DECODE_REFERENCE_CHROMA_PITCH = 41,
};
enum {
  DYNAMIC_LUMA_PITCH = 1,
  DYNAMIC_LUMA_ROWS = 2,
  DYNAMIC_LUMA_BYTES = 3,
  DYNAMIC_CHROMA_PITCH = 4,
  DYNAMIC_CHROMA_ROWS = 5,
  DYNAMIC_CHROMA_BYTES = 6,
  DYNAMIC_REFERENCE_COUNT = 7,
  DYNAMIC_CURRENT_LOW = 8,
  DYNAMIC_CURRENT_HIGH = 9,
  DYNAMIC_REFERENCE_LOW = 10,
  DYNAMIC_REFERENCE_HIGH = 26,
};

// AVC firmware fields mix DWORD and byte arrays. Keeping their units explicit
// avoids host-ABI padding and leaves unsupported MVC/slice-group fields zero.
enum {
  AVC_WORD_PROFILE = 0,
  AVC_WORD_LEVEL = 1,
  AVC_WORD_SEQUENCE_FLAGS = 2,
  AVC_WORD_PICTURE_FLAGS = 3,
  AVC_WORD_FRAME_NUMBER = 65,
  AVC_WORD_REFERENCE_FRAME_NUMBERS = 66,
  AVC_WORD_CURRENT_TOP_ORDER = 82,
  AVC_WORD_CURRENT_BOTTOM_ORDER = 83,
  AVC_WORD_REFERENCE_ORDERS = 84,
  AVC_WORD_CURRENT_SLOT = 116,
  AVC_WORD_REFERENCE_COUNT = 117,
  AVC_WORD_USED_FOR_REFERENCE = 280,
};
enum {
  AVC_BYTE_CHROMA_FORMAT = 16,
  AVC_BYTE_FRAME_NUMBER_BITS = 19,
  AVC_BYTE_ORDER_TYPE = 20,
  AVC_BYTE_ORDER_BITS = 21,
  AVC_BYTE_MAXIMUM_REFERENCES = 22,
  AVC_BYTE_INITIAL_QP = 24,
  AVC_BYTE_INITIAL_QS = 25,
  AVC_BYTE_CHROMA_QP = 26,
  AVC_BYTE_SECOND_CHROMA_QP = 27,
  AVC_BYTE_DEFAULT_L0_COUNT = 30,
  AVC_BYTE_DEFAULT_L1_COUNT = 31,
  AVC_BYTE_REFERENCE_SLOTS = 472,
};

// HEVC byte fields and DWORD fields within its 296-byte firmware message.
enum {
  HEVC_WORD_SEQUENCE_FLAGS = 0,
  HEVC_WORD_PICTURE_FLAGS = 1,
  HEVC_WORD_CURRENT_ORDER = 30,
  HEVC_WORD_REFERENCE_SET_BITS = 69,
};
enum {
  HEVC_BYTE_CHROMA_FORMAT = 8,
  HEVC_BYTE_LUMA_DEPTH = 9,
  HEVC_BYTE_CHROMA_DEPTH = 10,
  HEVC_BYTE_ORDER_BITS = 11,
  HEVC_BYTE_BUFFER_COUNT = 12,
  HEVC_BYTE_MINIMUM_BLOCK = 13,
  HEVC_BYTE_BLOCK_RANGE = 14,
  HEVC_BYTE_MINIMUM_TRANSFORM = 15,
  HEVC_BYTE_TRANSFORM_RANGE = 16,
  HEVC_BYTE_INTER_DEPTH = 17,
  HEVC_BYTE_INTRA_DEPTH = 18,
  HEVC_BYTE_SHORT_TERM_SETS = 24,
  HEVC_BYTE_LONG_TERM_COUNT = 25,
  HEVC_BYTE_EXTRA_HEADER_BITS = 23,
  HEVC_BYTE_DEFAULT_L0_COUNT = 26,
  HEVC_BYTE_DEFAULT_L1_COUNT = 27,
  HEVC_BYTE_CB_QP = 28,
  HEVC_BYTE_CR_QP = 29,
  HEVC_BYTE_BETA_OFFSET = 30,
  HEVC_BYTE_TC_OFFSET = 31,
  HEVC_BYTE_QP_DELTA_DEPTH = 32,
  HEVC_BYTE_TILE_COLUMNS = 33,
  HEVC_BYTE_TILE_ROWS = 34,
  HEVC_BYTE_MERGE_LEVEL = 35,
  HEVC_BYTE_INITIAL_QP = 116,
  HEVC_BYTE_PREDICTED_DELTA_COUNT = 117,
  HEVC_BYTE_CURRENT_SLOT = 118,
  HEVC_BYTE_HIGHEST_TEMPORAL_ID = 236,
  HEVC_BYTE_P010 = 238,
  HEVC_BYTE_MSB = 239,
  HEVC_BYTE_PCM_LUMA_DEPTH = 19,
  HEVC_BYTE_PCM_CHROMA_DEPTH = 20,
  HEVC_BYTE_PCM_MINIMUM_BLOCK = 21,
  HEVC_BYTE_PCM_BLOCK_RANGE = 22,
  HEVC_BYTE_TILE_WIDTHS = 36,
  HEVC_BYTE_TILE_HEIGHTS = 74,
  HEVC_BYTE_REFERENCE_SLOTS = 124,
  HEVC_BYTE_REFERENCE_ORDERS = 140,
  HEVC_BYTE_SHORT_TERM_BEFORE = 204,
  HEVC_BYTE_SHORT_TERM_AFTER = 212,
  HEVC_BYTE_LONG_TERM = 220,
  HEVC_BYTE_DC_16X16 = 228,
  HEVC_BYTE_DC_32X32 = 234,
};

// VP9 byte offsets; the first DWORD holds header flags and the final DWORDs
// locate its compressed and uncompressed header partitions.
enum {
  VP9_BYTE_CONTEXT = 4,
  VP9_BYTE_RESET_CONTEXT = 5,
  VP9_BYTE_CURRENT_SLOT = 6,
  VP9_BYTE_INTERPOLATION = 7,
  VP9_BYTE_FILTER_LEVEL = 8,
  VP9_BYTE_SHARPNESS = 9,
  VP9_BYTE_ADJUSTED_LEVELS = 10,
  VP9_BYTE_BASE_QUANTIZER = 74,
  VP9_BYTE_Y_DC_DELTA = 75,
  VP9_BYTE_UV_AC_DELTA = 76,
  VP9_BYTE_UV_DC_DELTA = 77,
  VP9_BYTE_TILE_COLUMNS = 78,
  VP9_BYTE_TILE_ROWS = 79,
  VP9_BYTE_CHROMA_FORMAT = 82,
  VP9_BYTE_REFERENCE_MAP = 83,
  VP9_BYTE_ACTIVE_REFERENCES = 91,
  VP9_BYTE_SIGN_BIASES = 94,
  VP9_BYTE_LUMA_DEPTH = 98,
  VP9_BYTE_CHROMA_DEPTH = 99,
  VP9_BYTE_P010 = 100,
  VP9_BYTE_MSB = 101,
  VP9_WORD_COMPRESSED_HEADER = 27,
  VP9_WORD_UNCOMPRESSED_HEADER = 28,
};

static void amdf_media_drm_vcn_address(uint32_t* words, uint64_t address) {
  words[0] = (uint32_t)(address >> 32);
  words[1] = (uint32_t)address;
}

static uint32_t amdf_media_drm_vcn_stream(amdf_media_codec_t codec) {
  return codec == AMDF_MEDIA_CODEC_H264   ? STREAM_H264
         : codec == AMDF_MEDIA_CODEC_HEVC ? STREAM_HEVC
         : codec == AMDF_MEDIA_CODEC_VP9  ? STREAM_VP9
                                          : STREAM_AV1;
}

bool amdf_media_drm_vcn_protocol(
    uint32_t ip_discovery_version,
    amdf_media_drm_vcn_protocol_info_t* out_protocol) {
  const uint32_t major = (ip_discovery_version >> 16) & 255;
  if (major == 4) {
    *out_protocol = (amdf_media_drm_vcn_protocol_info_t){
        .command_protocol = AMDF_MEDIA_DRM_COMMAND_PROTOCOL_VCN4,
        .decode_array_mode = ip_discovery_version == 0x040003 ? 1 : 3,
        .reference_swizzle_mode = 0,
        .reference_height_alignment = 32,
    };
    return true;
  }
  if (major == 5) {
    *out_protocol = (amdf_media_drm_vcn_protocol_info_t){
        .command_protocol = AMDF_MEDIA_DRM_COMMAND_PROTOCOL_VCN5,
        .decode_array_mode = ip_discovery_version == 0x050001 ? 1 : 3,
        .reference_swizzle_mode = AMDF_MEDIA_DRM_VCN_REFERENCE_SWIZZLE_256B_D,
        .reference_height_alignment = 64,
    };
    return true;
  }
  return false;
}

static uint32_t amdf_media_drm_vcn_ib(amdf_media_drm_buffer_t* storage,
                                      const amdf_media_drm_session_t* session,
                                      uint64_t compressed, uint64_t output) {
  uint32_t* words = storage->host_pointer;
  memset(words, 0, INDIRECT_BUFFER_BYTES);
  words[0] = ENGINE_PACKAGE_BYTES;   // ENGINE_INFO package bytes.
  words[1] = 0x30000001;             // ENGINE_INFO identifier.
  words[2] = 3;                      // Decode engine within the unified ring.
  words[3] = INDIRECT_BUFFER_BYTES;  // Complete engine-package extent.
  words[4] = DECODE_PACKAGE_BYTES;   // Sized decode-buffer package.
  words[5] = 1;                      // DECODE_BUFFER identifier.
  uint32_t* buffers = words + 6;
  buffers[BINDING_FLAGS] = 0x100001;  // Session context and message.
  amdf_media_drm_vcn_address(
      buffers + BINDING_MESSAGE,
      storage->address + AMDF_MEDIA_DRM_VCN_MESSAGE_OFFSET);
  amdf_media_drm_vcn_address(buffers + BINDING_SESSION,
                             session->storage->address);
  if (compressed != 0) {
    const amdf_media_codec_t codec = session->requirements.configuration.codec;
    const bool probabilities =
        codec == AMDF_MEDIA_CODEC_VP9 || codec == AMDF_MEDIA_CODEC_AV1;
    const uint32_t auxiliary_offset =
        codec == AMDF_MEDIA_CODEC_AV1 ? AMDF_MEDIA_DRM_VCN_AV1_AUXILIARY_OFFSET
        : probabilities               ? AMDF_MEDIA_DRM_VCN_PROBABILITY_OFFSET
                                      : AMDF_MEDIA_DRM_VCN_SCALING_OFFSET;
    // Bitstream, target, collocated state, and scaling or probabilities.
    buffers[BINDING_FLAGS] |= probabilities ? 0x180c : 0xa0c;
    amdf_media_drm_vcn_address(buffers + BINDING_OUTPUT, output);
    amdf_media_drm_vcn_address(buffers + BINDING_COMPRESSED, compressed);
    amdf_media_drm_vcn_address(
        buffers + BINDING_CONTEXT,
        session->storage->address + SESSION_CONTEXT_BYTES);
    amdf_media_drm_vcn_address(
        buffers + (probabilities ? BINDING_PROBABILITIES : BINDING_SCALING),
        storage->address + auxiliary_offset);
  }
  return INDIRECT_BUFFER_BYTES;
}

static void amdf_media_drm_vcn_index(uint32_t* index, uint32_t identifier,
                                     uint32_t offset, uint32_t size) {
  index[0] = identifier;
  index[1] = offset;
  index[2] = size;
  index[3] = 0;
}

uint32_t amdf_media_drm_vcn_control(amdf_media_drm_buffer_t* storage,
                                    const amdf_media_drm_session_t* session,
                                    amdf_media_command_operation_t operation) {
  uint32_t* message = (uint32_t*)((uint8_t*)storage->host_pointer +
                                  AMDF_MEDIA_DRM_VCN_MESSAGE_OFFSET);
  memset(message, 0, 56);
  const bool begin = operation == AMDF_MEDIA_COMMAND_OPERATION_BEGIN;
  message[0] = 40;
  message[1] = begin ? 56 : 24;
  message[2] = begin ? 1 : 0;
  message[3] = begin ? 0 : 2;
  message[4] = session->stream_handle;
  if (begin) {
    amdf_media_drm_vcn_index(message + 6, 1, 40, 16);
    message[10] =
        amdf_media_drm_vcn_stream(session->requirements.configuration.codec);
    message[12] = session->requirements.configuration.coded_width;
    message[13] = session->requirements.configuration.coded_height;
  }
  return amdf_media_drm_vcn_ib(storage, session, 0, 0);
}

static void amdf_media_drm_vcn_avc(
    uint8_t* bytes, const amdf_media_h264_configuration_t* configuration,
    uint32_t slot, const amdf_media_h264_decode_picture_info_t* picture) {
  uint32_t* words = (uint32_t*)bytes;
  const amdf_media_h264_sequence_parameters_t* sequence = &picture->sequence;
  const amdf_media_h264_picture_parameters_t* parameters = &picture->picture;
  words[AVC_WORD_PROFILE] =
      configuration->profile - AMDF_MEDIA_H264_PROFILE_CONSTRAINED_BASELINE;
  words[AVC_WORD_LEVEL] = configuration->level_idc;
  words[AVC_WORD_SEQUENCE_FLAGS] =
      4 | sequence->direct_8x8_inference |
      (sequence->delta_picture_order_always_zero << 3);
  words[AVC_WORD_PICTURE_FLAGS] =
      parameters->transform_8x8_mode |
      (parameters->redundant_picture_count_present << 1) |
      (parameters->constrained_intra_prediction << 2) |
      (parameters->deblocking_filter_control_present << 3) |
      (parameters->weighted_biprediction << 4) |
      (parameters->weighted_prediction << 6) |
      (parameters->bottom_field_picture_order_present << 7) |
      (parameters->entropy_coding_mode << 8);
  bytes[AVC_BYTE_CHROMA_FORMAT] =
      1;  // 4:2:0 chroma; the next two bytes select eight-bit samples.
  bytes[AVC_BYTE_FRAME_NUMBER_BITS] =
      (uint8_t)sequence->log2_max_frame_num_minus4;
  bytes[AVC_BYTE_ORDER_TYPE] = (uint8_t)sequence->picture_order_count_type;
  bytes[AVC_BYTE_ORDER_BITS] =
      (uint8_t)sequence->log2_max_picture_order_count_lsb_minus4;
  bytes[AVC_BYTE_MAXIMUM_REFERENCES] =
      (uint8_t)sequence->maximum_reference_frame_count;
  bytes[AVC_BYTE_INITIAL_QP] = (uint8_t)parameters->initial_qp_minus26;
  bytes[AVC_BYTE_INITIAL_QS] = (uint8_t)parameters->initial_qs_minus26;
  bytes[AVC_BYTE_CHROMA_QP] = (uint8_t)parameters->chroma_qp_index_offset;
  bytes[AVC_BYTE_SECOND_CHROMA_QP] =
      (uint8_t)parameters->second_chroma_qp_index_offset;
  bytes[AVC_BYTE_DEFAULT_L0_COUNT] =
      (uint8_t)parameters->default_reference_l0_count_minus1;
  bytes[AVC_BYTE_DEFAULT_L1_COUNT] =
      (uint8_t)parameters->default_reference_l1_count_minus1;
  words[AVC_WORD_FRAME_NUMBER] = picture->frame_number;
  words[AVC_WORD_CURRENT_TOP_ORDER] = (uint32_t)picture->top_field_order_count;
  words[AVC_WORD_CURRENT_BOTTOM_ORDER] =
      (uint32_t)picture->bottom_field_order_count;
  words[AVC_WORD_CURRENT_SLOT] = slot;
  words[AVC_WORD_REFERENCE_COUNT] = picture->reference_count;
  memset(bytes + AVC_BYTE_REFERENCE_SLOTS, 0xff, 16);
  for (uint32_t i = 0; i < picture->reference_count; ++i) {
    const amdf_media_h264_reference_t* reference = &picture->references[i];
    words[AVC_WORD_REFERENCE_FRAME_NUMBERS + i] = reference->frame_number;
    words[AVC_WORD_REFERENCE_ORDERS + i * 2] =
        (uint32_t)reference->top_field_order_count;
    words[AVC_WORD_REFERENCE_ORDERS + i * 2 + 1] =
        (uint32_t)reference->bottom_field_order_count;
    bytes[AVC_BYTE_REFERENCE_SLOTS + i] =
        (uint8_t)(reference->binding.slot_index | (reference->long_term << 7));
    words[AVC_WORD_USED_FOR_REFERENCE] |= 3u << (i * 2);
  }
}

static uint8_t* amdf_media_drm_vcn_decode(
    amdf_media_drm_buffer_t* storage, const amdf_media_drm_session_t* session,
    uint32_t compressed_byte_length, const amdf_media_drm_frame_t* output,
    const amdf_media_drm_frame_t* current,
    const amdf_media_drm_frame_t* references, uint32_t reference_count,
    uint32_t reference_array_count, uint32_t codec_identifier,
    uint32_t codec_byte_length) {
  uint32_t* message = (uint32_t*)((uint8_t*)storage->host_pointer +
                                  AMDF_MEDIA_DRM_VCN_MESSAGE_OFFSET);
  const uint32_t decode_offset = MESSAGE_HEADER_WORDS + 3 * MESSAGE_INDEX_WORDS;
  const uint32_t dynamic_offset = decode_offset + DECODE_WORDS;
  const uint32_t codec_offset = dynamic_offset + DYNAMIC_WORDS;
  const uint32_t byte_length = codec_offset * 4 + codec_byte_length;
  memset(message, 0, byte_length);
  message[0] = 40;
  message[1] = byte_length;
  message[2] = 3;
  message[3] = 1;  // DECODE.
  message[4] = session->stream_handle;
  amdf_media_drm_vcn_index(message + 6, 2, decode_offset * 4, DECODE_WORDS * 4);
  amdf_media_drm_vcn_index(message + 10, codec_identifier, codec_offset * 4,
                           codec_byte_length);
  amdf_media_drm_vcn_index(message + 14, 0x10, dynamic_offset * 4,
                           DYNAMIC_WORDS * 4);
  uint32_t* decode = message + decode_offset;
  const uint32_t sample_bytes =
      session->requirements.configuration.format == AMDF_MEDIA_PIXEL_FORMAT_P010
          ? 2
          : 1;
  decode[DECODE_STREAM] =
      amdf_media_drm_vcn_stream(session->requirements.configuration.codec);
  decode[DECODE_FLAGS] = 1;  // Explicit dynamic reference surfaces (tier 2).
  decode[DECODE_WIDTH] = session->requirements.configuration.coded_width;
  decode[DECODE_HEIGHT] = session->requirements.configuration.coded_height;
  decode[DECODE_COMPRESSED_BYTES] = compressed_byte_length;
  decode[DECODE_OUTPUT_BYTES] =
      (uint32_t)(output->addresses[1] - output->addresses[0]) +
      output->row_pitches[1] * output->row_counts[1];
  decode[DECODE_HARDWARE_CONTEXT_BYTES] = session->context_byte_length;
  decode[DECODE_SESSION_CONTEXT_BYTES] = SESSION_CONTEXT_BYTES;
  // Message, compressed, output, context, and codec-specific auxiliary storage.
  decode[DECODE_BUFFER_FLAGS] =
      (session->requirements.configuration.codec == AMDF_MEDIA_CODEC_VP9 ||
       session->requirements.configuration.codec == AMDF_MEDIA_CODEC_AV1)
          ? 0x180d
          : 0xa0d;
  decode[DECODE_REFERENCE_PITCH] = current->row_pitches[0] / sample_bytes;
  decode[DECODE_REFERENCE_ROWS] = current->row_counts[0];
  const amdf_media_drm_device_t* device =
      (const amdf_media_drm_device_t*)session->base.queue->device;
  const amdf_media_drm_engine_t* engine =
      &device->engines[session->base.queue->info.queue_family_ordinal];
  decode[DECODE_REFERENCE_SWIZZLE] = current->swizzle_mode;
  decode[DECODE_REFERENCE_ARRAY_MODE] = engine->decode_array_mode;
  decode[DECODE_OUTPUT_PITCH] = output->row_pitches[0] / sample_bytes;
  decode[DECODE_OUTPUT_CHROMA_PITCH] =
      output->row_pitches[1] /
      (2 * sample_bytes);  // UV pitch counts interleaved pairs.
  decode[DECODE_OUTPUT_ARRAY_MODE] = decode[DECODE_REFERENCE_ARRAY_MODE];
  decode[DECODE_OUTPUT_CHROMA_OFFSET] =
      (uint32_t)(output->addresses[1] - output->addresses[0]);
  decode[DECODE_WRITE_COHERENCY] =
      1;  // Media write coherency on VCN 3 and newer.
  decode[DECODE_REFERENCE_CHROMA_PITCH] =
      current->row_pitches[1] / (2 * sample_bytes);
  uint32_t* dynamic = message + dynamic_offset;
  dynamic[DYNAMIC_LUMA_PITCH] = current->row_pitches[0] / sample_bytes;
  dynamic[DYNAMIC_LUMA_ROWS] = current->row_counts[0];
  dynamic[DYNAMIC_LUMA_BYTES] =
      current->row_pitches[0] * current->row_counts[0];
  dynamic[DYNAMIC_CHROMA_PITCH] = current->row_pitches[1] / (2 * sample_bytes);
  dynamic[DYNAMIC_CHROMA_ROWS] = current->row_counts[1];
  dynamic[DYNAMIC_CHROMA_BYTES] =
      current->row_pitches[1] * current->row_counts[1];
  dynamic[DYNAMIC_REFERENCE_COUNT] = reference_array_count;
  dynamic[DYNAMIC_CURRENT_LOW] = (uint32_t)current->addresses[0];
  dynamic[DYNAMIC_CURRENT_HIGH] = (uint32_t)(current->addresses[0] >> 32);
  for (uint32_t i = 0; i < reference_count; ++i) {
    dynamic[DYNAMIC_REFERENCE_LOW + i] = (uint32_t)references[i].addresses[0];
    dynamic[DYNAMIC_REFERENCE_HIGH + i] =
        (uint32_t)(references[i].addresses[0] >> 32);
  }
  return (uint8_t*)(message + codec_offset);
}

uint32_t amdf_media_drm_vcn_h264(
    amdf_media_drm_buffer_t* storage, const amdf_media_drm_session_t* session,
    uint64_t compressed_address, uint32_t compressed_byte_length,
    const amdf_media_drm_frame_t* output, const amdf_media_drm_frame_t* current,
    const amdf_media_drm_frame_t* references, uint32_t current_slot,
    const amdf_media_h264_decode_picture_info_t* picture) {
  uint8_t* codec = amdf_media_drm_vcn_decode(
      storage, session, compressed_byte_length, output, current, references,
      picture->reference_count, picture->reference_count, 6, AVC_BYTES);
  amdf_media_drm_vcn_avc(codec, &session->codec.h264, current_slot, picture);
  // VCN AVC consumes raster scaling matrices; DXVA's zigzag transform is
  // a property of that separate native interface, not of the public schema.
  memcpy((uint8_t*)storage->host_pointer + AMDF_MEDIA_DRM_VCN_SCALING_OFFSET,
         picture->scaling_matrices, sizeof(*picture->scaling_matrices));
  return amdf_media_drm_vcn_ib(storage, session, compressed_address,
                               output->addresses[0]);
}

uint32_t amdf_media_drm_vcn_hevc(
    amdf_media_drm_buffer_t* storage, const amdf_media_drm_session_t* session,
    uint64_t compressed_address, uint32_t compressed_byte_length,
    const amdf_media_drm_frame_t* output, const amdf_media_drm_frame_t* current,
    const amdf_media_drm_frame_t* references, uint32_t current_slot,
    const amdf_media_hevc_decode_picture_info_t* picture) {
  const amdf_media_hevc_sequence_parameters_t* sequence = &picture->sequence;
  const amdf_media_hevc_picture_parameters_t* parameters = &picture->picture;
  uint8_t* codec = amdf_media_drm_vcn_decode(
      storage, session, compressed_byte_length, output, current, references,
      picture->reference_count,
      sequence->maximum_decoded_picture_buffer_count_minus1 + 1, 0x0d,
      HEVC_BYTES);
  uint32_t* words = (uint32_t*)codec;
  words[HEVC_WORD_SEQUENCE_FLAGS] =
      sequence->scaling_list_enabled |
      (sequence->asymmetric_motion_partitions_enabled << 1) |
      (sequence->sample_adaptive_offset_enabled << 2) |
      (sequence->pcm_enabled << 3) | (sequence->pcm_loop_filter_disabled << 4) |
      (sequence->long_term_references_present << 5) |
      (sequence->temporal_motion_vector_prediction_enabled << 6) |
      (sequence->strong_intra_smoothing_enabled << 7) |
      ((picture->short_term_reference_set_bit_count != 0) << 11);
  words[HEVC_WORD_PICTURE_FLAGS] =
      parameters->dependent_slice_segments_enabled |
      (parameters->output_flag_present << 1) |
      (parameters->sign_data_hiding_enabled << 2) |
      (parameters->cabac_initialization_present << 3) |
      (parameters->constrained_intra_prediction << 4) |
      (parameters->transform_skip_enabled << 5) |
      (parameters->coding_unit_qp_delta_enabled << 6) |
      (parameters->slice_chroma_qp_offsets_present << 7) |
      (parameters->weighted_prediction << 8) |
      (parameters->weighted_biprediction << 9) |
      (parameters->transform_quantization_bypass_enabled << 10) |
      (parameters->tiles_enabled << 11) |
      (parameters->entropy_coding_synchronization_enabled << 12) |
      (parameters->uniform_tile_spacing << 13) |
      (parameters->loop_filter_across_tiles_enabled << 14) |
      (parameters->loop_filter_across_slices_enabled << 15) |
      (parameters->deblocking_filter_override_enabled << 16) |
      (parameters->deblocking_filter_disabled << 17) |
      (parameters->reference_lists_modification_present << 18) |
      (parameters->slice_segment_header_extension_present << 19);
  codec[HEVC_BYTE_CHROMA_FORMAT] = 1;
  codec[HEVC_BYTE_LUMA_DEPTH] =
      (uint8_t)(session->codec.hevc.sample_bit_depth - 8);
  codec[HEVC_BYTE_CHROMA_DEPTH] =
      (uint8_t)(session->codec.hevc.sample_bit_depth - 8);
  codec[HEVC_BYTE_ORDER_BITS] =
      (uint8_t)(sequence->log2_max_picture_order_count_lsb_minus4);
  codec[HEVC_BYTE_BUFFER_COUNT] =
      (uint8_t)(sequence->maximum_decoded_picture_buffer_count_minus1);
  codec[HEVC_BYTE_MINIMUM_BLOCK] =
      (uint8_t)(sequence->log2_min_luma_coding_block_size_minus3);
  codec[HEVC_BYTE_BLOCK_RANGE] =
      (uint8_t)(sequence->log2_diff_max_min_luma_coding_block_size);
  codec[HEVC_BYTE_MINIMUM_TRANSFORM] =
      (uint8_t)(sequence->log2_min_luma_transform_block_size_minus2);
  codec[HEVC_BYTE_TRANSFORM_RANGE] =
      (uint8_t)(sequence->log2_diff_max_min_luma_transform_block_size);
  codec[HEVC_BYTE_INTER_DEPTH] =
      (uint8_t)sequence->maximum_transform_hierarchy_depth_inter;
  codec[HEVC_BYTE_INTRA_DEPTH] =
      (uint8_t)sequence->maximum_transform_hierarchy_depth_intra;
  codec[HEVC_BYTE_SHORT_TERM_SETS] =
      (uint8_t)sequence->short_term_reference_set_count;
  codec[HEVC_BYTE_LONG_TERM_COUNT] =
      (uint8_t)sequence->long_term_reference_count;
  codec[HEVC_BYTE_EXTRA_HEADER_BITS] =
      (uint8_t)parameters->extra_slice_header_bit_count;
  codec[HEVC_BYTE_DEFAULT_L0_COUNT] =
      (uint8_t)(parameters->default_reference_l0_count_minus1);
  codec[HEVC_BYTE_DEFAULT_L1_COUNT] =
      (uint8_t)(parameters->default_reference_l1_count_minus1);
  codec[HEVC_BYTE_CB_QP] = (uint8_t)parameters->cb_qp_offset;
  codec[HEVC_BYTE_CR_QP] = (uint8_t)parameters->cr_qp_offset;
  codec[HEVC_BYTE_BETA_OFFSET] = (uint8_t)(parameters->beta_offset_div2);
  codec[HEVC_BYTE_TC_OFFSET] = (uint8_t)(parameters->tc_offset_div2);
  codec[HEVC_BYTE_QP_DELTA_DEPTH] =
      (uint8_t)parameters->coding_unit_qp_delta_depth;
  codec[HEVC_BYTE_TILE_COLUMNS] =
      (uint8_t)(parameters->tile_column_count_minus1);
  codec[HEVC_BYTE_TILE_ROWS] = (uint8_t)(parameters->tile_row_count_minus1);
  codec[HEVC_BYTE_MERGE_LEVEL] =
      (uint8_t)(parameters->log2_parallel_merge_level_minus2);
  codec[HEVC_BYTE_INITIAL_QP] = (uint8_t)(parameters->initial_qp_minus26);
  codec[HEVC_BYTE_PREDICTED_DELTA_COUNT] =
      (uint8_t)picture->predicted_reference_set_delta_count;
  codec[HEVC_BYTE_CURRENT_SLOT] = (uint8_t)(current_slot);
  codec[HEVC_BYTE_HIGHEST_TEMPORAL_ID] = 255;
  codec[HEVC_BYTE_P010] = (uint8_t)(session->codec.hevc.sample_bit_depth == 10);
  codec[HEVC_BYTE_MSB] = (uint8_t)(session->codec.hevc.sample_bit_depth == 10);
  if (sequence->pcm_enabled) {
    codec[HEVC_BYTE_PCM_LUMA_DEPTH] =
        (uint8_t)sequence->pcm_luma_bit_depth_minus1;
    codec[HEVC_BYTE_PCM_CHROMA_DEPTH] =
        (uint8_t)sequence->pcm_chroma_bit_depth_minus1;
    codec[HEVC_BYTE_PCM_MINIMUM_BLOCK] =
        (uint8_t)sequence->log2_min_pcm_luma_coding_block_size_minus3;
    codec[HEVC_BYTE_PCM_BLOCK_RANGE] =
        (uint8_t)sequence->log2_diff_max_min_pcm_luma_coding_block_size;
  }
  words[HEVC_WORD_CURRENT_ORDER] = (uint32_t)picture->picture_order_count;
  words[HEVC_WORD_REFERENCE_SET_BITS] =
      picture->short_term_reference_set_bit_count;
  if (parameters->tiles_enabled && !parameters->uniform_tile_spacing) {
    uint16_t* widths = (uint16_t*)(codec + HEVC_BYTE_TILE_WIDTHS);
    uint16_t* heights = (uint16_t*)(codec + HEVC_BYTE_TILE_HEIGHTS);
    for (uint32_t i = 0; i < parameters->tile_column_count_minus1; ++i) {
      widths[i] = (uint16_t)parameters->tile_column_width_minus1[i];
    }
    for (uint32_t i = 0; i < parameters->tile_row_count_minus1; ++i) {
      heights[i] = (uint16_t)parameters->tile_row_height_minus1[i];
    }
  }
  memset(codec + HEVC_BYTE_REFERENCE_SLOTS, 0x7f, 16);
  uint32_t* orders = (uint32_t*)(codec + HEVC_BYTE_REFERENCE_ORDERS);
  for (uint32_t i = 0; i < picture->reference_count; ++i) {
    codec[HEVC_BYTE_REFERENCE_SLOTS + i] =
        (uint8_t)picture->references[i].binding.slot_index;
    orders[i] = (uint32_t)picture->references[i].picture_order_count;
  }
  const amdf_media_hevc_reference_set_t* sets[] = {&picture->short_term_before,
                                                   &picture->short_term_after,
                                                   &picture->long_term};
  const uint32_t offsets[] = {HEVC_BYTE_SHORT_TERM_BEFORE,
                              HEVC_BYTE_SHORT_TERM_AFTER, HEVC_BYTE_LONG_TERM};
  for (uint32_t i = 0; i < 3; ++i) {
    memset(codec + offsets[i], 0xff, 8);
    memcpy(codec + offsets[i], sets[i]->indices, sets[i]->count);
  }
  // Every picture has an explicit reconstructed target, including pictures not
  // selected by the application's later reference sets. isNonRef remains zero.
  uint8_t* scaling =
      (uint8_t*)storage->host_pointer + AMDF_MEDIA_DRM_VCN_SCALING_OFFSET;
  if (sequence->scaling_list_enabled) {
    const amdf_media_hevc_scaling_matrices_t* matrices =
        picture->scaling_matrices;
    for (uint32_t i = 0; i < 6; ++i) {
      amdf_media_hevc_scan_scaling_list(matrices->scaling_4x4[i], 4,
                                        scaling + i * 16);
      amdf_media_hevc_scan_scaling_list(matrices->scaling_8x8[i], 8,
                                        scaling + 96 + i * 64);
      amdf_media_hevc_scan_scaling_list(matrices->scaling_16x16[i], 8,
                                        scaling + 480 + i * 64);
      codec[HEVC_BYTE_DC_16X16 + i] = matrices->dc_16x16[i];
    }
    for (uint32_t i = 0; i < 2; ++i) {
      amdf_media_hevc_scan_scaling_list(matrices->scaling_32x32[i], 8,
                                        scaling + 864 + i * 64);
      codec[HEVC_BYTE_DC_32X32 + i] = matrices->dc_32x32[i];
    }
  } else {
    memset(scaling, 0, 992);
  }
  return amdf_media_drm_vcn_ib(storage, session, compressed_address,
                               output->addresses[0]);
}

static void amdf_media_drm_vcn_reference_map(
    const uint8_t map[8], const amdf_media_reference_t* bindings,
    const amdf_media_drm_frame_t* references, uint32_t* dynamic,
    uint8_t slots[8]) {
  for (uint32_t i = 0; i < 8; ++i) {
    const uint32_t index = map[i];
    slots[i] = index == 255 ? 127 : (uint8_t)bindings[index].slot_index;
    if (index != 255) {
      dynamic[DYNAMIC_REFERENCE_LOW + i] =
          (uint32_t)references[index].addresses[0];
      dynamic[DYNAMIC_REFERENCE_HIGH + i] =
          (uint32_t)(references[index].addresses[0] >> 32);
    }
  }
}

static uint8_t amdf_media_drm_vcn_vp9_filter_level(int32_t value) {
  return (uint8_t)(value < 0 ? 0 : value > 63 ? 63 : value);
}

uint32_t amdf_media_drm_vcn_vp9(
    amdf_media_drm_buffer_t* storage, const amdf_media_drm_session_t* session,
    uint64_t compressed_address, uint32_t compressed_byte_length,
    const amdf_media_drm_frame_t* output, const amdf_media_drm_frame_t* current,
    const amdf_media_drm_frame_t* references, uint32_t current_slot,
    const amdf_media_vp9_decode_picture_info_t* picture) {
  uint8_t* codec =
      amdf_media_drm_vcn_decode(storage, session, compressed_byte_length,
                                output, current, NULL, 0, 8, 0x0e, VP9_BYTES);
  uint32_t* words = (uint32_t*)codec;
  const amdf_media_vp9_loop_filter_t* filter = &picture->loop_filter;
  const amdf_media_vp9_segmentation_t* segments = &picture->segmentation;
  words[0] = (picture->frame_type << 1) | (picture->error_resilient << 2) |
             (picture->intra_only << 3) |
             (picture->high_precision_motion_vectors << 4) |
             (picture->refresh_frame_context << 5) |
             (picture->parallel_decoding << 6) | (segments->enabled << 7) |
             (segments->update_map << 8) | (segments->temporal_update << 9) |
             (segments->update_data << 10) | (filter->deltas_enabled << 11) |
             (filter->deltas_updated << 12) |
             (picture->use_previous_frame_motion_vectors << 13) |
             (1u << 14);  // The payload includes its uncompressed header.
  words[VP9_WORD_COMPRESSED_HEADER] = picture->compressed_header_byte_length;
  words[VP9_WORD_UNCOMPRESSED_HEADER] =
      picture->uncompressed_header_byte_length;
  codec[VP9_BYTE_CONTEXT] = (uint8_t)picture->frame_context_index;
  codec[VP9_BYTE_RESET_CONTEXT] = (uint8_t)picture->reset_frame_context;
  codec[VP9_BYTE_CURRENT_SLOT] = (uint8_t)current_slot;
  codec[VP9_BYTE_INTERPOLATION] = (uint8_t)picture->interpolation_filter;
  codec[VP9_BYTE_FILTER_LEVEL] = (uint8_t)filter->level;
  codec[VP9_BYTE_SHARPNESS] = (uint8_t)filter->sharpness;
  codec[VP9_BYTE_BASE_QUANTIZER] = (uint8_t)picture->base_quantizer_index;
  codec[VP9_BYTE_Y_DC_DELTA] = (uint8_t)picture->y_dc_quantizer_delta;
  codec[VP9_BYTE_UV_AC_DELTA] = (uint8_t)picture->uv_ac_quantizer_delta;
  codec[VP9_BYTE_UV_DC_DELTA] = (uint8_t)picture->uv_dc_quantizer_delta;
  codec[VP9_BYTE_TILE_COLUMNS] = (uint8_t)picture->log2_tile_columns;
  codec[VP9_BYTE_TILE_ROWS] = (uint8_t)picture->log2_tile_rows;
  codec[VP9_BYTE_CHROMA_FORMAT] = 1;
  codec[VP9_BYTE_LUMA_DEPTH] =
      (uint8_t)(session->codec.vp9.sample_bit_depth - 8);
  codec[VP9_BYTE_CHROMA_DEPTH] = codec[VP9_BYTE_LUMA_DEPTH];
  codec[VP9_BYTE_P010] = session->codec.vp9.sample_bit_depth == 10;
  codec[VP9_BYTE_MSB] = codec[VP9_BYTE_P010];
  const int32_t scale = 1 << (filter->level >> 5);
  for (uint32_t segment = 0; segment < 8; ++segment) {
    int32_t base = (int32_t)filter->level;
    if (segments->enabled && (segments->feature_masks[segment] & 2) != 0) {
      base = amdf_media_drm_vcn_vp9_filter_level(
          segments->feature_values[segment][1] +
          (segments->absolute_values ? 0 : base));
    }
    for (uint32_t reference = 0; reference < 4; ++reference) {
      for (uint32_t mode = 0; mode < 2; ++mode) {
        const int32_t delta =
            filter->deltas_enabled
                ? filter->reference_deltas[reference] +
                      (reference ? filter->mode_deltas[mode] : 0)
                : 0;
        codec[VP9_BYTE_ADJUSTED_LEVELS + (segment * 4 + reference) * 2 + mode] =
            amdf_media_drm_vcn_vp9_filter_level(base + delta * scale);
      }
    }
  }
  // Tier-2 VP9 addresses follow the eight logical map entries. Aliasing those
  // entries is legal; the residency list retains each unique physical binding.
  uint32_t* dynamic = (uint32_t*)(codec - DYNAMIC_WORDS * 4);
  amdf_media_drm_vcn_reference_map(picture->reference_map, picture->references,
                                   references, dynamic,
                                   codec + VP9_BYTE_REFERENCE_MAP);
  for (uint32_t i = 0; i < 3; ++i) {
    const uint32_t index = picture->active_reference_indices[i];
    codec[VP9_BYTE_ACTIVE_REFERENCES + i] =
        index == 255 ? 127 : codec[VP9_BYTE_REFERENCE_MAP + index];
    codec[VP9_BYTE_SIGN_BIASES + i] = picture->reference_sign_bias[i];
  }
  uint8_t* probabilities =
      (uint8_t*)storage->host_pointer + AMDF_MEDIA_DRM_VCN_PROBABILITY_OFFSET;
  memset(probabilities, 0, 2560);
  if (segments->enabled) {
    uint32_t* values = (uint32_t*)(probabilities + 2304);
    for (uint32_t i = 0; i < 8; ++i) {
      values[i] = (uint16_t)segments->feature_values[i][0] |
                  ((uint32_t)(uint8_t)segments->feature_values[i][1] << 16) |
                  ((uint32_t)segments->feature_values[i][2] << 24);
    }
    memcpy(probabilities + 2336, segments->tree_probabilities, 7);
    memcpy(probabilities + 2343, segments->prediction_probabilities, 3);
    probabilities[2346] = (uint8_t)segments->absolute_values;
    memcpy(probabilities + 2347, segments->feature_masks, 8);
  }
  return amdf_media_drm_vcn_ib(storage, session, compressed_address,
                               output->addresses[0]);
}

// Byte offsets in the VCN4 AV1 picture message.
enum {
  AV1_FRAME_HEADER_FLAGS = 0,
  AV1_CURRENT_FRAME_ID = 4,
  AV1_FRAME_OFFSET = 8,
  AV1_PROFILE = 12,
  AV1_IS_ANNEXB = 13,
  AV1_FRAME_TYPE = 14,
  AV1_PRIMARY_REF_FRAME = 15,
  AV1_CURR_PIC_IDX = 16,
  AV1_SB_SIZE = 17,
  AV1_INTERP_FILTER = 18,
  AV1_FILTER_LEVEL = 19,
  AV1_FILTER_LEVEL_U = 21,
  AV1_FILTER_LEVEL_V = 22,
  AV1_SHARPNESS_LEVEL = 23,
  AV1_REF_DELTAS = 24,
  AV1_MODE_DELTAS = 32,
  AV1_BASE_QINDEX = 34,
  AV1_Y_DC_DELTA_Q = 35,
  AV1_U_DC_DELTA_Q = 36,
  AV1_V_DC_DELTA_Q = 37,
  AV1_U_AC_DELTA_Q = 38,
  AV1_V_AC_DELTA_Q = 39,
  AV1_QM_Y = 40,
  AV1_QM_U = 41,
  AV1_QM_V = 42,
  AV1_DELTA_Q_RES = 43,
  AV1_DELTA_LF_RES = 44,
  AV1_TILE_COLS = 45,
  AV1_TILE_ROWS = 46,
  AV1_TX_MODE = 47,
  AV1_REFERENCE_MODE = 48,
  AV1_CHROMA_FORMAT = 49,
  AV1_TILE_SIZE_BYTES = 52,
  AV1_CONTEXT_UPDATE_TILE_ID = 56,
  AV1_TILE_COL_START_SB = 60,
  AV1_TILE_ROW_START_SB = 320,
  AV1_MAX_WIDTH = 580,
  AV1_MAX_HEIGHT = 584,
  AV1_WIDTH = 588,
  AV1_HEIGHT = 592,
  AV1_SUPERRES_UPSCALED_WIDTH = 596,
  AV1_SUPERRES_SCALE_DENOMINATOR = 600,
  AV1_ORDER_HINT_BITS = 601,
  AV1_REF_FRAME_MAP = 602,
  AV1_REF_FRAME_OFFSET = 612,
  AV1_FRAME_REFS = 644,
  AV1_REF_FRAME_SIGN_BIAS = 651,
  AV1_BIT_DEPTH_LUMA_MINUS8 = 658,
  AV1_BIT_DEPTH_CHROMA_MINUS8 = 659,
  AV1_FEATURE_DATA = 660,
  AV1_FEATURE_MASK = 916,
  AV1_CDEF_DAMPING = 924,
  AV1_CDEF_BITS = 925,
  AV1_CDEF_STRENGTHS = 926,
  AV1_CDEF_UV_STRENGTHS = 958,
  AV1_FRAME_RESTORATION_TYPE = 990,
  AV1_LOG2_RESTORATION_UNIT_SIZE_MINUS5 = 993,
  AV1_P010_MODE = 996,
  AV1_MSB_MODE = 997,
  AV1_LUMA_10TO8 = 998,
  AV1_CHROMA_10TO8 = 999,
  AV1_PRESKIP_SEGID = 1000,
  AV1_LAST_ACTIVE_SEGID = 1001,
  AV1_SEG_LOSSLESS_FLAG = 1002,
  AV1_CODED_LOSSLESS = 1003,
  AV1_FILM_GRAIN = 1004,
  AV1_UNCOMPRESSED_HEADER_SIZE = 1172,
  AV1_GLOBAL_MOTION = 1176,
  AV1_TILE_INFO = 1528,
  AV1_RESERVED = 3576,
  AV1_AV1_INTRABC_WORKAROUND = 3580,
  AV1_SIZE = 3584,
};

uint32_t amdf_media_drm_vcn_av1(
    amdf_media_drm_buffer_t* storage, const amdf_media_drm_session_t* session,
    uint64_t compressed_address, uint32_t compressed_byte_length,
    const amdf_media_drm_frame_t* output, const amdf_media_drm_frame_t* current,
    const amdf_media_drm_frame_t* references, uint32_t current_slot,
    const amdf_media_av1_decode_picture_info_t* picture) {
  uint8_t* codec =
      amdf_media_drm_vcn_decode(storage, session, compressed_byte_length,
                                output, current, NULL, 0, 8, 0x11, AV1_SIZE);
  const amdf_media_av1_sequence_parameters_t* sequence = &picture->sequence;
  const amdf_media_av1_loop_filter_t* filter = &picture->loop_filter;
  const amdf_media_av1_quantization_t* quantization = &picture->quantization;
  const amdf_media_av1_segmentation_t* segments = &picture->segmentation;
  uint32_t flags = picture->show_frame |
                   ((!picture->disable_frame_end_cdf_update) << 2) |
                   ((picture->frame_type == 2) << 3) |
                   ((quantization->matrix_levels[0] != 255) << 8) |
                   ((sequence->order_hint_bit_count != 0) << 15) |
                   ((uint32_t)!sequence->reference_frame_motion_vectors << 31);
  flags |= picture->disable_cdf_update << 1;
  flags |= picture->intra_block_copy << 4;
  flags |= picture->high_precision_motion_vectors << 5;
  flags |= picture->skip_mode << 7;
  flags |= sequence->filter_intra << 9;
  flags |= sequence->intra_edge_filter << 10;
  flags |= sequence->interintra_compound << 11;
  flags |= sequence->masked_compound << 12;
  flags |= picture->warped_motion << 13;
  flags |= sequence->dual_filter << 14;
  flags |= sequence->joint_compound << 16;
  flags |= picture->use_reference_motion_vectors << 17;
  flags |= picture->screen_content_tools << 18;
  flags |= picture->integer_motion_vectors << 19;
  flags |= filter->deltas_enabled << 20;
  flags |= filter->deltas_updated << 21;
  flags |= quantization->delta_present << 22;
  flags |= filter->delta_present << 23;
  flags |= picture->reduced_transform_set << 24;
  flags |= segments->enabled << 25;
  flags |= segments->update_map << 26;
  flags |= segments->temporal_update << 27;
  flags |= filter->delta_multi << 28;
  flags |= picture->switchable_motion_mode << 29;
  *(uint32_t*)(codec + AV1_FRAME_HEADER_FLAGS) = flags;
  *(uint32_t*)(codec + AV1_FRAME_OFFSET) = picture->order_hint;
  *(uint32_t*)(codec + AV1_TILE_SIZE_BYTES) = 255;
  *(uint32_t*)(codec + AV1_CONTEXT_UPDATE_TILE_ID) =
      picture->context_update_tile;
  *(uint32_t*)(codec + AV1_MAX_WIDTH) =
      session->requirements.configuration.coded_width;
  *(uint32_t*)(codec + AV1_MAX_HEIGHT) =
      session->requirements.configuration.coded_height;
  *(uint32_t*)(codec + AV1_WIDTH) =
      (session->requirements.configuration.coded_width * 8 +
       picture->superres_denominator / 2) /
      picture->superres_denominator;
  *(uint32_t*)(codec + AV1_HEIGHT) =
      session->requirements.configuration.coded_height;
  *(uint32_t*)(codec + AV1_SUPERRES_UPSCALED_WIDTH) =
      session->requirements.configuration.coded_width;
  codec[AV1_PROFILE] = (uint8_t)(session->codec.av1.profile);
  codec[AV1_FRAME_TYPE] = (uint8_t)(picture->frame_type);
  codec[AV1_PRIMARY_REF_FRAME] = (uint8_t)(picture->primary_reference);
  codec[AV1_CURR_PIC_IDX] = (uint8_t)(current_slot);
  codec[AV1_SB_SIZE] = (uint8_t)(sequence->use_128x128_superblock);
  codec[AV1_INTERP_FILTER] = (uint8_t)(picture->interpolation_filter);
  codec[AV1_FILTER_LEVEL_U] = (uint8_t)(filter->levels[2]);
  codec[AV1_FILTER_LEVEL_V] = (uint8_t)(filter->levels[3]);
  codec[AV1_SHARPNESS_LEVEL] = (uint8_t)(filter->sharpness);
  codec[AV1_BASE_QINDEX] = (uint8_t)(quantization->base_index);
  codec[AV1_Y_DC_DELTA_Q] = (uint8_t)(quantization->y_dc_delta);
  codec[AV1_U_DC_DELTA_Q] = (uint8_t)(quantization->chroma_dc_deltas[0]);
  codec[AV1_V_DC_DELTA_Q] = (uint8_t)(quantization->chroma_dc_deltas[1]);
  codec[AV1_U_AC_DELTA_Q] = (uint8_t)(quantization->chroma_ac_deltas[0]);
  codec[AV1_V_AC_DELTA_Q] = (uint8_t)(quantization->chroma_ac_deltas[1]);
  codec[AV1_QM_Y] = (uint8_t)(quantization->matrix_levels[0]);
  codec[AV1_QM_U] = (uint8_t)(quantization->matrix_levels[1]);
  codec[AV1_QM_V] = (uint8_t)(quantization->matrix_levels[2]);
  codec[AV1_DELTA_Q_RES] = (uint8_t)(1u << quantization->delta_resolution);
  codec[AV1_DELTA_LF_RES] = (uint8_t)(1u << filter->delta_resolution);
  codec[AV1_TILE_COLS] = (uint8_t)(picture->tile_column_count);
  codec[AV1_TILE_ROWS] = (uint8_t)(picture->tile_row_count);
  codec[AV1_TX_MODE] = (uint8_t)(picture->transform_mode);
  codec[AV1_REFERENCE_MODE] = (uint8_t)(picture->reference_select ? 2 : 0);
  codec[AV1_CHROMA_FORMAT] = (uint8_t)(1);
  codec[AV1_SUPERRES_SCALE_DENOMINATOR] =
      (uint8_t)(picture->superres_denominator);
  codec[AV1_ORDER_HINT_BITS] = (uint8_t)(sequence->order_hint_bit_count);
  codec[AV1_BIT_DEPTH_LUMA_MINUS8] =
      (uint8_t)(session->codec.av1.sample_bit_depth - 8);
  codec[AV1_BIT_DEPTH_CHROMA_MINUS8] =
      (uint8_t)(session->codec.av1.sample_bit_depth - 8);
  codec[AV1_P010_MODE] = (uint8_t)(session->codec.av1.sample_bit_depth == 10);
  codec[AV1_MSB_MODE] = (uint8_t)(session->codec.av1.sample_bit_depth == 10);
  codec[AV1_CDEF_DAMPING] = (uint8_t)(picture->cdef.damping_minus3 + 3);
  codec[AV1_CDEF_BITS] = (uint8_t)(picture->cdef.bits);
  memcpy(codec + AV1_FILTER_LEVEL, filter->levels, 2);
  memcpy(codec + AV1_REF_DELTAS, filter->reference_deltas, 8);
  memcpy(codec + AV1_MODE_DELTAS, filter->mode_deltas, 2);
  uint32_t* columns = (uint32_t*)(codec + AV1_TILE_COL_START_SB);
  uint32_t* rows = (uint32_t*)(codec + AV1_TILE_ROW_START_SB);
  for (uint32_t i = 0; i <= picture->tile_column_count; ++i) {
    columns[i] = picture->tile_column_starts[i];
  }
  for (uint32_t i = 0; i <= picture->tile_row_count; ++i) {
    rows[i] = picture->tile_row_starts[i];
  }
  uint32_t* dynamic = (uint32_t*)(codec - DYNAMIC_WORDS * 4);
  amdf_media_drm_vcn_reference_map(picture->reference_map, picture->references,
                                   references, dynamic,
                                   codec + AV1_REF_FRAME_MAP);
  for (uint32_t i = 0; i < 7; ++i) {
    const uint32_t index = picture->active_reference_indices[i];
    codec[AV1_FRAME_REFS + i] =
        index == 255 ? 127 : codec[AV1_REF_FRAME_MAP + index];
    codec[AV1_REF_FRAME_SIGN_BIAS + i] = picture->reference_sign_bias[i];
    // The firmware array includes INTRA at zero. LAST through ALTREF follow.
    uint32_t* motion = (uint32_t*)(codec + AV1_GLOBAL_MOTION + (i + 1) * 44);
    motion[0] = picture->global_motion[i].type;
    memcpy(motion + 1, picture->global_motion[i].parameters, 24);
  }
  uint8_t* auxiliary =
      (uint8_t*)storage->host_pointer + AMDF_MEDIA_DRM_VCN_AV1_AUXILIARY_OFFSET;
  memset(auxiliary, 0, AMDF_MEDIA_DRM_VCN_AV1_AUXILIARY_BYTES);
  const bool zero_deltas = quantization->y_dc_delta == 0 &&
                           quantization->chroma_dc_deltas[0] == 0 &&
                           quantization->chroma_dc_deltas[1] == 0 &&
                           quantization->chroma_ac_deltas[0] == 0 &&
                           quantization->chroma_ac_deltas[1] == 0;
  int32_t* features = (int32_t*)(codec + AV1_FEATURE_DATA);
  for (uint32_t segment = 0; segment < 8; ++segment) {
    const uint32_t mask =
        segments->enabled ? segments->feature_masks[segment] : 0;
    int32_t index = (int32_t)quantization->base_index;
    if ((mask & 1) != 0) {
      index += segments->feature_values[segment][0];
    }
    if (index <= 0 && zero_deltas) {
      codec[AV1_SEG_LOSSLESS_FLAG] |= (uint8_t)(1u << segment);
    }
    if (segments->enabled) {
      codec[AV1_FEATURE_MASK + segment] = (uint8_t)mask;
      for (uint32_t feature = 0; feature < 8; ++feature) {
        features[segment * 8 + feature] =
            segments->feature_values[segment][feature];
        if ((mask & (1u << feature)) != 0) {
          codec[AV1_LAST_ACTIVE_SEGID] = (uint8_t)segment;
          if (feature >= 5) {
            codec[AV1_PRESKIP_SEGID] = 1;
          }
        }
      }
    }
  }
  if (segments->enabled) {
    memcpy(auxiliary, segments->feature_values,
           sizeof(segments->feature_values));
    memcpy(auxiliary + 128, segments->feature_masks,
           sizeof(segments->feature_masks));
  }
  uint16_t* cdef_luma = (uint16_t*)(codec + AV1_CDEF_STRENGTHS);
  uint16_t* cdef_chroma = (uint16_t*)(codec + AV1_CDEF_UV_STRENGTHS);
  for (uint32_t i = 0; i < 8; ++i) {
    cdef_luma[i] = (uint16_t)((picture->cdef.luma_primary[i] << 2) |
                              picture->cdef.luma_secondary[i]);
    cdef_chroma[i] = (uint16_t)((picture->cdef.chroma_primary[i] << 2) |
                                picture->cdef.chroma_secondary[i]);
  }
  for (uint32_t i = 0; i < 3; ++i) {
    codec[AV1_FRAME_RESTORATION_TYPE + i] = filter->restoration_types[i];
    codec[AV1_LOG2_RESTORATION_UNIT_SIZE_MINUS5 + i] =
        (uint8_t)(filter->log2_restoration_unit_sizes[i] - 5);
  }
  if (picture->film_grain.apply) {
    amdf_media_drm_av1_grain(&picture->film_grain,
                             session->codec.av1.sample_bit_depth,
                             codec + AV1_FILM_GRAIN, auxiliary + 256);
  }
  uint32_t* tiles = (uint32_t*)(codec + AV1_TILE_INFO);
  for (uint32_t i = 0; i < picture->tile_column_count * picture->tile_row_count;
       ++i) {
    tiles[i * 2] = picture->tiles[i].byte_offset;
    tiles[i * 2 + 1] = picture->tiles[i].byte_length;
  }
  return amdf_media_drm_vcn_ib(storage, session, compressed_address,
                               output->addresses[0]);
}
