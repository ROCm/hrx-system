// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/drm/vcn.h"

#include <string.h>

// The firmware ABI consists of little-endian DWORD messages. These offsets
// are the VCN decode protocol, independent of host C layout and pointer width.
// Every reserved word remains zero. See ac_vcn_dec.h in the native reference.
enum {
  MESSAGE_HEADER_WORDS = 6,
  MESSAGE_INDEX_WORDS = 4,
  DECODE_WORDS = 47,
  DYNAMIC_WORDS = 42,
  AVC_BYTES = 1124,
  ENGINE_PACKAGE_BYTES = 16,
  DECODE_PACKAGE_BYTES = 140,
  INDIRECT_BUFFER_BYTES = ENGINE_PACKAGE_BYTES + DECODE_PACKAGE_BYTES,
  SESSION_CONTEXT_BYTES = 131072,
  STREAM_H264 = 7,
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

static void amdf_media_drm_vcn_address(uint32_t* words, uint64_t address) {
  words[0] = (uint32_t)(address >> 32);
  words[1] = (uint32_t)address;
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
    buffers[BINDING_FLAGS] |=
        0xa0c;  // Bitstream, target, collocated state, and scaling.
    amdf_media_drm_vcn_address(buffers + BINDING_OUTPUT, output);
    amdf_media_drm_vcn_address(buffers + BINDING_COMPRESSED, compressed);
    amdf_media_drm_vcn_address(
        buffers + BINDING_CONTEXT,
        session->storage->address + SESSION_CONTEXT_BYTES);
    amdf_media_drm_vcn_address(
        buffers + BINDING_SCALING,
        storage->address + AMDF_MEDIA_DRM_VCN_SCALING_OFFSET);
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
    message[10] = STREAM_H264;  // H264 performance decode.
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

uint32_t amdf_media_drm_vcn_h264(
    amdf_media_drm_buffer_t* storage, const amdf_media_drm_session_t* session,
    uint64_t compressed_address, uint32_t compressed_byte_length,
    const amdf_media_drm_frame_t* output, const amdf_media_drm_frame_t* current,
    const amdf_media_drm_frame_t* references, uint32_t current_slot,
    const amdf_media_h264_decode_picture_info_t* picture) {
  uint32_t* message = (uint32_t*)((uint8_t*)storage->host_pointer +
                                  AMDF_MEDIA_DRM_VCN_MESSAGE_OFFSET);
  const uint32_t decode_offset = MESSAGE_HEADER_WORDS + 3 * MESSAGE_INDEX_WORDS;
  const uint32_t dynamic_offset = decode_offset + DECODE_WORDS;
  const uint32_t codec_offset = dynamic_offset + DYNAMIC_WORDS;
  const uint32_t byte_length = codec_offset * 4 + AVC_BYTES;
  memset(message, 0, byte_length);
  message[0] = 40;
  message[1] = byte_length;
  message[2] = 3;
  message[3] = 1;  // DECODE.
  message[4] = session->stream_handle;
  amdf_media_drm_vcn_index(message + 6, 2, decode_offset * 4, DECODE_WORDS * 4);
  amdf_media_drm_vcn_index(message + 10, 0x10, dynamic_offset * 4,
                           DYNAMIC_WORDS * 4);
  amdf_media_drm_vcn_index(message + 14, 6, codec_offset * 4, AVC_BYTES);
  uint32_t* decode = message + decode_offset;
  decode[DECODE_STREAM] = STREAM_H264;  // H264 performance decode.
  decode[DECODE_FLAGS] = 1;  // Explicit dynamic reference surfaces (tier 2).
  decode[DECODE_WIDTH] = session->requirements.configuration.coded_width;
  decode[DECODE_HEIGHT] = session->requirements.configuration.coded_height;
  decode[DECODE_COMPRESSED_BYTES] = compressed_byte_length;
  decode[DECODE_OUTPUT_BYTES] =
      (uint32_t)(output->addresses[1] - output->addresses[0]) +
      output->row_pitches[1] * output->row_counts[1];
  decode[DECODE_HARDWARE_CONTEXT_BYTES] = session->context_byte_length;
  decode[DECODE_SESSION_CONTEXT_BYTES] = SESSION_CONTEXT_BYTES;
  decode[DECODE_BUFFER_FLAGS] =
      0xa0d;  // Message, compressed, output, context, and scaling.
  decode[DECODE_REFERENCE_PITCH] = current->row_pitches[0];
  decode[DECODE_REFERENCE_ROWS] = current->row_counts[0];
  const amdf_media_drm_device_t* device =
      (const amdf_media_drm_device_t*)session->base.queue->device;
  const uint32_t version =
      device->engines[session->base.queue->info.queue_family_ordinal]
          .information.ip_discovery_version;
  decode[DECODE_REFERENCE_ARRAY_MODE] =
      version == 0x040003 ? 1 : 3;  // Native address-library selector.
  decode[DECODE_OUTPUT_PITCH] = output->row_pitches[0];
  decode[DECODE_OUTPUT_CHROMA_PITCH] =
      output->row_pitches[1] / 2;  // UV pitch counts interleaved pairs.
  decode[DECODE_OUTPUT_ARRAY_MODE] = decode[DECODE_REFERENCE_ARRAY_MODE];
  decode[DECODE_OUTPUT_CHROMA_OFFSET] =
      (uint32_t)(output->addresses[1] - output->addresses[0]);
  decode[DECODE_WRITE_COHERENCY] =
      1;  // Media write coherency on VCN 3 and newer.
  decode[DECODE_REFERENCE_CHROMA_PITCH] = current->row_pitches[1] / 2;
  uint32_t* dynamic = message + dynamic_offset;
  dynamic[DYNAMIC_LUMA_PITCH] = current->row_pitches[0];
  dynamic[DYNAMIC_LUMA_ROWS] = current->row_counts[0];
  dynamic[DYNAMIC_LUMA_BYTES] =
      current->row_pitches[0] * current->row_counts[0];
  dynamic[DYNAMIC_CHROMA_PITCH] = current->row_pitches[1] / 2;
  dynamic[DYNAMIC_CHROMA_ROWS] = current->row_counts[1];
  dynamic[DYNAMIC_CHROMA_BYTES] =
      current->row_pitches[1] * current->row_counts[1];
  dynamic[DYNAMIC_REFERENCE_COUNT] = picture->reference_count;
  dynamic[DYNAMIC_CURRENT_LOW] = (uint32_t)current->addresses[0];
  dynamic[DYNAMIC_CURRENT_HIGH] = (uint32_t)(current->addresses[0] >> 32);
  for (uint32_t i = 0; i < picture->reference_count; ++i) {
    dynamic[DYNAMIC_REFERENCE_LOW + i] = (uint32_t)references[i].addresses[0];
    dynamic[DYNAMIC_REFERENCE_HIGH + i] =
        (uint32_t)(references[i].addresses[0] >> 32);
  }
  amdf_media_drm_vcn_avc((uint8_t*)(message + codec_offset), &session->h264,
                         current_slot, picture);
  // VCN consumes raster scaling matrices directly; DXVA's zigzag transform is
  // a property of that separate native interface, not of the public schema.
  memcpy((uint8_t*)storage->host_pointer + AMDF_MEDIA_DRM_VCN_SCALING_OFFSET,
         picture->scaling_matrices, sizeof(*picture->scaling_matrices));
  return amdf_media_drm_vcn_ib(storage, session, compressed_address,
                               output->addresses[0]);
}
