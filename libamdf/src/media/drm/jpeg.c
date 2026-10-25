// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/drm/jpeg.h"

typedef struct amdf_media_drm_jpeg_registers_t {
  // Per-engine decode reset register.
  uint32_t soft_reset;
  // High half of the compressed-input base address.
  uint32_t compressed_high;
  // Low half of the compressed-input base address.
  uint32_t compressed_low;
  // Luma output pitch register.
  uint32_t luma_pitch;
  // Chroma output pitch register.
  uint32_t chroma_pitch;
  // Output address interpretation register.
  uint32_t address_mode;
  // Luma output swizzle register.
  uint32_t luma_tiling;
  // Chroma output swizzle register.
  uint32_t chroma_tiling;
  // High half of the output allocation base address.
  uint32_t output_high;
  // Low half of the output allocation base address.
  uint32_t output_low;
  // Tier-two output control register.
  uint32_t tier_control;
  // Output-ring read pointer register.
  uint32_t output_read;
  // Output-ring format and scheduling register.
  uint32_t output_control;
  // Output-ring write pointer register.
  uint32_t output_write;
} amdf_media_drm_jpeg_registers_t;

static const amdf_media_drm_jpeg_registers_t
    amdf_media_drm_jpeg_indexed_registers = {
        .soft_reset = 0x402f,
        .compressed_high = 0x40e1,
        .compressed_low = 0x40e0,
        .luma_pitch = 0x401f,
        .chroma_pitch = 0x4020,
        .address_mode = 0x4027,
        .luma_tiling = 0x4024,
        .chroma_tiling = 0x4025,
        .output_high = 0x40e3,
        .output_low = 0x40e2,
        .tier_control = 0x400f,
        .output_read = 0x401e,
        .output_control = 0x401c,
        .output_write = 0x401d,
};

static const amdf_media_drm_jpeg_registers_t
    amdf_media_drm_jpeg_direct_registers = {
        .soft_reset = 0x4051,
        .compressed_high = 0x40b3,
        .compressed_low = 0x40b2,
        .luma_pitch = 0x4043,
        .chroma_pitch = 0x4044,
        .address_mode = 0x404b,
        .luma_tiling = 0x4048,
        .chroma_tiling = 0x4049,
        .output_high = 0x40b5,
        .output_low = 0x40b4,
        .tier_control = 0x400e,
        .output_read = 0x4042,
        .output_control = 0x4040,
        .output_write = 0x4041,
};

static void amdf_media_drm_jpeg_register(uint32_t** cursor, uint32_t address,
                                         uint32_t value) {
  *(*cursor)++ = address;
  *(*cursor)++ = value;
}

bool amdf_media_drm_jpeg_protocol(
    uint32_t ip_discovery_version,
    amdf_media_drm_command_protocol_t* out_protocol) {
  const uint32_t major = (ip_discovery_version >> 16) & 255;
  if (ip_discovery_version == 0x040003 || major == 5) {
    *out_protocol = AMDF_MEDIA_DRM_COMMAND_PROTOCOL_JPEG_DIRECT;
    return true;
  }
  if (major >= 2 && major <= 4) {
    *out_protocol = AMDF_MEDIA_DRM_COMMAND_PROTOCOL_JPEG_INDEXED;
    return true;
  }
  return false;
}

uint32_t amdf_media_drm_jpeg_emit(amdf_media_drm_command_protocol_t protocol,
                                  uint32_t* words, uint64_t compressed_address,
                                  uint32_t compressed_byte_length,
                                  const amdf_media_drm_frame_t* output) {
  const bool direct = protocol == AMDF_MEDIA_DRM_COMMAND_PROTOCOL_JPEG_DIRECT;
  const amdf_media_drm_jpeg_registers_t* registers =
      direct ? &amdf_media_drm_jpeg_direct_registers
             : &amdf_media_drm_jpeg_indexed_registers;
  uint32_t* cursor = words;
  // PACKETJ type 3/condition 3 waits for the masked hardware register value.
  // This per-picture decoder reset is the ordinary JPEG engine protocol, not
  // a DRM device/context reset or a device-fault recovery operation.
  amdf_media_drm_jpeg_register(&cursor, registers->soft_reset, 1);
  amdf_media_drm_jpeg_register(&cursor, 0x408e, 0x01400200);
  amdf_media_drm_jpeg_register(&cursor, 0x408f, 1u << 16);
  amdf_media_drm_jpeg_register(&cursor, 0x33000000 | registers->soft_reset,
                               1u << 16);
  amdf_media_drm_jpeg_register(&cursor, registers->soft_reset, 0);
  amdf_media_drm_jpeg_register(&cursor, 0x408f, 0);
  amdf_media_drm_jpeg_register(&cursor, 0x33000000 | registers->soft_reset,
                               1u << 16);
  // Complete JPEG bytes include marker/table headers and zero padding to 128B.
  amdf_media_drm_jpeg_register(&cursor, registers->compressed_high,
                               (uint32_t)(compressed_address >> 32));
  amdf_media_drm_jpeg_register(&cursor, registers->compressed_low,
                               (uint32_t)compressed_address);
  amdf_media_drm_jpeg_register(&cursor, 0x4001, 0);
  amdf_media_drm_jpeg_register(&cursor, 0x4004, 0xfffffff0u);
  amdf_media_drm_jpeg_register(&cursor, 0x4002, compressed_byte_length / 4);
  amdf_media_drm_jpeg_register(&cursor, registers->luma_pitch,
                               output->row_pitches[0] / 16);
  amdf_media_drm_jpeg_register(&cursor, registers->chroma_pitch,
                               output->row_pitches[1] / 16);
  // Linear luma/chroma addressing, without compression metadata.
  amdf_media_drm_jpeg_register(&cursor, registers->address_mode, 0);
  amdf_media_drm_jpeg_register(&cursor, registers->luma_tiling, 0);
  amdf_media_drm_jpeg_register(&cursor, registers->chroma_tiling, 0);
  amdf_media_drm_jpeg_register(&cursor, registers->output_high,
                               (uint32_t)(output->addresses[0] >> 32));
  amdf_media_drm_jpeg_register(&cursor, registers->output_low,
                               (uint32_t)output->addresses[0]);
  const uint32_t offsets[3] = {
      0, (uint32_t)(output->addresses[1] - output->addresses[0]), 0};
  if (direct) {
    amdf_media_drm_jpeg_register(&cursor, 0x41c0, offsets[0]);
    amdf_media_drm_jpeg_register(&cursor, 0x41c1, offsets[1]);
    amdf_media_drm_jpeg_register(&cursor, 0x41c2, offsets[2]);
    // The public JPEG schema exposes the complete coded image, not ROI crop.
    amdf_media_drm_jpeg_register(&cursor, 0x401b, 0);
    amdf_media_drm_jpeg_register(&cursor, 0x401c, 0x00010001);
    // VCN 5's format-conversion block still needs the native NV12 selector.
    amdf_media_drm_jpeg_register(&cursor, 0x4052, 0x0000ff21);
  } else {
    for (uint32_t i = 0; i < 3; ++i) {
      amdf_media_drm_jpeg_register(&cursor, 0x402c, i);
      amdf_media_drm_jpeg_register(&cursor, 0x402d, offsets[i]);
    }
  }
  amdf_media_drm_jpeg_register(&cursor, registers->tier_control, 0);
  amdf_media_drm_jpeg_register(&cursor, registers->output_read, 0);
  amdf_media_drm_jpeg_register(&cursor, registers->output_control, 0x14c7);
  amdf_media_drm_jpeg_register(&cursor, 0x400a, 0xfffffffeu);
  amdf_media_drm_jpeg_register(&cursor, 0x4000, 6);
  amdf_media_drm_jpeg_register(&cursor, 0x408f, compressed_byte_length / 4);
  amdf_media_drm_jpeg_register(&cursor, 0x408e, 0x01400200);
  amdf_media_drm_jpeg_register(&cursor, 0x33004003, UINT32_MAX);
  amdf_media_drm_jpeg_register(&cursor, 0x408f, UINT32_MAX);
  amdf_media_drm_jpeg_register(&cursor, 0x33000000 | registers->output_write,
                               1);
  amdf_media_drm_jpeg_register(&cursor, 0x4000, 4);
  while ((cursor - words) % 16 != 0) {
    amdf_media_drm_jpeg_register(&cursor, 0x60000000, 0);
  }
  return (uint32_t)(cursor - words) * 4;
}
