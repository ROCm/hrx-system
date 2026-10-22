// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_MEDIA_GFX11_H_
#define AMDF_MEDIA_GFX11_H_

#include "amdf/media/image.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AMDF_STRUCTURE_TYPE_MEDIA_GFX11_IMAGE_LAYOUT \
  ((amdf_structure_type_t)0x00040B01u)

/// Qualified gfx11 interpretation of one native plane for a selected consumer.
/// These are AMD ISA descriptor words, never a D3D12 descriptor handle. Their
/// addresses borrow the exact core memory access named by image_query_info.
typedef struct amdf_media_gfx11_plane_layout_t {
  /// gfx11 image-resource descriptor rebased to the consumer's GPU address.
  /// This is a single-plane view of the queried layer. Array image operations
  /// use view-relative array coordinate zero, not the underlying array_layer.
  uint32_t descriptor[8];
  /// Plane base relative to the memory object's logical byte zero, excluding
  /// address-encoded pipe/bank XOR. Tiled storage has no linear row-pitch
  /// claim.
  uint64_t byte_offset;
  /// Address-encoded pipe/bank XOR in byte-address units.
  uint32_t pipe_bank_xor;
  /// Native swizzle mode from the qualified descriptor.
  uint32_t swizzle_mode;
} amdf_media_gfx11_plane_layout_t;

/// Optional output on amdf_media_image_info_t for a qualified gfx11 consumer.
typedef struct amdf_media_gfx11_image_layout_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_GFX11_IMAGE_LAYOUT.
  amdf_structure_type_t type;
  /// Size of caller-provided result storage.
  uint32_t structure_size;
  /// Further typed outputs, or NULL.
  void* next;
  /// Number of initialized component-plane records.
  uint32_t plane_count;
  /// Selected array layer encoded in each native descriptor.
  uint32_t array_layer;
  /// Luma followed by interleaved chroma for NV12 and P010.
  amdf_media_gfx11_plane_layout_t planes[2];
} amdf_media_gfx11_image_layout_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_MEDIA_GFX11_H_
