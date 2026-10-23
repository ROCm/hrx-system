// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_MEDIA_IMAGE_H_
#define AMDF_MEDIA_IMAGE_H_

#include "amdf/amdf.h"

#ifdef __cplusplus
extern "C" {
#endif

/// One native image identity borrowing application-owned core memory.
typedef struct amdf_media_image_t amdf_media_image_t;

#define AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_CREATE_INFO \
  ((amdf_structure_type_t)0x00040101u)
#define AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_QUERY \
  ((amdf_structure_type_t)0x00040102u)
#define AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_INFO \
  ((amdf_structure_type_t)0x00040103u)
#define AMDF_STRUCTURE_TYPE_MEDIA_FRAME ((amdf_structure_type_t)0x00040104u)
#define AMDF_STRUCTURE_TYPE_MEDIA_LINEAR_IMAGE_LAYOUT \
  ((amdf_structure_type_t)0x00040105u)

/// Component storage, independent of transfer function and native tiling.
typedef uint32_t amdf_media_pixel_format_t;
enum amdf_media_pixel_format_e {
  AMDF_MEDIA_PIXEL_FORMAT_UNKNOWN = 0,
  /// Eight-bit Y followed by interleaved U,V at half width and height.
  AMDF_MEDIA_PIXEL_FORMAT_NV12 = 1,
  /// Ten-bit samples in the most significant bits of sixteen-bit NV12 words.
  AMDF_MEDIA_PIXEL_FORMAT_P010 = 2,
};

/// Native uses established when an image identity is constructed.
typedef uint32_t amdf_media_image_roles_t;
enum amdf_media_image_role_bits_e {
  /// Compute-readable decoded pixels, independent of decoder reference storage.
  AMDF_MEDIA_IMAGE_ROLE_DECODE_OUTPUT = 1u << 0,
  /// Decoder-private reference storage; raw consumers are not implied.
  AMDF_MEDIA_IMAGE_ROLE_DECODE_REFERENCE = 1u << 1,
};

/// Immutable native image shape. Construction never adjusts these dimensions.
typedef struct amdf_media_image_description_t {
  /// Component format, including bit depth and subsampling.
  amdf_media_pixel_format_t format;
  /// Full-resolution storage width in pixels.
  uint32_t width;
  /// Full-resolution storage height in pixels.
  uint32_t height;
  /// Number of independently addressed array layers.
  uint32_t array_layer_count;
  /// Native uses requested for this identity.
  amdf_media_image_roles_t roles;
} amdf_media_image_description_t;

/// Exact placement requirements for one immutable native image description.
typedef struct amdf_media_image_requirements_t {
  /// Shape whose complete native storage is described below.
  amdf_media_image_description_t description;
  /// Required contiguous byte extent inside the application's heap.
  uint64_t byte_length;
  /// Power-of-two alignment of its heap-relative placement.
  uint64_t byte_offset_alignment;
} amdf_media_image_requirements_t;

/// Cold construction of an image identity, without allocating pixel backing.
typedef struct amdf_media_image_create_info_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_CREATE_INFO.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Typed input extensions, or NULL.
  const void* next;
  /// Borrowed backing that remains alive through image destruction.
  amdf_memory_t* memory;
  /// Access ordinal naming the constructing media device in memory.
  uint32_t media_access_ordinal;
  /// Immutable shape and uses to establish exactly.
  amdf_media_image_description_t description;
  /// Placement relative to the memory object's logical byte zero.
  uint64_t byte_offset;
  /// Number of additional memory accesses needing a qualified native layout.
  uint32_t layout_access_count;
  /// Borrowed access ordinals, consumed during this call. No lazy imports
  /// occur.
  const uint32_t* layout_access_ordinals;
} amdf_media_image_create_info_t;

/// Selects one array layer for a retained-layout query.
typedef struct amdf_media_image_query_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_QUERY.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Typed query extensions, or NULL.
  const void* next;
  /// Layer within the image's immutable array extent.
  uint32_t array_layer;
} amdf_media_image_query_t;

/// Copies immutable identity and placement facts without native work.
typedef struct amdf_media_image_info_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_INFO.
  amdf_structure_type_t type;
  /// Size of caller-provided result storage.
  uint32_t structure_size;
  /// Caller-provided typed layout outputs, or NULL.
  void* next;
  /// Direct borrowed backing owner; no memory wrapper is introduced.
  amdf_memory_t* memory;
  /// Exact immutable image and placement requirements.
  amdf_media_image_requirements_t requirements;
  /// Placement relative to the memory object's logical byte zero.
  uint64_t byte_offset;
} amdf_media_image_info_t;

/// Borrowed linear plane storage. Native tiled images use their image identity.
typedef struct amdf_media_frame_plane_t {
  /// Direct backing owner, borrowed through accepted use.
  amdf_memory_t* memory;
  /// Exact consumer in memory's caller-ordered access list.
  uint32_t access_ordinal;
  /// Plane start relative to the memory's logical byte zero.
  uint64_t byte_offset;
  /// Accessible byte extent, including allocated row padding.
  uint64_t byte_length;
  /// Byte distance between consecutive rows.
  uint64_t row_pitch;
} amdf_media_frame_plane_t;

/// Optional image_query_info output for a native linear image. Plane offsets
/// include image placement and the queried array layer, relative to the direct
/// memory owner. The returned access ordinal is the query's exact consumer.
typedef struct amdf_media_linear_image_layout_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_LINEAR_IMAGE_LAYOUT.
  amdf_structure_type_t type;
  /// Size of caller-provided result storage.
  uint32_t structure_size;
  /// Further typed outputs, or NULL.
  void* next;
  /// Number of initialized component-plane records.
  uint32_t plane_count;
  /// Layer represented by the plane offsets.
  uint32_t array_layer;
  /// Luma followed by interleaved chroma for NV12 and P010.
  amdf_media_frame_plane_t planes[2];
} amdf_media_linear_image_layout_t;

typedef uint32_t amdf_media_frame_storage_kind_t;
enum amdf_media_frame_storage_kind_e {
  AMDF_MEDIA_FRAME_STORAGE_KIND_LINEAR_PLANES = 1,
  AMDF_MEDIA_FRAME_STORAGE_KIND_NATIVE_IMAGE = 2,
};

/// A borrowed image operand. Copying this description acquires no resources.
typedef struct amdf_media_frame_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_FRAME.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Typed frame metadata, or NULL.
  const void* next;
  /// Component format, including bit depth and subsampling.
  amdf_media_pixel_format_t format;
  /// Full-resolution storage width in pixels.
  uint32_t width;
  /// Full-resolution storage height in pixels.
  uint32_t height;
  /// Selects the active storage member.
  amdf_media_frame_storage_kind_t storage_kind;
  /// Direct borrowed backing and its interpretation.
  union {
    struct {
      /// Number of planes required by the component format.
      uint32_t plane_count;
      /// Caller-owned plane array, consumed during preparation.
      const amdf_media_frame_plane_t* planes;
    } linear;
    struct {
      /// Direct backing owner, identical to the image's backing.
      amdf_memory_t* memory;
      /// Placement identical to the image's heap-relative offset.
      uint64_t byte_offset;
      /// Borrowed native identity; owns no pixel allocation.
      amdf_media_image_t* image;
      /// Selected layer within the native image.
      uint32_t array_layer;
    } image;
  } storage;
} amdf_media_frame_t;

/// Application-authored ownership change for potentially aliased image bytes.
typedef struct amdf_media_image_activation_t {
  /// Prior identity, or NULL for first activation. Its last use is retired or
  /// covered by an explicit submission dependency supplied by the application.
  amdf_media_image_t* before;
  /// Identity activated before this command's media access.
  amdf_media_image_t* after;
} amdf_media_image_activation_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_MEDIA_IMAGE_H_
