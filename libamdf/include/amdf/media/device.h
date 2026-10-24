// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_MEDIA_DEVICE_H_
#define AMDF_MEDIA_DEVICE_H_

#include "amdf/media/image.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AMDF_STRUCTURE_TYPE_MEDIA_DEVICE_CREATE_INFO \
  ((amdf_structure_type_t)0x00040001u)
#define AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_QUERY \
  ((amdf_structure_type_t)0x00040002u)
#define AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_INFO \
  ((amdf_structure_type_t)0x00040003u)
#define AMDF_STRUCTURE_TYPE_MEDIA_KERNEL_QUEUE_CREATE_INFO \
  ((amdf_structure_type_t)0x00040004u)

typedef uint32_t amdf_media_codec_t;
enum amdf_media_codec_e {
  AMDF_MEDIA_CODEC_UNKNOWN = 0,
  AMDF_MEDIA_CODEC_H264 = 1,
  AMDF_MEDIA_CODEC_HEVC = 2,
  /// Eight-bit baseline sequential 4:2:0 JPEG to NV12. The complete input
  /// includes its marker headers; no parsed-picture extension is required.
  AMDF_MEDIA_CODEC_JPEG = 3,
  AMDF_MEDIA_CODEC_VP9 = 4,
  AMDF_MEDIA_CODEC_AV1 = 5,
};

typedef uint32_t amdf_media_direction_t;
enum amdf_media_direction_e {
  AMDF_MEDIA_DIRECTION_DECODE = 1,
  AMDF_MEDIA_DIRECTION_ENCODE = 2,
};

/// Activates an independently owned media execution domain on an endpoint.
/// This does not activate compute, create a queue, or allocate image storage.
typedef struct amdf_media_device_create_info_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_DEVICE_CREATE_INFO.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Typed activation extensions, or NULL.
  const void* next;
} amdf_media_device_create_info_t;

/// An implemented codec schema on one live execution family. Exact tuple
/// support is established separately by device_query_configuration.
typedef struct amdf_media_codec_info_t {
  /// Parsed codec schema understood by preparation.
  amdf_media_codec_t codec;
  /// Decode or encode direction on this family.
  amdf_media_direction_t direction;
} amdf_media_codec_info_t;

/// Session envelope. Storage and visible crop are not inferred from one
/// another.
typedef struct amdf_media_configuration_t {
  /// Parsed codec schema.
  amdf_media_codec_t codec;
  /// Requested operation direction.
  amdf_media_direction_t direction;
  /// Uncompressed component storage format.
  amdf_media_pixel_format_t format;
  /// Exact bitstream-coded width in pixels, before the consumer's render crop.
  /// Native storage padding is reported separately by configuration admission.
  uint32_t coded_width;
  /// Exact bitstream-coded height in pixels, before the consumer's render crop.
  /// Native storage padding is reported separately by configuration admission.
  uint32_t coded_height;
  /// Number of addressable reference slots, including the current reconstructed
  /// target whether it shares the decoded output or has separate storage.
  uint32_t reference_image_count;
} amdf_media_configuration_t;

typedef uint32_t amdf_media_configuration_support_t;
enum amdf_media_configuration_support_e {
  AMDF_MEDIA_CONFIGURATION_SUPPORT_UNSUPPORTED = 0,
  AMDF_MEDIA_CONFIGURATION_SUPPORT_SUPPORTED = 1,
};

typedef uint32_t amdf_media_configuration_reason_t;
enum amdf_media_configuration_reason_e {
  AMDF_MEDIA_CONFIGURATION_REASON_NONE = 0,
  AMDF_MEDIA_CONFIGURATION_REASON_CODEC = 1,
  AMDF_MEDIA_CONFIGURATION_REASON_FORMAT = 2,
  AMDF_MEDIA_CONFIGURATION_REASON_DIMENSIONS = 3,
  AMDF_MEDIA_CONFIGURATION_REASON_REFERENCE_STORAGE = 4,
  AMDF_MEDIA_CONFIGURATION_REASON_NATIVE_CONFIGURATION = 5,
};

/// Exact admission query. No dimension is padded, clamped, or silently changed.
typedef struct amdf_media_configuration_query_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_QUERY.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Typed codec configuration, or NULL when the codec needs no extra controls.
  const void* next;
  /// Ordinal from this live device's queue-family enumeration.
  uint32_t queue_family_ordinal;
  /// Complete requested session envelope.
  amdf_media_configuration_t configuration;
} amdf_media_configuration_query_t;

/// Exact admission and placement requirements. Unsupported is an answered
/// query; native-query failure is a failed call that leaves output storage
/// unchanged.
typedef struct amdf_media_configuration_info_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_INFO.
  amdf_structure_type_t type;
  /// Size of caller-provided result storage.
  uint32_t structure_size;
  /// Caller-provided typed outputs, or NULL.
  void* next;
  /// Whether this exact configuration is supported.
  amdf_media_configuration_support_t support;
  /// Unsupported predicate, or NONE for a supported tuple.
  amdf_media_configuration_reason_t reason;
  /// The queried envelope, unchanged by admission.
  amdf_media_configuration_t configuration;
  /// Nonzero when this session needs explicit BEGIN before its first picture
  /// and END after its last picture. Creation and destruction submit neither.
  uint32_t requires_begin_end;
  /// Borrowed core scope for application-owned image heaps. Include this media
  /// device and every required peer in the core profile query/access set.
  amdf_memory_scope_t* image_heap_memory_scope;
  /// Core scope-local profile for application-owned image heaps.
  uint32_t image_heap_memory_profile_ordinal;
  /// Borrowed core scope for compressed input or output bytes; may equal the
  /// image-heap scope.
  amdf_memory_scope_t* compressed_memory_scope;
  /// Core scope-local profile for compressed input or output backing.
  uint32_t compressed_memory_profile_ordinal;
  /// Borrowed core scope for application-owned host-readable result storage,
  /// or NULL when feedback_byte_length is zero.
  amdf_memory_scope_t* feedback_memory_scope;
  /// Core scope-local profile for result storage, with this media device in
  /// the explicit access set. A profile query qualifies any additional peers.
  uint32_t feedback_memory_profile_ordinal;
  /// Required alignment of compressed input or output offsets.
  uint64_t compressed_byte_offset_alignment;
  /// Required alignment of complete compressed input or output extents.
  uint64_t compressed_byte_length_alignment;
  /// Minimum accessible compressed range for each picture. Decode reports zero
  /// because the stream supplies its input extent; encode reports the complete
  /// native writable bound that must remain owned through retirement.
  uint64_t minimum_compressed_byte_length;
  /// Feedback bytes provisioned for one reusable recording owner.
  uint64_t feedback_byte_length;
  /// Required alignment of the feedback range within its memory.
  uint64_t feedback_byte_offset_alignment;
  /// Placement requirements for one uncompressed encode input, or zero
  /// byte_length for decode.
  amdf_media_image_requirements_t input;
  /// Placement requirements for one independently consumable decoded output.
  /// Encode reports zero byte_length.
  amdf_media_image_requirements_t output;
  /// Native reconstructed-reference placement. Decode describes one image, or
  /// zero byte_length when output carries that role. Encode describes the
  /// complete application-owned array containing every reconstruction slot.
  amdf_media_image_requirements_t reference;
} amdf_media_configuration_info_t;

/// Explicit creation of one media KMQ; no session or recording is implied.
typedef struct amdf_media_kernel_queue_create_info_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_KERNEL_QUEUE_CREATE_INFO.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Typed queue extensions, or NULL.
  const void* next;
  /// Ordinal from this live device's queue-family enumeration.
  uint32_t queue_family_ordinal;
  /// Capabilities that must be provisioned, including native dependencies when
  /// EXTERNAL_SYNCHRONIZATION is requested.
  amdf_kernel_queue_capabilities_t required_capabilities;
  /// Nonzero bound on accepted points that may remain unretired. This admission
  /// bound neither reserves driver capacity nor allocates a recording pool.
  uint32_t maximum_pending_submission_count;
  /// Native dependency operands provisioned per publication, or zero for the
  /// provider default. The realized bound is reported by
  /// kernel_queue_query_info.
  uint32_t maximum_wait_count;
} amdf_media_kernel_queue_create_info_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_MEDIA_DEVICE_H_
