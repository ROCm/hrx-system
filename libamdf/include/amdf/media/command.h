// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_MEDIA_COMMAND_H_
#define AMDF_MEDIA_COMMAND_H_

#include "amdf/media/session.h"

#ifdef __cplusplus
extern "C" {
#endif

/// One application-selected reusable native recording and result association.
typedef struct amdf_media_command_t amdf_media_command_t;

#define AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_CREATE_INFO \
  ((amdf_structure_type_t)0x00040301u)
#define AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_PREPARE_INFO \
  ((amdf_structure_type_t)0x00040302u)
#define AMDF_STRUCTURE_TYPE_MEDIA_DECODE_PICTURE_INFO \
  ((amdf_structure_type_t)0x00040303u)
#define AMDF_STRUCTURE_TYPE_MEDIA_KERNEL_QUEUE_SUBMISSION_INFO \
  ((amdf_structure_type_t)0x00040304u)
#define AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_RESULT_INFO \
  ((amdf_structure_type_t)0x00040305u)

/// Native recording capacity and explicit result storage, provisioned cold.
typedef struct amdf_media_command_create_info_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_CREATE_INFO.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Typed provisioning extensions, or NULL.
  const void* next;
  /// Maximum complete slices accepted by one preparation.
  uint32_t maximum_slice_count;
  /// Maximum reference bindings accepted by one preparation.
  uint32_t maximum_reference_count;
  /// Maximum explicit image activations accepted by one preparation.
  uint32_t maximum_activation_count;
  /// Caller-owned feedback range satisfying the exact configuration query.
  amdf_media_buffer_range_t feedback;
} amdf_media_command_create_info_t;

/// Shared operands of a decode picture. Native input framing and alignment come
/// from configuration admission; incompatible bytes need application repacking.
typedef struct amdf_media_decode_picture_info_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_DECODE_PICTURE_INFO.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Codec-specific parsed picture description, or NULL for complete JPEG
  /// input.
  const void* next;
  /// Borrowed compressed bytes; preparation performs no readback or copy.
  amdf_media_buffer_range_t compressed;
  /// Borrowed independently consumable decoded output.
  const amdf_media_frame_t* output;
  /// Separate reconstructed target when required by configuration, otherwise
  /// NULL. Its reference lifetime is controlled by the application.
  const amdf_media_frame_t* reference_output;
  /// Application-owned slot for this reconstructed picture, less than the
  /// session's reference_image_count. Every later reference to this picture
  /// preserves this index, including when its position in a reference list
  /// changes. This also identifies output when no separate reference is needed.
  uint32_t reference_slot_index;
} amdf_media_decode_picture_info_t;

typedef uint32_t amdf_media_command_operation_t;
enum amdf_media_command_operation_e {
  /// Starts a session whose configuration requires_begin_end is nonzero.
  /// Submission is accepted only once, before any picture in that session.
  AMDF_MEDIA_COMMAND_OPERATION_BEGIN = 1,
  AMDF_MEDIA_COMMAND_OPERATION_PICTURE = 2,
  /// Finishes a begun session. No further operation may be submitted to it.
  /// Command retirement still precedes recording and session destruction.
  AMDF_MEDIA_COMMAND_OPERATION_END = 3,
};

/// Prepares one previously retired recording. Temporary arrays are consumed in
/// this call. Native recording failure leaves cleanup with this live command;
/// it never returns a partially constructed owner or selects another slot.
typedef struct amdf_media_command_prepare_info_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_PREPARE_INFO.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Typed preparation extensions, or NULL.
  const void* next;
  /// Explicit native operation; unsupported operations reject before recording.
  amdf_media_command_operation_t operation;
  /// Number of application-authored image ownership changes.
  uint32_t activation_count;
  /// Borrowed activation array, consumed before this command's media accesses.
  const amdf_media_image_activation_t* activations;
  /// Required decode operands for a decode PICTURE operation, otherwise NULL.
  const amdf_media_decode_picture_info_t* decode;
} amdf_media_command_prepare_info_t;

/// Publishes a prepared recording and already-realized native dependencies.
/// This path allocates no library state and performs no codec lowering, import,
/// payload access, or host completion wait. An accepted native prefix returns a
/// nonzero point even if a later native operation fails; it retains every
/// operand until checked retirement. Failure alone never grants reuse.
typedef struct amdf_media_kernel_queue_submission_info_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_KERNEL_QUEUE_SUBMISSION_INFO.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Typed publication extensions, or NULL.
  const void* next;
  /// Prepared recording owned by the caller and bound to this queue.
  amdf_media_command_t* command;
  /// Number of already-accepted producer points to wait for natively.
  uint32_t wait_count;
  /// Borrowed native bindings; future or wrong-consumer points are rejected.
  const amdf_kernel_queue_dependency_wait_t* waits;
} amdf_media_kernel_queue_submission_info_t;

typedef uint32_t amdf_media_codec_result_t;
enum amdf_media_codec_result_e {
  AMDF_MEDIA_CODEC_RESULT_UNAVAILABLE = 0,
  AMDF_MEDIA_CODEC_RESULT_SUCCEEDED = 1,
  AMDF_MEDIA_CODEC_RESULT_FAILED = 2,
};

/// A retired command's media outcome. Codec failure can retire normally and is
/// separate from queue failure. Native success does not independently validate
/// the caller's parsed metadata or certify pixel correctness.
typedef struct amdf_media_command_result_info_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_RESULT_INFO.
  amdf_structure_type_t type;
  /// Size of caller-provided result storage.
  uint32_t structure_size;
  /// Caller-provided typed result extensions, or NULL.
  void* next;
  /// Explicit operation that produced this result.
  amdf_media_command_operation_t operation;
  /// Accepted queue point whose checked retirement covers this recording.
  uint64_t submission;
  /// Codec outcome supplied by native execution, or UNAVAILABLE.
  amdf_media_codec_result_t codec_result;
} amdf_media_command_result_info_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_MEDIA_COMMAND_H_
