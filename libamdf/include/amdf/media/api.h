// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_MEDIA_API_H_
#define AMDF_MEDIA_API_H_

#include "amdf/media/command.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AMDF_MEDIA_EXTENSION_VERSION_1 1u
#define AMDF_MEDIA_EXTENSION_VERSION_LATEST AMDF_MEDIA_EXTENSION_VERSION_1

/// Optional typed media operations on core devices, memory, and kernel queues.
/// Activation is independent of compute. Loading or negotiating this table
/// loads no native provider and creates no native state. Table growth appends
/// fields; covered function pointers stay valid until the library is unloaded.
///
/// Constructors are transactional and leave output storage unchanged on error.
/// Preparation instead mutates an already-live command with explicit cleanup
/// ownership. The application owns pixel/input/result backing, image placement,
/// reference scheduling, command slots, and all last-use dependencies. A frame
/// is borrowed metadata over direct amdf_memory_t storage, never an allocator.
typedef struct amdf_media_api_t {
  /// Size in bytes of this table version.
  uint32_t structure_size;
  /// Negotiated media extension version.
  uint32_t extension_version;

  /// Activates native media services, borrowing endpoint until core device
  /// destruction. Cold path: may allocate and load native system APIs. It does
  /// not activate compute or create pixel memory, a queue, or codec state.
  amdf_status_t(AMDF_CALL* device_create)(
      amdf_endpoint_t* endpoint,
      const amdf_media_device_create_info_t* create_info,
      amdf_device_t** out_device);

  /// Enumerates implemented schemas for one live family. count receives the
  /// full available count; capacity zero with codecs NULL is a count query.
  /// Insufficient nonzero capacity returns BUFFER_TOO_SMALL with count filled
  /// and the array unchanged. Cached query: no native calls or allocation.
  amdf_status_t(AMDF_CALL* device_enumerate_codecs)(
      amdf_device_t* device, uint32_t queue_family_ordinal, uint32_t capacity,
      amdf_media_codec_info_t* codecs, uint32_t* count);

  /// Answers exact native admission and storage requirements. This cold query
  /// may call the driver; it reserves no execution capacity or pixel memory.
  /// Unsupported tuples are result data; native errors leave outputs unchanged.
  amdf_status_t(AMDF_CALL* device_query_configuration)(
      amdf_device_t* device, const amdf_media_configuration_query_t* query,
      amdf_media_configuration_info_t* out_info);

  /// Answers the exact heap placement for an application-selected image shape,
  /// including array layers. Cold query: may call the driver, allocates no
  /// pixel backing, and creates no image identity. An unsupported shape returns
  /// UNSUPPORTED; all errors leave out_requirements unchanged. Codec admission
  /// remains separate from an image's storage requirements.
  amdf_status_t(AMDF_CALL* device_query_image_requirements)(
      amdf_device_t* device, const amdf_media_image_description_t* description,
      amdf_media_image_requirements_t* out_requirements);

  /// Creates a native queue with explicit capabilities and an admission bound.
  /// The core queue owns notifications, checked retirement, and destruction.
  amdf_status_t(AMDF_CALL* kernel_queue_create)(
      amdf_device_t* device,
      const amdf_media_kernel_queue_create_info_t* create_info,
      amdf_kernel_queue_t** out_queue);

  /// Creates one native identity in caller-owned heap bytes and establishes
  /// every requested consumer layout. Cold path; no pixel backing allocation.
  /// Memory and its devices remain borrowed until image destruction.
  amdf_status_t(AMDF_CALL* image_create)(
      amdf_device_t* device, const amdf_media_image_create_info_t* create_info,
      amdf_media_image_t** out_image);

  /// Copies retained layout for the exact memory access and array layer.
  /// No import, descriptor creation, native query, allocation, or device wait.
  amdf_status_t(AMDF_CALL* image_query_info)(
      amdf_media_image_t* image, uint32_t access_ordinal,
      const amdf_media_image_query_t* query, amdf_media_image_info_t* out_info);

  /// Releases one native identity after application-established last use. It
  /// neither waits nor frees its borrowed memory. NULL is a no-op.
  void(AMDF_CALL* image_destroy)(amdf_media_image_t* image);

  /// Creates native codec state for one queue and exact envelope. No image
  /// pool, recording pool, hidden BEGIN submission, or reference scheduler is
  /// created.
  amdf_status_t(AMDF_CALL* session_create)(
      amdf_kernel_queue_t* queue,
      const amdf_media_session_create_info_t* create_info,
      amdf_media_session_t** out_session);

  /// Copies immutable state/charge facts without native work or allocation.
  amdf_status_t(AMDF_CALL* session_query_info)(
      amdf_media_session_t* session, amdf_media_session_info_t* out_info);

  /// Provisions one native recording and association with caller-owned feedback
  /// storage. The command borrows its queue and supplied backing through
  /// release.
  amdf_status_t(AMDF_CALL* command_create)(
      amdf_kernel_queue_t* queue,
      const amdf_media_command_create_info_t* create_info,
      amdf_media_command_t** out_command);

  /// Records one operation in an idle or checked-retired command. Temporary CPU
  /// arrays are consumed before return; native owners remain borrowed. BUSY
  /// never waits, picks another slot, or grows a pool. Preparation can allocate
  /// inside the native recording implementation. The caller serializes access
  /// to a command and preserves session-required recording/execution order.
  amdf_status_t(AMDF_CALL* command_prepare)(
      amdf_media_session_t* session, amdf_media_command_t* command,
      const amdf_media_command_prepare_info_t* prepare_info);

  /// Publishes one prepared command. Zero accepted_submission means rejection
  /// with no native prefix accepted. A nonzero point remains owned even when
  /// the returned status fails. No library allocation or host completion wait.
  amdf_status_t(AMDF_CALL* kernel_queue_submit)(
      amdf_kernel_queue_t* queue,
      const amdf_media_kernel_queue_submission_info_t* submission_info,
      uint64_t* accepted_submission);

  /// Reads one accepted, checked-retired operation's feedback. Before checked
  /// retirement returns BUSY without waiting. A failed codec outcome is result
  /// data, not proof of queue failure or permission to ignore other last uses.
  amdf_status_t(AMDF_CALL* command_query_result)(
      amdf_media_command_t* command,
      amdf_media_command_result_info_t* out_info);

  /// Rejects unretired accepted use with BUSY and retains the owner. Otherwise
  /// releases native recording state and its session/queue borrows. NULL is an
  /// invalid argument.
  amdf_status_t(AMDF_CALL* command_destroy)(amdf_media_command_t* command);

  /// Rejects retained command borrows with BUSY. Otherwise releases native
  /// codec state without a hidden END submission. NULL is an invalid argument.
  amdf_status_t(AMDF_CALL* session_destroy)(amdf_media_session_t* session);
} amdf_media_api_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_MEDIA_API_H_
