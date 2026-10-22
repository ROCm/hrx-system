// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_MEDIA_SESSION_H_
#define AMDF_MEDIA_SESSION_H_

#include "amdf/media/device.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Native codec state bound to one queue, without a reference or output pool.
typedef struct amdf_media_session_t amdf_media_session_t;

#define AMDF_STRUCTURE_TYPE_MEDIA_SESSION_CREATE_INFO \
  ((amdf_structure_type_t)0x00040201u)
#define AMDF_STRUCTURE_TYPE_MEDIA_SESSION_INFO \
  ((amdf_structure_type_t)0x00040202u)

/// A borrowed byte range with its explicit native consumer identity. Compressed
/// input uses a linear view of its backing; its memory owner must be distinct
/// from every image backing accessed by the same media command.
typedef struct amdf_media_buffer_range_t {
  /// Direct core backing owner, retained by the application through last use.
  amdf_memory_t* memory;
  /// Exact consumer in memory's caller-ordered access list.
  uint32_t access_ordinal;
  /// Range start relative to the memory's logical byte zero.
  uint64_t byte_offset;
  /// Accessible range length in bytes.
  uint64_t byte_length;
} amdf_media_buffer_range_t;

/// Cold construction of native codec state. Creation publishes no commands.
typedef struct amdf_media_session_create_info_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_SESSION_CREATE_INFO.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Typed codec configuration, consumed during creation.
  const void* next;
  /// Exact session envelope previously admitted by the device.
  amdf_media_configuration_t configuration;
} amdf_media_session_create_info_t;

/// Immutable session facts. Native charges are reported separately from the
/// explicit image, command, input, and feedback memory owned by the caller.
typedef struct amdf_media_session_info_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_SESSION_INFO.
  amdf_structure_type_t type;
  /// Size of caller-provided result storage.
  uint32_t structure_size;
  /// Caller-provided typed outputs, or NULL.
  void* next;
  /// Borrowed queue whose order defines this session's native execution order.
  amdf_kernel_queue_t* queue;
  /// Exact realized session envelope.
  amdf_media_configuration_t configuration;
  /// Native system-pool codec-state bytes reported by the provider.
  uint64_t system_state_byte_length;
  /// Native local-pool codec-state bytes reported by the provider.
  uint64_t local_state_byte_length;
} amdf_media_session_info_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_MEDIA_SESSION_H_
