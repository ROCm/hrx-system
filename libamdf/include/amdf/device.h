// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_DEVICE_H_
#define AMDF_DEVICE_H_

#include "amdf/base.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Immutable identity and execution families of one activated device.
///
/// An endpoint identifies physical hardware. Each extension activates its own
/// device implementation on that hardware, with its own identity and families.
/// Family ordinals select services on this device, not on another activation
/// of the same endpoint. Query these families before creating queues or naming
/// device execution sites in memory visibility queries.
typedef struct amdf_device_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_DEVICE_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_device_info_t)`.
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are currently defined.
  void* next;
  /// Opaque identity of this live materialized device.
  amdf_device_id_t id;
  /// Provider epoch in which this activation's resources remain valid.
  uint64_t reset_epoch;
  /// Extension whose constructor created this device implementation.
  amdf_extension_id_t extension_id;
  /// Number of immutable families returned by device_query_queue_family_info.
  uint32_t queue_family_count;
} amdf_device_info_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_DEVICE_H_
