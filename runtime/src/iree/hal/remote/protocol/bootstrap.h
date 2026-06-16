// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL remote bootstrap metadata carried in iree_net_session_topology_t
// application data.

#ifndef IREE_HAL_REMOTE_PROTOCOL_BOOTSTRAP_H_
#define IREE_HAL_REMOTE_PROTOCOL_BOOTSTRAP_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_device_spec_t iree_hal_device_spec_t;

// Magic value for iree_hal_remote_bootstrap_device_catalog_header_t.
#define IREE_HAL_REMOTE_BOOTSTRAP_DEVICE_CATALOG_MAGIC 0x52484443u

// Current device catalog payload version.
#define IREE_HAL_REMOTE_BOOTSTRAP_DEVICE_CATALOG_VERSION 1u

// Device catalog flags.
typedef uint32_t iree_hal_remote_bootstrap_device_catalog_flags_t;
typedef enum iree_hal_remote_bootstrap_device_catalog_flag_bits_e {
  IREE_HAL_REMOTE_BOOTSTRAP_DEVICE_CATALOG_FLAG_NONE = 0u,
} iree_hal_remote_bootstrap_device_catalog_flag_bits_t;

// Per-device catalog entry flags.
typedef uint32_t iree_hal_remote_bootstrap_device_spec_entry_flags_t;
typedef enum iree_hal_remote_bootstrap_device_spec_entry_flag_bits_e {
  IREE_HAL_REMOTE_BOOTSTRAP_DEVICE_SPEC_ENTRY_FLAG_NONE = 0u,
} iree_hal_remote_bootstrap_device_spec_entry_flag_bits_t;

// Header for a device catalog advertised during session bootstrap.
typedef struct iree_hal_remote_bootstrap_device_catalog_header_t {
  // Must be IREE_HAL_REMOTE_BOOTSTRAP_DEVICE_CATALOG_MAGIC.
  uint32_t magic;
  // Must be IREE_HAL_REMOTE_BOOTSTRAP_DEVICE_CATALOG_VERSION.
  uint16_t version;
  // Reserved for iree_hal_remote_bootstrap_device_catalog_flags_t.
  uint16_t flags;
  // Number of iree_hal_remote_bootstrap_device_spec_entry_t records following.
  uint32_t device_count;
  // Must be zero.
  uint32_t reserved;
} iree_hal_remote_bootstrap_device_catalog_header_t;
static_assert(sizeof(iree_hal_remote_bootstrap_device_catalog_header_t) == 16,
              "");

// Entry describing one serialized HAL device spec within the catalog.
typedef struct iree_hal_remote_bootstrap_device_spec_entry_t {
  // Server-local device ordinal.
  uint32_t device_ordinal;
  // Reserved for iree_hal_remote_bootstrap_device_spec_entry_flags_t.
  uint32_t flags;
  // Offset from the start of the catalog to the serialized spec bytes.
  uint64_t spec_offset;
  // Length in bytes of the serialized spec image.
  uint64_t spec_length;
} iree_hal_remote_bootstrap_device_spec_entry_t;
static_assert(sizeof(iree_hal_remote_bootstrap_device_spec_entry_t) == 24, "");

// Selects serialized spec bytes for |device_ordinal| from |catalog_bytes|.
//
// The returned span aliases |catalog_bytes| and remains valid only while the
// catalog storage remains valid.
IREE_API_EXPORT iree_status_t
iree_hal_remote_bootstrap_device_catalog_select_spec(
    iree_const_byte_span_t catalog_bytes, uint32_t device_ordinal,
    iree_const_byte_span_t* out_spec_bytes);

// Parses the serialized spec for |device_ordinal| from |catalog_bytes|.
//
// The returned spec is retained for the caller and must be released with
// iree_hal_device_spec_release.
IREE_API_EXPORT iree_status_t
iree_hal_remote_bootstrap_device_catalog_parse_spec(
    iree_const_byte_span_t catalog_bytes, uint32_t device_ordinal,
    iree_allocator_t host_allocator, iree_hal_device_spec_t** out_spec);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_PROTOCOL_BOOTSTRAP_H_
