// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_LAUNCH_CONFIG_H_
#define LOOMC_LAUNCH_CONFIG_H_

#include <stdint.h>

#include "loomc/module.h"

/// @file
/// Evaluated kernel launch configuration values.

#ifdef __cplusplus
extern "C" {
#endif

/// Launch configuration fields that may be present after evaluation.
typedef enum loomc_launch_config_field_flag_bits_e {
  /// `loomc_launch_config_t::workgroup_count` is present.
  LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT = 1u << 0,

  /// `loomc_launch_config_t::workgroup_size` is present.
  LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE = 1u << 1,

  /// `loomc_launch_config_t::subgroup_size` is present.
  LOOMC_LAUNCH_CONFIG_FIELD_FLAG_SUBGROUP_SIZE = 1u << 2,

  /// `loomc_launch_config_t::workgroup_storage_bytes` is present.
  LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_STORAGE_BYTES = 1u << 3,
} loomc_launch_config_field_flag_bits_t;

/// Bitmask of `loomc_launch_config_field_flag_bits_t` values.
typedef uint32_t loomc_launch_config_field_flags_t;

/// Evaluated launch configuration.
///
/// Callers zero-initialize this structure, set `type` to
/// `LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG`, set `structure_size` to
/// `sizeof(loomc_launch_config_t)`, and pass it to an API that returns a launch
/// configuration. Fields are meaningful only when the corresponding `fields`
/// bit is present.
typedef struct loomc_launch_config_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG` when
  /// nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Extension chain for future launch configuration result payloads.
  void* next;

  /// Present evaluated fields.
  loomc_launch_config_field_flags_t fields;

  /// Optional concrete workgroup count.
  loomc_dimension3_t workgroup_count;

  /// Optional concrete local workgroup size.
  loomc_dimension3_t workgroup_size;

  /// Optional concrete subgroup size.
  uint32_t subgroup_size;

  /// Optional concrete workgroup-local storage byte count.
  uint64_t workgroup_storage_bytes;
} loomc_launch_config_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_LAUNCH_CONFIG_H_
