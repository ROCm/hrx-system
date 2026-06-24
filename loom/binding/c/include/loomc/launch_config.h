// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_LAUNCH_CONFIG_H_
#define LOOMC_LAUNCH_CONFIG_H_

#include <stdint.h>

#include "loomc/config.h"
#include "loomc/module.h"

/// @file
/// Kernel launch configuration evaluation.
///
/// Launch configuration evaluation answers the host-side question "which launch
/// parameters resolve to concrete values under this invocation config?" It is a
/// cold-path convenience API for embedders that compile kernel shape sets and
/// launch the resulting executable themselves. The result uses ordinary Loom C
/// API versioned structs and presence flags; serialized launch config artifacts
/// use a separate compact binary format.

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
/// `sizeof(loomc_launch_config_t)`, and pass it to
/// `loomc_module_evaluate_launch_config`. Fields are meaningful only
/// when the corresponding `fields` bit is present.
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

/// Launch configuration evaluation options.
///
/// Callers zero-initialize this descriptor, set `type` to
/// `LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_EVAL_OPTIONS`, set
/// `structure_size` to `sizeof(loomc_launch_config_eval_options_t)`,
/// and fill the requested fields.
typedef struct loomc_launch_config_eval_options_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_EVAL_OPTIONS` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Extension chain for future launch configuration evaluation options.
  const void* next;

  /// Kernel function symbol to evaluate, with or without a leading `@`.
  loomc_string_view_t function_symbol;

  /// Per-invocation config dialect bindings and optional JSON object.
  loomc_config_options_t config;

  /// Workload argument values supplied as signed 64-bit integers.
  ///
  /// Values map positionally to the kernel launch-config region arguments.
  /// Scalar index, offset, and integer arguments are seeded as exact value
  /// facts. Unsupported argument types produce a failed result diagnostic.
  const int64_t* workload_arguments;

  /// Number of entries in `workload_arguments`.
  loomc_host_size_t workload_argument_count;

  /// Fields that must be present for the result to succeed.
  ///
  /// Missing required fields are reported as result diagnostics. Missing fields
  /// that are not required are simply absent from
  /// `loomc_launch_config_t::fields`.
  loomc_launch_config_field_flags_t required_fields;
} loomc_launch_config_eval_options_t;

/// Evaluates a kernel's launch configuration under config bindings.
///
/// @param module Module containing the kernel. It is not mutated.
/// @param workspace Scratch workspace used for the internal cloned module,
/// config materialization, target fact storage, and value-fact analysis.
/// @param options Evaluation options. `function_symbol` is required.
/// @param allocator Host allocator used for returned result storage.
/// @param out_config Receives evaluated launch config fields.
/// @param out_result Receives a retained result for diagnostics.
/// @return OK when evaluation ran to a result. Non-OK statuses represent API
/// misuse or infrastructure failures before a result could be produced.
///
/// @ownership
/// The caller owns `out_result` on an OK return and releases it with
/// `loomc_result_release`.
LOOMC_API_EXPORT loomc_status_t loomc_module_evaluate_launch_config(
    const loomc_module_t* module, loomc_workspace_t* workspace,
    const loomc_launch_config_eval_options_t* options,
    loomc_allocator_t allocator, loomc_launch_config_t* out_config,
    loomc_result_t** out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_LAUNCH_CONFIG_H_
