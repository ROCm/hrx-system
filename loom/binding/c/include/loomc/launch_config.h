// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_LAUNCH_CONFIG_H_
#define LOOMC_LAUNCH_CONFIG_H_

#include <stdbool.h>
#include <stdint.h>

#include "loomc/artifact.h"

/// @file
/// Compiled kernel launch configurations.
///
/// A launch-config program is the host companion produced by one kernel
/// compilation invocation. It contains one function for each exported kernel
/// and evaluates that kernel's authored workload arguments into the complete
/// configuration required to launch the matching compiled entry.
///
/// Launch-config programs are compiler products. Loading and invocation do
/// not parse kernel source, apply configuration, select a target, or compile
/// device code. All such decisions were made when the compiled kernel module
/// and its companion launch-config artifact were produced.
///
/// @par Example
/// Bind a compiled kernel and evaluate its launch configuration:
///
/// @code{.c}
/// loomc_launch_config_program_t* launch_program = NULL;
/// loomc_status_t status = loomc_launch_config_program_load(
///     launch_config_artifact, loomc_allocator_system(), &launch_program);
/// if (!loomc_status_is_ok(status)) return status;
///
/// loomc_launch_config_function_t launch_function =
///     loomc_launch_config_function_invalid();
/// status = loomc_launch_config_program_lookup_function(
///     launch_program, kernel_export_name, &launch_function);
/// if (!loomc_status_is_ok(status)) {
///   loomc_launch_config_program_release(launch_program);
///   return status;
/// }
///
/// const uint64_t workload_argument_bits[] = {element_count};
/// loomc_launch_config_t launch_config = {
///     .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
///     .structure_size = sizeof(launch_config),
/// };
/// status = loomc_launch_config_program_invoke(
///     launch_program, launch_function, workload_argument_bits,
///     /*workload_argument_count=*/1, &launch_config);
/// if (loomc_status_is_ok(status)) {
///   // Use launch_config to populate the target launch ABI for the compiled
///   // executable entry identified by kernel_export_name.
/// }
///
/// loomc_launch_config_program_release(launch_program);
/// return status;
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// Loaded program containing compiled kernel launch configurations.
///
/// A program corresponds to one kernel compilation invocation and may describe
/// entries later emitted across multiple executable artifacts. Function names
/// match the public export names in those artifacts; launch-config function
/// tokens and executable function tokens remain local to their respective
/// objects.
///
/// @thread_safety
/// Programs are thread-compatible. Invocation reuses program-owned scratch and
/// must not overlap another invocation on the same program. Callers requiring
/// overlapping invocation use distinct loaded programs or externally
/// synchronize access.
typedef struct loomc_launch_config_program_t loomc_launch_config_program_t;

/// Program-local launch-config function token.
///
/// Tokens are meaningful only with the program that returned them. Public
/// export names are the stable identity used to bind independently loaded
/// executable and launch-config products.
typedef struct loomc_launch_config_function_t {
  /// Program-local function value.
  uint64_t value;
} loomc_launch_config_function_t;

/// Invalid launch-config function token value.
#define LOOMC_LAUNCH_CONFIG_FUNCTION_INVALID_VALUE UINT64_MAX

/// Returns an invalid launch-config function token.
static inline loomc_launch_config_function_t
loomc_launch_config_function_invalid(void) {
  loomc_launch_config_function_t function = {
      LOOMC_LAUNCH_CONFIG_FUNCTION_INVALID_VALUE};
  return function;
}

/// Returns true when `function` is not the invalid token value.
///
/// This does not prove that the token belongs to a particular program.
static inline bool loomc_launch_config_function_is_valid(
    loomc_launch_config_function_t function) {
  return function.value != LOOMC_LAUNCH_CONFIG_FUNCTION_INVALID_VALUE;
}

/// Complete concrete configuration for one compiled kernel launch.
///
/// Callers zero-initialize this structure, set `type` to
/// `LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG`, set `structure_size` to
/// `sizeof(loomc_launch_config_t)`, and pass it to
/// `loomc_launch_config_program_invoke`.
typedef struct loomc_launch_config_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG` when
  /// nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Unordered extension chain for additional launch properties.
  void* next;

  /// Number of workgroups dispatched along each dimension.
  loomc_dimension3_t workgroup_count;

  /// Number of invocations in each workgroup along each dimension.
  ///
  /// Every dimension is nonzero after successful evaluation.
  loomc_dimension3_t workgroup_size;

  /// Number of workgroups in each cooperative cluster dimension.
  ///
  /// Kernels that do not use clustered launch report `{1, 1, 1}`.
  loomc_dimension3_t workgroup_cluster_size;

  /// Number of invocations in each hardware subgroup.
  ///
  /// Zero indicates that the compiled target has no fixed subgroup size.
  uint32_t subgroup_size;

  /// Total workgroup-local storage required per workgroup, in bytes.
  ///
  /// This is the final total capacity required by the compiled kernel,
  /// including storage derived from workload facts during compilation. Target
  /// adapters map this total requirement to their launch ABI; it is not
  /// necessarily an API's additional dynamic-shared-memory argument.
  uint64_t workgroup_storage_bytes;
} loomc_launch_config_t;

/// Loads and verifies a compiled launch-config program.
///
/// The artifact must be the launch-config companion emitted from the same
/// kernel compilation used to produce the matching executable artifacts. The
/// loader validates the external bytes and constructs a program ready for
/// invocation.
///
/// @param artifact Launch-config artifact to load. The artifact descriptor and
/// its strings are borrowed for the duration of the call.
/// @param allocator Host allocator used for program-owned state.
/// @param out_program Receives one retained program on success.
/// @return OK when the artifact is valid and every exported function has a
/// complete launch-config contract.
///
/// @ownership
/// The caller releases the returned program with
/// `loomc_launch_config_program_release`.
LOOMC_API_EXPORT loomc_status_t loomc_launch_config_program_load(
    const loomc_artifact_t* artifact, loomc_allocator_t allocator,
    loomc_launch_config_program_t** out_program);

/// Retains `program` for another owner.
///
/// @param program Program to retain.
LOOMC_API_EXPORT void loomc_launch_config_program_retain(
    loomc_launch_config_program_t* program);

/// Releases `program` from one owner. Passing `NULL` is allowed.
///
/// @param program Program to release. Passing `NULL` is allowed.
LOOMC_API_EXPORT void loomc_launch_config_program_release(
    loomc_launch_config_program_t* program);

/// Resolves a compiled kernel export name to a program-local function token.
///
/// @param program Program to search.
/// @param export_name Exact public name of the matching executable entry.
/// A leading `@` is not accepted because artifact export names are not Loom
/// symbol references.
/// @param out_function Receives a program-local token on success.
/// @return OK when the function exists, `LOOMC_STATUS_NOT_FOUND` when the
/// program has no matching function, or an argument status for malformed
/// input.
LOOMC_API_EXPORT loomc_status_t loomc_launch_config_program_lookup_function(
    const loomc_launch_config_program_t* program,
    loomc_string_view_t export_name,
    loomc_launch_config_function_t* out_function);

/// Invokes one compiled launch-config function for one kernel workload.
///
/// Workload arguments map positionally to the selected kernel entry's authored
/// launch workload. Each slot contains the raw bits of one scalar argument in
/// its least-significant N bits, where N is the scalar bit width; higher bits
/// are ignored. Signed integers and index values use two's-complement
/// representation, offsets use unsigned representation, and floating-point
/// values use their declared FP8, IEEE, or bfloat representation. Index and
/// offset arguments consume all 64 bits. An offset greater than `INT64_MAX`
/// cannot be represented by the current exact-fact domain and is rejected.
///
/// The function token must have been returned by `program`. Invocation
/// validates argument count, scalar values, and authored value predicates
/// before writing `out_config`. A successful return provides every launch
/// property required by the compiled kernel entry. The operation performs no
/// source compilation or string lookup. The program caches invocation scratch;
/// an invocation may acquire allocator storage when the selected function
/// requires more scratch than previous invocations.
///
/// @param program Loaded launch-config program.
/// @param function Program-local function to invoke.
/// @param workload_argument_bits Positional raw scalar argument bits, or
/// `NULL` when `workload_argument_count` is zero.
/// @param workload_argument_count Number of argument slots.
/// @param out_config Caller-initialized complete result storage.
/// @return OK when invocation produced a complete launch configuration.
LOOMC_API_EXPORT loomc_status_t
loomc_launch_config_program_invoke(loomc_launch_config_program_t* program,
                                   loomc_launch_config_function_t function,
                                   const uint64_t* workload_argument_bits,
                                   loomc_host_size_t workload_argument_count,
                                   loomc_launch_config_t* out_config);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_LAUNCH_CONFIG_H_
