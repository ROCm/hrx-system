// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_LAUNCH_CONFIG_MODULE_H_
#define LOOMC_LAUNCH_CONFIG_MODULE_H_

#include <stdbool.h>
#include <stdint.h>

#include "loomc/artifact.h"
#include "loomc/launch_config.h"

/// @file
/// Prepared runtime launch-configuration modules.
///
/// Launch modules load compiler-produced functions extracted after source
/// configuration and specialization. Reusable contexts map changing workload
/// arguments to packed workgroup counts without retaining or replaying the
/// source compilation.

#ifdef __cplusplus
extern "C" {
#endif

/// Immutable loaded launch-configuration module.
///
/// A module owns a verified runtime representation and all function metadata.
/// It also owns any artifact storage retained according to the load options.
///
/// @thread_safety
/// Modules are immutable after loading and may be shared by evaluation
/// contexts and queried concurrently.
typedef struct loomc_launch_config_module_t loomc_launch_config_module_t;

/// Mutable reusable state for launch-configuration evaluation.
///
/// @thread_safety
/// A context is exclusive mutable state and must not be entered concurrently.
/// Independent contexts may evaluate functions from the same immutable module
/// concurrently without synchronization.
typedef struct loomc_launch_config_context_t loomc_launch_config_context_t;

/// Module-local launch-configuration function token.
///
/// Tokens are meaningful only with the module that returned them. Public
/// function names are the stable identity across artifact loads.
typedef struct loomc_launch_config_function_t {
  /// Module-local function table value.
  uint64_t value;
} loomc_launch_config_function_t;

/// Invalid launch-configuration function token value.
#define LOOMC_LAUNCH_CONFIG_FUNCTION_INVALID_VALUE UINT64_MAX

/// Returns an invalid launch-configuration function token.
static inline loomc_launch_config_function_t
loomc_launch_config_function_invalid(void) {
  loomc_launch_config_function_t function = {
      LOOMC_LAUNCH_CONFIG_FUNCTION_INVALID_VALUE};
  return function;
}

/// Returns a launch-configuration function token for a dense module index.
static inline loomc_launch_config_function_t
loomc_launch_config_function_from_index(uint32_t index) {
  loomc_launch_config_function_t function = {index};
  return function;
}

/// Returns true when `function` contains a valid token value.
static inline bool loomc_launch_config_function_is_valid(
    loomc_launch_config_function_t function) {
  return function.value != LOOMC_LAUNCH_CONFIG_FUNCTION_INVALID_VALUE;
}

/// Launch-configuration module loading options.
///
/// Callers zero-initialize this descriptor, set `type` to
/// `LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_MODULE_LOAD_OPTIONS`, set
/// `structure_size` to
/// `sizeof(loomc_launch_config_module_load_options_t)`, and select the artifact
/// storage policy. Zero initialization borrows artifact bytes for the call.
typedef struct loomc_launch_config_module_load_options_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_MODULE_LOAD_OPTIONS` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Unordered extension chain for representation-specific loader options.
  const void* next;

  /// Storage policy for artifact contents.
  ///
  /// Borrowed contents need only remain valid for the duration of the call.
  /// Copied contents permit a backing representation to retain a private copy.
  /// External contents transfer ownership to the loader and are released once
  /// the backing representation no longer needs them.
  loomc_source_storage_t storage;

  /// Callback used when `storage` is `LOOMC_SOURCE_STORAGE_EXTERNAL`.
  loomc_source_release_fn_t release;

  /// User data passed to `release`.
  void* release_user_data;
} loomc_launch_config_module_load_options_t;

/// Immutable metadata for one launch-configuration function.
///
/// Each result is one tightly packed `loomc_dimension3_t` workgroup count.
/// Several dispatches may reference the same result when compiler tuple
/// interning proves their launch calculations identical.
typedef struct loomc_launch_config_function_info_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_FUNCTION_INFO` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Extension chain for future function metadata.
  void* next;

  /// Public function name borrowed from the loaded module.
  loomc_string_view_t name;

  /// Number of positional scalar workload arguments.
  loomc_host_size_t workload_argument_count;

  /// Number of unique workgroup-count results.
  loomc_host_size_t result_count;

  /// Required caller-provided output storage length in bytes.
  loomc_host_size_t output_byte_length;

  /// Required output storage alignment in bytes.
  loomc_host_size_t output_alignment;
} loomc_launch_config_function_info_t;

/// Positional invocation arguments for a prepared launch function.
typedef struct loomc_launch_config_arguments_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_ARGUMENTS`
  /// when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Unordered extension chain for future argument representations.
  const void* next;

  /// Positional scalar workload argument bit patterns.
  ///
  /// Each slot carries one argument using the scalar type in the prepared
  /// function signature. The value occupies the least-significant N bits of
  /// the slot, where N is the scalar bit width; higher bits are ignored.
  /// Signed integer and index values use two's-complement representation,
  /// offset values use unsigned representation, and floating-point values use
  /// their declared IEEE or FP8 representation. Index and offset values use all
  /// 64 bits; offsets above `INT64_MAX` are outside the current fact domain.
  const uint64_t* workload_argument_bits;

  /// Number of entries in `workload_argument_bits`.
  loomc_host_size_t workload_argument_count;
} loomc_launch_config_arguments_t;

/// Writable packed output storage for prepared launch evaluation.
typedef struct loomc_launch_config_outputs_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_OUTPUTS` when
  /// nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Extension chain for future output representations.
  void* next;

  /// Caller-owned writable output storage.
  ///
  /// The storage contains `result_count` tightly packed
  /// `loomc_dimension3_t` values and may point directly into a persistently
  /// mapped host-local/device-visible indirect-parameter buffer.
  uint8_t* storage;

  /// Number of writable bytes available in `storage`.
  loomc_host_size_t storage_length;
} loomc_launch_config_outputs_t;

/// Loads and verifies a runtime launch-configuration artifact.
///
/// @param artifact Launch-configuration artifact to load. The initial runtime
/// representation accepts `LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE`; future
/// representations may use other formats without changing this API.
/// @param options Artifact storage ownership, or `NULL` to borrow the contents
/// for the duration of the call.
/// @param allocator Host allocator used for all module-owned storage.
/// @param out_module Receives one retained immutable module on success.
/// @return OK when the artifact was loaded and every exported launch function
/// satisfied the runtime ABI.
///
/// @ownership
/// The returned module owns its materialized representation and does not borrow
/// artifact bytes beyond the lifetime selected by `options`. External contents
/// transfer to this call after options validation and are released exactly once
/// before return or the module's final release, including when loading fails.
/// The caller releases the module with `loomc_launch_config_module_release`.
LOOMC_API_EXPORT loomc_status_t loomc_launch_config_module_load(
    const loomc_artifact_t* artifact,
    const loomc_launch_config_module_load_options_t* options,
    loomc_allocator_t allocator, loomc_launch_config_module_t** out_module);

/// Retains `module` for another owner.
LOOMC_API_EXPORT void loomc_launch_config_module_retain(
    loomc_launch_config_module_t* module);

/// Releases `module` from one owner. Passing `NULL` is allowed.
LOOMC_API_EXPORT void loomc_launch_config_module_release(
    loomc_launch_config_module_t* module);

/// Returns the number of exported launch functions in `module`.
///
/// Dense indices in `[0, function_count)` may be converted to tokens with
/// `loomc_launch_config_function_from_index` for enumeration. Function indices
/// are local to one loaded module; only public names are stable across loads.
LOOMC_API_EXPORT loomc_host_size_t loomc_launch_config_module_function_count(
    const loomc_launch_config_module_t* module);

/// Queries immutable metadata for `function`.
///
/// Returned string views borrow from `module` and remain valid until its final
/// release.
LOOMC_API_EXPORT loomc_status_t loomc_launch_config_module_function_info(
    const loomc_launch_config_module_t* module,
    loomc_launch_config_function_t function,
    loomc_launch_config_function_info_t* out_info);

/// Resolves a public function name to a module-local token.
LOOMC_API_EXPORT loomc_status_t
loomc_launch_config_module_lookup_function_by_name(
    const loomc_launch_config_module_t* module, loomc_string_view_t name,
    loomc_launch_config_function_t* out_function);

/// Creates reusable exclusive evaluation state for `module`.
///
/// Context creation prepares reusable representation-specific invocation
/// state. Evaluation performs no parsing, source configuration, source
/// specialization, string lookup, or global locking.
LOOMC_API_EXPORT loomc_status_t loomc_launch_config_context_create(
    loomc_launch_config_module_t* module, loomc_allocator_t allocator,
    loomc_launch_config_context_t** out_context);

/// Retains `context` for another owner.
///
/// Retention does not make concurrent evaluation legal; all owners must still
/// serialize use of this mutable context.
LOOMC_API_EXPORT void loomc_launch_config_context_retain(
    loomc_launch_config_context_t* context);

/// Releases `context` from one owner. Passing `NULL` is allowed.
LOOMC_API_EXPORT void loomc_launch_config_context_release(
    loomc_launch_config_context_t* context);

/// Evaluates `function` directly into caller-provided packed storage.
///
/// The function token must come from the context's module. Invocation values
/// are checked against the authored scalar domains and predicates captured in
/// the artifact. Evaluation never allocates output storage.
LOOMC_API_EXPORT loomc_status_t loomc_launch_config_context_evaluate(
    loomc_launch_config_context_t* context,
    loomc_launch_config_function_t function,
    const loomc_launch_config_arguments_t* arguments,
    loomc_launch_config_outputs_t* outputs);

/// Evaluates a single-result launch function into a launch configuration.
///
/// This is a convenience projection of `loomc_launch_config_context_evaluate`.
/// It rejects aggregate functions and reports only the dynamic workgroup count;
/// other dispatch properties are fixed when the matching executable is
/// compiled.
LOOMC_API_EXPORT loomc_status_t loomc_launch_config_context_evaluate_one(
    loomc_launch_config_context_t* context,
    loomc_launch_config_function_t function,
    const loomc_launch_config_arguments_t* arguments,
    loomc_launch_config_t* out_config);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_LAUNCH_CONFIG_MODULE_H_
