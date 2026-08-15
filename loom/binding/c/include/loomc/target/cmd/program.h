// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_CMD_PROGRAM_H_
#define LOOMC_TARGET_CMD_PROGRAM_H_

#include "loomc/artifact.h"

/// @file
/// Loaded portable command-program packages and their runtime ABI.
///
/// One package contains several named command-program roots. Loading validates
/// the complete external byte representation once and preserves immutable
/// indexed views for parameter placement, executable entry resolution, and
/// runtime materialization. It performs no source compilation or target
/// selection.

#ifdef __cplusplus
extern "C" {
#endif

/// Portable single-root command-program artifact format.
#define LOOMC_ARTIFACT_FORMAT_COMMAND_PROGRAM "loom-cmd"

/// Portable multi-root command-program package artifact format.
#define LOOMC_ARTIFACT_FORMAT_COMMAND_PACKAGE "loom-cmd-package"

/// Immutable loaded multi-root command-program package.
///
/// @thread_safety
/// Packages are immutable after loading. Retained handles may be queried and
/// used for concurrent materialization.
typedef struct loomc_cmd_program_package_t loomc_cmd_program_package_t;

/// Package-local command-program export token.
typedef struct loomc_cmd_program_export_t {
  /// Dense value in the owning package's export table.
  uint64_t value;
} loomc_cmd_program_export_t;

/// Invalid command-program export token value.
#define LOOMC_CMD_PROGRAM_EXPORT_INVALID_VALUE UINT64_MAX

/// Invalid fixed-buffer or issue-time binding index.
#define LOOMC_CMD_PROGRAM_BINDING_INVALID UINT32_MAX

/// Returns an invalid command-program export token.
static inline loomc_cmd_program_export_t loomc_cmd_program_export_invalid(
    void) {
  loomc_cmd_program_export_t program_export = {
      LOOMC_CMD_PROGRAM_EXPORT_INVALID_VALUE,
  };
  return program_export;
}

/// Returns true when `program_export` is not the invalid token value.
///
/// This does not prove that the token belongs to a particular package.
static inline bool loomc_cmd_program_export_is_valid(
    loomc_cmd_program_export_t program_export) {
  return program_export.value != LOOMC_CMD_PROGRAM_EXPORT_INVALID_VALUE;
}

/// One aggregate buffer requirement in a command-program ABI.
typedef struct loomc_cmd_program_buffer_requirement_t {
  /// Dense fixed-buffer or issue-time binding index, or
  /// `LOOMC_CMD_PROGRAM_BINDING_INVALID` when the storage is absent.
  uint32_t binding_index;

  /// Minimum required byte length of the supplied buffer range.
  uint64_t required_byte_length;

  /// Minimum alignment required by the portable command representation.
  uint64_t minimum_alignment;
} loomc_cmd_program_buffer_requirement_t;

/// Immutable runtime ABI metadata for one command-program export.
typedef struct loomc_cmd_program_info_t {
  /// Structure type. May be `LOOMC_STRUCTURE_TYPE_NONE` when zero-initialized.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  void* next;

  /// Public export name borrowed from the package.
  loomc_string_view_t name;

  /// Number of materialization-time fixed buffer slots.
  loomc_host_size_t fixed_buffer_count;

  /// Number of issue-time rebindable buffer slots.
  loomc_host_size_t rebindable_binding_count;

  /// Number of root-local live executable slots.
  loomc_host_size_t executable_count;

  /// Number of executable entry requirements.
  loomc_host_size_t entry_count;

  /// Number of fixed parameter-buffer root requirements.
  loomc_host_size_t parameter_root_count;

  /// Number of concrete immutable parameter requirements.
  loomc_host_size_t parameter_count;

  /// Number of portable commands recorded by the root.
  loomc_host_size_t command_count;

  /// Packed issue-time transient slab requirement.
  loomc_cmd_program_buffer_requirement_t transient;

  /// Issue-time command config-data requirement.
  ///
  /// The exact byte length is authoritative. Runtime adapters may impose a
  /// stronger placement alignment without changing the payload capacity.
  loomc_cmd_program_buffer_requirement_t config;
} loomc_cmd_program_info_t;

/// Immutable metadata for one fixed parameter-buffer root.
typedef struct loomc_cmd_program_parameter_root_info_t {
  /// Structure type. May be `LOOMC_STRUCTURE_TYPE_NONE` when zero-initialized.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  void* next;

  /// Dense materialization-time fixed-buffer index populated by this root.
  uint32_t fixed_buffer_index;

  /// Minimum byte length required by all parameters assigned to the root.
  uint64_t required_byte_length;

  /// Minimum alignment required for the supplied fixed-buffer range.
  uint64_t minimum_alignment;
} loomc_cmd_program_parameter_root_info_t;

/// Immutable metadata for one concrete command-program parameter.
typedef struct loomc_cmd_program_parameter_info_t {
  /// Structure type. May be `LOOMC_STRUCTURE_TYPE_NONE` when zero-initialized.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  void* next;

  /// Fully substituted parameter key borrowed from the package.
  loomc_string_view_t key;

  /// Dense fixed-buffer index containing the parameter.
  uint32_t fixed_buffer_index;

  /// Root-relative byte offset of the parameter payload.
  uint64_t byte_offset;

  /// Exact byte length of the parameter payload.
  uint64_t byte_length;

  /// Minimum required alignment of the parameter payload.
  uint64_t minimum_alignment;
} loomc_cmd_program_parameter_info_t;

/// Immutable executable-entry association for one command-program export.
typedef struct loomc_cmd_program_entry_info_t {
  /// Structure type. May be `LOOMC_STRUCTURE_TYPE_NONE` when zero-initialized.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  void* next;

  /// Root-local executable slot providing the entry.
  uint32_t executable_index;

  /// Public entry name borrowed from the package.
  loomc_string_view_t name;
} loomc_cmd_program_entry_info_t;

/// Loads and validates one portable multi-root command-program package.
///
/// `artifact` must be an executable artifact in
/// `LOOMC_ARTIFACT_FORMAT_COMMAND_PACKAGE` format. Loading validates the
/// package and every nested command program exactly once.
///
/// @param artifact Package artifact to load. Its descriptor and strings are
/// borrowed for the duration of the call.
/// @param release Optional callback transferring ownership of
/// `artifact->contents` to the package. On success it runs when the final
/// package reference is released; on failure it runs before this call returns.
/// Without a callback the loader copies the package bytes.
/// @param release_user_data Value passed to `release`.
/// @param allocator Host allocator used for package-owned state.
/// @param out_package Receives one retained package on success.
/// @return OK when the complete artifact is valid.
LOOMC_API_EXPORT loomc_status_t loomc_cmd_program_package_load(
    const loomc_artifact_t* artifact, loomc_artifact_release_fn_t release,
    void* release_user_data, loomc_allocator_t allocator,
    loomc_cmd_program_package_t** out_package);

/// Retains `package` for another owner.
LOOMC_API_EXPORT void loomc_cmd_program_package_retain(
    loomc_cmd_program_package_t* package);

/// Releases `package` from one owner. Passing `NULL` is allowed.
LOOMC_API_EXPORT void loomc_cmd_program_package_release(
    loomc_cmd_program_package_t* package);

/// Returns the number of command-program exports, or zero for `NULL`.
LOOMC_API_EXPORT loomc_host_size_t loomc_cmd_program_package_export_count(
    const loomc_cmd_program_package_t* package);

/// Returns the token at `index`, or an invalid token when out of range.
LOOMC_API_EXPORT loomc_cmd_program_export_t loomc_cmd_program_package_export_at(
    const loomc_cmd_program_package_t* package, loomc_host_size_t index);

/// Looks up a command-program export by canonical public name.
///
/// A leading `@` is not accepted. `out_export` receives an invalid token on
/// failure.
LOOMC_API_EXPORT loomc_status_t loomc_cmd_program_package_lookup_export(
    const loomc_cmd_program_package_t* package, loomc_string_view_t name,
    loomc_cmd_program_export_t* out_export);

/// Returns the runtime ABI for one command-program export.
LOOMC_API_EXPORT loomc_status_t loomc_cmd_program_package_export_info(
    const loomc_cmd_program_package_t* package,
    loomc_cmd_program_export_t program_export,
    loomc_cmd_program_info_t* out_info);

/// Returns one executable-entry association by root-local entry index.
LOOMC_API_EXPORT loomc_status_t loomc_cmd_program_package_entry_info(
    const loomc_cmd_program_package_t* package,
    loomc_cmd_program_export_t program_export, loomc_host_size_t index,
    loomc_cmd_program_entry_info_t* out_info);

/// Returns one fixed parameter-buffer root requirement by dense index.
LOOMC_API_EXPORT loomc_status_t loomc_cmd_program_package_parameter_root_info(
    const loomc_cmd_program_package_t* package,
    loomc_cmd_program_export_t program_export, loomc_host_size_t index,
    loomc_cmd_program_parameter_root_info_t* out_info);

/// Returns one concrete immutable parameter requirement by dense index.
LOOMC_API_EXPORT loomc_status_t loomc_cmd_program_package_parameter_info(
    const loomc_cmd_program_package_t* package,
    loomc_cmd_program_export_t program_export, loomc_host_size_t index,
    loomc_cmd_program_parameter_info_t* out_info);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_CMD_PROGRAM_H_
