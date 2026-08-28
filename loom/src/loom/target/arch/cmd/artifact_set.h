// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Complete in-memory products for prepared portable command programs.

#ifndef LOOM_TARGET_ARCH_CMD_ARTIFACT_SET_H_
#define LOOM_TARGET_ARCH_CMD_ARTIFACT_SET_H_

#include "iree/base/api.h"
#include "loom/target/arch/cmd/lower/program_plan.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Stable public format name for one serialized portable command program.
#define LOOM_CMD_PROGRAM_ARTIFACT_FORMAT "loom-command"

// File extension convention for a serialized portable command program.
#define LOOM_CMD_PROGRAM_ARTIFACT_EXTENSION ".loomcmd"

// Stable format name for a portable command artifact-set manifest.
#define LOOM_CMD_PROGRAM_ARTIFACT_SET_MANIFEST_FORMAT "loom-command-set"

// Current portable command artifact-set manifest schema version.
#define LOOM_CMD_PROGRAM_ARTIFACT_SET_SCHEMA_VERSION 2

// Maximum storage required by a canonical root artifact filename.
#define LOOM_CMD_PROGRAM_ARTIFACT_FILENAME_CAPACITY 64

// Maximum storage required by a canonical kernel request filename.
#define LOOM_CMD_KERNEL_REQUEST_FILENAME_CAPACITY 64

// One plan-wide executable-entry requirement.
typedef struct loom_cmd_program_artifact_entry_t {
  // Logical kernel symbol resolved by the embedding executable catalog.
  iree_string_view_t symbol;

  // True when the plan published an ordinary Loom source request artifact.
  bool has_source_request;
} loom_cmd_program_artifact_entry_t;

// One serialized command root and its plan-wide entry projection.
typedef struct loom_cmd_program_artifact_t {
  // Root symbol without an '@' sigil.
  iree_string_view_t symbol;

  // Complete portable command-program bytes.
  iree_byte_span_t data;

  // Plan-wide entry indices in root-local executable/entry slot order.
  const uint32_t* entry_requirement_indices;

  // Number of entries in |entry_requirement_indices|.
  uint32_t entry_requirement_count;
} loom_cmd_program_artifact_t;

// Owned serialized roots and their shared executable-entry requirements.
typedef struct loom_cmd_program_artifact_set_t {
  // Serialized command roots in caller order.
  struct {
    // Owned artifact table.
    loom_cmd_program_artifact_t* values;
    // Number of entries in |values|.
    iree_host_size_t count;
  } programs;

  // Unique plan-wide executable-entry requirements.
  struct {
    // Owned entry requirement table.
    loom_cmd_program_artifact_entry_t* values;
    // Number of entries in |values|.
    iree_host_size_t count;
  } entries;

  // Contiguous root-local entry projection storage.
  uint32_t* entry_requirement_index_storage;

  // Contiguous storage borrowed by every symbol string view.
  char* string_storage;

  // Host allocator used for all owned storage.
  iree_allocator_t host_allocator;
} loom_cmd_program_artifact_set_t;

// Serializes every root and snapshots its external entry requirements.
//
// The returned set retains no module, operation, value, or symbol storage and
// may outlive |plan|. Artifact bytes and metadata are owned by the set and must
// be released with loom_cmd_program_artifact_set_deinitialize.
iree_status_t loom_cmd_program_artifact_set_build(
    const loom_cmd_program_plan_t* plan,
    loom_cmd_program_artifact_set_t* out_artifact_set,
    iree_allocator_t host_allocator);

// Releases all storage owned by |artifact_set| and resets it to empty.
void loom_cmd_program_artifact_set_deinitialize(
    loom_cmd_program_artifact_set_t* artifact_set);

// Formats the canonical relative artifact filename for |program_ordinal|.
//
// The returned view borrows |buffer| and excludes its trailing NUL.
iree_status_t loom_cmd_program_artifact_format_filename(
    iree_host_size_t program_ordinal, iree_host_size_t buffer_capacity,
    char* buffer, iree_string_view_t* out_filename);

// Formats the canonical relative request filename for |entry_ordinal|.
//
// The returned view borrows |buffer| and excludes its trailing NUL.
iree_status_t loom_cmd_kernel_request_format_filename(
    iree_host_size_t entry_ordinal, iree_host_size_t buffer_capacity,
    char* buffer, iree_string_view_t* out_filename);

// Writes the schema-versioned artifact-set manifest as JSON.
//
// Program artifact filenames use loom_cmd_program_artifact_format_filename and
// are relative to the manifest's embedding-owned artifact directory. Source
// request filenames use loom_cmd_kernel_request_format_filename and are
// relative to the embedding-owned kernel request directory.
iree_status_t loom_cmd_program_artifact_set_format_manifest_json(
    const loom_cmd_program_artifact_set_t* artifact_set,
    loom_output_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_ARTIFACT_SET_H_
