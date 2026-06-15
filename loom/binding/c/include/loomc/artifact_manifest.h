// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_ARTIFACT_MANIFEST_H_
#define LOOMC_ARTIFACT_MANIFEST_H_

#include "loomc/base.h"

/// @file
/// Artifact manifest emission controls.
///
/// Artifact manifests are optional machine-readable sidecars for emitted target
/// artifacts. They describe the emitted artifact bytes, exported functions,
/// exported globals, target facts, and launch/call interface facts available at
/// the selected verbosity.
///
/// Manifest generation is opt-in. Omitting this descriptor, or setting
/// `mode` to `LOOMC_ARTIFACT_MANIFEST_MODE_NONE`, keeps emission on the normal
/// artifact-only path.

#ifdef __cplusplus
extern "C" {
#endif

/// Loom artifact manifest JSON report artifact format.
#define LOOMC_ARTIFACT_FORMAT_ARTIFACT_MANIFEST_JSON \
  "loom-artifact-manifest-json"

/// Artifact manifest detail mode.
typedef enum loomc_artifact_manifest_mode_e {
  /// Does not request an artifact manifest.
  LOOMC_ARTIFACT_MANIFEST_MODE_NONE = 0,

  /// Requests stable artifact, target, function, global, and summary ABI facts.
  LOOMC_ARTIFACT_MANIFEST_MODE_SUMMARY = 1,

  /// Requests summary facts plus detail arrays such as function parameters.
  LOOMC_ARTIFACT_MANIFEST_MODE_DETAILS = 2,

  /// Requests artifact-bound analysis facts when available.
  LOOMC_ARTIFACT_MANIFEST_MODE_ANALYSIS = 3,
} loomc_artifact_manifest_mode_t;

/// Artifact manifest emission options.
///
/// Attach this descriptor through `loomc_emit_options_t::next`. The descriptor
/// controls sidecar manifest production for the emitted target artifact. It
/// does not compile source, run passes, or write filesystem paths.
typedef struct loomc_artifact_manifest_options_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_ARTIFACT_MANIFEST_OPTIONS` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Next invocation option extension.
  const void* next;

  /// Selected manifest detail mode.
  loomc_artifact_manifest_mode_t mode;

  /// Result artifact identifier for the manifest JSON. Empty derives from the
  /// emitted artifact identifier by appending `.manifest.json`.
  loomc_string_view_t identifier;
} loomc_artifact_manifest_options_t;

/// Parses a stable artifact manifest mode spelling.
///
/// Accepts `""`, `"none"`, `"summary"`, `"details"`, or `"analysis"`.
LOOMC_API_EXPORT loomc_status_t loomc_artifact_manifest_mode_parse(
    loomc_string_view_t value, loomc_artifact_manifest_mode_t* out_mode);

/// Returns the stable JSON/CLI spelling for `mode`.
LOOMC_API_EXPORT loomc_string_view_t
loomc_artifact_manifest_mode_name(loomc_artifact_manifest_mode_t mode);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_ARTIFACT_MANIFEST_H_
