// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Offline artifact compilation shared by command-line tools and live runners.
//
// Target providers own family selectors and immutable profiles. Artifact
// providers consume those profiles and emit prepared target-low IR. They have
// no device discovery or runtime loading responsibilities. Live execution
// layers may project a device into an ordinary artifact target and then use the
// same provider that deterministic offline compilation uses.

#ifndef LOOM_TOOLING_COMPILE_ARTIFACT_H_
#define LOOM_TOOLING_COMPILE_ARTIFACT_H_

#include "iree/base/api.h"
#include "iree/base/byte_sequence.h"
#include "loom/ir/module.h"
#include "loom/target/pipeline_options.h"
#include "loom/target/profile.h"
#include "loom/target/provider.h"
#include "loom/target/reporting/report.h"
#include "loom/target/types.h"
#include "loom/tooling/compile/options.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_artifact_provider_t loom_artifact_provider_t;

typedef uint32_t loom_artifact_provider_flags_t;
enum loom_artifact_provider_flag_bits_e {
  // Provider is the default format for its target family.
  LOOM_ARTIFACT_PROVIDER_FLAG_CANONICAL = 1u << 0,
};

// Borrowed concrete target for emitting already-prepared target-low IR.
typedef struct loom_artifact_target_t {
  // Target-neutral bundle used to select compatible prepared entries. NULL
  // requests the module's authored target records.
  const loom_target_bundle_t* target_bundle;
  // Family-owned target selector used for diagnostics and artifact metadata.
  iree_string_view_t target_key;
} loom_artifact_target_t;

// Returns the target-neutral bundle projected by |target|, or NULL.
static inline const loom_target_bundle_t* loom_artifact_target_bundle(
    const loom_artifact_target_t* target) {
  return target ? target->target_bundle : NULL;
}

// Compiler artifact bytes produced without creating or querying a device.
typedef struct loom_artifact_t {
  // Provider-facing target key used to emit the artifact.
  iree_string_view_t target_key;
  // Durable target-neutral bundle resolved for the artifact.
  const loom_target_bundle_t* target_bundle;
  // Target-native artifact format.
  loom_target_artifact_format_t target_artifact_format;
  // Borrowed target-native artifact contents owned by |storage|.
  iree_byte_sequence_t* target_artifact_data;
  // Target-owned textual listing format, such as `amdgpu-assembly`.
  iree_string_view_t target_listing_format;
  // Borrowed textual listing contents owned by |storage|.
  iree_byte_sequence_t* target_listing_data;
  // Optional sidecar artifacts produced beside |executable_data|.
  const loom_target_emit_sidecar_artifact_t* sidecars;
  // Number of entries in |sidecars|.
  iree_host_size_t sidecar_count;
  // Borrowed primary executable contents owned by |storage|.
  iree_byte_sequence_t* executable_data;
  // Provider-owned storage released by |deinitialize_artifact|.
  void* storage;
} loom_artifact_t;

// Emits already-prepared target-low IR to a compiler artifact. This does not
// specialize source IR. When |out_emitted| is true the artifact has a target
// bundle, non-empty target-native and executable contents, and a valid
// descriptor and contents for every sidecar. Returning OK with |out_emitted|
// false is reserved for product diagnostics emitted through |diagnostic_sink|;
// infrastructure failures return a non-OK status.
typedef iree_status_t (*loom_artifact_provider_emit_fn_t)(
    const loom_artifact_provider_t* provider, loom_module_t* module,
    const loom_artifact_target_t* target, const loom_compile_options_t* options,
    iree_allocator_t allocator, bool* out_emitted,
    loom_artifact_t* out_artifact);

typedef void (*loom_artifact_provider_deinitialize_artifact_fn_t)(
    const loom_artifact_provider_t* provider, loom_artifact_t* artifact,
    iree_allocator_t allocator);

// Linked offline compiler for one target artifact family.
struct loom_artifact_provider_t {
  // Stable provider name used in diagnostics and execution tooling.
  iree_string_view_t name;
  // Public artifact format produced by this provider.
  iree_string_view_t public_artifact_format;
  // Format selection flags.
  loom_artifact_provider_flags_t flags;
  // Required target-family profile representation.
  const loom_target_profile_type_t* target_profile_type;
  // Artifact kind recorded in structured compile reports.
  loom_target_compile_artifact_kind_t artifact_kind;
  // Target-owned defaults used to prepare target-low IR before emission.
  loom_target_pipeline_options_t default_pipeline_options;
  // Emits a prepared target-low module to executable artifact bytes.
  loom_artifact_provider_emit_fn_t emit_artifact;
  // Releases storage owned by an artifact returned from |emit_artifact|.
  loom_artifact_provider_deinitialize_artifact_fn_t deinitialize_artifact;
};

// A registry of artifact providers linked into a compiler binary.
typedef struct loom_artifact_provider_registry_t {
  // Linked artifact provider table; entries are non-NULL when count is nonzero.
  const loom_artifact_provider_t* const* providers;
  // Number of entries in |providers|.
  iree_host_size_t provider_count;
} loom_artifact_provider_registry_t;

// Artifact candidate produced by an offline compiler provider.
typedef struct loom_artifact_candidate_t {
  // Host allocator used for owned candidate storage.
  iree_allocator_t host_allocator;
  // Structured report for this candidate.
  loom_target_compile_report_t compile_report;
  // Artifact provider that produced |artifact|.
  const loom_artifact_provider_t* provider;
  // True when artifact bytes were produced.
  bool compiled;
  // Artifact bytes produced by |provider|.
  loom_artifact_t artifact;
} loom_artifact_candidate_t;

// Emits prepared target-low |module| using a caller-owned explicit target.
iree_status_t loom_artifact_candidate_emit_target(
    const loom_artifact_provider_t* provider,
    const loom_artifact_target_t* target, loom_module_t* module,
    const loom_compile_options_t* options, iree_allocator_t allocator,
    loom_artifact_candidate_t* out_candidate);

// Emits |module| using its authored target records.
iree_status_t loom_artifact_candidate_emit_module_target(
    const loom_artifact_provider_t* provider, loom_module_t* module,
    const loom_compile_options_t* options, iree_allocator_t allocator,
    loom_artifact_candidate_t* out_candidate);

// Releases all artifact storage owned by |candidate|.
void loom_artifact_candidate_deinitialize(loom_artifact_candidate_t* candidate);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_COMPILE_ARTIFACT_H_
