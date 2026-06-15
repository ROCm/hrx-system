// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL artifact provider contract for Loom execution tools.
//
// The shared execution layer owns HAL device setup, executable loading,
// dispatch, binding/result handling, correctness checking, and measurement.
// Linked artifact providers contribute only the target-family facts needed to
// specialize a module for an active device and the compiler-owned emission path
// that turns a prepared target-low Loom module into bytes accepted by the
// production HAL executable loader.

#ifndef LOOM_TOOLING_EXECUTION_HAL_ARTIFACT_H_
#define LOOM_TOOLING_EXECUTION_HAL_ARTIFACT_H_

#include "iree/base/api.h"
#include "loom/error/diagnostic.h"
#include "loom/ir/module.h"
#include "loom/target/compile_report.h"
#include "loom/target/pipeline_options.h"
#include "loom/target/provider.h"
#include "loom/target/types.h"
#include "loom/tooling/execution/compile_options.h"
#include "loom/verify/verify.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_run_hal_artifact_provider_t
    loom_run_hal_artifact_provider_t;
struct loom_run_hal_runtime_t;

// Target facts selected from one HAL runtime device.
typedef struct loom_run_hal_device_target_t {
  // Provider-owned target payload. Usually points at static target info. NULL
  // requests emission from the module's target records without a runtime
  // processor override.
  const void* data;
  // Per-device target-bundle storage owned by this selection. If a selection
  // containing this storage is copied, the copy must rebind the storage before
  // publishing |target_bundle|.
  loom_target_bundle_storage_t target_storage;
  // Target-neutral bundle resolved for the selected device target.
  const loom_target_bundle_t* target_bundle;
  // Provider-facing target key selected for emission and diagnostics, if any.
  iree_string_view_t target_key;
} loom_run_hal_device_target_t;

// Loadable HAL artifact bytes ready for iree_hal_executable_cache_prepare.
typedef struct loom_run_hal_artifact_t {
  // HAL executable format string consumed by the selected HAL loader.
  iree_string_view_t executable_format;
  // Target-neutral bundle resolved for the artifact, when available.
  const loom_target_bundle_t* target_bundle;
  // Target-native artifact format.
  loom_target_artifact_format_t target_artifact_format;
  // Target-native artifact bytes.
  iree_const_byte_span_t target_artifact_data;
  // Target-owned textual listing format, such as `amdgpu-assembly`.
  iree_string_view_t target_listing_format;
  // Target-owned textual listing bytes for debug artifact bundles.
  iree_const_byte_span_t target_listing_data;
  // Optional sidecar artifacts produced beside executable_data.
  const loom_target_emit_sidecar_artifact_t* sidecars;
  // Number of entries in |sidecars|.
  iree_host_size_t sidecar_count;
  // Provider-owned executable bytes passed to the HAL loader.
  iree_const_byte_span_t executable_data;
  // Provider-owned storage released by |deinitialize_artifact|.
  void* storage;
} loom_run_hal_artifact_t;

typedef iree_status_t (*loom_run_hal_select_device_target_fn_t)(
    const loom_run_hal_artifact_provider_t* provider,
    const struct loom_run_hal_runtime_t* runtime, iree_allocator_t allocator,
    loom_run_hal_device_target_t* out_target);

typedef iree_status_t (*loom_run_hal_select_target_key_fn_t)(
    const loom_run_hal_artifact_provider_t* provider,
    iree_string_view_t target_key, iree_allocator_t allocator,
    loom_run_hal_device_target_t* out_target);

typedef void (*loom_run_hal_deinitialize_device_target_fn_t)(
    const loom_run_hal_artifact_provider_t* provider,
    loom_run_hal_device_target_t* target, iree_allocator_t allocator);

// Emits a loader-ready HAL artifact. Returning OK with |out_emitted| false is
// reserved for product diagnostics emitted through |diagnostic_sink|; provider
// contract failures return a non-OK status.
typedef iree_status_t (*loom_run_hal_emit_artifact_fn_t)(
    const loom_run_hal_artifact_provider_t* provider, loom_module_t* module,
    const loom_run_hal_device_target_t* target,
    loom_diagnostic_sink_t diagnostic_sink,
    loom_source_resolver_t source_resolver, uint32_t max_errors,
    loom_run_candidate_artifact_flags_t artifact_flags,
    const loom_run_candidate_artifact_manifest_options_t* artifact_manifest,
    loom_target_compile_report_t* report, iree_allocator_t allocator,
    bool* out_emitted, loom_run_hal_artifact_t* out_artifact);

typedef void (*loom_run_hal_deinitialize_artifact_fn_t)(
    const loom_run_hal_artifact_provider_t* provider,
    loom_run_hal_artifact_t* artifact, iree_allocator_t allocator);

// Linked compiler artifact provider for one HAL driver/target family.
struct loom_run_hal_artifact_provider_t {
  // User-facing provider name accepted by execution tools.
  iree_string_view_t name;
  // IREE HAL driver name used to create the runtime device.
  iree_string_view_t hal_driver_name;
  // Human-readable target-family name used in status messages.
  iree_string_view_t target_family_name;
  // Target-owned defaults used when the shared HAL testbench builds the
  // prepared-low compile pipeline before artifact emission.
  loom_target_pipeline_options_t default_pipeline_options;
  // Selects a concrete target supported by the active HAL executable cache.
  loom_run_hal_select_device_target_fn_t select_device_target;
  // Selects a concrete offline target by provider-owned key. This is used by
  // compilation tools that do not have an active HAL runtime device.
  loom_run_hal_select_target_key_fn_t select_target_key;
  // Releases storage owned by a target returned from a target selection hook.
  loom_run_hal_deinitialize_device_target_fn_t deinitialize_device_target;
  // Emits a prepared target-low Loom module to a HAL loadable artifact.
  loom_run_hal_emit_artifact_fn_t emit_artifact;
  // Releases storage owned by an artifact returned from |emit_artifact|.
  loom_run_hal_deinitialize_artifact_fn_t deinitialize_artifact;
};

// A registry of HAL artifact providers linked into a runner binary.
typedef struct loom_run_hal_artifact_provider_registry_t {
  // Linked artifact provider table; entries are non-NULL when count is nonzero.
  const loom_run_hal_artifact_provider_t* const* providers;
  // Number of entries in |providers|.
  iree_host_size_t provider_count;
} loom_run_hal_artifact_provider_registry_t;

// Initializes |out_registry| from a caller-owned provider table.
void loom_run_hal_artifact_provider_registry_initialize_from_entries(
    const loom_run_hal_artifact_provider_t* const* providers,
    iree_host_size_t provider_count,
    loom_run_hal_artifact_provider_registry_t* out_registry);

// Looks up a HAL artifact provider by user-facing provider name.
const loom_run_hal_artifact_provider_t*
loom_run_hal_artifact_provider_registry_lookup(
    const loom_run_hal_artifact_provider_registry_t* registry,
    iree_string_view_t name);

// Appends a comma-separated list of registered HAL artifact provider names.
iree_status_t loom_run_hal_artifact_provider_registry_format_names(
    const loom_run_hal_artifact_provider_registry_t* registry,
    iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_ARTIFACT_H_
