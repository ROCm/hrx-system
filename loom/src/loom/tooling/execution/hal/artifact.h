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
#include "iree/hal/api.h"
#include "iree/io/byte_sequence.h"
#include "loom/error/diagnostic.h"
#include "loom/ir/module.h"
#include "loom/target/pipeline_options.h"
#include "loom/target/profile.h"
#include "loom/target/provider.h"
#include "loom/target/reporting/report.h"
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
  // Exact target row borrowed from the active HAL device spec. NULL for
  // offline target selections that cannot be loaded into a device.
  const iree_hal_executable_target_t* hal_target;
  // Provider-owned structured target profile. NULL requests emission from the
  // module's authored target records.
  const loom_target_profile_t* target_profile;
  // Provider-facing target key selected for emission and diagnostics, if any.
  iree_string_view_t target_key;
} loom_run_hal_device_target_t;

// Returns the target-neutral bundle projected by |target|, or NULL.
static inline const loom_target_bundle_t* loom_run_hal_device_target_bundle(
    const loom_run_hal_device_target_t* target) {
  return target ? loom_target_profile_bundle(target->target_profile) : NULL;
}

// Loadable HAL artifact bytes ready for iree_hal_device_load_executable.
//
// Sequence pointers are borrowed from |storage| for the artifact lifetime.
// Callers may retain individual sequences when they must outlive the artifact.
typedef struct loom_run_hal_artifact_t {
  // Exact target row borrowed from the active HAL device spec.
  const iree_hal_executable_target_t* hal_target;
  // Family-owned target key used to emit the artifact.
  iree_string_view_t target_key;
  // Durable target-neutral bundle resolved for the artifact.
  const loom_target_bundle_t* target_bundle;
  // Target-native artifact format.
  loom_target_artifact_format_t target_artifact_format;
  // Borrowed target-native artifact contents owned by |storage|.
  iree_io_byte_sequence_t* target_artifact_data;
  // Target-owned textual listing format, such as `amdgpu-assembly`.
  iree_string_view_t target_listing_format;
  // Borrowed textual listing contents owned by |storage|.
  iree_io_byte_sequence_t* target_listing_data;
  // Optional sidecar artifacts produced beside executable_data.
  const loom_target_emit_sidecar_artifact_t* sidecars;
  // Number of entries in |sidecars|.
  iree_host_size_t sidecar_count;
  // Borrowed executable contents owned by |storage|.
  iree_io_byte_sequence_t* executable_data;
  // Provider-owned storage released by |deinitialize_artifact|.
  void* storage;
} loom_run_hal_artifact_t;

typedef iree_status_t (*loom_run_hal_select_device_target_fn_t)(
    const loom_run_hal_artifact_provider_t* provider,
    const struct loom_run_hal_runtime_t* runtime, iree_allocator_t allocator,
    loom_run_hal_device_target_t* out_target);

typedef iree_status_t (*loom_run_hal_select_compatible_device_target_fn_t)(
    const loom_run_hal_artifact_provider_t* provider,
    const struct loom_run_hal_runtime_t* runtime,
    const loom_target_facts_t* target_requirement, iree_allocator_t allocator,
    loom_run_hal_device_target_t* out_target);

typedef iree_status_t (*loom_run_hal_select_target_key_fn_t)(
    const loom_run_hal_artifact_provider_t* provider,
    iree_string_view_t target_key, iree_allocator_t allocator,
    loom_run_hal_device_target_t* out_target);

typedef void (*loom_run_hal_deinitialize_device_target_fn_t)(
    const loom_run_hal_artifact_provider_t* provider,
    loom_run_hal_device_target_t* target, iree_allocator_t allocator);

// Emits a loader-ready HAL artifact. When |out_emitted| is true the artifact
// has a target bundle, non-empty target-native and executable contents, and a
// valid descriptor and contents for every sidecar. Returning OK with
// |out_emitted| false is reserved for product diagnostics emitted through
// |diagnostic_sink|; infrastructure failures return a non-OK status.
typedef iree_status_t (*loom_run_hal_emit_artifact_fn_t)(
    const loom_run_hal_artifact_provider_t* provider, loom_module_t* module,
    const loom_run_hal_device_target_t* target,
    const loom_run_candidate_compile_options_t* options,
    iree_allocator_t allocator, bool* out_emitted,
    loom_run_hal_artifact_t* out_artifact);

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
  // Selects a concrete target supported by the active HAL device.
  loom_run_hal_select_device_target_fn_t select_device_target;
  // Selects the most specific concrete device target satisfying an immutable
  // target requirement. NULL represents target-independent code. Facts are
  // borrowed only for the duration of the call and must not be retained.
  loom_run_hal_select_compatible_device_target_fn_t
      select_compatible_device_target;
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

// Asks |provider| to select a target from |runtime| satisfying
// |target_requirement|. NULL represents target-independent code. The caller
// retains the immutable requirement for the duration of the call.
iree_status_t loom_run_hal_artifact_provider_select_compatible_device_target(
    const loom_run_hal_artifact_provider_t* provider,
    const struct loom_run_hal_runtime_t* runtime,
    const loom_target_facts_t* target_requirement, iree_allocator_t allocator,
    loom_run_hal_device_target_t* out_target);

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
