// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Live device projection for Loom execution tools.
//
// Device providers create no artifacts themselves. They select target facts
// from an active HAL device and project those facts into an ordinary offline
// artifact target. The shared execution layer then emits through the nested
// artifact provider and loads the resulting bytes into that same device.

#ifndef LOOM_TOOLING_EXECUTION_HAL_DEVICE_PROVIDER_H_
#define LOOM_TOOLING_EXECUTION_HAL_DEVICE_PROVIDER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loom/target/profile.h"
#include "loom/tooling/compile/artifact.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_device_provider_t loom_device_provider_t;
struct loom_run_hal_runtime_t;

// Artifact target selected from one active HAL device.
typedef struct loom_device_target_t {
  // Exact executable target row borrowed from the active device spec.
  const iree_hal_executable_target_t* executable_target;
  // Offline artifact target projected from the device.
  loom_artifact_target_t artifact_target;
} loom_device_target_t;

// Returns the target-neutral bundle projected by |target|, or NULL.
static inline const loom_target_bundle_t* loom_device_target_bundle(
    const loom_device_target_t* target) {
  return target ? loom_artifact_target_bundle(&target->artifact_target) : NULL;
}

typedef iree_status_t (*loom_device_provider_select_target_fn_t)(
    const loom_device_provider_t* provider,
    const struct loom_run_hal_runtime_t* runtime, iree_allocator_t allocator,
    loom_device_target_t* out_target);

typedef iree_status_t (*loom_device_provider_select_compatible_target_fn_t)(
    const loom_device_provider_t* provider,
    const struct loom_run_hal_runtime_t* runtime,
    const loom_target_facts_t* target_requirement, iree_allocator_t allocator,
    loom_device_target_t* out_target);

typedef void (*loom_device_provider_deinitialize_target_fn_t)(
    const loom_device_provider_t* provider, loom_device_target_t* target,
    iree_allocator_t allocator);

// Live device adapter for one offline artifact provider and HAL driver.
struct loom_device_provider_t {
  // Offline provider used to emit artifacts after device target selection.
  const loom_artifact_provider_t* artifact_provider;
  // IREE HAL driver name used to create the runtime device.
  iree_string_view_t driver_name;
  // Selects a concrete target supported by the active device.
  loom_device_provider_select_target_fn_t select_target;
  // Selects the most specific concrete device target satisfying an immutable
  // target requirement. NULL represents target-independent code. Facts are
  // borrowed only for the duration of the call and must not be retained.
  loom_device_provider_select_compatible_target_fn_t select_compatible_target;
  // Releases storage owned by a target returned from a selection hook.
  loom_device_provider_deinitialize_target_fn_t deinitialize_target;
};

// Asks |provider| to select a target from |runtime| satisfying
// |target_requirement|. NULL represents target-independent code. The caller
// retains the immutable requirement for the duration of the call.
iree_status_t loom_device_provider_select_compatible_target(
    const loom_device_provider_t* provider,
    const struct loom_run_hal_runtime_t* runtime,
    const loom_target_facts_t* target_requirement, iree_allocator_t allocator,
    loom_device_target_t* out_target);

// A registry of device providers linked into a runner binary.
typedef struct loom_device_provider_registry_t {
  // Linked device provider table; entries are non-NULL when count is nonzero.
  const loom_device_provider_t* const* providers;
  // Number of entries in |providers|.
  iree_host_size_t provider_count;
} loom_device_provider_registry_t;

// Initializes |out_registry| from a caller-owned provider table.
void loom_device_provider_registry_initialize_from_entries(
    const loom_device_provider_t* const* providers,
    iree_host_size_t provider_count,
    loom_device_provider_registry_t* out_registry);

// Looks up a device provider by its artifact provider name.
const loom_device_provider_t* loom_device_provider_registry_lookup(
    const loom_device_provider_registry_t* registry, iree_string_view_t name);

// Appends a comma-separated list of registered device provider names.
iree_status_t loom_device_provider_registry_format_names(
    const loom_device_provider_registry_t* registry,
    iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_DEVICE_PROVIDER_H_
