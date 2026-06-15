// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TEST_TARGET_IREE_HAL_EXECUTION_H_
#define LOOMC_TEST_TARGET_IREE_HAL_EXECUTION_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loomc/loomc.h"
#include "loomc/target/iree_hal.h"

namespace loomc::testing::target {

struct IreeHalKernelExecutionTarget;

// Creates the target environment used by a live HAL execution test.
using IreeHalTargetEnvironmentCreateFn =
    loomc_status_t (*)(loomc_allocator_t host_allocator,
                       loomc_target_environment_t** out_target_environment);

// Validates target-specific profile facts before the executable is compiled.
using IreeHalTargetProfileValidateFn = loomc_status_t (*)(
    loomc_target_profile_t* target_profile, const char** out_skip_reason);

// Emits a target-specific executable artifact from the compiled module.
using IreeHalTargetModuleEmitFn = loomc_status_t (*)(
    loomc_target_environment_t* target_environment,
    loomc_workspace_t* workspace, loomc_module_t* module,
    loomc_target_selection_t* target_selection,
    loomc_string_view_t artifact_format,
    loomc_string_view_t artifact_identifier, loomc_result_t** out_result);

// Target-specific inputs for the shared kernel-to-HAL execution test.
struct IreeHalKernelExecutionTarget {
  // Human-readable target name used in skip and failure messages.
  const char* label;

  // HAL device URI used to create the live device.
  iree_string_view_t device_uri;

  // Executable cache identifier used when preparing the HAL executable.
  iree_string_view_t executable_cache_identifier;

  // Profile identifier passed to `loomc_target_profile_create_iree_hal`.
  loomc_string_view_t target_profile_identifier;

  // Source identifier reported in parse diagnostics.
  loomc_string_view_t source_identifier;

  // Borrowed `.loom` source contents for the target-specific kernel.
  loomc_string_view_t source_text;

  // Module name passed to the compile invocation.
  loomc_string_view_t module_name;

  // Exported kernel function symbol compiled and dispatched by the test.
  loomc_string_view_t kernel_function_symbol;

  // Pipeline identifier reported by pipeline creation diagnostics.
  loomc_string_view_t target_pipeline_identifier;

  // Target pipeline kind used to lower the source module.
  loomc_target_pipeline_kind_t target_pipeline_kind;

  // Control-flow lowering policy used by the target pipeline.
  loomc_target_control_flow_lowering_t control_flow_lowering;

  // Maximum source-to-low errors accepted before pipeline creation fails.
  uint32_t source_to_low_max_errors;

  // Artifact format expected from the target-specific emitter.
  loomc_string_view_t artifact_format;

  // Artifact identifier reported by emission diagnostics.
  loomc_string_view_t artifact_identifier;

  // HAL executable format used by `iree_hal_executable_cache_prepare`.
  iree_string_view_t executable_format;

  // Static provider array used to project HAL device facts into Loom facts.
  const loomc_iree_hal_profile_provider_t* const* profile_providers;

  // Number of provider entries in `profile_providers`.
  loomc_host_size_t profile_provider_count;

  // Target environment factory for this backend.
  IreeHalTargetEnvironmentCreateFn create_target_environment;

  // Target-specific profile validation callback.
  IreeHalTargetProfileValidateFn validate_target_profile;

  // Target-specific module emission callback.
  IreeHalTargetModuleEmitFn emit_module;
};

// Runs the shared public Loom C API to live IREE HAL kernel execution flow.
// Uses GTest assertions and skips; intended to be called as a `TEST` body.
void RunIreeHalKernelExecutionTest(const IreeHalKernelExecutionTarget& target);

}  // namespace loomc::testing::target

#endif  // LOOMC_TEST_TARGET_IREE_HAL_EXECUTION_H_
