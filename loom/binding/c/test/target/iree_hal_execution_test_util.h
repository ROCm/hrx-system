// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TEST_TARGET_IREE_HAL_EXECUTION_H_
#define LOOMC_TEST_TARGET_IREE_HAL_EXECUTION_H_

#include <stdint.h>

#include <functional>

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

// Target-specific inputs for the shared kernel-to-HAL execution test.
struct IreeHalKernelExecutionTarget {
  // Human-readable target name used in skip and failure messages.
  const char* label;

  // HAL device URI used to create the live device.
  iree_string_view_t device_uri;

  // Profile identifier passed to `loomc_target_select_iree_hal`.
  loomc_string_view_t target_profile_identifier;

  // Source identifier reported in parse diagnostics.
  loomc_string_view_t source_identifier;

  // Borrowed `.loom` source contents for the target-specific kernel.
  loomc_string_view_t source_text;

  // Public kernel export compiled and dispatched by the test.
  loomc_string_view_t kernel_export_name;

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

  // Static provider array used to select compiler-and-loader targets.
  const loomc_iree_hal_target_provider_t* const* target_providers;

  // Number of provider entries in `target_providers`.
  loomc_host_size_t target_provider_count;

  // Target environment factory for this backend.
  IreeHalTargetEnvironmentCreateFn create_target_environment;

  // Target-specific profile validation callback.
  IreeHalTargetProfileValidateFn validate_target_profile;
};

// Borrowed execution objects valid for the duration of an execution callback.
struct IreeHalKernelExecution {
  // Live device that owns the executable and queues.
  iree_hal_device_t* device;
  // Queue used for explicitly synchronized uploads and downloads.
  iree_hal_queue_t* transfer_queue;
  // Queue compatible with the loaded executable.
  iree_hal_queue_t* dispatch_queue;
  // Native executable compiled through the public Loom C API.
  iree_hal_executable_t* executable;
  // Evaluated launch geometry for the selected kernel.
  loomc_launch_config_t launch_config;
};

// Compiles and loads a kernel through the public Loom C API and invokes
// |execute| while all compiler and HAL owners remain alive. The callback must
// wait for its submitted work before returning. Uses GTest assertions/skips.
void RunIreeHalKernelExecutionTest(
    const IreeHalKernelExecutionTarget& target,
    const std::function<void(const IreeHalKernelExecution&)>& execute);

// Dispatches the selected kernel with two whole-buffer bindings and the given
// push constants. Submission ordering is defined by the supplied semaphores.
iree_status_t DispatchIreeHalKernel(
    const IreeHalKernelExecution& execution, iree_const_byte_span_t constants,
    iree_hal_buffer_t* input_buffer, iree_hal_buffer_t* output_buffer,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores);

// Checks the standard two-element doubling kernel at a four-byte offset.
void RunIreeHalByteOffsetExecution(const IreeHalKernelExecution& execution);

}  // namespace loomc::testing::target

#endif  // LOOMC_TEST_TARGET_IREE_HAL_EXECUTION_H_
