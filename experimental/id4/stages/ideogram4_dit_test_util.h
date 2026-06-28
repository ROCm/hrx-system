// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_STAGES_IDEOGRAM4_DIT_TEST_UTIL_H_
#define EXPERIMENTAL_ID4_STAGES_IDEOGRAM4_DIT_TEST_UTIL_H_

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/stages/hal_integration_util.h"
#include "experimental/id4/stages/ideogram4_dit_program.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

namespace id4::test {

enum class Ideogram4DitBranch {
  // Conditioned DiT branch using text condition tokens.
  kConditioned,
  // Unconditioned DiT branch using image tokens only.
  kUnconditioned,
};

typedef struct Ideogram4DitBranchConfig {
  // Parameter provider scope containing the selected branch weights.
  iree_string_view_t parameter_scope;
  // Fixture stage containing the branch boundary input tensors.
  iree_string_view_t fixture_stage;
  // Expected fixture tensor name for the branch velocity output.
  iree_string_view_t expected_velocity_name;
  // Diagnostic tap prefix for all semantic tensors in the branch.
  iree_string_view_t diagnostic_tap_prefix;
  // Dynamic conditioning mode used when planning the DiT request.
  id4_ideogram4_dit_conditioning_mode_t conditioning_mode;
} Ideogram4DitBranchConfig;

// Returns the static fixture/parameter configuration for |branch|.
Ideogram4DitBranchConfig Ideogram4DitBranchConfigFor(Ideogram4DitBranch branch);

// Returns a stable benchmark/display name for |branch|.
iree_string_view_t Ideogram4DitBranchName(Ideogram4DitBranch branch);

// Returns the plan JSON filename used for |branch| diagnostics.
iree_string_view_t Ideogram4DitBranchPlanFileName(Ideogram4DitBranch branch);

// Infers the dynamic DiT request shape from a branch fixture.
iree_status_t Ideogram4DitConfigureRequestFromFixture(
    const FixtureTensorSet& fixture_tensors, Ideogram4DitBranchConfig branch,
    id4_ideogram4_dit_request_config_t* out_request);

// Queues all initialized branch boundary tensors from a fixture.
iree_status_t Ideogram4DitQueueInitializedBoundaryTensorsFromFixture(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, const BufferBindingSet& binding_set,
    const FixtureTensorSet& fixture_tensors, Ideogram4DitBranchConfig branch,
    iree_hal_semaphore_t* update_semaphore, uint64_t* inout_update_value);

}  // namespace id4::test

#endif  // EXPERIMENTAL_ID4_STAGES_IDEOGRAM4_DIT_TEST_UTIL_H_
