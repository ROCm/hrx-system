// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/ideogram4_dit_test_util.h"

#include <limits>

namespace id4::test {

static bool TensorShapeEquals(id4_pipeline_tensor_shape_t lhs,
                              id4_pipeline_tensor_shape_t rhs) {
  if (lhs.rank != rhs.rank) return false;
  for (uint32_t i = 0; i < lhs.rank; ++i) {
    if (lhs.dims[i] != rhs.dims[i]) return false;
  }
  return true;
}

static iree_status_t FindFixtureTensor(const FixtureTensorSet& fixture_tensors,
                                       iree_string_view_t role,
                                       iree_string_view_t stage,
                                       iree_string_view_t name,
                                       const FixtureTensor** out_tensor) {
  *out_tensor = fixture_tensors.FindTensor(role, stage, name);
  if (*out_tensor) return iree_ok_status();
  return iree_make_status(
      IREE_STATUS_NOT_FOUND, "fixture tensor `%.*s/%.*s/%.*s` not found",
      static_cast<int>(role.size), role.data, static_cast<int>(stage.size),
      stage.data, static_cast<int>(name.size), name.data);
}

static iree_status_t MakeProgramShape(
    id4_pipeline_tensor_shape_t tensor_shape,
    id4_pipeline_program_shape_t* out_program_shape) {
  if (tensor_shape.rank > IREE_ARRAYSIZE(out_program_shape->dims)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "fixture tensor rank %u exceeds program max rank",
                            tensor_shape.rank);
  }
  *out_program_shape = id4_pipeline_program_shape_t{};
  out_program_shape->rank = tensor_shape.rank;
  for (uint32_t i = 0; i < tensor_shape.rank; ++i) {
    out_program_shape->dims[i] = tensor_shape.dims[i];
  }
  return iree_ok_status();
}

static iree_string_view_t FixtureNameForBoundary(
    const id4_pipeline_boundary_tensor_plan_t* boundary) {
  if (iree_string_view_equal(boundary->layout.name, IREE_SV("condition"))) {
    return IREE_SV("context");
  }
  return boundary->layout.name;
}

Ideogram4DitBranchConfig Ideogram4DitBranchConfigFor(
    Ideogram4DitBranch branch) {
  switch (branch) {
    case Ideogram4DitBranch::kConditioned:
      return Ideogram4DitBranchConfig{
          // Conditioned DiT parameter scope.
          /*.parameter_scope=*/IREE_SV("dit_cond"),
          // Fixture stage carrying conditioned DiT inputs.
          /*.fixture_stage=*/IREE_SV("ideogram4.cond.input"),
          // Expected conditioned velocity fixture tensor.
          /*.expected_velocity_name=*/IREE_SV("ideogram4.cond.output.velocity"),
          // Conditioned diagnostic tap prefix.
          /*.diagnostic_tap_prefix=*/IREE_SV("ideogram4.cond."),
          // Conditioned request consumes Qwen context tokens.
          /*.conditioning_mode=*/
          ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED,
      };
    case Ideogram4DitBranch::kUnconditioned:
      return Ideogram4DitBranchConfig{
          // Unconditioned DiT parameter scope.
          /*.parameter_scope=*/IREE_SV("dit_uncond"),
          // Fixture stage carrying unconditioned DiT inputs.
          /*.fixture_stage=*/IREE_SV("ideogram4.uncond.input"),
          // Expected unconditioned velocity fixture tensor.
          /*.expected_velocity_name=*/
          IREE_SV("ideogram4.uncond.output.velocity"),
          // Unconditioned diagnostic tap prefix.
          /*.diagnostic_tap_prefix=*/IREE_SV("ideogram4.uncond."),
          // Unconditioned request consumes image tokens only.
          /*.conditioning_mode=*/
          ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED,
      };
  }
  return Ideogram4DitBranchConfig{};
}

iree_string_view_t Ideogram4DitBranchName(Ideogram4DitBranch branch) {
  switch (branch) {
    case Ideogram4DitBranch::kConditioned:
      return IREE_SV("conditioned");
    case Ideogram4DitBranch::kUnconditioned:
      return IREE_SV("unconditioned");
  }
  return IREE_SV("invalid");
}

iree_string_view_t Ideogram4DitBranchPlanFileName(Ideogram4DitBranch branch) {
  switch (branch) {
    case Ideogram4DitBranch::kConditioned:
      return IREE_SV("cond.json");
    case Ideogram4DitBranch::kUnconditioned:
      return IREE_SV("uncond.json");
  }
  return iree_string_view_empty();
}

iree_status_t Ideogram4DitConfigureRequestFromFixture(
    const FixtureTensorSet& fixture_tensors, Ideogram4DitBranchConfig branch,
    id4_ideogram4_dit_request_config_t* out_request) {
  *out_request = id4_ideogram4_dit_request_config_t{};

  const FixtureTensor* latent = nullptr;
  IREE_RETURN_IF_ERROR(FindFixtureTensor(fixture_tensors, IREE_SV("input"),
                                         branch.fixture_stage, IREE_SV("x"),
                                         &latent));
  IREE_RETURN_IF_ERROR(
      MakeProgramShape(latent->shape, &out_request->latent_shape));
  out_request->conditioning_mode = branch.conditioning_mode;

  if (branch.conditioning_mode ==
      ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED) {
    const FixtureTensor* condition = nullptr;
    IREE_RETURN_IF_ERROR(FindFixtureTensor(fixture_tensors, IREE_SV("input"),
                                           branch.fixture_stage,
                                           IREE_SV("context"), &condition));
    if (condition->shape.rank != 2 ||
        condition->shape.dims[1] >
            static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "condition fixture tensor must be rank-2 with uint32 token count");
    }
    out_request->text_token_count =
        static_cast<uint32_t>(condition->shape.dims[1]);
  }
  return iree_ok_status();
}

iree_status_t Ideogram4DitQueueInitializedBoundaryTensorsFromFixture(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, const BufferBindingSet& binding_set,
    const FixtureTensorSet& fixture_tensors, Ideogram4DitBranchConfig branch,
    iree_hal_semaphore_t* update_semaphore, uint64_t* inout_update_value) {
  for (iree_host_size_t i = 0; i < binding_set.count; ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (!boundary ||
        !iree_all_bits_set(boundary->flags,
                           ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_INITIALIZED)) {
      continue;
    }
    const FixtureTensor* tensor = nullptr;
    IREE_RETURN_IF_ERROR(FindFixtureTensor(
        fixture_tensors, IREE_SV("input"), branch.fixture_stage,
        FixtureNameForBoundary(boundary), &tensor));
    if (tensor->dtype != boundary->layout.dtype) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "fixture tensor `%s/%s` dtype does not match boundary `%.*s`",
          tensor->stage.c_str(), tensor->name.c_str(),
          static_cast<int>(boundary->layout.name.size),
          boundary->layout.name.data);
    }
    if (!TensorShapeEquals(tensor->shape, boundary->layout.shape)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "fixture tensor `%s/%s` shape does not match boundary `%.*s`",
          tensor->stage.c_str(), tensor->name.c_str(),
          static_cast<int>(boundary->layout.name.size),
          boundary->layout.name.data);
    }
    IREE_RETURN_IF_ERROR(QueueReadBindingFromHostAllocation(
        device, queue_affinity, &binding_set.bindings[i],
        tensor->payload.data(), tensor->payload.size(), update_semaphore,
        inout_update_value));
  }
  return iree_ok_status();
}

}  // namespace id4::test
