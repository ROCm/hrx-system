// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <vector>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/hal_integration_util.h"
#include "experimental/id4/stages/ideogram4_decode.h"
#include "experimental/id4/stages/ideogram4_dit.h"
#include "experimental/id4/stages/qwen3_vl.h"
#include "experimental/id4/stages/sampler.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

IREE_FLAG(string, id4_qwen_fixture_dir, "",
          "Directory containing the Qwen3-VL reduced BF16 fixture.");
IREE_FLAG(string, id4_dit_cond_fixture_dir, "",
          "Directory containing the conditioned Ideogram4 DiT reduced BF16 "
          "fixture.");
IREE_FLAG(string, id4_dit_uncond_fixture_dir, "",
          "Directory containing the unconditioned Ideogram4 DiT reduced BF16 "
          "fixture.");
IREE_FLAG(string, id4_sampler_fixture_dir, "",
          "Directory containing the sampler reduced BF16 fixture.");

namespace {

constexpr uint8_t kOutputSentinel = 0xA5;

enum class DitExpectedTapSet {
  kAll,
  kConditionedPrelude,
};

static iree_string_view_t FixtureTensorName(
    const id4::test::FixtureTensor& tensor) {
  return iree_make_string_view(tensor.name.data(), tensor.name.size());
}

static iree_string_view_t FixtureTensorRole(
    const id4::test::FixtureTensor& tensor) {
  return iree_make_string_view(tensor.role.data(), tensor.role.size());
}

static iree_status_t FindFixtureTensor(
    const id4::test::FixtureTensorSet& fixture_tensors, iree_string_view_t role,
    iree_string_view_t name, const id4::test::FixtureTensor** out_tensor) {
  *out_tensor = fixture_tensors.FindTensor(role, name);
  if (*out_tensor) return iree_ok_status();
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "fixture tensor `%.*s` with role `%.*s` not found",
                          static_cast<int>(name.size), name.data,
                          static_cast<int>(role.size), role.data);
}

static iree_status_t MakeProgramShape(
    id4_pipeline_tensor_shape_t tensor_shape,
    id4_pipeline_program_shape_t* out_program_shape) {
  if (tensor_shape.rank > IREE_ARRAYSIZE(out_program_shape->dims)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "fixture tensor rank %u exceeds program max rank",
                            tensor_shape.rank);
  }
  std::memset(out_program_shape, 0, sizeof(*out_program_shape));
  out_program_shape->rank = tensor_shape.rank;
  for (uint32_t i = 0; i < tensor_shape.rank; ++i) {
    out_program_shape->dims[i] = tensor_shape.dims[i];
  }
  return iree_ok_status();
}

static iree_status_t ConfigureDitRequestFromFixture(
    const id4::test::FixtureTensorSet& fixture_tensors,
    id4_ideogram4_dit_request_config_t* out_request) {
  std::memset(out_request, 0, sizeof(*out_request));

  const id4::test::FixtureTensor* latent = nullptr;
  IREE_RETURN_IF_ERROR(FindFixtureTensor(fixture_tensors, IREE_SV("input"),
                                         IREE_SV("x"), &latent));
  IREE_RETURN_IF_ERROR(
      MakeProgramShape(latent->shape, &out_request->latent_shape));

  const id4::test::FixtureTensor* condition =
      fixture_tensors.FindTensor(IREE_SV("input"), IREE_SV("condition"));
  if (!condition) {
    out_request->conditioning_mode =
        ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED;
    return iree_ok_status();
  }
  if (condition->shape.rank != 2 || condition->shape.dims[1] > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "condition fixture tensor must be rank-2 with uint32 token count");
  }
  out_request->conditioning_mode =
      ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED;
  out_request->text_token_count =
      static_cast<uint32_t>(condition->shape.dims[1]);
  return iree_ok_status();
}

static iree_status_t CreateQwenStage(const id4::test::LiveStageContext& context,
                                     iree_string_view_t parameter_scope,
                                     id4_pipeline_stage_t** out_stage) {
  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = context.device_group.get();
  services.executable_cache = context.executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_qwen3_vl_stage_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.services = services;
  options.kernel_cache = context.kernel_cache.get();
  options.parameter_scope = parameter_scope;
  options.parameter_format = ID4_QWEN3_VL_PARAMETER_FORMAT_BF16;
  options.model = *id4_qwen3_vl_program_ideogram4_model_config();
  return id4_qwen3_vl_stage_create(&options, iree_allocator_system(),
                                   out_stage);
}

static iree_status_t CreateDitStage(const id4::test::LiveStageContext& context,
                                    iree_string_view_t parameter_scope,
                                    id4_pipeline_stage_t** out_stage) {
  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = context.device_group.get();
  services.executable_cache = context.executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_ideogram4_dit_stage_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.services = services;
  options.kernel_cache = context.kernel_cache.get();
  options.parameter_scope = parameter_scope;
  options.model = *id4_ideogram4_dit_program_ideogram4_model_config();
  return id4_ideogram4_dit_stage_create(&options, iree_allocator_system(),
                                        out_stage);
}

static iree_status_t CreateSamplerStage(
    const id4::test::LiveStageContext& context,
    id4_pipeline_stage_t** out_stage) {
  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = context.device_group.get();
  services.executable_cache = context.executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_sampler_denoise_stage_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.services = services;
  options.kernel_cache = context.kernel_cache.get();
  return id4_sampler_denoise_stage_create(&options, iree_allocator_system(),
                                          out_stage);
}

static iree_status_t CreateDecodeStage(
    const id4::test::LiveStageContext& context,
    iree_string_view_t parameter_scope, id4_pipeline_stage_t** out_stage) {
  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = context.device_group.get();
  services.executable_cache = context.executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_ideogram4_decode_stage_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.services = services;
  options.kernel_cache = context.kernel_cache.get();
  options.parameter_scope = parameter_scope;
  options.model = *id4_ideogram4_decode_program_ideogram4_model_config();
  options.vae_activation_format = ID4_VAE_ACTIVATION_FORMAT_F32_CANONICAL;
  return id4_ideogram4_decode_stage_create(&options, iree_allocator_system(),
                                           out_stage);
}

static iree_status_t PlanQwenStage(
    id4_pipeline_stage_t* stage, uint32_t token_count, const int32_t* token_ids,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_pipeline_plan_t** out_plan) {
  id4_qwen3_vl_stage_plan_options_t qwen_options;
  std::memset(&qwen_options, 0, sizeof(qwen_options));
  qwen_options.structure_size = sizeof(qwen_options);
  qwen_options.request.token_count = token_count;
  qwen_options.request.token_ids = token_ids;
  qwen_options.weight_execution_strategy =
      ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_ROW_MAJOR;
  qwen_options.attention_implementation =
      ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_AUTO;

  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &qwen_options;
  const iree_string_view_t diagnostic_tap_names[] = {
      IREE_SV("selected_hidden_states"),
  };
  plan_options.flags = ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
  plan_options.diagnostic_tap_names = (iree_string_view_list_t){
      IREE_ARRAYSIZE(diagnostic_tap_names),
      diagnostic_tap_names,
  };
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = diagnostics_sink;
  return id4_pipeline_stage_plan(stage, &plan_options, out_plan);
}

static bool ShouldSkipDitExpectedTap(const id4::test::FixtureTensor& tensor,
                                     DitExpectedTapSet tap_set) {
  return tap_set == DitExpectedTapSet::kConditionedPrelude &&
         iree_string_view_equal(FixtureTensorName(tensor),
                                IREE_SV("ideogram4.cond.output.velocity"));
}

static iree_status_t PlanDitStage(
    id4_pipeline_stage_t* stage, id4_ideogram4_dit_request_config_t request,
    const id4::test::FixtureTensorSet& fixture_tensors,
    DitExpectedTapSet expected_tap_set,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_pipeline_plan_t** out_plan) {
  id4_ideogram4_dit_stage_plan_options_t dit_options;
  std::memset(&dit_options, 0, sizeof(dit_options));
  dit_options.structure_size = sizeof(dit_options);
  dit_options.request = request;
  dit_options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  dit_options.weight_execution_format =
      ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_BF16_RESIDENT;
  dit_options.attention_implementation =
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING;
  dit_options.feed_forward_implementation =
      ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_PYTORCH_PARITY;

  std::vector<iree_string_view_t> diagnostic_tap_names;
  for (const id4::test::FixtureTensor& tensor : fixture_tensors.tensors) {
    if (!iree_string_view_equal(FixtureTensorRole(tensor),
                                IREE_SV("expected"))) {
      continue;
    }
    if (ShouldSkipDitExpectedTap(tensor, expected_tap_set)) {
      continue;
    }
    diagnostic_tap_names.push_back(FixtureTensorName(tensor));
  }
  if (diagnostic_tap_names.empty()) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "DiT fixture contains no valid expected taps");
  }

  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &dit_options;
  plan_options.flags = ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
  plan_options.diagnostic_tap_names = (iree_string_view_list_t){
      diagnostic_tap_names.size(),
      diagnostic_tap_names.data(),
  };
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = diagnostics_sink;
  return id4_pipeline_stage_plan(stage, &plan_options, out_plan);
}

static iree_status_t PlanSamplerStage(
    id4_pipeline_stage_t* stage, id4_pipeline_program_shape_t latent_shape,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_pipeline_plan_t** out_plan) {
  id4_sampler_denoise_stage_plan_options_t sampler_options;
  std::memset(&sampler_options, 0, sizeof(sampler_options));
  sampler_options.structure_size = sizeof(sampler_options);
  sampler_options.request.latent_shape = latent_shape;

  const iree_string_view_t diagnostic_tap_names[] = {IREE_SV("guided_pred")};
  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &sampler_options;
  plan_options.flags = ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
  plan_options.diagnostic_tap_names = (iree_string_view_list_t){
      IREE_ARRAYSIZE(diagnostic_tap_names),
      diagnostic_tap_names,
  };
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = diagnostics_sink;
  return id4_pipeline_stage_plan(stage, &plan_options, out_plan);
}

static iree_status_t PlanDecodeStage(
    id4_pipeline_stage_t* stage,
    id4_pipeline_program_shape_t diffusion_latent_shape,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_pipeline_plan_t** out_plan) {
  const id4_ideogram4_decode_model_config_t* model =
      id4_ideogram4_decode_program_ideogram4_model_config();
  id4_ideogram4_decode_stage_plan_options_t decode_options;
  std::memset(&decode_options, 0, sizeof(decode_options));
  decode_options.structure_size = sizeof(decode_options);
  decode_options.request.diffusion_latent_shape = diffusion_latent_shape;
  if (diffusion_latent_shape.dims[0] < model->vae.min_tile_size_x ||
      diffusion_latent_shape.dims[1] < model->vae.min_tile_size_y) {
    decode_options.request.vae_tiling.mode = ID4_VAE_TILING_MODE_DISABLED;
  } else {
    decode_options.request.vae_tiling.mode =
        ID4_VAE_TILING_MODE_EXPLICIT_TILE_SIZE;
    decode_options.request.vae_tiling.tile_size_x = 32;
    decode_options.request.vae_tiling.tile_size_y = 32;
    decode_options.request.vae_tiling.overlap = 0.5f;
  }

  const iree_string_view_t diagnostic_tap_names[] = {
      IREE_SV("vae.flux2.internal_latent"),
  };
  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &decode_options;
  plan_options.flags = ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
  plan_options.diagnostic_tap_names = (iree_string_view_list_t){
      IREE_ARRAYSIZE(diagnostic_tap_names),
      diagnostic_tap_names,
  };
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = diagnostics_sink;
  return id4_pipeline_stage_plan(stage, &plan_options, out_plan);
}

static iree_status_t PrepareStage(
    id4_pipeline_stage_t* stage, const id4_pipeline_plan_t* plan,
    iree_io_parameter_provider_t* parameter_provider,
    id4_pipeline_kernel_library_t* kernel_library,
    iree_hal_semaphore_list_t wait_list, iree_hal_semaphore_list_t signal_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_pipeline_bundle_t** out_bundle) {
  id4_pipeline_stage_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.parameter_provider = parameter_provider;
  prepare_options.kernel_library = kernel_library;
  prepare_options.wait_semaphore_list = wait_list;
  prepare_options.signal_semaphore_list = signal_list;
  prepare_options.diagnostics_sink = diagnostics_sink;
  return id4_pipeline_stage_prepare(stage, plan, &prepare_options, out_bundle);
}

static iree_status_t ReplaceBoundaryBinding(
    const id4_pipeline_plan_t* plan, id4::test::BufferBindingSet* binding_set,
    iree_string_view_t name, iree_hal_buffer_binding_t replacement) {
  for (iree_host_size_t i = 0;
       i < id4_pipeline_plan_boundary_tensor_count(plan); ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (boundary && iree_string_view_equal(boundary->layout.name, name)) {
      binding_set->bindings[i] = replacement;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "boundary tensor `%.*s` not found",
                          static_cast<int>(name.size), name.data);
}

static iree_status_t CompareExpectedDiagnosticTaps(
    iree_hal_device_t* device, const id4_pipeline_plan_t* plan,
    const id4::test::BufferBindingSet& diagnostic_tap_bindings,
    const id4::test::FixtureTensorSet& fixture_tensors,
    DitExpectedTapSet expected_tap_set, iree_hal_semaphore_list_t wait_list) {
  iree_host_size_t expected_count = 0;
  for (const id4::test::FixtureTensor& tensor : fixture_tensors.tensors) {
    if (!iree_string_view_equal(FixtureTensorRole(tensor),
                                IREE_SV("expected"))) {
      continue;
    }
    if (ShouldSkipDitExpectedTap(tensor, expected_tap_set)) {
      continue;
    }
    ++expected_count;
    iree_hal_buffer_binding_t binding = {};
    IREE_RETURN_IF_ERROR(id4::test::FindDiagnosticTapBinding(
        plan, diagnostic_tap_bindings, FixtureTensorName(tensor), &binding));
    IREE_RETURN_IF_ERROR(id4::test::CompareF32BindingWithFixtureTensor(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, &binding, wait_list, tensor));
  }
  if (expected_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "fixture contains no expected diagnostic taps");
  }
  return iree_ok_status();
}

static iree_status_t VerifyBindingWasWritten(
    iree_hal_device_t* device, const iree_hal_buffer_binding_t* binding,
    iree_hal_semaphore_list_t wait_list, uint8_t sentinel) {
  std::vector<uint8_t> bytes;
  IREE_RETURN_IF_ERROR(id4::test::ReadBindingToHost(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, binding, wait_list, &bytes));
  for (uint8_t byte : bytes) {
    if (byte != sentinel) return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "binding was not written");
}

static iree_status_t IssueStage(
    id4_pipeline_stage_t* stage, id4_pipeline_bundle_t* bundle,
    const id4::test::BufferBindingSet& boundary_bindings,
    const id4::test::BufferBindingSet& diagnostic_tap_bindings,
    iree_hal_semaphore_list_t wait_list, iree_hal_semaphore_list_t signal_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_pipeline_stage_issue_options_t issue_options;
  std::memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.region_submission_window = 1;
  issue_options.boundary_binding_count = boundary_bindings.count;
  issue_options.boundary_bindings = boundary_bindings.bindings;
  issue_options.diagnostic_tap_binding_count = diagnostic_tap_bindings.count;
  issue_options.diagnostic_tap_bindings = diagnostic_tap_bindings.bindings;
  issue_options.wait_semaphore_list = wait_list;
  issue_options.signal_semaphore_list = signal_list;
  issue_options.diagnostics_sink = diagnostics_sink;
  return id4_pipeline_stage_issue(stage, bundle, &issue_options);
}

TEST(Ideogram4OneStepIntegration,
     RunsQwenAndReferenceDitSamplerWithDeviceHandoffs) {
  const iree_string_view_t qwen_fixture_dir =
      iree_make_cstring_view(FLAG_id4_qwen_fixture_dir);
  const iree_string_view_t dit_cond_fixture_dir =
      iree_make_cstring_view(FLAG_id4_dit_cond_fixture_dir);
  const iree_string_view_t dit_uncond_fixture_dir =
      iree_make_cstring_view(FLAG_id4_dit_uncond_fixture_dir);
  const iree_string_view_t sampler_fixture_dir =
      iree_make_cstring_view(FLAG_id4_sampler_fixture_dir);
  ASSERT_FALSE(iree_string_view_is_empty(qwen_fixture_dir));
  ASSERT_FALSE(iree_string_view_is_empty(dit_cond_fixture_dir));
  ASSERT_FALSE(iree_string_view_is_empty(dit_uncond_fixture_dir));
  ASSERT_FALSE(iree_string_view_is_empty(sampler_fixture_dir));

  id4::test::FixtureTensorSet qwen_tensors;
  IREE_ASSERT_OK(
      id4::test::LoadFixtureTensors(qwen_fixture_dir, &qwen_tensors));
  id4::test::FixtureTensorSet dit_cond_tensors;
  IREE_ASSERT_OK(
      id4::test::LoadFixtureTensors(dit_cond_fixture_dir, &dit_cond_tensors));
  id4::test::FixtureTensorSet dit_uncond_tensors;
  IREE_ASSERT_OK(id4::test::LoadFixtureTensors(dit_uncond_fixture_dir,
                                               &dit_uncond_tensors));
  id4::test::FixtureTensorSet sampler_tensors;
  IREE_ASSERT_OK(
      id4::test::LoadFixtureTensors(sampler_fixture_dir, &sampler_tensors));

  uint32_t qwen_token_count = 0;
  IREE_ASSERT_OK(id4::test::InferRank1TensorLengthFromFixture(
      qwen_tensors, IREE_SV("token_ids"), ID4_PIPELINE_TENSOR_DTYPE_I32,
      &qwen_token_count));
  const id4::test::FixtureTensor* qwen_token_ids_fixture =
      qwen_tensors.FindTensor(IREE_SV("input"), IREE_SV("token_ids"));
  ASSERT_NE(qwen_token_ids_fixture, nullptr);
  ASSERT_EQ(qwen_token_ids_fixture->payload.size(),
            qwen_token_count * sizeof(int32_t));
  std::vector<int32_t> qwen_token_ids(qwen_token_count);
  std::memcpy(qwen_token_ids.data(), qwen_token_ids_fixture->payload.data(),
              qwen_token_ids_fixture->payload.size());
  id4_ideogram4_dit_request_config_t dit_cond_request;
  IREE_ASSERT_OK(
      ConfigureDitRequestFromFixture(dit_cond_tensors, &dit_cond_request));
  id4_ideogram4_dit_request_config_t dit_uncond_request;
  IREE_ASSERT_OK(
      ConfigureDitRequestFromFixture(dit_uncond_tensors, &dit_uncond_request));

  id4::test::LiveStageContext context;
  IREE_ASSERT_OK(id4::test::CreateLiveStageContextFromFlags(&context));
  id4::test::KernelLibraryRef kernel_library;
  IREE_ASSERT_OK(id4::test::CreateEmbeddedKernelLibrary(kernel_library.out()));

  id4::test::OwningRef<iree_io_parameter_provider_t,
                       iree_io_parameter_provider_release>
      qwen_provider;
  IREE_ASSERT_OK(id4::test::CreateParameterProviderFromFlags(
      IREE_SV("qwen"), qwen_provider.out()));
  id4::test::OwningRef<iree_io_parameter_provider_t,
                       iree_io_parameter_provider_release>
      dit_cond_provider;
  IREE_ASSERT_OK(id4::test::CreateParameterProviderFromFlags(
      IREE_SV("dit_cond"), dit_cond_provider.out()));
  id4::test::OwningRef<iree_io_parameter_provider_t,
                       iree_io_parameter_provider_release>
      dit_uncond_provider;
  IREE_ASSERT_OK(id4::test::CreateParameterProviderFromFlags(
      IREE_SV("dit_uncond"), dit_uncond_provider.out()));
  id4::test::OwningRef<iree_io_parameter_provider_t,
                       iree_io_parameter_provider_release>
      vae_provider;
  IREE_ASSERT_OK(id4::test::CreateParameterProviderFromFlags(
      IREE_SV("vae"), vae_provider.out()));

  id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release>
      qwen_stage;
  IREE_ASSERT_OK(CreateQwenStage(context, IREE_SV("qwen"), qwen_stage.out()));
  id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release>
      dit_cond_stage;
  IREE_ASSERT_OK(
      CreateDitStage(context, IREE_SV("dit_cond"), dit_cond_stage.out()));
  id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release>
      dit_uncond_stage;
  IREE_ASSERT_OK(
      CreateDitStage(context, IREE_SV("dit_uncond"), dit_uncond_stage.out()));
  id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release>
      sampler_stage;
  IREE_ASSERT_OK(CreateSamplerStage(context, sampler_stage.out()));
  id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release>
      decode_stage;
  IREE_ASSERT_OK(
      CreateDecodeStage(context, IREE_SV("vae"), decode_stage.out()));

  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);
  id4_pipeline_stage_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(qwen_stage.get(), &load_options));
  IREE_ASSERT_OK(id4_pipeline_stage_load(dit_cond_stage.get(), &load_options));
  IREE_ASSERT_OK(
      id4_pipeline_stage_load(dit_uncond_stage.get(), &load_options));
  IREE_ASSERT_OK(id4_pipeline_stage_load(sampler_stage.get(), &load_options));
  IREE_ASSERT_OK(id4_pipeline_stage_load(decode_stage.get(), &load_options));

  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release>
      qwen_plan;
  IREE_ASSERT_OK(PlanQwenStage(qwen_stage.get(), qwen_token_count,
                               qwen_token_ids.data(), &diagnostics_sink,
                               qwen_plan.out()));
  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release>
      dit_cond_plan;
  IREE_ASSERT_OK(PlanDitStage(dit_cond_stage.get(), dit_cond_request,
                              dit_cond_tensors,
                              DitExpectedTapSet::kConditionedPrelude,
                              &diagnostics_sink, dit_cond_plan.out()));
  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release>
      dit_uncond_plan;
  IREE_ASSERT_OK(PlanDitStage(dit_uncond_stage.get(), dit_uncond_request,
                              dit_uncond_tensors, DitExpectedTapSet::kAll,
                              &diagnostics_sink, dit_uncond_plan.out()));
  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release>
      sampler_plan;
  IREE_ASSERT_OK(PlanSamplerStage(sampler_stage.get(),
                                  dit_cond_request.latent_shape,
                                  &diagnostics_sink, sampler_plan.out()));
  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release>
      decode_plan;
  IREE_ASSERT_OK(PlanDecodeStage(decode_stage.get(),
                                 dit_cond_request.latent_shape,
                                 &diagnostics_sink, decode_plan.out()));

  id4::test::BufferBindingSet qwen_boundaries;
  IREE_ASSERT_OK(id4::test::AllocateBoundaryBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, qwen_plan.get(),
      &qwen_boundaries));
  id4::test::BufferBindingSet qwen_taps;
  IREE_ASSERT_OK(id4::test::AllocateDiagnosticTapBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, qwen_plan.get(),
      &qwen_taps));
  id4::test::BufferBindingSet dit_cond_boundaries;
  IREE_ASSERT_OK(id4::test::AllocateBoundaryBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, dit_cond_plan.get(),
      &dit_cond_boundaries));
  id4::test::BufferBindingSet dit_cond_taps;
  IREE_ASSERT_OK(id4::test::AllocateDiagnosticTapBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, dit_cond_plan.get(),
      &dit_cond_taps));
  id4::test::BufferBindingSet dit_uncond_boundaries;
  IREE_ASSERT_OK(id4::test::AllocateBoundaryBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, dit_uncond_plan.get(),
      &dit_uncond_boundaries));
  id4::test::BufferBindingSet dit_uncond_taps;
  IREE_ASSERT_OK(id4::test::AllocateDiagnosticTapBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, dit_uncond_plan.get(),
      &dit_uncond_taps));
  id4::test::BufferBindingSet sampler_boundaries;
  IREE_ASSERT_OK(id4::test::AllocateBoundaryBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, sampler_plan.get(),
      &sampler_boundaries));
  id4::test::BufferBindingSet sampler_taps;
  IREE_ASSERT_OK(id4::test::AllocateDiagnosticTapBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, sampler_plan.get(),
      &sampler_taps));
  id4::test::BufferBindingSet decode_boundaries;
  IREE_ASSERT_OK(id4::test::AllocateBoundaryBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, decode_plan.get(),
      &decode_boundaries));
  id4::test::BufferBindingSet decode_taps;
  IREE_ASSERT_OK(id4::test::AllocateDiagnosticTapBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, decode_plan.get(),
      &decode_taps));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      update_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, update_semaphore.out()));
  uint64_t update_value = 0;
  IREE_ASSERT_OK(id4::test::QueueUpdateInitializedBoundaryTensorsFromFixture(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, qwen_plan.get(),
      qwen_boundaries, qwen_tensors, update_semaphore.get(), &update_value));
  IREE_ASSERT_OK(id4::test::QueueUpdateInitializedBoundaryTensorsFromFixture(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, dit_cond_plan.get(),
      dit_cond_boundaries, dit_cond_tensors, update_semaphore.get(),
      &update_value));
  IREE_ASSERT_OK(id4::test::QueueUpdateInitializedBoundaryTensorsFromFixture(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, dit_uncond_plan.get(),
      dit_uncond_boundaries, dit_uncond_tensors, update_semaphore.get(),
      &update_value));
  IREE_ASSERT_OK(id4::test::QueueUpdateBoundaryTensorFromFixture(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, sampler_plan.get(),
      sampler_boundaries, IREE_SV("x_t"), dit_cond_tensors, IREE_SV("x"),
      update_semaphore.get(), &update_value));
  IREE_ASSERT_OK(id4::test::QueueUpdateBoundaryTensorFromFixture(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, sampler_plan.get(),
      sampler_boundaries, IREE_SV("scalings"), sampler_tensors,
      IREE_SV("scalings"), update_semaphore.get(), &update_value));
  const float step_sigmas[2] = {1.0f, 0.0f};
  iree_hal_buffer_binding_t sigmas_binding = {};
  IREE_ASSERT_OK(
      id4::test::FindBoundaryBinding(sampler_plan.get(), sampler_boundaries,
                                     IREE_SV("sigmas"), &sigmas_binding));
  IREE_ASSERT_OK(id4::test::QueueUpdateBinding(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &sigmas_binding,
      step_sigmas, sizeof(step_sigmas), update_semaphore.get(), &update_value));
  IREE_ASSERT_OK(id4::test::QueueUpdateBoundaryTensorFromFixture(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, sampler_plan.get(),
      sampler_boundaries, IREE_SV("guidance"), sampler_tensors,
      IREE_SV("guidance"), update_semaphore.get(), &update_value));
  IREE_ASSERT_OK(id4::test::QueueFillBoundaryTensors(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, qwen_plan.get(),
      qwen_boundaries, ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED,
      &kOutputSentinel, sizeof(kOutputSentinel), update_semaphore.get(),
      &update_value));
  IREE_ASSERT_OK(id4::test::QueueFillBoundaryTensors(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, dit_cond_plan.get(),
      dit_cond_boundaries, ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED,
      &kOutputSentinel, sizeof(kOutputSentinel), update_semaphore.get(),
      &update_value));
  IREE_ASSERT_OK(id4::test::QueueFillBoundaryTensors(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, dit_uncond_plan.get(),
      dit_uncond_boundaries, ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED,
      &kOutputSentinel, sizeof(kOutputSentinel), update_semaphore.get(),
      &update_value));
  IREE_ASSERT_OK(id4::test::QueueFillBoundaryTensors(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, sampler_plan.get(),
      sampler_boundaries, ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED,
      &kOutputSentinel, sizeof(kOutputSentinel), update_semaphore.get(),
      &update_value));
  IREE_ASSERT_OK(id4::test::QueueFillBoundaryTensors(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, decode_plan.get(),
      decode_boundaries, ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED,
      &kOutputSentinel, sizeof(kOutputSentinel), update_semaphore.get(),
      &update_value));
  IREE_ASSERT_OK(id4::test::QueueFillDiagnosticTapTensors(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, dit_cond_taps,
      &kOutputSentinel, sizeof(kOutputSentinel), update_semaphore.get(),
      &update_value));
  IREE_ASSERT_OK(id4::test::QueueFillDiagnosticTapTensors(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, dit_uncond_taps,
      &kOutputSentinel, sizeof(kOutputSentinel), update_semaphore.get(),
      &update_value));
  IREE_ASSERT_OK(id4::test::QueueFillDiagnosticTapTensors(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, sampler_taps,
      &kOutputSentinel, sizeof(kOutputSentinel), update_semaphore.get(),
      &update_value));
  IREE_ASSERT_OK(id4::test::QueueFillDiagnosticTapTensors(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, decode_taps,
      &kOutputSentinel, sizeof(kOutputSentinel), update_semaphore.get(),
      &update_value));

  iree_hal_buffer_binding_t qwen_condition = {};
  IREE_ASSERT_OK(id4::test::FindBoundaryBinding(
      qwen_plan.get(), qwen_boundaries, IREE_SV("condition"), &qwen_condition));
  iree_hal_buffer_binding_t dit_cond_velocity = {};
  IREE_ASSERT_OK(
      id4::test::FindBoundaryBinding(dit_cond_plan.get(), dit_cond_boundaries,
                                     IREE_SV("velocity"), &dit_cond_velocity));
  iree_hal_buffer_binding_t dit_uncond_velocity = {};
  IREE_ASSERT_OK(id4::test::FindBoundaryBinding(
      dit_uncond_plan.get(), dit_uncond_boundaries, IREE_SV("velocity"),
      &dit_uncond_velocity));
  // The conditioned DiT fixture carries the condition tensor from its reference
  // trace. Qwen is validated against its own fixture above; a Qwen-to-DiT
  // semantic handoff test requires both fixtures to come from the same
  // reference request.
  IREE_ASSERT_OK(ReplaceBoundaryBinding(sampler_plan.get(), &sampler_boundaries,
                                        IREE_SV("cond_out"),
                                        dit_cond_velocity));
  IREE_ASSERT_OK(ReplaceBoundaryBinding(sampler_plan.get(), &sampler_boundaries,
                                        IREE_SV("uncond_out"),
                                        dit_uncond_velocity));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      qwen_done;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, qwen_done.out()));
  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      qwen_prepare_done;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, qwen_prepare_done.out()));
  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      dit_cond_done;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, dit_cond_done.out()));
  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      dit_cond_prepare_done;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, dit_cond_prepare_done.out()));
  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      dit_uncond_done;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, dit_uncond_done.out()));
  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      dit_uncond_prepare_done;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, dit_uncond_prepare_done.out()));
  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      sampler_done;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, sampler_done.out()));
  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      decode_done;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, decode_done.out()));
  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      decode_prepare_done;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, decode_prepare_done.out()));

  id4::test::SemaphoreListStorage qwen_prepare_signal;
  qwen_prepare_signal.semaphore = qwen_prepare_done.get();
  qwen_prepare_signal.payload_value = 1;
  id4::test::FixedSemaphoreListStorage qwen_wait;
  IREE_ASSERT_OK(qwen_wait.push(update_semaphore.get(), update_value));
  IREE_ASSERT_OK(qwen_wait.push(qwen_prepare_done.get(), 1));
  id4::test::SemaphoreListStorage qwen_signal;
  qwen_signal.semaphore = qwen_done.get();
  qwen_signal.payload_value = 1;
  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      qwen_bundle;
  IREE_ASSERT_OK(PrepareStage(
      qwen_stage.get(), qwen_plan.get(), qwen_provider.get(),
      kernel_library.get(), iree_hal_semaphore_list_empty(),
      qwen_prepare_signal.list(), &diagnostics_sink, qwen_bundle.out()));
  IREE_ASSERT_OK(IssueStage(qwen_stage.get(), qwen_bundle.get(),
                            qwen_boundaries, qwen_taps, qwen_wait.list(),
                            qwen_signal.list(), &diagnostics_sink));

  id4::test::SemaphoreListStorage qwen_read_wait;
  qwen_read_wait.semaphore = qwen_done.get();
  qwen_read_wait.payload_value = 1;
  const id4::test::FixtureTensor* expected_condition = nullptr;
  IREE_ASSERT_OK(FindFixtureTensor(qwen_tensors, IREE_SV("expected"),
                                   IREE_SV("condition"), &expected_condition));
  iree_hal_buffer_binding_t qwen_selected_hidden_states = {};
  IREE_ASSERT_OK(id4::test::FindDiagnosticTapBinding(
      qwen_plan.get(), qwen_taps, IREE_SV("selected_hidden_states"),
      &qwen_selected_hidden_states));
  IREE_ASSERT_OK(id4::test::CompareF32BindingWithFixtureTensor(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      &qwen_selected_hidden_states, qwen_read_wait.list(),
      *expected_condition));
  IREE_ASSERT_OK(id4::test::CompareF32BindingWithFixtureTensor(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &qwen_condition,
      qwen_read_wait.list(), *expected_condition));
  qwen_bundle.reset();

  id4::test::SemaphoreListStorage dit_cond_prepare_signal;
  dit_cond_prepare_signal.semaphore = dit_cond_prepare_done.get();
  dit_cond_prepare_signal.payload_value = 1;
  id4::test::FixedSemaphoreListStorage dit_cond_wait;
  IREE_ASSERT_OK(dit_cond_wait.push(update_semaphore.get(), update_value));
  IREE_ASSERT_OK(dit_cond_wait.push(dit_cond_prepare_done.get(), 1));
  id4::test::SemaphoreListStorage dit_cond_signal;
  dit_cond_signal.semaphore = dit_cond_done.get();
  dit_cond_signal.payload_value = 1;
  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      dit_cond_bundle;
  IREE_ASSERT_OK(PrepareStage(dit_cond_stage.get(), dit_cond_plan.get(),
                              dit_cond_provider.get(), kernel_library.get(),
                              iree_hal_semaphore_list_empty(),
                              dit_cond_prepare_signal.list(), &diagnostics_sink,
                              dit_cond_bundle.out()));
  IREE_ASSERT_OK(IssueStage(dit_cond_stage.get(), dit_cond_bundle.get(),
                            dit_cond_boundaries, dit_cond_taps,
                            dit_cond_wait.list(), dit_cond_signal.list(),
                            &diagnostics_sink));

  id4::test::SemaphoreListStorage cond_read_wait;
  cond_read_wait.semaphore = dit_cond_done.get();
  cond_read_wait.payload_value = 1;
  IREE_ASSERT_OK(CompareExpectedDiagnosticTaps(
      context.device.get(), dit_cond_plan.get(), dit_cond_taps,
      dit_cond_tensors, DitExpectedTapSet::kConditionedPrelude,
      cond_read_wait.list()));
  dit_cond_bundle.reset();

  id4::test::SemaphoreListStorage dit_uncond_prepare_signal;
  dit_uncond_prepare_signal.semaphore = dit_uncond_prepare_done.get();
  dit_uncond_prepare_signal.payload_value = 1;
  id4::test::FixedSemaphoreListStorage dit_uncond_wait;
  IREE_ASSERT_OK(dit_uncond_wait.push(update_semaphore.get(), update_value));
  IREE_ASSERT_OK(dit_uncond_wait.push(dit_uncond_prepare_done.get(), 1));
  id4::test::SemaphoreListStorage dit_uncond_signal;
  dit_uncond_signal.semaphore = dit_uncond_done.get();
  dit_uncond_signal.payload_value = 1;
  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      dit_uncond_bundle;
  IREE_ASSERT_OK(PrepareStage(dit_uncond_stage.get(), dit_uncond_plan.get(),
                              dit_uncond_provider.get(), kernel_library.get(),
                              iree_hal_semaphore_list_empty(),
                              dit_uncond_prepare_signal.list(),
                              &diagnostics_sink, dit_uncond_bundle.out()));
  IREE_ASSERT_OK(IssueStage(dit_uncond_stage.get(), dit_uncond_bundle.get(),
                            dit_uncond_boundaries, dit_uncond_taps,
                            dit_uncond_wait.list(), dit_uncond_signal.list(),
                            &diagnostics_sink));

  id4::test::SemaphoreListStorage uncond_read_wait;
  uncond_read_wait.semaphore = dit_uncond_done.get();
  uncond_read_wait.payload_value = 1;
  IREE_ASSERT_OK(CompareExpectedDiagnosticTaps(
      context.device.get(), dit_uncond_plan.get(), dit_uncond_taps,
      dit_uncond_tensors, DitExpectedTapSet::kAll, uncond_read_wait.list()));
  dit_uncond_bundle.reset();

  id4::test::FixedSemaphoreListStorage sampler_wait;
  IREE_ASSERT_OK(sampler_wait.push(update_semaphore.get(), update_value));
  IREE_ASSERT_OK(sampler_wait.push(dit_cond_done.get(), 1));
  IREE_ASSERT_OK(sampler_wait.push(dit_uncond_done.get(), 1));
  id4::test::SemaphoreListStorage sampler_signal;
  sampler_signal.semaphore = sampler_done.get();
  sampler_signal.payload_value = 1;
  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      sampler_bundle;
  IREE_ASSERT_OK(PrepareStage(
      sampler_stage.get(), sampler_plan.get(), nullptr, kernel_library.get(),
      iree_hal_semaphore_list_empty(), iree_hal_semaphore_list_empty(),
      &diagnostics_sink, sampler_bundle.out()));
  IREE_ASSERT_OK(IssueStage(sampler_stage.get(), sampler_bundle.get(),
                            sampler_boundaries, sampler_taps,
                            sampler_wait.list(), sampler_signal.list(),
                            &diagnostics_sink));

  iree_hal_buffer_binding_t denoised = {};
  IREE_ASSERT_OK(id4::test::FindBoundaryBinding(
      sampler_plan.get(), sampler_boundaries, IREE_SV("denoised"), &denoised));
  IREE_ASSERT_OK(ReplaceBoundaryBinding(decode_plan.get(), &decode_boundaries,
                                        IREE_SV("media.latent.diffusion"),
                                        denoised));
  id4::test::SemaphoreListStorage sampler_read_wait;
  sampler_read_wait.semaphore = sampler_done.get();
  sampler_read_wait.payload_value = 1;
  IREE_ASSERT_OK(VerifyBindingWasWritten(context.device.get(), &denoised,
                                         sampler_read_wait.list(),
                                         kOutputSentinel));

  id4::test::SemaphoreListStorage decode_prepare_signal;
  decode_prepare_signal.semaphore = decode_prepare_done.get();
  decode_prepare_signal.payload_value = 1;
  id4::test::FixedSemaphoreListStorage decode_wait;
  IREE_ASSERT_OK(decode_wait.push(update_semaphore.get(), update_value));
  IREE_ASSERT_OK(decode_wait.push(sampler_done.get(), 1));
  IREE_ASSERT_OK(decode_wait.push(decode_prepare_done.get(), 1));
  id4::test::SemaphoreListStorage decode_signal;
  decode_signal.semaphore = decode_done.get();
  decode_signal.payload_value = 1;
  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      decode_bundle;
  IREE_ASSERT_OK(PrepareStage(
      decode_stage.get(), decode_plan.get(), vae_provider.get(),
      kernel_library.get(), iree_hal_semaphore_list_empty(),
      decode_prepare_signal.list(), &diagnostics_sink, decode_bundle.out()));
  IREE_ASSERT_OK(IssueStage(decode_stage.get(), decode_bundle.get(),
                            decode_boundaries, decode_taps, decode_wait.list(),
                            decode_signal.list(), &diagnostics_sink));

  iree_hal_buffer_binding_t decoded = {};
  IREE_ASSERT_OK(
      id4::test::FindBoundaryBinding(decode_plan.get(), decode_boundaries,
                                     IREE_SV("media.image.decoded"), &decoded));
  id4::test::SemaphoreListStorage decode_read_wait;
  decode_read_wait.semaphore = decode_done.get();
  decode_read_wait.payload_value = 1;
  IREE_ASSERT_OK(VerifyBindingWasWritten(context.device.get(), &decoded,
                                         decode_read_wait.list(),
                                         kOutputSentinel));
}

}  // namespace
