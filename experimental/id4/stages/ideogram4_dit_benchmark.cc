// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <limits>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/hal_integration_util.h"
#include "experimental/id4/stages/ideogram4_dit.h"
#include "experimental/id4/stages/ideogram4_dit_parameters.h"
#include "experimental/id4/tooling/filesystem.h"
#include "experimental/id4/tooling/runtime.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/io/file_contents.h"
#include "iree/io/parameter_provider.h"
#include "iree/testing/benchmark.h"
#include "iree/tooling/device_util.h"

IREE_FLAG(
    string, id4_fixture_dir, "",
    "Directory containing a full or DiT-only Ideogram4 fixture manifest.");
IREE_FLAG(string, id4_plan_output_dir, "",
          "Optional directory receiving benchmark DiT stage plan JSON files.");
IREE_FLAG(string, dit_parameter_format, "bf16",
          "DiT parameter format: bf16, mixed_bf16_fp8_e4m3, or "
          "mixed_bf16_fp8_e4m3_all_supported.");
IREE_FLAG(string, dit_conditioned_fp8_scope, "dit_cond_fp8",
          "Conditioned DiT native-FP8 parameter scope.");
IREE_FLAG(string, dit_unconditioned_fp8_scope, "dit_uncond_fp8",
          "Unconditioned DiT native-FP8 parameter scope.");

namespace {

enum class DitBenchmarkBranch {
  // Conditioned DiT branch using text condition tokens.
  kConditioned,
  // Unconditioned DiT branch using image tokens only.
  kUnconditioned,
};

enum class DitBenchmarkIssueMode {
  // Include submission and completion wait in the timed region.
  kEndToEnd,
  // Time submission only and pause timing around completion waits.
  kSubmitOnly,
};

struct DitBenchmarkBranchConfig {
  // Parameter scope expected by the stage and --parameters flag.
  iree_string_view_t parameter_scope;
  // Native-FP8 parameter scope used in mixed parameter formats.
  iree_string_view_t fp8_parameter_scope;
  // Fixture stage containing boundary input tensors.
  iree_string_view_t fixture_stage;
  // Dynamic conditioning mode used when planning the DiT request.
  id4_ideogram4_dit_conditioning_mode_t conditioning_mode;
};

struct DitBenchmarkContext {
  // Live HAL, executable cache, and kernel-cache context selected by flags.
  id4::test::LiveStageContext live;
  // Embedded Loom source library used during stage preparation.
  id4::test::KernelLibraryRef kernel_library;
  // Parameter provider created from standard --parameters= flags.
  id4::test::OwningRef<iree_io_parameter_provider_t,
                       iree_io_parameter_provider_release>
      parameter_provider;
  // Loaded DiT stage under benchmark.
  id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release> stage;
  // Fixture tensors used to configure requests and initialize inputs.
  id4::test::FixtureTensorSet fixture_tensors;
  // Dynamic request dimensions inferred from the fixture.
  id4_ideogram4_dit_request_config_t request = {};
  // Diagnostic event counters collected by lifecycle calls.
  id4::test::StageDiagnostics diagnostics = {};
  // Diagnostics sink passed to stage lifecycle calls.
  id4_pipeline_diagnostics_sink_t diagnostics_sink = {};
};

static DitBenchmarkBranchConfig BranchConfig(DitBenchmarkBranch branch) {
  switch (branch) {
    case DitBenchmarkBranch::kConditioned:
      return DitBenchmarkBranchConfig{
          // Conditioned DiT parameter scope.
          /*.parameter_scope=*/IREE_SV("dit_cond"),
          // Conditioned DiT native-FP8 parameter scope.
          /*.fp8_parameter_scope=*/
          iree_make_cstring_view(FLAG_dit_conditioned_fp8_scope),
          // Fixture stage carrying conditioned DiT inputs.
          /*.fixture_stage=*/IREE_SV("ideogram4.cond.input"),
          // Conditioned request consumes Qwen context tokens.
          /*.conditioning_mode=*/
          ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED,
      };
    case DitBenchmarkBranch::kUnconditioned:
      return DitBenchmarkBranchConfig{
          // Unconditioned DiT parameter scope.
          /*.parameter_scope=*/IREE_SV("dit_uncond"),
          // Unconditioned DiT native-FP8 parameter scope.
          /*.fp8_parameter_scope=*/
          iree_make_cstring_view(FLAG_dit_unconditioned_fp8_scope),
          // Fixture stage carrying unconditioned DiT inputs.
          /*.fixture_stage=*/IREE_SV("ideogram4.uncond.input"),
          // Unconditioned request consumes image tokens only.
          /*.conditioning_mode=*/
          ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED,
      };
  }
  return DitBenchmarkBranchConfig{};
}

static iree_string_view_t BranchPlanFileName(DitBenchmarkBranch branch) {
  switch (branch) {
    case DitBenchmarkBranch::kConditioned:
      return IREE_SV("cond.json");
    case DitBenchmarkBranch::kUnconditioned:
      return IREE_SV("uncond.json");
  }
  return iree_string_view_empty();
}

static iree_status_t ParseDitParameterFormat(
    id4_ideogram4_dit_parameter_format_t* out_format) {
  iree_status_t status = id4_ideogram4_dit_parameter_format_parse(
      iree_make_cstring_view(FLAG_dit_parameter_format), out_format);
  if (iree_status_is_ok(status)) return status;
  return iree_status_annotate(status, IREE_SV("--dit_parameter_format"));
}

static iree_status_t WritePlanJsonIfRequested(DitBenchmarkBranch branch,
                                              const id4_pipeline_plan_t* plan) {
  iree_string_view_t output_dir =
      iree_make_cstring_view(FLAG_id4_plan_output_dir);
  if (iree_string_view_is_empty(output_dir)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      id4_tooling_ensure_directory(output_dir, iree_allocator_system()));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  iree_status_t status = id4_pipeline_plan_format_json(plan, &builder);
  iree_string_view_t path = iree_string_view_empty();
  if (iree_status_is_ok(status)) {
    status = id4_tooling_format_child_path(
        output_dir, BranchPlanFileName(branch), iree_allocator_system(), &path);
  }
  if (iree_status_is_ok(status)) {
    iree_string_view_t json = iree_string_builder_view(&builder);
    status = iree_io_file_contents_write(
        path, iree_make_const_byte_span(json.data, json.size),
        iree_allocator_system());
  }
  id4_tooling_free_path(&path, iree_allocator_system());
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t FindFixtureTensor(
    const id4::test::FixtureTensorSet& fixture_tensors, iree_string_view_t role,
    iree_string_view_t stage, iree_string_view_t name,
    const id4::test::FixtureTensor** out_tensor) {
  *out_tensor = fixture_tensors.FindTensor(role, stage, name);
  if (*out_tensor) return iree_ok_status();
  return iree_make_status(
      IREE_STATUS_NOT_FOUND, "fixture tensor `%.*s/%.*s/%.*s` not found",
      static_cast<int>(role.size), role.data, static_cast<int>(stage.size),
      stage.data, static_cast<int>(name.size), name.data);
}

static bool TensorShapeEquals(id4_pipeline_tensor_shape_t lhs,
                              id4_pipeline_tensor_shape_t rhs) {
  if (lhs.rank != rhs.rank) return false;
  for (uint32_t i = 0; i < lhs.rank; ++i) {
    if (lhs.dims[i] != rhs.dims[i]) return false;
  }
  return true;
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

static iree_status_t ConfigureRequestFromFixture(
    const id4::test::FixtureTensorSet& fixture_tensors,
    DitBenchmarkBranchConfig branch,
    id4_ideogram4_dit_request_config_t* out_request) {
  std::memset(out_request, 0, sizeof(*out_request));

  const id4::test::FixtureTensor* latent = nullptr;
  IREE_RETURN_IF_ERROR(FindFixtureTensor(fixture_tensors, IREE_SV("input"),
                                         branch.fixture_stage, IREE_SV("x"),
                                         &latent));
  IREE_RETURN_IF_ERROR(
      MakeProgramShape(latent->shape, &out_request->latent_shape));
  out_request->conditioning_mode = branch.conditioning_mode;

  if (branch.conditioning_mode ==
      ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED) {
    const id4::test::FixtureTensor* condition = nullptr;
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

static iree_status_t CreateDitStage(const id4::test::LiveStageContext& live,
                                    DitBenchmarkBranchConfig branch,
                                    id4_ideogram4_dit_parameter_format_t format,
                                    id4_pipeline_stage_t** out_stage) {
  IREE_ASSERT_ARGUMENT(out_stage);
  *out_stage = nullptr;

  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = live.device_group.get();
  services.executable_cache = live.executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_ideogram4_dit_parameter_source_rule_list_t source_rules;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_parameter_source_rule_list_initialize(
      format, *id4_ideogram4_dit_program_ideogram4_model_config(),
      branch.fp8_parameter_scope, iree_allocator_system(), &source_rules));

  id4_ideogram4_dit_stage_create_options_t create_options;
  std::memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.kernel_cache = live.kernel_cache.get();
  create_options.parameter_scope = branch.parameter_scope;
  create_options.parameter_source_rule_count = source_rules.count;
  create_options.parameter_source_rules = source_rules.values;
  create_options.model = *id4_ideogram4_dit_program_ideogram4_model_config();
  iree_status_t status = id4_ideogram4_dit_stage_create(
      &create_options, iree_allocator_system(), out_stage);
  id4_ideogram4_dit_parameter_source_rule_list_deinitialize(
      &source_rules, iree_allocator_system());
  return status;
}

static iree_status_t LoadFixtureAndConfigureRequest(
    DitBenchmarkContext* context, DitBenchmarkBranchConfig branch) {
  const iree_string_view_t fixture_directory =
      iree_make_cstring_view(FLAG_id4_fixture_dir);
  if (iree_string_view_is_empty(fixture_directory)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--id4_fixture_dir is required");
  }
  IREE_RETURN_IF_ERROR(id4::test::LoadFixtureTensors(
      fixture_directory, &context->fixture_tensors));
  return ConfigureRequestFromFixture(context->fixture_tensors, branch,
                                     &context->request);
}

static iree_status_t CreateLoadedDitStageContext(
    DitBenchmarkBranch branch, DitBenchmarkContext* out_context) {
  IREE_ASSERT_ARGUMENT(out_context);
  const DitBenchmarkBranchConfig branch_config = BranchConfig(branch);
  id4_ideogram4_dit_parameter_format_t parameter_format =
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_INVALID;
  IREE_RETURN_IF_ERROR(ParseDitParameterFormat(&parameter_format));
  out_context->diagnostics_sink =
      id4::test::DiagnosticsSink(&out_context->diagnostics);
  IREE_RETURN_IF_ERROR(
      id4::test::CreateLiveStageContextFromFlags(&out_context->live));
  IREE_RETURN_IF_ERROR(
      LoadFixtureAndConfigureRequest(out_context, branch_config));
  IREE_RETURN_IF_ERROR(CreateDitStage(out_context->live, branch_config,
                                      parameter_format,
                                      out_context->stage.out()));

  id4_pipeline_stage_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &out_context->diagnostics_sink;
  return id4_pipeline_stage_load(out_context->stage.get(), &load_options);
}

static iree_status_t AttachDitPreparationInputs(DitBenchmarkBranch branch,
                                                DitBenchmarkContext* context) {
  IREE_ASSERT_ARGUMENT(context);
  const DitBenchmarkBranchConfig branch_config = BranchConfig(branch);
  IREE_RETURN_IF_ERROR(
      id4::test::CreateEmbeddedKernelLibrary(context->kernel_library.out()));
  id4_ideogram4_dit_parameter_format_t parameter_format =
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_INVALID;
  IREE_RETURN_IF_ERROR(ParseDitParameterFormat(&parameter_format));
  if (parameter_format == ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16) {
    return id4::test::CreateParameterProviderFromFlags(
        branch_config.parameter_scope, context->parameter_provider.out());
  }

  id4::test::OwningRef<iree_io_parameter_provider_t,
                       iree_io_parameter_provider_release>
      bf16_provider;
  id4::test::OwningRef<iree_io_parameter_provider_t,
                       iree_io_parameter_provider_release>
      fp8_provider;
  id4_tooling_parameter_provider_request_t requests[] = {
      {
          // BF16-expanded fallback parameter scope.
          .scope = branch_config.parameter_scope,
          // BF16-expanded provider output.
          .out_provider = bf16_provider.out(),
      },
      {
          // Native-FP8 parameter scope.
          .scope = branch_config.fp8_parameter_scope,
          // Native-FP8 provider output.
          .out_provider = fp8_provider.out(),
      },
  };
  IREE_RETURN_IF_ERROR(id4_tooling_create_parameter_providers_from_flags(
      IREE_ARRAYSIZE(requests), requests, iree_allocator_system()));

  const id4_tooling_parameter_provider_set_entry_t entries[] = {
      {
          // BF16-expanded fallback parameter scope.
          .scope = branch_config.parameter_scope,
          // BF16-expanded provider.
          .provider = bf16_provider.get(),
      },
      {
          // Native-FP8 parameter scope.
          .scope = branch_config.fp8_parameter_scope,
          // Native-FP8 provider.
          .provider = fp8_provider.get(),
      },
  };
  return id4_tooling_create_parameter_provider_set(
      IREE_ARRAYSIZE(entries), entries, iree_allocator_system(),
      context->parameter_provider.out());
}

static iree_status_t CreateLoadedDitBenchmarkContext(
    DitBenchmarkBranch branch, DitBenchmarkContext* out_context) {
  IREE_RETURN_IF_ERROR(CreateLoadedDitStageContext(branch, out_context));
  return AttachDitPreparationInputs(branch, out_context);
}

static iree_status_t CreateDitPlan(DitBenchmarkContext* context,
                                   id4_pipeline_plan_t** out_plan) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = nullptr;

  id4_ideogram4_dit_stage_plan_options_t dit_options;
  std::memset(&dit_options, 0, sizeof(dit_options));
  dit_options.structure_size = sizeof(dit_options);
  dit_options.request = context->request;
  dit_options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;

  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &dit_options;
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &context->diagnostics_sink;
  return id4_pipeline_stage_plan(context->stage.get(), &plan_options, out_plan);
}

static iree_status_t PrepareDitBundle(DitBenchmarkContext* context,
                                      const id4_pipeline_plan_t* plan,
                                      iree_hal_semaphore_t* prepare_semaphore,
                                      uint64_t signal_value,
                                      id4_pipeline_bundle_t** out_bundle) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(out_bundle);
  *out_bundle = nullptr;

  id4::test::SemaphoreListStorage signal;
  signal.semaphore = prepare_semaphore;
  signal.payload_value = signal_value;

  id4_pipeline_stage_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.parameter_provider = context->parameter_provider.get();
  prepare_options.kernel_library = context->kernel_library.get();
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = signal.list();
  prepare_options.command_buffer_mode = context->live.command_buffer_mode;
  prepare_options.diagnostics_sink = &context->diagnostics_sink;
  return id4_pipeline_stage_prepare(context->stage.get(), plan,
                                    &prepare_options, out_bundle);
}

static iree_status_t FindBoundaryPlan(
    const id4_pipeline_plan_t* plan, iree_string_view_t name,
    const id4_pipeline_boundary_tensor_plan_t** out_boundary) {
  for (iree_host_size_t i = 0;
       i < id4_pipeline_plan_boundary_tensor_count(plan); ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (boundary && iree_string_view_equal(boundary->layout.name, name)) {
      *out_boundary = boundary;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "boundary tensor `%.*s` not found",
                          static_cast<int>(name.size), name.data);
}

static iree_status_t QueueUpdateBoundaryFromFixture(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan,
    const id4::test::BufferBindingSet& binding_set,
    iree_string_view_t boundary_name, const id4::test::FixtureTensor& tensor,
    iree_hal_semaphore_t* update_semaphore, uint64_t* inout_update_value) {
  const id4_pipeline_boundary_tensor_plan_t* boundary = nullptr;
  IREE_RETURN_IF_ERROR(FindBoundaryPlan(plan, boundary_name, &boundary));
  if (tensor.dtype != boundary->layout.dtype) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fixture tensor `%s/%s` dtype does not match boundary `%.*s` dtype",
        tensor.stage.c_str(), tensor.name.c_str(),
        static_cast<int>(boundary_name.size), boundary_name.data);
  }
  if (!TensorShapeEquals(tensor.shape, boundary->layout.shape)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fixture tensor `%s/%s` shape does not match boundary `%.*s` shape",
        tensor.stage.c_str(), tensor.name.c_str(),
        static_cast<int>(boundary_name.size), boundary_name.data);
  }

  iree_hal_buffer_binding_t binding = {};
  IREE_RETURN_IF_ERROR(id4::test::FindBoundaryBinding(plan, binding_set,
                                                      boundary_name, &binding));
  return id4::test::QueueReadBindingFromHostAllocation(
      device, queue_affinity, &binding, tensor.payload.data(),
      tensor.payload.size(), update_semaphore, inout_update_value);
}

static iree_status_t QueueUpdateBoundaryFromFixture(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan,
    const id4::test::BufferBindingSet& binding_set,
    const id4::test::FixtureTensorSet& fixture_tensors,
    iree_string_view_t fixture_stage, iree_string_view_t boundary_name,
    iree_string_view_t fixture_name, iree_hal_semaphore_t* update_semaphore,
    uint64_t* inout_update_value) {
  const id4::test::FixtureTensor* tensor = nullptr;
  IREE_RETURN_IF_ERROR(FindFixtureTensor(fixture_tensors, IREE_SV("input"),
                                         fixture_stage, fixture_name, &tensor));
  return QueueUpdateBoundaryFromFixture(device, queue_affinity, plan,
                                        binding_set, boundary_name, *tensor,
                                        update_semaphore, inout_update_value);
}

static iree_status_t QueueUpdateBranchInputsFromFixture(
    DitBenchmarkBranch branch, DitBenchmarkContext* context,
    const id4_pipeline_plan_t* plan,
    const id4::test::BufferBindingSet& boundary_bindings,
    iree_hal_semaphore_t* update_semaphore, uint64_t* inout_update_value) {
  const DitBenchmarkBranchConfig branch_config = BranchConfig(branch);
  IREE_RETURN_IF_ERROR(QueueUpdateBoundaryFromFixture(
      context->live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan,
      boundary_bindings, context->fixture_tensors, branch_config.fixture_stage,
      IREE_SV("x"), IREE_SV("x"), update_semaphore, inout_update_value));
  IREE_RETURN_IF_ERROR(QueueUpdateBoundaryFromFixture(
      context->live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan,
      boundary_bindings, context->fixture_tensors, branch_config.fixture_stage,
      IREE_SV("timestep"), IREE_SV("timestep"), update_semaphore,
      inout_update_value));
  IREE_RETURN_IF_ERROR(QueueUpdateBoundaryFromFixture(
      context->live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan,
      boundary_bindings, context->fixture_tensors, branch_config.fixture_stage,
      IREE_SV("image_indicator"), IREE_SV("image_indicator"), update_semaphore,
      inout_update_value));
  IREE_RETURN_IF_ERROR(QueueUpdateBoundaryFromFixture(
      context->live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan,
      boundary_bindings, context->fixture_tensors, branch_config.fixture_stage,
      IREE_SV("position_embedding"), IREE_SV("position_embedding"),
      update_semaphore, inout_update_value));
  if (branch_config.conditioning_mode ==
      ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED) {
    IREE_RETURN_IF_ERROR(QueueUpdateBoundaryFromFixture(
        context->live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan,
        boundary_bindings, context->fixture_tensors,
        branch_config.fixture_stage, IREE_SV("condition"), IREE_SV("context"),
        update_semaphore, inout_update_value));
  }
  return iree_ok_status();
}

static iree_status_t IssueDitBundle(
    DitBenchmarkContext* context, id4_pipeline_bundle_t* bundle,
    const id4::test::BufferBindingSet& boundary_bindings,
    iree_hal_semaphore_t* wait_semaphore, uint64_t wait_value,
    iree_hal_semaphore_t* signal_semaphore, uint64_t signal_value) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(bundle);

  id4::test::SemaphoreListStorage wait;
  wait.semaphore = wait_semaphore;
  wait.payload_value = wait_value;
  id4::test::SemaphoreListStorage signal;
  signal.semaphore = signal_semaphore;
  signal.payload_value = signal_value;

  id4_pipeline_stage_issue_options_t issue_options;
  std::memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.boundary_binding_count = boundary_bindings.count;
  issue_options.boundary_bindings = boundary_bindings.bindings;
  issue_options.wait_semaphore_list = wait.list();
  issue_options.signal_semaphore_list = signal.list();
  issue_options.diagnostics_sink = &context->diagnostics_sink;
  return id4_pipeline_stage_issue(context->stage.get(), bundle, &issue_options);
}

static iree_status_t WaitForSemaphore(iree_hal_semaphore_t* semaphore,
                                      uint64_t payload_value) {
  return iree_hal_semaphore_wait(semaphore, payload_value,
                                 iree_infinite_timeout(),
                                 IREE_ASYNC_WAIT_FLAG_NONE);
}

static const iree_benchmark_def_t* RegisterDitBenchmark(
    iree_string_view_t name, iree_benchmark_fn_t run,
    iree_benchmark_unit_t time_unit) {
  iree_benchmark_def_t* benchmark = iree_make_function_benchmark(run);
  benchmark->flags = IREE_BENCHMARK_FLAG_USE_REAL_TIME;
  benchmark->time_unit = time_unit;
  return iree_benchmark_register(name, benchmark);
}

#define ID4_DIT_BENCHMARK_REGISTER(name, time_unit)                 \
  static const iree_benchmark_def_t* name##_registration            \
      IREE_ATTRIBUTE_UNUSED =                                       \
          RegisterDitBenchmark(iree_make_cstring_view(#name), name, \
                               IREE_BENCHMARK_UNIT_##time_unit)

static iree_status_t RunPlanBenchmark(iree_benchmark_state_t* benchmark_state,
                                      DitBenchmarkBranch branch) {
  DitBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedDitStageContext(branch, &context));

  uint64_t iteration_count = 0;
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    id4_pipeline_plan_t* plan = nullptr;
    IREE_RETURN_IF_ERROR(CreateDitPlan(&context, &plan));
    iree_optimization_barrier(plan);
    id4_pipeline_plan_release(plan);
    ++iteration_count;
  }
  iree_benchmark_set_items_processed(
      benchmark_state,
      static_cast<int64_t>(iteration_count *
                           context.request.latent_shape.dims[0] *
                           context.request.latent_shape.dims[1]));
  return iree_ok_status();
}

IREE_BENCHMARK_FN(BM_Ideogram4DitStagePlanConditionedFixture) {
  return RunPlanBenchmark(benchmark_state, DitBenchmarkBranch::kConditioned);
}
ID4_DIT_BENCHMARK_REGISTER(BM_Ideogram4DitStagePlanConditionedFixture,
                           MICROSECOND);

IREE_BENCHMARK_FN(BM_Ideogram4DitStagePlanUnconditionedFixture) {
  return RunPlanBenchmark(benchmark_state, DitBenchmarkBranch::kUnconditioned);
}
ID4_DIT_BENCHMARK_REGISTER(BM_Ideogram4DitStagePlanUnconditionedFixture,
                           MICROSECOND);

static iree_status_t RunPrepareBenchmark(
    iree_benchmark_state_t* benchmark_state, DitBenchmarkBranch branch) {
  DitBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedDitBenchmarkContext(branch, &context));

  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_RETURN_IF_ERROR(CreateDitPlan(&context, plan.out()));
  IREE_RETURN_IF_ERROR(WritePlanJsonIfRequested(branch, plan.get()));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      prepare_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, prepare_semaphore.out()));

  uint64_t prepare_value = 1;
  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      warm_bundle;
  IREE_RETURN_IF_ERROR(PrepareDitBundle(&context, plan.get(),
                                        prepare_semaphore.get(), prepare_value,
                                        warm_bundle.out()));
  IREE_RETURN_IF_ERROR(
      WaitForSemaphore(prepare_semaphore.get(), prepare_value));
  warm_bundle.reset();

  uint64_t iteration_count = 0;
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    ++prepare_value;
    id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
        bundle;
    IREE_RETURN_IF_ERROR(PrepareDitBundle(&context, plan.get(),
                                          prepare_semaphore.get(),
                                          prepare_value, bundle.out()));
    IREE_RETURN_IF_ERROR(
        WaitForSemaphore(prepare_semaphore.get(), prepare_value));
    iree_optimization_barrier(bundle.get());
    ++iteration_count;
  }
  iree_benchmark_set_items_processed(
      benchmark_state,
      static_cast<int64_t>(iteration_count *
                           context.request.latent_shape.dims[0] *
                           context.request.latent_shape.dims[1]));
  return iree_ok_status();
}

IREE_BENCHMARK_FN(BM_Ideogram4DitStagePrepareConditionedFixture) {
  return RunPrepareBenchmark(benchmark_state, DitBenchmarkBranch::kConditioned);
}
ID4_DIT_BENCHMARK_REGISTER(BM_Ideogram4DitStagePrepareConditionedFixture,
                           MILLISECOND);

IREE_BENCHMARK_FN(BM_Ideogram4DitStagePrepareUnconditionedFixture) {
  return RunPrepareBenchmark(benchmark_state,
                             DitBenchmarkBranch::kUnconditioned);
}
ID4_DIT_BENCHMARK_REGISTER(BM_Ideogram4DitStagePrepareUnconditionedFixture,
                           MILLISECOND);

static iree_status_t RunIssueBenchmark(iree_benchmark_state_t* benchmark_state,
                                       DitBenchmarkBranch branch,
                                       DitBenchmarkIssueMode issue_mode) {
  DitBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedDitBenchmarkContext(branch, &context));

  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_RETURN_IF_ERROR(CreateDitPlan(&context, plan.out()));
  IREE_RETURN_IF_ERROR(WritePlanJsonIfRequested(branch, plan.get()));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      prepare_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, prepare_semaphore.out()));

  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      bundle;
  IREE_RETURN_IF_ERROR(PrepareDitBundle(
      &context, plan.get(), prepare_semaphore.get(), 1, bundle.out()));
  IREE_RETURN_IF_ERROR(WaitForSemaphore(prepare_semaphore.get(), 1));

  id4::test::BufferBindingSet boundary_bindings;
  IREE_RETURN_IF_ERROR(id4::test::AllocateBoundaryBindings(
      context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      &boundary_bindings));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      update_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, update_semaphore.out()));
  uint64_t update_value = 0;
  IREE_RETURN_IF_ERROR(QueueUpdateBranchInputsFromFixture(
      branch, &context, plan.get(), boundary_bindings, update_semaphore.get(),
      &update_value));
  const uint32_t sentinel_pattern = 0xA5A5A5A5u;
  IREE_RETURN_IF_ERROR(id4::test::QueueFillBoundaryTensors(
      context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      boundary_bindings, ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED,
      &sentinel_pattern, sizeof(sentinel_pattern), update_semaphore.get(),
      &update_value));
  if (update_value != 0) {
    IREE_RETURN_IF_ERROR(
        WaitForSemaphore(update_semaphore.get(), update_value));
  }

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      issue_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, issue_semaphore.out()));

  iree_hal_semaphore_t* wait_semaphore = update_semaphore.get();
  uint64_t wait_value = update_value;
  uint64_t signal_value = 0;
  uint64_t iteration_count = 0;
  iree_hal_profiling_from_flags_t* profiling = nullptr;
  iree_status_t status = iree_hal_begin_device_group_profiling_from_flags(
      context.live.device_group.get(), iree_allocator_system(), &profiling);
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    ++signal_value;
    status = IssueDitBundle(&context, bundle.get(), boundary_bindings,
                            wait_semaphore, wait_value, issue_semaphore.get(),
                            signal_value);
    bool timing_paused = false;
    if (iree_status_is_ok(status) &&
        issue_mode == DitBenchmarkIssueMode::kSubmitOnly) {
      iree_benchmark_pause_timing(benchmark_state);
      timing_paused = true;
    }
    if (iree_status_is_ok(status)) {
      status = WaitForSemaphore(issue_semaphore.get(), signal_value);
    }
    if (timing_paused) {
      iree_benchmark_resume_timing(benchmark_state);
    }
    if (iree_status_is_ok(status)) {
      wait_semaphore = issue_semaphore.get();
      wait_value = signal_value;
      ++iteration_count;
    }
  }
  status =
      iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
  IREE_RETURN_IF_ERROR(status);
  iree_benchmark_set_items_processed(
      benchmark_state,
      static_cast<int64_t>(iteration_count *
                           context.request.latent_shape.dims[0] *
                           context.request.latent_shape.dims[1]));
  return iree_ok_status();
}

IREE_BENCHMARK_FN(BM_Ideogram4DitStageIssueConditionedSubmitOnly) {
  return RunIssueBenchmark(benchmark_state, DitBenchmarkBranch::kConditioned,
                           DitBenchmarkIssueMode::kSubmitOnly);
}
ID4_DIT_BENCHMARK_REGISTER(BM_Ideogram4DitStageIssueConditionedSubmitOnly,
                           MICROSECOND);

IREE_BENCHMARK_FN(BM_Ideogram4DitStageIssueConditionedEndToEnd) {
  return RunIssueBenchmark(benchmark_state, DitBenchmarkBranch::kConditioned,
                           DitBenchmarkIssueMode::kEndToEnd);
}
ID4_DIT_BENCHMARK_REGISTER(BM_Ideogram4DitStageIssueConditionedEndToEnd,
                           MILLISECOND);

IREE_BENCHMARK_FN(BM_Ideogram4DitStageIssueUnconditionedSubmitOnly) {
  return RunIssueBenchmark(benchmark_state, DitBenchmarkBranch::kUnconditioned,
                           DitBenchmarkIssueMode::kSubmitOnly);
}
ID4_DIT_BENCHMARK_REGISTER(BM_Ideogram4DitStageIssueUnconditionedSubmitOnly,
                           MICROSECOND);

IREE_BENCHMARK_FN(BM_Ideogram4DitStageIssueUnconditionedEndToEnd) {
  return RunIssueBenchmark(benchmark_state, DitBenchmarkBranch::kUnconditioned,
                           DitBenchmarkIssueMode::kEndToEnd);
}
ID4_DIT_BENCHMARK_REGISTER(BM_Ideogram4DitStageIssueUnconditionedEndToEnd,
                           MILLISECOND);

#undef ID4_DIT_BENCHMARK_REGISTER

}  // namespace
