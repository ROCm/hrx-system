// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <limits>
#include <vector>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/hal_integration_util.h"
#include "experimental/id4/stages/ideogram4_dit.h"
#include "experimental/id4/stages/ideogram4_dit_parameters.h"
#include "experimental/id4/tooling/capture.h"
#include "experimental/id4/tooling/runtime.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

IREE_FLAG(string, id4_capture_dir, "",
          "Directory that receives exported ID4 DiT boundary and diagnostic "
          "tap tensor captures.");
IREE_FLAG_LIST(string, id4_diagnostic_tap,
               "Additional diagnostic tap names to capture.");
IREE_FLAG(
    string, id4_fixture_dir, "",
    "Directory containing an ID4 DiT fixture manifest and tensor "
    "payloads used to initialize stage inputs and verify reference taps.");
IREE_FLAG(string, id4_plan_output, "",
          "Path that receives the planned ID4 DiT stage JSON before "
          "preparation.");
IREE_FLAG(string, dit_parameter_format, "bf16",
          "DiT parameter format: bf16, mixed_bf16_fp8_e4m3, or "
          "mixed_bf16_fp8_e4m3_all_supported.");
IREE_FLAG(string, dit_conditioned_fp8_scope, "dit_cond_fp8",
          "Conditioned DiT native-FP8 parameter scope.");
IREE_FLAG(string, dit_unconditioned_fp8_scope, "dit_uncond_fp8",
          "Unconditioned DiT native-FP8 parameter scope.");

namespace {

constexpr uint8_t kOutputSentinel = 0xA5;

using ParameterProviderRef =
    id4::test::OwningRef<iree_io_parameter_provider_t,
                         iree_io_parameter_provider_release>;

static iree_status_t ParseDitParameterFormat(
    id4_ideogram4_dit_parameter_format_t* out_format) {
  iree_status_t status = id4_ideogram4_dit_parameter_format_parse(
      iree_make_cstring_view(FLAG_dit_parameter_format), out_format);
  if (iree_status_is_ok(status)) return status;
  return iree_status_annotate(status, IREE_SV("--dit_parameter_format"));
}

static iree_string_view_t Fp8ParameterScopeForRequest(
    const id4_ideogram4_dit_request_config_t& request) {
  switch (request.conditioning_mode) {
    case ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED:
      return iree_make_cstring_view(FLAG_dit_conditioned_fp8_scope);
    case ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED:
      return iree_make_cstring_view(FLAG_dit_unconditioned_fp8_scope);
    default:
      return iree_string_view_empty();
  }
}

static iree_status_t CreateIdeogram4DitStage(
    const id4::test::LiveStageContext& context,
    id4_ideogram4_dit_parameter_format_t parameter_format,
    iree_string_view_t fp8_parameter_scope, id4_pipeline_stage_t** out_stage) {
  IREE_ASSERT_ARGUMENT(out_stage);
  *out_stage = nullptr;

  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = context.device_group.get();
  services.executable_cache = context.executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_ideogram4_dit_parameter_source_rule_list_t source_rules;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_parameter_source_rule_list_initialize(
      parameter_format, *id4_ideogram4_dit_program_ideogram4_model_config(),
      fp8_parameter_scope, iree_allocator_system(), &source_rules));

  id4_ideogram4_dit_stage_create_options_t create_options;
  std::memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.kernel_cache = context.kernel_cache.get();
  create_options.model = *id4_ideogram4_dit_program_ideogram4_model_config();
  create_options.parameter_source_rule_count = source_rules.count;
  create_options.parameter_source_rules = source_rules.values;
  iree_status_t status = id4_ideogram4_dit_stage_create(
      &create_options, iree_allocator_system(), out_stage);
  id4_ideogram4_dit_parameter_source_rule_list_deinitialize(
      &source_rules, iree_allocator_system());
  return status;
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

static iree_status_t FindFixtureTensor(
    const id4::test::FixtureTensorSet& fixture_tensors, iree_string_view_t role,
    iree_string_view_t stage, iree_string_view_t name,
    const id4::test::FixtureTensor** out_tensor) {
  *out_tensor = fixture_tensors.FindTensor(role, stage, name);
  if (*out_tensor) return iree_ok_status();
  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "fixture tensor `%.*s` with role `%.*s` and stage `%.*s` not found",
      static_cast<int>(name.size), name.data, static_cast<int>(role.size),
      role.data, static_cast<int>(stage.size), stage.data);
}

static iree_string_view_t FixtureTensorName(
    const id4::test::FixtureTensor& tensor) {
  return iree_make_string_view(tensor.name.data(), tensor.name.size());
}

static iree_string_view_t FixtureTensorRole(
    const id4::test::FixtureTensor& tensor) {
  return iree_make_string_view(tensor.role.data(), tensor.role.size());
}

static iree_string_view_t FixtureTensorStage(
    const id4::test::FixtureTensor& tensor) {
  return iree_make_string_view(tensor.stage.data(), tensor.stage.size());
}

static bool TensorShapeEquals(id4_pipeline_tensor_shape_t lhs,
                              id4_pipeline_tensor_shape_t rhs) {
  if (lhs.rank != rhs.rank) return false;
  for (uint32_t i = 0; i < lhs.rank; ++i) {
    if (lhs.dims[i] != rhs.dims[i]) return false;
  }
  return true;
}

static iree_string_view_t FixtureNameForBoundary(
    const id4_pipeline_boundary_tensor_plan_t* boundary) {
  if (iree_string_view_equal(boundary->layout.name, IREE_SV("condition"))) {
    return IREE_SV("context");
  }
  return boundary->layout.name;
}

static bool ExpectedTapMatchesRequestExtent(
    const id4::test::FixtureTensor& tensor,
    const id4_ideogram4_dit_request_config_t& request) {
  const iree_string_view_t tensor_name = FixtureTensorName(tensor);
  if (!iree_string_view_ends_with(tensor_name, IREE_SV(".output.velocity"))) {
    return true;
  }
  const id4_pipeline_program_shape_t latent_shape = request.latent_shape;
  return tensor.source_shape.rank == 3 && latent_shape.rank == 4 &&
         tensor.source_shape.dims[0] == latent_shape.dims[0] &&
         tensor.source_shape.dims[1] == latent_shape.dims[1] &&
         tensor.source_shape.dims[2] == latent_shape.dims[2];
}

static iree_status_t FindDiagnosticTapPlan(
    const id4_pipeline_plan_t* plan, iree_string_view_t name,
    const id4_pipeline_diagnostic_tap_plan_t** out_diagnostic_tap) {
  *out_diagnostic_tap = nullptr;
  const iree_host_size_t diagnostic_tap_count =
      id4_pipeline_plan_diagnostic_tap_count(plan);
  for (iree_host_size_t i = 0; i < diagnostic_tap_count; ++i) {
    const id4_pipeline_diagnostic_tap_plan_t* diagnostic_tap =
        id4_pipeline_plan_diagnostic_tap_at(plan, i);
    if (diagnostic_tap && iree_string_view_equal(diagnostic_tap->name, name)) {
      *out_diagnostic_tap = diagnostic_tap;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "diagnostic tap `%.*s` not found",
                          static_cast<int>(name.size), name.data);
}

static iree_status_t FindBoundaryPlan(
    const id4_pipeline_plan_t* plan, iree_string_view_t name,
    const id4_pipeline_boundary_tensor_plan_t** out_boundary) {
  *out_boundary = nullptr;
  const iree_host_size_t boundary_count =
      id4_pipeline_plan_boundary_tensor_count(plan);
  for (iree_host_size_t i = 0; i < boundary_count; ++i) {
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
    iree_string_view_t input_stage,
    id4_ideogram4_dit_request_config_t* out_request) {
  std::memset(out_request, 0, sizeof(*out_request));

  const id4::test::FixtureTensor* latent = nullptr;
  IREE_RETURN_IF_ERROR(FindFixtureTensor(fixture_tensors, IREE_SV("input"),
                                         input_stage, IREE_SV("x"), &latent));
  IREE_RETURN_IF_ERROR(
      MakeProgramShape(latent->shape, &out_request->latent_shape));

  const id4::test::FixtureTensor* context = fixture_tensors.FindTensor(
      IREE_SV("input"), input_stage, IREE_SV("context"));
  if (!context) {
    out_request->conditioning_mode =
        ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED;
    return iree_ok_status();
  }
  if (context->shape.rank != 2 ||
      context->shape.dims[1] >
          static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "context fixture tensor must be rank-2 with uint32 token count");
  }
  out_request->conditioning_mode =
      ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED;
  out_request->text_token_count = static_cast<uint32_t>(context->shape.dims[1]);
  return iree_ok_status();
}

static iree_status_t WritePlanJsonIfRequested(const id4_pipeline_plan_t* plan) {
  iree_string_view_t plan_output = iree_make_cstring_view(FLAG_id4_plan_output);
  if (iree_string_view_is_empty(plan_output)) return iree_ok_status();

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  iree_status_t status = id4_pipeline_plan_format_json(plan, &builder);
  if (iree_status_is_ok(status)) {
    iree_string_view_t json = iree_string_builder_view(&builder);
    status = iree_io_file_contents_write(
        plan_output, iree_make_const_byte_span(json.data, json.size),
        iree_allocator_system());
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t CreateDitParameterProviderFromFlags(
    id4_ideogram4_dit_parameter_format_t parameter_format,
    iree_string_view_t fp8_parameter_scope,
    iree_io_parameter_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = nullptr;
  if (parameter_format == ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16) {
    return id4::test::CreateParameterProviderFromFlags(iree_string_view_empty(),
                                                       out_provider);
  }

  ParameterProviderRef bf16_provider;
  ParameterProviderRef fp8_provider;
  id4_tooling_parameter_provider_request_t requests[] = {
      {
          // BF16-expanded fallback parameter scope.
          .scope = iree_string_view_empty(),
          // BF16-expanded provider output.
          .out_provider = bf16_provider.out(),
      },
      {
          // Native-FP8 parameter scope.
          .scope = fp8_parameter_scope,
          // Native-FP8 provider output.
          .out_provider = fp8_provider.out(),
      },
  };
  IREE_RETURN_IF_ERROR(id4_tooling_create_parameter_providers_from_flags(
      IREE_ARRAYSIZE(requests), requests, iree_allocator_system()));

  const id4_tooling_parameter_provider_set_entry_t entries[] = {
      {
          // BF16-expanded fallback parameter scope.
          .scope = iree_string_view_empty(),
          // BF16-expanded provider.
          .provider = bf16_provider.get(),
      },
      {
          // Native-FP8 parameter scope.
          .scope = fp8_parameter_scope,
          // Native-FP8 provider.
          .provider = fp8_provider.get(),
      },
  };
  return id4_tooling_create_parameter_provider_set(
      IREE_ARRAYSIZE(entries), entries, iree_allocator_system(), out_provider);
}

static id4_pipeline_tensor_layout_t FixtureComparisonLayout(
    const id4_pipeline_tensor_layout_t* actual_layout,
    const id4::test::FixtureTensor& expected_tensor) {
  id4_pipeline_tensor_layout_t layout = *actual_layout;
  if (layout.shape.rank == expected_tensor.source_shape.rank + 1 &&
      layout.shape.dims[expected_tensor.source_shape.rank] == 1) {
    layout.shape = expected_tensor.source_shape;
  }
  return layout;
}

static iree_status_t CompareExpectedDiagnosticTaps(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan,
    const id4::test::BufferBindingSet& diagnostic_tap_bindings,
    const id4::test::FixtureTensorSet& fixture_tensors,
    const id4_ideogram4_dit_request_config_t& request,
    iree_string_view_t expected_tap_prefix,
    iree_hal_semaphore_list_t wait_list) {
  iree_host_size_t expected_count = 0;
  for (const id4::test::FixtureTensor& tensor : fixture_tensors.tensors) {
    if (!iree_string_view_equal(FixtureTensorRole(tensor),
                                IREE_SV("expected"))) {
      continue;
    }
    if (!iree_string_view_starts_with(FixtureTensorName(tensor),
                                      expected_tap_prefix)) {
      continue;
    }
    if (!ExpectedTapMatchesRequestExtent(tensor, request)) continue;
    ++expected_count;
    iree_hal_buffer_binding_t binding = {};
    const id4_pipeline_diagnostic_tap_plan_t* diagnostic_tap = nullptr;
    IREE_RETURN_IF_ERROR(FindDiagnosticTapPlan(plan, FixtureTensorName(tensor),
                                               &diagnostic_tap));
    IREE_RETURN_IF_ERROR(id4::test::FindDiagnosticTapBinding(
        plan, diagnostic_tap_bindings, FixtureTensorName(tensor), &binding));
    const id4_pipeline_tensor_layout_t comparison_layout =
        FixtureComparisonLayout(&diagnostic_tap->layout, tensor);
    IREE_RETURN_IF_ERROR(id4::test::CompareBindingWithFixtureTensor(
        device, queue_affinity, &binding, wait_list, &comparison_layout,
        tensor));
  }
  if (expected_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "DiT fixture contains no expected tensors");
  }
  return iree_ok_status();
}

static iree_status_t QueueUpdateInitializedBoundariesFromFixture(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan,
    const id4::test::BufferBindingSet& boundary_bindings,
    const id4::test::FixtureTensorSet& fixture_tensors,
    iree_string_view_t input_stage, iree_hal_semaphore_t* update_semaphore,
    uint64_t* inout_update_value) {
  for (iree_host_size_t i = 0; i < boundary_bindings.count; ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (!boundary ||
        !iree_all_bits_set(boundary->flags,
                           ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_INITIALIZED)) {
      continue;
    }
    const id4::test::FixtureTensor* fixture_tensor = nullptr;
    IREE_RETURN_IF_ERROR(
        FindFixtureTensor(fixture_tensors, IREE_SV("input"), input_stage,
                          FixtureNameForBoundary(boundary), &fixture_tensor));
    if (fixture_tensor->dtype != boundary->layout.dtype) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "fixture tensor `%.*s/%.*s` dtype does not match boundary `%.*s`",
          static_cast<int>(FixtureTensorStage(*fixture_tensor).size),
          FixtureTensorStage(*fixture_tensor).data,
          static_cast<int>(FixtureTensorName(*fixture_tensor).size),
          FixtureTensorName(*fixture_tensor).data,
          static_cast<int>(boundary->layout.name.size),
          boundary->layout.name.data);
    }
    if (!TensorShapeEquals(fixture_tensor->shape, boundary->layout.shape)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "fixture tensor `%.*s/%.*s` shape does not match boundary `%.*s`",
          static_cast<int>(FixtureTensorStage(*fixture_tensor).size),
          FixtureTensorStage(*fixture_tensor).data,
          static_cast<int>(FixtureTensorName(*fixture_tensor).size),
          FixtureTensorName(*fixture_tensor).data,
          static_cast<int>(boundary->layout.name.size),
          boundary->layout.name.data);
    }
    IREE_RETURN_IF_ERROR(id4::test::QueueReadBindingFromHostAllocation(
        device, queue_affinity, &boundary_bindings.bindings[i],
        fixture_tensor->payload.data(), fixture_tensor->payload.size(),
        update_semaphore, inout_update_value));
  }
  return iree_ok_status();
}

static iree_status_t CompareExpectedOutputBoundary(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan,
    const id4::test::BufferBindingSet& boundary_bindings,
    const id4::test::FixtureTensorSet& fixture_tensors,
    iree_string_view_t expected_output_name,
    iree_hal_semaphore_list_t wait_list) {
  const id4::test::FixtureTensor* expected_output = nullptr;
  IREE_RETURN_IF_ERROR(FindFixtureTensor(fixture_tensors, IREE_SV("expected"),
                                         expected_output_name,
                                         &expected_output));

  const id4_pipeline_boundary_tensor_plan_t* boundary = nullptr;
  IREE_RETURN_IF_ERROR(FindBoundaryPlan(plan, IREE_SV("velocity"), &boundary));
  if (!iree_all_bits_set(boundary->flags,
                         ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "velocity boundary is not exported");
  }

  iree_hal_buffer_binding_t binding = {};
  IREE_RETURN_IF_ERROR(id4::test::FindBoundaryBinding(
      plan, boundary_bindings, IREE_SV("velocity"), &binding));
  const id4_pipeline_tensor_layout_t comparison_layout =
      FixtureComparisonLayout(&boundary->layout, *expected_output);
  return id4::test::CompareBindingWithFixtureTensor(
      device, queue_affinity, &binding, wait_list, &comparison_layout,
      *expected_output);
}

static iree_status_t VerifyExportedBoundariesWereWritten(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan,
    const id4::test::BufferBindingSet& boundary_bindings,
    iree_hal_semaphore_list_t wait_list) {
  iree_host_size_t exported_count = 0;
  for (iree_host_size_t i = 0;
       i < id4_pipeline_plan_boundary_tensor_count(plan); ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (!boundary ||
        !iree_all_bits_set(boundary->flags,
                           ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED)) {
      continue;
    }
    ++exported_count;
    std::vector<uint8_t> bytes;
    IREE_RETURN_IF_ERROR(id4::test::ReadBindingToHost(
        device, queue_affinity, &boundary_bindings.bindings[i], wait_list,
        &bytes));
    bool all_sentinel = true;
    for (uint8_t byte : bytes) {
      if (byte != kOutputSentinel) {
        all_sentinel = false;
        break;
      }
    }
    if (all_sentinel) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "exported boundary `%.*s` was not written",
                              static_cast<int>(boundary->layout.name.size),
                              boundary->layout.name.data);
    }
  }
  if (exported_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "DiT plan contains no exported boundaries");
  }
  return iree_ok_status();
}

static iree_status_t VerifyDiagnosticTapsWereWritten(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan,
    const id4::test::BufferBindingSet& diagnostic_tap_bindings,
    iree_hal_semaphore_list_t wait_list) {
  const iree_host_size_t diagnostic_tap_count =
      id4_pipeline_plan_diagnostic_tap_count(plan);
  if (diagnostic_tap_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "DiT plan contains no diagnostic taps");
  }
  for (iree_host_size_t i = 0; i < diagnostic_tap_count; ++i) {
    const id4_pipeline_diagnostic_tap_plan_t* diagnostic_tap =
        id4_pipeline_plan_diagnostic_tap_at(plan, i);
    std::vector<uint8_t> bytes;
    IREE_RETURN_IF_ERROR(id4::test::ReadBindingToHost(
        device, queue_affinity, &diagnostic_tap_bindings.bindings[i], wait_list,
        &bytes));
    bool all_sentinel = true;
    for (uint8_t byte : bytes) {
      if (byte != kOutputSentinel) {
        all_sentinel = false;
        break;
      }
    }
    if (all_sentinel) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "diagnostic tap `%.*s` was not written",
                              static_cast<int>(diagnostic_tap->name.size),
                              diagnostic_tap->name.data);
    }
  }
  return iree_ok_status();
}

typedef uint32_t DitFixtureRunFlags;

typedef enum DitFixtureRunFlagBits {
  ID4_DIT_FIXTURE_RUN_FLAG_CAPTURE_EXPECTED_TAPS = 1u << 0,
  ID4_DIT_FIXTURE_RUN_FLAG_COMPARE_EXPECTED_TAPS = 1u << 1,
  ID4_DIT_FIXTURE_RUN_FLAG_COMPARE_OUTPUT_BOUNDARY = 1u << 2,
  ID4_DIT_FIXTURE_RUN_FLAG_VERIFY_DIAGNOSTIC_TAPS_WRITTEN = 1u << 3,
} DitFixtureRunFlagBits;

typedef struct DitFixtureRunOptions {
  // Activation format requested from the DiT stage planner.
  id4_ideogram4_dit_activation_format_t activation_format;
  // Attention implementation requested from the DiT stage planner.
  id4_ideogram4_dit_attention_implementation_t attention_implementation;
  // Fixture input stage used to initialize imported boundary tensors.
  iree_string_view_t input_stage;
  // Expected diagnostic tap name prefix selected from the fixture.
  iree_string_view_t expected_tap_prefix;
  // Additional diagnostic tap names requested by this run.
  iree_string_view_list_t diagnostic_tap_names;
  // Expected fixture tensor compared against the exported velocity boundary.
  iree_string_view_t expected_output_name;
  // Integration-run capture identifier used when --id4_capture_dir is set.
  iree_string_view_t capture_run_id;
  // Fixture run behavior selected by DitFixtureRunFlagBits.
  DitFixtureRunFlags flags;
} DitFixtureRunOptions;

static void RunDitFixture(const DitFixtureRunOptions& options) {
  iree_string_view_t fixture_directory =
      iree_make_cstring_view(FLAG_id4_fixture_dir);
  ASSERT_FALSE(iree_string_view_is_empty(fixture_directory))
      << "--id4_fixture_dir is required for DiT integration correctness";

  id4::test::FixtureTensorSet fixture_tensors;
  IREE_ASSERT_OK(
      id4::test::LoadFixtureTensors(fixture_directory, &fixture_tensors));

  id4_ideogram4_dit_request_config_t request;
  IREE_ASSERT_OK(ConfigureRequestFromFixture(fixture_tensors,
                                             options.input_stage, &request));
  id4_ideogram4_dit_parameter_format_t parameter_format =
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_INVALID;
  IREE_ASSERT_OK(ParseDitParameterFormat(&parameter_format));
  iree_string_view_t fp8_parameter_scope = Fp8ParameterScopeForRequest(request);

  id4::test::LiveStageContext context;
  IREE_ASSERT_OK(id4::test::CreateLiveStageContextFromFlags(&context));

  id4::test::KernelLibraryRef kernel_library;
  IREE_ASSERT_OK(id4::test::CreateEmbeddedKernelLibrary(kernel_library.out()));

  id4::test::OwningRef<iree_io_parameter_provider_t,
                       iree_io_parameter_provider_release>
      parameter_provider;
  IREE_ASSERT_OK(CreateDitParameterProviderFromFlags(
      parameter_format, fp8_parameter_scope, parameter_provider.out()));

  id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release> stage;
  IREE_ASSERT_OK(CreateIdeogram4DitStage(context, parameter_format,
                                         fp8_parameter_scope, stage.out()));

  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);

  id4_pipeline_stage_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage.get(), &load_options));

  id4_ideogram4_dit_stage_plan_options_t dit_options;
  std::memset(&dit_options, 0, sizeof(dit_options));
  dit_options.structure_size = sizeof(dit_options);
  dit_options.request = request;
  dit_options.activation_format = options.activation_format;
  dit_options.attention_implementation = options.attention_implementation;

  std::vector<iree_string_view_t> diagnostic_tap_names;
  if (iree_all_bits_set(options.flags,
                        ID4_DIT_FIXTURE_RUN_FLAG_CAPTURE_EXPECTED_TAPS)) {
    for (const id4::test::FixtureTensor& tensor : fixture_tensors.tensors) {
      if (iree_string_view_equal(FixtureTensorRole(tensor),
                                 IREE_SV("expected")) &&
          iree_string_view_starts_with(FixtureTensorName(tensor),
                                       options.expected_tap_prefix) &&
          ExpectedTapMatchesRequestExtent(tensor, request)) {
        diagnostic_tap_names.push_back(FixtureTensorName(tensor));
      }
    }
    ASSERT_FALSE(diagnostic_tap_names.empty())
        << "DiT fixture contains no expected tensors";
  }
  for (iree_host_size_t i = 0; i < options.diagnostic_tap_names.count; ++i) {
    diagnostic_tap_names.push_back(options.diagnostic_tap_names.values[i]);
  }
  const iree_flag_string_list_t extra_diagnostic_taps =
      FLAG_id4_diagnostic_tap_list();
  for (iree_host_size_t i = 0; i < extra_diagnostic_taps.count; ++i) {
    diagnostic_tap_names.push_back(extra_diagnostic_taps.values[i]);
  }
  const bool captures_diagnostic_taps = !diagnostic_tap_names.empty();

  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &dit_options;
  if (captures_diagnostic_taps) {
    plan_options.flags = ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
  }
  plan_options.diagnostic_tap_names = (iree_string_view_list_t){
      diagnostic_tap_names.size(),
      diagnostic_tap_names.data(),
  };
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;

  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_ASSERT_OK(
      id4_pipeline_stage_plan(stage.get(), &plan_options, plan.out()));
  IREE_ASSERT_OK(WritePlanJsonIfRequested(plan.get()));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      prepare_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, prepare_semaphore.out()));
  id4::test::SemaphoreListStorage prepare_signal;
  prepare_signal.semaphore = prepare_semaphore.get();
  prepare_signal.payload_value = 1;

  id4_pipeline_stage_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.parameter_provider = parameter_provider.get();
  prepare_options.kernel_library = kernel_library.get();
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = prepare_signal.list();
  prepare_options.diagnostics_sink = &diagnostics_sink;

  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      bundle;
  IREE_ASSERT_OK(id4_pipeline_stage_prepare(stage.get(), plan.get(),
                                            &prepare_options, bundle.out()));

  id4::test::BufferBindingSet boundary_bindings;
  IREE_ASSERT_OK(id4::test::AllocateBoundaryBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      &boundary_bindings));
  id4::test::BufferBindingSet diagnostic_tap_bindings;
  IREE_ASSERT_OK(id4::test::AllocateDiagnosticTapBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      &diagnostic_tap_bindings));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      update_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, update_semaphore.out()));
  uint64_t update_value = 0;
  IREE_ASSERT_OK(QueueUpdateInitializedBoundariesFromFixture(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      boundary_bindings, fixture_tensors, options.input_stage,
      update_semaphore.get(), &update_value));
  IREE_ASSERT_OK(id4::test::QueueFillBoundaryTensors(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      boundary_bindings, ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED,
      &kOutputSentinel, sizeof(kOutputSentinel), update_semaphore.get(),
      &update_value));
  IREE_ASSERT_OK(id4::test::QueueFillDiagnosticTapTensors(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      diagnostic_tap_bindings, &kOutputSentinel, sizeof(kOutputSentinel),
      update_semaphore.get(), &update_value));
  ASSERT_GT(update_value, 0u);

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      issue_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, issue_semaphore.out()));

  id4::test::SemaphoreListStorage issue_wait;
  issue_wait.semaphore = update_semaphore.get();
  issue_wait.payload_value = update_value;
  id4::test::SemaphoreListStorage issue_signal;
  issue_signal.semaphore = issue_semaphore.get();
  issue_signal.payload_value = 1;

  id4_pipeline_stage_issue_options_t issue_options;
  std::memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.boundary_binding_count = boundary_bindings.count;
  issue_options.boundary_bindings = boundary_bindings.bindings;
  issue_options.diagnostic_tap_binding_count = diagnostic_tap_bindings.count;
  issue_options.diagnostic_tap_bindings = diagnostic_tap_bindings.bindings;
  issue_options.wait_semaphore_list = issue_wait.list();
  issue_options.signal_semaphore_list = issue_signal.list();
  issue_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(
      id4_pipeline_stage_issue(stage.get(), bundle.get(), &issue_options));

  id4::test::SemaphoreListStorage read_wait;
  read_wait.semaphore = issue_semaphore.get();
  read_wait.payload_value = issue_signal.payload_value;
  if (iree_all_bits_set(
          options.flags,
          ID4_DIT_FIXTURE_RUN_FLAG_VERIFY_DIAGNOSTIC_TAPS_WRITTEN)) {
    IREE_ASSERT_OK(VerifyDiagnosticTapsWereWritten(
        context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
        diagnostic_tap_bindings, read_wait.list()));
  }
  iree_string_view_t capture_directory =
      iree_make_cstring_view(FLAG_id4_capture_dir);
  if (!iree_string_view_is_empty(capture_directory)) {
    id4_tooling_capture_execution_options_t capture_options;
    std::memset(&capture_options, 0, sizeof(capture_options));
    capture_options.structure_size = sizeof(capture_options);
    capture_options.run_id = options.capture_run_id;
    capture_options.output_directory = capture_directory;
    capture_options.plan = plan.get();
    capture_options.device = context.device.get();
    capture_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
    capture_options.boundary_binding_count = boundary_bindings.count;
    capture_options.boundary_bindings = boundary_bindings.bindings;
    capture_options.diagnostic_tap_binding_count =
        diagnostic_tap_bindings.count;
    capture_options.diagnostic_tap_bindings = diagnostic_tap_bindings.bindings;
    capture_options.wait_semaphore_list = read_wait.list();
    capture_options.host_allocator = iree_allocator_system();
    IREE_ASSERT_OK(id4_tooling_capture_execution(&capture_options));
    IREE_ASSERT_OK(id4::test::VerifyCapturedExportedBoundaryTensorsWereWritten(
        plan.get(), capture_directory, kOutputSentinel));
    if (captures_diagnostic_taps) {
      IREE_ASSERT_OK(id4::test::VerifyCapturedDiagnosticTapTensorsWereWritten(
          plan.get(), capture_directory, kOutputSentinel));
    }
  }
  if (iree_all_bits_set(options.flags,
                        ID4_DIT_FIXTURE_RUN_FLAG_COMPARE_EXPECTED_TAPS)) {
    IREE_ASSERT_OK(CompareExpectedDiagnosticTaps(
        context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
        diagnostic_tap_bindings, fixture_tensors, request,
        options.expected_tap_prefix, read_wait.list()));
  }
  if (iree_all_bits_set(options.flags,
                        ID4_DIT_FIXTURE_RUN_FLAG_COMPARE_OUTPUT_BOUNDARY)) {
    IREE_ASSERT_OK(CompareExpectedOutputBoundary(
        context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
        boundary_bindings, fixture_tensors, options.expected_output_name,
        read_wait.list()));
  }
  IREE_ASSERT_OK(VerifyExportedBoundariesWereWritten(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      boundary_bindings, read_wait.list()));
}

TEST(Ideogram4DitStageIntegration, PrepareAndIssueForwardPreludeFixture) {
  RunDitFixture(DitFixtureRunOptions{
      .activation_format = ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_F32_CANONICAL,
      .attention_implementation =
          ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING,
      .input_stage = IREE_SV("ideogram4.cond.input"),
      .expected_tap_prefix = IREE_SV("ideogram4.cond."),
      .expected_output_name = IREE_SV("ideogram4.cond.output.velocity"),
      .capture_run_id = IREE_SV("ideogram4_dit_forward_integration"),
      .flags = ID4_DIT_FIXTURE_RUN_FLAG_CAPTURE_EXPECTED_TAPS |
               ID4_DIT_FIXTURE_RUN_FLAG_COMPARE_EXPECTED_TAPS |
               ID4_DIT_FIXTURE_RUN_FLAG_COMPARE_OUTPUT_BOUNDARY,
  });
}

TEST(Ideogram4DitStageIntegration, PrepareAndIssueBf16LinearInputFixture) {
  RunDitFixture(DitFixtureRunOptions{
      .activation_format =
          ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT,
      .attention_implementation =
          ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING,
      .input_stage = IREE_SV("ideogram4.cond.input"),
      .expected_tap_prefix = IREE_SV("ideogram4.cond."),
      .expected_output_name = IREE_SV("ideogram4.cond.output.velocity"),
      .capture_run_id = IREE_SV("ideogram4_dit_bf16_linear_input_integration"),
      .flags = ID4_DIT_FIXTURE_RUN_FLAG_COMPARE_OUTPUT_BOUNDARY,
  });
}

TEST(Ideogram4DitStageIntegration,
     PrepareAndIssueMaterializedWmmaAttentionFixture) {
  const iree_string_view_t diagnostic_tap_names[] = {
      IREE_SV("ideogram4.cond.layers.0.attention.context"),
      IREE_SV("ideogram4.cond.layers.0.attention.output"),
  };
  RunDitFixture(DitFixtureRunOptions{
      .activation_format =
          ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT,
      .attention_implementation =
          ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_MATERIALIZED_WMMA,
      .input_stage = IREE_SV("ideogram4.cond.input"),
      .expected_tap_prefix = iree_string_view_empty(),
      .diagnostic_tap_names =
          {
              IREE_ARRAYSIZE(diagnostic_tap_names),
              diagnostic_tap_names,
          },
      .expected_output_name = iree_string_view_empty(),
      .capture_run_id =
          IREE_SV("ideogram4_dit_materialized_wmma_attention_integration"),
      .flags = ID4_DIT_FIXTURE_RUN_FLAG_VERIFY_DIAGNOSTIC_TAPS_WRITTEN,
  });
}

}  // namespace
