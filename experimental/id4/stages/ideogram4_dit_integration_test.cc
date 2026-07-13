// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/hal_integration_util.h"
#include "experimental/id4/stages/ideogram4_dit.h"
#include "experimental/id4/stages/ideogram4_dit_parameters.h"
#include "experimental/id4/stages/ideogram4_dit_test_util.h"
#include "experimental/id4/tooling/capture.h"
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
IREE_FLAG(
    string, id4_pytorch_oracle_dir, "",
    "Directory containing PyTorch DiT oracle .npy tensors using cond/ and "
    "uncond/ subdirectories.");
IREE_FLAG(string, dit_parameter_format, "fp8_e4m3",
          "DiT parameter format: bf16 or fp8_e4m3.");
IREE_FLAG(string, dit_conditioned_fp8_scope, "dit_cond_fp8",
          "Conditioned DiT FP8 e4m3 source parameter scope.");
IREE_FLAG(string, dit_unconditioned_fp8_scope, "dit_uncond_fp8",
          "Unconditioned DiT FP8 e4m3 source parameter scope.");

namespace {

constexpr uint8_t kOutputSentinel = 0xA5;

typedef struct DitBranchParameterScopes {
  // BF16-expanded parameter scope for the selected branch.
  iree_string_view_t bf16;
  // FP8 e4m3 parameter scope for the selected branch.
  iree_string_view_t fp8_e4m3;
} DitBranchParameterScopes;

static iree_string_view_t DitBranchParameterScopeForFormat(
    id4_ideogram4_dit_parameter_format_t parameter_format,
    DitBranchParameterScopes parameter_scopes) {
  switch (parameter_format) {
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16:
      return parameter_scopes.bf16;
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_FP8_E4M3:
      return parameter_scopes.fp8_e4m3;
    default:
      return iree_string_view_empty();
  }
}

static iree_status_t ParseDitParameterFormat(
    id4_ideogram4_dit_parameter_format_t* out_format) {
  iree_status_t status = id4_ideogram4_dit_parameter_format_parse(
      iree_make_cstring_view(FLAG_dit_parameter_format), out_format);
  if (iree_status_is_ok(status)) return status;
  return iree_status_annotate(status, IREE_SV("--dit_parameter_format"));
}

static iree_status_t DitBranchParameterScopesForRequest(
    const id4_ideogram4_dit_request_config_t& request,
    DitBranchParameterScopes* out_scopes) {
  IREE_ASSERT_ARGUMENT(out_scopes);
  std::memset(out_scopes, 0, sizeof(*out_scopes));
  switch (request.conditioning_mode) {
    case ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED:
      out_scopes->bf16 = IREE_SV("dit_cond");
      out_scopes->fp8_e4m3 =
          iree_make_cstring_view(FLAG_dit_conditioned_fp8_scope);
      return iree_ok_status();
    case ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED:
      out_scopes->bf16 = IREE_SV("dit_uncond");
      out_scopes->fp8_e4m3 =
          iree_make_cstring_view(FLAG_dit_unconditioned_fp8_scope);
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "DiT conditioning mode %" PRIu32
                              " has no parameter scopes",
                              (uint32_t)request.conditioning_mode);
  }
}

static iree_status_t CreateIdeogram4DitStage(
    const id4::test::LiveStageContext& context,
    id4_ideogram4_dit_parameter_format_t parameter_format,
    DitBranchParameterScopes parameter_scopes,
    id4_pipeline_stage_t** out_stage) {
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
      parameter_scopes.fp8_e4m3, iree_allocator_system(), &source_rules));

  id4_ideogram4_dit_stage_create_options_t create_options;
  std::memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.kernel_cache = context.kernel_cache.get();
  create_options.parameter_scope =
      DitBranchParameterScopeForFormat(parameter_format, parameter_scopes);
  if (iree_string_view_is_empty(create_options.parameter_scope)) {
    id4_ideogram4_dit_parameter_source_rule_list_deinitialize(
        &source_rules, iree_allocator_system());
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "DiT parameter format has no source scope");
  }
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

static iree_string_view_t FixtureTensorName(
    const id4::test::FixtureTensor& tensor) {
  return iree_make_string_view(tensor.name.data(), tensor.name.size());
}

static iree_string_view_t FixtureTensorRole(
    const id4::test::FixtureTensor& tensor) {
  return iree_make_string_view(tensor.role.data(), tensor.role.size());
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
    DitBranchParameterScopes parameter_scopes,
    iree_io_parameter_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = nullptr;
  const iree_string_view_t parameter_scope =
      DitBranchParameterScopeForFormat(parameter_format, parameter_scopes);
  if (iree_string_view_is_empty(parameter_scope)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "DiT parameter format has no provider scope");
  }
  return id4::test::CreateParameterProviderFromFlags(parameter_scope,
                                                     out_provider);
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

typedef struct PytorchOracleTolerance {
  // Aggregate mean absolute error limit.
  double mean_absolute_error;
  // Aggregate p99 absolute error limit.
  double p99_absolute_error;
  // Aggregate maximum absolute error limit.
  double max_absolute_error;
  // Elementwise relative error limit for values above the absolute limit.
  double max_relative_error;
} PytorchOracleTolerance;

typedef enum PytorchOracleLayoutMappingKind {
  // Dense physical order matches the PyTorch oracle payload order.
  PYTORCH_ORACLE_LAYOUT_MAPPING_DENSE = 0,
  // Physical rank-2 tensor is [feature, token]; oracle is [1, token, feature].
  PYTORCH_ORACLE_LAYOUT_MAPPING_HIDDEN_MAJOR_RANK2 = 1,
  // Physical rank-2 tensor is [padded_token, feature]; oracle is [1, token,
  // feature].
  PYTORCH_ORACLE_LAYOUT_MAPPING_PADDED_TOKEN_MAJOR_RANK2 = 2,
  // Physical rank-2 tensor is [feature, image_token]; oracle is [1, token,
  // feature] and starts after a text-token prefix.
  PYTORCH_ORACLE_LAYOUT_MAPPING_IMAGE_HIDDEN_MAJOR_RANK2 = 3,
} PytorchOracleLayoutMappingKind;

typedef struct PytorchOracleLayoutMapping {
  // Physical-to-oracle index mapping class.
  PytorchOracleLayoutMappingKind kind;
  // Logical oracle shape expected in the PyTorch payload.
  id4_pipeline_tensor_shape_t expected_shape;
  // Logical element count expected in the PyTorch payload.
  iree_host_size_t expected_element_count;
  // Logical element count compared against the PyTorch payload.
  iree_host_size_t comparison_element_count;
  // First oracle token ordinal compared for token-sliced mappings.
  uint32_t expected_token_offset;
} PytorchOracleLayoutMapping;

static float LoadF32(const uint8_t* bytes) {
  float value = 0.0f;
  std::memcpy(&value, bytes, sizeof(value));
  return value;
}

static uint16_t LoadU16(const uint8_t* bytes) {
  uint16_t value = 0;
  std::memcpy(&value, bytes, sizeof(value));
  return value;
}

static float LoadBf16AsF32(const uint8_t* bytes) {
  const uint32_t bits = static_cast<uint32_t>(LoadU16(bytes)) << 16;
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

static iree_status_t LoadTensorElementAsF32(id4_pipeline_tensor_dtype_t dtype,
                                            const std::vector<uint8_t>& bytes,
                                            iree_host_size_t index,
                                            float* out_value) {
  const iree_device_size_t dtype_byte_length =
      id4_pipeline_tensor_dtype_byte_length(dtype);
  if (dtype_byte_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "actual tensor dtype is invalid");
  }
  if (index >
      std::numeric_limits<iree_host_size_t>::max() / dtype_byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "actual tensor byte offset overflow");
  }
  const iree_host_size_t byte_offset =
      index * static_cast<iree_host_size_t>(dtype_byte_length);
  if (byte_offset > bytes.size() ||
      bytes.size() - byte_offset < dtype_byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "actual tensor element index is out of range");
  }
  switch (dtype) {
    case ID4_PIPELINE_TENSOR_DTYPE_F32:
      *out_value = LoadF32(&bytes[byte_offset]);
      return iree_ok_status();
    case ID4_PIPELINE_TENSOR_DTYPE_BF16:
      *out_value = LoadBf16AsF32(&bytes[byte_offset]);
      return iree_ok_status();
    default: {
      iree_string_view_t dtype_name = id4_pipeline_tensor_dtype_format(dtype);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "actual tensor dtype `%.*s` cannot be compared against PyTorch "
          "oracle f32 payloads",
          static_cast<int>(dtype_name.size), dtype_name.data);
    }
  }
}

static iree_status_t ShapeElementCount(id4_pipeline_tensor_shape_t shape,
                                       iree_host_size_t* out_element_count) {
  iree_host_size_t element_count = 1;
  for (uint32_t i = 0; i < shape.rank; ++i) {
    if (shape.dims[i] > std::numeric_limits<iree_host_size_t>::max()) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "tensor dimension %u exceeds host size", i);
    }
    const iree_host_size_t dim = static_cast<iree_host_size_t>(shape.dims[i]);
    if (dim == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "tensor dimension %u is zero", i);
    }
    if (element_count > std::numeric_limits<iree_host_size_t>::max() / dim) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "tensor element count overflow");
    }
    element_count *= dim;
  }
  *out_element_count = element_count;
  return iree_ok_status();
}

static id4_pipeline_tensor_shape_t MakeShapeRank3(uint64_t dim0, uint64_t dim1,
                                                  uint64_t dim2) {
  id4_pipeline_tensor_shape_t shape = {};
  shape.rank = 3;
  shape.dims[0] = dim0;
  shape.dims[1] = dim1;
  shape.dims[2] = dim2;
  return shape;
}

static iree_status_t PytorchOracleTotalTokenCount(
    const id4_ideogram4_dit_request_config_t& request,
    uint32_t* out_total_token_count) {
  *out_total_token_count = 0;
  if (request.latent_shape.rank != 4 || request.latent_shape.dims[3] != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PyTorch oracle DiT latent shape must be rank-4 "
                            "with batch size 1");
  }
  if (request.latent_shape.dims[0] == 0 || request.latent_shape.dims[1] == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PyTorch oracle image token dimensions must be "
                            "nonzero");
  }
  uint64_t image_token_count = 0;
  if (request.latent_shape.dims[0] >
      std::numeric_limits<uint64_t>::max() / request.latent_shape.dims[1]) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PyTorch oracle image token count overflow");
  }
  image_token_count =
      request.latent_shape.dims[0] * request.latent_shape.dims[1];
  uint64_t total_token_count = image_token_count;
  switch (request.conditioning_mode) {
    case ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED:
      if (total_token_count >
          std::numeric_limits<uint64_t>::max() - request.text_token_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "PyTorch oracle total token count overflow");
      }
      total_token_count += request.text_token_count;
      break;
    case ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "DiT conditioning mode %" PRIu32
                              " has no PyTorch oracle token count",
                              (uint32_t)request.conditioning_mode);
  }
  if (total_token_count == 0 ||
      total_token_count > std::numeric_limits<uint32_t>::max()) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PyTorch oracle total token count %" PRIu64
                            " is outside uint32 range",
                            total_token_count);
  }
  *out_total_token_count = (uint32_t)total_token_count;
  return iree_ok_status();
}

static std::string JoinPath(iree_string_view_t root, const char* first,
                            const char* second) {
  std::string path(root.data, root.size);
  if (!path.empty() && path.back() != '/') path.push_back('/');
  path.append(first);
  if (second && second[0] != '\0') {
    path.push_back('/');
    path.append(second);
  }
  return path;
}

static iree_status_t PytorchOracleBranchName(
    id4_ideogram4_dit_conditioning_mode_t conditioning_mode,
    const char** out_branch_name) {
  switch (conditioning_mode) {
    case ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED:
      *out_branch_name = "cond";
      return iree_ok_status();
    case ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED:
      *out_branch_name = "uncond";
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "DiT conditioning mode %" PRIu32
                              " has no PyTorch oracle branch",
                              (uint32_t)conditioning_mode);
  }
}

static iree_status_t PytorchOraclePathForBoundary(
    iree_string_view_t oracle_directory,
    id4_ideogram4_dit_conditioning_mode_t conditioning_mode,
    iree_string_view_t boundary_name, std::string* out_path) {
  if (!iree_string_view_equal(boundary_name, IREE_SV("velocity"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "boundary `%.*s` has no PyTorch oracle path",
                            static_cast<int>(boundary_name.size),
                            boundary_name.data);
  }
  const char* branch_name = nullptr;
  IREE_RETURN_IF_ERROR(
      PytorchOracleBranchName(conditioning_mode, &branch_name));
  *out_path = JoinPath(oracle_directory, branch_name, "velocity.npy");
  return iree_ok_status();
}

static iree_status_t PytorchOraclePathForTap(
    iree_string_view_t oracle_directory,
    id4_ideogram4_dit_conditioning_mode_t conditioning_mode,
    iree_string_view_t tap_name, std::string* out_path) {
  typedef struct TapPathAlias {
    // ID4 semantic tap suffix.
    iree_string_view_t source_suffix;
    // PyTorch oracle filename suffix.
    iree_string_view_t target_suffix;
  } TapPathAlias;
  static const TapPathAlias kTapPathAliases[] = {
      {IREE_SV(".attention.qkv_projection.output"), IREE_SV(".attention.qkv")},
      {IREE_SV(".ffn.input"), IREE_SV(".mlp_input")},
      {IREE_SV(".ffn.output"), IREE_SV(".mlp_output")},
      {IREE_SV("final.projected"), IREE_SV("final.raw_output")},
      {IREE_SV(".post_attention_hidden"), IREE_SV(".after_attention")},
      {IREE_SV(".attention.query_rotary"), IREE_SV(".attention.query")},
      {IREE_SV(".attention.key_rotary"), IREE_SV(".attention.key")},
  };

  const char* branch_name = nullptr;
  IREE_RETURN_IF_ERROR(
      PytorchOracleBranchName(conditioning_mode, &branch_name));
  const iree_string_view_t prefix =
      conditioning_mode == ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED
          ? IREE_SV("ideogram4.cond.")
          : IREE_SV("ideogram4.uncond.");
  if (!iree_string_view_starts_with(tap_name, prefix)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "diagnostic tap `%.*s` does not match the requested oracle branch",
        static_cast<int>(tap_name.size), tap_name.data);
  }
  const iree_string_view_t relative_name =
      iree_string_view_remove_prefix(tap_name, prefix.size);
  iree_string_view_t oracle_relative_prefix = relative_name;
  iree_string_view_t oracle_relative_suffix = iree_string_view_empty();
  for (const TapPathAlias& alias : kTapPathAliases) {
    if (iree_string_view_ends_with(relative_name, alias.source_suffix)) {
      oracle_relative_prefix = iree_string_view_remove_suffix(
          relative_name, alias.source_suffix.size);
      oracle_relative_suffix = alias.target_suffix;
      break;
    }
  }
  *out_path = JoinPath(oracle_directory, branch_name, "taps");
  out_path->push_back('/');
  out_path->append(oracle_relative_prefix.data, oracle_relative_prefix.size);
  out_path->append(oracle_relative_suffix.data, oracle_relative_suffix.size);
  out_path->append(".npy");
  return iree_ok_status();
}

static iree_status_t PytorchOracleLayoutMappingForTensor(
    const id4_pipeline_tensor_layout_t* actual_layout,
    const id4_ideogram4_dit_request_config_t& request,
    PytorchOracleLayoutMapping* out_mapping) {
  std::memset(out_mapping, 0, sizeof(*out_mapping));
  const id4_pipeline_tensor_shape_t actual_shape = actual_layout->shape;
  if (actual_shape.rank == 4 && actual_shape.dims[3] == 1) {
    out_mapping->kind = PYTORCH_ORACLE_LAYOUT_MAPPING_DENSE;
    out_mapping->expected_shape = MakeShapeRank3(
        actual_shape.dims[0], actual_shape.dims[1], actual_shape.dims[2]);
    IREE_RETURN_IF_ERROR(ShapeElementCount(
        out_mapping->expected_shape, &out_mapping->expected_element_count));
    out_mapping->comparison_element_count = out_mapping->expected_element_count;
    return iree_ok_status();
  }
  if (actual_shape.rank == 2) {
    uint32_t total_token_count = 0;
    IREE_RETURN_IF_ERROR(
        PytorchOracleTotalTokenCount(request, &total_token_count));
    uint64_t image_token_count64 = 0;
    if (request.latent_shape.dims[0] >
        std::numeric_limits<uint64_t>::max() / request.latent_shape.dims[1]) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "PyTorch oracle image token count overflow");
    }
    image_token_count64 =
        request.latent_shape.dims[0] * request.latent_shape.dims[1];
    if (image_token_count64 > std::numeric_limits<uint32_t>::max()) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "PyTorch oracle image token count exceeds "
                              "uint32 range");
    }
    const uint32_t image_token_count = (uint32_t)image_token_count64;
    if (iree_string_view_ends_with(actual_layout->name,
                                   IREE_SV(".final.projected")) &&
        actual_shape.dims[1] == image_token_count) {
      out_mapping->kind =
          PYTORCH_ORACLE_LAYOUT_MAPPING_IMAGE_HIDDEN_MAJOR_RANK2;
      out_mapping->expected_shape =
          MakeShapeRank3(1, total_token_count, actual_shape.dims[0]);
      out_mapping->expected_token_offset =
          request.conditioning_mode ==
                  ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED
              ? request.text_token_count
              : 0;
      IREE_RETURN_IF_ERROR(ShapeElementCount(
          out_mapping->expected_shape, &out_mapping->expected_element_count));
      IREE_RETURN_IF_ERROR(ShapeElementCount(
          actual_shape, &out_mapping->comparison_element_count));
    } else if (actual_shape.dims[0] >= total_token_count &&
               actual_shape.dims[1] > actual_shape.dims[0]) {
      out_mapping->kind =
          PYTORCH_ORACLE_LAYOUT_MAPPING_PADDED_TOKEN_MAJOR_RANK2;
      out_mapping->expected_shape =
          MakeShapeRank3(1, total_token_count, actual_shape.dims[1]);
      IREE_RETURN_IF_ERROR(ShapeElementCount(
          out_mapping->expected_shape, &out_mapping->expected_element_count));
      out_mapping->comparison_element_count =
          out_mapping->expected_element_count;
    } else {
      out_mapping->kind = PYTORCH_ORACLE_LAYOUT_MAPPING_HIDDEN_MAJOR_RANK2;
      out_mapping->expected_shape =
          MakeShapeRank3(1, actual_shape.dims[1], actual_shape.dims[0]);
      IREE_RETURN_IF_ERROR(ShapeElementCount(
          out_mapping->expected_shape, &out_mapping->expected_element_count));
      out_mapping->comparison_element_count =
          out_mapping->expected_element_count;
    }
    return iree_ok_status();
  }
  if (actual_shape.rank == 1) {
    out_mapping->kind = PYTORCH_ORACLE_LAYOUT_MAPPING_DENSE;
    out_mapping->expected_shape = MakeShapeRank3(1, 1, actual_shape.dims[0]);
    IREE_RETURN_IF_ERROR(ShapeElementCount(
        out_mapping->expected_shape, &out_mapping->expected_element_count));
    out_mapping->comparison_element_count = out_mapping->expected_element_count;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "actual tensor `%.*s` shape rank %u has no PyTorch "
                          "oracle mapping",
                          static_cast<int>(actual_layout->name.size),
                          actual_layout->name.data, actual_shape.rank);
}

static iree_status_t PytorchOracleIndexMapping(
    const id4_pipeline_tensor_layout_t* actual_layout,
    const PytorchOracleLayoutMapping* mapping,
    iree_host_size_t comparison_index, iree_host_size_t* out_actual_index,
    iree_host_size_t* out_expected_index) {
  const id4_pipeline_tensor_shape_t actual_shape = actual_layout->shape;
  switch (mapping->kind) {
    case PYTORCH_ORACLE_LAYOUT_MAPPING_DENSE:
    case PYTORCH_ORACLE_LAYOUT_MAPPING_PADDED_TOKEN_MAJOR_RANK2:
      *out_actual_index = comparison_index;
      *out_expected_index = comparison_index;
      return iree_ok_status();
    case PYTORCH_ORACLE_LAYOUT_MAPPING_HIDDEN_MAJOR_RANK2: {
      const iree_host_size_t hidden_size =
          static_cast<iree_host_size_t>(actual_shape.dims[0]);
      const iree_host_size_t token_count =
          static_cast<iree_host_size_t>(actual_shape.dims[1]);
      const iree_host_size_t hidden_ordinal = comparison_index / token_count;
      const iree_host_size_t token_ordinal = comparison_index % token_count;
      *out_actual_index = comparison_index;
      *out_expected_index = token_ordinal * hidden_size + hidden_ordinal;
      return iree_ok_status();
    }
    case PYTORCH_ORACLE_LAYOUT_MAPPING_IMAGE_HIDDEN_MAJOR_RANK2: {
      const iree_host_size_t hidden_size =
          static_cast<iree_host_size_t>(actual_shape.dims[0]);
      const iree_host_size_t image_token_count =
          static_cast<iree_host_size_t>(actual_shape.dims[1]);
      const iree_host_size_t hidden_ordinal =
          comparison_index / image_token_count;
      const iree_host_size_t image_token_ordinal =
          comparison_index % image_token_count;
      const iree_host_size_t expected_token_ordinal =
          static_cast<iree_host_size_t>(mapping->expected_token_offset) +
          image_token_ordinal;
      *out_actual_index = comparison_index;
      *out_expected_index =
          expected_token_ordinal * hidden_size + hidden_ordinal;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "actual tensor `%.*s` has no PyTorch oracle index "
                          "mapping",
                          static_cast<int>(actual_layout->name.size),
                          actual_layout->name.data);
}

static iree_status_t CompareBindingWithPytorchOracle(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_buffer_binding_t* binding,
    iree_hal_semaphore_list_t wait_list,
    const id4_pipeline_tensor_layout_t* actual_layout,
    const id4_ideogram4_dit_request_config_t& request,
    iree_string_view_t oracle_path, PytorchOracleTolerance tolerance) {
  PytorchOracleLayoutMapping mapping = {};
  IREE_RETURN_IF_ERROR(
      PytorchOracleLayoutMappingForTensor(actual_layout, request, &mapping));
  std::vector<uint8_t> expected_payload;
  IREE_RETURN_IF_ERROR(id4::test::LoadReferenceTensorPayload(
      oracle_path, ID4_PIPELINE_TENSOR_DTYPE_F32, mapping.expected_shape,
      &expected_payload));

  std::vector<uint8_t> actual_payload;
  IREE_RETURN_IF_ERROR(id4::test::ReadBindingToHost(
      device, queue_affinity, binding, wait_list, &actual_payload));
  iree_host_size_t actual_element_count = 0;
  IREE_RETURN_IF_ERROR(
      ShapeElementCount(actual_layout->shape, &actual_element_count));
  const iree_device_size_t actual_dtype_byte_length =
      id4_pipeline_tensor_dtype_byte_length(actual_layout->dtype);
  if (actual_dtype_byte_length == 0 ||
      actual_dtype_byte_length > std::numeric_limits<iree_host_size_t>::max()) {
    iree_string_view_t dtype_name =
        id4_pipeline_tensor_dtype_format(actual_layout->dtype);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "actual tensor `%.*s` dtype `%.*s` cannot be "
                            "compared against PyTorch oracle payloads",
                            static_cast<int>(actual_layout->name.size),
                            actual_layout->name.data,
                            static_cast<int>(dtype_name.size), dtype_name.data);
  }
  if (actual_element_count >
      std::numeric_limits<iree_host_size_t>::max() /
          static_cast<iree_host_size_t>(actual_dtype_byte_length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE, "actual tensor `%.*s` byte length overflow",
        static_cast<int>(actual_layout->name.size), actual_layout->name.data);
  }
  const iree_host_size_t actual_byte_count =
      actual_element_count *
      static_cast<iree_host_size_t>(actual_dtype_byte_length);
  if (actual_payload.size() != actual_byte_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "actual tensor `%.*s` byte length %zu does not match shape byte "
        "length %" PRIhsz,
        static_cast<int>(actual_layout->name.size), actual_layout->name.data,
        actual_payload.size(), actual_byte_count);
  }
  if (mapping.comparison_element_count >
      std::numeric_limits<iree_host_size_t>::max() / sizeof(float)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE, "PyTorch oracle `%.*s` byte length overflow",
        static_cast<int>(oracle_path.size), oracle_path.data);
  }
  const iree_host_size_t expected_byte_count =
      mapping.expected_element_count * sizeof(float);
  if (expected_payload.size() != expected_byte_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PyTorch oracle `%.*s` byte length %zu does not match actual element "
        "byte length %" PRIhsz,
        static_cast<int>(oracle_path.size), oracle_path.data,
        expected_payload.size(), expected_byte_count);
  }

  std::vector<float> absolute_errors;
  absolute_errors.reserve(mapping.comparison_element_count);
  double mean_absolute_error = 0.0;
  double max_absolute_error = 0.0;
  double max_relative_error = 0.0;
  iree_host_size_t outlier_count = 0;
  iree_host_size_t max_index = 0;
  iree_host_size_t max_relative_index = 0;
  for (iree_host_size_t comparison_index = 0;
       comparison_index < mapping.comparison_element_count;
       ++comparison_index) {
    iree_host_size_t actual_index = 0;
    iree_host_size_t expected_index = 0;
    IREE_RETURN_IF_ERROR(
        PytorchOracleIndexMapping(actual_layout, &mapping, comparison_index,
                                  &actual_index, &expected_index));
    float actual_value = 0.0f;
    IREE_RETURN_IF_ERROR(LoadTensorElementAsF32(
        actual_layout->dtype, actual_payload, actual_index, &actual_value));
    const float expected_value =
        LoadF32(&expected_payload[expected_index * sizeof(float)]);
    const float absolute_error = std::fabs(actual_value - expected_value);
    const double relative_error =
        (double)absolute_error /
        std::max<double>(std::fabs((double)expected_value), 1.0);
    absolute_errors.push_back(absolute_error);
    mean_absolute_error += absolute_error;
    if (absolute_error > max_absolute_error) {
      max_absolute_error = absolute_error;
      max_index = actual_index;
    }
    if (relative_error > max_relative_error) {
      max_relative_error = relative_error;
      max_relative_index = actual_index;
    }
    if (absolute_error > tolerance.max_absolute_error &&
        relative_error > tolerance.max_relative_error) {
      ++outlier_count;
    }
  }
  mean_absolute_error /= static_cast<double>(
      std::max<iree_host_size_t>(mapping.comparison_element_count, 1));
  std::sort(absolute_errors.begin(), absolute_errors.end());
  const iree_host_size_t p99_index =
      (absolute_errors.size() * 99) / 100 >= absolute_errors.size()
          ? absolute_errors.size() - 1
          : (absolute_errors.size() * 99) / 100;
  const double p99_absolute_error = absolute_errors[p99_index];
  if (mean_absolute_error <= tolerance.mean_absolute_error &&
      p99_absolute_error <= tolerance.p99_absolute_error &&
      outlier_count == 0) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "PyTorch oracle mismatch for `%.*s`: mean_abs=%g/%g p99_abs=%g/%g "
      "max_abs=%g/%g max_abs_index=%" PRIhsz
      " max_rel=%g/%g "
      "max_rel_index=%" PRIhsz " outliers=%" PRIhsz,
      static_cast<int>(actual_layout->name.size), actual_layout->name.data,
      mean_absolute_error, tolerance.mean_absolute_error, p99_absolute_error,
      tolerance.p99_absolute_error, max_absolute_error,
      tolerance.max_absolute_error, max_index, max_relative_error,
      tolerance.max_relative_error, max_relative_index, outlier_count);
}

static PytorchOracleTolerance PytorchOracleVelocityTolerance(void) {
  return PytorchOracleTolerance{
      // Mean absolute error limit after the BF16 residual trajectory amplifies
      // locally bounded block error.
      /*.mean_absolute_error=*/0.015625,
      // P99 limit at the next BF16 bucket above the observed trajectories.
      /*.p99_absolute_error=*/0.0625,
      // Maximum absolute error limit for exported velocity.
      /*.max_absolute_error=*/0.125,
      // Exported velocity uses an absolute gate.
      /*.max_relative_error=*/0.0,
  };
}

static PytorchOracleTolerance PytorchOracleToleranceForTap(
    iree_string_view_t tap_name,
    const id4_ideogram4_dit_request_config_t& request) {
  if (iree_string_view_ends_with(tap_name, IREE_SV(".ffn.hidden"))) {
    return PytorchOracleTolerance{
        // Mean absolute error limit for BF16 taps with target-specific
        // reduction and matmul accumulation order.
        /*.mean_absolute_error=*/0.004,
        // P99 absolute error limit for one BF16-scale bucket of drift.
        /*.p99_absolute_error=*/0.016,
        // SiLU/product compounds W1/W3 projection drift at high magnitude.
        /*.max_absolute_error=*/4.000,
        // Relative gate for rare high-magnitude BF16 product outliers.
        /*.max_relative_error=*/0.350,
    };
  }
  if (iree_string_view_ends_with(tap_name, IREE_SV(".ffn.output"))) {
    return PytorchOracleTolerance{
        // Mean absolute error limit for BF16 down-projection output across
        // dynamic token and image-token counts.
        /*.mean_absolute_error=*/0.005,
        // P99 absolute error limit after one amplified BF16 product matmul.
        /*.p99_absolute_error=*/0.050,
        // Maximum absolute error limit after the BF16 down projection.
        /*.max_absolute_error=*/0.750,
        // Relative gate for rare down-projection outliers.
        /*.max_relative_error=*/0.200,
    };
  }
  if (iree_string_view_find(tap_name, IREE_SV(".layers."), 0) !=
          IREE_STRING_VIEW_NPOS &&
      iree_string_view_ends_with(tap_name, IREE_SV(".hidden"))) {
    return PytorchOracleTolerance{
        // Mean absolute error limit for accumulated BF16 layer hidden states.
        /*.mean_absolute_error=*/0.004,
        // P99 absolute error limit for accumulated BF16 layer hidden states.
        /*.p99_absolute_error=*/0.016,
        // Rare high-magnitude BF16 matmul outliers can accumulate by mid-layer.
        /*.max_absolute_error=*/1.000,
        // Relative gate for rare accumulated hidden-state outliers.
        /*.max_relative_error=*/0.050,
    };
  }
  if (iree_string_view_ends_with(tap_name, IREE_SV("final.projected"))) {
    if (request.conditioning_mode ==
        ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED) {
      return PytorchOracleTolerance{
          // Mean absolute error limit for image-only final BF16 projection.
          /*.mean_absolute_error=*/0.012,
          // P99 absolute error limit after all BF16 transformer blocks.
          /*.p99_absolute_error=*/0.040,
          // Maximum absolute error limit for final projected logits.
          /*.max_absolute_error=*/0.250,
          // Relative limit for final projected logits.
          /*.max_relative_error=*/0.020,
      };
    }
    return PytorchOracleTolerance{
        // Mean absolute error limit after all BF16 transformer blocks.
        /*.mean_absolute_error=*/0.008,
        // P99 absolute error limit after all BF16 transformer blocks.
        /*.p99_absolute_error=*/0.040,
        // Maximum absolute error limit for final projected logits.
        /*.max_absolute_error=*/0.250,
        // Relative limit for final projected logits.
        /*.max_relative_error=*/0.020,
    };
  }
  if (iree_string_view_ends_with(tap_name, IREE_SV("final.normalized"))) {
    if (request.conditioning_mode ==
        ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED) {
      return PytorchOracleTolerance{
          // Mean absolute error limit for image-only final BF16 hidden state.
          /*.mean_absolute_error=*/0.012,
          // P99 absolute error limit for image-only final BF16 hidden state.
          /*.p99_absolute_error=*/0.064,
          // Maximum absolute error limit for final normalized hidden state.
          /*.max_absolute_error=*/1.500,
          // Relative gate for rare accumulated BF16 activation outliers.
          /*.max_relative_error=*/0.450,
      };
    }
    return PytorchOracleTolerance{
        // Mean absolute error limit for final BF16 normalized hidden state.
        /*.mean_absolute_error=*/0.008,
        // P99 absolute error limit after all BF16 transformer blocks across
        // dynamic token and image-token counts.
        /*.p99_absolute_error=*/0.050,
        // Maximum absolute error limit for final normalized hidden state.
        /*.max_absolute_error=*/1.500,
        // Relative gate for rare accumulated BF16 activation outliers.
        /*.max_relative_error=*/0.450,
    };
  }
  return PytorchOracleTolerance{
      // Mean absolute error limit for BF16 taps with target-specific
      // reduction and matmul accumulation order.
      /*.mean_absolute_error=*/0.004,
      // P99 absolute error limit for one BF16-scale bucket of drift.
      /*.p99_absolute_error=*/0.016,
      // Maximum absolute error limit for large-magnitude BF16 tap values.
      /*.max_absolute_error=*/0.250,
      // Relative limit for high-magnitude BF16 activations where one or two
      // BF16 buckets exceed the absolute gate.
      /*.max_relative_error=*/0.020,
  };
}

static iree_status_t ComparePytorchOracleOutputBoundary(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan,
    const id4::test::BufferBindingSet& boundary_bindings,
    const id4_ideogram4_dit_request_config_t& request,
    iree_string_view_t oracle_directory, iree_hal_semaphore_list_t wait_list) {
  const id4_pipeline_boundary_tensor_plan_t* boundary = nullptr;
  IREE_RETURN_IF_ERROR(FindBoundaryPlan(plan, IREE_SV("velocity"), &boundary));
  iree_hal_buffer_binding_t binding = {};
  IREE_RETURN_IF_ERROR(id4::test::FindBoundaryBinding(
      plan, boundary_bindings, IREE_SV("velocity"), &binding));
  std::string oracle_path;
  IREE_RETURN_IF_ERROR(
      PytorchOraclePathForBoundary(oracle_directory, request.conditioning_mode,
                                   IREE_SV("velocity"), &oracle_path));
  return CompareBindingWithPytorchOracle(
      device, queue_affinity, &binding, wait_list, &boundary->layout, request,
      iree_make_string_view(oracle_path.data(), oracle_path.size()),
      PytorchOracleVelocityTolerance());
}

static iree_status_t ComparePytorchOracleDiagnosticTaps(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan,
    const id4::test::BufferBindingSet& diagnostic_tap_bindings,
    const id4_ideogram4_dit_request_config_t& request,
    iree_string_view_t oracle_directory, iree_hal_semaphore_list_t wait_list) {
  for (iree_host_size_t i = 0; i < id4_pipeline_plan_diagnostic_tap_count(plan);
       ++i) {
    const id4_pipeline_diagnostic_tap_plan_t* diagnostic_tap =
        id4_pipeline_plan_diagnostic_tap_at(plan, i);
    if (!diagnostic_tap) continue;
    iree_hal_buffer_binding_t binding = {};
    IREE_RETURN_IF_ERROR(id4::test::FindDiagnosticTapBinding(
        plan, diagnostic_tap_bindings, diagnostic_tap->name, &binding));
    std::string oracle_path;
    IREE_RETURN_IF_ERROR(
        PytorchOraclePathForTap(oracle_directory, request.conditioning_mode,
                                diagnostic_tap->name, &oracle_path));
    IREE_RETURN_IF_ERROR(CompareBindingWithPytorchOracle(
        device, queue_affinity, &binding, wait_list, &diagnostic_tap->layout,
        request, iree_make_string_view(oracle_path.data(), oracle_path.size()),
        PytorchOracleToleranceForTap(diagnostic_tap->name, request)));
  }
  return iree_ok_status();
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
  // Linear weight execution strategy requested from the DiT stage planner.
  id4_ideogram4_dit_weight_execution_format_t weight_execution_format;
  // Attention implementation requested from the DiT stage planner.
  id4_ideogram4_dit_attention_implementation_t attention_implementation;
  // Feed-forward implementation requested from the DiT stage planner.
  id4_ideogram4_dit_feed_forward_implementation_t feed_forward_implementation;
  // Conditioned or unconditioned DiT branch selected for this run.
  id4::test::Ideogram4DitBranch branch;
  // Additional diagnostic tap names requested by this run.
  iree_string_view_list_t diagnostic_tap_names;
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

  const id4::test::Ideogram4DitBranchConfig branch =
      id4::test::Ideogram4DitBranchConfigFor(options.branch);
  id4_ideogram4_dit_request_config_t request;
  IREE_ASSERT_OK(id4::test::Ideogram4DitConfigureRequestFromFixture(
      fixture_tensors, branch, &request));
  id4_ideogram4_dit_parameter_format_t parameter_format =
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_INVALID;
  IREE_ASSERT_OK(ParseDitParameterFormat(&parameter_format));
  DitBranchParameterScopes parameter_scopes;
  IREE_ASSERT_OK(
      DitBranchParameterScopesForRequest(request, &parameter_scopes));

  id4::test::LiveStageContext context;
  IREE_ASSERT_OK(id4::test::CreateLiveStageContextFromFlags(&context));

  id4::test::KernelLibraryRef kernel_library;
  IREE_ASSERT_OK(id4::test::CreateEmbeddedKernelLibrary(kernel_library.out()));

  id4::test::OwningRef<iree_io_parameter_provider_t,
                       iree_io_parameter_provider_release>
      parameter_provider;
  IREE_ASSERT_OK(CreateDitParameterProviderFromFlags(
      parameter_format, parameter_scopes, parameter_provider.out()));

  id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release> stage;
  IREE_ASSERT_OK(CreateIdeogram4DitStage(context, parameter_format,
                                         parameter_scopes, stage.out()));

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
  dit_options.weight_execution_format = options.weight_execution_format;
  dit_options.attention_implementation = options.attention_implementation;
  dit_options.feed_forward_implementation = options.feed_forward_implementation;

  std::vector<iree_string_view_t> diagnostic_tap_names;
  if (iree_all_bits_set(options.flags,
                        ID4_DIT_FIXTURE_RUN_FLAG_CAPTURE_EXPECTED_TAPS)) {
    for (const id4::test::FixtureTensor& tensor : fixture_tensors.tensors) {
      if (iree_string_view_equal(FixtureTensorRole(tensor),
                                 IREE_SV("expected")) &&
          iree_string_view_starts_with(FixtureTensorName(tensor),
                                       branch.diagnostic_tap_prefix) &&
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
  ASSERT_EQ(id4_pipeline_plan_region_count(plan.get()), 1u);
  EXPECT_EQ(id4_pipeline_plan_shared_tensor_count(plan.get()), 0u);
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
  prepare_options.parameter_policy = id4_pipeline_stage_parameters(
      id4_pipeline_checkpoint_parameter_source(parameter_provider.get()),
      ID4_PIPELINE_STAGE_PARAMETER_RESIDENCY_RESIDENT,
      /*maximum_parameter_window_byte_length=*/0);
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
  IREE_ASSERT_OK(
      id4::test::Ideogram4DitQueueInitializedBoundaryTensorsFromFixture(
          context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
          boundary_bindings, fixture_tensors, branch, update_semaphore.get(),
          &update_value));
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
  issue_options.execution_segment_submission_window = 1;
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
        branch.diagnostic_tap_prefix, read_wait.list()));
  }
  iree_string_view_t pytorch_oracle_directory =
      iree_make_cstring_view(FLAG_id4_pytorch_oracle_dir);
  if (!iree_string_view_is_empty(pytorch_oracle_directory)) {
    if (captures_diagnostic_taps) {
      IREE_ASSERT_OK(ComparePytorchOracleDiagnosticTaps(
          context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
          diagnostic_tap_bindings, request, pytorch_oracle_directory,
          read_wait.list()));
    }
    if (iree_all_bits_set(options.flags,
                          ID4_DIT_FIXTURE_RUN_FLAG_COMPARE_OUTPUT_BOUNDARY)) {
      IREE_ASSERT_OK(ComparePytorchOracleOutputBoundary(
          context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
          boundary_bindings, request, pytorch_oracle_directory,
          read_wait.list()));
    }
  } else if (iree_all_bits_set(
                 options.flags,
                 ID4_DIT_FIXTURE_RUN_FLAG_COMPARE_OUTPUT_BOUNDARY)) {
    IREE_ASSERT_OK(CompareExpectedOutputBoundary(
        context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
        boundary_bindings, fixture_tensors, branch.expected_velocity_name,
        read_wait.list()));
  }
  IREE_ASSERT_OK(VerifyExportedBoundariesWereWritten(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      boundary_bindings, read_wait.list()));
}

TEST(Ideogram4DitStageIntegration, PrepareAndIssueForwardPreludeFixture) {
  RunDitFixture(DitFixtureRunOptions{
      .activation_format = ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_F32_CANONICAL,
      .weight_execution_format =
          ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_BF16_RESIDENT,
      .attention_implementation =
          ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING,
      .feed_forward_implementation =
          ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_FUSED_PRODUCT,
      .branch = id4::test::Ideogram4DitBranch::kConditioned,
      .capture_run_id = IREE_SV("ideogram4_dit_forward_integration"),
      .flags = ID4_DIT_FIXTURE_RUN_FLAG_CAPTURE_EXPECTED_TAPS |
               ID4_DIT_FIXTURE_RUN_FLAG_COMPARE_EXPECTED_TAPS |
               ID4_DIT_FIXTURE_RUN_FLAG_COMPARE_OUTPUT_BOUNDARY,
  });
}

static void RunBf16LinearInputFixture(id4::test::Ideogram4DitBranch branch,
                                      iree_string_view_t capture_run_id) {
  RunDitFixture(DitFixtureRunOptions{
      .activation_format =
          ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT,
      .weight_execution_format =
          ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_BF16_RESIDENT,
      .attention_implementation =
          ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING,
      .feed_forward_implementation =
          ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_PYTORCH_PARITY,
      .branch = branch,
      .capture_run_id = capture_run_id,
      .flags = ID4_DIT_FIXTURE_RUN_FLAG_COMPARE_OUTPUT_BOUNDARY,
  });
}

TEST(Ideogram4DitStageIntegration, PrepareAndIssueBf16LinearInputFixture) {
  RunBf16LinearInputFixture(
      id4::test::Ideogram4DitBranch::kConditioned,
      IREE_SV("ideogram4_dit_bf16_linear_input_integration"));
}

TEST(Ideogram4DitStageIntegration,
     PrepareAndIssueBf16LinearInputUnconditionedFixture) {
  RunBf16LinearInputFixture(
      id4::test::Ideogram4DitBranch::kUnconditioned,
      IREE_SV("ideogram4_dit_bf16_linear_input_unconditioned_integration"));
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
      .weight_execution_format =
          ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_BF16_RESIDENT,
      .attention_implementation =
          ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_MATERIALIZED_WMMA,
      .feed_forward_implementation =
          ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_PYTORCH_PARITY,
      .branch = id4::test::Ideogram4DitBranch::kConditioned,
      .diagnostic_tap_names =
          {
              IREE_ARRAYSIZE(diagnostic_tap_names),
              diagnostic_tap_names,
          },
      .capture_run_id =
          IREE_SV("ideogram4_dit_materialized_wmma_attention_integration"),
      .flags = ID4_DIT_FIXTURE_RUN_FLAG_VERIFY_DIAGNOSTIC_TAPS_WRITTEN,
  });
}

static void RunFp8CompactRhsFixture(
    id4::test::Ideogram4DitBranch branch,
    id4_ideogram4_dit_attention_implementation_t attention_implementation,
    iree_string_view_t capture_run_id) {
  id4_ideogram4_dit_parameter_format_t parameter_format =
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_INVALID;
  IREE_ASSERT_OK(ParseDitParameterFormat(&parameter_format));
  ASSERT_EQ(parameter_format, ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_FP8_E4M3)
      << "compact FP8 integration requires --dit_parameter_format=fp8_e4m3";
  ASSERT_FALSE(iree_string_view_is_empty(
      iree_make_cstring_view(FLAG_id4_pytorch_oracle_dir)))
      << "compact FP8 integration requires --id4_pytorch_oracle_dir";
  RunDitFixture(DitFixtureRunOptions{
      .activation_format =
          ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT,
      .weight_execution_format =
          ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_COMPACT_RHS,
      .attention_implementation = attention_implementation,
      .feed_forward_implementation =
          ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_PYTORCH_PARITY,
      .branch = branch,
      .capture_run_id = capture_run_id,
      .flags = ID4_DIT_FIXTURE_RUN_FLAG_COMPARE_OUTPUT_BOUNDARY,
  });
}

TEST(Ideogram4DitStageIntegration,
     PrepareAndIssueFp8CompactRhsMaterializedFixture) {
  RunFp8CompactRhsFixture(
      id4::test::Ideogram4DitBranch::kConditioned,
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_MATERIALIZED_WMMA,
      IREE_SV("ideogram4_dit_fp8_compact_rhs_materialized"));
}

TEST(Ideogram4DitStageIntegration,
     PrepareAndIssueFp8CompactRhsMaterializedUnconditionedFixture) {
  RunFp8CompactRhsFixture(
      id4::test::Ideogram4DitBranch::kUnconditioned,
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_MATERIALIZED_WMMA,
      IREE_SV("ideogram4_dit_fp8_compact_rhs_materialized_unconditioned"));
}

TEST(Ideogram4DitStageIntegration,
     PrepareAndIssueFp8CompactRhsStreamingFixture) {
  RunFp8CompactRhsFixture(id4::test::Ideogram4DitBranch::kConditioned,
                          ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING,
                          IREE_SV("ideogram4_dit_fp8_compact_rhs_streaming"));
}

TEST(Ideogram4DitStageIntegration, PrepareAndIssueBlockedWmmaAttentionFixture) {
  const iree_string_view_t diagnostic_tap_names[] = {
      IREE_SV("ideogram4.cond.layers.0.attention.context"),
      IREE_SV("ideogram4.cond.layers.0.attention.output"),
  };
  RunDitFixture(DitFixtureRunOptions{
      .activation_format =
          ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT,
      .weight_execution_format =
          ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_BF16_RESIDENT,
      .attention_implementation =
          ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_BLOCKED_WMMA,
      .feed_forward_implementation =
          ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_PYTORCH_PARITY,
      .branch = id4::test::Ideogram4DitBranch::kConditioned,
      .diagnostic_tap_names =
          {
              IREE_ARRAYSIZE(diagnostic_tap_names),
              diagnostic_tap_names,
          },
      .capture_run_id =
          IREE_SV("ideogram4_dit_blocked_wmma_attention_integration"),
      .flags = ID4_DIT_FIXTURE_RUN_FLAG_VERIFY_DIAGNOSTIC_TAPS_WRITTEN,
  });
}

TEST(Ideogram4DitStageIntegration, PrepareAndIssueOnlineWmmaAttentionFixture) {
  const iree_string_view_t diagnostic_tap_names[] = {
      IREE_SV("ideogram4.cond.layers.0.attention.context"),
      IREE_SV("ideogram4.cond.layers.0.attention.output"),
  };
  RunDitFixture(DitFixtureRunOptions{
      .activation_format =
          ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT,
      .weight_execution_format =
          ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_BF16_RESIDENT,
      .attention_implementation =
          ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_ONLINE_WMMA,
      .feed_forward_implementation =
          ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_PYTORCH_PARITY,
      .branch = id4::test::Ideogram4DitBranch::kConditioned,
      .diagnostic_tap_names =
          {
              IREE_ARRAYSIZE(diagnostic_tap_names),
              diagnostic_tap_names,
          },
      .capture_run_id =
          IREE_SV("ideogram4_dit_online_wmma_attention_integration"),
      .flags = ID4_DIT_FIXTURE_RUN_FLAG_VERIFY_DIAGNOSTIC_TAPS_WRITTEN,
  });
}

TEST(Ideogram4DitStageIntegration,
     PrepareAndIssueOnlineWmmaAttentionFp8CompactRhsFixture) {
  RunFp8CompactRhsFixture(
      id4::test::Ideogram4DitBranch::kConditioned,
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_ONLINE_WMMA,
      IREE_SV("ideogram4_dit_online_wmma_attention_fp8_compact_rhs"));
}

TEST(Ideogram4DitStageIntegration,
     PrepareAndIssueOnlineWmmaAttentionFp8CompactRhsFusedFeedForwardFixture) {
  const iree_string_view_t diagnostic_tap_names[] = {
      IREE_SV("ideogram4.cond.layers.0.attention.context"),
      IREE_SV("ideogram4.cond.layers.0.attention.output"),
  };
  RunDitFixture(DitFixtureRunOptions{
      .activation_format =
          ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT,
      .weight_execution_format =
          ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_COMPACT_RHS,
      .attention_implementation =
          ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_ONLINE_WMMA,
      .feed_forward_implementation =
          ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_FUSED_PRODUCT,
      .branch = id4::test::Ideogram4DitBranch::kConditioned,
      .diagnostic_tap_names =
          {
              IREE_ARRAYSIZE(diagnostic_tap_names),
              diagnostic_tap_names,
          },
      .capture_run_id = IREE_SV(
          "ideogram4_dit_online_wmma_attention_fp8_compact_rhs_fused_ffn"),
      .flags = ID4_DIT_FIXTURE_RUN_FLAG_VERIFY_DIAGNOSTIC_TAPS_WRITTEN,
  });
}

static void RunFusedFeedForwardFixture(
    id4::test::Ideogram4DitBranch branch,
    id4_ideogram4_dit_weight_execution_format_t weight_execution_format,
    iree_string_view_t hidden_tap_name, iree_string_view_t capture_run_id) {
  const iree_string_view_t diagnostic_tap_names[] = {
      hidden_tap_name,
  };
  RunDitFixture(DitFixtureRunOptions{
      .activation_format =
          ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT,
      .weight_execution_format = weight_execution_format,
      .attention_implementation =
          ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_ONLINE_WMMA,
      .feed_forward_implementation =
          ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_FUSED_PRODUCT,
      .branch = branch,
      .diagnostic_tap_names =
          {
              IREE_ARRAYSIZE(diagnostic_tap_names),
              diagnostic_tap_names,
          },
      .capture_run_id = capture_run_id,
      .flags = ID4_DIT_FIXTURE_RUN_FLAG_VERIFY_DIAGNOSTIC_TAPS_WRITTEN,
  });
}

TEST(Ideogram4DitStageIntegration,
     PrepareAndIssueOnlineWmmaFusedFeedForwardPytorchOracleFixture) {
  RunFusedFeedForwardFixture(
      id4::test::Ideogram4DitBranch::kConditioned,
      ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_BF16_RESIDENT,
      IREE_SV("ideogram4.cond.layers.0.ffn.hidden"),
      IREE_SV("ideogram4_dit_online_wmma_fused_ffn_pytorch_oracle"));
}

TEST(
    Ideogram4DitStageIntegration,
    PrepareAndIssueOnlineWmmaFusedFeedForwardUnconditionedPytorchOracleFixture) {
  RunFusedFeedForwardFixture(
      id4::test::Ideogram4DitBranch::kUnconditioned,
      ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_BF16_RESIDENT,
      IREE_SV("ideogram4.uncond.layers.0.ffn.hidden"),
      IREE_SV(
          "ideogram4_dit_online_wmma_fused_ffn_unconditioned_pytorch_oracle"));
}

TEST(
    Ideogram4DitStageIntegration,
    PrepareAndIssueOnlineWmmaFp8CompactRhsFusedFeedForwardPytorchOracleFixture) {
  RunFusedFeedForwardFixture(
      id4::test::Ideogram4DitBranch::kConditioned,
      ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_COMPACT_RHS,
      IREE_SV("ideogram4.cond.layers.0.ffn.hidden"),
      IREE_SV("ideogram4_dit_online_wmma_fp8_compact_rhs_fused_ffn_pytorch_"
              "oracle"));
}

TEST(
    Ideogram4DitStageIntegration,
    PrepareAndIssueOnlineWmmaFp8CompactRhsFusedFeedForwardUnconditionedPytorchOracleFixture) {
  RunFusedFeedForwardFixture(
      id4::test::Ideogram4DitBranch::kUnconditioned,
      ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_COMPACT_RHS,
      IREE_SV("ideogram4.uncond.layers.0.ffn.hidden"),
      IREE_SV(
          "ideogram4_dit_online_wmma_fp8_compact_rhs_fused_ffn_unconditioned_"
          "pytorch_oracle"));
}

TEST(Ideogram4DitStageIntegration,
     PrepareAndIssueOnlineWmmaAttentionUnconditionedFixture) {
  RunFp8CompactRhsFixture(
      id4::test::Ideogram4DitBranch::kUnconditioned,
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_ONLINE_WMMA,
      IREE_SV(
          "ideogram4_dit_online_wmma_attention_fp8_compact_rhs_unconditioned"));
}

}  // namespace
