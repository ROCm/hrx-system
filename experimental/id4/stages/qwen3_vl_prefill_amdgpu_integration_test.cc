// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "experimental/id4/pipeline/kernel_cache.h"
#include "experimental/id4/pipeline/parameter_slab.h"
#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/hal_integration_util.h"
#include "experimental/id4/stages/qwen3_vl_prefill.h"
#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/io/file_contents.h"
#include "iree/io/parameter_provider.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

IREE_FLAG(string, rmsnorm_loom_source, "",
          "Path to kernels/qwen3_vl/rmsnorm.loom.");
IREE_FLAG(string, linear_loom_source, "",
          "Path to kernels/qwen3_vl/linear_bf16_f32.loom.");
IREE_FLAG(string, device_uri, "", "HAL device URI, such as amdgpu://0.");
IREE_FLAG(string, amdgpu_processor, "", "AMDGPU processor, such as gfx1100.");

namespace {

static constexpr int32_t kTestTokenCount = 1;
static constexpr int32_t kTestHiddenSize =
    ID4_QWEN3_VL_PREFILL_STAGE_HIDDEN_SIZE;

using BundleRef =
    id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>;
using ExecutableCacheRef =
    id4::test::OwningRef<iree_hal_executable_cache_t,
                         iree_hal_executable_cache_release>;
using FileContentsRef =
    id4::test::OwningRef<iree_io_file_contents_t, iree_io_file_contents_free>;
using KernelCacheRef = id4::test::OwningRef<id4_pipeline_kernel_cache_t,
                                            id4_pipeline_kernel_cache_release>;
using KernelLibraryRef =
    id4::test::OwningRef<id4_pipeline_kernel_library_t,
                         id4_pipeline_kernel_library_release>;
using HalBufferRef =
    id4::test::OwningRef<iree_hal_buffer_t, iree_hal_buffer_release>;
using ParameterProviderRef =
    id4::test::OwningRef<iree_io_parameter_provider_t,
                         iree_io_parameter_provider_release>;
using PlanRef =
    id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release>;
using SemaphoreRef =
    id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>;
using StageRef =
    id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release>;

static bool ValidateRequiredFlags() {
  bool is_valid = true;
  if (std::strlen(FLAG_rmsnorm_loom_source) == 0) {
    std::fprintf(stderr, "--rmsnorm_loom_source is required\n");
    is_valid = false;
  }
  if (std::strlen(FLAG_linear_loom_source) == 0) {
    std::fprintf(stderr, "--linear_loom_source is required\n");
    is_valid = false;
  }
  if (std::strlen(FLAG_device_uri) == 0) {
    std::fprintf(stderr, "--device_uri is required\n");
    is_valid = false;
  }
  if (std::strlen(FLAG_amdgpu_processor) == 0) {
    std::fprintf(stderr, "--amdgpu_processor is required\n");
    is_valid = false;
  }
  return is_valid;
}

static iree_device_size_t HiddenStatesByteLength() {
  return static_cast<iree_device_size_t>(kTestTokenCount) * kTestHiddenSize *
         sizeof(float);
}

static iree_device_size_t RmsnormWeightByteLength() {
  return kTestHiddenSize * sizeof(float);
}

static iree_device_size_t QProjectionWeightByteLength() {
  return static_cast<iree_device_size_t>(kTestHiddenSize) * kTestHiddenSize *
         sizeof(uint16_t);
}

static iree_status_t GatherOracleWeights(
    iree_io_parameter_provider_t* parameter_provider, iree_hal_device_t* device,
    iree_hal_buffer_t** out_weight_buffer) {
  IREE_ASSERT_ARGUMENT(out_weight_buffer);
  *out_weight_buffer = nullptr;

  const iree_device_size_t rmsnorm_byte_length = RmsnormWeightByteLength();
  const iree_device_size_t q_projection_byte_length =
      QProjectionWeightByteLength();
  const iree_device_size_t oracle_weight_byte_length =
      rmsnorm_byte_length + q_projection_byte_length;

  iree_hal_buffer_params_t buffer_params = {};
  buffer_params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                       IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
                       IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
  buffer_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  buffer_params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  buffer_params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  buffer_params.min_alignment = alignof(float);

  HalBufferRef weight_buffer;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), buffer_params,
      oracle_weight_byte_length, weight_buffer.out()));

  const iree_io_parameter_span_t rmsnorm_span = {
      // Source byte offset within the RMSNorm parameter.
      .parameter_offset = 0,
      // Target byte offset within the oracle buffer.
      .buffer_offset = 0,
      // Number of RMSNorm weight bytes to gather.
      .length = rmsnorm_byte_length,
  };
  const iree_io_parameter_span_t q_projection_span = {
      // Source byte offset within the Q projection parameter.
      .parameter_offset = 0,
      // Target byte offset within the oracle buffer.
      .buffer_offset = rmsnorm_byte_length,
      // Number of Q projection weight bytes to gather.
      .length = q_projection_byte_length,
  };
  const id4_pipeline_parameter_request_t requests[] = {
      id4_pipeline_parameter_request(
          IREE_SV(ID4_QWEN3_VL_PREFILL_STAGE_INPUT_RMSNORM_WEIGHT_KEY),
          rmsnorm_span),
      id4_pipeline_parameter_request(
          IREE_SV(ID4_QWEN3_VL_PREFILL_STAGE_Q_PROJECTION_WEIGHT_KEY),
          q_projection_span),
  };
  const id4_pipeline_parameter_slab_plan_t slab =
      id4_pipeline_make_parameter_slab_plan(
          IREE_SV(ID4_QWEN3_VL_PREFILL_STAGE_PARAMETER_SCOPE),
          /*placement_id=*/0, buffer_params, oracle_weight_byte_length,
          alignof(float), IREE_ARRAYSIZE(requests), requests);
  id4_pipeline_parameter_slab_enumerator_state_t enumerator_state = {
      // Slab plan supplying request keys and spans.
      .slab = &slab,
  };
  iree_io_parameter_enumerator_t enumerator =
      id4_pipeline_parameter_slab_enumerator(&enumerator_state);

  SemaphoreRef gather_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, gather_semaphore.out()));
  iree_hal_semaphore_t* gather_signal_semaphores[] = {gather_semaphore.get()};
  uint64_t gather_signal_payload_values[] = {1};
  iree_hal_semaphore_list_t gather_signal_list = {
      // Number of oracle gather signal semaphores.
      .count = IREE_ARRAYSIZE(gather_signal_semaphores),
      // Oracle gather signal semaphore pointer array.
      .semaphores = gather_signal_semaphores,
      // Oracle gather signal payload values.
      .payload_values = gather_signal_payload_values,
  };
  IREE_RETURN_IF_ERROR(iree_io_parameter_provider_gather(
      parameter_provider, device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), gather_signal_list, slab.scope,
      weight_buffer.get(), slab.request_count, enumerator));
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_wait(
      gather_semaphore.get(), gather_signal_payload_values[0],
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  *out_weight_buffer = weight_buffer.get();
  weight_buffer.reset();
  return iree_ok_status();
}

static void ComputeExpectedHiddenStates(
    const std::vector<float>& input_hidden_states,
    const void* oracle_weight_data,
    std::vector<float>* expected_hidden_states) {
  const iree_device_size_t rmsnorm_byte_length = RmsnormWeightByteLength();
  const float* rmsnorm_weights = static_cast<const float*>(oracle_weight_data);
  const uint16_t* q_projection_weights = reinterpret_cast<const uint16_t*>(
      static_cast<const uint8_t*>(oracle_weight_data) + rmsnorm_byte_length);

  expected_hidden_states->assign(
      static_cast<size_t>(kTestTokenCount) * kTestHiddenSize, 0.0f);
  std::vector<float> normalized_hidden_states(
      static_cast<size_t>(kTestTokenCount) * kTestHiddenSize, 0.0f);
  for (int32_t token = 0; token < kTestTokenCount; ++token) {
    float square_sum = 0.0f;
    const size_t row_offset = static_cast<size_t>(token) * kTestHiddenSize;
    for (int32_t column = 0; column < kTestHiddenSize; ++column) {
      const float value = input_hidden_states[row_offset + column];
      square_sum += value * value;
    }
    const float row_scale =
        1.0f / std::sqrt(square_sum / kTestHiddenSize + 1.0e-6f);
    for (int32_t column = 0; column < kTestHiddenSize; ++column) {
      const float value = input_hidden_states[row_offset + column];
      normalized_hidden_states[row_offset + column] =
          value * row_scale * rmsnorm_weights[column];
    }
  }
  for (int32_t token = 0; token < kTestTokenCount; ++token) {
    const size_t row_offset = static_cast<size_t>(token) * kTestHiddenSize;
    for (int32_t output_row = 0; output_row < kTestHiddenSize; ++output_row) {
      float dot = 0.0f;
      const size_t weight_row_offset =
          static_cast<size_t>(output_row) * kTestHiddenSize;
      for (int32_t input_column = 0; input_column < kTestHiddenSize;
           ++input_column) {
        const uint16_t weight_bits =
            q_projection_weights[weight_row_offset + input_column];
        const float weight_value = iree_math_bf16_to_f32(weight_bits);
        const float input_value =
            normalized_hidden_states[row_offset + input_column];
        dot += input_value * weight_value;
      }
      (*expected_hidden_states)[row_offset + output_row] = dot;
    }
  }
}

TEST(Qwen3VlPrefillAmdgpuIntegrationTest, PrepareIssuePrefillForward) {
  id4::test::LiveHalDevice live_device;
  IREE_ASSERT_OK(id4::test::CreateLiveHalDevice(
      iree_make_cstring_view(FLAG_device_uri), &live_device));

  ExecutableCacheRef executable_cache;
  IREE_ASSERT_OK(iree_hal_executable_cache_create(
      live_device.device.get(), IREE_SV("id4-qwen3-vl-prefill-integration"),
      executable_cache.out()));

  KernelCacheRef kernel_cache;
  id4_pipeline_kernel_cache_create_options_t kernel_cache_options;
  std::memset(&kernel_cache_options, 0, sizeof(kernel_cache_options));
  kernel_cache_options.structure_size = sizeof(kernel_cache_options);
  kernel_cache_options.amdgpu_processor =
      iree_make_cstring_view(FLAG_amdgpu_processor);
  IREE_ASSERT_OK(id4_pipeline_kernel_cache_create(
      &kernel_cache_options, iree_allocator_system(), kernel_cache.out()));

  FileContentsRef rmsnorm_source_file;
  IREE_ASSERT_OK(iree_io_file_contents_read(
      iree_make_cstring_view(FLAG_rmsnorm_loom_source), iree_allocator_system(),
      rmsnorm_source_file.out()));
  FileContentsRef linear_source_file;
  IREE_ASSERT_OK(iree_io_file_contents_read(
      iree_make_cstring_view(FLAG_linear_loom_source), iree_allocator_system(),
      linear_source_file.out()));
  id4_pipeline_kernel_module_t modules[] = {
      {
          // Stable module path selected by the Qwen3-VL prefill stage.
          .module_path = IREE_SV("qwen3_vl/rmsnorm"),
          // Diagnostic source identifier for the current VFS entry.
          .source_identifier = IREE_SV("kernels/qwen3_vl/rmsnorm.loom"),
          // Source bytes loaded by the integration test harness.
          .source_contents = rmsnorm_source_file.get()->const_buffer,
      },
      {
          // Stable module path selected by the Qwen3-VL prefill stage.
          .module_path = IREE_SV("qwen3_vl/linear_bf16_f32"),
          // Diagnostic source identifier for the current VFS entry.
          .source_identifier = IREE_SV("kernels/qwen3_vl/linear_bf16_f32.loom"),
          // Source bytes loaded by the integration test harness.
          .source_contents = linear_source_file.get()->const_buffer,
      },
  };
  id4_pipeline_kernel_library_create_options_t library_options;
  std::memset(&library_options, 0, sizeof(library_options));
  library_options.structure_size = sizeof(library_options);
  library_options.module_count = IREE_ARRAYSIZE(modules);
  library_options.modules = modules;
  KernelLibraryRef kernel_library;
  IREE_ASSERT_OK(id4_pipeline_kernel_library_create(
      &library_options, iree_allocator_system(), kernel_library.out()));

  ParameterProviderRef parameter_provider;
  IREE_ASSERT_OK(id4::test::CreateParameterProviderFromFlags(
      IREE_SV(ID4_QWEN3_VL_PREFILL_STAGE_PARAMETER_SCOPE),
      parameter_provider.out()));

  HalBufferRef oracle_weight_buffer;
  IREE_ASSERT_OK(GatherOracleWeights(parameter_provider.get(),
                                     live_device.device.get(),
                                     oracle_weight_buffer.out()));

  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = live_device.device_group.get();
  services.executable_cache = executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_qwen3_vl_prefill_stage_create_options_t create_options;
  std::memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.kernel_cache = kernel_cache.get();
  create_options.token_count = kTestTokenCount;

  StageRef stage;
  IREE_ASSERT_OK(id4_qwen3_vl_prefill_stage_create(
      &create_options, iree_allocator_system(), stage.out()));

  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);

  id4_pipeline_stage_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage.get(), &load_options));

  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;

  PlanRef plan;
  IREE_ASSERT_OK(
      id4_pipeline_stage_plan(stage.get(), &plan_options, plan.out()));

  SemaphoreRef readiness_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      live_device.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
      readiness_semaphore.out()));
  iree_hal_semaphore_t* readiness_semaphores[] = {readiness_semaphore.get()};
  uint64_t readiness_payload_values[] = {1};
  iree_hal_semaphore_list_t readiness_signal_list = {
      // Number of prepare readiness semaphores.
      .count = IREE_ARRAYSIZE(readiness_semaphores),
      // Prepare readiness semaphore pointer array.
      .semaphores = readiness_semaphores,
      // Prepare readiness payload values.
      .payload_values = readiness_payload_values,
  };

  id4_pipeline_stage_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.kernel_library = kernel_library.get();
  prepare_options.parameter_provider = parameter_provider.get();
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = readiness_signal_list;
  prepare_options.diagnostics_sink = &diagnostics_sink;

  BundleRef bundle;
  IREE_ASSERT_OK(id4_pipeline_stage_prepare(stage.get(), plan.get(),
                                            &prepare_options, bundle.out()));

  std::vector<float> input_hidden_states(
      static_cast<size_t>(kTestTokenCount) *
          ID4_QWEN3_VL_PREFILL_STAGE_HIDDEN_SIZE,
      2.0f);
  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      live_device.device.get(), input_hidden_states.data(),
      id4_qwen3_vl_prefill_stage_bundle_input_buffer(bundle.get()),
      /*target_offset=*/0, HiddenStatesByteLength(),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

  SemaphoreRef completion_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      live_device.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
      completion_semaphore.out()));
  iree_hal_semaphore_t* signal_semaphores[] = {completion_semaphore.get()};
  uint64_t signal_payload_values[] = {1};
  iree_hal_semaphore_list_t signal_list = {
      // Number of final signal semaphores.
      .count = IREE_ARRAYSIZE(signal_semaphores),
      // Final signal semaphore pointer array.
      .semaphores = signal_semaphores,
      // Final signal payload values.
      .payload_values = signal_payload_values,
  };

  id4_pipeline_stage_issue_options_t issue_options;
  std::memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  issue_options.signal_semaphore_list = signal_list;
  issue_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(
      id4_pipeline_stage_issue(stage.get(), bundle.get(), &issue_options));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      completion_semaphore.get(), signal_payload_values[0],
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_mapping_t oracle_weight_mapping;
  std::memset(&oracle_weight_mapping, 0, sizeof(oracle_weight_mapping));
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      oracle_weight_buffer.get(), IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, /*byte_offset=*/0,
      iree_hal_buffer_byte_length(oracle_weight_buffer.get()),
      &oracle_weight_mapping));
  std::vector<float> expected_hidden_states;
  ComputeExpectedHiddenStates(input_hidden_states,
                              oracle_weight_mapping.contents.data,
                              &expected_hidden_states);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&oracle_weight_mapping));

  std::vector<float> actual_hidden_states(
      static_cast<size_t>(kTestTokenCount) *
          ID4_QWEN3_VL_PREFILL_STAGE_HIDDEN_SIZE,
      0.0f);
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      live_device.device.get(),
      id4_qwen3_vl_prefill_stage_bundle_output_buffer(bundle.get()),
      /*source_offset=*/0, actual_hidden_states.data(),
      HiddenStatesByteLength(), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));

  for (size_t i = 0; i < actual_hidden_states.size(); ++i) {
    const float tolerance =
        std::max(1.0e-2f, std::abs(expected_hidden_states[i]) * 1.0e-4f);
    EXPECT_NEAR(actual_hidden_states[i], expected_hidden_states[i], tolerance)
        << "prefill hidden-state element " << i;
  }
  EXPECT_GT(diagnostics.event_count, 0u);
  EXPECT_GT(diagnostics.kernel_event_count, 0u);
}

}  // namespace

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "qwen3_vl_prefill_amdgpu_integration_test",
      "Runs the Qwen3-VL prefill stage against real HAL AMDGPU execution.\n"
      "Pass --parameters=<file> to provide Qwen3-VL weights.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK |
                               IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP,
                           &argc, &argv);
  if (!ValidateRequiredFlags()) return 1;
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
