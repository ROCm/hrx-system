// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/testbench_actual.h"

#include "iree/base/internal/arena.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/tooling/testbench/testbench.h"
#include "loomc/interop.h"
#include "loomc/iree.h"

namespace loom {
namespace {

class HalTestbenchActualTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &plan_arena_);

    IREE_ASSERT_OK(iree_status_from_loomc(loomc_context_create(
        /*options=*/nullptr, loomc_allocator_system(), &compiler_context_)));
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_workspace_create(
        /*options=*/nullptr, loomc_allocator_system(), &compiler_workspace_)));
  }

  void TearDown() override {
    loomc_workspace_release(compiler_workspace_);
    loomc_context_release(compiler_context_);
    iree_arena_deinitialize(&plan_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void ParseAndPlan(iree_string_view_t source, loomc_module_t** out_module,
                    const loom_module_t** out_native_module,
                    loom_testbench_module_plan_t* out_plan) {
    *out_module = ParsePublicModule(source);
    loomc_module_interop_view_t view = {};
    loomc_result_t* result = nullptr;
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_module_get_interop_view(
        *out_module, loomc_allocator_system(), &view, &result)));
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(loomc_result_succeeded(result));
    loomc_result_release(result);
    if (out_native_module != nullptr) {
      *out_native_module = view.module;
    }
    IREE_ASSERT_OK(loom_testbench_plan_module(view.module, nullptr,
                                              &plan_arena_, out_plan));
    ASSERT_EQ(out_plan->issue_count, 0u);
  }

  loomc_module_t* ParsePublicModule(iree_string_view_t source);

  loomc_launch_config_program_t* LoadLaunchConfigProgram(
      iree_string_view_t source);

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t plan_arena_;
  loomc_context_t* compiler_context_ = nullptr;
  loomc_workspace_t* compiler_workspace_ = nullptr;
};

loomc_module_t* HalTestbenchActualTest::ParsePublicModule(
    iree_string_view_t source) {
  const loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(source_options),
      .next = nullptr,
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view("hal_testbench_actual_test.loom"),
      .contents = loomc_make_byte_span(source.data, source.size),
      .storage = LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* compiler_source = nullptr;
  IREE_EXPECT_OK(iree_status_from_loomc(loomc_source_create(
      &source_options, loomc_allocator_system(), &compiler_source)));
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  IREE_EXPECT_OK(iree_status_from_loomc(loomc_module_deserialize_from_source(
      compiler_context_, compiler_workspace_, compiler_source,
      /*options=*/nullptr, loomc_allocator_system(), &module, &result)));
  EXPECT_NE(result, nullptr);
  EXPECT_TRUE(result != nullptr && loomc_result_succeeded(result));
  loomc_result_release(result);
  loomc_source_release(compiler_source);
  return module;
}

loomc_launch_config_program_t* HalTestbenchActualTest::LoadLaunchConfigProgram(
    iree_string_view_t source) {
  loomc_module_t* module = ParsePublicModule(source);
  const loomc_module_serialize_options_t serialize_options = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      .structure_size = sizeof(serialize_options),
      .next = nullptr,
      .format = LOOMC_SOURCE_FORMAT_BYTECODE,
      .identifier = loomc_make_cstring_view("launch_config.loombc"),
  };
  loomc_source_t* serialized_source = nullptr;
  IREE_EXPECT_OK(iree_status_from_loomc(loomc_module_serialize_to_source(
      module, &serialize_options, loomc_allocator_system(),
      &serialized_source)));
  loomc_byte_sequence_t* contents = nullptr;
  IREE_EXPECT_OK(iree_status_from_loomc(
      loomc_byte_sequence_create_copy(loomc_source_contents(serialized_source),
                                      loomc_allocator_system(), &contents)));
  const loomc_artifact_t artifact = {
      .kind = LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
      .format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE),
      .identifier = loomc_make_cstring_view("launch_config.loombc"),
      .contents = contents,
  };
  loomc_launch_config_program_t* program = nullptr;
  IREE_EXPECT_OK(iree_status_from_loomc(loomc_launch_config_program_load(
      &artifact, loomc_allocator_system(), &program)));
  loomc_byte_sequence_release(contents);
  loomc_source_release(serialized_source);
  loomc_module_release(module);
  return program;
}

static loom_testbench_value_t I32Value(int32_t value) {
  loom_testbench_value_t result = {.kind = LOOM_TESTBENCH_VALUE_KIND_SCALAR};
  result.scalar.kind = IREE_TOOLING_VALUE_KIND_I32;
  result.scalar.storage.i32 = value;
  return result;
}

static loom_testbench_value_t RawU32Value(uint32_t value) {
  loom_testbench_value_t result = {.kind = LOOM_TESTBENCH_VALUE_KIND_SCALAR};
  result.scalar.kind = IREE_TOOLING_VALUE_KIND_RAW_U32;
  result.scalar.storage.u32 = value;
  return result;
}

static loom_testbench_value_t I64Value(int64_t value) {
  loom_testbench_value_t result = {.kind = LOOM_TESTBENCH_VALUE_KIND_SCALAR};
  result.scalar.kind = IREE_TOOLING_VALUE_KIND_I64;
  result.scalar.storage.i64 = value;
  return result;
}

static loom_testbench_value_t F32Value(float value) {
  loom_testbench_value_t result = {.kind = LOOM_TESTBENCH_VALUE_KIND_SCALAR};
  result.scalar.kind = IREE_TOOLING_VALUE_KIND_F32;
  result.scalar.storage.f32 = value;
  return result;
}

static loom_testbench_value_t F64Value(double value) {
  loom_testbench_value_t result = {.kind = LOOM_TESTBENCH_VALUE_KIND_SCALAR};
  result.scalar.kind = IREE_TOOLING_VALUE_KIND_F64;
  result.scalar.storage.f64 = value;
  return result;
}

static loom_target_snapshot_t AddressTargetSnapshot(uint32_t index_bitwidth,
                                                    uint32_t offset_bitwidth) {
  loom_target_snapshot_t snapshot = {.index_bitwidth = index_bitwidth,
                                     .offset_bitwidth = offset_bitwidth};
  return snapshot;
}

static loom_target_snapshot_t FakeTargetSnapshot() {
  loom_target_snapshot_t snapshot = AddressTargetSnapshot(32, 64);
  snapshot.name = IREE_SV("fake-snapshot");
  return snapshot;
}

static const loom_target_snapshot_t kFakeTargetSnapshot = FakeTargetSnapshot();
static const loom_target_snapshot_t kIndex32Offset64TargetSnapshot =
    AddressTargetSnapshot(32, 64);
static const loom_target_snapshot_t kIndex64Offset32TargetSnapshot =
    AddressTargetSnapshot(64, 32);

static iree_status_t InitializeFakeHalContext(
    loom_run_hal_testbench_context_t* context,
    iree_hal_queue_t* out_dispatch_queue) {
  const iree_hal_queue_priority_t normal_priority =
      IREE_HAL_QUEUE_PRIORITY_NORMAL;
  const iree_hal_queue_family_spec_t queue_family = {
      .name = IREE_SV("dispatch"),
      .provisioned_queue_count = 0,
      .priority_count = 1,
      .priorities = &normal_priority,
      .execution_unit_count = 0,
      .execution_resource_group_count = 0,
      .execution_resource_groups = nullptr,
      .execution_resource_count = 0,
      .execution_resources = nullptr,
      .supported_queue_features = IREE_HAL_QUEUE_FEATURE_FLAG_NONE,
      .timestamp_valid_bits = 0,
      .timestamp_frequency_hz = 0,
      .physical_device_affinity = 1,
      .role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH,
      .atomic_capabilities = {},
      .zero_compute_atomic_capabilities = {},
      .flags = IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_NONE,
  };
  const iree_hal_device_queue_spec_t queues = {
      .family_count = 1,
      .families = &queue_family,
  };
  iree_hal_device_spec_params_t params = {.queues = &queues};
  iree_hal_device_spec_t* device_spec = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_create(
      &params, iree_allocator_system(), &device_spec));

  iree_hal_mock_device_options_t mock_options;
  iree_hal_mock_device_options_initialize(&mock_options);
  mock_options.identifier = IREE_SV("fake");
  mock_options.device_spec = device_spec;
  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_mock_device_create(
      &mock_options, iree_allocator_system(), &device);
  iree_hal_device_spec_release(device_spec);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  out_dispatch_queue->queue_family = iree_hal_device_queue_family(device, 0);
  context->runtime = (loom_run_hal_runtime_t){
      .device = device,
      .dispatch_queue = out_dispatch_queue,
  };
  context->runtime_initialized = true;
  return iree_ok_status();
}

TEST_F(HalTestbenchActualTest, RequiresExplicitDevice) {
  loom_run_hal_testbench_context_t context = {};
  loom_run_hal_testbench_context_initialize(
      /*target_environment=*/nullptr, /*target_routes=*/nullptr,
      /*target_route_count=*/0, iree_allocator_system(), &context);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_run_hal_testbench_context_ensure_runtime(&context));

  loom_run_hal_testbench_context_deinitialize(&context);
}

TEST_F(HalTestbenchActualTest,
       AddsAuthoredSanitizerRequirementsBeforeRuntimeInitialization) {
  static constexpr char kSource[] = R"(
kernel.def @entry() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch(%condition: i1) {
  kernel.assert %condition : i1
  kernel.return
}
)";
  loomc_module_t* module = ParsePublicModule(IREE_SV(kSource));

  loom_run_hal_testbench_context_t context = {};
  loom_run_hal_testbench_context_initialize(
      /*target_environment=*/nullptr, /*target_routes=*/nullptr,
      /*target_route_count=*/0, iree_allocator_system(), &context);
  IREE_ASSERT_OK(loom_run_hal_testbench_context_add_module_runtime_requirements(
      &context, module, /*sanitizer_options=*/nullptr));
  EXPECT_EQ(context.runtime_features,
            IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_FEEDBACK);

  loom_run_hal_testbench_context_deinitialize(&context);
  loomc_module_release(module);
}

TEST_F(HalTestbenchActualTest,
       RejectsNewSanitizerRequirementsAfterRuntimeInitialization) {
  static constexpr char kSource[] = R"(
kernel.def @entry() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch(%condition: i1) {
  kernel.assert %condition : i1
  kernel.return
}
)";
  loomc_module_t* module = ParsePublicModule(IREE_SV(kSource));

  loom_run_hal_testbench_context_t context = {};
  loom_run_hal_testbench_context_initialize(
      /*target_environment=*/nullptr, /*target_routes=*/nullptr,
      /*target_route_count=*/0, iree_allocator_system(), &context);
  iree_hal_queue_t dispatch_queue = {};
  IREE_ASSERT_OK(InitializeFakeHalContext(&context, &dispatch_queue));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_run_hal_testbench_context_add_module_runtime_requirements(
          &context, module, /*sanitizer_options=*/nullptr));

  loom_run_hal_testbench_context_deinitialize(&context);
  loomc_module_release(module);
}

TEST_F(HalTestbenchActualTest, ScalarInputsPackDispatchConstantWords) {
  loom_testbench_value_t inputs[] = {
      I32Value(0x12345678),
      I64Value(static_cast<int64_t>(0x1122334455667788ull)),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      loom_type_scalar(LOOM_SCALAR_TYPE_I64),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_values(
      inputs, input_types, &kIndex32Offset64TargetSnapshot,
      /*input_parameters=*/nullptr, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_byte_length, 12u);
  EXPECT_EQ(iree_unaligned_load_le_u32(options.constants), 0x12345678u);
  EXPECT_EQ(iree_unaligned_load_le_u32(options.constants + 4), 0x55667788u);
  EXPECT_EQ(iree_unaligned_load_le_u32(options.constants + 8), 0x11223344u);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest, ReflectionPlacesAlignedAndNarrowConstants) {
  loom_testbench_value_t inputs[] = {
      I32Value(0x12345678), I64Value(INT64_C(0x1122334455667788)),
      I32Value(INT8_MIN),   I32Value(INT16_MIN),
      I32Value(1),          RawU32Value(0x38),
      RawU32Value(0x3c),    RawU32Value(0x3c00),
      RawU32Value(0x3f80),  I32Value(127),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      loom_type_scalar(LOOM_SCALAR_TYPE_I64),
      loom_type_scalar(LOOM_SCALAR_TYPE_I8),
      loom_type_scalar(LOOM_SCALAR_TYPE_I16),
      loom_type_scalar(LOOM_SCALAR_TYPE_I1),
      loom_type_scalar(LOOM_SCALAR_TYPE_F8E4M3),
      loom_type_scalar(LOOM_SCALAR_TYPE_F8E5M2),
      loom_type_scalar(LOOM_SCALAR_TYPE_F16),
      loom_type_scalar(LOOM_SCALAR_TYPE_BF16),
      loom_type_scalar(LOOM_SCALAR_TYPE_I8),
  };
  const uint16_t offsets[] = {0, 8, 16, 18, 20, 21, 22, 24, 26, 28};
  const uint8_t sizes[] = {4, 8, 1, 2, 1, 1, 1, 2, 2, 1};
  iree_hal_executable_function_parameter_t parameters[IREE_ARRAYSIZE(inputs)] =
      {};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(inputs); ++i) {
    parameters[i].type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT;
    parameters[i].size = sizes[i];
    parameters[i].offset = offsets[i];
  }
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  // Padding must be initialized even when the caller reuses its argument
  // storage.
  memset(options.constants, 0xcd, sizeof(options.constants));
  loom_run_hal_binding_list_t bindings = {};
  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_values(
      inputs, input_types, nullptr, parameters, IREE_ARRAYSIZE(inputs),
      &options, iree_allocator_system(), &bindings));

  EXPECT_EQ(options.constant_byte_length, 29u);
  EXPECT_EQ(iree_unaligned_load_le_u32(options.constants), 0x12345678u);
  EXPECT_EQ(iree_unaligned_load_le_u32(options.constants + 4), 0u);
  EXPECT_EQ(iree_unaligned_load_le_u64(options.constants + 8),
            UINT64_C(0x1122334455667788));
  EXPECT_EQ(options.constants[16], 0x80u);
  EXPECT_EQ(options.constants[17], 0u);
  EXPECT_EQ(iree_unaligned_load_le_u16(options.constants + 18), 0x8000u);
  EXPECT_EQ(options.constants[20], 1u);
  EXPECT_EQ(options.constants[21], 0x38u);
  EXPECT_EQ(options.constants[22], 0x3cu);
  EXPECT_EQ(options.constants[23], 0u);
  EXPECT_EQ(iree_unaligned_load_le_u16(options.constants + 24), 0x3c00u);
  EXPECT_EQ(iree_unaligned_load_le_u16(options.constants + 26), 0x3f80u);
  EXPECT_EQ(options.constants[28], 127u);
  EXPECT_EQ(options.constants[29], 0xcdu);
  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest, F64InputsPackDispatchConstantWords) {
  loom_testbench_value_t inputs[] = {
      F64Value(1.0),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_F64),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_values(
      inputs, input_types, &kIndex32Offset64TargetSnapshot,
      /*input_parameters=*/nullptr, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_byte_length, 8u);
  EXPECT_EQ(iree_unaligned_load_le_u32(options.constants), 0x00000000u);
  EXPECT_EQ(iree_unaligned_load_le_u32(options.constants + 4), 0x3ff00000u);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest, IndexInputUsesSelected32BitTargetCarrier) {
  loom_testbench_value_t inputs[] = {
      I64Value(3584),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_values(
      inputs, input_types, &kIndex32Offset64TargetSnapshot,
      /*input_parameters=*/nullptr, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_byte_length, 4u);
  EXPECT_EQ(iree_unaligned_load_le_u32(options.constants), 3584u);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest, IndexInputUsesSelected64BitTargetCarrier) {
  loom_testbench_value_t inputs[] = {
      I64Value(static_cast<int64_t>(0x1122334455667788ull)),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_values(
      inputs, input_types, &kIndex64Offset32TargetSnapshot,
      /*input_parameters=*/nullptr, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_byte_length, 8u);
  EXPECT_EQ(iree_unaligned_load_le_u32(options.constants), 0x55667788u);
  EXPECT_EQ(iree_unaligned_load_le_u32(options.constants + 4), 0x11223344u);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest,
       RejectsAddressInputWithoutSelectedTargetCarrier) {
  loom_testbench_value_t inputs[] = {
      I64Value(1),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_run_hal_testbench_invocation_inputs_from_values(
          inputs, input_types, /*target_snapshot=*/nullptr,
          /*input_parameters=*/nullptr, IREE_ARRAYSIZE(inputs), &options,
          iree_allocator_system(), &bindings));
}

TEST_F(HalTestbenchActualTest, IndexInputUsesSignedReflectedFourByteRange) {
  loom_testbench_value_t inputs[] = {
      I64Value(INT32_MIN),
      I64Value(INT32_MAX),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
  };
  iree_hal_executable_function_parameter_t input_parameters[] = {
      {
          .type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
          .flags = {},
          .size = 4,
          .offset = 0,
      },
      {
          .type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
          .flags = {},
          .size = 4,
          .offset = 4,
      },
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_values(
      inputs, input_types, &kIndex32Offset64TargetSnapshot, input_parameters,
      IREE_ARRAYSIZE(inputs), &options, iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_byte_length, 8u);
  EXPECT_EQ(iree_unaligned_load_le_u32(options.constants), 0x80000000u);
  EXPECT_EQ(iree_unaligned_load_le_u32(options.constants + 4), 0x7fffffffu);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest,
       RejectsIndexOutsideSignedReflectedFourByteRange) {
  loom_testbench_value_t inputs[] = {
      I64Value(INT64_C(2147483648)),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
  };
  iree_hal_executable_function_parameter_t input_parameters[] = {{
      .type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
      .flags = {},
      .size = 4,
      .offset = 0,
  }};
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_run_hal_testbench_invocation_inputs_from_values(
          inputs, input_types, &kIndex32Offset64TargetSnapshot,
          input_parameters, IREE_ARRAYSIZE(inputs), &options,
          iree_allocator_system(), &bindings));
}

TEST_F(HalTestbenchActualTest, MixedInputsUseReflectedWidthsAndOffsets) {
  loom_testbench_value_t inputs[] = {
      I64Value(3584),
      F32Value(4.0f),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_F32),
  };
  iree_hal_executable_function_parameter_t input_parameters[] = {
      {
          .type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
          .flags = {},
          .size = 8,
          .offset = 0,
      },
      {
          .type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
          .flags = {},
          .size = 4,
          .offset = 8,
      },
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_values(
      inputs, input_types, &kIndex32Offset64TargetSnapshot, input_parameters,
      IREE_ARRAYSIZE(inputs), &options, iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_byte_length, 12u);
  EXPECT_EQ(iree_unaligned_load_le_u32(options.constants), 3584u);
  EXPECT_EQ(iree_unaligned_load_le_u32(options.constants + 4), 0u);
  EXPECT_EQ(iree_unaligned_load_le_u32(options.constants + 8), 0x40800000u);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest, OffsetInputUsesSelected64BitTargetCarrier) {
  loom_testbench_value_t inputs[] = {
      I64Value(static_cast<int64_t>(0x1122334455667788ull)),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_values(
      inputs, input_types, &kIndex32Offset64TargetSnapshot,
      /*input_parameters=*/nullptr, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_byte_length, 8u);
  EXPECT_EQ(iree_unaligned_load_le_u32(options.constants), 0x55667788u);
  EXPECT_EQ(iree_unaligned_load_le_u32(options.constants + 4), 0x11223344u);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest, OffsetInputUsesSelected32BitTargetCarrier) {
  loom_testbench_value_t inputs[] = {
      I64Value(INT64_C(4294967295)),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_values(
      inputs, input_types, &kIndex64Offset32TargetSnapshot,
      /*input_parameters=*/nullptr, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_byte_length, 4u);
  EXPECT_EQ(iree_unaligned_load_le_u32(options.constants), UINT32_MAX);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest, OffsetInputUsesReflectedFourByteWidth) {
  loom_testbench_value_t inputs[] = {
      I64Value(INT64_C(4294967295)),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
  };
  iree_hal_executable_function_parameter_t input_parameters[] = {{
      .type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
      .flags = {},
      .size = 4,
      .offset = 0,
  }};
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_values(
      inputs, input_types, &kIndex32Offset64TargetSnapshot, input_parameters,
      IREE_ARRAYSIZE(inputs), &options, iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_byte_length, 4u);
  EXPECT_EQ(iree_unaligned_load_le_u32(options.constants), UINT32_MAX);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest,
       RejectsOffsetOutsideUnsignedReflectedFourByteRange) {
  loom_testbench_value_t inputs[] = {
      I64Value(INT64_C(4294967296)),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
  };
  iree_hal_executable_function_parameter_t input_parameters[] = {{
      .type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
      .flags = {},
      .size = 4,
      .offset = 0,
  }};
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_run_hal_testbench_invocation_inputs_from_values(
          inputs, input_types, &kIndex32Offset64TargetSnapshot,
          input_parameters, IREE_ARRAYSIZE(inputs), &options,
          iree_allocator_system(), &bindings));
}

TEST_F(HalTestbenchActualTest, RejectsNegativeOffsetInput) {
  loom_testbench_value_t inputs[] = {
      I64Value(-1),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_run_hal_testbench_invocation_inputs_from_values(
          inputs, input_types, &kIndex32Offset64TargetSnapshot,
          /*input_parameters=*/nullptr, IREE_ARRAYSIZE(inputs), &options,
          iree_allocator_system(), &bindings));
}

TEST_F(HalTestbenchActualTest, RejectsNestedKernelLaunchSchedules) {
  static constexpr char kSource[] = R"(
kernel.decl @step() launch(%output: buffer)

check.case @nested_launch_schedule {
  %output = check.generate.fill value(0) : tensor<1xi32>
  kernel.launch.serial {
    kernel.launch.concurrent {
      kernel.launch @step(%output) : (tensor<1xi32>)
    }
  }
  check.return
}
)";
  loomc_module_t* module = nullptr;
  loom_testbench_module_plan_t module_plan = {};
  ParseAndPlan(IREE_SV(kSource), &module, /*out_native_module=*/nullptr,
               &module_plan);
  ASSERT_EQ(module_plan.case_count, 1u);

  iree_host_size_t kernel_launch_count = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        loom_run_hal_testbench_count_kernel_launches(
                            &module_plan.cases[0], &kernel_launch_count));

  loomc_module_release(module);
}

TEST_F(HalTestbenchActualTest, RejectsCaseWithoutKernelLaunch) {
  static constexpr char kSource[] = R"(
check.case @host_only {
  %value = check.literal value(1) : i32
  check.expect.equal actual(%value) expected(%value) : i32
  check.return
}
)";
  loomc_module_t* module = nullptr;
  loom_testbench_module_plan_t module_plan = {};
  ParseAndPlan(IREE_SV(kSource), &module, /*out_native_module=*/nullptr,
               &module_plan);
  ASSERT_EQ(module_plan.case_count, 1u);

  const loom_testbench_invocation_plan_t* kernel_launch = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        loom_run_hal_testbench_select_kernel_launch(
                            &module_plan.cases[0], &kernel_launch));
  EXPECT_EQ(kernel_launch, nullptr);

  loomc_module_release(module);
}

TEST_F(HalTestbenchActualTest, EvaluatesCompiledLaunchConfigForExactWorkload) {
  static constexpr char kSource[] = R"(
kernel.def @dynamic(%workgroup_count: index) {
  %unit = index.constant 1 : index
  %size = index.constant 64 : index
  kernel.launch.config workgroups(%workgroup_count, %unit, %unit) workgroup_size(%size, %unit, %unit) : index
} launch(%workgroup_count: index) {
  kernel.return
}

check.case @dynamic_case {
  %workgroup_count = check.param.choice values([1, 4]) name("workgroup_count") : index
  kernel.launch @dynamic[%workgroup_count](%workgroup_count) : [index](index)
  check.return
}
)";
  static constexpr char kLaunchConfigSource[] = R"(
func.def public pure @device_dynamic(%workgroup_count: index) -> (index, index, index, index, index, index, index, index, index, index, index) {
  %one = index.constant 1 : index
  %size = index.constant 64 : index
  %subgroup_size = index.constant 32 : index
  %storage = index.constant 0 : index
  func.return %workgroup_count, %one, %one, %size, %one, %one, %one, %one, %one, %subgroup_size, %storage : index, index, index, index, index, index, index, index, index, index, index
}
)";
  loomc_module_t* module = nullptr;
  const loom_module_t* native_module = nullptr;
  loom_testbench_module_plan_t module_plan = {};
  ParseAndPlan(IREE_SV(kSource), &module, &native_module, &module_plan);
  ASSERT_EQ(module_plan.case_count, 1u);
  const loom_testbench_case_plan_t* case_plan = &module_plan.cases[0];
  const loom_testbench_invocation_plan_t* kernel_launch = nullptr;
  IREE_ASSERT_OK(
      loom_run_hal_testbench_select_kernel_launch(case_plan, &kernel_launch));

  uint64_t workload_argument_bits[1] = {};
  loom_run_hal_testbench_context_t context = {.host_allocator =
                                                  iree_allocator_system()};
  loom_run_hal_testbench_actual_provider_t provider = {};
  provider.context = &context;
  provider.native_module = native_module;
  provider.kernel_launch = kernel_launch;
  provider.launch_config_program =
      LoadLaunchConfigProgram(IREE_SV(kLaunchConfigSource));
  provider.launch_config_function = loomc_launch_config_function_invalid();
  IREE_ASSERT_OK(
      iree_status_from_loomc(loomc_launch_config_program_lookup_function(
          provider.launch_config_program,
          loomc_make_cstring_view("device_dynamic"),
          &provider.launch_config_function)));
  provider.workload_argument_bits = workload_argument_bits;
  provider.target_snapshot = &kFakeTargetSnapshot;
  provider.invocation_options.function_name = IREE_SV("device_dynamic");

  loom_testbench_value_materializer_options_t materializer_options = {};
  loom_testbench_value_materializer_options_initialize(&materializer_options);
  loom_testbench_value_table_t value_table = {};
  IREE_ASSERT_OK(loom_testbench_value_table_initialize_case(
      native_module, case_plan, iree_allocator_system(), &value_table));
  for (iree_host_size_t sample_ordinal = 0; sample_ordinal < 2;
       ++sample_ordinal) {
    loom_testbench_value_table_reset(&value_table);
    IREE_ASSERT_OK(loom_testbench_materialize_case_sample(
        &materializer_options, case_plan, sample_ordinal, &value_table));
    loom_run_hal_invocation_options_t invocation_options = {};
    loom_run_hal_binding_list_t bindings = {};
    IREE_ASSERT_OK(loom_run_hal_testbench_materialize_invocation_from_table(
        &value_table, &provider, iree_allocator_system(), &invocation_options,
        &bindings));

    const uint32_t expected_workgroup_count = sample_ordinal == 0 ? 1 : 4;
    EXPECT_EQ(provider.workload_argument_bits[0], expected_workgroup_count);
    EXPECT_EQ(invocation_options.workgroup_count[0], expected_workgroup_count);
    EXPECT_EQ(provider.resolved_launch_config.workgroup_count.x,
              expected_workgroup_count);
    EXPECT_EQ(provider.resolved_launch_config.workgroup_size.x, 64u);
    EXPECT_EQ(provider.resolved_launch_config.subgroup_size, 32u);
    EXPECT_EQ(provider.resolved_launch_config.workgroup_storage_bytes, 0u);

    const loom_testbench_value_t* workload = nullptr;
    const loom_testbench_value_t* input = nullptr;
    IREE_ASSERT_OK(loom_testbench_value_table_lookup_borrow(
        &value_table, kernel_launch->workload_value_ids[0], &workload));
    IREE_ASSERT_OK(loom_testbench_value_table_lookup_borrow(
        &value_table, kernel_launch->input_value_ids[0], &input));
    loom_run_hal_invocation_options_t prepared_options = {};
    loom_run_hal_binding_list_t prepared_bindings = {};
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_FAILED_PRECONDITION,
        loom_run_hal_testbench_actual_provider_materialize_invocation(
            &provider, /*workload_count=*/1, workload, /*input_count=*/1, input,
            &prepared_options, &prepared_bindings));
    provider.prepared_candidate_initialized = true;
    IREE_ASSERT_OK(
        loom_run_hal_testbench_actual_provider_materialize_invocation(
            &provider, /*workload_count=*/1, workload, /*input_count=*/1, input,
            &prepared_options, &prepared_bindings));
    provider.prepared_candidate_initialized = false;
    EXPECT_EQ(prepared_options.workgroup_count[0], expected_workgroup_count);
    EXPECT_EQ(prepared_options.constant_byte_length, 4u);
    EXPECT_EQ(iree_unaligned_load_le_u32(prepared_options.constants),
              expected_workgroup_count);
    EXPECT_EQ(prepared_bindings.count, 0u);
    loom_run_hal_binding_list_deinitialize(&prepared_bindings);
    loom_run_hal_binding_list_deinitialize(&bindings);
  }

  loom_testbench_value_table_deinitialize(&value_table);
  loomc_launch_config_program_release(provider.launch_config_program);
  loomc_module_release(module);
}

}  // namespace
}  // namespace loom
