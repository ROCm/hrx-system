// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vulkan BDA-specific validation coverage. These cases assert failures at the
// HAL boundary before malformed pointer tables can reach the device.

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

using iree::testing::status::StatusIs;

static constexpr uint32_t kRequiredBindingAlignment = 2;
static constexpr uint64_t kRequiredBindingLength = 17;

// Descriptor-free BDA-environment no-op shader. The shader does not need to
// consume bindings because these tests fail while publishing the host-validated
// BDA table.
static const uint32_t kBdaNoopSpirv[] = {
    0x07230203u,
    0x00010600u,
    0u,
    5u,
    0u,
    // Declares OpCapability Shader.
    0x00020011u,
    1u,
    // Declares OpCapability PhysicalStorageBufferAddresses.
    0x00020011u,
    5347u,
    // Declares OpMemoryModel PhysicalStorageBuffer64 GLSL450.
    0x0003000eu,
    5348u,
    1u,
    // Declares OpEntryPoint GLCompute %main "main".
    0x0005000fu,
    5u,
    3u,
    0x6e69616du,
    0u,
    // Declares OpExecutionMode %main LocalSize 1 1 1.
    0x00060010u,
    3u,
    17u,
    1u,
    1u,
    1u,
    // Declares OpTypeVoid %void.
    0x00020013u,
    1u,
    // Declares OpTypeFunction %fn %void.
    0x00030021u,
    2u,
    1u,
    // Defines %main as an empty compute function.
    0x00050036u,
    1u,
    3u,
    0u,
    2u,
    0x000200f8u,
    4u,
    0x000100fdu,
    0x00010038u,
};

static void AppendOpModuleProcessed(std::vector<uint32_t>* module,
                                    const char* text) {
  const size_t byte_length = std::strlen(text) + 1;
  const size_t string_word_count =
      (byte_length + sizeof(uint32_t) - 1) / sizeof(uint32_t);
  std::vector<uint32_t> instruction(1 + string_word_count, 0);
  instruction[0] = ((uint32_t)instruction.size() << 16) | 330u;
  std::memcpy(&instruction[1], text, byte_length);
  module->insert(module->begin() + 12, instruction.begin(), instruction.end());
}

static std::vector<uint32_t> MakeBdaExecutableWords(
    std::initializer_list<const char*> metadata_strings) {
  std::vector<uint32_t> words(kBdaNoopSpirv,
                              kBdaNoopSpirv + IREE_ARRAYSIZE(kBdaNoopSpirv));
  for (const char* metadata_string : metadata_strings) {
    AppendOpModuleProcessed(&words, metadata_string);
  }
  return words;
}

class BdaDispatchValidationTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) return;

    IREE_ASSERT_OK(iree_hal_executable_cache_create(
        device_, iree_make_cstring_view("default"), &executable_cache_));

    std::vector<uint32_t> executable_words = MakeBdaExecutableWords({
        "iree.vulkan.bda.v1.constant_length=8",
        "iree.vulkan.bda.v1.bindings=2",
        "iree.vulkan.bda.v1",
    });
    IREE_ASSERT_OK(PrepareBdaExecutable(executable_words, &executable_));

    std::vector<uint32_t> requirement_executable_words =
        MakeBdaExecutableWords({
            "iree.vulkan.bda.v1.binding.1=2,0",
            "iree.vulkan.bda.v1.binding.0=1,17",
            "iree.vulkan.bda.v1.bindings=2",
            "iree.vulkan.bda.v1",
        });
    IREE_ASSERT_OK(PrepareBdaExecutable(requirement_executable_words,
                                        &requirement_executable_));
  }

  void TearDown() override {
    iree_hal_executable_release(requirement_executable_);
    requirement_executable_ = nullptr;
    iree_hal_executable_release(executable_);
    executable_ = nullptr;
    iree_hal_executable_cache_release(executable_cache_);
    executable_cache_ = nullptr;
    CtsTestBase::TearDown();
  }

  iree_const_byte_span_t constants() const {
    return iree_make_const_byte_span(constant_data_, sizeof(constant_data_));
  }

  iree_status_t PrepareBdaExecutable(
      const std::vector<uint32_t>& executable_words,
      iree_hal_executable_t** out_executable) {
    iree_hal_executable_params_t executable_params;
    iree_hal_executable_params_initialize(&executable_params);
    executable_params.caching_mode =
        IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA |
        IREE_HAL_EXECUTABLE_CACHING_MODE_DISABLE_VERIFICATION;
    executable_params.executable_format =
        iree_make_cstring_view("vulkan-spirv-bda");
    executable_params.executable_data = iree_make_const_byte_span(
        executable_words.data(), executable_words.size() * sizeof(uint32_t));
    return iree_hal_executable_cache_prepare_executable(
        executable_cache_, &executable_params, out_executable);
  }

  iree_status_t CreateInputOutputBuffers(
      iree_hal_buffer_t** out_input_buffer,
      iree_hal_buffer_t** out_output_buffer) {
    *out_input_buffer = nullptr;
    *out_output_buffer = nullptr;
    const uint32_t input_data[4] = {1, 2, 3, 4};
    IREE_RETURN_IF_ERROR(CreateDeviceBufferWithData(
        input_data, sizeof(input_data), out_input_buffer));
    iree_status_t status =
        CreateZeroedDeviceBuffer(sizeof(input_data), out_output_buffer);
    if (!iree_status_is_ok(status)) {
      iree_hal_buffer_release(*out_input_buffer);
      *out_input_buffer = nullptr;
    }
    return status;
  }

  static constexpr uint32_t constant_data_[2] = {3, 10};

  iree_hal_executable_cache_t* executable_cache_ = nullptr;
  iree_hal_executable_t* executable_ = nullptr;
  iree_hal_executable_t* requirement_executable_ = nullptr;
};

TEST_P(BdaDispatchValidationTest, QueueDispatchRejectsBindingCountMismatch) {
  iree_hal_buffer_t* input_buffer = nullptr;
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(CreateInputOutputBuffers(&input_buffer, &output_buffer));

  iree_hal_buffer_ref_t binding_refs[1] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  EXPECT_THAT(
      Status(iree_hal_device_queue_dispatch(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), executable_,
          iree_hal_executable_function_from_index(0),
          iree_hal_make_static_dispatch_config(1, 1, 1), constants(), bindings,
          IREE_HAL_DISPATCH_FLAG_NONE)),
      StatusIs(StatusCode::kInvalidArgument));

  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
}

TEST_P(BdaDispatchValidationTest,
       CommandBufferDispatchRejectsBindingCountMismatch) {
  iree_hal_buffer_t* input_buffer = nullptr;
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(CreateInputOutputBuffers(&input_buffer, &output_buffer));

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  iree_hal_buffer_ref_t binding_refs[1] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  EXPECT_THAT(Status(iree_hal_command_buffer_dispatch(
                  command_buffer, executable_,
                  iree_hal_executable_function_from_index(0),
                  iree_hal_make_static_dispatch_config(1, 1, 1), constants(),
                  bindings, IREE_HAL_DISPATCH_FLAG_NONE)),
              StatusIs(StatusCode::kInvalidArgument));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
}

TEST_P(BdaDispatchValidationTest, QueueDispatchRejectsEmptyBindingRange) {
  iree_hal_buffer_t* input_buffer = nullptr;
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(CreateInputOutputBuffers(&input_buffer, &output_buffer));

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_buffer_ref(output_buffer, /*offset=*/0, /*length=*/0),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  EXPECT_THAT(
      Status(iree_hal_device_queue_dispatch(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), executable_,
          iree_hal_executable_function_from_index(0),
          iree_hal_make_static_dispatch_config(1, 1, 1), constants(), bindings,
          IREE_HAL_DISPATCH_FLAG_NONE)),
      StatusIs(StatusCode::kInvalidArgument));

  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
}

TEST_P(BdaDispatchValidationTest,
       CommandBufferExecuteRejectsEmptyBindingRange) {
  iree_hal_buffer_t* input_buffer = nullptr;
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(CreateInputOutputBuffers(&input_buffer, &output_buffer));

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_buffer_ref(output_buffer, /*offset=*/0, /*length=*/0),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants(), bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  EXPECT_THAT(
      Status(iree_hal_device_queue_execute(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), command_buffer,
          iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE)),
      StatusIs(StatusCode::kInvalidArgument));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
}

TEST_P(BdaDispatchValidationTest, QueueDispatchRejectsMinimumBindingLength) {
  iree_hal_buffer_t* input_buffer = nullptr;
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(kRequiredBindingLength - 1, &input_buffer));
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(64, &output_buffer));

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_buffer_ref(output_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(output_buffer)),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  EXPECT_THAT(
      Status(iree_hal_device_queue_dispatch(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), requirement_executable_,
          iree_hal_executable_function_from_index(0),
          iree_hal_make_static_dispatch_config(1, 1, 1),
          iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE)),
      StatusIs(StatusCode::kOutOfRange));

  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
}

TEST_P(BdaDispatchValidationTest,
       CommandBufferExecuteRejectsMinimumBindingLength) {
  iree_hal_buffer_t* input_buffer = nullptr;
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(kRequiredBindingLength - 1, &input_buffer));
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(64, &output_buffer));

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_buffer_ref(output_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(output_buffer)),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, requirement_executable_,
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  EXPECT_THAT(
      Status(iree_hal_device_queue_execute(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), command_buffer,
          iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE)),
      StatusIs(StatusCode::kOutOfRange));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
}

TEST_P(BdaDispatchValidationTest, QueueDispatchRejectsMinimumBindingAlignment) {
  iree_hal_buffer_t* input_buffer = nullptr;
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(kRequiredBindingLength, &input_buffer));
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(64, &output_buffer));

  const iree_device_size_t output_offset = 1;
  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_buffer_ref(
          output_buffer, output_offset,
          iree_hal_buffer_byte_length(output_buffer) - output_offset),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  EXPECT_THAT(
      Status(iree_hal_device_queue_dispatch(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), requirement_executable_,
          iree_hal_executable_function_from_index(0),
          iree_hal_make_static_dispatch_config(1, 1, 1),
          iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE)),
      StatusIs(StatusCode::kInvalidArgument));

  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
}

TEST_P(BdaDispatchValidationTest,
       CommandBufferExecuteRejectsMinimumBindingAlignment) {
  iree_hal_buffer_t* input_buffer = nullptr;
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(kRequiredBindingLength, &input_buffer));
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(64, &output_buffer));

  const iree_device_size_t output_offset = 1;

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_buffer_ref(
          output_buffer, output_offset,
          iree_hal_buffer_byte_length(output_buffer) - output_offset),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, requirement_executable_,
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  EXPECT_THAT(
      Status(iree_hal_device_queue_execute(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), command_buffer,
          iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE)),
      StatusIs(StatusCode::kInvalidArgument));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
}

CTS_REGISTER_TEST_SUITE(BdaDispatchValidationTest);

}  // namespace iree::hal::cts
