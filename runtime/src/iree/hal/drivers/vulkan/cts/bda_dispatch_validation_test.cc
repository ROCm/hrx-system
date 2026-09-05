// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vulkan BDA-specific validation coverage. These cases assert failures at the
// HAL boundary before malformed pointer tables can reach the device.

#include <cstdint>
#include <cstring>

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/vulkan/cts/bda_spirv_test_spv.h"

namespace iree::hal::cts {

using iree::testing::status::StatusIs;

static constexpr uint32_t kRequiredBindingAlignment = 2;
static constexpr uint64_t kRequiredBindingLength = 17;

constexpr char kBdaSpirvNoopBindingRequirements[] =
    "bda_noop_binding_requirements.spv";
constexpr char kBdaSpirvNoopBindings2ConstantLength8[] =
    "bda_noop_bindings_2_constant_length_8.spv";

static iree_const_byte_span_t BdaDispatchValidationSpirvFixture(
    const char* file_name) {
  const iree_file_toc_t* toc = iree_hal_vulkan_cts_bda_spirv_test_spv_create();
  for (iree_host_size_t i = 0;
       i < iree_hal_vulkan_cts_bda_spirv_test_spv_size(); ++i) {
    if (std::strcmp(toc[i].name, file_name) == 0) {
      return iree_make_const_byte_span(toc[i].data, toc[i].size);
    }
  }
  ADD_FAILURE() << "BDA SPIR-V fixture not found: " << file_name;
  return iree_const_byte_span_empty();
}

class BdaDispatchValidationTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) return;

    dispatch_queue_ =
        QueueForCommandCategories(IREE_HAL_COMMAND_CATEGORY_DISPATCH);
    if (!dispatch_queue_) {
      GTEST_SKIP() << "device has no dispatch-capable queue";
    }
    const iree_hal_executable_target_selection_result_t target_result =
        SelectExecutableTarget(IREE_SV("spirv"), IREE_SV("vulkan1.3+bda"));
    ASSERT_EQ(IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED,
              target_result.outcome);
    executable_target_ = target_result.target;

    IREE_ASSERT_OK(
        PrepareBdaExecutable(BdaDispatchValidationSpirvFixture(
                                 kBdaSpirvNoopBindings2ConstantLength8),
                             &executable_));
    IREE_ASSERT_OK(PrepareBdaExecutable(
        BdaDispatchValidationSpirvFixture(kBdaSpirvNoopBindingRequirements),
        &requirement_executable_));
  }

  void TearDown() override {
    iree_hal_executable_release(requirement_executable_);
    requirement_executable_ = nullptr;
    iree_hal_executable_release(executable_);
    executable_ = nullptr;
    executable_target_ = nullptr;
    CtsTestBase::TearDown();
  }

  iree_const_byte_span_t constants() const {
    return iree_make_const_byte_span(constant_data_, sizeof(constant_data_));
  }

  iree_status_t PrepareBdaExecutable(iree_const_byte_span_t executable_data,
                                     iree_hal_executable_t** out_executable) {
    return LoadExecutable(executable_target_,
                          IREE_HAL_EXECUTABLE_LOAD_FLAG_DISABLE_VERIFICATION,
                          executable_data, out_executable);
  }

  iree_status_t CreateInputOutputBuffers(
      iree_hal_buffer_t** out_input_buffer,
      iree_hal_buffer_t** out_output_buffer) {
    *out_input_buffer = nullptr;
    *out_output_buffer = nullptr;
    const uint32_t input_data[4] = {1, 2, 3, 4};
    iree_hal_buffer_t* input_buffer = nullptr;
    iree_hal_buffer_t* output_buffer = nullptr;
    iree_status_t status = CreateDeviceBufferWithData(
        input_data, sizeof(input_data), &input_buffer);
    if (iree_status_is_ok(status)) {
      status = CreateZeroedDeviceBuffer(sizeof(input_data), &output_buffer);
    }
    if (iree_status_is_ok(status)) {
      *out_input_buffer = input_buffer;
      *out_output_buffer = output_buffer;
    } else {
      iree_hal_buffer_release(output_buffer);
      iree_hal_buffer_release(input_buffer);
    }
    return status;
  }

  static constexpr uint32_t constant_data_[2] = {3, 10};

  iree_hal_queue_t* dispatch_queue_ = nullptr;
  const iree_hal_executable_target_t* executable_target_ = nullptr;
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

  EXPECT_THAT(Status(iree_hal_queue_dispatch(
                  dispatch_queue_, iree_hal_semaphore_list_empty(),
                  iree_hal_semaphore_list_empty(), executable_,
                  iree_hal_executable_function_from_index(0),
                  iree_hal_make_static_dispatch_config(1, 1, 1), constants(),
                  bindings, IREE_HAL_DISPATCH_FLAG_NONE)),
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
  IREE_ASSERT_OK(CreateCommandBuffer(IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                                     IREE_HAL_COMMAND_CATEGORY_DISPATCH,
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

  EXPECT_THAT(Status(iree_hal_queue_dispatch(
                  dispatch_queue_, iree_hal_semaphore_list_empty(),
                  iree_hal_semaphore_list_empty(), executable_,
                  iree_hal_executable_function_from_index(0),
                  iree_hal_make_static_dispatch_config(1, 1, 1), constants(),
                  bindings, IREE_HAL_DISPATCH_FLAG_NONE)),
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
  IREE_ASSERT_OK(CreateCommandBuffer(IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                                     IREE_HAL_COMMAND_CATEGORY_DISPATCH,
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

  iree_hal_queue_t* queue = QueueForCommandBuffer(command_buffer);
  ASSERT_NE(nullptr, queue);
  EXPECT_THAT(Status(iree_hal_queue_execute(
                  queue, iree_hal_semaphore_list_empty(),
                  iree_hal_semaphore_list_empty(), command_buffer,
                  iree_hal_buffer_binding_table_empty(),
                  IREE_HAL_QUEUE_EXECUTE_FLAG_NONE)),
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
      Status(iree_hal_queue_dispatch(
          dispatch_queue_, iree_hal_semaphore_list_empty(),
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
  IREE_ASSERT_OK(CreateCommandBuffer(IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                                     IREE_HAL_COMMAND_CATEGORY_DISPATCH,
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

  iree_hal_queue_t* queue = QueueForCommandBuffer(command_buffer);
  ASSERT_NE(nullptr, queue);
  EXPECT_THAT(Status(iree_hal_queue_execute(
                  queue, iree_hal_semaphore_list_empty(),
                  iree_hal_semaphore_list_empty(), command_buffer,
                  iree_hal_buffer_binding_table_empty(),
                  IREE_HAL_QUEUE_EXECUTE_FLAG_NONE)),
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
      Status(iree_hal_queue_dispatch(
          dispatch_queue_, iree_hal_semaphore_list_empty(),
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
  IREE_ASSERT_OK(CreateCommandBuffer(IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                                     IREE_HAL_COMMAND_CATEGORY_DISPATCH,
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

  iree_hal_queue_t* queue = QueueForCommandBuffer(command_buffer);
  ASSERT_NE(nullptr, queue);
  EXPECT_THAT(Status(iree_hal_queue_execute(
                  queue, iree_hal_semaphore_list_empty(),
                  iree_hal_semaphore_list_empty(), command_buffer,
                  iree_hal_buffer_binding_table_empty(),
                  IREE_HAL_QUEUE_EXECUTE_FLAG_NONE)),
              StatusIs(StatusCode::kInvalidArgument));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
}

CTS_REGISTER_TEST_SUITE(BdaDispatchValidationTest);

}  // namespace iree::hal::cts
