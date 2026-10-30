// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

class CommandBufferBasicTest : public CtsTestBase<> {};

TEST_P(CommandBufferBasicTest, Create) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(CreateCommandBuffer(IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                                     IREE_HAL_COMMAND_CATEGORY_DISPATCH,
                                     /*binding_capacity=*/0, &command_buffer));

  EXPECT_TRUE((iree_hal_command_buffer_allowed_categories(command_buffer) &
               IREE_HAL_COMMAND_CATEGORY_DISPATCH) ==
              IREE_HAL_COMMAND_CATEGORY_DISPATCH);

  iree_hal_command_buffer_release(command_buffer);
}

TEST_P(CommandBufferBasicTest, CreateFailurePreservesOutput) {
  iree_hal_queue_t* queue =
      QueueForCommandCategories(IREE_HAL_COMMAND_CATEGORY_DISPATCH);
  ASSERT_NE(nullptr, queue);
  Ref<iree_hal_command_buffer_t> original;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      iree_hal_queue_family(queue), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*binding_capacity=*/0,
      original.out()));
  iree_hal_command_buffer_t* output = original.get();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_command_buffer_create(
          iree_hal_queue_family(queue), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
          ~IREE_HAL_COMMAND_CATEGORY_ANY, /*binding_capacity=*/0, &output));
  EXPECT_EQ(original.get(), output);
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(original));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(original));
  IREE_ASSERT_OK(SubmitCommandBufferAndWait(original));
}

TEST_P(CommandBufferBasicTest, BeginEnd) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(CreateCommandBuffer(IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                                     IREE_HAL_COMMAND_CATEGORY_DISPATCH,
                                     /*binding_capacity=*/0, &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_hal_command_buffer_release(command_buffer);
}

TEST_P(CommandBufferBasicTest, SubmitEmpty) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(CreateCommandBuffer(IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                                     IREE_HAL_COMMAND_CATEGORY_DISPATCH,
                                     /*binding_capacity=*/0, &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer));

  iree_hal_command_buffer_release(command_buffer);
}

TEST_P(CommandBufferBasicTest, TimestampCapabilitiesApplyWithoutValidation) {
  auto* queue = QueueForCommandCategories(IREE_HAL_COMMAND_CATEGORY_TRANSFER);
  ASSERT_NE(nullptr, queue);
  const auto* family = iree_hal_queue_family(queue);
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(16, buffer.out()));
  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      family, IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, 1, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_EXPECT_STATUS_IS(
      (family->spec->flags & IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_COMMAND_TIMESTAMPS)
          ? IREE_STATUS_OK
          : IREE_STATUS_UNIMPLEMENTED,
      iree_hal_command_buffer_timestamp(
          command_buffer, 0, 0, iree_hal_make_buffer_ref(buffer, 0, 8), 0));
  IREE_EXPECT_STATUS_IS(
      (family->spec->flags &
       IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_INDIRECT_COMMAND_TIMESTAMPS)
          ? IREE_STATUS_OK
          : IREE_STATUS_UNIMPLEMENTED,
      iree_hal_command_buffer_timestamp(
          command_buffer, 0, 0, iree_hal_make_indirect_buffer_ref(0, 8, 8), 0));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
}

CTS_REGISTER_TEST_SUITE(CommandBufferBasicTest);

}  // namespace iree::hal::cts
