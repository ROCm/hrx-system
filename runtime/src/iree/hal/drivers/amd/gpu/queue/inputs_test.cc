// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/inputs.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class InputsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const uint16_t types[] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS};
    const iree_hal_buffer_binding_layout_t layout = {
        sizeof(iree_hal_buffer_native_binding_t), 1, 0, types};
    IREE_ASSERT_OK(iree_hal_memory_contract_create(
        this, 3, &layout, iree_allocator_system(), &contract_));
    contract_->scopes[1].interfaces =
        1u << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
    contract_->scopes[1].bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS] = 0;
    static const iree_hal_buffer_vtable_t vtable = {
        .destroy = [](iree_hal_buffer_t*) {},
    };
    iree_hal_buffer_initialize(
        {}, &buffer_, 64, 0, 64, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_STORAGE, &vtable,
        &buffer_);
    buffer_.memory = {.contract = contract_, .bindings = &binding_};
    family_.memory = {.domain = this, .queue_scope_id = 1};
    command_buffer_.queue_family = &family_;
    command_buffer_.binding_capacity = 1;
    iree_hal_amd_gpu_command_inputs_initialize(
        &command_buffer_, 1, nullptr, requirements_, iree_allocator_system(),
        &inputs_);
  }

  void TearDown() override {
    iree_hal_amd_gpu_command_inputs_deinitialize(&inputs_);
    EXPECT_EQ(iree_atomic_ref_count_load(&buffer_.resource.ref_count), 1);
    iree_hal_buffer_release(&buffer_);
    iree_hal_memory_contract_release(contract_);
  }

  // Native allocation dependency publishes this address after capture.
  iree_hal_buffer_native_binding_t binding_ = {};
  // Prepared site guarantees owned by the dependency, independent of address.
  iree_hal_memory_contract_t* contract_ = nullptr;
  // Refcounted binding whose retained invocation ownership is observed.
  iree_hal_buffer_t buffer_ = {};
  // Canonical queue site in this test's sealed domain.
  iree_hal_queue_family_t family_ = {};
  // Recording identity and binding count consumed by the real input collector.
  iree_hal_command_buffer_t command_buffer_ = {};
  // Inline storage for the constant root and one external binding.
  iree_hal_amd_gpu_binding_requirement_t requirements_[2] = {};
  // Production deduplication, qualification, and deferred address resolution.
  iree_hal_amd_gpu_command_inputs_t inputs_ = {};
};

TEST_F(InputsTest, IndirectAtomicUsesShareOneQualifiedInput) {
  const auto add = iree_hal_amd_gpu_control_rmw({
      .operand = 1,
      .width = IREE_HAL_ATOMIC_WIDTH_32,
      .operation = IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
  });
  const auto store = iree_hal_amd_gpu_control_store({
      .value = 1,
      .width = IREE_HAL_ATOMIC_WIDTH_64,
      .target_error_mode = IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE,
  });
  for (uint32_t i = 0; i < 1000; ++i) {
    uint32_t add_value = 0, store_value = 0;
    IREE_ASSERT_OK(iree_hal_amd_gpu_command_inputs_add_atomic(
        &inputs_, iree_hal_make_indirect_buffer_ref(0, 4, 4), add, &add_value));
    IREE_ASSERT_OK(iree_hal_amd_gpu_command_inputs_add_atomic(
        &inputs_, iree_hal_make_indirect_buffer_ref(0, 16, 8), store,
        &store_value));
    EXPECT_EQ(add_value, 1u);
    EXPECT_EQ(store_value, add_value);
  }
  iree_hal_amd_gpu_command_inputs_seal(&inputs_);
  EXPECT_EQ(inputs_.count, 1u);
  EXPECT_EQ(requirements_[1].minimum_length, 24u);
  const iree_hal_buffer_binding_t binding = {&buffer_, 0, 32};
  iree_hal_amd_gpu_command_binding_t captured = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        iree_hal_amd_gpu_command_inputs_capture(
                            &inputs_, {1, &binding}, 0, &captured));
  EXPECT_EQ(captured.buffer, nullptr);
  contract_->scopes[1].atomic_operations =
      IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INCOMPATIBLE,
                        iree_hal_amd_gpu_command_inputs_capture(
                            &inputs_, {1, &binding}, 0, &captured));
  EXPECT_EQ(captured.buffer, nullptr);
  contract_->scopes[1].atomic_operations |= IREE_HAL_ATOMIC_OPERATION_FLAG_STORE
                                            << 8;
  IREE_ASSERT_OK(iree_hal_amd_gpu_command_inputs_capture(
      &inputs_, {1, &binding}, 0, &captured));
  EXPECT_EQ(iree_atomic_ref_count_load(&buffer_.resource.ref_count), 2);
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_amd_gpu_command_inputs_write(&inputs_, &captured, 0, &value));
  binding_.device_address = 0x1000;
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_command_inputs_write(&inputs_, &captured, 0, &value));
  EXPECT_EQ(value, 0x1000u);
  binding_.device_address = 0x1004;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INCOMPATIBLE,
      iree_hal_amd_gpu_command_inputs_write(&inputs_, &captured, 0, &value));
  binding_.device_address = 0x1002;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_gpu_command_inputs_write(&inputs_, &captured, 0, &value));
  iree_hal_buffer_release(captured.buffer);
}

TEST_F(InputsTest, DirectAtomicRequiresTheQueueSiteAtRecording) {
  contract_->scopes[2].atomic_operations = IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT;
  uint32_t value = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INCOMPATIBLE,
      iree_hal_amd_gpu_command_inputs_add_atomic(
          &inputs_, iree_hal_make_buffer_ref(&buffer_, 0, 4),
          iree_hal_amd_gpu_control_wait({
              .width = IREE_HAL_ATOMIC_WIDTH_32,
              .target_error_mode =
                  IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE,
          }),
          &value));
}

TEST_F(InputsTest, DirectAtomicDefersAddressResolutionUntilDependencies) {
  contract_->scopes[1].atomic_operations = IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT;
  uint32_t input_value = 0;
  IREE_ASSERT_OK(iree_hal_amd_gpu_command_inputs_add_atomic(
      &inputs_, iree_hal_make_buffer_ref(&buffer_, 4, 4),
      iree_hal_amd_gpu_control_wait({.width = IREE_HAL_ATOMIC_WIDTH_32}),
      &input_value));
  iree_hal_amd_gpu_command_inputs_seal(&inputs_);
  EXPECT_EQ(command_buffer_.binding_count, 0u);
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_amd_gpu_command_inputs_write(&inputs_, nullptr, 0, &value));
  binding_.device_address = 0x1002;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_gpu_command_inputs_write(&inputs_, nullptr, 0, &value));
  binding_.device_address = 0x1000;
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_command_inputs_write(&inputs_, nullptr, 0, &value));
  EXPECT_EQ(value, 0x1000u);
}

}  // namespace
