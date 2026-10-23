// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/bank.h"

#include "loom/serve/runtime/parameter/weights_test_base.h"

namespace loom_serve::testing {

class ParameterBankTest : public WeightsTest {
 protected:
  void TearDown() override {
    if (execution_) {
      IREE_EXPECT_OK(loom_serve_execution_drain(execution_));
    }
    for (auto*& command : commands_) {
      iree_hal_command_buffer_release(command);
      command = nullptr;
    }
    loom_serve_parameter_bank_retire(bank_, nullptr, nullptr);
    loom_serve_parameter_bank_retire(adapter_bank_, nullptr, nullptr);
    IREE_EXPECT_OK(loom_serve_parameter_bank_await_retirement(bank_));
    IREE_EXPECT_OK(loom_serve_parameter_bank_await_retirement(adapter_bank_));
    loom_serve_residency_free(bank_residency_);
    IREE_EXPECT_OK(loom_serve_parameter_bank_free(bank_));
    IREE_EXPECT_OK(loom_serve_parameter_bank_free(adapter_bank_));
    WeightsTest::TearDown();
  }

  iree_status_t Prepare(size_t index, const char* path) {
    return PrepareBank(bank_, index, path);
  }

  iree_status_t PrepareBank(loom_serve_parameter_bank_t* bank, size_t index,
                            const char* path) {
    return WeightsTest::Prepare(loom_serve_parameter_bank_weights(bank, index),
                                {path});
  }

  // Owned checkpoint roots, including partially prepared construction.
  loom_serve_parameter_bank_t* bank_ = nullptr;
  // Independently owned selected adapter domain, sharing no base allocations.
  loom_serve_parameter_bank_t* adapter_bank_ = nullptr;
  // Borrowed bank group, released before the bank itself.
  loom_serve_residency_t* bank_residency_ = nullptr;
};

TEST_P(ParameterBankTest, RecordingsRetainPreparedRootsAfterBankRetirement) {
  WriteCheckpoint(adapter_path_.path(), 100);
  IREE_ASSERT_OK(Compile(0, "domains"));
  IREE_ASSERT_OK(loom_serve_parameter_bank_allocate(
      device_owner_, loom_serve_jit_stage_program(compiled_[0]),
      host_allocator_, &bank_));
  ASSERT_EQ(loom_serve_parameter_bank_count(bank_), 2u);
  IREE_ASSERT_OK(Prepare(0, path_.path().c_str()));
  IREE_ASSERT_OK(Prepare(1, adapter_path_.path().c_str()));
  auto* group = loom_serve_parameter_bank_group(bank_);
  IREE_ASSERT_OK(loom_serve_residency_allocate(
      loom_serve_device_residency_cache(device_owner_), execution_, 1, &group,
      host_allocator_, &bank_residency_));
  IREE_ASSERT_OK(loom_serve_jit_stage_record(
      compiled_[0], iree_hal_queue_family(dispatch_),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, nullptr,
      loom_serve_parameter_bank_buffers(bank_), &commands_[0]));
  // A recorded consumer and its bank no longer borrow compiler reflection.
  loom_serve_jit_stage_free(compiled_[0]);
  compiled_[0] = nullptr;
  IREE_ASSERT_OK(loom_serve_residency_acquire_wait(bank_residency_));
  ASSERT_NO_FATAL_FAILURE(Run(0, 290));
  loom_serve_residency_release(bank_residency_);
  loom_serve_parameter_bank_retire(bank_, nullptr, nullptr);
  EXPECT_FALSE(loom_serve_parameter_bank_is_retired(bank_));
  // Retiring twice does not invalidate the independently retained recording.
  loom_serve_parameter_bank_retire(bank_, nullptr, nullptr);
  EXPECT_FALSE(loom_serve_parameter_bank_is_retired(bank_));
  iree_hal_command_buffer_release(commands_[0]);
  commands_[0] = nullptr;
  IREE_ASSERT_OK(loom_serve_parameter_bank_await_retirement(bank_));
  EXPECT_TRUE(loom_serve_parameter_bank_is_retired(bank_));
}

TEST_P(ParameterBankTest, FailedDomainPreparationRetiresAllOwnedRoots) {
  IREE_ASSERT_OK(Compile(0, "domains"));
  IREE_ASSERT_OK(loom_serve_parameter_bank_allocate(
      device_owner_, loom_serve_jit_stage_program(compiled_[0]),
      host_allocator_, &bank_));
  IREE_ASSERT_OK(Prepare(0, path_.path().c_str()));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        Prepare(1, adapter_path_.path().c_str()));
  loom_serve_parameter_bank_retire(bank_, nullptr, nullptr);
  IREE_ASSERT_OK(loom_serve_parameter_bank_await_retirement(bank_));
  EXPECT_TRUE(loom_serve_parameter_bank_is_retired(bank_));
  for (size_t i = 0; i < loom_serve_parameter_bank_count(bank_); ++i) {
    EXPECT_EQ(
        loom_serve_weights_activity(loom_serve_parameter_bank_weights(bank_, i))
            .activations,
        0u);
  }
}

TEST_P(ParameterBankTest, ChecksRetainedLayoutAfterCompilerProductRetires) {
  WriteCheckpoint(adapter_path_.path(), 100);
  IREE_ASSERT_OK(Compile(0, "domains"));
  const auto* original = loom_serve_jit_stage_program(compiled_[0]);
  IREE_ASSERT_OK(loom_serve_parameter_bank_allocate(device_owner_, original,
                                                    host_allocator_, &bank_));
  EXPECT_EQ(loom_cmd_program_parameter_at(original, 0).fixed_buffer_index, 0u);
  IREE_ASSERT_OK(Prepare(0, path_.path().c_str()));
  IREE_ASSERT_OK(Prepare(1, adapter_path_.path().c_str()));
  loom_serve_jit_stage_free(compiled_[0]);
  compiled_[0] = nullptr;

  IREE_ASSERT_OK(Compile(1, "domains_interleaved"));
  const auto* candidate = loom_serve_jit_stage_program(compiled_[1]);
  EXPECT_EQ(loom_cmd_program_parameter_at(candidate, 0).fixed_buffer_index, 1u);
  IREE_ASSERT_OK(loom_serve_parameter_bank_check_layout(bank_, candidate));
  auto* group = loom_serve_parameter_bank_group(bank_);
  IREE_ASSERT_OK(loom_serve_residency_allocate(
      loom_serve_device_residency_cache(device_owner_), execution_, 1, &group,
      host_allocator_, &bank_residency_));
  IREE_ASSERT_OK(loom_serve_jit_stage_record(
      compiled_[1], iree_hal_queue_family(dispatch_),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, nullptr,
      loom_serve_parameter_bank_buffers(bank_), &commands_[1]));
  IREE_ASSERT_OK(loom_serve_residency_acquire_wait(bank_residency_));
  ASSERT_NO_FATAL_FAILURE(Run(1, 290));
  loom_serve_residency_release(bank_residency_);
}

TEST_P(ParameterBankTest, RejectsDifferentParameterKeysAndExtents) {
  IREE_ASSERT_OK(Compile(0, "target"));
  IREE_ASSERT_OK(loom_serve_parameter_bank_allocate(
      device_owner_, loom_serve_jit_stage_program(compiled_[0]),
      host_allocator_, &bank_));
  IREE_ASSERT_OK(Compile(1, "partial"));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_parameter_bank_check_layout(
                            bank_, loom_serve_jit_stage_program(compiled_[1])));
  IREE_ASSERT_OK(Compile(2, "raw"));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_parameter_bank_check_layout(
                            bank_, loom_serve_jit_stage_program(compiled_[2])));
}

TEST_P(ParameterBankTest, SelectedDomainReusesBaseAtChangedBindingOrdinal) {
  WriteCheckpoint(adapter_path_.path(), 100);
  IREE_ASSERT_OK(Compile(0, "raw"));
  IREE_ASSERT_OK(loom_serve_parameter_bank_allocate(
      device_owner_, loom_serve_jit_stage_program(compiled_[0]),
      host_allocator_, &bank_));
  IREE_ASSERT_OK(Prepare(0, path_.path().c_str()));
  auto* base = loom_serve_parameter_bank_buffers(bank_)[0];
  auto* base_group = loom_serve_parameter_bank_group(bank_);
  IREE_ASSERT_OK(loom_serve_residency_allocate(
      loom_serve_device_residency_cache(device_owner_), execution_, 1,
      &base_group, host_allocator_, &bank_residency_));
  IREE_ASSERT_OK(loom_serve_jit_stage_record(
      compiled_[0], iree_hal_queue_family(dispatch_),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, nullptr,
      loom_serve_parameter_bank_buffers(bank_), &commands_[0]));
  IREE_ASSERT_OK(loom_serve_residency_acquire_wait(bank_residency_));
  ASSERT_NO_FATAL_FAILURE(Run(0, 90));
  loom_serve_residency_release(bank_residency_);
  loom_serve_residency_free(bank_residency_);
  bank_residency_ = nullptr;
  loom_serve_jit_stage_free(compiled_[0]);
  compiled_[0] = nullptr;
  const auto before = loom_serve_device_memory_statistics(device_owner_);

  IREE_ASSERT_OK(Compile(1, "domains"));
  IREE_ASSERT_OK(loom_serve_parameter_bank_allocate_domain(
      device_owner_, loom_serve_jit_stage_program(compiled_[1]), 1,
      host_allocator_, &adapter_bank_));
  ASSERT_EQ(loom_serve_parameter_bank_count(adapter_bank_), 1u);
  const auto after = loom_serve_device_memory_statistics(device_owner_);
  EXPECT_EQ(after.retained.reserved_bytes - before.retained.reserved_bytes,
            GetParam()
                ? loom_serve_device_weight_geometry(device_owner_).minimum_size
                : 0u);
  IREE_ASSERT_OK(PrepareBank(adapter_bank_, 0, adapter_path_.path().c_str()));
  loom_serve_jit_stage_free(compiled_[1]);
  compiled_[1] = nullptr;
  IREE_ASSERT_OK(Compile(2, "domains_swapped"));
  const auto* program = loom_serve_jit_stage_program(compiled_[2]);
  const loom_serve_parameter_binding_t bindings[] = {{adapter_bank_, 0},
                                                     {bank_, 0}};
  IREE_ASSERT_OK(loom_serve_parameter_bindings_check_layout(
      IREE_ARRAYSIZE(bindings), bindings, program, host_allocator_));
  iree_hal_buffer_t* buffers[] = {
      loom_serve_parameter_bank_buffers(adapter_bank_)[0], base};
  IREE_ASSERT_OK(loom_serve_jit_stage_record(
      compiled_[2], iree_hal_queue_family(dispatch_),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, nullptr, buffers, &commands_[2]));
  loom_serve_residency_group_t* groups[] = {
      base_group, loom_serve_parameter_bank_group(adapter_bank_)};
  IREE_ASSERT_OK(loom_serve_residency_allocate(
      loom_serve_device_residency_cache(device_owner_), execution_,
      IREE_ARRAYSIZE(groups), groups, host_allocator_, &bank_residency_));
  IREE_ASSERT_OK(loom_serve_residency_acquire_wait(bank_residency_));
  ASSERT_NO_FATAL_FAILURE(Run(2, 290));
  loom_serve_residency_release(bank_residency_);
  EXPECT_EQ(loom_serve_parameter_bank_buffers(bank_)[0], base);
  EXPECT_EQ(
      loom_serve_weights_activity(loom_serve_parameter_bank_weights(bank_, 0))
          .activations,
      1u);
  EXPECT_EQ(loom_serve_weights_activity(
                loom_serve_parameter_bank_weights(adapter_bank_, 0))
                .activations,
            1u);
}

TEST_P(ParameterBankTest, RejectsChangedDomainCount) {
  IREE_ASSERT_OK(Compile(0, "domains"));
  IREE_ASSERT_OK(loom_serve_parameter_bank_allocate(
      device_owner_, loom_serve_jit_stage_program(compiled_[0]),
      host_allocator_, &bank_));
  IREE_ASSERT_OK(Compile(1, "raw"));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_parameter_bank_check_layout(
                            bank_, loom_serve_jit_stage_program(compiled_[1])));
  IREE_ASSERT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_parameter_bank_check_layout(
          nullptr, loom_serve_jit_stage_program(compiled_[1])));
}

TEST_P(ParameterBankTest, ChecksMoreDomainsThanInlineCursorStorage) {
  IREE_ASSERT_OK(Compile(0, "many_domains"));
  IREE_ASSERT_OK(loom_serve_parameter_bank_allocate(
      device_owner_, loom_serve_jit_stage_program(compiled_[0]),
      host_allocator_, &bank_));
  ASSERT_EQ(loom_serve_parameter_bank_count(bank_), 9u);
  loom_serve_jit_stage_free(compiled_[0]);
  compiled_[0] = nullptr;
  IREE_ASSERT_OK(Compile(1, "many_domains"));
  IREE_ASSERT_OK(loom_serve_parameter_bank_check_layout(
      bank_, loom_serve_jit_stage_program(compiled_[1])));
}

TEST_P(ParameterBankTest, ParameterFreeBankNeedsNoNativeBacking) {
  const auto before = loom_serve_device_memory_statistics(device_owner_);
  IREE_ASSERT_OK(loom_serve_parameter_bank_allocate(device_owner_, nullptr,
                                                    host_allocator_, &bank_));
  EXPECT_EQ(bank_, nullptr);
  EXPECT_EQ(loom_serve_parameter_bank_count(bank_), 0u);
  EXPECT_EQ(loom_serve_parameter_bank_buffers(bank_), nullptr);
  EXPECT_EQ(loom_serve_parameter_bank_group(bank_), nullptr);
  IREE_EXPECT_OK(loom_serve_parameter_bank_check_layout(bank_, nullptr));
  EXPECT_TRUE(loom_serve_parameter_bank_is_retired(bank_));
  const auto after = loom_serve_device_memory_statistics(device_owner_);
  EXPECT_EQ(after.retained.reserved_bytes, before.retained.reserved_bytes);
  EXPECT_EQ(after.retained.committed_bytes, before.retained.committed_bytes);
}

INSTANTIATE_TEST_SUITE_P(Backing, ParameterBankTest, ::testing::Bool());

}  // namespace loom_serve::testing
