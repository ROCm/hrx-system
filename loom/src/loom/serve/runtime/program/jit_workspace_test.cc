// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/sync.h"
#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/testing/jit_module.h"

namespace {

using loom_serve::testing::Compilation;

class JitWorkspaceTest : public loom_serve::testing::JitModuleTest {
 protected:
  void TearDown() override {
    // Open the external predecessor even after an assertion failure, while
    // both models' upload and feedback backing remain alive.
    if (held_.gate) {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(held_.gate, 1, nullptr));
    }
    iree_hal_buffer_release(held_.buffer);
    if (held_.execution) {
      IREE_EXPECT_OK(loom_serve_execution_drain(held_.execution));
    }
    loom_serve_execution_release(held_.execution);
    iree_hal_semaphore_release(held_.gate);
    JitModuleTest::TearDown();
  }

  // Independent model with an externally gated dependency and retained I/O.
  struct {
    // Owned model timeline sharing the fixture's workspace owner.
    loom_serve_execution_t* execution = nullptr;
    // Owned external predecessor, released only after independent output.
    iree_hal_semaphore_t* gate = nullptr;
    // Private root, released after its dealloca is accepted.
    iree_hal_buffer_t* buffer = nullptr;
    // Borrowed feedback backing kept alive through teardown.
    std::array<int32_t, 8> output = {};
  } held_;
};

TEST_P(JitWorkspaceTest, UnequalPrivateWorkspacesPreservePoolPolicyAndOutput) {
  const auto* sanitizer =
      iree_hal_device_spec_sanitizer(iree_hal_device_spec(device_));
  const bool guarded =
      iree_any_bit_set(sanitizer->flags, IREE_HAL_DEVICE_SANITIZER_FLAG_ASAN);
  const uint64_t quarantine_limit =
      guarded ? sanitizer->asan.pool_options.quarantine_size : 0;
  IREE_ASSERT_OK(Compile("5", 0, IREE_SV("temporary_advance"), "64"));
  // A non-class-boundary maximum exercises the pool's backing geometry.
  IREE_ASSERT_OK(Compile("7", 1, IREE_SV("temporary_advance"), "4160"));
  ASSERT_EQ(transients_[0].required_byte_length, 64u);
  ASSERT_EQ(transients_[1].required_byte_length, 4160u);
  IREE_ASSERT_OK(PrepareVm());
  EXPECT_EQ(loom_serve_device_memory_statistics(device_owner_)
                .workspace.bytes_committed,
            0u);
  loom_serve_jit_free(jit_);
  jit_ = nullptr;
  iree_vm_function_t selected_step;
  IREE_ASSERT_OK(iree_vm_process_lookup_function(
      process_, IREE_SV("model"), IREE_SV("selected_step"), &selected_step));
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      loom_serve_device_storage_pool(device_owner_),
      iree_hal_make_pool_reservation_request(sizeof(input_), 0),
      iree_infinite_timeout(), &buffer_));
  iree_hal_transfer_operation_t upload = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
      .upload = {input_.data(), buffer_, 0, sizeof(input_)},
  };
  uint64_t completion = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(execution_, 1, &upload, &completion));
  const int32_t stages[] = {0, 1, 0, 1};
  int32_t expected = 100;
  for (int32_t stage : stages) {
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_i32(stage),
        iree_hal_buffer_variant_from_ptr_borrowed(&types_, buffer_)};
    iree_vm_variant_t results[1] = {};
    auto status = iree_vm_invoke(invocation_, selected_step,
                                 iree_vm_variant_span_from_array(arguments),
                                 iree_vm_variant_span_from_array(results));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
    IREE_ASSERT_OK(status);
    expected += stage ? 7 : 5;
  }
  iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download = {buffer_, 0, output_.data(), sizeof(output_)},
  };
  IREE_ASSERT_OK(
      loom_serve_execution_feedback(execution_, 1, &download, &completion));
  IREE_ASSERT_OK(loom_serve_execution_drain(execution_));
  const auto statistics =
      loom_serve_device_memory_statistics(device_owner_).workspace;
  EXPECT_EQ(statistics.reservation_count, 0u);
  EXPECT_EQ(statistics.reserve_count, 4u);
  EXPECT_EQ(statistics.release_count, 4u);
  if (guarded) {
    // Sanitized deallocation retires before returning an empty frontier. These
    // reservations are fresh in HAL's dependency accounting, even on reused
    // backing; they are not frontier-dominated reuse.
    EXPECT_EQ(statistics.reuse_count, 0u);
    EXPECT_EQ(statistics.fresh_count, statistics.reserve_count);
  } else {
    EXPECT_GT(statistics.reuse_count, 0u);
  }
  EXPECT_LE(statistics.bytes_quarantined, quarantine_limit);
  if (quarantine_limit) {
    EXPECT_TRUE(statistics.bytes_quarantined > 0 ||
                statistics.quarantine_eviction_count > 0);
  }
  EXPECT_GT(statistics.bytes_committed, 0u);
  // A different model has no dependency edge to the retired producer. HAL's
  // completed-frontier query must recover reuse without a device-wide chain.
  IREE_ASSERT_OK(
      loom_serve_device_create_execution(device_owner_, &held_.execution));
  const iree_hal_pool_reservation_request_t request = {
      .allocation_size = 4160,
      .min_alignment = 16,
  };
  IREE_ASSERT_OK(
      loom_serve_execution_alloca(held_.execution, &request, &held_.buffer));
  ASSERT_NE(held_.buffer->memory.contract, nullptr);
  EXPECT_EQ(held_.buffer->memory.contract->domain,
            iree_hal_device_group_memory_domain(
                loom_serve_device_group(device_owner_)));
  IREE_ASSERT_OK(iree_hal_buffer_validate_family_usage(
      held_.buffer, iree_hal_queue_family(dispatch_),
      IREE_HAL_BUFFER_USAGE_STORAGE));
  IREE_ASSERT_OK(iree_hal_buffer_validate_family_usage(
      held_.buffer,
      iree_hal_queue_family(loom_serve_device_transfer_queue(device_owner_)),
      IREE_HAL_BUFFER_USAGE_TRANSFER));
  iree_hal_transfer_operation_t transfers[] = {
      {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
          .upload = {input_.data(), held_.buffer, 0, sizeof(input_)},
      },
      {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
          .download = {held_.buffer, 0, held_.output.data(),
                       sizeof(held_.output)},
      },
  };
  IREE_ASSERT_OK(loom_serve_execution_transfer(held_.execution, 1, transfers,
                                               &completion));
  IREE_ASSERT_OK(loom_serve_execution_transfer(held_.execution, 1,
                                               transfers + 1, &completion));
  IREE_ASSERT_OK(loom_serve_execution_dealloca(held_.execution, held_.buffer,
                                               &completion));
  iree_hal_buffer_release(held_.buffer);
  held_.buffer = nullptr;
  IREE_ASSERT_OK(loom_serve_execution_drain(held_.execution));
  EXPECT_EQ(held_.output, input_);
  const auto shared =
      loom_serve_device_memory_statistics(device_owner_).workspace;
  if (quarantine_limit) {
    // Quarantined ranges cannot satisfy this independent model's request.
    EXPECT_GE(shared.bytes_committed, statistics.bytes_committed);
    EXPECT_LE(shared.bytes_quarantined, quarantine_limit);
  } else {
    EXPECT_EQ(shared.bytes_committed, statistics.bytes_committed);
  }
  if (guarded) {
    EXPECT_EQ(shared.reuse_count, 0u);
    EXPECT_EQ(shared.fresh_count, statistics.fresh_count + 1);
  } else {
    EXPECT_GT(shared.reuse_count, statistics.reuse_count);
  }
  IREE_ASSERT_OK(loom_serve_device_trim(device_owner_, 0));
  const auto trimmed =
      loom_serve_device_memory_statistics(device_owner_).workspace;
  // Completed consumers do not imply retirement of every raw queue capture.
  // Nonwaiting trim may retain backing until those captures retire; final
  // group teardown joins them independently of model work.
  EXPECT_EQ(trimmed.reservation_count, 0u);
  EXPECT_LE(trimmed.bytes_committed, shared.bytes_committed);
  EXPECT_LE(trimmed.bytes_quarantined, quarantine_limit);
  if (quarantine_limit) {
    EXPECT_GT(trimmed.quarantine_eviction_count, 0u);
  }
  // Feedback owns the final output independently of private workspace reuse.
  for (int32_t value : output_) {
    EXPECT_EQ(value, expected);
  }
}

TEST_P(JitWorkspaceTest, IndependentModelCompletesWhileSharedWorkspaceGrows) {
  IREE_ASSERT_OK(Compile("5", 0));
  IREE_ASSERT_OK(Compile("7", 1, IREE_SV("temporary_advance"), "4160"));
  IREE_ASSERT_OK(
      loom_serve_device_create_execution(device_owner_, &held_.execution));
  IREE_ASSERT_OK(
      loom_serve_execution_reserve_workspace(held_.execution, 64, 16));
  const iree_hal_pool_reservation_request_t request = {
      .allocation_size = 64,
      .min_alignment = 16,
  };
  IREE_ASSERT_OK(
      loom_serve_execution_alloca(held_.execution, &request, &held_.buffer));
  iree_hal_transfer_operation_t upload = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
      .upload = {input_.data(), held_.buffer, 0, sizeof(input_)},
  };
  uint64_t completion = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(held_.execution, 1, &upload, &completion));
  IREE_ASSERT_OK(loom_serve_execution_wait(held_.execution, completion));
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &held_.gate));
  uint64_t gate_value = 1;
  IREE_ASSERT_OK(loom_serve_execution_barrier(held_.execution,
                                              {1, &held_.gate, &gate_value}));
  const iree_hal_buffer_binding_t binding = {held_.buffer, 0, sizeof(input_)};
  IREE_ASSERT_OK(loom_serve_execution_execute(held_.execution, commands_[0],
                                              {1, &binding}, &completion));
  iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download = {held_.buffer, 0, held_.output.data(), sizeof(held_.output)},
  };
  IREE_ASSERT_OK(loom_serve_execution_feedback(held_.execution, 1, &download,
                                               &completion));
  auto held_frontier = loom_serve_execution_frontier(held_.execution);
  // Feedback is an independent branch. Explicitly join its read before freeing
  // storage that another model may immediately reuse.
  IREE_ASSERT_OK(loom_serve_execution_barrier(
      held_.execution,
      {1, &held_frontier.semaphores[1], &held_frontier.values[1]}));
  IREE_ASSERT_OK(loom_serve_execution_dealloca(held_.execution, held_.buffer,
                                               &completion));
  iree_hal_buffer_release(held_.buffer);
  held_.buffer = nullptr;

  // Real VM registration grows slab geometry while the old pool has a live
  // consumer. Neither registration nor the second model may join that consumer.
  IREE_ASSERT_OK(PrepareVm());
  iree_vm_function_t selected_step;
  IREE_ASSERT_OK(iree_vm_process_lookup_function(
      process_, IREE_SV("model"), IREE_SV("selected_step"), &selected_step));
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      loom_serve_device_storage_pool(device_owner_),
      iree_hal_make_pool_reservation_request(sizeof(input_), 0),
      iree_infinite_timeout(), &buffer_));
  upload.upload.target_buffer = buffer_;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(execution_, 1, &upload, &completion));
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_i32(1),
      iree_hal_buffer_variant_from_ptr_borrowed(&types_, buffer_)};
  iree_vm_variant_t results[1] = {};
  auto status = iree_vm_invoke(invocation_, selected_step,
                               iree_vm_variant_span_from_array(arguments),
                               iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  IREE_ASSERT_OK(status);
  download.download = {buffer_, 0, output_.data(), sizeof(output_)};
  IREE_ASSERT_OK(
      loom_serve_execution_feedback(execution_, 1, &download, &completion));
  IREE_ASSERT_OK(loom_serve_execution_drain(execution_));
  for (int32_t value : output_) {
    EXPECT_EQ(value, 107);
  }
  uint64_t observed = 0;
  IREE_ASSERT_OK(
      iree_hal_semaphore_query(held_frontier.semaphores[1], &observed));
  EXPECT_LT(observed, held_frontier.values[1]);
  const auto live =
      loom_serve_device_memory_statistics(device_owner_).workspace;
  EXPECT_EQ(live.reservation_count, 1u);
  IREE_ASSERT_OK(loom_serve_device_trim(device_owner_, 0));
  EXPECT_GT(loom_serve_device_memory_statistics(device_owner_)
                .workspace.bytes_committed,
            0u);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(held_.gate, 1, nullptr));
  iree_hal_semaphore_release(held_.gate);
  held_.gate = nullptr;
  IREE_ASSERT_OK(loom_serve_execution_drain(held_.execution));
  for (size_t i = 0; i < held_.output.size(); ++i) {
    EXPECT_EQ(held_.output[i], 105 + i);
  }
  IREE_ASSERT_OK(loom_serve_device_trim(device_owner_, 0));
  const auto trimmed =
      loom_serve_device_memory_statistics(device_owner_).workspace;
  // The model has retired, but a raw allocation capture can still preserve an
  // older pool generation. Trim must neither wait for it nor grow backing.
  EXPECT_LE(trimmed.bytes_committed, live.bytes_committed);
  EXPECT_EQ(trimmed.reservation_count, 0u);
  EXPECT_EQ(trimmed.reserve_count, 2u);
  EXPECT_EQ(trimmed.release_count, 2u);
}

TEST_P(JitWorkspaceTest, RetiredModelDoesNotJoinRawQueueAllocationCaptures) {
  IREE_ASSERT_OK(
      loom_serve_device_create_execution(device_owner_, &held_.execution));
  IREE_ASSERT_OK(
      loom_serve_execution_reserve_workspace(held_.execution, 64, 16));
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &held_.gate));
  uint64_t value = 1;
  IREE_ASSERT_OK(
      loom_serve_execution_barrier(held_.execution, {1, &held_.gate, &value}));
  const iree_hal_pool_reservation_request_t request = {
      .allocation_size = 64,
      .min_alignment = 16,
  };
  IREE_ASSERT_OK(
      loom_serve_execution_alloca(held_.execution, &request, &held_.buffer));
  IREE_ASSERT_OK(
      loom_serve_execution_dealloca(held_.execution, held_.buffer, &value));
  iree_hal_buffer_release(held_.buffer);
  held_.buffer = nullptr;
  EXPECT_TRUE(loom_serve_execution_is_retired(held_.execution));

  // Neither raw queue operation can run yet, but no model consumer remains.
  // A hidden join in release blocks the shared worker and prevents this real
  // source compilation from completing. The gate opens only after GPU output.
  Compilation compilation;
  compilation.jit = jit_;
  compilation.root = IREE_SV("temporary_advance");
  compilation.retired_execution = held_.execution;
  held_.execution = nullptr;
  const loomc_config_binding_t bindings[] = {
      {loomc_make_cstring_view("increment.delta"),
       loomc_make_cstring_view("7")},
      {loomc_make_cstring_view("increment.temporary_bytes"),
       loomc_make_cstring_view("4160")},
  };
  const loomc_config_options_t config = {
      bindings,
      IREE_ARRAYSIZE(bindings),
      {},
      LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED};
  compilation.config = &config;
  compilation.Start(
      loom_serve_host_executor(loom_serve_device_host(device_owner_)));
  loom_serve_jit_stage_t* stage = nullptr;
  IREE_ASSERT_OK(compilation.Await(&stage));
  auto status = loom_serve_jit_stage_record(
      stage, iree_hal_queue_family(dispatch_),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, nullptr, nullptr, &commands_[1]);
  transients_[1] = loom_serve_jit_stage_program(stage)->requirements.transient;
  loom_serve_jit_stage_free(stage);
  IREE_ASSERT_OK(status);
  IREE_ASSERT_OK(Compile("5", 0));
  IREE_ASSERT_OK(PrepareVm());
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      loom_serve_device_storage_pool(device_owner_),
      iree_hal_make_pool_reservation_request(sizeof(input_), 0),
      iree_infinite_timeout(), &buffer_));
  const iree_hal_transfer_operation_t upload = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
      .upload = {input_.data(), buffer_, 0, sizeof(input_)},
  };
  IREE_ASSERT_OK(loom_serve_execution_transfer(execution_, 1, &upload, &value));
  iree_vm_function_t selected_step;
  IREE_ASSERT_OK(iree_vm_process_lookup_function(
      process_, IREE_SV("model"), IREE_SV("selected_step"), &selected_step));
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_i32(1),
      iree_hal_buffer_variant_from_ptr_borrowed(&types_, buffer_)};
  iree_vm_variant_t results[1] = {};
  status = iree_vm_invoke(invocation_, selected_step,
                          iree_vm_variant_span_from_array(arguments),
                          iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  IREE_ASSERT_OK(status);
  const iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download = {buffer_, 0, output_.data(), sizeof(output_)},
  };
  IREE_ASSERT_OK(
      loom_serve_execution_feedback(execution_, 1, &download, &value));
  IREE_ASSERT_OK(loom_serve_execution_drain(execution_));
  for (int32_t output : output_) {
    EXPECT_EQ(output, 107);
  }
  uint64_t observed = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(held_.gate, &observed));
  EXPECT_EQ(observed, 0u);
  IREE_ASSERT_OK(loom_serve_device_trim(device_owner_, 0));
  // Final application teardown owns the remaining raw captures. It opens the
  // predecessor and joins them while keeping the shared pool/device alive.
}

INSTANTIATE_TEST_SUITE_P(SharedWorkers, JitWorkspaceTest,
                         ::testing::Values(1, 4));

}  // namespace
