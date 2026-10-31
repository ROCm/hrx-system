// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/weights_test_base.h"

namespace loom_serve::testing {

class WeightPackingTest : public WeightsTest {
 protected:
  void WriteFactors(const std::string& path, int32_t offset, size_t width) {
    std::string header = "{";
    for (size_t i = 0; i < 6; ++i) {
      if (i) {
        header += ",";
      }
      header += "\"" + std::string(1, 'a' + i) + "\":{\"shape\":[2," +
                std::to_string(width / 2) +
                "],\"dtype\":\"F32\",\"data_offsets\":[" +
                std::to_string(i * width * 4) + "," +
                std::to_string((i + 1) * width * 4) + "]}";
    }
    header += "}";
    std::ofstream file(path, std::ios::binary);
    for (unsigned i = 0; i < 8; ++i) {
      file.put(
          static_cast<char>(static_cast<uint64_t>(header.size()) >> (8 * i)));
    }
    file.write(header.data(), header.size());
    for (size_t tensor = 0; tensor < 6; ++tensor) {
      for (size_t element = 0; element < width; ++element) {
        const float value = offset + (tensor + 1) * 10 + element;
        file.write(reinterpret_cast<const char*>(&value), sizeof(value));
      }
    }
    file.close();
    ASSERT_TRUE(file.good());
  }

  static int32_t PackedValue(size_t tensor, size_t element) {
    // Odd ASCII keys interleave columns; even keys concatenate rows.
    const bool interleaved = (tensor % 2) == 0;
    const bool first = interleaved ? element % 6 < 4 : element < 8;
    const size_t index = interleaved
                             ? (element / 6) * (first ? 4 : 2) +
                                   (first ? element % 6 : element % 6 - 4)
                             : (first ? element : element - 8);
    return (tensor + 1) * 10 + (first ? 0 : 100) + index;
  }

  void TrimWorkspace() {
    IREE_ASSERT_OK(loom_serve_test_workspace_await_idle(
        loom_serve_pool_workspace(storage_owner_)));
    iree_device_size_t retained = 0;
    IREE_ASSERT_OK(loom_serve_workspace_trim(
        loom_serve_pool_workspace(storage_owner_), &retained));
    ASSERT_NO_FATAL_FAILURE(DrainNativeReturns());
    EXPECT_EQ(
        loom_serve_pool_statistics(storage_owner_).workspace.bytes_committed,
        0u);
  }

  void CheckPackedOn(loom_serve_execution_t* execution) {
    const auto* program = loom_serve_jit_stage_program(compiled_[0]);
    std::vector<uint16_t> packed(iree_hal_buffer_byte_length(buffers_[0][0]) /
                                 sizeof(uint16_t));
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = buffers_[0][0],
                .target = packed.data(),
                .length = packed.size() * sizeof(uint16_t),
            },
    };
    uint64_t completion = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_feedback(execution, 1, &download, &completion));
    IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution, completion));
    ASSERT_EQ(program->parameters.count, 6u);
    for (uint32_t i = 0; i < program->parameters.count; ++i) {
      const auto parameter = loom_cmd_program_parameter_at(program, i);
      ASSERT_EQ(parameter.key.size, 1u);
      ASSERT_EQ(parameter.byte_length, 24u);
      const size_t tensor = parameter.key.data[0] - 'a';
      for (size_t element = 0; element < 12; ++element) {
        const float value = PackedValue(tensor, element);
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        EXPECT_EQ(packed[parameter.byte_offset / 2 + element], bits >> 16)
            << parameter.key.data[0] << "[" << element << "]";
      }
    }
    ASSERT_NO_FATAL_FAILURE(ReadOutputOn(execution, 0));
    for (size_t element = 0; element < output_.size(); ++element) {
      int32_t expected = 0;
      for (size_t tensor = 0; tensor < 6; ++tensor) {
        expected += PackedValue(tensor, element);
      }
      EXPECT_EQ(output_[element], expected);
    }
  }
};

TEST_P(WeightPackingTest, ActivationWorkingSetIncludesCapturedInputLanes) {
  WriteFactors(path_.path(), 0, 8);
  WriteFactors(adapter_path_.path(), 100, 4);
  IREE_ASSERT_OK(Compile(0, "packed_target"));
  // Final parameters fit the full budget by themselves. Their independently
  // captured source inputs still need four reusable staging lanes while the
  // final output storage is live.
  const auto root = Root(0, 0, 2 * kSlabBytes);
  IREE_ASSERT_OK(CreateSourceRoots(0, 1, &root,
                                   {{path_.path()}, {adapter_path_.path()}},
                                   "packing_policy.loom"));
  const auto staging = loom_serve_weights_activation_workspace(plans_[0]);
  EXPECT_EQ(staging.allocation_size, 512u);
  EXPECT_EQ(staging.min_alignment, 64u);
  if (pool_) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Register(0, 0, 1));
    EXPECT_EQ(residencies_[0], nullptr);
    EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes, 0u);
    EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).pending_bytes, 0u);
  } else {
    IREE_ASSERT_OK(Register(0, 0, 1));
  }
  const auto activity = loom_serve_weights_activity(plans_[0]);
  EXPECT_EQ(activity.activations, 0u);
  EXPECT_EQ(activity.read_bytes, 0u);
  EXPECT_EQ(loom_serve_execution_frontier(execution_).values[0], 0u);
}

TEST_P(WeightPackingTest,
       InvocationWorkingSetChecksTheConfiguredLimitBeforeLoading) {
  WriteFactors(path_.path(), 0, 8);
  WriteFactors(adapter_path_.path(), 100, 4);
  IREE_ASSERT_OK(Compile(0, "packed_target"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateSourceRoots(0, 1, &root,
                                   {{path_.path()}, {adapter_path_.path()}},
                                   "packing_policy.loom"));
  IREE_ASSERT_OK(Register(0, 0, 1));
  const uint64_t weights =
      loom_serve_residency_statistics(residencies_[0]).reserved_bytes;
  const uint64_t limit =
      pool_ ? loom_serve_memory_pool_limit(pool_) : UINT64_MAX;
  const uint64_t headroom = limit - weights;
  const iree_device_size_t fitting[] = {headroom / 2, headroom - headroom / 2};
  IREE_ASSERT_OK(loom_serve_residency_check_working_set(
      residencies_[0], IREE_ARRAYSIZE(fitting), fitting));
  const iree_device_size_t impossible[] = {headroom / 2,
                                           headroom - headroom / 2, 1};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_serve_residency_check_working_set(
          residencies_[0], IREE_ARRAYSIZE(impossible), impossible));
  const auto before = loom_serve_residency_snapshot(residencies_[0]);
  EXPECT_FALSE(before.active);
  EXPECT_EQ(before.pin_count, 0u);
  EXPECT_FALSE(before.admission.waiting);
  EXPECT_EQ(before.activity.read_bytes, 0u);
  EXPECT_EQ(loom_serve_execution_frontier(execution_).values[0], 0u);
  // A denied shape has not poisoned either the parameter plan or execution.
  IREE_ASSERT_OK(loom_serve_residency_acquire_wait(residencies_[0]));
  ASSERT_NO_FATAL_FAILURE(CheckPackedOn(execution_));
  loom_serve_residency_release(residencies_[0]);
}

TEST_P(WeightPackingTest,
       InvocationWorkingSetAdmitsStateAndParametersTogether) {
  // One weight slab, one state slab, and captured input lanes fit together.
  // Adding a second state slab leaves no space for those activation lanes.
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking(3 * kSlabBytes));
  WriteFactors(path_.path(), 0, 8);
  WriteFactors(adapter_path_.path(), 100, 4);
  IREE_ASSERT_OK(Compile(0, "packed_target"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateSourceRoots(0, 1, &root,
                                   {{path_.path()}, {adapter_path_.path()}},
                                   "packing_policy.loom"));
  IREE_ASSERT_OK(Register(0, 0, 1));
  loom_serve_residency_admission_t admission;
  loom_serve_memory_operation_t* operation = nullptr;
  loom_serve_memory_range_t range = {};
  if (pool_) {
    IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
        pool_, loom_serve_pool_weight_geometry(storage_owner_), 3 * kSlabBytes,
        256, &state_statistics_, &state_));
    range = {state_, 0, 2 * kSlabBytes};
    // Either resource fits alone. Their union plus activation staging does
    // not: neither backing nor parameter I/O may start, and the caller retains
    // no partial pin.
    IREE_ASSERT_OK(loom_serve_residency_try_admit_state(
        residencies_[0], 1, &range, nullptr, 0, &admission, &operation));
    EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
    EXPECT_EQ(operation, nullptr);
    EXPECT_EQ(state_statistics_.pending_bytes, 0u);
    EXPECT_EQ(state_statistics_.committed_bytes, 0u);
    const auto denied = loom_serve_residency_snapshot(residencies_[0]);
    EXPECT_EQ(denied.pin_count, 0u);
    EXPECT_FALSE(denied.active);
    EXPECT_EQ(denied.activity.activations, 0u);
    EXPECT_EQ(denied.activity.read_bytes, 0u);
    EXPECT_EQ(denied.memory.pending_bytes, 0u);
    EXPECT_EQ(denied.memory.committed_bytes, 0u);
    range.length = kSlabBytes;
  }
  IREE_ASSERT_OK(loom_serve_residency_try_admit_state(
      residencies_[0], pool_ ? 1 : 0, &range, nullptr, 0, &admission,
      &operation));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  loom_serve_memory_operation_await(operation);
  IREE_EXPECT_OK(loom_serve_memory_operation_query_status(operation));
  loom_serve_memory_operation_free(operation);
  IREE_ASSERT_OK(loom_serve_residency_await(residencies_[0]));
  loom_serve_residency_cancel_admission(residencies_[0]);
  ASSERT_NO_FATAL_FAILURE(CheckPackedOn(execution_));
  if (pool_) {
    EXPECT_EQ(state_statistics_.committed_bytes, kSlabBytes);
    const auto before = loom_serve_residency_snapshot(residencies_[0]);
    // Warm loading credit is gone, but three state slabs plus the parameters
    // still exceed capacity. Denial preserves the accepted consumer's pin.
    range.length = 3 * kSlabBytes;
    IREE_ASSERT_OK(loom_serve_residency_try_admit_state(
        residencies_[0], 1, &range, nullptr, 0, &admission, &operation));
    EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
    EXPECT_EQ(operation, nullptr);
    const auto denied = loom_serve_residency_snapshot(residencies_[0]);
    EXPECT_EQ(denied.pin_count, before.pin_count);
    EXPECT_EQ(denied.activity.activations, before.activity.activations);
    EXPECT_EQ(state_statistics_.committed_bytes, kSlabBytes);
    ASSERT_NO_FATAL_FAILURE(CheckPackedOn(execution_));
    loom_serve_residency_cancel_admission(residencies_[0]);
  }
  loom_serve_residency_release(residencies_[0]);
}

TEST_P(WeightPackingTest, InvocationWorkingSetRetiresColdActivationCredit) {
  WriteFactors(path_.path(), 0, 8);
  WriteFactors(adapter_path_.path(), 100, 4);
  IREE_ASSERT_OK(Compile(0, "packed_target"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateSourceRoots(0, 1, &root,
                                   {{path_.path()}, {adapter_path_.path()}},
                                   "packing_policy.loom"));
  IREE_ASSERT_OK(Register(0, 0, 1));
  IREE_ASSERT_OK(
      loom_serve_residency_prepare_working_set(residencies_[0], 0, 0));
  loom_serve_residency_admission_t admission;
  loom_serve_memory_operation_t* operation = nullptr;
  loom_serve_memory_range_t range = {};
  if (pool_) {
    IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
        pool_, loom_serve_pool_weight_geometry(storage_owner_), kSlabBytes, 256,
        &state_statistics_, &state_));
    range = {state_, 0, kSlabBytes};
    IREE_ASSERT_OK(loom_serve_residency_try_admit_state(
        residencies_[0], 1, &range, nullptr, 0, &admission, &operation));
    EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
    EXPECT_EQ(operation, nullptr);
    EXPECT_EQ(loom_serve_weights_activity(plans_[0]).activations, 0u);
    loom_serve_residency_cancel_admission(residencies_[0]);
  }
  // Keep the prepared weights while retiring and trimming only cold scratch.
  loom_serve_residency_set_pinned(residencies_[0], true);
  IREE_ASSERT_OK(loom_serve_residency_acquire_wait(residencies_[0]));
  ASSERT_NO_FATAL_FAILURE(CheckPackedOn(execution_));
  loom_serve_residency_release(residencies_[0]);
  ASSERT_NO_FATAL_FAILURE(TrimWorkspace());
  IREE_ASSERT_OK(loom_serve_residency_try_admit_state(
      residencies_[0], pool_ ? 1 : 0, &range, nullptr, 0, &admission,
      &operation));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  loom_serve_memory_operation_await(operation);
  IREE_EXPECT_OK(loom_serve_memory_operation_query_status(operation));
  loom_serve_memory_operation_free(operation);
  if (admission == LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED) {
    ASSERT_NO_FATAL_FAILURE(CheckPackedOn(execution_));
    loom_serve_residency_release(residencies_[0]);
  }
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).activations, 1u);
  loom_serve_residency_cancel_admission(residencies_[0]);
  if (pool_) {
    // Eviction restores the cold requirement even though the same residency
    // and source plan survive. Retained state cannot spend that credit.
    loom_serve_residency_set_pinned(residencies_[0], false);
    bool complete = false;
    IREE_ASSERT_OK(loom_serve_residency_deactivate(residencies_[0], &complete));
    IREE_ASSERT_OK(loom_serve_memory_pool_await(pool_));
    IREE_ASSERT_OK(loom_serve_residency_deactivate(residencies_[0], &complete));
    EXPECT_TRUE(complete);
    IREE_ASSERT_OK(loom_serve_residency_try_admit_state(
        residencies_[0], 1, &range, nullptr, 0, &admission, &operation));
    EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
    EXPECT_EQ(operation, nullptr);
    EXPECT_EQ(loom_serve_weights_activity(plans_[0]).activations, 1u);
    loom_serve_residency_cancel_admission(residencies_[0]);
    IREE_ASSERT_OK(loom_serve_virtual_buffer_free(state_));
    state_ = nullptr;
    IREE_ASSERT_OK(loom_serve_residency_acquire_wait(residencies_[0]));
    ASSERT_NO_FATAL_FAILURE(CheckPackedOn(execution_));
    loom_serve_residency_release(residencies_[0]);
    EXPECT_EQ(loom_serve_weights_activity(plans_[0]).activations, 2u);
  }
}

TEST_P(WeightPackingTest,
       InvocationWorkingSetSharesStagingRetirementAcrossPinnedBorrowers) {
  WriteFactors(path_.path(), 0, 8);
  WriteFactors(adapter_path_.path(), 100, 4);
  IREE_ASSERT_OK(Compile(0, "packed_target"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateSourceRoots(0, 1, &root,
                                   {{path_.path()}, {adapter_path_.path()}},
                                   "packing_policy.loom"));
  IREE_ASSERT_OK(Register(0, 0, 1));
  IREE_ASSERT_OK(loom_serve_device_create_execution(
      device_owner_, storage_owner_, &shared_execution_));
  IREE_ASSERT_OK(loom_serve_residency_allocate(
      ResidencyCache(), shared_execution_, 1, &parameter_groups_[0],
      host_allocator_, &residencies_[1]));
  loom_serve_residency_admission_t admission;
  loom_serve_memory_operation_t* operation = nullptr;
  loom_serve_memory_range_t range = {};
  if (pool_) {
    IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
        pool_, loom_serve_pool_weight_geometry(storage_owner_), kSlabBytes, 256,
        &state_statistics_, &state_));
    range = {state_, 0, kSlabBytes};
  }
  {
    ScopedNativeHold hold(device_owner_);
    IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[0]));
    if (pool_) {
      hold.WaitEntered();
    }
    IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[1]));
    if (pool_) {
      // Submission and an explicit policy unpin do not return staging credit.
      loom_serve_residency_set_pinned(residencies_[0], true);
      loom_serve_residency_set_pinned(residencies_[0], false);
      IREE_ASSERT_OK(loom_serve_residency_try_admit_state(
          residencies_[1], 1, &range, nullptr, 0, &admission, &operation));
      EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED);
      EXPECT_EQ(operation, nullptr);
      EXPECT_EQ(loom_serve_residency_snapshot(residencies_[1]).pin_count, 1u);
    }
  }
  IREE_ASSERT_OK(loom_serve_residency_await(residencies_[0]));
  ASSERT_NO_FATAL_FAILURE(CheckPackedOn(execution_));
  // Neither borrower's invocation has returned its pin. Completion of their
  // shared producer is enough to admit state without another parameter load.
  ASSERT_NO_FATAL_FAILURE(TrimWorkspace());
  IREE_ASSERT_OK(loom_serve_residency_try_admit_state(
      residencies_[1], pool_ ? 1 : 0, &range, nullptr, 0, &admission,
      &operation));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  loom_serve_memory_operation_await(operation);
  IREE_EXPECT_OK(loom_serve_memory_operation_query_status(operation));
  loom_serve_memory_operation_free(operation);
  if (admission == LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED) {
    ASSERT_NO_FATAL_FAILURE(CheckPackedOn(shared_execution_));
    loom_serve_residency_release(residencies_[1]);
  }
  if (pool_) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        loom_serve_residency_check_private_peak(residencies_[0], 1));
  }
  loom_serve_residency_release(residencies_[1]);
  loom_serve_residency_release(residencies_[0]);
  loom_serve_residency_cancel_admission(residencies_[1]);
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).activations, 1u);
}

TEST_P(WeightPackingTest,
       InvocationWorkingSetRetainsPrivateAndEscapingOutputCredit) {
  WriteFactors(path_.path(), 0, 8);
  WriteFactors(adapter_path_.path(), 100, 4);
  IREE_ASSERT_OK(Compile(0, "packed_target"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateSourceRoots(0, 1, &root,
                                   {{path_.path()}, {adapter_path_.path()}},
                                   "packing_policy.loom"));
  IREE_ASSERT_OK(Register(0, 0, 1));
  IREE_ASSERT_OK(
      loom_serve_residency_prepare_working_set(residencies_[0], 0, 1));
  loom_serve_residency_set_pinned(residencies_[0], true);
  IREE_ASSERT_OK(loom_serve_residency_acquire_wait(residencies_[0]));
  ASSERT_NO_FATAL_FAILURE(CheckPackedOn(execution_));
  iree_hal_buffer_t* escaping = output_buffer_;
  iree_hal_buffer_retain(escaping);
  auto status = loom_serve_residency_track_output(residencies_[0], &escaping);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(escaping);
  }
  IREE_ASSERT_OK(status);
  loom_serve_residency_release(residencies_[0]);
  ASSERT_NO_FATAL_FAILURE(TrimWorkspace());
  // The next invocation produces no output, but the previous consumer's view
  // retains its independently charged bound after the producer has finished.
  IREE_ASSERT_OK(
      loom_serve_residency_prepare_working_set(residencies_[0], 0, 0));
  loom_serve_residency_admission_t admission;
  loom_serve_memory_operation_t* operation = nullptr;
  loom_serve_memory_range_t range = {};
  if (pool_) {
    IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
        pool_, loom_serve_pool_weight_geometry(storage_owner_), kSlabBytes, 256,
        &state_statistics_, &state_));
    range = {state_, 0, kSlabBytes};
    IREE_EXPECT_OK(loom_serve_residency_try_admit_state(
        residencies_[0], 1, &range, nullptr, 0, &admission, &operation));
    EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
    EXPECT_EQ(operation, nullptr);
  }
  iree_hal_buffer_release(escaping);
  IREE_ASSERT_OK(
      loom_serve_residency_prepare_working_set(residencies_[0], 1, 0));
  if (pool_) {
    IREE_ASSERT_OK(loom_serve_residency_try_admit_state(
        residencies_[0], 1, &range, nullptr, 0, &admission, &operation));
    EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
    EXPECT_EQ(operation, nullptr);
  }
  IREE_ASSERT_OK(loom_serve_residency_check_private_peak(residencies_[0], 0));
  loom_serve_residency_set_private_peak(residencies_[0], 0);
  IREE_ASSERT_OK(loom_serve_residency_try_admit_state(
      residencies_[0], pool_ ? 1 : 0, &range, nullptr, 0, &admission,
      &operation));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  loom_serve_memory_operation_await(operation);
  IREE_EXPECT_OK(loom_serve_memory_operation_query_status(operation));
  loom_serve_memory_operation_free(operation);
  if (admission == LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED) {
    ASSERT_NO_FATAL_FAILURE(CheckPackedOn(execution_));
    loom_serve_residency_release(residencies_[0]);
  }
  loom_serve_residency_cancel_admission(residencies_[0]);
}

TEST_P(WeightPackingTest,
       InvocationWorkingSetFailedStagingDoesNotPromiseCapacity) {
  WriteFactors(path_.path(), 0, 8);
  WriteFactors(adapter_path_.path(), 100, 4);
  IREE_ASSERT_OK(Compile(0, "packed_target"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateSourceRoots(0, 1, &root,
                                   {{path_.path()}, {adapter_path_.path()}},
                                   "packing_policy.loom"));
  IREE_ASSERT_OK(Register(0, 0, 1));
  IREE_ASSERT_OK(loom_serve_device_create_execution(
      device_owner_, storage_owner_, &shared_execution_));
  IREE_ASSERT_OK(loom_serve_residency_allocate(
      ResidencyCache(), shared_execution_, 0, nullptr, host_allocator_,
      &residencies_[1]));
  IREE_ASSERT_OK(
      loom_serve_residency_prepare_working_set(residencies_[1], kSlabBytes, 0));
  execution_code_ = IREE_STATUS_RESOURCE_EXHAUSTED;
  {
    ScopedNativeHold hold(device_owner_);
    NativeAllocator::ForDevice(device_owner_)
        .FailNextAllocation(IREE_ALLOCATOR_COMMAND_CALLOC, true);
    IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[0]));
    hold.WaitEntered();
    EXPECT_TRUE(loom_serve_residency_cache_is_reclaiming(ResidencyCache()));
  }
  IREE_EXPECT_STATUS_IS(execution_code_,
                        loom_serve_residency_await(residencies_[0]));
  uint64_t staging_bytes = 0;
  EXPECT_FALSE(
      loom_serve_weights_query_activation_workspace(plans_[0], &staging_bytes));
  EXPECT_EQ(staging_bytes,
            loom_serve_weights_activation_workspace(plans_[0]).allocation_size);
  // Producer retirement does not join raw queue-allocation bookkeeping or
  // backing returns. Settle those owners before inspecting global capacity;
  // the failed producer's declared staging requirement remains charged.
  ASSERT_NO_FATAL_FAILURE(TrimWorkspace());
  IREE_ASSERT_OK(loom_serve_residency_cache_query_status(ResidencyCache()));
  EXPECT_FALSE(loom_serve_residency_cache_is_reclaiming(ResidencyCache()));
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[1], &admission));
  if (pool_) {
    // The failed producer still owns its working set until terminal teardown,
    // but accepted loading is no longer a reason to wait for future capacity.
    EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
  } else {
    EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
    loom_serve_residency_release(residencies_[1]);
  }
  loom_serve_residency_cancel_admission(residencies_[1]);
}

TEST_P(WeightPackingTest,
       IndependentInputsPackWithBoundedLaneReuseAndWarmSharing) {
  WriteFactors(path_.path(), 0, 8);
  WriteFactors(adapter_path_.path(), 100, 4);
  IREE_ASSERT_OK(Compile(0, "packed_target"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateSourceRoots(0, 1, &root,
                                   {{path_.path()}, {adapter_path_.path()}},
                                   "packing_policy.loom"));
  EXPECT_EQ(compilation_.commands, 3u);
  const auto initial_workspace =
      loom_serve_pool_statistics(storage_owner_).workspace;
  IREE_ASSERT_OK(loom_serve_weights_activate(plans_[0]));
  IREE_ASSERT_OK(loom_serve_weights_import_readiness(plans_[0], execution_));
  CheckPackedOn(execution_);
  const auto packed_workspace =
      loom_serve_pool_statistics(storage_owner_).workspace;
  EXPECT_EQ(packed_workspace.reserve_count,
            initial_workspace.reserve_count + 1);
  EXPECT_EQ(packed_workspace.release_count,
            initial_workspace.release_count + 1);
  EXPECT_EQ(packed_workspace.reservation_count, 0u);
  EXPECT_EQ(packed_workspace.bytes_reserved, 0u);
  const auto before = loom_serve_weights_activity(plans_[0]);
  EXPECT_EQ(before.activations, 1u);
  EXPECT_EQ(before.read_bytes, 6u * (32u + 16u));
  EXPECT_EQ(before.prepared_bytes, 6u * 24u);
  EXPECT_EQ(before.groups, 12u);
  IREE_ASSERT_OK(loom_serve_device_create_execution(
      device_owner_, storage_owner_, &shared_execution_));
  IREE_ASSERT_OK(loom_serve_weights_activate(plans_[0]));
  IREE_ASSERT_OK(
      loom_serve_weights_import_readiness(plans_[0], shared_execution_));
  CheckPackedOn(shared_execution_);
  const auto after = loom_serve_weights_activity(plans_[0]);
  EXPECT_EQ(after.activations, before.activations);
  EXPECT_EQ(after.read_bytes, before.read_bytes);
  EXPECT_EQ(after.prepared_bytes, before.prepared_bytes);
  EXPECT_EQ(after.groups, before.groups);
  EXPECT_EQ(compilation_.commands, 3u);
  const auto warm_workspace =
      loom_serve_pool_statistics(storage_owner_).workspace;
  EXPECT_EQ(warm_workspace.reserve_count, packed_workspace.reserve_count);
  EXPECT_EQ(warm_workspace.release_count, packed_workspace.release_count);
}

TEST_P(WeightPackingTest,
       CancelledPackingRetainsStagingUntilAcceptedWorkRetires) {
  WriteFactors(path_.path(), 0, 8);
  WriteFactors(adapter_path_.path(), 100, 4);
  IREE_ASSERT_OK(Compile(0, "packed_target"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateSourceRoots(0, 1, &root,
                                   {{path_.path()}, {adapter_path_.path()}},
                                   "packing_policy.loom"));
  IREE_ASSERT_OK(Register(0, 0, 1));
  loom_serve_test_weight_activation_wake_t wake;
  loom_serve_activation_t* activation = nullptr;
  IREE_ASSERT_OK(loom_serve_activation_allocate(
      residencies_[0],
      loom_serve_host_proactor_pool(loom_serve_device_host(device_owner_)),
      loom_serve_test_weight_activation_wake_t::Notify, &wake, host_allocator_,
      &activation));
  loom_serve_activation_phase_t phase;
  {
    loom_serve::testing::ScopedNativeHold hold(device_owner_);
    IREE_ASSERT_OK(loom_serve_activation_advance(activation, 0, &phase));
    ASSERT_EQ(phase, LOOM_SERVE_ACTIVATION_LOADING);
    hold.WaitEntered();
    IREE_ASSERT_OK(loom_serve_activation_advance(
        activation, LOOM_SERVE_ACTIVATION_FLAG_CANCEL, &phase));
    EXPECT_EQ(phase, LOOM_SERVE_ACTIVATION_LOADING);
    IREE_ASSERT_OK(loom_serve_pool_trim(storage_owner_, 0));
  }
  // The outer physical owner services cold staging while joining the accepted
  // activation; its completion callback still owns the cancellation boundary.
  // Consume terminal completion and release the activation even when loading
  // fails, so an assertion does not abandon its retained status and observer.
  IREE_EXPECT_OK(loom_serve_residency_await(residencies_[0]));
  wake.Await();
  IREE_EXPECT_OK(loom_serve_activation_advance(
      activation, LOOM_SERVE_ACTIVATION_FLAG_CANCEL, &phase));
  EXPECT_EQ(phase, LOOM_SERVE_ACTIVATION_COMPLETE);
  loom_serve_activation_free(activation);
  IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[0]));
  CheckPackedOn(execution_);
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).activations, 1u);
  loom_serve_residency_release(residencies_[0]);
  IREE_ASSERT_OK(loom_serve_pool_trim(storage_owner_, 0));
  DrainNativeReturns();
  EXPECT_EQ(
      loom_serve_pool_statistics(storage_owner_).workspace.bytes_committed, 0u);
  // Elastic eviction drops only the result backing, not indexes or commands.
  IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[0]));
  CheckPackedOn(execution_);
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).activations,
            GetParam() ? 2u : 1u);
  EXPECT_EQ(compilation_.commands, 3u);
  loom_serve_residency_release(residencies_[0]);
}

TEST_P(WeightPackingTest,
       PackingAllocationFailureRetiresBothInputAndOutputOwnership) {
  WriteFactors(path_.path(), 0, 8);
  WriteFactors(adapter_path_.path(), 100, 4);
  IREE_ASSERT_OK(Compile(0, "packed_target"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateSourceRoots(0, 1, &root,
                                   {{path_.path()}, {adapter_path_.path()}},
                                   "packing_policy.loom"));
  auto& allocator =
      loom_serve::testing::NativeAllocator::ForDevice(device_owner_);
  allocator.FailNextAllocation(IREE_ALLOCATOR_COMMAND_CALLOC, true);
  // Queue capture succeeds; physical acquisition fails asynchronously.
  IREE_ASSERT_OK(loom_serve_weights_activate(plans_[0]));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_weights_await(plans_[0]));
  uint64_t retained = 0;
  EXPECT_FALSE(
      loom_serve_weights_query_activation_workspace(plans_[0], &retained));
  EXPECT_EQ(retained,
            loom_serve_weights_activation_workspace(plans_[0]).allocation_size);
  iree_hal_buffer_release(buffers_[0][0]);
  buffers_[0][0] = nullptr;
  IREE_ASSERT_OK(loom_serve_weights_await_retirement(plans_[0]));
  EXPECT_TRUE(loom_serve_weights_is_retired(plans_[0]));
  // Failed readiness is not a successful staging deallocation, even after
  // terminal payload cleanup. Normal consumers still receive the load error.
  EXPECT_FALSE(
      loom_serve_weights_query_activation_workspace(plans_[0], &retained));
  EXPECT_EQ(retained,
            loom_serve_weights_activation_workspace(plans_[0]).allocation_size);
  const auto before = loom_serve_weights_activity(plans_[0]);
  IREE_ASSERT_OK(loom_serve_weights_activate(plans_[0]));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_weights_await(plans_[0]));
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).activations,
            before.activations);
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).read_bytes,
            before.read_bytes);
}

TEST_P(WeightPackingTest,
       SourceMetadataRejectsWrongFactorDtypeBeforeCompilation) {
  WriteFactors(adapter_path_.path(), 100, 4);
  IREE_ASSERT_OK(Compile(0, "packed_target"));
  const auto root = Root(0, 0);
  iree::Status status(
      CreateSourceRoots(0, 1, &root, {{path_.path()}, {adapter_path_.path()}},
                        "packing_policy.loom"));
  EXPECT_THAT(status, iree::testing::status::StatusIs(
                          iree::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("expected two F32 factor checkpoints"));
  EXPECT_EQ(compilation_.commands, 1u);
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).read_bytes, 0u);
}

INSTANTIATE_TEST_SUITE_P(Backing, WeightPackingTest,
                         ::testing::Values(false, true));

}  // namespace loom_serve::testing
