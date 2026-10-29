// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/weights_test_base.h"
#include "loom/serve/testing/worker_gate.h"

namespace loom_serve::testing {

class WindowedWeightsTest : public WeightsTest {
 protected:
  uint64_t SlabSize() const override { return 2 * 1024 * 1024; }
  void SetUpPool() override {
    // The captured-source recipe packs two aligned halves into each of four
    // staging lanes. These overlap both final parameter slabs during loading.
    const uint64_t staging_bytes = 4 * 2 * iree_device_align(1048592, 64);
    const uint64_t memory_limit =
        2 * SlabSize() + iree_device_align(staging_bytes, SlabSize());
    ASSERT_NO_FATAL_FAILURE(AllocatePool(memory_limit));
  }
  void WriteWindowedCheckpoint(const std::string& path, int32_t offset = 0) {
    constexpr size_t kElements = 524296;
    const std::string header =
        "{\"a\":{\"dtype\":\"I32\",\"shape\":[8],\"data_offsets\":[0,32]},"
        "\"wide\":{\"dtype\":\"I32\",\"shape\":[524296],"
        "\"data_offsets\":[32,2097216]},"
        "\"b\":{\"dtype\":\"I32\",\"shape\":[8],"
        "\"data_offsets\":[2097216,2097248]}}";
    std::vector<int32_t> values(8 + kElements + 8);
    for (size_t i = 0; i < 8; ++i) {
      values[i] = offset + 10 + i;
      values[8 + kElements + i] = offset + 20 + i;
    }
    for (size_t i = 0; i < kElements; ++i) {
      values[8 + i] = offset + 1000 + i;
    }
    std::ofstream file(path, std::ios::binary);
    for (unsigned i = 0; i < 8; ++i) {
      file.put(
          static_cast<char>(static_cast<uint64_t>(header.size()) >> (8 * i)));
    }
    file.write(header.data(), header.size());
    file.write(reinterpret_cast<const char*>(values.data()),
               values.size() * sizeof(int32_t));
    file.close();
    ASSERT_TRUE(file.good());
  }

  void VerifyWindowedStorage(bool captured_inputs = false) {
    const auto* program = loom_serve_jit_stage_program(compiled_[0]);
    std::vector<int32_t> values(524296);
    for (uint32_t i = 0; i < program->parameters.count; ++i) {
      const auto parameter = loom_cmd_program_parameter_at(program, i);
      const iree_hal_transfer_operation_t download = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
          .download =
              {
                  .source_buffer = buffers_[0][0],
                  .source_offset = parameter.byte_offset,
                  .target = values.data(),
                  .length = parameter.byte_length,
              },
      };
      uint64_t completion = 0;
      IREE_ASSERT_OK(
          loom_serve_execution_feedback(execution_, 1, &download, &completion));
      IREE_ASSERT_OK(
          loom_serve_execution_feedback_wait(execution_, completion));
      const bool first = iree_string_view_equal(parameter.key, IREE_SV("a"));
      const bool second = iree_string_view_equal(parameter.key, IREE_SV("b"));
      const int32_t base = first ? 10 : second ? 20 : 1000;
      for (size_t j = 0; j < parameter.byte_length / sizeof(int32_t); ++j) {
        const int32_t prepared =
            captured_inputs ? (!first && !second && j >= 262148 ? 100000 : 0)
                            : (second ? 7 : 3);
        ASSERT_EQ(values[j], base + static_cast<int32_t>(j) + prepared)
            << "parameter " << i << " element " << j;
      }
    }
    Run(0, captured_inputs ? 625318 : 525331);
  }

  void Reload() {
    bool complete = false;
    IREE_ASSERT_OK(loom_serve_weights_deactivate(plans_[0], &complete));
    IREE_ASSERT_OK(loom_serve_memory_pool_await(pool_));
    if (!complete) {
      IREE_ASSERT_OK(loom_serve_weights_deactivate(plans_[0], &complete));
    }
    ASSERT_TRUE(complete);
    IREE_ASSERT_OK(loom_serve_weights_activate(plans_[0]));
    IREE_ASSERT_OK(loom_serve_weights_import_readiness(plans_[0], execution_));
    IREE_ASSERT_OK(loom_serve_weights_await(plans_[0]));
  }
};

TEST_P(WindowedWeightsTest, CrossSlabTensorPreparesOnceAndReloads) {
  ASSERT_NO_FATAL_FAILURE(WriteWindowedCheckpoint(path_.path()));
  IREE_ASSERT_OK(Compile(0, "windowed"));
  IREE_ASSERT_OK(Compile(1, "windowed"));
  IREE_ASSERT_OK(Load(2, 2));
  EXPECT_EQ(buffers_[0][0], buffers_[1][0]);
  ASSERT_NO_FATAL_FAILURE(VerifyWindowedStorage());
  const auto activity = loom_serve_weights_activity(plans_[0]);
  EXPECT_EQ(activity.read_bytes, 2097248u);
  EXPECT_EQ(activity.prepared_bytes, activity.read_bytes);
  EXPECT_EQ(activity.groups, GetParam() ? 2u : 3u);
  if (GetParam()) {
    ASSERT_NO_FATAL_FAILURE(Reload());
    ASSERT_NO_FATAL_FAILURE(VerifyWindowedStorage());
    EXPECT_EQ(loom_serve_weights_activity(plans_[0]).prepared_bytes,
              activity.prepared_bytes * 2);
  }
}

TEST_P(WindowedWeightsTest, CapturedSourcesWaitForEveryDestinationSlab) {
  ASSERT_NO_FATAL_FAILURE(WriteWindowedCheckpoint(path_.path()));
  ASSERT_NO_FATAL_FAILURE(
      WriteWindowedCheckpoint(adapter_path_.path(), 100000));
  IREE_ASSERT_OK(Compile(0, "windowed"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateSourceRoots(0, 1, &root,
                                   {{path_.path()}, {adapter_path_.path()}},
                                   "streaming_policy.loom"));
  IREE_ASSERT_OK(Register(0, 0, 1));
  IREE_ASSERT_OK(loom_serve_residency_acquire_wait(residencies_[0]));
  ASSERT_NO_FATAL_FAILURE(VerifyWindowedStorage(true));
  loom_serve_residency_release(residencies_[0]);
  const auto activity = loom_serve_weights_activity(plans_[0]);
  EXPECT_EQ(activity.read_bytes, 2097248u);
  EXPECT_EQ(activity.prepared_bytes, 2097184u);
  if (GetParam()) {
    ASSERT_NO_FATAL_FAILURE(Reload());
    ASSERT_NO_FATAL_FAILURE(VerifyWindowedStorage(true));
    EXPECT_EQ(loom_serve_weights_activity(plans_[0]).read_bytes,
              activity.read_bytes * 2);
  }
}

TEST_P(WindowedWeightsTest, CapturedInputSubmissionReturnsWithTheWorkerHeld) {
  ASSERT_NO_FATAL_FAILURE(WriteWindowedCheckpoint(path_.path()));
  ASSERT_NO_FATAL_FAILURE(
      WriteWindowedCheckpoint(adapter_path_.path(), 100000));
  IREE_ASSERT_OK(Compile(0, "windowed"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateSourceRoots(0, 1, &root,
                                   {{path_.path()}, {adapter_path_.path()}},
                                   "streaming_policy.loom"));
  auto* host = loom_serve_device_host(device_owner_);
  {
    WorkerGate gate(loom_serve_host_executor(host));
    IREE_ASSERT_OK(loom_serve_weights_activate(plans_[0]));
    IREE_ASSERT_OK(loom_serve_weights_import_readiness(plans_[0], execution_));
    EXPECT_FALSE(loom_serve_weights_is_retired(plans_[0]));
    EXPECT_EQ(loom_serve_weights_activity(plans_[0]).read_bytes, 2097248u);
  }
  IREE_ASSERT_OK(loom_serve_weights_await(plans_[0]));
  ASSERT_NO_FATAL_FAILURE(VerifyWindowedStorage(true));
}

class RejectedReadWeightsTest : public WindowedWeightsTest {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(WindowedWeightsTest::SetUp());
    const iree_hal_queue_params_t params = {
        .priority = transfer_->priority,
        .features = transfer_->features,
        .execution_resources = transfer_->execution_resources,
    };
    static const iree_hal_queue_vtable_t vtable = {
        .destroy = [](iree_hal_queue_t*) {},
        .read =
            [](iree_hal_queue_t*, iree_hal_semaphore_list_t,
               iree_hal_semaphore_list_t, iree_hal_file_t*, uint64_t,
               iree_hal_buffer_t*, iree_device_size_t, iree_device_size_t,
               const iree_hal_queue_barriers_t*, iree_hal_read_flags_t) {
              return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
            },
    };
    iree_hal_queue_initialize(iree_hal_queue_family(transfer_), &params,
                              &vtable, &rejected_transfer_);
    initialized_ = true;
  }

  void TearDown() override {
    WindowedWeightsTest::TearDown();
    if (initialized_) {
      iree_hal_queue_release(&rejected_transfer_);
    }
  }

  // Native file submission rejects after the cold host task was accepted.
  iree_hal_queue_t rejected_transfer_ = {};
  // The queue is released only after successful fixture initialization.
  bool initialized_ = false;
};

TEST_P(RejectedReadWeightsTest,
       FailedReadRetiresTheAcceptedProducerWithoutReplay) {
  ASSERT_NO_FATAL_FAILURE(WriteWindowedCheckpoint(path_.path()));
  IREE_ASSERT_OK(Compile(0, "windowed"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(loom_serve_weights_allocate(
      device_, &rejected_transfer_, dispatch_,
      loom_serve_pool_storage(storage_owner_), pool_,
      loom_serve_pool_weight_geometry(storage_owner_), nullptr,
      loom_serve_pool_workspace(storage_owner_), 0, 1, &root, host_allocator_,
      &plans_[0]));
  IREE_ASSERT_OK(Prepare(plans_[0], {path_.path()}));
  {
    WorkerGate gate(
        loom_serve_host_executor(loom_serve_device_host(device_owner_)));
    IREE_ASSERT_OK(loom_serve_weights_activate(plans_[0]));
    iree_hal_buffer_release(buffers_[0][0]);
    buffers_[0][0] = nullptr;
    EXPECT_FALSE(loom_serve_weights_is_retired(plans_[0]));
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_weights_await(plans_[0]));
  IREE_ASSERT_OK(loom_serve_weights_await_retirement(plans_[0]));
  EXPECT_TRUE(loom_serve_weights_is_retired(plans_[0]));
  const auto activity = loom_serve_weights_activity(plans_[0]);
  EXPECT_EQ(activity.activations, 1u);
  EXPECT_EQ(activity.read_bytes, 2097248u);
  IREE_ASSERT_OK(loom_serve_weights_activate(plans_[0]));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_weights_await(plans_[0]));
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).activations,
            activity.activations);
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).read_bytes,
            activity.read_bytes);
}

class CoarseWindowWeightsTest : public WindowedWeightsTest {
 protected:
  // Two populated 64 MiB read windows share one exact physical tail.
  static constexpr uint64_t kCoarseSlabBytes = 128 * 1024 * 1024;
  uint64_t SlabSize() const override { return kCoarseSlabBytes; }
};

TEST_P(CoarseWindowWeightsTest, ReadWindowsShareOnePendingSlabAndReload) {
  constexpr uint64_t kWindowBytes = 64 * 1024 * 1024;
  constexpr uint64_t kTensorBytes = kWindowBytes + 32;
  const std::string header =
      "{\"raw\":{\"dtype\":\"I32\",\"shape\":[16777224],"
      "\"data_offsets\":[0,67108896]}}";
  std::ofstream file(path_.path(), std::ios::binary);
  for (unsigned i = 0; i < 8; ++i) {
    file.put(
        static_cast<char>(static_cast<uint64_t>(header.size()) >> (8 * i)));
  }
  file.write(header.data(), header.size());
  for (int32_t i = 0; i < 8; ++i) {
    const int32_t value = 10 + i;
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
  }
  file.seekp(8 + header.size() + kWindowBytes);
  for (int32_t i = 0; i < 8; ++i) {
    const int32_t value = 30 + i;
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
  }
  file.close();
  ASSERT_TRUE(file.good());
  IREE_ASSERT_OK(Compile(0, "coarse_raw"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateRoots(0, 1, &root, {path_.path()}));
  const auto geometry = loom_serve_pool_weight_geometry(storage_owner_);
  const auto backing_bytes =
      iree_device_align(kTensorBytes, geometry.granularity);
  ASSERT_LT(backing_bytes, kCoarseSlabBytes);
  {
    ScopedNativeHold hold(device_owner_);
    IREE_ASSERT_OK(loom_serve_weights_activate(plans_[0]));
    IREE_ASSERT_OK(loom_serve_weights_import_readiness(plans_[0], execution_));
    hold.WaitEntered();
    EXPECT_FALSE(loom_serve_weights_is_retired(plans_[0]));
    EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).pending_bytes,
              backing_bytes);
  }
  IREE_ASSERT_OK(loom_serve_weights_await(plans_[0]));
  ASSERT_NO_FATAL_FAILURE(Run(0, 50));
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).read_bytes, kTensorBytes);
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).groups, 2u);
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes,
            backing_bytes);
  ASSERT_NO_FATAL_FAILURE(Reload());
  ASSERT_NO_FATAL_FAILURE(Run(0, 50));
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).read_bytes,
            2 * kTensorBytes);
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).groups, 4u);
}

INSTANTIATE_TEST_SUITE_P(Streaming, WindowedWeightsTest,
                         ::testing::Values(false, true));
INSTANTIATE_TEST_SUITE_P(Streaming, CoarseWindowWeightsTest,
                         ::testing::Values(true));
INSTANTIATE_TEST_SUITE_P(Streaming, RejectedReadWeightsTest,
                         ::testing::Values(true));

}  // namespace loom_serve::testing
