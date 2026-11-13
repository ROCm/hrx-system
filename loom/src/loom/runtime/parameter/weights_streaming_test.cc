// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>

#include "loom/runtime/parameter/telemetry.h"
#include "loom/runtime/parameter/weights_test_base.h"
#include "loom/runtime/testing/worker_gate.h"

namespace loom_runtime::testing {

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
    const auto* program = loom_runtime_jit_stage_program(compiled_[0]);
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
      IREE_ASSERT_OK(loom_runtime_execution_feedback(execution_, 1, &download,
                                                     &completion));
      IREE_ASSERT_OK(
          loom_runtime_execution_feedback_wait(execution_, completion));
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
    IREE_ASSERT_OK(loom_runtime_weights_deactivate(plans_[0], &complete));
    IREE_ASSERT_OK(loom_runtime_memory_pool_await(pool_));
    if (!complete) {
      IREE_ASSERT_OK(loom_runtime_weights_deactivate(plans_[0], &complete));
    }
    ASSERT_TRUE(complete);
    IREE_ASSERT_OK(loom_runtime_weights_activate(plans_[0]));
    IREE_ASSERT_OK(
        loom_runtime_weights_import_readiness(plans_[0], execution_));
    IREE_ASSERT_OK(loom_runtime_weights_await(plans_[0]));
  }
};

TEST_P(WindowedWeightsTest, CrossSlabTensorPreparesOnceAndReloads) {
  ASSERT_NO_FATAL_FAILURE(WriteWindowedCheckpoint(path_.path()));
  IREE_ASSERT_OK(Compile(0, "windowed"));
  IREE_ASSERT_OK(Compile(1, "windowed"));
  IREE_ASSERT_OK(Load(2, 2));
  EXPECT_EQ(buffers_[0][0], buffers_[1][0]);
  ASSERT_NO_FATAL_FAILURE(VerifyWindowedStorage());
  const auto activity = loom_runtime_weights_activity(plans_[0]);
  EXPECT_EQ(activity.read_bytes, 2097248u);
  EXPECT_EQ(activity.prepared_bytes, activity.read_bytes);
  EXPECT_EQ(activity.groups, GetParam() ? 2u : 3u);
  loom_runtime_telemetry_stream_t* stream = nullptr;
  IREE_ASSERT_OK(loom_runtime_telemetry_stream_allocate(
      2, iree_allocator_system(), &stream));
  loom_runtime_parameter_record_activity(stream, 1, &activity);
  if (GetParam()) {
    ASSERT_NO_FATAL_FAILURE(Reload());
    ASSERT_NO_FATAL_FAILURE(VerifyWindowedStorage());
    EXPECT_EQ(loom_runtime_weights_activity(plans_[0]).prepared_bytes,
              activity.prepared_bytes * 2);
  }
  const auto completed = loom_runtime_weights_activity(plans_[0]);
  loom_runtime_parameter_record_activity(stream, 2, &completed);
  std::array<loom_runtime_telemetry_record_t, 2> records;
  uint64_t cursor = 0;
  for (auto& record : records) {
    ASSERT_EQ(loom_runtime_telemetry_stream_read(stream, &cursor, &record),
              LOOM_RUNTIME_TELEMETRY_READ_RECORD);
  }
  loom_runtime_telemetry_stream_free(stream);
  // Retire the actual plans, commands, buffers, and storage owners before
  // interpreting observations. The fixture's outer cleanup is idempotent.
  TearDown();
  for (const auto& record : records) {
    auto expected = record.time_ns == 1 ? activity : completed;
    IREE_ASSERT_OK(record.event->read_fields(
        &record,
        {&expected, [](void* self, iree_host_size_t count,
                       const loom_runtime_telemetry_field_t* fields) {
           const auto& expected =
               *static_cast<const loom_runtime_parameter_activity_t*>(self);
           EXPECT_EQ(count, 5u);
           if (count == 5) {
             EXPECT_EQ(fields[0].value.uint64_value, expected.activations);
             EXPECT_EQ(fields[1].value.uint64_value, expected.deactivations);
             EXPECT_EQ(fields[2].value.uint64_value, expected.read_bytes);
             EXPECT_EQ(fields[3].value.uint64_value, expected.prepared_bytes);
             EXPECT_EQ(fields[4].value.uint64_value, expected.groups);
           }
           return iree_ok_status();
         }}));
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
  IREE_ASSERT_OK(loom_runtime_residency_acquire_wait(residencies_[0]));
  ASSERT_NO_FATAL_FAILURE(VerifyWindowedStorage(true));
  loom_runtime_residency_release(residencies_[0]);
  const auto activity = loom_runtime_weights_activity(plans_[0]);
  EXPECT_EQ(activity.read_bytes, 2097248u);
  EXPECT_EQ(activity.prepared_bytes, 2097184u);
  if (GetParam()) {
    ASSERT_NO_FATAL_FAILURE(Reload());
    ASSERT_NO_FATAL_FAILURE(VerifyWindowedStorage(true));
    EXPECT_EQ(loom_runtime_weights_activity(plans_[0]).read_bytes,
              activity.read_bytes * 2);
  }
}

TEST_P(WindowedWeightsTest, DirectSourcesAndHostSnapshotsNeedNoStaging) {
  ASSERT_NO_FATAL_FAILURE(WriteWindowedCheckpoint(path_.path()));
  ASSERT_NO_FATAL_FAILURE(
      WriteWindowedCheckpoint(adapter_path_.path(), 100000));
  IREE_ASSERT_OK(Compile(0, "windowed"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateSourceRoots(0, 1, &root,
                                   {{path_.path()}, {adapter_path_.path()}},
                                   "direct_policy.loom"));
  EXPECT_EQ(
      loom_runtime_weights_activation_workspace(plans_[0]).allocation_size, 0u);
  IREE_ASSERT_OK(Register(0, 0, 1));
  IREE_ASSERT_OK(loom_runtime_residency_acquire_wait(residencies_[0]));
  ASSERT_NO_FATAL_FAILURE(VerifyWindowedStorage(true));
  loom_runtime_residency_release(residencies_[0]);
  const auto activity = loom_runtime_weights_activity(plans_[0]);
  EXPECT_EQ(activity.read_bytes, 2097248u);
  EXPECT_EQ(activity.prepared_bytes, 0u);
  // Each captured half spans five bounded gathers. The host snapshot adds
  // another; canonical input shares the first source's partially filled gather.
  EXPECT_EQ(activity.groups, 11u);
  if (GetParam()) {
    ASSERT_NO_FATAL_FAILURE(Reload());
    ASSERT_NO_FATAL_FAILURE(VerifyWindowedStorage(true));
    EXPECT_EQ(loom_runtime_weights_activity(plans_[0]).read_bytes,
              activity.read_bytes * 2);
  }
}

TEST_P(WindowedWeightsTest, IncompleteDirectRecipesRejectBeforeSubmission) {
  size_t index = 0;
  for (const char* command :
       {"direct_gap", "direct_short", "direct_incomplete"}) {
    SCOPED_TRACE(command);
    IREE_ASSERT_OK(Compile(index, command));
    const auto root = Root(index, 0);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        CreateRoots(0, 1, &root, {path_.path()}, "direct_invalid_policy.loom"));
    EXPECT_EQ(loom_runtime_weights_activity(plans_[plan_count_ - 1]).read_bytes,
              0u);
    ++index;
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
  auto* host = loom_runtime_device_host(device_owner_);
  {
    WorkerGate gate(loom_runtime_host_executor(host));
    IREE_ASSERT_OK(loom_runtime_weights_activate(plans_[0]));
    IREE_ASSERT_OK(
        loom_runtime_weights_import_readiness(plans_[0], execution_));
    EXPECT_FALSE(loom_runtime_weights_is_retired(plans_[0]));
    EXPECT_EQ(loom_runtime_weights_activity(plans_[0]).read_bytes, 2097248u);
  }
  IREE_ASSERT_OK(loom_runtime_weights_await(plans_[0]));
  ASSERT_NO_FATAL_FAILURE(VerifyWindowedStorage(true));
}

class ReadQueueWeightsTest : public WindowedWeightsTest {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(WindowedWeightsTest::SetUp());
    const iree_hal_queue_params_t params = {
        .priority = transfer_->priority,
        .features = transfer_->features,
        .execution_resources = transfer_->execution_resources,
    };
    read_queue_.target = transfer_;
    static const iree_hal_queue_vtable_t vtable = {
        .destroy = [](iree_hal_queue_t*) {},
        .barrier =
            [](iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
               iree_hal_semaphore_list_t signals,
               const iree_hal_queue_barriers_t* barriers,
               iree_hal_queue_barrier_flags_t flags) {
              auto* self = reinterpret_cast<ReadQueue*>(queue);
              return iree_hal_queue_barrier(self->target, waits, signals,
                                            barriers, flags);
            },
        .transfer =
            [](iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
               iree_hal_semaphore_list_t signals,
               iree_host_size_t operation_count,
               const iree_hal_transfer_operation_t* operations,
               const iree_hal_queue_barriers_t* barriers) {
              auto* self = reinterpret_cast<ReadQueue*>(queue);
              return iree_hal_queue_transfer(self->target, waits, signals,
                                             operation_count, operations,
                                             barriers);
            },
        .read =
            [](iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
               iree_hal_semaphore_list_t signals, iree_hal_file_t* file,
               uint64_t offset, iree_hal_buffer_t* buffer,
               iree_device_size_t buffer_offset, iree_device_size_t length,
               const iree_hal_queue_barriers_t* barriers,
               iree_hal_read_flags_t flags) {
              auto* self = reinterpret_cast<ReadQueue*>(queue);
              if (++self->count == self->reject_ordinal) {
                return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
              }
              {
                std::lock_guard<std::mutex> lock(self->mutex);
                if (self->retain_read && !self->retained) {
                  iree_hal_buffer_retain(buffer);
                  self->retained = buffer;
                }
              }
              return iree_hal_queue_read(self->target, waits, signals, file,
                                         offset, buffer, buffer_offset, length,
                                         barriers, flags);
            },
    };
    iree_hal_queue_initialize(iree_hal_queue_family(transfer_), &params,
                              &vtable, &read_queue_.base);
    initialized_ = true;
  }

  void TearDown() override {
    ReleaseRead();
    if (storage_owner_) {
      loom_runtime_residency_cache_set_observer(ResidencyCache(), {});
      loom_runtime_weights_set_retirement_observer(plans_[0], nullptr, nullptr);
    }
    WindowedWeightsTest::TearDown();
    if (initialized_) {
      iree_hal_queue_release(&read_queue_.base);
    }
  }

  void VerifyRejectedRead(const char* policy, size_t reject_ordinal);
  void PrepareRetainedRead();

  void ReleaseRead() {
    iree_hal_buffer_t* retained = nullptr;
    {
      std::lock_guard<std::mutex> lock(read_queue_.mutex);
      read_queue_.retain_read = false;
      retained = read_queue_.retained;
      read_queue_.retained = nullptr;
    }
    iree_hal_buffer_release(retained);
  }

  struct ReadQueue {
    // Queue prefix for callbacks forwarding to the real transfer
    // implementation.
    iree_hal_queue_t base = {};
    // Borrowed fixture-owned queue, valid until every read has retired.
    iree_hal_queue_t* target = nullptr;
    // Observed read submissions from independently progressing lane tasks.
    std::atomic<size_t> count = 0;
    // One native submission to reject; all earlier reads execute normally.
    size_t reject_ordinal = 0;
    // Serializes final capture release with concurrent read submission.
    std::mutex mutex;
    // Retains one native read capture beyond successful device readiness.
    bool retain_read = false;
    // Owned read view, released explicitly or during fixture cleanup.
    iree_hal_buffer_t* retained = nullptr;
  };
  // Real file-submission dependency with controlled rejection or retirement.
  ReadQueue read_queue_;
  // Admission-owner notification, independent of final model removal.
  loom_runtime_test_weight_activation_wake_t capacity_wake_;
  // Model-owner notification whose detachment must preserve capacity wakes.
  loom_runtime_test_weight_activation_wake_t retirement_wake_;
  // The queue is released only after successful fixture initialization.
  bool initialized_ = false;
};

void ReadQueueWeightsTest::VerifyRejectedRead(const char* policy,
                                              size_t reject_ordinal) {
  read_queue_.reject_ordinal = reject_ordinal;
  ASSERT_NO_FATAL_FAILURE(WriteWindowedCheckpoint(path_.path()));
  ASSERT_NO_FATAL_FAILURE(
      WriteWindowedCheckpoint(adapter_path_.path(), 100000));
  IREE_ASSERT_OK(Compile(0, "windowed"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(loom_runtime_weights_allocate(
      device_, &read_queue_.base, dispatch_,
      loom_runtime_pool_storage(storage_owner_), pool_,
      loom_runtime_pool_weight_geometry(storage_owner_), nullptr,
      loom_runtime_pool_workspace(storage_owner_), 0, 1, &root, host_allocator_,
      &plans_[0]));
  IREE_ASSERT_OK(PrepareSources(
      plans_[0], {{path_.path()}, {adapter_path_.path()}}, policy));
  {
    WorkerGate gate(
        loom_runtime_host_executor(loom_runtime_device_host(device_owner_)));
    IREE_ASSERT_OK(loom_runtime_weights_activate(plans_[0]));
    iree_hal_buffer_release(buffers_[0][0]);
    buffers_[0][0] = nullptr;
    EXPECT_FALSE(loom_runtime_weights_is_retired(plans_[0]));
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_runtime_weights_await(plans_[0]));
  IREE_ASSERT_OK(loom_runtime_weights_await_retirement(plans_[0]));
  EXPECT_TRUE(loom_runtime_weights_is_retired(plans_[0]));
  EXPECT_GE(read_queue_.count.load(), reject_ordinal);
  const auto activity = loom_runtime_weights_activity(plans_[0]);
  EXPECT_EQ(activity.activations, 1u);
  EXPECT_EQ(activity.read_bytes, 2097248u);
  IREE_ASSERT_OK(loom_runtime_weights_activate(plans_[0]));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_runtime_weights_await(plans_[0]));
  EXPECT_EQ(loom_runtime_weights_activity(plans_[0]).activations,
            activity.activations);
  EXPECT_EQ(loom_runtime_weights_activity(plans_[0]).read_bytes,
            activity.read_bytes);
  IREE_ASSERT_OK(loom_runtime_weights_deactivate_wait(plans_[0]));
  EXPECT_EQ(loom_runtime_weights_memory_usage(plans_[0]).committed_bytes, 0u);
}

TEST_P(ReadQueueWeightsTest,
       FailedReadRetiresTheAcceptedProducerWithoutReplay) {
  ASSERT_NO_FATAL_FAILURE(VerifyRejectedRead("direct_policy.loom", 1));
}

TEST_P(ReadQueueWeightsTest,
       RefillFailureRetiresAcceptedInputWindowsWithoutReplay) {
  ASSERT_NO_FATAL_FAILURE(VerifyRejectedRead("streaming_policy.loom", 257));
}

void ReadQueueWeightsTest::PrepareRetainedRead() {
  ASSERT_NO_FATAL_FAILURE(WriteWindowedCheckpoint(path_.path()));
  IREE_ASSERT_OK(Compile(0, "windowed"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(loom_runtime_weights_allocate(
      device_, &read_queue_.base, dispatch_,
      loom_runtime_pool_storage(storage_owner_), pool_,
      loom_runtime_pool_weight_geometry(storage_owner_), nullptr,
      loom_runtime_pool_workspace(storage_owner_), 0, 1, &root, host_allocator_,
      &plans_[0]));
  IREE_ASSERT_OK(Prepare(plans_[0], {path_.path()}));
  plan_count_ = 1;
  IREE_ASSERT_OK(Register(0, 0, 1));
}

TEST_P(ReadQueueWeightsTest, ReadinessDoesNotReleaseActivationCaptures) {
  ASSERT_NO_FATAL_FAILURE(PrepareRetainedRead());
  loom_runtime_residency_cache_set_observer(
      ResidencyCache(),
      {loom_runtime_test_weight_activation_wake_t::Notify, &capacity_wake_});
  for (bool observe_model : {true, false}) {
    SCOPED_TRACE(observe_model);
    {
      std::lock_guard<std::mutex> lock(read_queue_.mutex);
      read_queue_.retain_read = true;
    }
    IREE_ASSERT_OK(loom_runtime_residency_acquire(residencies_[0]));
    ASSERT_NO_FATAL_FAILURE(Run(0, 525331));
    loom_runtime_residency_release(residencies_[0]);
    if (observe_model) {
      loom_runtime_weights_set_retirement_observer(
          plans_[0], loom_runtime_test_weight_activation_wake_t::Notify,
          &retirement_wake_);
    }
    const auto before = loom_runtime_weights_memory_usage(plans_[0]);
    const auto activity = loom_runtime_weights_activity(plans_[0]);
    bool complete = true;
    IREE_ASSERT_OK(
        loom_runtime_residency_deactivate(residencies_[0], &complete));
    EXPECT_FALSE(complete);
    EXPECT_EQ(loom_runtime_weights_activity(plans_[0]).deactivations,
              activity.deactivations);
    const auto pending = loom_runtime_weights_memory_usage(plans_[0]);
    EXPECT_EQ(pending.committed_bytes + pending.pending_bytes,
              before.committed_bytes + before.pending_bytes);
    EXPECT_EQ(pending.releasing_bytes, 0u);
    loom_runtime_residency_admission_t admission;
    IREE_ASSERT_OK(
        loom_runtime_residency_try_admit(residencies_[0], &admission));
    EXPECT_EQ(admission, LOOM_RUNTIME_RESIDENCY_ADMISSION_DEFERRED);
    ReleaseRead();
    capacity_wake_.Await();
    if (observe_model) {
      retirement_wake_.Await();
      loom_runtime_weights_set_retirement_observer(plans_[0], nullptr, nullptr);
    }
    // The wake advances the existing release interest without a loader wait.
    IREE_ASSERT_OK(loom_runtime_residency_cache_query_status(ResidencyCache()));
    IREE_ASSERT_OK(loom_runtime_memory_pool_await(pool_));
    IREE_ASSERT_OK(
        loom_runtime_residency_deactivate(residencies_[0], &complete));
    ASSERT_TRUE(complete);
    EXPECT_EQ(loom_runtime_weights_memory_usage(plans_[0]).committed_bytes, 0u);
  }
  // Cached roots and command recordings survived both evictions.
  ASSERT_NO_FATAL_FAILURE(RunPinned(0, 525331));
}

TEST_P(ReadQueueWeightsTest, ModelRemovalPreservesPendingEvictionWake) {
  ASSERT_NO_FATAL_FAILURE(PrepareRetainedRead());
  read_queue_.retain_read = true;
  loom_runtime_residency_cache_set_observer(
      ResidencyCache(),
      {loom_runtime_test_weight_activation_wake_t::Notify, &capacity_wake_});
  IREE_ASSERT_OK(loom_runtime_residency_acquire(residencies_[0]));
  ASSERT_NO_FATAL_FAILURE(Run(0, 525331));
  loom_runtime_residency_release(residencies_[0]);
  bool complete = true;
  IREE_ASSERT_OK(loom_runtime_residency_deactivate(residencies_[0], &complete));
  EXPECT_FALSE(complete);
  loom_runtime_weights_set_retirement_observer(
      plans_[0], loom_runtime_test_weight_activation_wake_t::Notify,
      &retirement_wake_);
  iree_hal_command_buffer_release(commands_[0]);
  commands_[0] = nullptr;
  iree_hal_buffer_release(buffers_[0][0]);
  buffers_[0][0] = nullptr;
  EXPECT_FALSE(loom_runtime_weights_is_retired(plans_[0]));
  ReleaseRead();
  capacity_wake_.Await();
  while (!loom_runtime_weights_is_retired(plans_[0])) {
    retirement_wake_.Await();
  }
  IREE_ASSERT_OK(loom_runtime_residency_cache_query_status(ResidencyCache()));
  IREE_ASSERT_OK(loom_runtime_memory_pool_await(pool_));
  IREE_ASSERT_OK(loom_runtime_residency_deactivate(residencies_[0], &complete));
  ASSERT_TRUE(complete);
  EXPECT_TRUE(loom_runtime_weights_is_retired(plans_[0]));
}

TEST_P(ReadQueueWeightsTest, PendingActivationReleasePreservesResidentPeer) {
  ASSERT_NO_FATAL_FAILURE(PrepareRetainedRead());
  IREE_ASSERT_OK(Compile(1, "windowed"));
  const auto root = Root(1, 0);
  IREE_ASSERT_OK(CreateRoots(0, 1, &root, {path_.path()}));
  IREE_ASSERT_OK(Register(1, 1, 1));
  read_queue_.retain_read = true;
  loom_runtime_residency_cache_set_observer(
      ResidencyCache(),
      {loom_runtime_test_weight_activation_wake_t::Notify, &capacity_wake_});
  IREE_ASSERT_OK(loom_runtime_residency_acquire(residencies_[0]));
  ASSERT_NO_FATAL_FAILURE(Run(0, 525331));
  loom_runtime_residency_release(residencies_[0]);
  ASSERT_NO_FATAL_FAILURE(RunPinned(1, 525331));
  const auto peer = loom_runtime_weights_memory_usage(plans_[1]);
  const auto memory = loom_runtime_memory_pool_statistics(pool_);
  const auto first = loom_runtime_weights_memory_usage(plans_[0]);
  const uint64_t target = memory.committed_bytes + memory.pending_bytes -
                          first.committed_bytes - first.pending_bytes;
  IREE_ASSERT_OK(loom_runtime_residency_cache_trim(ResidencyCache(), target));
  // The first requested release covers the target even before native unmapping
  // can begin. A second observation must not select another useful model.
  IREE_ASSERT_OK(loom_runtime_residency_cache_trim(ResidencyCache(), target));
  EXPECT_EQ(loom_runtime_weights_activity(plans_[1]).deactivations, 0u);
  EXPECT_EQ(loom_runtime_weights_memory_usage(plans_[1]).committed_bytes,
            peer.committed_bytes);
  ReleaseRead();
  capacity_wake_.Await();
  IREE_ASSERT_OK(loom_runtime_residency_cache_query_status(ResidencyCache()));
  IREE_ASSERT_OK(loom_runtime_memory_pool_await(pool_));
  IREE_ASSERT_OK(loom_runtime_residency_cache_query_status(ResidencyCache()));
  EXPECT_EQ(loom_runtime_weights_memory_usage(plans_[0]).committed_bytes, 0u);
  ASSERT_NO_FATAL_FAILURE(RunPinned(1, 525331));
  EXPECT_EQ(loom_runtime_weights_activity(plans_[1]).activations, 1u);
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
  const auto geometry = loom_runtime_pool_weight_geometry(storage_owner_);
  const auto backing_bytes =
      iree_device_align(kTensorBytes, geometry.granularity);
  ASSERT_LT(backing_bytes, kCoarseSlabBytes);
  {
    ScopedNativeHold hold(device_owner_);
    IREE_ASSERT_OK(loom_runtime_weights_activate(plans_[0]));
    IREE_ASSERT_OK(
        loom_runtime_weights_import_readiness(plans_[0], execution_));
    hold.WaitEntered();
    EXPECT_FALSE(loom_runtime_weights_is_retired(plans_[0]));
    EXPECT_EQ(loom_runtime_weights_statistics(plans_[0]).pending_bytes,
              backing_bytes);
  }
  IREE_ASSERT_OK(loom_runtime_weights_await(plans_[0]));
  ASSERT_NO_FATAL_FAILURE(Run(0, 50));
  EXPECT_EQ(loom_runtime_weights_activity(plans_[0]).read_bytes, kTensorBytes);
  EXPECT_EQ(loom_runtime_weights_activity(plans_[0]).groups, 2u);
  EXPECT_EQ(loom_runtime_weights_statistics(plans_[0]).committed_bytes,
            backing_bytes);
  ASSERT_NO_FATAL_FAILURE(Reload());
  ASSERT_NO_FATAL_FAILURE(Run(0, 50));
  EXPECT_EQ(loom_runtime_weights_activity(plans_[0]).read_bytes,
            2 * kTensorBytes);
  EXPECT_EQ(loom_runtime_weights_activity(plans_[0]).groups, 4u);
}

INSTANTIATE_TEST_SUITE_P(Streaming, WindowedWeightsTest,
                         ::testing::Values(false, true));
INSTANTIATE_TEST_SUITE_P(Streaming, CoarseWindowWeightsTest,
                         ::testing::Values(true));
INSTANTIATE_TEST_SUITE_P(Streaming, ReadQueueWeightsTest,
                         ::testing::Values(true));

}  // namespace loom_runtime::testing
