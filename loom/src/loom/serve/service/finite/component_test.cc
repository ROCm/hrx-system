// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/finite/component.h"

#include <array>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <utility>

#include "iree/task/executor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/serve/service/finite/result.h"
#include "loom/serve/testing/device.h"
#include "loom/serve/testing/service_application.h"

namespace {

// Actual serialized owner on the same executor as preparation. Only the outer
// test waits, and it waits for process release rather than merely completion.
class ComponentOwner {
 public:
  loom_serve_wake_callback_t observer() { return {Wake, this}; }

  iree_status_t Run(loom_serve_device_t* device,
                    std::function<iree_status_t(bool*)> advance) {
    executor_ = loom_serve_host_executor(loom_serve_device_host(device));
    pool_ = loom_serve_device_memory_pool(device);
    advance_ = std::move(advance);
    complete_ = false;
    released_ = false;
    iree_task_process_initialize(Drain, 0, 1, &process_);
    process_.user_data = this;
    process_.release_fn = Release;
    if (pool_) {
      loom_serve_memory_pool_set_observer(pool_, Wake, this);
    }
    Wake(this);
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return released_; });
    return std::exchange(status_, iree_ok_status());
  }

 private:
  static void Wake(void* user_data) {
    auto* owner = static_cast<ComponentOwner*>(user_data);
    iree_task_executor_schedule_process(owner->executor_, &owner->process_);
  }

  static iree_status_t Drain(iree_task_process_t* process,
                             const iree_task_worker_context_t* context,
                             iree_task_process_drain_result_t* result) {
    auto* owner = static_cast<ComponentOwner*>(process->user_data);
    if (!owner->complete_) {
      owner->status_ = owner->advance_(&owner->complete_);
      owner->complete_ |= !iree_status_is_ok(owner->status_);
    }
    const bool pending =
        owner->pool_ && loom_serve_memory_pool_is_pending(owner->pool_);
    result->completed = owner->complete_ && !pending;
    if (result->completed && owner->pool_) {
      owner->status_ = iree_status_join(
          owner->status_, loom_serve_memory_pool_query_status(owner->pool_));
      loom_serve_memory_pool_set_observer(owner->pool_, nullptr, nullptr);
    }
    return iree_ok_status();
  }

  static void Release(iree_task_process_t* process) {
    auto* owner = static_cast<ComponentOwner*>(process->user_data);
    std::lock_guard<std::mutex> lock(owner->mutex_);
    owner->released_ = true;
    owner->condition_.notify_one();
  }

  // Borrowed executor, alive through the final owner turn.
  iree_task_executor_t* executor_ = nullptr;
  // Borrowed native-capacity owner, joined before releasing its wake target.
  loom_serve_memory_pool_t* pool_ = nullptr;
  // Current serialized operation, retaining its captures through release.
  std::function<iree_status_t(bool*)> advance_;
  // Terminal outcome returned only after the executor releases the process.
  iree_status_t status_ = iree_ok_status();
  // Intrusive owner process, never reused before Release finishes.
  iree_task_process_t process_ = {};
  // The requested operation finished; accepted native work may still retire.
  bool complete_ = false;
  // Joins the executor's final access to the process.
  std::mutex mutex_;
  // Outer-thread readiness notification, not a worker wait.
  std::condition_variable condition_;
  // Published under mutex_ once process storage is no longer borrowed.
  bool released_ = false;
};

class FiniteComponentTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto application_options = loom_serve_test_application_options_default();
    application_options.slab_size = 65536;
    loom_serve_test_application_t* application = nullptr;
    IREE_ASSERT_OK(loom_serve_test_application_acquire(
        &application_options, host_allocator_, &application));
    device_ = loom_serve_test_application_device_at(application, 0);
    directory_ = iree::testing::MakeTempFilePath("finite_component");
    ASSERT_TRUE(std::filesystem::create_directory(directory_));
    const char* source_path = std::getenv("LOOM_SERVE_TEST_COMPONENT_SOURCE");
    ASSERT_NE(source_path, nullptr);
    const auto source_directory =
        std::filesystem::path(source_path).parent_path();
    for (const char* file : {"prepare.loom", "control.loom", "kernels.loom",
                             "copy.loom", "policy.loom", "demand_prepare.loom",
                             "demand_control.loom", "demand_kernels.loom"}) {
      ASSERT_TRUE(std::filesystem::copy_file(
          source_directory / file, std::filesystem::path(directory_) / file));
    }
    ASSERT_NO_FATAL_FAILURE(WriteCheckpoint("weights.safetensors", 1.0f));
    ASSERT_NO_FATAL_FAILURE(WriteCheckpoint("second.safetensors", 3.0f));
    options_ = {
        .sources =
            {
                .base_directory = iree_make_cstring_view(directory_.c_str()),
                .prepare = IREE_SV("prepare.loom"),
                .control = IREE_SV("control.loom"),
                .count = IREE_ARRAYSIZE(providers_),
                .values = providers_,
            },
        .asset_directory = IREE_SV("."),
        .options = IREE_SV("{}"),
    };
    IREE_ASSERT_OK(loom_serve_finite_component_allocate(
        device_, &options_, owner_.observer(), host_allocator_, &component_));
  }

  void TearDown() override {
    if (consumer_) {
      IREE_EXPECT_OK(loom_serve_execution_drain(consumer_));
      loom_serve_execution_release(consumer_);
    }
    for (auto* result : results_) {
      loom_serve_finite_result_release(result);
    }
    if (component_) {
      IREE_EXPECT_OK(Retire(component_));
      loom_serve_finite_component_free(component_);
    }
  }

  iree_status_t Prepare(loom_serve_finite_component_t* component,
                        loom_serve_finite_model_t** out_model) {
    return owner_.Run(device_, [&](bool* complete) {
      iree_status_t status =
          loom_serve_finite_component_prepare(component, out_model);
      *complete = *out_model != nullptr;
      return status;
    });
  }

  void WriteCheckpoint(const char* name, float factor) {
    std::string header =
        R"({"factor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
    header.append((8 - header.size() % 8) % 8, ' ');
    std::ofstream checkpoint(directory_ + "/" + name, std::ios::binary);
    uint8_t bytes[8];
    iree_unaligned_store_le_u64(bytes, header.size());
    checkpoint.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    checkpoint.write(header.data(), header.size());
    iree_unaligned_store_le_f32(bytes, factor);
    checkpoint.write(reinterpret_cast<const char*>(bytes), sizeof(factor));
    checkpoint.close();
    ASSERT_TRUE(checkpoint.good());
  }

  void UseDemandSources() {
    IREE_ASSERT_OK(Retire(component_));
    loom_serve_finite_component_free(std::exchange(component_, nullptr));
    options_.sources.prepare = IREE_SV("demand_prepare.loom");
    options_.sources.control = IREE_SV("demand_control.loom");
    options_.sources.count = 1;
    providers_[0] = IREE_SV("demand_kernels.loom");
    IREE_ASSERT_OK(loom_serve_finite_component_allocate(
        device_, &options_, owner_.observer(), host_allocator_, &component_));
  }

  iree_status_t Retire(loom_serve_finite_component_t* component) {
    return owner_.Run(device_, [&](bool* complete) {
      return loom_serve_finite_component_retire(component, complete);
    });
  }

  iree_status_t Trim() {
    return owner_.Run(device_, [&](bool* complete) {
      return loom_serve_finite_component_trim(component_, complete);
    });
  }

  void Submit(loom_serve_finite_model_t* model, size_t ordinal,
              bool reject_after_gather = false) {
    iree_vm_buffer_t* input = nullptr;
    IREE_ASSERT_OK(iree_vm_buffer_create(16, 16, host_allocator_, &input));
    iree_byte_span_t bytes;
    IREE_ASSERT_OK(iree_vm_buffer_map_write(input, 0, 16, &bytes));
    for (uint32_t i = 0; i < 4; ++i) {
      iree_unaligned_store_le_f32(bytes.data + i * 4,
                                  static_cast<float>(i + 1 + ordinal * 4) *
                                      (reject_after_gather ? -1.0f : 1.0f));
    }
    loom_serve_prepared_request_t prepared = {};
    IREE_ASSERT_OK(
        loom_serve_finite_model_prepare_request(model, input, &prepared));
    const iree_host_size_t rows = 1;
    IREE_ASSERT_OK(
        loom_serve_finite_model_prepare_batch(model, 1, &prepared, &rows));
    loom_serve_finite_model_set_observer(model, owner_.observer());
    IREE_ASSERT_OK(owner_.Run(device_, [&](bool* complete) {
      return loom_serve_finite_model_advance_batch(model, false, complete);
    }));
    iree_vm_buffer_release(prepared.bytes);
    iree_vm_buffer_release(input);
    loom_serve_residency_admission_t admission;
    IREE_ASSERT_OK(loom_serve_residency_try_admit(
        loom_serve_finite_model_residency(model), &admission));
    ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
    loom_serve_execution_frontier_t frontier;
    if (reject_after_gather) {
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_INVALID_ARGUMENT,
          loom_serve_finite_model_submit(model, &results_[ordinal], &frontier));
      EXPECT_EQ(results_[ordinal], nullptr);
      return;
    }
    IREE_ASSERT_OK(
        loom_serve_finite_model_submit(model, &results_[ordinal], &frontier));
    EXPECT_EQ(frontier.values[1], 0u);
    // Only this outer application thread joins device completion.
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        {2, frontier.semaphores, frontier.values}, iree_infinite_timeout(),
        IREE_ASYNC_WAIT_FLAG_NONE));
    loom_serve_finite_model_complete(model);
  }

  void CheckResult(size_t ordinal, float factor = 2.0f) {
    if (!consumer_) {
      IREE_ASSERT_OK(loom_serve_device_create_execution(device_, &consumer_));
    }
    auto ready = loom_serve_finite_result_ready(results_[ordinal]);
    IREE_ASSERT_OK(loom_serve_execution_barrier(
        consumer_, {2, ready.semaphores, ready.values}));
    const auto row = loom_serve_finite_result_row(results_[ordinal], 0, 0);
    ASSERT_EQ(row.length, sizeof(output_));
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = row.buffer,
                .source_offset = row.offset,
                .target = output_.data(),
                .length = row.length,
            },
    };
    uint64_t value = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_transfer(consumer_, 1, &download, &value));
    IREE_ASSERT_OK(loom_serve_execution_drain(consumer_));
    for (uint32_t i = 0; i < 4; ++i) {
      EXPECT_EQ(output_[i], static_cast<float>(i + 1 + ordinal * 4) * factor);
    }
  }

  void Invoke(loom_serve_finite_model_t* model) {
    ASSERT_NO_FATAL_FAILURE(Submit(model, 0));
    ASSERT_NO_FATAL_FAILURE(CheckResult(0));
    loom_serve_finite_result_release(std::exchange(results_[0], nullptr));
  }

  // Allocation policy for this real source/JIT/device fixture.
  iree_allocator_t host_allocator_ = iree_allocator_system();
  // Borrowed physical owner shared by both tested components.
  loom_serve_device_t* device_ = nullptr;
  // Real shared-executor owner for cold lifecycle operations.
  ComponentOwner owner_;
  // Stable path backing all borrowed source/parameter locators.
  std::string directory_;
  // Native providers borrowed by options_.sources, including the physical
  // entry.
  iree_string_view_t providers_[2] = {IREE_SV("kernels.loom"),
                                      IREE_SV("copy.loom")};
  // Stable borrowed definition through final component destruction.
  loom_serve_finite_model_options_t options_ = {};
  // Owned optional model handle under test.
  loom_serve_finite_component_t* component_ = nullptr;
  // Cohort outputs independent of component parameter and command ownership.
  std::array<loom_serve_finite_result_t*, 2> results_ = {};
  // Independent consumer used after component release or reconstruction.
  loom_serve_execution_t* consumer_ = nullptr;
  // Only the oracle observation downloads result payloads, after device reuse.
  std::array<float, 4> output_ = {};
};

TEST_F(FiniteComponentTest,
       DormantHandleDoesNotTouchMissingSourceOrCheckpoint) {
  ASSERT_TRUE(std::filesystem::remove(directory_ + "/prepare.loom"));
  ASSERT_TRUE(std::filesystem::remove(directory_ + "/weights.safetensors"));
  EXPECT_EQ(loom_serve_finite_component_phase(component_),
            LOOM_SERVE_FINITE_COMPONENT_DECLARED);
  EXPECT_FALSE(loom_serve_finite_component_advance(component_, false));
  EXPECT_FALSE(loom_serve_finite_component_is_pending(component_));
  EXPECT_EQ(loom_serve_finite_component_report(component_)->commands, 0u);
  IREE_ASSERT_OK(Trim());
  EXPECT_EQ(loom_serve_finite_component_phase(component_),
            LOOM_SERVE_FINITE_COMPONENT_DECLARED);
}

TEST_F(FiniteComponentTest, RepeatedDemandSharesOneUnactivatedPreparation) {
  loom_serve_finite_model_t* model = nullptr;
  IREE_ASSERT_OK(owner_.Run(device_, [&](bool* complete) {
    iree_status_t status =
        loom_serve_finite_component_prepare(component_, &model);
    if (iree_status_is_ok(status) && !model) {
      status = loom_serve_finite_component_prepare(component_, &model);
    }
    *complete = model != nullptr;
    return status;
  }));
  ASSERT_NE(model, nullptr);
  const auto report = *loom_serve_finite_component_report(component_);
  EXPECT_EQ(report.commands, 2u);
  EXPECT_EQ(report.workers, 1u);
  EXPECT_FALSE(
      loom_serve_residency_snapshot(loom_serve_finite_model_residency(model))
          .active);
  EXPECT_EQ(
      loom_serve_residency_activity(loom_serve_finite_model_residency(model))
          .read_bytes,
      0u);
  loom_serve_finite_model_t* repeated = nullptr;
  IREE_ASSERT_OK(Prepare(component_, &repeated));
  EXPECT_EQ(repeated, model);
  EXPECT_EQ(loom_serve_finite_component_report(component_)->commands,
            report.commands);
  ASSERT_NO_FATAL_FAILURE(Invoke(model));
}

TEST_F(FiniteComponentTest, PreparationFailureIsRetainedWithoutRetrying) {
  ASSERT_TRUE(std::filesystem::remove(directory_ + "/weights.safetensors"));
  loom_serve_finite_model_t* model = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, Prepare(component_, &model));
  EXPECT_EQ(model, nullptr);
  EXPECT_FALSE(loom_serve_finite_component_is_pending(component_));
  EXPECT_EQ(loom_serve_finite_component_phase(component_),
            LOOM_SERVE_FINITE_COMPONENT_FAILED);
  const auto report = *loom_serve_finite_component_report(component_);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, Prepare(component_, &model));
  EXPECT_EQ(loom_serve_finite_component_report(component_)->commands,
            report.commands);
  EXPECT_FALSE(loom_serve_finite_component_advance(component_, false));
}

TEST_F(FiniteComponentTest, RemovalJoinsAcceptedPreparation) {
  bool started = false;
  IREE_ASSERT_OK(owner_.Run(device_, [&](bool* complete) -> iree_status_t {
    if (!started) {
      loom_serve_finite_model_t* model = nullptr;
      IREE_RETURN_IF_ERROR(
          loom_serve_finite_component_prepare(component_, &model));
      EXPECT_EQ(model, nullptr);
      EXPECT_TRUE(loom_serve_finite_component_is_pending(component_));
      started = true;
    }
    return loom_serve_finite_component_retire(component_, complete);
  }));
  EXPECT_EQ(loom_serve_finite_component_phase(component_),
            LOOM_SERVE_FINITE_COMPONENT_RETIRED);
  EXPECT_FALSE(loom_serve_finite_component_is_pending(component_));
  EXPECT_EQ(loom_serve_finite_component_report(component_)->commands, 0u);
}

TEST_F(FiniteComponentTest, TrimReleasesBackingAndNextDemandExecutesAgain) {
  loom_serve_finite_model_t* model = nullptr;
  IREE_ASSERT_OK(Prepare(component_, &model));
  ASSERT_NO_FATAL_FAILURE(Invoke(model));
  const auto report = *loom_serve_finite_component_report(component_);
  auto* residency = loom_serve_finite_model_residency(model);
  EXPECT_EQ(loom_serve_residency_activity(residency).activations, 1u);
  EXPECT_EQ(loom_serve_residency_activity(residency).read_bytes, 4u);
  IREE_ASSERT_OK(Trim());
  if ((loom_serve_test_device()->service_backing ==
       LOOM_SERVE_DEVICE_BACKING_ELASTIC)) {
    EXPECT_EQ(loom_serve_finite_component_phase(component_),
              LOOM_SERVE_FINITE_COMPONENT_READY);
    EXPECT_FALSE(loom_serve_residency_snapshot(residency).active);
    EXPECT_EQ(loom_serve_residency_statistics(residency).committed_bytes, 0u);
  } else {
    EXPECT_EQ(loom_serve_finite_component_phase(component_),
              LOOM_SERVE_FINITE_COMPONENT_DECLARED);
  }
  IREE_ASSERT_OK(Prepare(component_, &model));
  ASSERT_NO_FATAL_FAILURE(Invoke(model));
  residency = loom_serve_finite_model_residency(model);
  EXPECT_EQ(loom_serve_finite_component_report(component_)->commands,
            report.commands * ((loom_serve_test_device()->service_backing ==
                                LOOM_SERVE_DEVICE_BACKING_ELASTIC)
                                   ? 1
                                   : 2));
  EXPECT_EQ(loom_serve_residency_activity(residency).activations,
            (loom_serve_test_device()->service_backing ==
             LOOM_SERVE_DEVICE_BACKING_ELASTIC)
                ? 2u
                : 1u);
  EXPECT_EQ(loom_serve_residency_activity(residency).read_bytes,
            (loom_serve_test_device()->service_backing ==
             LOOM_SERVE_DEVICE_BACKING_ELASTIC)
                ? 8u
                : 4u);
}

TEST_F(FiniteComponentTest, ComponentReleasePreservesResidentPeer) {
  loom_serve_finite_model_t* model = nullptr;
  IREE_ASSERT_OK(Prepare(component_, &model));
  ASSERT_NO_FATAL_FAILURE(Invoke(model));
  loom_serve_finite_component_t* peer = nullptr;
  IREE_ASSERT_OK(loom_serve_finite_component_allocate(
      device_, &options_, owner_.observer(), host_allocator_, &peer));
  loom_serve_finite_model_t* peer_model = nullptr;
  IREE_ASSERT_OK(Prepare(peer, &peer_model));
  ASSERT_NO_FATAL_FAILURE(Invoke(peer_model));
  auto* peer_residency = loom_serve_finite_model_residency(peer_model);
  loom_serve_residency_set_pinned(peer_residency, true);
  const auto activity = loom_serve_residency_activity(peer_residency);
  IREE_ASSERT_OK(Trim());
  EXPECT_TRUE(loom_serve_residency_snapshot(peer_residency).active);
  ASSERT_NO_FATAL_FAILURE(Invoke(peer_model));
  EXPECT_EQ(loom_serve_residency_activity(peer_residency).activations,
            activity.activations);
  EXPECT_EQ(loom_serve_residency_activity(peer_residency).read_bytes,
            activity.read_bytes);
  loom_serve_residency_set_pinned(peer_residency, false);
  IREE_ASSERT_OK(Retire(peer));
  loom_serve_finite_component_free(peer);
}

TEST_F(FiniteComponentTest, DeviceResultSurvivesTrimReloadAndRemoval) {
  loom_serve_finite_model_t* model = nullptr;
  IREE_ASSERT_OK(Prepare(component_, &model));
  ASSERT_NO_FATAL_FAILURE(Submit(model, 1));
  IREE_ASSERT_OK(Trim());
  IREE_ASSERT_OK(Prepare(component_, &model));
  ASSERT_NO_FATAL_FAILURE(Invoke(model));
  // The old result still names its original allocation and rows even after
  // elastic parameter reload or fixed-backing model reconstruction.
  ASSERT_NO_FATAL_FAILURE(CheckResult(1));
  IREE_ASSERT_OK(Retire(component_));
  loom_serve_finite_component_free(std::exchange(component_, nullptr));
  IREE_ASSERT_OK(loom_serve_device_trim(device_, 0));
  ASSERT_NO_FATAL_FAILURE(CheckResult(1));
}

TEST_F(FiniteComponentTest, TrimPreservesExplicitRetention) {
  loom_serve_finite_component_set_pinned(component_, true);
  EXPECT_EQ(loom_serve_finite_component_phase(component_),
            LOOM_SERVE_FINITE_COMPONENT_DECLARED);
  loom_serve_residency_snapshot_t parameters = {};
  EXPECT_FALSE(loom_serve_finite_component_parameters(component_, &parameters));
  loom_serve_finite_model_t* model = nullptr;
  IREE_ASSERT_OK(Prepare(component_, &model));
  ASSERT_NO_FATAL_FAILURE(Invoke(model));
  auto* residency = loom_serve_finite_model_residency(model);
  EXPECT_TRUE(loom_serve_residency_is_pinned(residency));
  IREE_ASSERT_OK(Trim());
  EXPECT_EQ(loom_serve_finite_component_phase(component_),
            LOOM_SERVE_FINITE_COMPONENT_READY);
  EXPECT_TRUE(loom_serve_residency_snapshot(residency).active);
  loom_serve_finite_component_set_pinned(component_, false);
  EXPECT_FALSE(loom_serve_residency_is_pinned(residency));
  IREE_ASSERT_OK(Trim());
  IREE_ASSERT_OK(Prepare(component_, &model));
  EXPECT_FALSE(
      loom_serve_residency_is_pinned(loom_serve_finite_model_residency(model)));
}

TEST_F(FiniteComponentTest, DemandSourcesRebindWithoutResidentWeightsOrJit) {
  ASSERT_NO_FATAL_FAILURE(UseDemandSources());
  loom_serve_finite_model_t* model = nullptr;
  IREE_ASSERT_OK(Prepare(component_, &model));
  const auto commands =
      loom_serve_finite_component_report(component_)->commands;
  EXPECT_EQ(commands, 2u);
  for (size_t ordinal : {0u, 1u, 0u}) {
    ASSERT_NO_FATAL_FAILURE(Submit(model, ordinal));
    ASSERT_NO_FATAL_FAILURE(CheckResult(ordinal, ordinal ? 4.0f : 2.0f));
    loom_serve_finite_result_release(std::exchange(results_[ordinal], nullptr));
    EXPECT_EQ(loom_serve_finite_component_report(component_)->commands,
              commands);
    const auto parameters = loom_serve_finite_model_parameters(model);
    EXPECT_EQ(parameters.memory.reserved_bytes, 0u);
    EXPECT_EQ(parameters.memory.committed_bytes, 0u);
    EXPECT_EQ(parameters.activity.read_bytes, 0u);
  }
  ASSERT_NO_FATAL_FAILURE(Submit(model, 1));
  IREE_ASSERT_OK(Retire(component_));
  loom_serve_finite_component_free(std::exchange(component_, nullptr));
  ASSERT_NO_FATAL_FAILURE(CheckResult(1, 4.0f));
}

TEST_F(FiniteComponentTest, UnjoinedDemandReadsRetireWithRejectedInvocation) {
  ASSERT_NO_FATAL_FAILURE(UseDemandSources());
  loom_serve_finite_model_t* model = nullptr;
  IREE_ASSERT_OK(Prepare(component_, &model));
  ASSERT_NO_FATAL_FAILURE(Submit(model, 0, true));
  // Source rejected after submitting both reads, without joining either one.
  // Removal releases the source VM before waiting for captured storage users.
  IREE_ASSERT_OK(Retire(component_));
  loom_serve_finite_component_free(std::exchange(component_, nullptr));
  IREE_ASSERT_OK(loom_serve_finite_component_allocate(
      device_, &options_, owner_.observer(), host_allocator_, &component_));
  IREE_ASSERT_OK(Prepare(component_, &model));
  ASSERT_NO_FATAL_FAILURE(Submit(model, 1));
  ASSERT_NO_FATAL_FAILURE(CheckResult(1, 4.0f));
}

}  // namespace
