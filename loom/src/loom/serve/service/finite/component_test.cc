// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/finite/component.h"

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
#include "loom/serve/testing/device.h"
#include "loom/serve/testing/service_application.h"

namespace {

// Actual serialized owner on the same executor as preparation. Only the outer
// test waits, and it waits for process release rather than merely completion.
class ComponentOwner {
 public:
  loom_serve_wake_callback_t observer() { return {Wake, this}; }

  iree_status_t Run(iree_task_executor_t* executor,
                    std::function<iree_status_t(bool*)> advance) {
    executor_ = executor;
    advance_ = std::move(advance);
    released_ = false;
    iree_task_process_initialize(Drain, 0, 1, &process_);
    process_.user_data = this;
    process_.release_fn = Release;
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
    owner->status_ = owner->advance_(&result->completed);
    result->completed |= !iree_status_is_ok(owner->status_);
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
  // Current serialized operation, retaining its captures through release.
  std::function<iree_status_t(bool*)> advance_;
  // Terminal outcome returned only after the executor releases the process.
  iree_status_t status_ = iree_ok_status();
  // Intrusive owner process, never reused before Release finishes.
  iree_task_process_t process_ = {};
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
    host_ = loom_serve_test_application_host(application);
    device_ = loom_serve_test_application_device_at(application, 0);
    directory_ = iree::testing::MakeTempFilePath("finite_component");
    ASSERT_TRUE(std::filesystem::create_directory(directory_));
    const char* source_path = std::getenv("LOOM_SERVE_TEST_COMPONENT_SOURCE");
    ASSERT_NE(source_path, nullptr);
    const auto source_directory =
        std::filesystem::path(source_path).parent_path();
    for (const char* file :
         {"prepare.loom", "control.loom", "kernels.loom", "policy.loom"}) {
      ASSERT_TRUE(std::filesystem::copy_file(
          source_directory / file, std::filesystem::path(directory_) / file));
    }
    std::string header =
        R"({"factor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
    header.append((8 - header.size() % 8) % 8, ' ');
    std::ofstream checkpoint(directory_ + "/weights.safetensors",
                             std::ios::binary);
    uint8_t bytes[8];
    iree_unaligned_store_le_u64(bytes, header.size());
    checkpoint.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    checkpoint.write(header.data(), header.size());
    iree_unaligned_store_le_u32(bytes,
                                0x3f800000);  // F32 1.0, prepared to 2.0.
    checkpoint.write(reinterpret_cast<const char*>(bytes), 4);
    checkpoint.close();
    ASSERT_TRUE(checkpoint.good());
    options_ = {
        .sources =
            {
                .base_directory = iree_make_cstring_view(directory_.c_str()),
                .prepare = IREE_SV("prepare.loom"),
                .control = IREE_SV("control.loom"),
                .count = 1,
                .values = &provider_,
            },
        .asset_directory = IREE_SV("."),
        .options = IREE_SV("{}"),
    };
    IREE_ASSERT_OK(loom_serve_finite_component_allocate(
        device_, &options_, owner_.observer(), host_allocator_, &component_));
  }

  void TearDown() override {
    if (component_) {
      IREE_EXPECT_OK(Retire(component_));
      loom_serve_finite_component_free(component_);
    }
  }

  iree_status_t Prepare(loom_serve_finite_component_t* component,
                        loom_serve_finite_model_t** out_model) {
    return owner_.Run(loom_serve_host_executor(host_), [&](bool* complete) {
      iree_status_t status =
          loom_serve_finite_component_prepare(component, out_model);
      *complete = *out_model != nullptr;
      return status;
    });
  }

  iree_status_t Retire(loom_serve_finite_component_t* component) {
    return owner_.Run(loom_serve_host_executor(host_), [&](bool* complete) {
      return loom_serve_finite_component_retire(component, complete);
    });
  }

  iree_status_t Trim() {
    return owner_.Run(loom_serve_host_executor(host_), [&](bool* complete) {
      return loom_serve_finite_component_trim(component_, complete);
    });
  }

  void Invoke(loom_serve_finite_model_t* model) {
    iree_vm_buffer_t* input = nullptr;
    IREE_ASSERT_OK(iree_vm_buffer_create(16, 16, host_allocator_, &input));
    iree_byte_span_t bytes;
    IREE_ASSERT_OK(iree_vm_buffer_map_write(input, 0, 16, &bytes));
    for (uint32_t i = 0; i < 4; ++i) {
      iree_unaligned_store_le_f32(bytes.data + i * 4,
                                  static_cast<float>(i + 1));
    }
    loom_serve_finite_prepared_t prepared = {};
    IREE_ASSERT_OK(
        loom_serve_finite_model_prepare_request(model, input, &prepared));
    const iree_host_size_t rows = 1;
    IREE_ASSERT_OK(
        loom_serve_finite_model_prepare_batch(model, 1, &prepared, &rows));
    iree_vm_buffer_release(prepared.bytes);
    iree_vm_buffer_release(input);
    bool admitted = false;
    IREE_ASSERT_OK(loom_serve_residency_try_acquire(
        loom_serve_finite_model_residency(model), &admitted));
    ASSERT_TRUE(admitted);
    loom_serve_execution_frontier_t frontier;
    IREE_ASSERT_OK(loom_serve_finite_model_submit(model, &frontier));
    // Only this outer application thread joins device completion.
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        {2, frontier.semaphores, frontier.values}, iree_infinite_timeout(),
        IREE_ASYNC_WAIT_FLAG_NONE));
    loom_serve_finite_model_complete(model);
    const auto output = loom_serve_finite_model_row(model, 0, 0);
    ASSERT_EQ(output.data_length, 16u);
    for (uint32_t i = 0; i < 4; ++i) {
      EXPECT_EQ(iree_unaligned_load_le_f32(output.data + i * 4),
                static_cast<float>((i + 1) * 2));
    }
    loom_serve_finite_model_release_batch(model);
  }

  // Allocation policy for this real source/JIT/device fixture.
  iree_allocator_t host_allocator_ = iree_allocator_system();
  // Borrowed process-wide worker and proactor infrastructure.
  loom_serve_host_t* host_ = nullptr;
  // Borrowed physical owner shared by both tested components.
  loom_serve_device_t* device_ = nullptr;
  // Real shared-executor owner for cold lifecycle operations.
  ComponentOwner owner_;
  // Stable path backing all borrowed source/parameter locators.
  std::string directory_;
  // Native provider name borrowed by options_.sources.
  iree_string_view_t provider_ = IREE_SV("kernels.loom");
  // Stable borrowed definition through final component destruction.
  loom_serve_finite_model_options_t options_ = {};
  // Owned optional model handle under test.
  loom_serve_finite_component_t* component_ = nullptr;
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
  IREE_ASSERT_OK(
      owner_.Run(loom_serve_host_executor(host_), [&](bool* complete) {
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
  IREE_ASSERT_OK(owner_.Run(
      loom_serve_host_executor(host_), [&](bool* complete) -> iree_status_t {
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

TEST_F(FiniteComponentTest, TrimPreservesExplicitRetention) {
  loom_serve_finite_model_t* model = nullptr;
  IREE_ASSERT_OK(Prepare(component_, &model));
  ASSERT_NO_FATAL_FAILURE(Invoke(model));
  auto* residency = loom_serve_finite_model_residency(model);
  loom_serve_residency_set_pinned(residency, true);
  EXPECT_TRUE(loom_serve_residency_is_pinned(residency));
  IREE_ASSERT_OK(Trim());
  EXPECT_EQ(loom_serve_finite_component_phase(component_),
            LOOM_SERVE_FINITE_COMPONENT_READY);
  EXPECT_TRUE(loom_serve_residency_snapshot(residency).active);
  loom_serve_residency_set_pinned(residency, false);
  EXPECT_FALSE(loom_serve_residency_is_pinned(residency));
  IREE_ASSERT_OK(Trim());
}

}  // namespace
