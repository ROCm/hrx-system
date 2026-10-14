// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_FINITE_COMPONENT_TEST_FIXTURE_H_
#define LOOM_SERVE_SERVICE_FINITE_COMPONENT_TEST_FIXTURE_H_

#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>

#include "iree/task/executor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/service/finite/component.h"

namespace loom_serve::testing::finite {

// Actual serialized owner on the same executor as preparation. Only the outer
// test waits, and it waits for process release rather than merely completion.
class ComponentOwner {
 public:
  loom_serve_wake_callback_t observer() { return {Wake, this}; }

  iree_status_t Run(iree_task_executor_t* executor,
                    std::function<iree_status_t(bool*)> advance);

 private:
  static void Wake(void* user_data);

  static iree_status_t Drain(iree_task_process_t* process,
                             const iree_task_worker_context_t* context,
                             iree_task_process_drain_result_t* result);

  static void Release(iree_task_process_t* process);

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
  void SetUp() override;

  void TearDown() override;

  iree_status_t Prepare(loom_serve_finite_component_t* component,
                        loom_serve_finite_model_t** out_model);

  iree_status_t Retire(loom_serve_finite_component_t* component);

  iree_status_t Trim();

  void Project(loom_serve_finite_model_t* model, float first,
               loom_serve_finite_result_t* out_result);

  void ExpectResult(const loom_serve_finite_result_t* result, float first);

  void Invoke(loom_serve_finite_model_t* model);

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

}  // namespace loom_serve::testing::finite

#endif  // LOOM_SERVE_SERVICE_FINITE_COMPONENT_TEST_FIXTURE_H_
