// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_TESTING_JIT_H_
#define LOOM_SERVE_TESTING_JIT_H_

#include <array>
#include <condition_variable>
#include <mutex>
#include <string>

#include "iree/task/executor.h"
#include "iree/testing/gtest.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/runtime/program/jit.h"

namespace loom_serve::testing {

// A scheduler process submits compilation from the same executor that must
// compile its native requests. Only the outer test thread waits. One worker
// makes a hidden synchronous join deterministic instead of timing-dependent.
struct Compilation {
  void Start(iree_task_executor_t* executor);
  iree_status_t Await(loom_serve_jit_stage_t** out_stage);

  // Catalog with no other compilation outstanding; owned when retiring it.
  loom_serve_jit_t* jit = nullptr;
  // Optional retired model released by the shared worker before compilation.
  loom_serve_execution_t* retired_execution = nullptr;
  // Exercises worker-side retirement at the actual final compilation callback.
  bool retire_catalog = false;
  // Requests a native executable instead of a reusable command program.
  bool direct_dispatch = false;
  // Borrowed source root retained through Await.
  iree_string_view_t root = {};
  // Borrowed specialization retained through Await.
  const loomc_config_options_t* config = nullptr;

 private:
  static iree_status_t Drain(iree_task_process_t* process,
                             const iree_task_worker_context_t* context,
                             iree_task_process_drain_result_t* result);
  static void Complete(void* user_data, iree_status_t status,
                       loom_serve_jit_stage_t* stage);
  static void Release(iree_task_process_t* process);

  // Immediate process representing a scheduler's compilation submission.
  iree_task_process_t process = {};
  // Joins both callback publication and the scheduler process's final access.
  std::mutex mutex;
  // Outer test's readiness event, unrelated to device completion.
  std::condition_variable condition;
  // Compilation delivered its final outcome.
  bool compiled = false;
  // Executor relinquished the submitting process storage.
  bool released = false;
  // Owned outcome transferred to the outer caller.
  iree_status_t status = iree_ok_status();
  // Owned command stage transferred to the outer caller.
  loom_serve_jit_stage_t* stage = nullptr;
};

// Owns a source catalog and model execution over one shared host/device group.
// Prepared commands and transfer backing survive until accepted work drains.
class JitTest : public ::testing::TestWithParam<int> {
 protected:
  void SetUp() override;
  void TearDown() override;

  // Null uses the source provider; text overlays its value for this request.
  iree_status_t Compile(const char* delta, size_t index,
                        iree_string_view_t root = IREE_SV("advance"),
                        const char* temporary_bytes = nullptr);

  // Releases a derived fixture's program after model completion and before
  // prepared stages, source catalog, execution, and device group retire.
  virtual void ReleaseProgram() {}

  // Host allocation policy for all fixture resources.
  iree_allocator_t host_allocator_ = iree_allocator_system();
  // Copied compiler evidence retained beyond individual results.
  loom_serve_compilation_report_t compilation_ = {};
  // Application group outliving all compiler, VM and I/O resources.
  loom_serve_device_group_t* group_ = nullptr;
  // Borrowed device record within the application group.
  loom_serve_device_t* device_owner_ = nullptr;
  // Borrowed live device used for profile extraction.
  iree_hal_device_t* device_ = nullptr;
  // Exact dispatch queue borrowed from device.
  iree_hal_queue_t* dispatch_ = nullptr;
  // Owned model submission/feedback timelines over shared device backing.
  loom_serve_execution_t* execution_ = nullptr;
  // Portable source location from runfiles.
  std::string directory_;
  // Model-resident compiler shared by both variants.
  loom_serve_jit_t* jit_ = nullptr;
  // Independently prepared variants retaining executable ownership.
  std::array<iree_hal_command_buffer_t*, 2> commands_ = {};
  // Direct stages retaining executables after source/catalog retirement.
  std::array<loom_serve_dispatch_t*, 2> dispatches_ = {};
  // Compiler requirements retained independently of the stage/compiler.
  std::array<loom_cmd_program_transient_requirement_t, 2> transients_ = {};
  // Device state shared across the variants.
  iree_hal_buffer_t* buffer_ = nullptr;
  // Upload backing retained until teardown drains accepted work.
  std::array<int32_t, 8> input_ = {100, 100, 100, 100, 100, 100, 100, 100};
  // Feedback backing retained until teardown drains accepted work.
  std::array<int32_t, 8> output_ = {};
};

}  // namespace loom_serve::testing

#endif  // LOOM_SERVE_TESTING_JIT_H_
