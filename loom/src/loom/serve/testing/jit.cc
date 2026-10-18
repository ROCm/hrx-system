// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/testing/jit.h"

#include <filesystem>

#include "iree/base/tooling/flags.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/testing/compiler.h"
#include "loom/serve/testing/device.h"

IREE_FLAG(string, jit_sources, "", "Portable test source catalog.");

namespace loom_serve::testing {

namespace {

struct JitTestApplication {
  // Host worker count selecting this application.
  iree_host_size_t worker_count;
  // Process-owned physical device group shared by matching test cases.
  loom_serve_device_group_t* group = nullptr;
};

std::array<JitTestApplication, 2>& GetJitTestApplications() {
  static std::array<JitTestApplication, 2> applications = {
      JitTestApplication{1}, JitTestApplication{4}};
  return applications;
}

iree_status_t AcquireJitTestApplication(iree_host_size_t worker_count,
                                        iree_allocator_t host_allocator,
                                        loom_serve_device_group_t** out_group) {
  *out_group = nullptr;
  JitTestApplication* selected = nullptr;
  for (auto& application : GetJitTestApplications()) {
    if (application.worker_count == worker_count) {
      selected = &application;
      break;
    }
  }
  if (!selected) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported JIT test worker count: %" PRIhsz,
                            worker_count);
  }
  if (selected->group) {
    *out_group = selected->group;
    return iree_ok_status();
  }

  loom_serve_host_t* host = nullptr;
  IREE_RETURN_IF_ERROR(
      loom_serve_test_host_create(worker_count, host_allocator, &host));
  iree_hal_driver_t* driver = nullptr;
  iree_status_t status = loom_serve_test_driver_acquire(host, &driver);
  if (iree_status_is_ok(status)) {
    const loom_serve_device_options_t options = {
        .driver = driver,
        .uri = loom_serve_test_device()->device_uri,
    };
    status = loom_serve_device_group_allocate(host, 1, &options, host_allocator,
                                              &selected->group);
  }
  iree_hal_driver_release(driver);
  loom_serve_host_release(host);
  if (iree_status_is_ok(status)) {
    *out_group = selected->group;
  }
  return status;
}

class JitTestEnvironment : public ::testing::Environment {
 public:
  void TearDown() override {
    iree_status_t status = iree_ok_status();
    for (auto& application : GetJitTestApplications()) {
      status = iree_status_join(
          status, loom_serve_device_group_free(application.group));
      application.group = nullptr;
    }
    IREE_EXPECT_OK(status);
  }
};

::testing::Environment* const jit_test_environment =
    ::testing::AddGlobalTestEnvironment(new JitTestEnvironment);

}  // namespace

iree_status_t loom_serve_test_compilation_t::Drain(
    iree_task_process_t* process, const iree_task_worker_context_t* context,
    iree_task_process_drain_result_t* result) {
  auto* self = static_cast<loom_serve_test_compilation_t*>(process->user_data);
  loom_serve_execution_release(self->retired_execution);
  self->retired_execution = nullptr;
  auto status = self->direct_dispatch
                    ? loom_serve_jit_compile_dispatch_async(
                          self->jit, self->root, self->config, Complete, self)
                    : loom_serve_jit_compile_async(
                          self->jit, self->root, self->config, Complete, self);
  if (!iree_status_is_ok(status)) {
    Complete(self, status, nullptr);
  }
  result->completed = true;
  result->did_work = true;
  return iree_ok_status();
}

void loom_serve_test_compilation_t::Complete(void* user_data,
                                             iree_status_t status,
                                             loom_serve_jit_stage_t* stage) {
  auto* self = static_cast<loom_serve_test_compilation_t*>(user_data);
  if (self->retire_catalog) {
    loom_serve_jit_release(self->jit);
    self->jit = nullptr;
  }
  std::lock_guard<std::mutex> lock(self->mutex);
  self->status = status;
  self->stage = stage;
  self->compiled = true;
  self->condition.notify_one();
}

void loom_serve_test_compilation_t::Release(iree_task_process_t* process) {
  auto* self = static_cast<loom_serve_test_compilation_t*>(process->user_data);
  std::lock_guard<std::mutex> lock(self->mutex);
  self->released = true;
  self->condition.notify_one();
}

void loom_serve_test_compilation_t::Start(iree_task_executor_t* executor) {
  iree_task_process_initialize(Drain, 0, 1, &process);
  process.user_data = this;
  process.release_fn = Release;
  iree_task_executor_schedule_process(executor, &process);
}

iree_status_t loom_serve_test_compilation_t::Await(
    loom_serve_jit_stage_t** out_stage) {
  std::unique_lock<std::mutex> lock(mutex);
  condition.wait(lock, [&] { return compiled && released; });
  *out_stage = stage;
  return status;
}

void JitTest::SetUp() {
  IREE_ASSERT_OK(
      AcquireJitTestApplication(GetParam(), host_allocator_, &group_));
  device_owner_ = loom_serve_device_group_at(group_, 0);
  device_ = loom_serve_device_handle(device_owner_);
  dispatch_ = loom_serve_device_dispatch_queue(device_owner_);
  workspace_baseline_ =
      loom_serve_device_memory_statistics(device_owner_).workspace;
  ASSERT_EQ(workspace_baseline_.reservation_count, 0u);
  ASSERT_EQ(workspace_baseline_.bytes_committed, 0u);
  IREE_ASSERT_OK(
      loom_serve_device_create_execution(device_owner_, &execution_));
  directory_ = std::filesystem::path(FLAG_jit_sources).parent_path().string();
  const iree_string_view_t providers[] = {IREE_SV("config.loom"),
                                          IREE_SV("stage.loom")};
  const loom_serve_sources_t sources = {
      .base_directory = iree_make_cstring_view(directory_.c_str()),
      .count = IREE_ARRAYSIZE(providers),
      .values = providers,
  };
  IREE_ASSERT_OK(loom_serve_jit_create(
      device_, dispatch_, loom_serve_device_host(device_owner_), &sources,
      nullptr, &compilation_, host_allocator_, &jit_));
}

void JitTest::TearDown() {
  if (execution_) {
    IREE_EXPECT_OK(loom_serve_execution_drain(execution_));
  }
  ReleaseProgram();
  for (auto* command : commands_) {
    iree_hal_command_buffer_release(command);
  }
  for (auto* dispatch : dispatches_) {
    loom_serve_dispatch_release(dispatch);
  }
  iree_hal_buffer_release(buffer_);
  loom_serve_jit_release(jit_);
  loom_serve_execution_release(execution_);
  if (device_owner_) {
    IREE_EXPECT_OK(loom_serve_test_device_trim_and_drain(device_owner_, 0));
    const auto workspace =
        loom_serve_device_memory_statistics(device_owner_).workspace;
    EXPECT_EQ(workspace.reservation_count, 0u);
    EXPECT_EQ(workspace.bytes_committed, 0u);
  }
}

iree_status_t JitTest::Compile(const char* delta, size_t index,
                               iree_string_view_t root,
                               const char* temporary_bytes) {
  loomc_config_binding_t bindings[2] = {};
  size_t count = 0;
  if (delta) {
    bindings[count++] = {loomc_make_cstring_view("increment.delta"),
                         loomc_make_cstring_view(delta)};
  }
  if (temporary_bytes) {
    bindings[count++] = {loomc_make_cstring_view("increment.temporary_bytes"),
                         loomc_make_cstring_view(temporary_bytes)};
  }
  const loomc_config_options_t config = {
      bindings, count, {}, LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED};
  loom_serve_jit_stage_t* stage = nullptr;
  loom_serve_test_compilation_t compilation;
  compilation.jit = jit_;
  compilation.root = root;
  compilation.config = &config;
  compilation.Start(
      loom_serve_host_executor(loom_serve_device_host(device_owner_)));
  auto status = compilation.Await(&stage);
  if (iree_status_is_ok(status)) {
    transients_[index] =
        loom_serve_jit_stage_program(stage)->requirements.transient;
    status =
        loom_serve_jit_stage_record(stage, iree_hal_queue_family(dispatch_),
                                    IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
                                    nullptr, nullptr, &commands_[index]);
  }
  loom_serve_jit_stage_free(stage);
  return status;
}

}  // namespace loom_serve::testing
