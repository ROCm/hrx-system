// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/jit.h"

#include <array>
#include <condition_variable>
#include <filesystem>
#include <mutex>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#endif

#include "experimental/loom_serve/runtime/device.h"
#include "experimental/loom_serve/runtime/execution.h"
#include "experimental/loom_serve/runtime/module.h"
#include "iree/base/tooling/flags.h"
#include "iree/task/executor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/sync.h"

IREE_FLAG(string, jit_sources, "", "Portable test source catalog.");
IREE_FLAG(string, invalid_sources, "", "Malformed source catalog.");

namespace {

#if defined(__linux__)
// A full blocking sink makes an accidental worker-side diagnostic write hang
// until the outer test harness reports failure. There is no consumer thread.
class FullStderr {
 public:
  int Redirect() {
    if (pipe2(pipe_, O_CLOEXEC | O_NONBLOCK) != 0) {
      return errno;
    }
    char bytes[4096] = {};
    while (write(pipe_[1], bytes, sizeof(bytes)) > 0) {
    }
    if (errno != EAGAIN) {
      return errno;
    }
    if (fcntl(pipe_[1], F_SETFL, 0) != 0) {
      return errno;
    }
    if (fflush(stderr) != 0) {
      return errno;
    }
    original_ = dup(STDERR_FILENO);
    if (original_ < 0) {
      return errno;
    }
    return dup2(pipe_[1], STDERR_FILENO) < 0 ? errno : 0;
  }
  ~FullStderr() {
    if (original_ >= 0) {
      // Restore before reporting cleanup failures to the test runner.
      const int restored = dup2(original_, STDERR_FILENO);
      EXPECT_GE(restored, 0);
      EXPECT_EQ(close(original_), 0);
    }
    if (pipe_[1] >= 0) {
      EXPECT_EQ(close(pipe_[1]), 0);
    }
    if (pipe_[0] >= 0) {
      EXPECT_EQ(close(pipe_[0]), 0);
    }
  }

 private:
  // Pipe endpoints owned through restoration of the process stderr handle.
  int pipe_[2] = {-1, -1};
  // Saved process stderr descriptor, or -1 before successful duplication.
  int original_ = -1;
};
#endif

// A scheduler process submits compilation from the same executor that must
// compile its native requests. Only the outer test thread waits. One worker
// makes a hidden synchronous join deterministic instead of timing-dependent.
struct Compilation {
  static iree_status_t Drain(iree_task_process_t* process,
                             const iree_task_worker_context_t*,
                             iree_task_process_drain_result_t* result) {
    auto* self = static_cast<Compilation*>(process->user_data);
    auto status = loom_serve_jit_compile_async(self->jit, self->root,
                                               self->config, Complete, self);
    if (!iree_status_is_ok(status)) {
      Complete(self, status, nullptr);
    }
    result->completed = true;
    result->did_work = true;
    return iree_ok_status();
  }

  static void Complete(void* user_data, iree_status_t status,
                       loom_serve_jit_stage_t* stage) {
    auto* self = static_cast<Compilation*>(user_data);
    if (self->retire_catalog) {
      loom_serve_jit_destroy(self->jit);
      self->jit = nullptr;
    }
    std::lock_guard<std::mutex> lock(self->mutex);
    self->status = status;
    self->stage = stage;
    self->compiled = true;
    self->condition.notify_one();
  }

  static void Release(iree_task_process_t* process) {
    auto* self = static_cast<Compilation*>(process->user_data);
    std::lock_guard<std::mutex> lock(self->mutex);
    self->released = true;
    self->condition.notify_one();
  }

  void Start(iree_task_executor_t* executor) {
    iree_task_process_initialize(Drain, 0, 1, &process);
    process.user_data = this;
    process.release_fn = Release;
    iree_task_executor_schedule_process(executor, &process);
  }

  iree_status_t Await(loom_serve_jit_stage_t** out_stage) {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return compiled && released; });
    *out_stage = stage;
    return status;
  }

  // Catalog with no other compilation outstanding; owned when retiring it.
  loom_serve_jit_t* jit = nullptr;
  // Exercises worker-side retirement at the actual final compilation callback.
  bool retire_catalog = false;
  // Borrowed source root retained through Await.
  iree_string_view_t root = {};
  // Borrowed specialization retained through Await.
  const loomc_config_options_t* config = nullptr;
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

class JitTest : public ::testing::TestWithParam<int> {
 protected:
  void SetUp() override {
    const loom_serve_device_options_t options = {.uri = IREE_SV("amdgpu")};
    loom_serve_host_t* host = nullptr;
    IREE_ASSERT_OK(loom_serve_host_create(GetParam(), &host, allocator_));
    auto status =
        loom_serve_device_group_create(host, 1, &options, &group_, allocator_);
    loom_serve_host_release(host);
    IREE_ASSERT_OK(status);
    device_owner_ = loom_serve_device_group_at(group_, 0);
    device_ = loom_serve_device_handle(device_owner_);
    dispatch_ = loom_serve_device_dispatch_queue(device_owner_);
    IREE_ASSERT_OK(
        loom_serve_device_create_execution(device_owner_, &execution_));
    directory_ = std::filesystem::path(FLAG_jit_sources).parent_path().string();
    IREE_ASSERT_OK(loom_serve_jit_create(
        device_, dispatch_, loom_serve_device_host(device_owner_),
        iree_make_cstring_view(directory_.c_str()), nullptr, &compilation_,
        allocator_, &jit_));
  }

  void TearDown() override {
    // Even an assertion before the explicit release must retire the held work
    // while fixture-owned uploads and feedback are still alive.
    if (held_.gate) {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(held_.gate, 1, nullptr));
    }
    iree_hal_buffer_release(held_.buffer);
    if (held_.execution) {
      IREE_EXPECT_OK(loom_serve_execution_drain(held_.execution));
    }
    if (execution_) {
      IREE_EXPECT_OK(loom_serve_execution_drain(execution_));
    }
    iree_vm_process_release(process_);
    if (invocation_) {
      iree_vm_invocation_deinitialize(invocation_);
    }
    iree_vm_program_release(program_);
    iree_vm_module_release(native_);
    iree_vm_module_release(bytecode_);
    iree_vm_environment_free(environment_);
    for (auto* command : commands_) {
      iree_hal_command_buffer_release(command);
    }
    iree_hal_buffer_release(buffer_);
    loom_serve_jit_destroy(jit_);
    loom_serve_execution_release(held_.execution);
    iree_hal_semaphore_release(held_.gate);
    loom_serve_execution_release(execution_);
    IREE_EXPECT_OK(loom_serve_device_group_destroy(group_));
  }

  // Null uses the source provider; text overlays its value for this request.
  iree_status_t Compile(const char* delta, size_t index,
                        iree_string_view_t root = IREE_SV("advance"),
                        const char* temporary_bytes = nullptr) {
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
    Compilation compilation;
    compilation.jit = jit_;
    compilation.root = root;
    compilation.config = &config;
    compilation.Start(
        loom_serve_host_executor(loom_serve_device_host(device_owner_)));
    auto status = compilation.Await(&stage);
    if (iree_status_is_ok(status)) {
      transients_[index] =
          loom_serve_jit_stage_program(stage)->requirements.transient;
      status = loom_serve_jit_stage_record(
          stage, iree_hal_queue_family(dispatch_),
          IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, nullptr, &commands_[index]);
    }
    loom_serve_jit_stage_destroy(stage);
    return status;
  }

  iree_status_t PrepareVm() {
    IREE_RETURN_IF_ERROR(
        iree_vm_environment_allocate(allocator_, &environment_));
    const iree_vm_ref_type_table_t* table = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_module_register_types(environment_, &table));
    IREE_RETURN_IF_ERROR(iree_hal_module_types_resolve(table, &types_));
    const loom_serve_stage_t stages[] = {
        {commands_[0], 1, transients_[0]},
        {commands_[1], 1, transients_[1]},
    };
    loom_serve_module_options_t options = {};
    options.binding_capacity = 1;
    options.stages = {commands_[1] ? 2u : 1u, stages};
    IREE_RETURN_IF_ERROR(loom_serve_module_create(&types_, execution_, options,
                                                  allocator_, &native_));
    const std::string path = directory_ + "/control.loom";
    iree_const_byte_span_t image;
    const iree_string_view_t roots[] = {IREE_SVL("step"),
                                        IREE_SVL("selected_step")};
    IREE_RETURN_IF_ERROR(loom_serve_jit_compile_vm(
        iree_make_cstring_view(path.c_str()), IREE_ARRAYSIZE(roots), roots,
        &compilation_, allocator_, &image));
    auto status = iree_vm_bytecode_module_create_trusted(
        environment_, IREE_SV("model"), {image, allocator_}, allocator_,
        &bytecode_);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(allocator_, (void*)image.data);
      return status;
    }
    iree_vm_module_t* libraries[] = {native_};
    IREE_RETURN_IF_ERROR(iree_vm_program_create(
        {bytecode_, iree_vm_module_span_from_array(libraries)}, allocator_,
        &program_));
    IREE_RETURN_IF_ERROR(iree_vm_invocation_initialize(
        iree_make_byte_span(invocation_storage_.data(),
                            invocation_storage_.size()),
        &invocation_));
    IREE_RETURN_IF_ERROR(iree_vm_process_create(program_, invocation_,
                                                iree_vm_variant_span_empty(),
                                                allocator_, &process_));
    return iree_vm_process_lookup_function(process_, IREE_SV("model"),
                                           IREE_SV("step"), &step_);
  }

  // Host allocation policy for all fixture resources.
  iree_allocator_t allocator_ = iree_allocator_system();
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
  // Portable source location from runfiles.
  std::string directory_;
  // Model-resident compiler shared by both variants.
  loom_serve_jit_t* jit_ = nullptr;
  // Independently prepared variants retaining executable ownership.
  std::array<iree_hal_command_buffer_t*, 2> commands_ = {};
  // Compiler requirements retained independently of the stage/compiler.
  std::array<loom_cmd_program_transient_requirement_t, 2> transients_ = {};
  // Device state shared across the variants.
  iree_hal_buffer_t* buffer_ = nullptr;
  // Upload backing retained until teardown drains accepted work.
  std::array<int32_t, 8> input_ = {100, 100, 100, 100, 100, 100, 100, 100};
  // Feedback backing retained until teardown drains accepted work.
  std::array<int32_t, 8> output_ = {};
  // VM reference type registry.
  iree_vm_environment_t* environment_ = nullptr;
  // HAL types borrowed from the environment.
  iree_hal_module_types_t types_ = {};
  // Runner module retaining the first variant.
  iree_vm_module_t* native_ = nullptr;
  // JIT bytecode module owning its image.
  iree_vm_module_t* bytecode_ = nullptr;
  // One linked model program.
  iree_vm_program_t* program_ = nullptr;
  // Aligned invocation backing.
  alignas(iree_max_align_t) std::array<uint8_t, 16384> invocation_storage_ = {};
  // Invocation borrowing its backing.
  iree_vm_invocation_t* invocation_ = nullptr;
  // One shared model process.
  iree_vm_process_t* process_ = nullptr;
  // Resolved control entry point.
  iree_vm_function_t step_ = {};
};

TEST_P(JitTest, SourceDefaultsAndOverridesShareVmAndGpuState) {
  // The same exact-output witness runs with a sink unable to accept any byte.
#if defined(__linux__)
  FullStderr blocked;
  ASSERT_EQ(blocked.Redirect(), 0);
#endif
  // A failed native task must join its accepted siblings before compiler reuse.
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Compile("3", 0, IREE_SV("invalid_native")));
  ASSERT_EQ(commands_[0], nullptr);
  EXPECT_GT(compilation_.severities[LOOMC_DIAGNOSTIC_SEVERITY_ERROR], 0u);
  ASSERT_GT(compilation_.diagnostic_count, 0u);
  EXPECT_GT(compilation_.diagnostics[0].message_length, 0u);
  IREE_ASSERT_OK(Compile(nullptr, 0));
  IREE_ASSERT_OK(Compile("7", 1));
  IREE_ASSERT_OK(PrepareVm());
  // Prepared commands must outlive all compiler/source storage.
  loom_serve_jit_destroy(jit_);
  jit_ = nullptr;
  EXPECT_EQ(compilation_.commands, 3u);
  EXPECT_EQ(compilation_.entries, 16u);
  EXPECT_EQ(compilation_.workers, GetParam());
  EXPECT_GT(compilation_.duration_ns, 0u);
  iree_hal_buffer_params_t params = {};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device_), params, sizeof(input_), &buffer_));
  iree_hal_transfer_operation_t upload = {};
  upload.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
  upload.upload.source = input_.data();
  upload.upload.target_buffer = buffer_;
  upload.upload.length = sizeof(input_);
  uint64_t completion = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(execution_, 1, &upload, &completion));
  iree_vm_variant_t arguments[] = {
      iree_hal_buffer_variant_from_ptr_borrowed(&types_, buffer_)};
  iree_vm_variant_t results[1] = {};
  auto status = iree_vm_invoke(invocation_, step_,
                               iree_vm_variant_span_from_array(arguments),
                               iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  IREE_ASSERT_OK(status);
  const iree_hal_buffer_binding_t binding = {buffer_, 0, sizeof(input_)};
  IREE_ASSERT_OK(loom_serve_execution_execute(execution_, commands_[1],
                                              {1, &binding}, &completion));
  iree_hal_transfer_operation_t download = {};
  download.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
  download.download.source_buffer = buffer_;
  download.download.target = output_.data();
  download.download.length = sizeof(output_);
  IREE_ASSERT_OK(
      loom_serve_execution_feedback(execution_, 1, &download, &completion));
  IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
  for (size_t i = 0; i < output_.size(); ++i) {
    EXPECT_EQ(output_[i], 112 + 2 * i) << "specialized element " << i;
  }
}

TEST_P(JitTest, SourceFailureOutlivesUnconstructedCatalogAndVm) {
#if defined(__linux__)
  FullStderr blocked;
  ASSERT_EQ(blocked.Redirect(), 0);
#endif
  const std::string directory =
      std::filesystem::path(FLAG_invalid_sources).parent_path().string();
  loom_serve_jit_t* invalid = nullptr;
  loom_serve_compilation_report_t report = {};
  auto status = loom_serve_jit_create(device_, dispatch_,
                                      loom_serve_device_host(device_owner_),
                                      iree_make_cstring_view(directory.c_str()),
                                      nullptr, &report, allocator_, &invalid);
  EXPECT_FALSE(iree_status_is_ok(status));
  EXPECT_EQ(invalid, nullptr);
  EXPECT_GT(report.severities[LOOMC_DIAGNOSTIC_SEVERITY_ERROR], 0u);
  ASSERT_GT(report.diagnostic_count, 0u);
  EXPECT_GT(report.diagnostics[0].line, 0u);
  EXPECT_GT(report.diagnostics[0].source_length, 0u);
  const auto message = iree_status_message(status);
  EXPECT_NE(std::string(message.data, message.size).find("invalid.loom"),
            std::string::npos);
  iree_status_free(status);
  loom_serve_jit_destroy(invalid);

  const std::string path = directory + "/invalid.loom";
  const iree_string_view_t root = IREE_SVL("broken");
  // More failures than retained slots leave complete severity counters.
  for (size_t i = 0; i < 10; ++i) {
    iree_const_byte_span_t image = {};
    status = loom_serve_jit_compile_vm(iree_make_cstring_view(path.c_str()), 1,
                                       &root, &report, allocator_, &image);
    EXPECT_FALSE(iree_status_is_ok(status));
    EXPECT_EQ(image.data, nullptr);
    iree_status_free(status);
  }
  EXPECT_EQ(report.diagnostic_count, IREE_ARRAYSIZE(report.diagnostics));
  EXPECT_GE(report.severities[LOOMC_DIAGNOSTIC_SEVERITY_ERROR], 11u);
  EXPECT_EQ(report.commands, 0u);
}

TEST_P(JitTest, CompletedCatalogsRetireWithoutClosingSharedCompilerQueue) {
  loom_serve_compilation_report_t peer_report = {};
  loom_serve_jit_t* peer = nullptr;
  auto* host = loom_serve_device_host(device_owner_);
  IREE_ASSERT_OK(loom_serve_jit_create(
      device_, dispatch_, host, iree_make_cstring_view(directory_.c_str()),
      nullptr, &peer_report, allocator_, &peer));
  const loomc_config_options_t config = {
      nullptr, 0, {}, LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED};
  Compilation jobs[2];
  jobs[0].jit = jit_;
  jobs[1].jit = peer;
  // Both catalogs are now owned through their worker completion callbacks.
  jit_ = nullptr;
  for (auto& job : jobs) {
    job.retire_catalog = true;
    job.root = IREE_SV("advance");
    job.config = &config;
    job.Start(loom_serve_host_executor(host));
  }
  auto status = iree_ok_status();
  for (size_t i = 0; i < IREE_ARRAYSIZE(jobs); ++i) {
    loom_serve_jit_stage_t* stage = nullptr;
    auto outcome = jobs[i].Await(&stage);
    EXPECT_EQ(jobs[i].jit, nullptr);
    if (iree_status_is_ok(outcome)) {
      outcome = loom_serve_jit_stage_record(
          stage, iree_hal_queue_family(dispatch_),
          IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, nullptr, &commands_[i]);
    }
    status = iree_status_join(status, outcome);
    loom_serve_jit_stage_destroy(stage);
  }
  IREE_ASSERT_OK(status);
  EXPECT_EQ(compilation_.entries, 8u);
  EXPECT_EQ(peer_report.entries, 8u);
  iree_hal_buffer_params_t params = {};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device_), params, sizeof(input_), &buffer_));
  iree_hal_transfer_operation_t upload = {};
  upload.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
  upload.upload = {input_.data(), buffer_, 0, sizeof(input_)};
  uint64_t completion = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(execution_, 1, &upload, &completion));
  const iree_hal_buffer_binding_t binding = {buffer_, 0, sizeof(input_)};
  for (auto* command : commands_) {
    IREE_ASSERT_OK(loom_serve_execution_execute(execution_, command,
                                                {1, &binding}, &completion));
  }
  iree_hal_transfer_operation_t download = {};
  download.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
  download.download = {buffer_, 0, output_.data(), sizeof(output_)};
  IREE_ASSERT_OK(
      loom_serve_execution_feedback(execution_, 1, &download, &completion));
  IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
  for (size_t i = 0; i < output_.size(); ++i) {
    EXPECT_EQ(output_[i], 110 + 2 * i);
  }
}

TEST_P(JitTest, InvalidConfigurationLeavesCompilerReusable) {
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Compile("99", 0));
  ASSERT_EQ(commands_[0], nullptr);
  IREE_ASSERT_OK(Compile("3", 0));
}

TEST_P(JitTest, InvalidRootLeavesCompilerReusable) {
  IREE_ASSERT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        Compile("3", 0, IREE_SV("absent")));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        Compile("3", 0, IREE_SV("missing")));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        Compile("3", 0, IREE_SV("ambiguous")));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Compile("3", 0, IREE_SV("not_command")));
  ASSERT_EQ(commands_[0], nullptr);
  IREE_ASSERT_OK(Compile("3", 0));
}

TEST_P(JitTest, CommandWithoutNativeRequests) {
  IREE_ASSERT_OK(Compile("3", 0, IREE_SV("idle")));
}

TEST_P(JitTest, UnequalPrivateWorkspacesPreservePoolPolicyAndOutput) {
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
  loom_serve_jit_destroy(jit_);
  jit_ = nullptr;
  iree_vm_function_t selected_step;
  IREE_ASSERT_OK(iree_vm_process_lookup_function(
      process_, IREE_SV("model"), IREE_SV("selected_step"), &selected_step));
  iree_hal_buffer_params_t params = {};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device_), params, sizeof(input_), &buffer_));
  iree_hal_transfer_operation_t upload = {};
  upload.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
  upload.upload = {input_.data(), buffer_, 0, sizeof(input_)};
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
  iree_hal_transfer_operation_t download = {};
  download.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
  download.download = {buffer_, 0, output_.data(), sizeof(output_)};
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
      .params = {.usage = IREE_HAL_BUFFER_USAGE_STORAGE |
                          IREE_HAL_BUFFER_USAGE_TRANSFER,
                 .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
                 .min_alignment = 16},
      .allocation_size = 4160};
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
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {input_.data(), held_.buffer, 0, sizeof(input_)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
       .download = {held_.buffer, 0, held_.output.data(),
                    sizeof(held_.output)}},
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
  EXPECT_EQ(trimmed.bytes_committed, 0u);
  EXPECT_EQ(trimmed.bytes_quarantined, 0u);
  if (quarantine_limit) {
    EXPECT_GT(trimmed.quarantine_eviction_count, 0u);
  }
  // Feedback owns the final output independently of private workspace reuse.
  for (int32_t value : output_) {
    EXPECT_EQ(value, expected);
  }
}

TEST_P(JitTest, IndependentModelCompletesWhileSharedWorkspaceGrows) {
  IREE_ASSERT_OK(Compile("5", 0));
  IREE_ASSERT_OK(Compile("7", 1, IREE_SV("temporary_advance"), "4160"));
  IREE_ASSERT_OK(
      loom_serve_device_create_execution(device_owner_, &held_.execution));
  IREE_ASSERT_OK(
      loom_serve_execution_reserve_workspace(held_.execution, 64, 16));
  const iree_hal_pool_reservation_request_t request = {
      .params = {.usage = IREE_HAL_BUFFER_USAGE_STORAGE |
                          IREE_HAL_BUFFER_USAGE_TRANSFER,
                 .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
                 .min_alignment = 16},
      .allocation_size = 64};
  IREE_ASSERT_OK(
      loom_serve_execution_alloca(held_.execution, &request, &held_.buffer));
  iree_hal_transfer_operation_t upload = {};
  upload.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
  upload.upload = {input_.data(), held_.buffer, 0, sizeof(input_)};
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
  iree_hal_transfer_operation_t download = {};
  download.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
  download.download = {held_.buffer, 0, held_.output.data(),
                       sizeof(held_.output)};
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
  iree_hal_buffer_params_t params = {};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device_), params, sizeof(input_), &buffer_));
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
  const auto retired =
      loom_serve_device_memory_statistics(device_owner_).workspace;
  EXPECT_EQ(retired.bytes_committed, 0u);
  EXPECT_EQ(retired.reservation_count, 0u);
  EXPECT_EQ(retired.reserve_count, 2u);
  EXPECT_EQ(retired.release_count, 2u);
}

INSTANTIATE_TEST_SUITE_P(SharedWorkers, JitTest, ::testing::Values(1, 4));

}  // namespace
