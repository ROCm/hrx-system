// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/weights.h"

#include <array>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>

#include "experimental/loom_serve/runtime/activation.h"
#include "experimental/loom_serve/runtime/device.h"
#include "experimental/loom_serve/runtime/execution.h"
#include "experimental/loom_serve/runtime/residency.h"
#include "iree/async/operations/semaphore.h"
#include "iree/base/tooling/flags.h"
#include "iree/task/executor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

IREE_FLAG(string, weight_sources, "", "Weight loader source catalog.");

namespace {

// Models the loader's owner turn: submit private policy/index/JIT preparation
// and return so that the same sole worker can execute the accepted work.
struct Preparation {
  static iree_status_t Submit(iree_task_process_t* process,
                              const iree_task_worker_context_t*,
                              iree_task_process_drain_result_t* result) {
    auto* self = static_cast<Preparation*>(process->user_data);
    auto status = loom_serve_weights_prepare_async(
        self->weights, self->host, self->jit,
        IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, self->checkpoint, self->policy,
        Complete, self);
    if (!iree_status_is_ok(status)) {
      Complete(self, status);
    }
    result->completed = true;
    result->did_work = true;
    return iree_ok_status();
  }

  static void Complete(void* user_data, iree_status_t status) {
    auto* self = static_cast<Preparation*>(user_data);
    std::lock_guard<std::mutex> lock(self->mutex);
    self->status = status;
    self->ready = true;
    self->condition.notify_one();
  }

  static void Release(iree_task_process_t* process) {
    auto* self = static_cast<Preparation*>(process->user_data);
    std::lock_guard<std::mutex> lock(self->mutex);
    self->released = true;
    self->condition.notify_one();
  }

  iree_status_t Run() {
    iree_task_process_initialize(Submit, 0, 1, &process);
    process.user_data = this;
    process.release_fn = Release;
    iree_task_executor_schedule_process(loom_serve_host_executor(host),
                                        &process);
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return ready && released; });
    return status;
  }

  // Borrowed unactivated model plan, owned by the fixture through completion.
  loom_serve_weights_t* weights = nullptr;
  // Borrowed one-worker host shared with source compilation and serving.
  loom_serve_host_t* host = nullptr;
  // Borrowed exclusive compiler catalog.
  loom_serve_jit_t* jit = nullptr;
  // Borrowed file locator retained by the outer test owner.
  iree_string_view_t checkpoint = {};
  // Borrowed source policy path retained through completion.
  iree_string_view_t policy = {};
  // Loader's finite submitting turn, not a thread or waiting worker.
  iree_task_process_t process = {};
  // Joins both terminal callbacks with the test thread's final access.
  std::mutex mutex;
  // Outer test notification; not part of the production preparation path.
  std::condition_variable condition;
  // Terminal source preparation has delivered ownership and status.
  bool ready = false;
  // Executor has made its final access to the loader's submitting process.
  bool released = false;
  // Owned terminal preparation status transferred to the outer test caller.
  iree_status_t status = iree_ok_status();
};

// The only worker submits activation and native consumption, then returns.
// Another task must run to release consumption's predecessor. Completion is a
// public proactor multi-wait, not a worker join or a driver-specific callback.
struct Activation {
  static iree_status_t Submit(iree_task_process_t* process,
                              const iree_task_worker_context_t*,
                              iree_task_process_drain_result_t* result) {
    auto* self = static_cast<Activation*>(process->user_data);
    auto status =
        loom_serve_residency_try_acquire(self->residency, &self->admitted);
    if (iree_status_is_ok(status) && !self->admitted) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "native invocation was not admitted");
    }
    uint64_t completion = 0;
    const iree_hal_buffer_binding_t binding = {self->output, 0,
                                               sizeof(*self->readback)};
    if (iree_status_is_ok(status)) {
      status = loom_serve_execution_execute(self->execution, self->command,
                                            {1, &binding}, &completion);
    }
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download = {.source_buffer = self->output,
                     .target = self->readback->data(),
                     .length = sizeof(*self->readback)}};
    if (iree_status_is_ok(status)) {
      status = loom_serve_execution_feedback(self->execution, 1, &download,
                                             &completion);
    }
    if (iree_status_is_ok(status)) {
      self->frontier = loom_serve_execution_frontier(self->execution);
      for (size_t i = 0; i < self->semaphores.size(); ++i) {
        self->semaphores[i] = reinterpret_cast<iree_async_semaphore_t*>(
            self->frontier.semaphores[i]);
      }
      self->wait.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
      self->wait.base.completion_fn = Ready;
      self->wait.base.user_data = self;
      self->wait.semaphores = self->semaphores.data();
      self->wait.values = self->frontier.values;
      self->wait.count = self->semaphores.size();
      self->wait.mode = IREE_ASYNC_WAIT_MODE_ALL;
      status = iree_async_proactor_submit_one(self->proactor, &self->wait.base);
    }
    if (!iree_status_is_ok(status)) {
      Ready(self, nullptr, status, IREE_ASYNC_COMPLETION_FLAG_NONE);
    }
    iree_task_executor_schedule_process(self->executor, &self->unblock);
    result->did_work = true;
    result->completed = true;
    return iree_ok_status();
  }

  static iree_status_t Unblock(iree_task_process_t* process,
                               const iree_task_worker_context_t*,
                               iree_task_process_drain_result_t* result) {
    auto* self = static_cast<Activation*>(process->user_data);
    auto status = iree_hal_semaphore_signal(self->gate, 1, nullptr);
    if (!iree_status_is_ok(status)) {
      iree_hal_semaphore_fail(self->gate, iree_status_clone(status));
    }
    {
      std::lock_guard<std::mutex> lock(self->mutex);
      self->status = iree_status_join(self->status, status);
    }
    result->did_work = true;
    result->completed = true;
    return iree_ok_status();
  }

  static void Ready(void* user_data, iree_async_operation_t*,
                    iree_status_t status, iree_async_completion_flags_t) {
    auto* self = static_cast<Activation*>(user_data);
    std::lock_guard<std::mutex> lock(self->mutex);
    self->status = iree_status_join(self->status, status);
    self->ready = true;
    self->condition.notify_one();
  }

  static void Release(iree_task_process_t* process) {
    auto* self = static_cast<Activation*>(process->user_data);
    std::lock_guard<std::mutex> lock(self->mutex);
    ++self->released;
    self->condition.notify_one();
  }

  iree_status_t Run() {
    iree_task_process_initialize(Submit, 0, 1, &submit);
    iree_task_process_initialize(Unblock, 0, 1, &unblock);
    for (auto* process : {&submit, &unblock}) {
      process->user_data = this;
      process->release_fn = Release;
    }
    iree_task_executor_schedule_process(executor, &submit);
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return ready && released == 2; });
    return status;
  }

  // Borrowed shared executor with exactly one worker.
  iree_task_executor_t* executor = nullptr;
  // Borrowed host proactor whose pool runner drives completion.
  iree_async_proactor_t* proactor = nullptr;
  // Borrowed model admission and activation group.
  loom_serve_residency_t* residency = nullptr;
  // Borrowed model work and feedback timelines.
  loom_serve_execution_t* execution = nullptr;
  // Borrowed native consumer, recorded before activation.
  iree_hal_command_buffer_t* command = nullptr;
  // Borrowed output storage retained through final queue retirement.
  iree_hal_buffer_t* output = nullptr;
  // Borrowed host feedback storage retained through completion.
  std::array<int32_t, 8>* readback = nullptr;
  // Borrowed explicit dependency that requires another task turn to resolve.
  iree_hal_semaphore_t* gate = nullptr;
  // Submitting model service turn.
  iree_task_process_t submit = {};
  // Independent work required for the model to complete.
  iree_task_process_t unblock = {};
  // Owned asynchronous wait storage retained through final callback.
  iree_async_semaphore_wait_operation_t wait = {};
  // Borrowed HAL timelines captured after the native consumer and feedback.
  loom_serve_execution_frontier_t frontier = {};
  // Toll-free async handles referring to the captured HAL semaphores.
  std::array<iree_async_semaphore_t*, 2> semaphores = {};
  // Joins final callback/process access with the outer test thread.
  std::mutex mutex;
  // Wakes only the outer thread, never a task worker.
  std::condition_variable condition;
  // Acquired pin remains held for the owner to consume the result.
  bool admitted = false;
  // Completion callback has published a terminal result.
  bool ready = false;
  // Number of executor-owned processes that have fully retired.
  unsigned released = 0;
  // Joined status transferred to the outer owner.
  iree_status_t status = iree_ok_status();
};

// Outer owner notification for explicit warmup. Advance joins the production
// callback's last access before this notification or the operation can die.
struct ActivationWake {
  static void Notify(void* user_data) {
    auto* self = static_cast<ActivationWake*>(user_data);
    std::lock_guard<std::mutex> lock(self->mutex);
    self->ready = true;
    self->condition.notify_one();
  }
  void Await() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this] { return ready; });
  }
  // Protects readiness publication to the outer owner.
  std::mutex mutex;
  // Outer-thread wait only; the one task worker remains available.
  std::condition_variable condition;
  // The production warmup callback has requested an owner turn.
  bool ready = false;
};

class WeightsTest : public ::testing::TestWithParam<bool> {
 protected:
  void WriteCheckpoint(const std::string& path, int32_t offset) {
    // Actual safetensors input exercises file indexing and queued reads. Each
    // tensor has eight distinct words; transformations are not idempotent.
    const std::string header =
        R"({"a":{"dtype":"I32","shape":[8],"data_offsets":[0,32]},)"
        R"("b":{"dtype":"I32","shape":[8],"data_offsets":[32,64]},)"
        R"("raw":{"dtype":"I32","shape":[8],"data_offsets":[64,96]},)"
        R"("d":{"dtype":"I32","shape":[8],"data_offsets":[96,128]}})";
    std::ofstream file(path, std::ios::binary);
    for (unsigned i = 0; i < 8; ++i) {
      file.put(
          static_cast<char>(static_cast<uint64_t>(header.size()) >> (8 * i)));
    }
    file.write(header.data(), header.size());
    for (int tensor = 1; tensor <= 4; ++tensor) {
      for (int element = 0; element < 8; ++element) {
        const int32_t value = offset + tensor * 10 + element;
        file.write(reinterpret_cast<const char*>(&value), sizeof(value));
      }
    }
    file.close();
    ASSERT_TRUE(file.good());
  }

  void SetUp() override {
    const loom_serve_device_options_t options = {
        .uri = IREE_SV("amdgpu"),
        .backing = GetParam() ? LOOM_SERVE_DEVICE_BACKING_ELASTIC
                              : LOOM_SERVE_DEVICE_BACKING_FIXED,
        .slab_size = kSlabBytes,
        .memory_limit = GetParam() ? 2 * kSlabBytes : 0};
    loom_serve_host_t* host = nullptr;
    IREE_ASSERT_OK(loom_serve_host_create(1, &host, allocator_));
    auto status =
        loom_serve_device_group_create(host, 1, &options, &group_, allocator_);
    loom_serve_host_release(host);
    IREE_ASSERT_OK(status);
    device_owner_ = loom_serve_device_group_at(group_, 0);
    device_ = loom_serve_device_handle(device_owner_);
    dispatch_ = loom_serve_device_dispatch_queue(device_owner_);
    transfer_ = loom_serve_device_transfer_queue(device_owner_);
    IREE_ASSERT_OK(
        loom_serve_device_create_execution(device_owner_, &execution_));
    pool_ = loom_serve_device_memory_pool(device_owner_);
    directory_ =
        std::filesystem::path(FLAG_weight_sources).parent_path().string();
    IREE_ASSERT_OK(loom_serve_jit_create(
        device_, dispatch_, loom_serve_device_host(device_owner_),
        iree_make_cstring_view(directory_.c_str()), nullptr, &compilation_,
        allocator_, &jit_));
    WriteCheckpoint(path_.path(), 0);
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        loom_serve_device_storage_pool(device_owner_),
        iree_hal_make_pool_reservation_request(sizeof(output_), 0),
        iree_infinite_timeout(), &output_buffer_));
  }

  void TearDown() override {
    iree_hal_buffer_release(workspace_);
    if (execution_) {
      IREE_EXPECT_OK(loom_serve_execution_drain(execution_));
    }
    for (auto* command : commands_) {
      iree_hal_command_buffer_release(command);
    }
    for (auto* stage : compiled_) {
      loom_serve_jit_stage_destroy(stage);
    }
    for (auto& stage : buffers_) {
      for (auto* buffer : stage) {
        iree_hal_buffer_release(buffer);
      }
    }
    for (auto* residency : residencies_) {
      loom_serve_residency_destroy(residency);
    }
    for (auto* plan : plans_) {
      loom_serve_weights_await_retirement(plan);
      IREE_EXPECT_OK(loom_serve_weights_destroy(plan));
    }
    IREE_EXPECT_OK(loom_serve_virtual_buffer_destroy(state_));
    iree_hal_buffer_release(output_buffer_);
    loom_serve_jit_destroy(jit_);
    loom_serve_execution_release(execution_);
    IREE_EXPECT_OK(loom_serve_device_group_destroy(group_));
    if (path_.Exists()) {
      EXPECT_TRUE(path_.Remove());
    }
    if (adapter_path_.Exists()) {
      EXPECT_TRUE(adapter_path_.Remove());
    }
  }

  iree_status_t Compile(iree_host_size_t index, const char* root) {
    const loomc_config_options_t config = {
        nullptr, 0, {}, LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED};
    return loom_serve_jit_compile(jit_, iree_make_cstring_view(root), &config,
                                  &compiled_[index]);
  }

  loom_serve_weight_root_t Root(iree_host_size_t stage_index,
                                uint32_t root_index) {
    const loom_cmd_program_t* program =
        loom_serve_jit_stage_program(compiled_[stage_index]);
    const loom_cmd_program_parameter_root_t root =
        loom_cmd_program_parameter_root_at(program, root_index);
    return {program, root, &buffers_[stage_index][root.fixed_buffer_index]};
  }

  iree_status_t CreateRoots(iree_host_size_t shared_count,
                            iree_host_size_t root_count,
                            const loom_serve_weight_root_t* roots,
                            const std::string& path) {
    const std::string policy = directory_ + "/policy.loom";
    auto** plan = &plans_[plan_count_++];
    IREE_RETURN_IF_ERROR(loom_serve_weights_create(
        device_, transfer_, dispatch_,
        loom_serve_device_storage_pool(device_owner_), pool_, shared_count,
        root_count, roots, plan, allocator_));
    const auto before = loom_serve_weights_statistics(*plan);
    Preparation preparation;
    preparation.weights = *plan;
    preparation.host = loom_serve_device_host(device_owner_);
    preparation.jit = jit_;
    preparation.checkpoint = iree_make_cstring_view(path.c_str());
    preparation.policy = iree_make_cstring_view(policy.c_str());
    auto status = preparation.Run();
    const auto after = loom_serve_weights_statistics(*plan);
    EXPECT_EQ(after.reserved_bytes, before.reserved_bytes);
    EXPECT_EQ(after.committed_bytes, before.committed_bytes);
    EXPECT_EQ(loom_serve_weights_activity(*plan).activations, 0u);
    return status;
  }

  iree_status_t LoadRoots(iree_host_size_t shared_count,
                          iree_host_size_t root_count,
                          const loom_serve_weight_root_t* roots,
                          const std::string& path) {
    IREE_RETURN_IF_ERROR(CreateRoots(shared_count, root_count, roots, path));
    auto* plan = plans_[plan_count_ - 1];
    auto status = loom_serve_weights_activate(plan, execution_);
    return iree_status_join(status, loom_serve_weights_await(plan));
  }

  iree_status_t Register(iree_host_size_t index, iree_host_size_t first_plan,
                         iree_host_size_t plan_count) {
    return loom_serve_residency_create(
        loom_serve_device_residency_cache(device_owner_), execution_,
        plan_count, plans_.data() + first_plan, &residencies_[index],
        allocator_);
  }

  void RunPinned(iree_host_size_t index, int32_t first) {
    IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[index]));
    ASSERT_NO_FATAL_FAILURE(Run(index, first));
    loom_serve_residency_release(residencies_[index]);
  }

  iree_status_t Load(iree_host_size_t shared_count, iree_host_size_t count) {
    std::array<loom_serve_weight_root_t, 6> roots = {};
    iree_host_size_t root_count = 0;
    for (iree_host_size_t i = 0; i < count; ++i) {
      const loom_cmd_program_t* program =
          loom_serve_jit_stage_program(compiled_[i]);
      for (uint32_t r = 0; r < program->parameter_roots.count; ++r) {
        roots[root_count++] = Root(i, r);
      }
    }
    return LoadRoots(shared_count, root_count, roots.data(), path_.path());
  }

  void Run(iree_host_size_t index, int32_t first) {
    if (!commands_[index]) {
      IREE_ASSERT_OK(loom_serve_jit_stage_record(
          compiled_[index], iree_hal_queue_family(dispatch_),
          IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, buffers_[index].data(),
          &commands_[index]));
    }
    const iree_hal_buffer_binding_t binding = {output_buffer_, 0,
                                               sizeof(output_)};
    uint64_t completion = 0;
    IREE_ASSERT_OK(loom_serve_execution_execute(execution_, commands_[index],
                                                {1, &binding}, &completion));
    iree_hal_transfer_operation_t download = {};
    download.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
    download.download.source_buffer = output_buffer_;
    download.download.target = output_.data();
    download.download.length = sizeof(output_);
    IREE_ASSERT_OK(
        loom_serve_execution_feedback(execution_, 1, &download, &completion));
    IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
    for (int32_t i = 0; i < 8; ++i) {
      EXPECT_EQ(output_[i], first + 3 * i);
    }
  }

  // Allocation policy for the real runtime and compiler.
  iree_allocator_t allocator_ = iree_allocator_system();
  // Copied compiler evidence retained beyond individual results.
  loom_serve_compilation_report_t compilation_ = {};
  // Application group outliving all compiler, VM and I/O resources.
  loom_serve_device_group_t* group_ = nullptr;
  // Borrowed device record within the application group.
  loom_serve_device_t* device_owner_ = nullptr;
  // Borrowed live GPU used by compilation and execution.
  iree_hal_device_t* device_ = nullptr;
  // Exact dispatch queue borrowed from device.
  iree_hal_queue_t* dispatch_ = nullptr;
  // Exact transfer queue borrowed from device.
  iree_hal_queue_t* transfer_ = nullptr;
  // Owned model timelines used by consuming commands.
  loom_serve_execution_t* execution_ = nullptr;
  // Explicitly elastic backing or NULL for the fixed comparison arm.
  loom_serve_memory_pool_t* pool_ = nullptr;
  // Physical slab size; the shared budget holds two unique roots.
  static constexpr uint64_t kSlabBytes = 2 * 1024 * 1024;
  // Owned checkpoint plans, including partially failed construction.
  std::array<loom_serve_weights_t*, 3> plans_ = {};
  // Admission groups borrowing their plans until final queue retirement.
  std::array<loom_serve_residency_t*, 3> residencies_ = {};
  // Independent live-state allocation sharing the parameter pressure policy.
  loom_serve_virtual_buffer_t* state_ = nullptr;
  // Mutable-state accounting retained through virtual storage destruction.
  loom_serve_memory_statistics_t state_statistics_ = {};
  // Number of plan slots initialized by the loading helper.
  size_t plan_count_ = 0;
  // Source catalog compiler shared across all stages and preparers.
  loom_serve_jit_t* jit_ = nullptr;
  // Source directory from runfiles.
  std::string directory_;
  // Owned test archive removed after queue resources retire.
  iree::testing::TempFilePath path_{"loom_weights", ".safetensors"};
  // Independent checkpoint with colliding keys and distinct values.
  iree::testing::TempFilePath adapter_path_{"loom_adapter", ".safetensors"};
  // Compiled immutable parameter reflections.
  std::array<loom_serve_jit_stage_t*, 3> compiled_ = {};
  // Populated owned weight roots, including partially failed loads.
  std::array<std::array<iree_hal_buffer_t*, 2>, 3> buffers_ = {};
  // Recorded consumers retaining fixed roots and executables.
  std::array<iree_hal_command_buffer_t*, 3> commands_ = {};
  // Mutable output independently allocated from immutable parameters.
  iree_hal_buffer_t* output_buffer_ = nullptr;
  // Private workspace owned until deallocation has been accepted.
  iree_hal_buffer_t* workspace_ = nullptr;
  // Readback backing retained until completion or terminal teardown.
  std::array<int32_t, 8> output_ = {};
};

TEST_P(WeightsTest, MultiplePreparersAndSharedRootsPrepareEachTensorOnce) {
  IREE_ASSERT_OK(Compile(0, "target"));
  IREE_ASSERT_OK(Compile(1, "target"));
  IREE_ASSERT_OK(Compile(2, "auxiliary"));
  IREE_ASSERT_OK(Load(2, 3));
  // Three consumer commands plus exactly two distinct source preparers.
  EXPECT_EQ(compilation_.commands, 5u);
  EXPECT_EQ(buffers_[0][0], buffers_[1][0]);
  Run(0, 70);
  Run(1, 70);
  Run(2, 93);
}

TEST_P(WeightsTest, OneWorkerActivatesAndConsumesThroughAsyncTimelineWait) {
  IREE_ASSERT_OK(Compile(0, "target"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateRoots(0, 1, &root, path_.path()));
  IREE_ASSERT_OK(Register(0, 0, 1));
  IREE_ASSERT_OK(loom_serve_jit_stage_record(
      compiled_[0], iree_hal_queue_family(dispatch_),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, buffers_[0].data(), &commands_[0]));
  auto* host = loom_serve_device_host(device_owner_);
  Activation activation;
  activation.executor = loom_serve_host_executor(host);
  IREE_ASSERT_OK(iree_async_proactor_pool_get(
      loom_serve_host_proactor_pool(host), 0, &activation.proactor));
  activation.residency = residencies_[0];
  activation.execution = execution_;
  activation.command = commands_[0];
  activation.output = output_buffer_;
  activation.readback = &output_;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &activation.gate));
  uint64_t held = 1;
  auto status =
      loom_serve_execution_barrier(execution_, {1, &activation.gate, &held});
  if (iree_status_is_ok(status)) {
    status = activation.Run();
  }
  iree_hal_semaphore_release(activation.gate);
  IREE_ASSERT_OK(status);
  ASSERT_TRUE(activation.admitted);
  for (int32_t i = 0; i < 8; ++i) {
    EXPECT_EQ(output_[i], 70 + 3 * i);
  }
  // Readiness publication retains the pin. Pressure cannot evict the result's
  // parameters before the owning scheduler consumes completion.
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(
      loom_serve_device_residency_cache(device_owner_), 0));
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes,
            GetParam() ? kSlabBytes : 0);
  loom_serve_residency_release(residencies_[0]);
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(
      loom_serve_device_residency_cache(device_owner_), 0));
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes, 0u);
  // The same compiled consumer survives a complete asynchronous reload.
  ASSERT_NO_FATAL_FAILURE(RunPinned(0, 70));
}

TEST_P(WeightsTest, CanonicalWeightsNeedNoPreparationOrLeadingSharedGroup) {
  IREE_ASSERT_OK(Compile(0, "raw"));
  IREE_ASSERT_OK(Load(0, 1));
  Run(0, 90);
}

TEST_P(WeightsTest, SeparateCheckpointDomainsRetainSharedBaseStorage) {
  WriteCheckpoint(adapter_path_.path(), 100);
  IREE_ASSERT_OK(Compile(0, "raw"));
  IREE_ASSERT_OK(Compile(1, "domains"));
  const loom_serve_weight_root_t base[] = {Root(1, 0), Root(0, 0)};
  IREE_ASSERT_OK(LoadRoots(2, 2, base, path_.path()));
  EXPECT_EQ(buffers_[0][0], buffers_[1][0]);
  EXPECT_EQ(buffers_[1][1], nullptr);
  Run(0, 90);

  const loom_serve_weight_root_t adapter = Root(1, 1);
  IREE_ASSERT_OK(LoadRoots(0, 1, &adapter, adapter_path_.path()));
  EXPECT_NE(buffers_[1][0], buffers_[1][1]);
  Run(1, 290);
  Run(1, 290);
  Run(0, 90);
}

TEST_P(WeightsTest, PolicySizeMismatchRejectsBeforeFileSubmission) {
  IREE_ASSERT_OK(Compile(0, "size"));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Load(0, 1));
}

TEST_P(WeightsTest, MissingPreparationRootFailsCompilation) {
  IREE_ASSERT_OK(Compile(0, "missing"));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_NOT_FOUND, Load(0, 1));
}

TEST_P(WeightsTest, CheckpointIndexFailureLeavesTheSharedCompilerUsable) {
  IREE_ASSERT_OK(Compile(0, "target"));
  const auto failed_root = Root(0, 0);
  IREE_ASSERT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      CreateRoots(0, 1, &failed_root, path_.path() + ".missing.safetensors"));
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).read_bytes, 0u);
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes, 0u);
  // The failed plan retains its own roots for normal owner-side cleanup while
  // another plan prepares and consumes through the same single-worker host.
  IREE_ASSERT_OK(Compile(1, "target"));
  const auto live_root = Root(1, 0);
  IREE_ASSERT_OK(LoadRoots(0, 1, &live_root, path_.path()));
  Run(1, 70);
}

TEST_P(WeightsTest, MissingFileParameterFailsTheLoad) {
  IREE_ASSERT_OK(Compile(0, "unavailable"));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_NOT_FOUND, Load(0, 1));
}

TEST_P(WeightsTest, LateMissingParameterRetiresEarlierAcceptedReads) {
  // The provider accepts separate prepared read groups for a/b before looking
  // up the absent third key. No preparation is submitted after that rejection.
  IREE_ASSERT_OK(Compile(0, "partial"));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_NOT_FOUND, Load(0, 1));
  iree_hal_buffer_release(buffers_[0][0]);
  buffers_[0][0] = nullptr;
  loom_serve_weights_await_retirement(plans_[0]);
  IREE_ASSERT_OK(loom_serve_weights_destroy(plans_[0]));
  plans_[0] = nullptr;
  if (GetParam()) {
    EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).committed_bytes, 0u);
  }
}

TEST_P(WeightsTest, RejectedActivationRetainsProtectionUntilActualRetirement) {
  IREE_ASSERT_OK(Compile(0, "partial"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateRoots(0, 1, &root, path_.path()));
  IREE_ASSERT_OK(Register(0, 0, 1));
  bool admitted = false;
  IREE_ASSERT_STATUS_IS(IREE_STATUS_NOT_FOUND, loom_serve_residency_try_acquire(
                                                   residencies_[0], &admitted));
  EXPECT_FALSE(admitted);
  // The rejection can precede earlier accepted file reads' final accesses.
  // It preserves a terminal pin even though no invocation was admitted.
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(
      loom_serve_device_residency_cache(device_owner_), 0));
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes,
            GetParam() ? kSlabBytes : 0);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        loom_serve_weights_await(plans_[0]));
  loom_serve_residency_destroy(residencies_[0]);
  residencies_[0] = nullptr;
  iree_hal_buffer_release(buffers_[0][0]);
  buffers_[0][0] = nullptr;
  loom_serve_weights_await_retirement(plans_[0]);
  IREE_ASSERT_OK(loom_serve_weights_destroy(plans_[0]));
  plans_[0] = nullptr;
  if (GetParam()) {
    EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).committed_bytes, 0u);
  }
}

TEST_P(WeightsTest, CachedConsumersSurviveBackingReuseAndPreparationReplay) {
  IREE_ASSERT_OK(Compile(0, "target"));
  IREE_ASSERT_OK(Compile(1, "target"));
  IREE_ASSERT_OK(Compile(2, "auxiliary"));
  IREE_ASSERT_OK(Load(2, 3));
  const auto first_activity = loom_serve_weights_activity(plans_[0]);
  EXPECT_EQ(first_activity.activations, 1u);
  EXPECT_EQ(first_activity.deactivations, 0u);
  EXPECT_GT(first_activity.read_bytes, 0u);
  EXPECT_GT(first_activity.prepared_bytes, 0u);
  EXPECT_GT(first_activity.groups, 0u);
  Run(0, 70);
  Run(2, 93);
  auto* const root = buffers_[0][0];
  auto* const command = commands_[0];
  if (!GetParam()) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                          loom_serve_weights_deactivate(plans_[0]));
    Run(0, 70);
    return;
  }
  IREE_ASSERT_OK(loom_serve_weights_deactivate(plans_[0]));
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes, 0u);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).committed_bytes, 0u);
  loom_serve_virtual_buffer_t* other = nullptr;
  loom_serve_memory_statistics_t other_statistics = {};
  IREE_ASSERT_OK(loom_serve_virtual_buffer_create(pool_, 2 * kSlabBytes, 256,
                                                  &other_statistics, &other));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(other, 0, 2 * kSlabBytes));
  // Another allocation consumes every physical byte the model gave up.
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).committed_bytes,
            2 * kSlabBytes);
  const uint32_t poison = 0xDEADBEEF;
  const iree_hal_transfer_operation_t fill = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
      .fill = {.target_buffer = loom_serve_virtual_buffer_handle(other),
               .length = 2 * kSlabBytes,
               .pattern = &poison,
               .pattern_length = sizeof(poison)}};
  uint64_t completion = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(execution_, 1, &fill, &completion));
  IREE_ASSERT_OK(loom_serve_execution_wait(execution_, completion));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_destroy(other));
  IREE_ASSERT_OK(loom_serve_weights_activate(plans_[0], execution_));
  IREE_ASSERT_OK(loom_serve_weights_activate(plans_[0], execution_));
  const auto replay_activity = loom_serve_weights_activity(plans_[0]);
  EXPECT_EQ(replay_activity.activations, 2u);
  EXPECT_EQ(replay_activity.deactivations, 1u);
  EXPECT_EQ(replay_activity.read_bytes, first_activity.read_bytes * 2);
  EXPECT_EQ(replay_activity.prepared_bytes, first_activity.prepared_bytes * 2);
  EXPECT_EQ(replay_activity.groups, first_activity.groups * 2);
  EXPECT_EQ(buffers_[0][0], root);
  EXPECT_EQ(commands_[0], command);
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes,
            2 * kSlabBytes);
  Run(0, 70);
  Run(1, 70);
  Run(2, 93);
}

TEST_P(WeightsTest, AdmissionUsesRecencyOnlyAmongUnpinnedGroups) {
  for (iree_host_size_t i = 0; i < 3; ++i) {
    IREE_ASSERT_OK(Compile(i, "target"));
    const auto root = Root(i, 0);
    IREE_ASSERT_OK(CreateRoots(0, 1, &root, path_.path()));
    IREE_ASSERT_OK(Register(i, i, 1));
  }
  ASSERT_NO_FATAL_FAILURE(RunPinned(0, 70));
  ASSERT_NO_FATAL_FAILURE(RunPinned(1, 70));
  // Touch the oldest entry, making the other resident entry the LRU victim.
  ASSERT_NO_FATAL_FAILURE(RunPinned(0, 70));
  ASSERT_NO_FATAL_FAILURE(RunPinned(2, 70));
  if (!GetParam()) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                          loom_serve_residency_deactivate(residencies_[0]));
    ASSERT_NO_FATAL_FAILURE(RunPinned(1, 70));
    return;
  }
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes,
            kSlabBytes);
  EXPECT_EQ(loom_serve_weights_statistics(plans_[1]).committed_bytes, 0u);
  EXPECT_EQ(loom_serve_weights_statistics(plans_[2]).committed_bytes,
            kSlabBytes);
  IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[0]));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_serve_residency_deactivate(residencies_[0]));
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(
      loom_serve_device_residency_cache(device_owner_), 0));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).committed_bytes,
            kSlabBytes);
  EXPECT_EQ(loom_serve_weights_statistics(plans_[2]).committed_bytes, 0u);
  ASSERT_NO_FATAL_FAILURE(Run(0, 70));
  loom_serve_residency_release(residencies_[0]);
  ASSERT_NO_FATAL_FAILURE(RunPinned(1, 70));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).peak_bytes,
            2 * kSlabBytes);
}

TEST_P(WeightsTest, ImpossibleParameterGroupCannotJoinPressureAdmission) {
  for (uint32_t i = 0; i < 3; ++i) {
    IREE_ASSERT_OK(Compile(i, "raw"));
    const auto root = Root(i, 0);
    IREE_ASSERT_OK(CreateRoots(0, 1, &root, path_.path()));
  }
  if (!GetParam()) {
    IREE_ASSERT_OK(Register(0, 0, 3));
    return;
  }
  EXPECT_EQ(loom_serve_memory_pool_limit(pool_), 2 * kSlabBytes);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Register(0, 0, 3));
  EXPECT_EQ(residencies_[0], nullptr);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).committed_bytes, 0u);
  // Rejection does not poison the shared admission queue or borrowed plans.
  IREE_ASSERT_OK(Register(0, 0, 1));
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  ASSERT_NO_FATAL_FAILURE(Run(0, 90));
  loom_serve_residency_release(residencies_[0]);
}

TEST_P(WeightsTest, PersistentPinningOnlyChangesParameterEvictionEligibility) {
  IREE_ASSERT_OK(Compile(0, "target"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateRoots(0, 1, &root, path_.path()));
  IREE_ASSERT_OK(Register(0, 0, 1));
  auto* residency = residencies_[0];
  auto* cache = loom_serve_device_residency_cache(device_owner_);
  loom_serve_residency_set_pinned(residency, true);
  loom_serve_residency_set_pinned(residency, true);
  EXPECT_FALSE(loom_serve_residency_snapshot(residency).active);
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).activations, 0u);
  ASSERT_NO_FATAL_FAILURE(RunPinned(0, 70));
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(cache, 0));
  EXPECT_TRUE(loom_serve_residency_snapshot(residency).active);
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes,
            GetParam() ? kSlabBytes : 0u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_serve_residency_deactivate(residency));
  loom_serve_residency_set_pinned(residency, false);
  loom_serve_residency_set_pinned(residency, false);
  EXPECT_TRUE(loom_serve_residency_snapshot(residency).active);
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(cache, 0));
  EXPECT_EQ(loom_serve_residency_snapshot(residency).active, !GetParam());
  ASSERT_NO_FATAL_FAILURE(RunPinned(0, 70));
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).activations,
            GetParam() ? 2u : 1u);
}

TEST_P(WeightsTest, ExplicitActivationCancellationJoinsAcceptedReadiness) {
  IREE_ASSERT_OK(Compile(0, "target"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateRoots(0, 1, &root, path_.path()));
  IREE_ASSERT_OK(Register(0, 0, 1));
  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &gate));
  uint64_t gate_value = 1;
  IREE_ASSERT_OK(
      loom_serve_execution_barrier(execution_, {1, &gate, &gate_value}));
  ActivationWake wake;
  loom_serve_activation_t* activation = nullptr;
  IREE_ASSERT_OK(loom_serve_activation_create(
      residencies_[0],
      loom_serve_host_proactor_pool(loom_serve_device_host(device_owner_)),
      ActivationWake::Notify, &wake, &activation, allocator_));
  loom_serve_activation_phase_t phase;
  IREE_ASSERT_OK(loom_serve_activation_advance(activation, 0, &phase));
  EXPECT_EQ(phase, LOOM_SERVE_ACTIVATION_LOADING);
  // Removing administrative retention cannot release accepted native work.
  loom_serve_residency_set_pinned(residencies_[0], true);
  loom_serve_residency_set_pinned(residencies_[0], false);
  IREE_ASSERT_OK(loom_serve_activation_advance(
      activation, LOOM_SERVE_ACTIVATION_FLAG_CANCEL, &phase));
  EXPECT_EQ(phase, LOOM_SERVE_ACTIVATION_LOADING);
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(
      loom_serve_device_residency_cache(device_owner_), 0));
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes,
            GetParam() ? kSlabBytes : 0u);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, gate_value, nullptr));
  wake.Await();
  IREE_ASSERT_OK(loom_serve_activation_advance(
      activation, LOOM_SERVE_ACTIVATION_FLAG_CANCEL, &phase));
  EXPECT_EQ(phase, LOOM_SERVE_ACTIVATION_COMPLETE);
  loom_serve_activation_destroy(activation);
  iree_hal_semaphore_release(gate);
  ASSERT_NO_FATAL_FAILURE(RunPinned(0, 70));
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).activations, 1u);
  // Warmup owns no persistent pin after completion.
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(
      loom_serve_device_residency_cache(device_owner_), 0));
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes, 0u);
}

TEST_P(WeightsTest, CancelledWarmupWithdrawsCapacityInterestWithoutLoading) {
  if (!GetParam()) {
    // Fixed allocations have no serving-imposed physical ceiling.
    return;
  }
  for (uint32_t i = 0; i < 3; ++i) {
    IREE_ASSERT_OK(Compile(i, "raw"));
    const auto root = Root(i, 0);
    IREE_ASSERT_OK(CreateRoots(0, 1, &root, path_.path()));
  }
  IREE_ASSERT_OK(Register(0, 0, 2));
  IREE_ASSERT_OK(Register(1, 2, 1));
  IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[0]));
  IREE_ASSERT_OK(loom_serve_residency_await(residencies_[0]));
  ActivationWake wake;
  loom_serve_activation_t* activation = nullptr;
  IREE_ASSERT_OK(loom_serve_activation_create(
      residencies_[1],
      loom_serve_host_proactor_pool(loom_serve_device_host(device_owner_)),
      ActivationWake::Notify, &wake, &activation, allocator_));
  loom_serve_activation_phase_t phase;
  IREE_ASSERT_OK(loom_serve_activation_advance(activation, 0, &phase));
  EXPECT_EQ(phase, LOOM_SERVE_ACTIVATION_WAITING);
  EXPECT_EQ(loom_serve_weights_activity(plans_[2]).activations, 0u);
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED);
  IREE_ASSERT_OK(loom_serve_activation_advance(
      activation, LOOM_SERVE_ACTIVATION_FLAG_CANCEL, &phase));
  EXPECT_EQ(phase, LOOM_SERVE_ACTIVATION_COMPLETE);
  loom_serve_activation_destroy(activation);
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  loom_serve_residency_release(residencies_[0]);
  loom_serve_residency_release(residencies_[0]);
  // A new operation can now reclaim the unpinned peer and finish normally.
  IREE_ASSERT_OK(loom_serve_activation_create(
      residencies_[1],
      loom_serve_host_proactor_pool(loom_serve_device_host(device_owner_)),
      ActivationWake::Notify, &wake, &activation, allocator_));
  IREE_ASSERT_OK(loom_serve_activation_advance(activation, 0, &phase));
  EXPECT_EQ(phase, LOOM_SERVE_ACTIVATION_LOADING);
  wake.Await();
  IREE_ASSERT_OK(loom_serve_activation_advance(activation, 0, &phase));
  EXPECT_EQ(phase, LOOM_SERVE_ACTIVATION_COMPLETE);
  loom_serve_activation_destroy(activation);
  IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[1]));
  ASSERT_NO_FATAL_FAILURE(Run(2, 90));
  loom_serve_residency_release(residencies_[1]);
}

TEST_P(WeightsTest, RejectedWarmupKeepsAcceptedLoadsProtectedUntilRetirement) {
  IREE_ASSERT_OK(Compile(0, "partial"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateRoots(0, 1, &root, path_.path()));
  IREE_ASSERT_OK(Register(0, 0, 1));
  ActivationWake wake;
  loom_serve_activation_t* activation = nullptr;
  IREE_ASSERT_OK(loom_serve_activation_create(
      residencies_[0],
      loom_serve_host_proactor_pool(loom_serve_device_host(device_owner_)),
      ActivationWake::Notify, &wake, &activation, allocator_));
  loom_serve_activation_phase_t phase;
  IREE_ASSERT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        loom_serve_activation_advance(activation, 0, &phase));
  EXPECT_EQ(phase, LOOM_SERVE_ACTIVATION_COMPLETE);
  loom_serve_activation_destroy(activation);
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(
      loom_serve_device_residency_cache(device_owner_), 0));
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes,
            GetParam() ? kSlabBytes : 0u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        loom_serve_weights_await(plans_[0]));
}

TEST_P(WeightsTest, CheckpointGroupsAdmitAtomicallyAndRetentionPinsNest) {
  WriteCheckpoint(adapter_path_.path(), 100);
  IREE_ASSERT_OK(Compile(0, "raw"));
  IREE_ASSERT_OK(Compile(1, "domains"));
  const auto first = Root(0, 0);
  IREE_ASSERT_OK(CreateRoots(0, 1, &first, path_.path()));
  IREE_ASSERT_OK(Register(0, 0, 1));
  for (uint32_t i = 0; i < 2; ++i) {
    const auto root = Root(1, i);
    IREE_ASSERT_OK(
        CreateRoots(0, 1, &root, i ? adapter_path_.path() : path_.path()));
  }
  IREE_ASSERT_OK(Register(1, 1, 2));
  auto check_activity = [&](uint64_t activations, uint64_t deactivations) {
    const auto total = loom_serve_residency_activity(residencies_[1]);
    const auto base = loom_serve_weights_activity(plans_[1]);
    const auto adapter = loom_serve_weights_activity(plans_[2]);
    EXPECT_EQ(total.activations, activations);
    EXPECT_EQ(total.deactivations, deactivations);
    EXPECT_EQ(total.read_bytes, base.read_bytes + adapter.read_bytes);
    EXPECT_EQ(total.prepared_bytes,
              base.prepared_bytes + adapter.prepared_bytes);
    EXPECT_EQ(total.groups, base.groups + adapter.groups);
  };
  check_activity(0, 0);
  IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[0]));
  ASSERT_NO_FATAL_FAILURE(RunPinned(0, 90));
  bool admitted = false;
  IREE_ASSERT_OK(loom_serve_residency_try_acquire(residencies_[1], &admitted));
  if (!GetParam()) {
    ASSERT_TRUE(admitted);
    check_activity(2, 0);
    ASSERT_NO_FATAL_FAILURE(Run(1, 290));
    loom_serve_residency_release(residencies_[1]);
    loom_serve_residency_release(residencies_[0]);
    return;
  }
  EXPECT_FALSE(admitted);
  check_activity(0, 0);
  EXPECT_EQ(loom_serve_weights_statistics(plans_[1]).peak_bytes, 0u);
  EXPECT_EQ(loom_serve_weights_statistics(plans_[2]).peak_bytes, 0u);
  // Failed admission neither consumes its own pin nor releases retention.
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes,
            kSlabBytes);
  loom_serve_residency_release(residencies_[0]);
  IREE_ASSERT_OK(loom_serve_residency_try_acquire(residencies_[1], &admitted));
  ASSERT_TRUE(admitted);
  check_activity(2, 0);
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes, 0u);
  ASSERT_NO_FATAL_FAILURE(Run(1, 290));
  loom_serve_residency_release(residencies_[1]);
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(
      loom_serve_device_residency_cache(device_owner_), 0));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).committed_bytes, 0u);
  check_activity(2, 2);
}

TEST_P(WeightsTest, PressureAdmissionDrainsNewPinsButNotAcceptedConsumers) {
  WriteCheckpoint(adapter_path_.path(), 100);
  IREE_ASSERT_OK(Compile(0, "raw"));
  IREE_ASSERT_OK(Compile(1, "domains"));
  const auto first = Root(0, 0);
  IREE_ASSERT_OK(CreateRoots(0, 1, &first, path_.path()));
  IREE_ASSERT_OK(Register(0, 0, 1));
  for (uint32_t i = 0; i < 2; ++i) {
    const auto root = Root(1, i);
    IREE_ASSERT_OK(
        CreateRoots(0, 1, &root, i ? adapter_path_.path() : path_.path()));
  }
  IREE_ASSERT_OK(Register(1, 1, 2));
  IREE_ASSERT_OK(Register(2, 0, 0));
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  ASSERT_NO_FATAL_FAILURE(Run(0, 90));
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[1], &admission));
  if (!GetParam()) {
    ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
    ASSERT_NO_FATAL_FAILURE(Run(1, 290));
    loom_serve_residency_release(residencies_[1]);
    loom_serve_residency_release(residencies_[0]);
    return;
  }
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED);
  // The already-admitted consumer still takes its nested execution pin and
  // produces exact device output. A parameter-free peer cannot obstruct it.
  ASSERT_NO_FATAL_FAILURE(RunPinned(0, 90));
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[2], &admission));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  loom_serve_residency_release(residencies_[2]);
  // Withdraw both tail and head, including repeated withdrawal. No owned
  // invocation pin is released by cancellation of a new request.
  loom_serve_residency_cancel_admission(residencies_[0]);
  loom_serve_residency_cancel_admission(residencies_[0]);
  loom_serve_residency_cancel_admission(residencies_[1]);
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  loom_serve_residency_release(residencies_[0]);
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[1], &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED);
  loom_serve_residency_release(residencies_[0]);
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[1], &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  ASSERT_NO_FATAL_FAILURE(Run(1, 290));
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
  loom_serve_residency_release(residencies_[1]);
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  ASSERT_NO_FATAL_FAILURE(Run(0, 90));
  loom_serve_residency_release(residencies_[0]);
}

TEST_P(WeightsTest, NewAdmissionsCoexistWithoutPhysicalPressure) {
  loom_serve_residency_admission_t admission;
  for (iree_host_size_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(Compile(i, "target"));
    const auto root = Root(i, 0);
    IREE_ASSERT_OK(CreateRoots(0, 1, &root, path_.path()));
    IREE_ASSERT_OK(Register(i, i, 1));
    IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[i], &admission));
    ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  }
  for (iree_host_size_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[i], &admission));
    ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
    ASSERT_NO_FATAL_FAILURE(Run(i, 70));
    loom_serve_residency_release(residencies_[i]);
    loom_serve_residency_release(residencies_[i]);
  }
}

TEST_P(WeightsTest, PinnedWeightsDeferIdleCacheReclamationUntilItCanHelp) {
  if (!GetParam()) {
    return;
  }
  IREE_ASSERT_OK(Compile(0, "domains"));
  for (uint32_t i = 0; i < 2; ++i) {
    const auto root = Root(0, i);
    IREE_ASSERT_OK(CreateRoots(0, 1, &root, path_.path()));
  }
  IREE_ASSERT_OK(Register(0, 0, 2));
  IREE_ASSERT_OK(Register(1, 0, 0));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_create(pool_, kSlabBytes, 256,
                                                  &state_statistics_, &state_));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(state_, 0, kSlabBytes));
  struct CacheOwner {
    // Logical ownership can be released once, independently of physical trim.
    bool retained = true;
    // Accepted maintenance keeps backing alive until its owner retires it.
    bool pending = false;
    // Counts requests, not repeated probes while retirement is pending.
    uint32_t requests = 0;
  } owner;
  loom_serve_residency_set_reclaimer(
      residencies_[1],
      {&owner,
       [](void* self) {
         auto* owner = static_cast<CacheOwner*>(self);
         if (!owner->retained) {
           return false;
         }
         owner->retained = false;
         owner->pending = true;
         ++owner->requests;
         return true;
       },
       [](void* self) { return static_cast<CacheOwner*>(self)->pending; }});
  auto* cache = loom_serve_device_residency_cache(device_owner_);
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(Compile(1, "raw"));
  const auto peer_root = Root(1, 0);
  IREE_ASSERT_OK(CreateRoots(0, 1, &peer_root, path_.path()));
  IREE_ASSERT_OK(Register(2, 2, 1));
  IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[2]));
  ASSERT_NO_FATAL_FAILURE(Run(1, 90));
  // Even discarding every idle history cannot admit the two-slab group while
  // this one-slab peer is pinned. Neither kind of pin permits useful eviction.
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
  EXPECT_EQ(owner.requests, 0u);
  EXPECT_TRUE(owner.retained);
  EXPECT_FALSE(loom_serve_residency_cache_is_reclaiming(cache));
  EXPECT_EQ(state_statistics_.committed_bytes, kSlabBytes);
  loom_serve_residency_set_pinned(residencies_[2], true);
  loom_serve_residency_release(residencies_[2]);
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
  EXPECT_EQ(owner.requests, 0u);
  EXPECT_TRUE(owner.retained);
  EXPECT_FALSE(loom_serve_residency_cache_is_reclaiming(cache));
  loom_serve_residency_set_pinned(residencies_[2], false);
  // Once the peer's parameters become evictable, retiring the idle history is
  // necessary and sufficient. The regular asynchronous reclaimer must run.
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED);
  EXPECT_TRUE(loom_serve_residency_cache_is_reclaiming(cache));
  EXPECT_EQ(state_statistics_.committed_bytes, kSlabBytes);
  EXPECT_EQ(loom_serve_residency_statistics(residencies_[0]).committed_bytes,
            0u);
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED);
  EXPECT_EQ(owner.requests, 1u);
  // Cancelling admission does not pretend accepted maintenance has retired.
  loom_serve_residency_cancel_admission(residencies_[0]);
  EXPECT_TRUE(loom_serve_residency_cache_is_reclaiming(cache));
  EXPECT_EQ(state_statistics_.committed_bytes, kSlabBytes);
  loom_serve_virtual_buffer_begin_trim(state_);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(state_));
  owner.pending = false;
  EXPECT_FALSE(loom_serve_residency_cache_is_reclaiming(cache));
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  ASSERT_NO_FATAL_FAILURE(Run(0, 90));
  loom_serve_residency_release(residencies_[0]);
  loom_serve_residency_set_reclaimer(residencies_[1], {});
  EXPECT_EQ(owner.requests, 1u);
}

TEST_P(WeightsTest, MutableGrowthReclaimsIdleParametersWithoutDiscardingState) {
  IREE_ASSERT_OK(Compile(0, "target"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateRoots(0, 1, &root, path_.path()));
  IREE_ASSERT_OK(Register(0, 0, 1));
  ASSERT_NO_FATAL_FAILURE(RunPinned(0, 70));
  if (!GetParam()) {
    return;
  }
  IREE_ASSERT_OK(loom_serve_virtual_buffer_create(pool_, 2 * kSlabBytes, 256,
                                                  &state_statistics_, &state_));
  IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[0]));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_serve_virtual_buffer_commit(state_, 0, 2 * kSlabBytes));
  EXPECT_EQ(state_statistics_.committed_bytes, 0u);
  loom_serve_residency_release(residencies_[0]);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(state_, 0, 2 * kSlabBytes));
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes, 0u);
  const uint32_t value = 0xA1B2C3D4;
  uint32_t observed = 0;
  const iree_hal_transfer_operation_t fill = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
      .fill = {.target_buffer = loom_serve_virtual_buffer_handle(state_),
               .length = sizeof(value),
               .pattern = &value,
               .pattern_length = sizeof(value)}};
  uint64_t completion = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(execution_, 1, &fill, &completion));
  IREE_ASSERT_OK(loom_serve_execution_wait(execution_, completion));
  bool admitted = true;
  IREE_ASSERT_OK(loom_serve_residency_try_acquire(residencies_[0], &admitted));
  EXPECT_FALSE(admitted);
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(
      loom_serve_device_residency_cache(device_owner_), 0));
  EXPECT_EQ(state_statistics_.committed_bytes, 2 * kSlabBytes);
  const iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download = {.source_buffer = loom_serve_virtual_buffer_handle(state_),
                   .target = &observed,
                   .length = sizeof(observed)}};
  IREE_ASSERT_OK(
      loom_serve_execution_feedback(execution_, 1, &download, &completion));
  IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
  EXPECT_EQ(observed, value);
  loom_serve_virtual_buffer_begin_trim(state_);
  loom_serve_virtual_buffer_keep(state_, 0, sizeof(value));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(state_));
  ASSERT_NO_FATAL_FAILURE(RunPinned(0, 70));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).committed_bytes,
            2 * kSlabBytes);
}

TEST_P(WeightsTest,
       DeviceTrimPrefersIdleWorkspaceAndPreservesPinnedParameters) {
  for (iree_host_size_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(Compile(i, "target"));
    const auto root = Root(i, 0);
    IREE_ASSERT_OK(CreateRoots(0, 1, &root, path_.path()));
    IREE_ASSERT_OK(Register(i, i, 1));
    ASSERT_NO_FATAL_FAILURE(RunPinned(i, 70));
  }
  IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[0]));
  IREE_ASSERT_OK(
      loom_serve_execution_reserve_workspace(execution_, kSlabBytes, 256));
  const iree_hal_pool_reservation_request_t request = {
      .allocation_size = kSlabBytes, .min_alignment = 256};
  IREE_ASSERT_OK(
      loom_serve_execution_alloca(execution_, &request, &workspace_));
  uint64_t completion = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_dealloca(execution_, workspace_, &completion));
  iree_hal_buffer_release(workspace_);
  workspace_ = nullptr;
  IREE_ASSERT_OK(loom_serve_execution_drain(execution_));
  const auto before = loom_serve_device_memory_statistics(device_owner_);
  EXPECT_EQ(before.retained.committed_bytes, GetParam() ? 2 * kSlabBytes : 0);
  EXPECT_EQ(before.workspace.bytes_committed, kSlabBytes);
  IREE_ASSERT_OK(
      loom_serve_device_trim(device_owner_, before.retained.committed_bytes));
  const auto idle = loom_serve_device_memory_statistics(device_owner_);
  EXPECT_EQ(idle.retained.committed_bytes, before.retained.committed_bytes);
  EXPECT_EQ(idle.workspace.bytes_committed, 0u);
  ASSERT_NO_FATAL_FAILURE(Run(0, 70));
  ASSERT_NO_FATAL_FAILURE(Run(1, 70));

  IREE_ASSERT_OK(
      loom_serve_execution_alloca(execution_, &request, &workspace_));
  // The model's real command completion follows allocation on the work chain.
  ASSERT_NO_FATAL_FAILURE(Run(0, 70));
  IREE_ASSERT_OK(loom_serve_device_trim(device_owner_, kSlabBytes - 1));
  const auto pinned = loom_serve_device_memory_statistics(device_owner_);
  EXPECT_EQ(pinned.workspace.bytes_committed, kSlabBytes);
  EXPECT_EQ(pinned.workspace.reservation_count, 1u);
  EXPECT_EQ(pinned.retained.committed_bytes, GetParam() ? kSlabBytes : 0);
  if (GetParam()) {
    EXPECT_EQ(loom_serve_weights_statistics(plans_[1]).committed_bytes, 0u);
  }
  ASSERT_NO_FATAL_FAILURE(Run(0, 70));
  loom_serve_residency_release(residencies_[0]);
  IREE_ASSERT_OK(
      loom_serve_execution_dealloca(execution_, workspace_, &completion));
  iree_hal_buffer_release(workspace_);
  workspace_ = nullptr;
  IREE_ASSERT_OK(loom_serve_execution_drain(execution_));
  IREE_ASSERT_OK(loom_serve_device_trim(device_owner_, 0));
  const auto cold = loom_serve_device_memory_statistics(device_owner_);
  EXPECT_EQ(cold.retained.committed_bytes, 0u);
  EXPECT_EQ(cold.workspace.bytes_committed, 0u);
  ASSERT_NO_FATAL_FAILURE(RunPinned(1, 70));
  EXPECT_EQ(loom_serve_device_memory_statistics(device_owner_)
                .retained.committed_bytes,
            GetParam() ? kSlabBytes : 0);
}

INSTANTIATE_TEST_SUITE_P(Backing, WeightsTest, ::testing::Values(false, true));

}  // namespace
