// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PARAMETER_WEIGHTS_TEST_BASE_H_
#define LOOM_SERVE_RUNTIME_PARAMETER_WEIGHTS_TEST_BASE_H_

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <mutex>
#include <vector>

#include "iree/async/operations/semaphore.h"
#include "iree/hal/testing/deferred_pool.h"
#include "iree/task/executor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/runtime/parameter/activation.h"
#include "loom/serve/runtime/parameter/residency.h"
#include "loom/serve/runtime/parameter/sources.h"
#include "loom/serve/runtime/parameter/weights.h"
#include "loom/serve/testing/compiler.h"
#include "loom/serve/testing/device.h"
#include "loom/serve/testing/native_allocator.h"

namespace loom_serve::testing {

// Source catalog selected by the owning hardware test target.
const char* WeightSourcePath();
// Optional URI including feedback requirements selected by the test target.
const char* WeightDeviceUri();

// Borrows the current immutable backing configuration. All previous borrowers
// must have retired before acquiring another configuration; the environment
// releases its native device and allocator before constructing the replacement.
iree_status_t AcquireWeightDeviceGroup(bool elastic, uint64_t slab_size,
                                       uint64_t memory_limit,
                                       loom_serve_device_group_t** out_group);

// Models the loader's owner turn: submit private policy/JIT preparation
// and return so that the same sole worker can execute the accepted work.
struct loom_serve_test_weight_preparation_t {
  static iree_status_t Submit(iree_task_process_t* process,
                              const iree_task_worker_context_t* context,
                              iree_task_process_drain_result_t* result) {
    auto* self =
        static_cast<loom_serve_test_weight_preparation_t*>(process->user_data);
    auto status = loom_serve_weights_prepare_async(
        self->weights, self->host, self->jit,
        IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, &self->sources, &self->parameter,
        self->indexed_sources, Complete, self);
    if (!iree_status_is_ok(status)) {
      Complete(self, status);
    }
    result->completed = true;
    result->did_work = true;
    return iree_ok_status();
  }

  static void Complete(void* user_data, iree_status_t status) {
    auto* self = static_cast<loom_serve_test_weight_preparation_t*>(user_data);
    std::lock_guard<std::mutex> lock(self->mutex);
    self->status = status;
    self->ready = true;
    self->condition.notify_one();
  }

  static void Release(iree_task_process_t* process) {
    auto* self =
        static_cast<loom_serve_test_weight_preparation_t*>(process->user_data);
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
  // Explicit local namespace borrowed through asynchronous preparation.
  const loom_serve_sources_t sources = {.base_directory = IREE_SV(".")};
  // Borrowed source declarations retained by the outer owner through
  // completion.
  loom_serve_preparation_parameter_t parameter = {};
  // Borrowed immutable catalog opened by cold registration before policy work.
  loom_serve_parameter_sources_t* indexed_sources = nullptr;
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
struct loom_serve_test_weight_activation_t {
  static iree_status_t Submit(iree_task_process_t* process,
                              const iree_task_worker_context_t* context,
                              iree_task_process_drain_result_t* result) {
    auto* self =
        static_cast<loom_serve_test_weight_activation_t*>(process->user_data);
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
        .download =
            {
                .source_buffer = self->output,
                .target = self->readback->data(),
                .length = sizeof(*self->readback),
            },
    };
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
                               const iree_task_worker_context_t* context,
                               iree_task_process_drain_result_t* result) {
    auto* self =
        static_cast<loom_serve_test_weight_activation_t*>(process->user_data);
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

  static void Ready(void* user_data, iree_async_operation_t* operation,
                    iree_status_t status, iree_async_completion_flags_t flags) {
    auto* self = static_cast<loom_serve_test_weight_activation_t*>(user_data);
    std::lock_guard<std::mutex> lock(self->mutex);
    self->status = iree_status_join(self->status, status);
    self->ready = true;
    self->condition.notify_one();
  }

  static void Release(iree_task_process_t* process) {
    auto* self =
        static_cast<loom_serve_test_weight_activation_t*>(process->user_data);
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
struct loom_serve_test_weight_activation_wake_t {
  static void Notify(void* user_data) {
    auto* self =
        static_cast<loom_serve_test_weight_activation_wake_t*>(user_data);
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
  virtual uint64_t SlabSize() const { return kSlabBytes; }

  virtual void SetUpDevice() {
    IREE_ASSERT_OK(AcquireWeightDeviceGroup(GetParam(), SlabSize(),
                                            2 * SlabSize(), &group_));
  }

  void WriteCheckpoint(const std::string& path, int32_t offset,
                       size_t first_tensor = 0, size_t tensor_count = 4,
                       size_t element_count = 8) {
    // Actual safetensors input exercises file indexing and queued reads.
    // Distinct words make the preparation transformations non-idempotent.
    const char* keys[] = {"a", "b", "raw", "d", "missing"};
    const size_t byte_length = element_count * sizeof(int32_t);
    std::string header = "{";
    for (size_t i = 0; i < tensor_count; ++i) {
      if (i) {
        header += ",";
      }
      header += "\"" + std::string(keys[first_tensor + i]) +
                "\":{\"dtype\":\"I32\",\"shape\":[" +
                std::to_string(element_count) + "],\"data_offsets\":[" +
                std::to_string(i * byte_length) + "," +
                std::to_string((i + 1) * byte_length) + "]}";
    }
    header += "}";
    std::ofstream file(path, std::ios::binary);
    for (unsigned i = 0; i < 8; ++i) {
      file.put(
          static_cast<char>(static_cast<uint64_t>(header.size()) >> (8 * i)));
    }
    file.write(header.data(), header.size());
    for (size_t i = 0; i < tensor_count; ++i) {
      for (size_t element = 0; element < element_count; ++element) {
        const int32_t value = offset + (first_tensor + i + 1) * 10 + element;
        file.write(reinterpret_cast<const char*>(&value), sizeof(value));
      }
    }
    file.close();
    ASSERT_TRUE(file.good());
  }

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(SetUpDevice());
    device_owner_ = loom_serve_device_group_at(group_, 0);
    device_ = loom_serve_device_handle(device_owner_);
    dispatch_ = loom_serve_device_dispatch_queue(device_owner_);
    transfer_ = loom_serve_device_transfer_queue(device_owner_);
    IREE_ASSERT_OK(
        loom_serve_device_create_execution(device_owner_, &execution_));
    pool_ = loom_serve_device_memory_pool(device_owner_);
    directory_ =
        std::filesystem::path(WeightSourcePath()).parent_path().string();
    const iree_string_view_t providers[] = {IREE_SV("stages.loom"),
                                            IREE_SV("packing.loom")};
    const loom_serve_sources_t sources = {
        .base_directory = iree_make_cstring_view(directory_.c_str()),
        .count = IREE_ARRAYSIZE(providers),
        .values = providers,
    };
    IREE_ASSERT_OK(loom_serve_jit_create(
        device_, dispatch_, loom_serve_device_host(device_owner_), &sources,
        nullptr, &compilation_, host_allocator_, &jit_));
    WriteCheckpoint(path_.path(), 0);
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        loom_serve_device_storage_pool(device_owner_),
        iree_hal_make_pool_reservation_request(sizeof(output_), 0),
        iree_infinite_timeout(), &output_buffer_));
  }

  void TearDown() override {
    if (!device_owner_) {
      return;
    }
    if (deferred_backing_) {
      deferred_backing_->CompleteReturns();
    }
    iree_hal_buffer_release(workspace_);
    if (consumer_gate_) {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(consumer_gate_, 2, nullptr));
    }
    if (shared_execution_) {
      IREE_EXPECT_OK(loom_serve_execution_drain(shared_execution_));
    }
    if (execution_) {
      IREE_EXPECT_OK(loom_serve_execution_drain(execution_));
    }
    for (auto* command : commands_) {
      iree_hal_command_buffer_release(command);
    }
    for (auto* stage : compiled_) {
      loom_serve_jit_stage_free(stage);
    }
    for (auto& stage : buffers_) {
      for (auto* buffer : stage) {
        iree_hal_buffer_release(buffer);
      }
    }
    for (auto* plan : plans_) {
      IREE_EXPECT_OK(loom_serve_weights_await_retirement(plan));
      if (arena_ && plan) {
        bool complete = false;
        IREE_EXPECT_OK(loom_serve_weights_deactivate(plan, &complete));
        IREE_EXPECT_OK(loom_serve_memory_pool_await(pool_));
        IREE_EXPECT_OK(loom_serve_weights_deactivate(plan, &complete));
        EXPECT_TRUE(complete);
      }
    }
    for (auto* residency : residencies_) {
      loom_serve_residency_free(residency);
    }
    for (auto* group : parameter_groups_) {
      loom_serve_residency_group_free(group);
    }
    for (auto* plan : plans_) {
      IREE_EXPECT_OK(loom_serve_weights_free(plan));
    }
    IREE_EXPECT_OK(loom_serve_virtual_buffer_free(state_));
    loom_serve_memory_arena_free(arena_);
    if (owned_pool_) {
      loom_serve_memory_pool_trim(owned_pool_, 0);
      IREE_EXPECT_STATUS_IS(retirement_code_,
                            loom_serve_memory_pool_await(owned_pool_));
      EXPECT_EQ(loom_serve_memory_pool_statistics(owned_pool_).reserved_bytes,
                0u);
      EXPECT_EQ(loom_serve_memory_pool_statistics(owned_pool_).committed_bytes,
                0u);
      loom_serve_residency_cache_free(owned_cache_);
      loom_serve_memory_pool_free(owned_pool_);
      pool_ = nullptr;
    }
    iree_hal_pool_release(deferred_backing_);
    iree_hal_buffer_release(output_buffer_);
    loom_serve_jit_release(jit_);
    loom_serve_execution_release(shared_execution_);
    loom_serve_execution_release(execution_);
    iree_hal_semaphore_release(consumer_gate_);
    IREE_EXPECT_OK(loom_serve_device_trim(device_owner_, 0));
    DrainNativeReturns();
    const auto memory = loom_serve_device_memory_statistics(device_owner_);
    EXPECT_EQ(memory.retained.reserved_bytes,
              loom_serve_workspace_memory_usage(
                  loom_serve_device_workspace(device_owner_))
                  .reserved_bytes);
    EXPECT_EQ(memory.retained.committed_bytes, 0u);
    EXPECT_EQ(memory.workspace.bytes_committed, 0u);
    if (path_.Exists()) {
      EXPECT_TRUE(path_.Remove());
    }
    if (adapter_path_.Exists()) {
      EXPECT_TRUE(adapter_path_.Remove());
    }
  }

  void UseDeferredBacking(uint64_t memory_limit = 2 * kSlabBytes) {
    if (!pool_) {
      return;
    }
    const auto* dispatch = iree_hal_queue_family(dispatch_);
    const auto* transfer = iree_hal_queue_family(transfer_);
    iree_hal_pool_family_access_t families[] = {
        {.family = dispatch,
         .usage =
             IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
         .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE},
        {.family = transfer,
         .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
         .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE},
    };
    const iree_hal_pool_scope_t scope = {
        .family_count = dispatch == transfer ? 1u : 2u, .families = families};
    auto* group = loom_serve_device_group(device_owner_);
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    iree_hal_pool_t* backing = nullptr;
    IREE_ASSERT_OK(iree_hal_slab_pool_create(group, scope, &options,
                                             host_allocator_, &backing));
    auto status = iree::hal::testing::DeferredBackingPool::Create(
        backing, &deferred_backing_);
    iree_hal_pool_release(backing);
    IREE_ASSERT_OK(status);
    for (auto& family : families) {
      family.interfaces = 0;
    }
    iree_hal_virtual_buffer_capabilities_t capabilities;
    IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
        group, scope, host_allocator_, &capabilities));
    loom_serve_memory_geometry_t geometry;
    IREE_ASSERT_OK(loom_serve_memory_pool_query_geometry(
        &capabilities, deferred_backing_, kSlabBytes, &geometry));
    const loom_serve_memory_geometry_t geometries[] = {geometry, geometry};
    IREE_ASSERT_OK(loom_serve_memory_pool_allocate(
        group, scope, deferred_backing_, transfer_, geometries, memory_limit,
        host_allocator_, &owned_pool_));
    pool_ = owned_pool_;
    IREE_ASSERT_OK(loom_serve_residency_cache_allocate(
        pool_, nullptr, host_allocator_, &owned_cache_));
  }

  loom_serve_residency_cache_t* ResidencyCache() {
    return owned_cache_ ? owned_cache_
                        : loom_serve_device_residency_cache(device_owner_);
  }

  iree_status_t Compile(iree_host_size_t index, const char* root) {
    const loomc_config_options_t config = {
        nullptr, 0, {}, LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED};
    return loom_serve_jit_compile(jit_, iree_make_cstring_view(root), &config,
                                  &compiled_[index]);
  }

  loom_serve_weight_root_t Root(iree_host_size_t stage_index,
                                uint32_t root_index,
                                uint64_t minimum_byte_length = kSlabBytes) {
    const loom_cmd_program_t* program =
        loom_serve_jit_stage_program(compiled_[stage_index]);
    loom_cmd_program_parameter_root_t root =
        loom_cmd_program_parameter_root_at(program, root_index);
    // Pressure cases request the full root extent they intend to consume.
    // Small roots otherwise commit only their native-aligned physical extent.
    root.required_byte_length =
        std::max(root.required_byte_length, minimum_byte_length);
    return {program, root, &buffers_[stage_index][root.fixed_buffer_index]};
  }

  iree_status_t CreateSourceRoots(
      iree_host_size_t shared_count, iree_host_size_t root_count,
      const loom_serve_weight_root_t* roots,
      std::initializer_list<std::initializer_list<std::string>> sources,
      const char* policy_name) {
    auto** plan = &plans_[plan_count_++];
    IREE_RETURN_IF_ERROR(loom_serve_weights_allocate(
        device_, transfer_, dispatch_,
        loom_serve_device_storage_pool(device_owner_), pool_,
        loom_serve_device_weight_geometry(device_owner_), arena_,
        loom_serve_device_workspace(device_owner_), shared_count, root_count,
        roots, host_allocator_, plan));
    return PrepareSources(*plan, sources, policy_name);
  }

  iree_status_t PrepareSources(
      loom_serve_weights_t* plan,
      std::initializer_list<std::initializer_list<std::string>> sources,
      const char* policy_name) {
    const std::string policy = directory_ + "/" + policy_name;
    const auto before = loom_serve_weights_statistics(plan);
    loom_serve_test_weight_preparation_t preparation;
    preparation.weights = plan;
    preparation.host = loom_serve_device_host(device_owner_);
    preparation.jit = jit_;
    std::vector<std::vector<loom_model_file_t>> shards;
    std::vector<loom_model_file_list_t> checkpoints;
    std::vector<uint32_t> ordinals;
    for (const auto& source : sources) {
      auto& paths = shards.emplace_back();
      for (const auto& path : source) {
        paths.push_back({iree_make_cstring_view(path.c_str()), {}});
      }
      checkpoints.push_back({paths.size(), paths.data()});
      ordinals.push_back(static_cast<uint32_t>(ordinals.size()));
    }
    preparation.parameter = {
        0, static_cast<uint32_t>(checkpoints.size()), checkpoints.data(),
        iree_make_cstring_view(policy.c_str()), ordinals.data()};
    auto status = loom_serve_parameter_sources_create(
        nullptr, &preparation.sources,
        static_cast<uint32_t>(checkpoints.size()), checkpoints.data(),
        host_allocator_, &preparation.indexed_sources);
    if (iree_status_is_ok(status)) {
      status = preparation.Run();
    }
    loom_serve_parameter_sources_release(preparation.indexed_sources);
    const auto after = loom_serve_weights_statistics(plan);
    EXPECT_EQ(after.reserved_bytes, before.reserved_bytes);
    EXPECT_EQ(after.committed_bytes, before.committed_bytes);
    EXPECT_EQ(loom_serve_weights_activity(plan).activations, 0u);
    return status;
  }

  iree_status_t Prepare(loom_serve_weights_t* plan,
                        std::initializer_list<std::string> paths,
                        const char* policy_name = "policy.loom") {
    return PrepareSources(plan, {paths}, policy_name);
  }

  iree_status_t CreateRoots(iree_host_size_t shared_count,
                            iree_host_size_t root_count,
                            const loom_serve_weight_root_t* roots,
                            std::initializer_list<std::string> paths,
                            const char* policy_name = "policy.loom") {
    return CreateSourceRoots(shared_count, root_count, roots, {paths},
                             policy_name);
  }

  iree_status_t LoadRoots(iree_host_size_t shared_count,
                          iree_host_size_t root_count,
                          const loom_serve_weight_root_t* roots,
                          std::initializer_list<std::string> paths) {
    IREE_RETURN_IF_ERROR(CreateRoots(shared_count, root_count, roots, paths));
    auto* plan = plans_[plan_count_ - 1];
    auto status = loom_serve_weights_activate(plan);
    if (iree_status_is_ok(status)) {
      status = loom_serve_weights_import_readiness(plan, execution_);
    }
    return iree_status_join(status, loom_serve_weights_await(plan));
  }

  iree_status_t Register(iree_host_size_t index, iree_host_size_t first_plan,
                         iree_host_size_t plan_count) {
    auto* cache = ResidencyCache();
    IREE_RETURN_IF_ERROR(loom_serve_residency_group_allocate(
        cache, plan_count, plans_.data() + first_plan, host_allocator_,
        &parameter_groups_[index]));
    auto status = loom_serve_residency_allocate(
        cache, execution_, 1, &parameter_groups_[index], host_allocator_,
        &residencies_[index]);
    if (!iree_status_is_ok(status)) {
      loom_serve_residency_group_free(parameter_groups_[index]);
      parameter_groups_[index] = nullptr;
    }
    return status;
  }

  void RunPinned(iree_host_size_t index, int32_t first) {
    IREE_ASSERT_OK(loom_serve_residency_acquire_wait(residencies_[index]));
    ASSERT_NO_FATAL_FAILURE(Run(index, first));
    loom_serve_residency_release(residencies_[index]);
  }

  void ExpectBacking(iree_host_size_t index, uint64_t bytes) {
    const auto memory = loom_serve_weights_statistics(plans_[index]);
    // Accepted native work owns budget before physical results are published.
    EXPECT_EQ(memory.committed_bytes + memory.pending_bytes, bytes);
  }

  void AdmitAfterRelease(iree_host_size_t index) {
    IREE_ASSERT_OK(loom_serve_residency_acquire_wait(residencies_[index]));
  }

  void DrainNativeReturns() {
    if (pool_) {
      IREE_ASSERT_STATUS_IS(retirement_code_,
                            loom_serve_memory_pool_await(pool_));
    }
    IREE_ASSERT_OK(
        loom_serve_test_device_trim_and_drain(device_owner_, UINT64_MAX));
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
    return LoadRoots(shared_count, root_count, roots.data(), {path_.path()});
  }

  void ReadOutputOn(loom_serve_execution_t* execution, iree_host_size_t index) {
    if (!commands_[index]) {
      IREE_ASSERT_OK(loom_serve_jit_stage_record(
          compiled_[index], iree_hal_queue_family(dispatch_),
          IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, nullptr, buffers_[index].data(),
          &commands_[index]));
    }
    const iree_hal_buffer_binding_t binding = {output_buffer_, 0,
                                               sizeof(output_)};
    uint64_t completion = 0;
    IREE_ASSERT_OK(loom_serve_execution_execute(execution, commands_[index],
                                                {1, &binding}, &completion));
    iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
    };
    download.download.source_buffer = output_buffer_;
    download.download.target = output_.data();
    download.download.length = sizeof(output_);
    IREE_ASSERT_OK(
        loom_serve_execution_feedback(execution, 1, &download, &completion));
    IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution, completion));
  }

  void RunOn(loom_serve_execution_t* execution, iree_host_size_t index,
             int32_t first) {
    ASSERT_NO_FATAL_FAILURE(ReadOutputOn(execution, index));
    for (int32_t i = 0; i < 8; ++i) {
      EXPECT_EQ(output_[i], first + 3 * i);
    }
  }

  void Run(iree_host_size_t index, int32_t first) {
    RunOn(execution_, index, first);
  }

  // Allocation policy for the real runtime and compiler.
  iree_allocator_t host_allocator_ = iree_allocator_system();
  // Copied compiler evidence retained beyond individual results.
  loom_serve_compilation_report_t compilation_ = {};
  // Borrowed process-owned group outliving all compiler, VM and I/O resources.
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
  // Independent consumer of the same immutable roots, retired before teardown.
  loom_serve_execution_t* shared_execution_ = nullptr;
  // Unrelated first-consumer work, excluded from shared weight readiness.
  iree_hal_semaphore_t* consumer_gate_ = nullptr;
  // Explicitly elastic backing or NULL for the fixed comparison arm.
  loom_serve_memory_pool_t* pool_ = nullptr;
  // Optional shared address owner, destroyed after all parameter ranges retire.
  loom_serve_memory_arena_t* arena_ = nullptr;
  // Terminal pool result expected while all accepted native work is drained.
  iree_status_code_t retirement_code_ = IREE_STATUS_OK;
  // Optional real pool whose parent reservations can outlive logical return.
  loom_serve_memory_pool_t* owned_pool_ = nullptr;
  // Admission policy attached to the same optional pool as its weight plans.
  loom_serve_residency_cache_t* owned_cache_ = nullptr;
  // Retains real native backing without blocking unmapping or queue work.
  iree::hal::testing::DeferredBackingPool* deferred_backing_ = nullptr;
  // Physical slab size; the shared budget holds two unique roots.
  static constexpr uint64_t kSlabBytes = 4096;
  // Owned checkpoint plans, including partially failed construction.
  std::array<loom_serve_weights_t*, 3> plans_ = {};
  // Unique physical groups, outliving all working sets that borrow them.
  std::array<loom_serve_residency_group_t*, 3> parameter_groups_ = {};
  // Invocation working sets borrowing groups until final queue retirement.
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

}  // namespace loom_serve::testing

#endif  // LOOM_SERVE_RUNTIME_PARAMETER_WEIGHTS_TEST_BASE_H_
