// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/executables.h"

#include <atomic>
#include <condition_variable>
#include <mutex>

#include "iree/base/tooling/flags.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"
#include "loom/serve/runtime/parameter/module.h"
#include "loom/serve/runtime/parameter/weights_test_base.h"
#include "loom/serve/runtime/program/program.h"

IREE_FLAG(string, executable_sources, "", "Request executable source fixture.");

namespace loom_serve::testing {

class ExecutablesTest : public WeightsTest {
 protected:
  virtual uint64_t SlabCount() const { return 2; }

  void SetUpDevice() override {
    // Native geometry is known after device construction. Reuse the
    // process-owned configurations for the required number of physical slabs.
    ASSERT_NO_FATAL_FAILURE(WeightsTest::SetUpDevice());
    if (GetParam()) {
      const uint64_t root_bytes = loom_serve_device_weight_geometry(
                                      loom_serve_device_group_at(group_, 0))
                                      .minimum_size;
      IREE_ASSERT_OK(AcquireWeightDeviceGroup(
          true, kSlabBytes, SlabCount() * root_bytes, &group_));
    }
  }

  void TearDown() override {
    if (!device_owner_) {
      return;
    }
    if (execution_) {
      IREE_EXPECT_OK(loom_serve_execution_drain(execution_));
    }
    if (executables_ && !retiring_) {
      loom_serve_executables_release(executables_);
    }
    loom_serve_executables_retire(executables_);
    IREE_EXPECT_OK(loom_serve_executables_await_retirement(executables_));
    IREE_EXPECT_OK(loom_serve_executables_free(executables_));
    loom_serve_program_free(control_);
    iree_vm_module_release(runner_);
    iree_vm_module_release(parameters_);
    loom_serve_stage_table_free(table_);
    loom_serve_preparation_free(preparation_);
    loom_serve_parameter_sources_release(parameter_sources_);
    iree_vm_environment_free(environment_);
    loom_serve_parameter_bank_retire(bank_, nullptr, nullptr);
    IREE_EXPECT_OK(loom_serve_parameter_bank_await_retirement(bank_));
    loom_serve_residency_free(residency_);
    IREE_EXPECT_OK(loom_serve_parameter_bank_free(bank_));
    iree_notification_deinitialize(&notification_);
    WeightsTest::TearDown();
  }

  iree_status_t Declare(const char* root, bool swapped, int32_t count,
                        int64_t tag, uint64_t limit,
                        loom_serve_preparation_t** out_preparation,
                        const char* entry = "declare") {
    const std::string policy = directory_ + "/policy.loom";
    const std::string strings[] = {
        root,
        selected_checkpoint_.empty()
            ? (swapped ? adapter_path_.path() : path_.path())
            : selected_checkpoint_,
        selected_right_.empty()
            ? (swapped ? path_.path() : adapter_path_.path())
            : selected_right_,
        policy};
    iree_vm_variant_t arguments[7] = {};
    iree_status_t status = iree_ok_status();
    for (size_t i = 0; i < IREE_ARRAYSIZE(strings) && iree_status_is_ok(status);
         ++i) {
      iree_vm_buffer_t* buffer = nullptr;
      status =
          iree_vm_buffer_create(strings[i].size(), 1, host_allocator_, &buffer);
      if (iree_status_is_ok(status)) {
        iree_byte_span_t bytes;
        status = iree_vm_buffer_map_write(buffer, 0, strings[i].size(), &bytes);
        if (iree_status_is_ok(status)) {
          memcpy(bytes.data, strings[i].data(), strings[i].size());
        }
        arguments[i] =
            iree_vm_buffer_variant_from_ptr_move(&vm_types_, &buffer);
      }
    }
    arguments[4] = iree_vm_variant_from_i32(count);
    arguments[5] = iree_vm_variant_from_i64(tag);
    arguments[6] = iree_vm_variant_from_i64(static_cast<int64_t>(limit));
    iree_vm_module_t* capture = nullptr;
    loom_serve_program_t* program = nullptr;
    if (iree_status_is_ok(status)) {
      status = loom_serve_preparation_module_create(
          environment_, nullptr, nullptr, host_allocator_, &capture);
    }
    if (iree_status_is_ok(status)) {
      loom_serve_preparation_module_begin(
          capture, preparation_ ? 1 : 0, preparation_,
          LOOM_SERVE_PREPARATION_FLAG_PARAMETERS |
              (preparation_ ? 0
                            : LOOM_SERVE_PREPARATION_FLAG_PARAMETER_ARENAS));
      loom_serve_compilation_report_t report = {};
      const auto root = iree_make_cstring_view(entry);
      status = loom_serve_program_allocate(
          loom_serve_host_target_environment(
              loom_serve_device_host(device_owner_)),
          environment_, &sources_,
          iree_make_cstring_view(FLAG_executable_sources), 1, &root,
          {&capture, 1}, &report, host_allocator_, &program);
      iree_vm_function_t function = {};
      if (iree_status_is_ok(status)) {
        status =
            iree_vm_process_lookup_function(loom_serve_program_process(program),
                                            IREE_SV("model"), root, &function);
      }
      if (iree_status_is_ok(status)) {
        status =
            iree_vm_invoke(loom_serve_program_invocation(program), function,
                           iree_vm_variant_span_from_array(arguments), {});
      }
      *out_preparation = loom_serve_preparation_module_take(capture);
      if (!iree_status_is_ok(status)) {
        loom_serve_preparation_free(*out_preparation);
        *out_preparation = nullptr;
      }
    }
    loom_serve_program_free(program);
    iree_vm_module_release(capture);
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    return status;
  }

  iree_status_t Create(iree_host_size_t capacity = 4,
                       iree_host_size_t max_bytes = 16 * 1024 * 1024,
                       iree_host_size_t parameter_capacity = 4,
                       iree_host_size_t parameter_bytes = 1024 * 1024,
                       bool single_domain = false, uint64_t arena_bytes = 0) {
    iree_notification_initialize(&notification_);
    WriteCheckpoint(adapter_path_.path(), 100);
    IREE_RETURN_IF_ERROR(
        iree_vm_environment_allocate(host_allocator_, &environment_));
    IREE_RETURN_IF_ERROR(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment_, IREE_SV("vm")),
        &vm_types_));
    const iree_vm_ref_type_table_t* types = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_module_register_types(environment_, &types));
    IREE_RETURN_IF_ERROR(iree_hal_module_types_resolve(types, &hal_types_));
    const char* root_name =
        arena_bytes ? "bank_cold" : (single_domain ? "raw" : "domains");
    IREE_RETURN_IF_ERROR(Declare(root_name, false, single_domain ? 1 : 2, 0,
                                 arena_bytes, &preparation_,
                                 arena_bytes ? "declare_arena" : "declare"));
    const loom_model_file_list_t* paths = nullptr;
    const uint32_t source_count =
        loom_serve_preparation_local_sources(preparation_, &paths);
    IREE_RETURN_IF_ERROR(loom_serve_parameter_sources_create(
        nullptr, &sources_, source_count, paths, host_allocator_,
        &parameter_sources_));
    IREE_RETURN_IF_ERROR(Compile(0, root_name));
    IREE_RETURN_IF_ERROR(loom_serve_parameter_bank_allocate(
        device_owner_, loom_serve_jit_stage_program(compiled_[0]),
        host_allocator_, &bank_));
    const auto* declaration = loom_serve_preparation_stage(preparation_, 0);
    iree_status_t status = iree_ok_status();
    for (size_t i = 0;
         i < declaration->parameter_count && iree_status_is_ok(status); ++i) {
      status = loom_serve_weights_prepare(
          loom_serve_parameter_bank_weights(bank_, i),
          loom_serve_device_host(device_owner_), jit_,
          IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, &sources_,
          &declaration->parameters[i], parameter_sources_);
    }
    IREE_RETURN_IF_ERROR(status);
    auto* group = loom_serve_parameter_bank_group(bank_);
    IREE_RETURN_IF_ERROR(loom_serve_residency_allocate(
        loom_serve_device_residency_cache(device_owner_), execution_, 1, &group,
        host_allocator_, &residency_));
    loom_serve_executable_options_t options = {
        .device = device_owner_,
        .jit = jit_,
        .sources = &sources_,
        .execution = execution_,
        .residency = residency_,
        .preparation = preparation_,
        .parameter_sources = parameter_sources_,
        .parameters = bank_,
        .recording =
            {
                .queue_family = iree_hal_queue_family(dispatch_),
                .parameters = loom_serve_parameter_bank_buffers(bank_),
                .binding_count = static_cast<uint16_t>(arena_bytes ? 2 : 1),
            },
        .cache = {capacity, max_bytes},
        .parameter_cache = {parameter_capacity, parameter_bytes},
    };
    IREE_RETURN_IF_ERROR(loom_serve_stage_table_allocate(
        &options.recording, nullptr, 1, compiled_.data(), host_allocator_,
        &table_));
    const loom_serve_module_options_t module_options = {
        .binding_capacity = static_cast<uint16_t>(arena_bytes ? 2 : 1),
        .stages = {table_->part_count, table_->parts},
    };
    IREE_RETURN_IF_ERROR(loom_serve_module_create(
        &hal_types_, execution_, module_options, host_allocator_, &runner_));
    options.stages = table_;
    options.runner = runner_;
    IREE_RETURN_IF_ERROR(loom_serve_parameters_module_create(
        environment_, &hal_types_,
        loom_serve_device_transfer_queue(device_owner_), parameter_sources_,
        host_allocator_, &parameters_));
    options.parameter_module = parameters_;
    IREE_RETURN_IF_ERROR(loom_serve_executables_allocate(
        &options, host_allocator_, &executables_));
    loom_serve_executables_set_observer(
        executables_, {[](void* self) {
                         iree_notification_post(
                             static_cast<iree_notification_t*>(self),
                             IREE_ALL_WAITERS);
                       },
                       &notification_});
    const iree_string_view_t roots[] = {IREE_SV("submit"),
                                        IREE_SV("submit_indexed")};
    loom_serve_compilation_report_t report = {};
    IREE_RETURN_IF_ERROR(loom_serve_program_allocate(
        loom_serve_host_target_environment(
            loom_serve_device_host(device_owner_)),
        environment_, &sources_,
        iree_make_cstring_view(FLAG_executable_sources), arena_bytes ? 2 : 1,
        roots, {&runner_, 1}, &report, host_allocator_, &control_));
    IREE_RETURN_IF_ERROR(
        iree_vm_process_lookup_function(loom_serve_program_process(control_),
                                        IREE_SV("model"), roots[0], &submit_));
    return arena_bytes ? iree_vm_process_lookup_function(
                             loom_serve_program_process(control_),
                             IREE_SV("model"), roots[1], &submit_indexed_)
                       : iree_ok_status();
  }

  iree_status_t Select(const char* root, bool swapped, int32_t count = 2,
                       int64_t tag = 0, uint64_t limit = 0) {
    IREE_RETURN_IF_ERROR(Begin(root, swapped, count, tag, limit));
    return Advance(false);
  }

  iree_status_t Begin(const char* root, bool swapped, int32_t count = 2,
                      int64_t tag = 0, uint64_t limit = 0,
                      const char* entry = "declare") {
    loom_serve_preparation_t* declaration = nullptr;
    IREE_RETURN_IF_ERROR(
        Declare(root, swapped, count, tag, limit, &declaration, entry));
    loom_serve_executables_capture(executables_);
    loom_serve_executables_declare(executables_, declaration);
    return iree_ok_status();
  }

  iree_status_t Advance(bool cancel) {
    bool complete = false;
    iree_status_t status = iree_ok_status();
    while (!complete && iree_status_is_ok(status)) {
      const uint32_t ticket = iree_notification_prepare_wait(&notification_);
      status = loom_serve_executables_advance(executables_, cancel, &complete);
      if (!complete && iree_status_is_ok(status)) {
        iree_notification_commit_wait(&notification_, ticket,
                                      IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      } else {
        iree_notification_cancel_wait(&notification_);
      }
    }
    return status;
  }

  void Execute(int32_t first, int32_t stride = 3, int32_t stage = 1) {
    auto* residency = loom_serve_executables_residency(executables_);
    IREE_ASSERT_OK(loom_serve_residency_acquire_wait(residency));
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_i32(stage),
        iree_hal_buffer_variant_from_ptr_borrowed(&hal_types_, output_buffer_)};
    iree_vm_variant_t results[1] = {};
    auto status =
        iree_vm_invoke(loom_serve_program_invocation(control_), submit_,
                       iree_vm_variant_span_from_array(arguments),
                       iree_vm_variant_span_from_array(results));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
    IREE_ASSERT_OK(status);
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = output_buffer_,
                .target = output_.data(),
                .length = sizeof(output_),
            },
    };
    uint64_t completion = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_feedback(execution_, 1, &download, &completion));
    IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
    loom_serve_residency_release(residency);
    for (size_t i = 0; i < output_.size(); ++i) {
      EXPECT_EQ(output_[i], first + stride * i);
    }
    loom_serve_executables_release(executables_);
  }

  void CollectIdle() {
    IREE_ASSERT_OK(loom_serve_executables_query_status(executables_));
    while (loom_serve_executables_is_pending(executables_)) {
      const uint32_t ticket = iree_notification_prepare_wait(&notification_);
      IREE_ASSERT_OK(loom_serve_executables_query_status(executables_));
      if (loom_serve_executables_is_pending(executables_)) {
        iree_notification_commit_wait(&notification_, ticket,
                                      IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      } else {
        iree_notification_cancel_wait(&notification_);
      }
    }
  }

  iree_status_t ReplaceSources() {
    const iree_string_view_t providers[] = {IREE_SV("stages.loom"),
                                            IREE_SV("packing.loom")};
    const loom_serve_sources_t sources = {
        .base_directory =
            iree_make_string_view(directory_.data(), directory_.size()),
        .count = IREE_ARRAYSIZE(providers),
        .values = providers,
    };
    struct Completion {
      // Joins the producer's final access before the owner takes its value.
      std::mutex mutex;
      // Notifies only the outer application thread, never the shared executor.
      std::condition_variable condition;
      // Owned producer outcome, transferred when ready is published.
      iree_status_t status = iree_ok_status();
      // Owned replacement generation until select/dispose below.
      void* value = nullptr;
      // Terminal compile/recording publication under mutex.
      bool ready = false;
    } completion;
    auto update = loom_serve_executables_update(executables_);
    IREE_RETURN_IF_ERROR(update.compile(
        update.self, &sources,
        [](void* self, iree_status_t status, void* value,
           const loom_serve_compilation_report_t* report) {
          auto* completion = static_cast<Completion*>(self);
          std::lock_guard<std::mutex> lock(completion->mutex);
          completion->status = status;
          completion->value = value;
          completion->ready = true;
          completion->condition.notify_one();
        },
        &completion));
    {
      std::unique_lock<std::mutex> lock(completion.mutex);
      completion.condition.wait(lock, [&] { return completion.ready; });
    }
    auto status = completion.status;
    if (iree_status_is_ok(status)) {
      status = update.select(update.self, completion.value);
    }
    update.dispose(completion.value);
    return status;
  }

  // Authored locators are rooted in the test's declared runfiles namespace.
  const loom_serve_sources_t sources_ = {.base_directory = IREE_SV(".")};
  // VM identities outliving every source capture and native module.
  iree_vm_environment_t* environment_ = nullptr;
  // Canonical builtin VM reference identities.
  iree_vm_ref_types_t vm_types_ = {};
  // Canonical native buffer identities.
  iree_hal_module_types_t hal_types_ = {};
  // Immutable registration declarations.
  loom_serve_preparation_t* preparation_ = nullptr;
  // Original checkpoint indexes shared by parameter banks and request choices.
  loom_serve_parameter_sources_t* parameter_sources_ = nullptr;
  // Registration's independently loaded roots with colliding tensor names.
  loom_serve_parameter_bank_t* bank_ = nullptr;
  // Atomic admission for all registration-owned parameters.
  loom_serve_residency_t* residency_ = nullptr;
  // Original recordings borrowed by the executable owner.
  loom_serve_stage_table_t* table_ = nullptr;
  // Native runner consuming the selected request table.
  iree_vm_module_t* runner_ = nullptr;
  // Demand-read capability selected atomically with the request's command
  // table.
  iree_vm_module_t* parameters_ = nullptr;
  // Source-selected request ownership under test.
  loom_serve_executables_t* executables_ = nullptr;
  // Real VM caller of the runner ABI.
  loom_serve_program_t* control_ = nullptr;
  // Bound request entry, borrowing control_.
  iree_vm_function_t submit_ = {};
  // Bound two-buffer consumer for indexed bank cohorts.
  iree_vm_function_t submit_indexed_ = {};
  // Outer-thread completion notification; no worker waits here.
  iree_notification_t notification_ = {};
  // Optional request checkpoint, distinct from both registered domains.
  std::string selected_checkpoint_;
  // Optional second request domain, including an alias of the first domain.
  std::string selected_right_;
  // Terminal retirement has already released request and recording consumers.
  bool retiring_ = false;
};

TEST_P(ExecutablesTest, RebindsCheckpointDataWithoutRecompilingSharedCode) {
  IREE_ASSERT_OK(Create(1));
  const auto* base = loom_serve_parameter_bank_buffers(bank_)[0];
  const auto before = compilation_.commands;
  IREE_ASSERT_OK(Select("domains", false));
  ASSERT_NO_FATAL_FAILURE(Execute(290));
  EXPECT_EQ(compilation_.commands, before + 1);
  const auto kernels = compilation_.kernels;
  // Same compiler inputs, different checkpoint bindings. Evicting the first
  // choice must not retire the code or retain its declaration as a cache key.
  IREE_ASSERT_OK(Select("domains", true));
  ASSERT_NO_FATAL_FAILURE(Execute(190));
  EXPECT_EQ(compilation_.commands, before + 1);
  EXPECT_EQ(compilation_.kernels, kernels);
  IREE_ASSERT_OK(Select("domains", false));
  ASSERT_NO_FATAL_FAILURE(Execute(290));
  EXPECT_EQ(compilation_.commands, before + 1);
  EXPECT_EQ(loom_serve_parameter_bank_buffers(bank_)[0], base);
  EXPECT_EQ(loom_serve_residency_activity(residency_).activations, 2u);
}

TEST_P(ExecutablesTest, SourceOnlyChoicesPreserveColdCodeAndResidentBindings) {
  IREE_ASSERT_OK(Create(1));
  const auto commands = compilation_.commands;
  const auto kernels = compilation_.kernels;
  const auto* base = loom_serve_parameter_bank_buffers(bank_)[0];
  for (bool swapped : {false, true, false}) {
    IREE_ASSERT_OK(Begin("domains", swapped, 0, 0, 0, "sources_only"));
    IREE_ASSERT_OK(Advance(false));
    ASSERT_NO_FATAL_FAILURE(Execute(290, 3, 0));
    EXPECT_EQ(compilation_.commands, commands);
    EXPECT_EQ(compilation_.kernels, kernels);
    EXPECT_EQ(loom_serve_parameter_bank_buffers(bank_)[0], base);
  }
}

TEST_P(ExecutablesTest, SelectsSubsetWithoutChangingParameterStorage) {
  IREE_ASSERT_OK(Create());
  IREE_ASSERT_OK(Select("raw", false, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(90));
  const auto commands = compilation_.commands;
  IREE_ASSERT_OK(Select("raw", true, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(390));
  EXPECT_EQ(compilation_.commands, commands);
  // An exact repeated selection uses the already-recorded table too.
  IREE_ASSERT_OK(Select("raw", true, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(390));
  EXPECT_EQ(compilation_.commands, commands);
}

TEST_P(ExecutablesTest, DisabledRetentionReleasesSharedCodeAndRecordsAgain) {
  IREE_ASSERT_OK(Create(0));
  const auto commands = compilation_.commands;
  IREE_ASSERT_OK(Select("domains", false));
  ASSERT_NO_FATAL_FAILURE(Execute(290));
  IREE_ASSERT_OK(Select("domains", true));
  ASSERT_NO_FATAL_FAILURE(Execute(190));
  EXPECT_EQ(compilation_.commands, commands + 2);
}

TEST_P(ExecutablesTest, ByteBudgetEvictsEvenWhenEntryCapacityRemains) {
  IREE_ASSERT_OK(Create(4, 1));
  const auto commands = compilation_.commands;
  IREE_ASSERT_OK(Select("domains", false));
  ASSERT_NO_FATAL_FAILURE(Execute(290));
  IREE_ASSERT_OK(Select("domains", true));
  ASSERT_NO_FATAL_FAILURE(Execute(190));
  EXPECT_EQ(compilation_.commands, commands + 2);
}

TEST_P(ExecutablesTest, CancellationJoinsRecordingWithoutActivatingParameters) {
  IREE_ASSERT_OK(Create());
  IREE_ASSERT_OK(Select("domains", false));
  ASSERT_NO_FATAL_FAILURE(Execute(290));
  const auto commands = compilation_.commands;
  const auto activity = loom_serve_residency_activity(residency_);
  IREE_ASSERT_OK(Begin("domains", true));
  bool complete = false;
  IREE_ASSERT_OK(
      loom_serve_executables_advance(executables_, false, &complete));
  IREE_ASSERT_OK(Advance(true));
  loom_serve_executables_release(executables_);
  EXPECT_EQ(compilation_.commands, commands);
  EXPECT_EQ(loom_serve_residency_activity(residency_).activations,
            activity.activations);
  IREE_ASSERT_OK(Select("domains", true));
  ASSERT_NO_FATAL_FAILURE(Execute(190));
  EXPECT_EQ(compilation_.commands, commands);
}

TEST_P(ExecutablesTest, MixedImplicitAndExplicitDomainsRejectBeforeBinding) {
  IREE_ASSERT_OK(Create());
  IREE_ASSERT_OK(Begin("domains", false, 2, 0, 0, "mixed"));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Advance(false));
  loom_serve_executables_release(executables_);
  IREE_ASSERT_OK(Select("domains", false));
  ASSERT_NO_FATAL_FAILURE(Execute(290));
}

TEST_P(ExecutablesTest, LayoutMismatchDoesNotPoisonSharedCode) {
  IREE_ASSERT_OK(Create());
  IREE_ASSERT_OK(Select("domains", false));
  ASSERT_NO_FATAL_FAILURE(Execute(290));
  const auto commands = compilation_.commands;
  // The same code must still validate each request's complete declaration.
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Select("domains", true, 1));
  loom_serve_executables_release(executables_);
  EXPECT_EQ(compilation_.commands, commands);
  IREE_ASSERT_OK(Select("domains", true));
  ASSERT_NO_FATAL_FAILURE(Execute(190));
  EXPECT_EQ(compilation_.commands, commands);
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Select("target", false, 1));
  loom_serve_executables_release(executables_);
  IREE_ASSERT_OK(Select("domains", false));
  ASSERT_NO_FATAL_FAILURE(Execute(290));
}

class WorkspaceExecutablesTest : public ExecutablesTest {
 protected:
  // Two parameter roots and a shared command workspace slab coexist.
  uint64_t SlabCount() const override { return 3; }
};

TEST_P(WorkspaceExecutablesTest, SharedCodeChecksEachRequestsWorkspaceLimit) {
  IREE_ASSERT_OK(Create());
  IREE_ASSERT_OK(Select("domains_workspace", false));
  ASSERT_NO_FATAL_FAILURE(Execute(870, 9));
  const auto commands = compilation_.commands;
  IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        Select("domains_workspace", false, 2, 1, 1));
  loom_serve_executables_release(executables_);
  EXPECT_EQ(compilation_.commands, commands);
  IREE_ASSERT_OK(Select("domains_workspace", true, 2, 2, 4096));
  ASSERT_NO_FATAL_FAILURE(Execute(570, 9));
  EXPECT_EQ(compilation_.commands, commands);
}

TEST_P(ExecutablesTest, PreparesNewCheckpointWithoutReplacingColdStorage) {
  IREE_ASSERT_OK(Create(4, 16 * 1024 * 1024, 4, 1024 * 1024, true));
  iree::testing::TempFilePath checkpoint("loom_request", ".safetensors");
  WriteCheckpoint(checkpoint.path(), 200);
  selected_checkpoint_ = checkpoint.path();
  const auto* base = loom_serve_parameter_bank_buffers(bank_)[0];
  IREE_ASSERT_OK(Select("raw", false, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(690));
  const auto commands = compilation_.commands;
  IREE_ASSERT_OK(Select("raw", false, 1));
  EXPECT_EQ(loom_serve_residency_activity(
                loom_serve_executables_residency(executables_))
                .activations,
            2u);
  ASSERT_NO_FATAL_FAILURE(Execute(690));
  EXPECT_EQ(compilation_.commands, commands);
  EXPECT_EQ(loom_serve_parameter_bank_buffers(bank_)[0], base);
}

TEST_P(ExecutablesTest, CodeEvictionKeepsPreparedCheckpoint) {
  IREE_ASSERT_OK(Create(0, 16 * 1024 * 1024, 4, 1024 * 1024, true));
  iree::testing::TempFilePath checkpoint("loom_request", ".safetensors");
  WriteCheckpoint(checkpoint.path(), 200);
  selected_checkpoint_ = checkpoint.path();
  IREE_ASSERT_OK(Select("raw", false, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(690));
  const auto commands = compilation_.commands;
  IREE_ASSERT_OK(Select("raw", false, 1));
  EXPECT_EQ(loom_serve_residency_activity(
                loom_serve_executables_residency(executables_))
                .activations,
            2u);
  ASSERT_NO_FATAL_FAILURE(Execute(690));
  EXPECT_EQ(compilation_.commands, commands + 1);
}

TEST_P(ExecutablesTest, ParameterByteBudgetRetiresRecordingsAndBank) {
  IREE_ASSERT_OK(Create(4, 16 * 1024 * 1024, 4, 1, true));
  iree::testing::TempFilePath checkpoint("loom_request", ".safetensors");
  WriteCheckpoint(checkpoint.path(), 200);
  selected_checkpoint_ = checkpoint.path();
  IREE_ASSERT_OK(Select("raw", false, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(690));
  ASSERT_NO_FATAL_FAILURE(CollectIdle());
  const auto first = loom_serve_executables_parameters(executables_);
  EXPECT_EQ(first.activity.activations, 2u);
  const uint64_t root_bytes =
      GetParam() ? loom_serve_device_weight_geometry(device_owner_).minimum_size
                 : 0;
  EXPECT_EQ(first.memory.reserved_bytes, root_bytes);
  EXPECT_EQ(first.memory.committed_bytes, root_bytes);
  EXPECT_EQ(first.memory.released_bytes, root_bytes);
  IREE_ASSERT_OK(Select("raw", false, 1));
  EXPECT_EQ(loom_serve_residency_activity(
                loom_serve_executables_residency(executables_))
                .activations,
            1u);
  ASSERT_NO_FATAL_FAILURE(Execute(690));
  ASSERT_NO_FATAL_FAILURE(CollectIdle());
  const auto second = loom_serve_executables_parameters(executables_);
  EXPECT_EQ(second.activity.activations, 3u);
  EXPECT_EQ(second.memory.reserved_bytes, first.memory.reserved_bytes);
  EXPECT_EQ(second.memory.committed_bytes, first.memory.committed_bytes);
  EXPECT_EQ(second.memory.released_bytes, 2 * first.memory.released_bytes);
}

TEST_P(ExecutablesTest, RetainedCheckpointsDoNotAddIndependentPeaks) {
  if (!GetParam()) {
    GTEST_SKIP() << "Fixed storage has no physical-pool accounting";
  }
  IREE_ASSERT_OK(Create(4, 16 * 1024 * 1024, 4, 1024 * 1024, true));
  iree::testing::TempFilePath first("loom_first", ".safetensors");
  iree::testing::TempFilePath second("loom_second", ".safetensors");
  WriteCheckpoint(first.path(), 200);
  WriteCheckpoint(second.path(), 300);
  selected_checkpoint_ = first.path();
  IREE_ASSERT_OK(Select("raw", false, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(690));
  selected_checkpoint_ = second.path();
  IREE_ASSERT_OK(Select("raw", false, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(990));
  const auto memory = loom_serve_executables_parameters(executables_).memory;
  const uint64_t root_bytes =
      loom_serve_device_weight_geometry(device_owner_).minimum_size;
  EXPECT_EQ(memory.reserved_bytes, 3 * root_bytes);
  EXPECT_EQ(memory.committed_bytes, 2 * root_bytes);
  EXPECT_EQ(memory.released_bytes, root_bytes);
  // The pool observes the actual overlap. A retained checkpoint's historical
  // peak cannot be added to the independently activated replacement's peak.
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).peak_bytes,
            2 * root_bytes);
}

TEST_P(ExecutablesTest, MissingNewCheckpointDoesNotPoisonNextSelection) {
  IREE_ASSERT_OK(Create());
  selected_checkpoint_ = "absent-checkpoint.safetensors";
  IREE_ASSERT_STATUS_IS(IREE_STATUS_NOT_FOUND, Select("raw", false, 1));
  loom_serve_executables_release(executables_);
  ASSERT_NO_FATAL_FAILURE(CollectIdle());
  selected_checkpoint_.clear();
  IREE_ASSERT_OK(Select("raw", false, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(90));
}

TEST_P(ExecutablesTest, SourceOnlyIndexFailurePreservesCodeAndColdWeights) {
  IREE_ASSERT_OK(Create());
  const auto commands = compilation_.commands;
  const auto kernels = compilation_.kernels;
  selected_checkpoint_ = "absent-checkpoint.safetensors";
  IREE_ASSERT_OK(Begin("domains", false, 0, 0, 0, "sources_only"));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_NOT_FOUND, Advance(false));
  EXPECT_EQ(compilation_.commands, commands);
  EXPECT_EQ(compilation_.kernels, kernels);
  EXPECT_EQ(
      loom_serve_executables_parameters(executables_).activity.activations, 0u);
  loom_serve_executables_release(executables_);
  selected_checkpoint_.clear();
  IREE_ASSERT_OK(Begin("domains", false, 0, 0, 0, "sources_only"));
  IREE_ASSERT_OK(Advance(false));
  ASSERT_NO_FATAL_FAILURE(Execute(290, 3, 0));
  EXPECT_EQ(compilation_.commands, commands);
  EXPECT_EQ(compilation_.kernels, kernels);
}

TEST_P(ExecutablesTest, AdmissionRejectsCompleteSetBeforeAnyActivation) {
  if (!GetParam()) {
    GTEST_SKIP() << "Fixed storage has no physical-pool budget";
  }
  IREE_ASSERT_OK(Create());
  iree::testing::TempFilePath checkpoint("loom_request", ".safetensors");
  WriteCheckpoint(checkpoint.path(), 200);
  selected_checkpoint_ = checkpoint.path();
  IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        Select("raw", false, 1));
  EXPECT_EQ(
      loom_serve_executables_parameters(executables_).activity.activations, 0u);
  loom_serve_executables_release(executables_);
  selected_checkpoint_.clear();
  IREE_ASSERT_OK(Select("domains", false));
  ASSERT_NO_FATAL_FAILURE(Execute(290));
}

TEST_P(ExecutablesTest, PackedLayoutIsPartOfCheckpointIdentity) {
  IREE_ASSERT_OK(Create(4, 16 * 1024 * 1024, 4, 1024 * 1024, true));
  iree::testing::TempFilePath checkpoint("loom_request", ".safetensors");
  WriteCheckpoint(checkpoint.path(), 200);
  selected_checkpoint_ = checkpoint.path();
  IREE_ASSERT_OK(Select("raw", false, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(690));
  IREE_ASSERT_OK(Select("target", false, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(670));
  const auto commands = compilation_.commands;
  IREE_ASSERT_OK(Select("raw", false, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(690));
  EXPECT_EQ(compilation_.commands, commands);
}

TEST_P(ExecutablesTest, PinnedBanksSurviveCodeEvictionAndDrainReloads) {
  if (!GetParam()) {
    GTEST_SKIP() << "Fixed storage cannot deactivate";
  }
  IREE_ASSERT_OK(Create(0, 16 * 1024 * 1024, 4, 1024 * 1024, true));
  iree::testing::TempFilePath checkpoint("loom_request", ".safetensors");
  WriteCheckpoint(checkpoint.path(), 200);
  selected_checkpoint_ = checkpoint.path();
  loom_serve_executables_set_pinned(executables_, true);
  IREE_ASSERT_OK(Select("raw", false, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(690));
  const auto before = loom_serve_executables_parameters(executables_);
  EXPECT_EQ(before.activity.activations, 2u);
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(
      loom_serve_device_residency_cache(device_owner_), 0));
  EXPECT_EQ(
      loom_serve_executables_parameters(executables_).memory.committed_bytes,
      before.memory.committed_bytes);
  bool complete = false;
  IREE_ASSERT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_serve_executables_deactivate(executables_, &complete));
  loom_serve_executables_set_pinned(executables_, false);
  IREE_ASSERT_OK(loom_serve_executables_deactivate(executables_, &complete));
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool_));
  IREE_ASSERT_OK(loom_serve_executables_deactivate(executables_, &complete));
  EXPECT_TRUE(complete);
  const auto drained = loom_serve_executables_parameters(executables_);
  EXPECT_EQ(drained.memory.committed_bytes, 0u);
  EXPECT_EQ(drained.activity.deactivations, 2u);
  IREE_ASSERT_OK(Select("raw", false, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(690));
  EXPECT_EQ(
      loom_serve_executables_parameters(executables_).activity.activations, 4u);
}

TEST_P(WorkspaceExecutablesTest, RepeatedDomainSharesOnePhysicalBank) {
  IREE_ASSERT_OK(Create(4, 16 * 1024 * 1024, 4, 1024 * 1024, true));
  iree::testing::TempFilePath checkpoint("loom_request", ".safetensors");
  WriteCheckpoint(checkpoint.path(), 200);
  selected_checkpoint_ = selected_right_ = checkpoint.path();
  IREE_ASSERT_OK(Select("domains", false));
  ASSERT_NO_FATAL_FAILURE(Execute(690));
  EXPECT_EQ(
      loom_serve_executables_parameters(executables_).activity.activations, 2u);
  IREE_ASSERT_OK(Select("domains_workspace", false));
  ASSERT_NO_FATAL_FAILURE(Execute(2070, 9));
  EXPECT_EQ(
      loom_serve_executables_parameters(executables_).activity.activations, 2u);
}

TEST_P(ExecutablesTest, CancellationJoinsNewBankAllocationBeforeReuse) {
  if (GetParam()) {
    GTEST_SKIP() << "Elastic roots allocate backing at activation";
  }
  IREE_ASSERT_OK(Create(4, 16 * 1024 * 1024, 4, 1024 * 1024, true));
  IREE_ASSERT_OK(Select("raw", false, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(90));
  iree::testing::TempFilePath checkpoint("loom_request", ".safetensors");
  WriteCheckpoint(checkpoint.path(), 200);
  selected_checkpoint_ = checkpoint.path();
  IREE_ASSERT_OK(Begin("raw", false, 1));
  std::atomic<bool> cancel = false;
  iree_status_t status = iree_ok_status();
  ScopedNativeHold hold(device_owner_);
  std::thread owner([&] {
    bool complete = false;
    while (!complete && iree_status_is_ok(status)) {
      const uint32_t ticket = iree_notification_prepare_wait(&notification_);
      status = loom_serve_executables_advance(executables_, cancel.load(),
                                              &complete);
      if (!complete && iree_status_is_ok(status)) {
        iree_notification_commit_wait(&notification_, ticket,
                                      IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      } else {
        iree_notification_cancel_wait(&notification_);
      }
    }
  });
  hold.WaitEntered();
  cancel.store(true);
  hold.Release();
  owner.join();
  IREE_ASSERT_OK(status);
  loom_serve_executables_release(executables_);
  EXPECT_EQ(
      loom_serve_executables_parameters(executables_).activity.activations, 1u);
  IREE_ASSERT_OK(Select("raw", false, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(690));
}

TEST_P(ExecutablesTest, ExactChoiceHitsRefreshCheckpointLru) {
  IREE_ASSERT_OK(Create(4, 16 * 1024 * 1024, 2, 1024 * 1024, true));
  iree::testing::TempFilePath first("loom_first", ".safetensors");
  iree::testing::TempFilePath second("loom_second", ".safetensors");
  iree::testing::TempFilePath third("loom_third", ".safetensors");
  WriteCheckpoint(first.path(), 200);
  WriteCheckpoint(second.path(), 300);
  WriteCheckpoint(third.path(), 400);
  selected_checkpoint_ = first.path();
  IREE_ASSERT_OK(Select("target", false, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(670));
  selected_checkpoint_ = second.path();
  IREE_ASSERT_OK(Select("target", false, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(970));
  selected_checkpoint_ = first.path();
  IREE_ASSERT_OK(Select("target", false, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(670));
  selected_checkpoint_ = third.path();
  IREE_ASSERT_OK(Select("target", false, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(1270));
  ASSERT_NO_FATAL_FAILURE(CollectIdle());
  const auto commands = compilation_.commands;
  const auto activations =
      loom_serve_executables_parameters(executables_).activity.activations;
  selected_checkpoint_ = first.path();
  IREE_ASSERT_OK(Select("target", false, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(670));
  // An exact choice hit refreshed the first checkpoint, not just its code.
  EXPECT_EQ(compilation_.commands, commands);
  selected_checkpoint_ = second.path();
  IREE_ASSERT_OK(Select("target", false, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(970));
  EXPECT_EQ(compilation_.commands, commands + 2);
  ASSERT_NO_FATAL_FAILURE(CollectIdle());
  EXPECT_GT(
      loom_serve_executables_parameters(executables_).activity.activations,
      activations);
}

TEST_P(ExecutablesTest, ReplacementRecordingsRetireBeforeCheckpointJoin) {
  IREE_ASSERT_OK(Create(4, 16 * 1024 * 1024, 4, 1024 * 1024, true));
  IREE_ASSERT_OK(ReplaceSources());
  iree::testing::TempFilePath checkpoint("loom_request", ".safetensors");
  WriteCheckpoint(checkpoint.path(), 200);
  selected_checkpoint_ = checkpoint.path();
  IREE_ASSERT_OK(Select("raw", false, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(690));
  auto update = loom_serve_executables_update(executables_);
  IREE_ASSERT_OK(update.select(update.self, nullptr));
  IREE_ASSERT_OK(loom_serve_execution_drain(execution_));
  retiring_ = true;
  loom_serve_executables_retire(executables_);
  loom_serve_program_free(control_);
  control_ = nullptr;
  iree_vm_module_release(runner_);
  runner_ = nullptr;
  loom_serve_stage_table_free(table_);
  table_ = nullptr;
  loom_serve_parameter_bank_retire(bank_, nullptr, nullptr);
  // No producer remains. A retained replacement table would keep a root view
  // alive forever and make the model's bank-retirement join self-dependent.
  EXPECT_TRUE(loom_serve_parameter_bank_is_retired(bank_));
}

INSTANTIATE_TEST_SUITE_P(Native, ExecutablesTest, ::testing::Bool());
INSTANTIATE_TEST_SUITE_P(Native, WorkspaceExecutablesTest, ::testing::Bool());

class ArenaExecutablesTest : public ExecutablesTest {
 protected:
  uint64_t SlabCount() const override { return 8; }

  iree_status_t CreateArena(uint64_t banks) {
    return Create(
        4, 16 * 1024 * 1024, 16, 1024 * 1024, true,
        banks * loom_serve_device_weight_geometry(device_owner_).minimum_size);
  }

  iree_status_t BeginBanks(const std::vector<std::string>& paths,
                           const char* root = "bank_cohort") {
    selected_checkpoint_.clear();
    for (const auto& path : paths) {
      const size_t position = selected_checkpoint_.size();
      selected_checkpoint_.resize(position + 8 + path.size());
      iree_unaligned_store_le_u64(selected_checkpoint_.data() + position,
                                  path.size());
      memcpy(selected_checkpoint_.data() + position + 8, path.data(),
             path.size());
    }
    auto status =
        Begin(root, false, static_cast<int32_t>(paths.size()), 0, 0, "banks");
    selected_checkpoint_.clear();
    return status;
  }

  std::vector<uint64_t> Offsets() {
    const auto bytes = loom_serve_executables_parameter_offsets(executables_);
    std::vector<uint64_t> offsets(bytes.data_length / sizeof(uint64_t));
    for (size_t i = 0; i < offsets.size(); ++i) {
      offsets[i] =
          iree_unaligned_load_le_u64(bytes.data + i * sizeof(uint64_t));
    }
    return offsets;
  }

  void ExecuteBanks(std::array<uint32_t, 2> counts,
                    const std::vector<int32_t>& strengths,
                    const std::vector<int32_t>& checkpoint_values) {
    const auto offsets = Offsets();
    ASSERT_EQ(offsets.size(), counts[0] + counts[1]);
    ASSERT_EQ(offsets.size(), strengths.size());
    ASSERT_EQ(offsets.size(), checkpoint_values.size());
    std::vector<uint8_t> payload(16 + offsets.size() * 16);
    iree_unaligned_store_le_u32(payload.data(), 0);
    iree_unaligned_store_le_u32(payload.data() + 4, counts[0]);
    iree_unaligned_store_le_u32(payload.data() + 8, counts[0]);
    iree_unaligned_store_le_u32(payload.data() + 12, counts[1]);
    for (size_t i = 0; i < offsets.size(); ++i) {
      iree_unaligned_store_le_u64(payload.data() + 16 + i * 16, offsets[i]);
      iree_unaligned_store_le_u32(payload.data() + 24 + i * 16,
                                  static_cast<uint32_t>(strengths[i]));
    }
    iree_hal_buffer_release(workspace_);
    workspace_ = nullptr;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        loom_serve_device_storage_pool(device_owner_),
        iree_hal_make_pool_reservation_request(payload.size(), 8),
        iree_infinite_timeout(), &workspace_));
    auto* residency = loom_serve_executables_residency(executables_);
    IREE_ASSERT_OK(loom_serve_residency_acquire_wait(residency));
    const iree_hal_transfer_operation_t upload = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
        .upload =
            {
                .source = payload.data(),
                .target_buffer = workspace_,
                .length = payload.size(),
            },
    };
    uint64_t completion = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_transfer(execution_, 1, &upload, &completion));
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_i32(1),
        iree_hal_buffer_variant_from_ptr_borrowed(&hal_types_, workspace_),
        iree_hal_buffer_variant_from_ptr_borrowed(&hal_types_, output_buffer_)};
    iree_vm_variant_t results[1] = {};
    auto status =
        iree_vm_invoke(loom_serve_program_invocation(control_), submit_indexed_,
                       iree_vm_variant_span_from_array(arguments),
                       iree_vm_variant_span_from_array(results));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
    IREE_ASSERT_OK(status);
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = output_buffer_,
                .target = output_.data(),
                .length = sizeof(output_),
            },
    };
    IREE_ASSERT_OK(
        loom_serve_execution_feedback(execution_, 1, &download, &completion));
    IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
    loom_serve_residency_release(residency);
    size_t first = 0;
    for (size_t item = 0; item < counts.size(); ++item) {
      for (size_t element = 0; element < 4; ++element) {
        int32_t expected = 0;
        for (size_t i = first; i < first + counts[item]; ++i) {
          expected += strengths[i] * (checkpoint_values[i] + 13 +
                                      static_cast<int32_t>(element));
        }
        EXPECT_EQ(output_[item * 4 + element], expected);
      }
      first += counts[item];
    }
    loom_serve_executables_release(executables_);
  }
};

TEST_P(ArenaExecutablesTest, MixedCountsSharePreparedBanksAndNativeCode) {
  IREE_ASSERT_OK(CreateArena(4));
  iree::testing::TempFilePath third("loom_bank_c", ".safetensors");
  iree::testing::TempFilePath fourth("loom_bank_d", ".safetensors");
  WriteCheckpoint(third.path(), 200);
  WriteCheckpoint(fourth.path(), 300);
  IREE_ASSERT_OK(BeginBanks({path_.path(), adapter_path_.path(), third.path(),
                             adapter_path_.path(), fourth.path()}));
  IREE_ASSERT_OK(Advance(false));
  const auto offsets = Offsets();
  ASSERT_EQ(offsets.size(), 5u);
  EXPECT_EQ(offsets[1], offsets[3]);
  EXPECT_NE(offsets[0], offsets[1]);
  EXPECT_NE(offsets[1], offsets[2]);
  EXPECT_NE(offsets[2], offsets[4]);
  ASSERT_NO_FATAL_FAILURE(
      ExecuteBanks({3, 2}, {1, 2, -1, -3, 2}, {0, 100, 200, 100, 300}));
  const auto activity =
      loom_serve_executables_parameters(executables_).activity;
  EXPECT_EQ(activity.activations, 5u);  // One cold root and four shared banks.
  EXPECT_EQ(activity.prepared_bytes, 4u * 8 * sizeof(int32_t));
  const auto commands = compilation_.commands;
  const auto kernels = compilation_.kernels;
  IREE_ASSERT_OK(
      BeginBanks({fourth.path(), adapter_path_.path(), path_.path()}));
  IREE_ASSERT_OK(Advance(false));
  ASSERT_NO_FATAL_FAILURE(ExecuteBanks({1, 2}, {3, -2, 7}, {300, 100, 0}));
  EXPECT_EQ(compilation_.commands, commands);
  EXPECT_EQ(compilation_.kernels, kernels);
  EXPECT_EQ(
      loom_serve_executables_parameters(executables_).activity.activations,
      activity.activations);
  EXPECT_EQ(
      loom_serve_executables_parameters(executables_).activity.prepared_bytes,
      activity.prepared_bytes);
}

TEST_P(ArenaExecutablesTest, CapacityRecyclingProtectsLaterHits) {
  IREE_ASSERT_OK(CreateArena(2));
  iree::testing::TempFilePath third("loom_bank_c", ".safetensors");
  WriteCheckpoint(third.path(), 200);
  IREE_ASSERT_OK(BeginBanks({path_.path(), adapter_path_.path()}));
  IREE_ASSERT_OK(Advance(false));
  const auto original = Offsets();
  ASSERT_NO_FATAL_FAILURE(ExecuteBanks({1, 1}, {1, 1}, {0, 100}));
  // A is the older hit but appears after C's miss. It must survive resolution.
  IREE_ASSERT_OK(BeginBanks({third.path(), path_.path()}));
  IREE_ASSERT_OK(Advance(false));
  const auto replacement = Offsets();
  EXPECT_EQ(replacement[0], original[1]);
  EXPECT_EQ(replacement[1], original[0]);
  ASSERT_NO_FATAL_FAILURE(ExecuteBanks({1, 1}, {2, -1}, {200, 0}));
  EXPECT_EQ(
      loom_serve_executables_parameters(executables_).activity.activations, 4u);
}

TEST_P(ArenaExecutablesTest, FailedFirstPreparationDoesNotPublishLayout) {
  IREE_ASSERT_OK(CreateArena(2));
  const auto before = loom_serve_memory_pool_statistics(pool_).reserved_bytes;
  IREE_ASSERT_OK(BeginBanks({path_.path()}, "bank_missing"));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_NOT_FOUND, Advance(false));
  loom_serve_executables_release(executables_);
  ASSERT_NO_FATAL_FAILURE(CollectIdle());
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).reserved_bytes, before);
  IREE_ASSERT_OK(BeginBanks({path_.path(), adapter_path_.path()}));
  IREE_ASSERT_OK(Advance(false));
  ASSERT_NO_FATAL_FAILURE(ExecuteBanks({1, 1}, {1, 2}, {0, 100}));
}

TEST_P(ArenaExecutablesTest, CancelledFirstCaptureReleasesProvisionalArena) {
  IREE_ASSERT_OK(CreateArena(2));
  const auto before = loom_serve_memory_pool_statistics(pool_).reserved_bytes;
  IREE_ASSERT_OK(BeginBanks({path_.path(), adapter_path_.path()}));
  bool complete = false;
  while (loom_serve_memory_pool_statistics(pool_).reserved_bytes == before) {
    const uint32_t ticket = iree_notification_prepare_wait(&notification_);
    auto status =
        loom_serve_executables_advance(executables_, false, &complete);
    if (iree_status_is_ok(status) && !complete &&
        loom_serve_memory_pool_statistics(pool_).reserved_bytes == before) {
      iree_notification_commit_wait(&notification_, ticket, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    } else {
      iree_notification_cancel_wait(&notification_);
    }
    IREE_ASSERT_OK(status);
    ASSERT_FALSE(complete);
  }
  IREE_ASSERT_OK(Advance(true));
  loom_serve_executables_release(executables_);
  ASSERT_NO_FATAL_FAILURE(CollectIdle());
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).reserved_bytes, before);
  IREE_ASSERT_OK(BeginBanks({adapter_path_.path(), path_.path()}));
  IREE_ASSERT_OK(Advance(false));
  ASSERT_NO_FATAL_FAILURE(ExecuteBanks({1, 1}, {1, 3}, {100, 0}));
}

TEST_P(ArenaExecutablesTest, RepeatedSelectionsDoNotMultiplyBankStorage) {
  IREE_ASSERT_OK(CreateArena(1));
  std::vector<std::string> paths(33, path_.path());
  IREE_ASSERT_OK(BeginBanks(paths));
  IREE_ASSERT_OK(Advance(false));
  const auto offsets = Offsets();
  ASSERT_EQ(offsets.size(), 33u);
  for (uint64_t offset : offsets) {
    EXPECT_EQ(offset, offsets[0]);
  }
  ASSERT_NO_FATAL_FAILURE(ExecuteBanks({17, 16}, std::vector<int32_t>(33, 2),
                                       std::vector<int32_t>(33, 0)));
  const auto activity =
      loom_serve_executables_parameters(executables_).activity;
  EXPECT_EQ(activity.activations, 2u);
  EXPECT_EQ(activity.prepared_bytes, 8u * sizeof(int32_t));
  const auto commands = compilation_.commands;
  IREE_ASSERT_OK(BeginBanks({path_.path()}));
  IREE_ASSERT_OK(Advance(false));
  ASSERT_NO_FATAL_FAILURE(ExecuteBanks({0, 1}, {-3}, {0}));
  EXPECT_EQ(compilation_.commands, commands);
  EXPECT_EQ(
      loom_serve_executables_parameters(executables_).activity.activations,
      activity.activations);
}

TEST_P(ArenaExecutablesTest, PublishedLayoutRejectsIncompatiblePacking) {
  IREE_ASSERT_OK(CreateArena(2));
  IREE_ASSERT_OK(BeginBanks({path_.path()}));
  IREE_ASSERT_OK(Advance(false));
  ASSERT_NO_FATAL_FAILURE(ExecuteBanks({1, 0}, {1}, {0}));
  IREE_ASSERT_OK(BeginBanks({path_.path()}, "bank_missing"));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Advance(false));
  loom_serve_executables_release(executables_);
  IREE_ASSERT_OK(BeginBanks({path_.path()}));
  IREE_ASSERT_OK(Advance(false));
  ASSERT_NO_FATAL_FAILURE(ExecuteBanks({1, 0}, {2}, {0}));
  EXPECT_EQ(
      loom_serve_executables_parameters(executables_).activity.activations, 2u);
}

TEST_P(ArenaExecutablesTest, ImpossibleUnionPreservesUsableBanks) {
  IREE_ASSERT_OK(CreateArena(2));
  iree::testing::TempFilePath third("loom_bank_c", ".safetensors");
  WriteCheckpoint(third.path(), 200);
  IREE_ASSERT_OK(BeginBanks({path_.path(), adapter_path_.path()}));
  IREE_ASSERT_OK(Advance(false));
  ASSERT_NO_FATAL_FAILURE(ExecuteBanks({1, 1}, {1, 1}, {0, 100}));
  IREE_ASSERT_OK(
      BeginBanks({third.path(), path_.path(), adapter_path_.path()}));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Advance(false));
  loom_serve_executables_release(executables_);
  IREE_ASSERT_OK(BeginBanks({path_.path(), adapter_path_.path()}));
  IREE_ASSERT_OK(Advance(false));
  ASSERT_NO_FATAL_FAILURE(ExecuteBanks({1, 1}, {2, 3}, {0, 100}));
  EXPECT_EQ(
      loom_serve_executables_parameters(executables_).activity.activations, 3u);
}

TEST_P(ArenaExecutablesTest, ReconfigurationPreservesCapturedBanksAndArena) {
  IREE_ASSERT_OK(CreateArena(2));
  IREE_ASSERT_OK(BeginBanks({path_.path(), adapter_path_.path()}));
  IREE_ASSERT_OK(Advance(false));
  const auto offsets = Offsets();
  const auto reserved = loom_serve_memory_pool_statistics(pool_).reserved_bytes;
  // Selecting a new source generation cannot change the captured invocation.
  IREE_ASSERT_OK(ReplaceSources());
  EXPECT_EQ(Offsets(), offsets);
  ASSERT_NO_FATAL_FAILURE(ExecuteBanks({1, 1}, {2, -3}, {0, 100}));
  const auto activations =
      loom_serve_executables_parameters(executables_).activity.activations;

  // The next capture uses new preparers but the same canonical arena layout.
  // Finite capacity requires retirement of the old generation's idle banks.
  IREE_ASSERT_OK(BeginBanks({adapter_path_.path(), path_.path()}));
  IREE_ASSERT_OK(Advance(false));
  ASSERT_NO_FATAL_FAILURE(ExecuteBanks({1, 1}, {-1, 7}, {100, 0}));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).reserved_bytes, reserved);
  EXPECT_EQ(
      loom_serve_executables_parameters(executables_).activity.activations,
      activations + 2);

  auto update = loom_serve_executables_update(executables_);
  IREE_ASSERT_OK(update.select(update.self, nullptr));
  IREE_ASSERT_OK(BeginBanks({path_.path(), adapter_path_.path()}));
  IREE_ASSERT_OK(Advance(false));
  ASSERT_NO_FATAL_FAILURE(ExecuteBanks({1, 1}, {5, 3}, {0, 100}));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).reserved_bytes, reserved);
  EXPECT_EQ(
      loom_serve_executables_parameters(executables_).activity.activations,
      activations + 4);
}

INSTANTIATE_TEST_SUITE_P(Elastic, ArenaExecutablesTest,
                         ::testing::Values(true));

}  // namespace loom_serve::testing
