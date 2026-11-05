// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <tuple>

#include "iree/base/alignment.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"
#include "loom/serve/runtime/program/program.h"
#include "loom/serve/runtime/program/stage_set.h"
#include "loom/serve/service/text/stages.h"
#include "loom/serve/testing/compiler.h"
#include "loom/serve/testing/device.h"

IREE_FLAG(string, placement_sources, "", "Source-defined stage placement.");
IREE_FLAG_LIST(string, placement_device,
               "Three explicit device URIs for physical-owner qualification.");

namespace {

class TextPlacementTest
    : public ::testing::TestWithParam<std::tuple<std::array<size_t, 3>, bool>> {
 protected:
  static void SetUpTestSuite() {
    const auto allocator = iree_allocator_system();
    IREE_ASSERT_OK(loom_serve_test_host_create(4, allocator, &host_));
    iree_hal_driver_t* driver = nullptr;
    IREE_ASSERT_OK(loom_serve_test_driver_acquire(host_, &driver));
    const auto uris = FLAG_placement_device_list();
    ASSERT_TRUE(uris.count == 0 || uris.count == 3);
    const iree_string_view_t names[] = {IREE_SV("first"), IREE_SV("second"),
                                        IREE_SV("third")};
    std::array<loom_serve_device_options_t, 3> options = {};
    for (size_t i = 0; i < options.size(); ++i) {
      options[i] = {
          .driver = driver,
          .uri = uris.count ? uris.values[i]
                            : loom_serve_test_device()->device_uri,
          .name = names[i],
      };
    }
    auto status = loom_serve_device_group_allocate(
        host_, options.size(), options.data(), allocator, &group_);
    iree_hal_driver_release(driver);
    IREE_ASSERT_OK(status);
    for (size_t backing = 0; backing < 2; ++backing) {
      for (size_t i = 0; i < 3; ++i) {
        const std::string name = "pool" + std::to_string(backing * 3 + i);
        const loom_serve_pool_options_t pool_options = {
            .elastic = backing != 0,
            .weight_slab_size = 4096,
            .state_slab_size = 4096,
            .placement =
                {
                    .kind = uris.count ? LOOM_SERVE_POOL_PLACEMENT_DEVICE
                                       : LOOM_SERVE_POOL_PLACEMENT_AUTOMATIC,
                    .ordinal = static_cast<uint32_t>(i),
                },
        };
        loom_serve_pool_t* pool = nullptr;
        IREE_ASSERT_OK(loom_serve_device_group_create_pool(
            group_, {iree_make_cstring_view(name.c_str()), 7}, &pool_options,
            &pool));
      }
    }
  }

  static void TearDownTestSuite() {
    IREE_EXPECT_OK(loom_serve_device_group_free(group_));
    group_ = nullptr;
    loom_serve_host_release(host_);
    host_ = nullptr;
  }

  void SetUp() override {
    directory_ =
        std::filesystem::path(FLAG_placement_sources).parent_path().string();
    sources_ = {
        .base_directory = iree_make_cstring_view(directory_.c_str()),
        .count = providers_.size(),
        .values = providers_.data(),
    };
    for (size_t i = 0; i < devices_.size(); ++i) {
      devices_[i] = loom_serve_device_group_at(group_, i);
      pools_[i] =
          loom_serve_device_group_pool_at(group_, std::get<0>(GetParam())[i]);
      IREE_ASSERT_OK(loom_serve_device_create_execution(devices_[i], pools_[i],
                                                        &executions_[i]));
      IREE_ASSERT_OK(loom_serve_device_create_jit(
          devices_[i], &sources_, nullptr, &reports_[i], &jits_[i]));
    }
    IREE_ASSERT_OK(iree_vm_environment_allocate(allocator_, &environment_));
    const iree_vm_ref_type_table_t* table = nullptr;
    IREE_ASSERT_OK(iree_hal_module_register_types(environment_, &table));
    IREE_ASSERT_OK(iree_hal_module_types_resolve(table, &types_));
    const std::string header =
        R"({"raw":{"dtype":"I32","shape":[8],"data_offsets":[0,32]}})";
    std::ofstream file(checkpoint_.path(), std::ios::binary);
    for (unsigned i = 0; i < 8; ++i) {
      file.put(static_cast<char>(uint64_t{header.size()} >> (8 * i)));
    }
    file.write(header.data(), header.size());
    for (int32_t i = 0; i < 8; ++i) {
      const int32_t value = 30 + i;
      uint8_t bytes[sizeof(value)];
      iree_unaligned_store_le_u32(bytes, value);
      file.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    }
    file.close();
    ASSERT_TRUE(file.good());
  }

  void TearDown() override {
    IREE_EXPECT_OK(
        loom_serve_execution_drain_all(executions_.size(), executions_.data()));
    loom_serve_program_free(program_);
    iree_vm_module_release(module_);
    ReleaseStages(stages_);
    loom_serve_stage_set_free(compiled_);
    for (auto* residency : residencies_) {
      loom_serve_residency_free(residency);
    }
    for (auto* group : parameter_groups_) {
      loom_serve_residency_group_free(group);
    }
    for (size_t i = 0; i < parameters_.count; ++i) {
      IREE_EXPECT_OK(loom_serve_weights_await_retirement(parameters_.plans[i]));
      IREE_EXPECT_OK(loom_serve_weights_free(parameters_.plans[i]));
    }
    iree_allocator_free(allocator_, parameters_.plans);
    iree_allocator_free(allocator_, parameters_.sources);
    loom_serve_parameter_sources_release(indexed_);
    loom_serve_preparation_free(preparation_);
    iree_vm_environment_free(environment_);
    for (auto* buffer : outputs_) {
      iree_hal_buffer_release(buffer);
    }
    ReleaseCatalogs();
    for (size_t i = 0; i < devices_.size(); ++i) {
      loom_serve_execution_release(executions_[i]);
      if (devices_[i]) {
        loom_serve_device_trim_programs(devices_[i]);
        IREE_EXPECT_OK(loom_serve_test_workspace_await_idle(
            loom_serve_pool_workspace(pools_[i])));
        IREE_EXPECT_OK(loom_serve_test_pool_trim_and_drain(pools_[i], 0));
        const auto memory = loom_serve_pool_statistics(pools_[i]);
        EXPECT_EQ(memory.retained.committed_bytes, 0u);
        EXPECT_EQ(memory.workspace.bytes_committed, 0u);
      }
    }
    if (checkpoint_.Exists()) {
      EXPECT_TRUE(checkpoint_.Remove());
    }
    IREE_EXPECT_OK(status_);
  }

  iree_status_t Wait(iree_hal_semaphore_t* semaphore, uint64_t value) {
    std::array<loom_serve_workspace_t*, 3> workspaces;
    for (size_t i = 0; i < pools_.size(); ++i) {
      workspaces[i] = loom_serve_pool_workspace(pools_[i]);
    }
    return loom_serve_workspace_wait(workspaces.size(), workspaces.data(),
                                     semaphore, value);
  }

  void ReleaseCatalogs() {
    for (auto*& jit : jits_) {
      loom_serve_jit_release(jit);
      jit = nullptr;
    }
  }

  iree_status_t Prepare(bool fail) {
    iree_vm_ref_types_t vm_types;
    IREE_RETURN_IF_ERROR(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment_, IREE_SV("vm")),
        &vm_types));
    iree_vm_buffer_t* path = nullptr;
    const std::string name = checkpoint_.path();
    IREE_RETURN_IF_ERROR(iree_vm_buffer_clone(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_const_byte_span(name.data(), name.size()), 1, allocator_,
        &path));
    iree_vm_variant_t arguments[] = {
        iree_vm_buffer_variant_from_ptr_move(&vm_types, &path),
        iree_vm_variant_from_i32(fail)};
    iree_vm_variant_t result = {};
    loom_serve_compilation_report_t report = {};
    auto status = loom_serve_preparation_allocate(
        loom_serve_host_target_environment(host_), environment_, &sources_,
        IREE_SV("prepare.loom"), IREE_SV("prepare"), devices_.size(),
        LOOM_SERVE_PREPARATION_FLAG_PARAMETERS,
        iree_vm_variant_span_from_array(arguments), {&result, 1},
        iree_vm_module_span_empty(), &report, allocator_, &preparation_);
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    IREE_RETURN_IF_ERROR(status);
    int32_t count = 0;
    IREE_RETURN_IF_ERROR(iree_vm_i32_from_variant(result, &count));
    EXPECT_EQ(count, devices_.size());
    return iree_ok_status();
  }

  iree_status_t Compile(bool release_catalogs = false) {
    auto status = std::get<1>(GetParam())
                      ? loom_serve_stage_set_compile_async(
                            devices_.data(), preparation_, &sources_, Complete,
                            this, allocator_)
                      : loom_serve_stage_set_compile_catalog_async(
                            jits_.data(), preparation_, nullptr, Complete, this,
                            allocator_);
    IREE_RETURN_IF_ERROR(status);
    if (release_catalogs) {
      ReleaseCatalogs();
    }
    return AwaitCompletion();
  }

  iree_status_t AwaitCompletion() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return complete_; });
    const auto status = status_;
    status_ = iree_ok_status();
    return status;
  }

  void ReleaseStages(const std::array<loom_serve_text_stage_t, 3>& stages) {
    for (const auto& stage : stages) {
      iree_hal_command_buffer_release(stage.command);
      for (uint32_t i = 0; i < stage.fixed_buffer_count; ++i) {
        iree_hal_buffer_release(stage.fixed_buffers[i]);
      }
      iree_allocator_free(allocator_, stage.fixed_buffers);
      loom_serve_jit_stage_free(stage.compiled);
    }
  }

  iree_status_t Record() {
    complete_ = false;
    IREE_RETURN_IF_ERROR(loom_serve_text_stages_record_async(
        loom_serve_host_preparation_sink(host_),
        loomc_task_pool_worker_count(loom_serve_host_task_pool(host_)),
        IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, stages_.size(), stages_.data(),
        [](void* user_data, iree_status_t status) {
          auto* self = static_cast<TextPlacementTest*>(user_data);
          std::lock_guard<std::mutex> lock(self->mutex_);
          self->status_ = status;
          self->complete_ = true;
          self->condition_.notify_one();
        },
        this, allocator_));
    return AwaitCompletion();
  }

  void Bind(bool rebind = false) {
    ASSERT_EQ(compiled_->count, 4u);
    for (size_t i = 0; i < stages_.size(); ++i) {
      ASSERT_NE(compiled_->values[i], nullptr);
      stages_[i].program = *loom_serve_jit_stage_program(compiled_->values[i]);
      stages_[i].compiled = compiled_->values[i];
      compiled_->values[i] = nullptr;
    }
    // The target prefix uses the production six-buffer text ABI; auxiliary
    // commands use its ordinary one-buffer form. Direct stages have no roots.
    IREE_ASSERT_OK(loom_serve_text_stages_bind(
        devices_.data(), pools_.data(), preparation_, 1, stages_.size(),
        stages_.data(), &parameters_, allocator_));
    ASSERT_EQ(parameters_.count, 3u);
    const loom_model_file_list_t* paths = nullptr;
    const auto count =
        loom_serve_preparation_local_sources(preparation_, &paths);
    IREE_ASSERT_OK(loom_serve_parameter_sources_create(
        nullptr, &sources_, count, paths, allocator_, &indexed_));
    const std::array<uint32_t, 3> owners = {2, 0, 1};
    for (size_t i = 0; i < parameters_.count; ++i) {
      const auto* declaration = parameters_.sources[i];
      ASSERT_EQ(declaration->owner_index, owners[i]);
      const uint32_t owner = declaration->owner_index;
      IREE_ASSERT_OK(
          loom_serve_weights_prepare(parameters_.plans[i], host_, jits_[owner],
                                     IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
                                     &sources_, declaration, indexed_));
      auto* cache = loom_serve_pool_residency_cache(pools_[owner]);
      IREE_ASSERT_OK(loom_serve_residency_group_allocate(
          cache, 1, &parameters_.plans[i], allocator_,
          &parameter_groups_[owner]));
      IREE_ASSERT_OK(loom_serve_residency_allocate(
          cache, executions_[owner], 1, &parameter_groups_[owner], allocator_,
          &residencies_[owner]));
    }
    if (rebind) {
      auto baseline = stages_;
      stages_ = {};
      for (size_t i = 0; i < stages_.size(); ++i) {
        stages_[i].program = baseline[i].program;
        stages_[i].compiled = baseline[i].compiled;
        baseline[i].compiled = nullptr;
      }
      auto status = loom_serve_text_stages_rebind(
          preparation_, 1, stages_.size(), baseline.data(), stages_.data(),
          allocator_);
      ReleaseStages(baseline);
      IREE_ASSERT_OK(status);
    }
    IREE_ASSERT_OK(Record());
    std::array<loom_serve_stage_t, 4> bindings = {};
    for (size_t i = 0; i < stages_.size(); ++i) {
      const uint32_t execution =
          loom_serve_preparation_stage(preparation_, i)->execution_index;
      EXPECT_EQ(iree_hal_command_buffer_queue_family(stages_[i].command),
                iree_hal_queue_family(
                    loom_serve_device_dispatch_queue(devices_[execution])));
      bindings[i] = loom_serve_text_stage_binding(&stages_[i], execution);
    }
    bindings[3] = {
        .binding_count = 1,
        .execution_index =
            loom_serve_preparation_stage(preparation_, 3)->execution_index,
        .dispatch = loom_serve_jit_stage_dispatch(compiled_->values[3]),
    };
    const loom_serve_module_feedback_t feedback = {
        .execution_index = 0,
        .destination = iree_make_byte_span(output_.data(), sizeof(output_)),
    };
    const loom_serve_module_options_t options = {
        .binding_capacity = 6,
        .executions = {executions_.size(), executions_.data()},
        .stages = {bindings.size(), bindings.data()},
        .feedback = {1, &feedback},
    };
    IREE_ASSERT_OK(
        loom_serve_module_create(&types_, options, allocator_, &module_));
    const iree_string_view_t root = IREE_SV("step");
    loom_serve_compilation_report_t report = {};
    IREE_ASSERT_OK(loom_serve_program_allocate(
        loom_serve_host_target_environment(host_), environment_, &sources_,
        IREE_SV("control.loom"), 1, &root, {&module_, 1}, &report, allocator_,
        &program_));
    IREE_ASSERT_OK(iree_vm_process_lookup_function(
        loom_serve_program_process(program_), IREE_SV("model"), root, &step_));
    // Native commands, plans, and the VM retain no borrowed source/JIT state.
    ReleaseCatalogs();
    loom_serve_preparation_free(preparation_);
    preparation_ = nullptr;
    loom_serve_parameter_sources_release(indexed_);
    indexed_ = nullptr;
    loom_serve_stage_set_free(compiled_);
    compiled_ = nullptr;
  }

  void Run() {
    for (auto* residency : residencies_) {
      loom_serve_residency_admission_t admission;
      IREE_ASSERT_OK(loom_serve_residency_try_admit(1, &residency, &admission));
      ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
    }
    for (auto* residency : residencies_) {
      loom_serve_residency_cancel_admission(residency);
    }
    std::array<iree_hal_pool_reservation_request_t, 6> requests;
    requests.fill(iree_hal_make_pool_reservation_request(sizeof(output_), 0));
    IREE_ASSERT_OK(loom_serve_execution_alloca_from_pool(
        executions_[0], loom_serve_pool_storage(pools_[0]), requests.size(),
        requests.data(), outputs_.data()));
    // Every consuming execution imports loading and output-backing readiness.
    // Capture first, so installing peer waits cannot create circular edges.
    std::array<loom_serve_execution_frontier_t, 3> frontiers;
    for (size_t i = 0; i < frontiers.size(); ++i) {
      frontiers[i] = loom_serve_execution_frontier(executions_[i]);
    }
    for (auto* execution : executions_) {
      for (auto& frontier : frontiers) {
        IREE_ASSERT_OK(loom_serve_execution_barrier(
            execution, {2, frontier.semaphores, frontier.values}));
      }
    }
    std::array<iree_vm_variant_t, 6> arguments = {};
    for (size_t i = 0; i < outputs_.size(); ++i) {
      arguments[i] =
          iree_hal_buffer_variant_from_ptr_borrowed(&types_, outputs_[i]);
    }
    iree_vm_variant_t result = {};
    auto status =
        iree_vm_invoke(loom_serve_program_invocation(program_), step_,
                       {arguments.data(), arguments.size()}, {&result, 1});
    iree_vm_variant_span_reset({arguments.data(), arguments.size()});
    IREE_ASSERT_OK(status);
    int64_t completion = 0;
    IREE_ASSERT_OK(iree_vm_i64_from_variant(result, &completion));
    auto* feedback =
        loom_serve_execution_frontier(executions_[0]).semaphores[1];
    IREE_ASSERT_OK(Wait(feedback, completion));
    for (int32_t i = 0; i < 8; ++i) {
      EXPECT_EQ(output_[i], 97 + 3 * i);
    }
    for (size_t index = 1; index < outputs_.size(); ++index) {
      const iree_hal_transfer_operation_t download = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
          .download = {outputs_[index], 0, output_.data(), sizeof(output_)},
      };
      uint64_t value = 0;
      IREE_ASSERT_OK(
          loom_serve_execution_feedback(executions_[0], 1, &download, &value));
      IREE_ASSERT_OK(Wait(feedback, value));
      for (int32_t i = 0; i < 8; ++i) {
        EXPECT_EQ(output_[i], 30 + i);
      }
    }
    IREE_ASSERT_OK(
        loom_serve_execution_drain_all(executions_.size(), executions_.data()));
    for (auto* pool : pools_) {
      if (loom_serve_pool_memory(pool)) {
        EXPECT_GE(loom_serve_pool_statistics(pool).retained.committed_bytes,
                  sizeof(output_));
      } else {
        iree_hal_pool_stats_t statistics;
        iree_hal_pool_query_stats(loom_serve_pool_storage(pool), &statistics);
        EXPECT_GE(statistics.bytes_reserved, sizeof(output_));
      }
    }
    for (auto* residency : residencies_) {
      loom_serve_residency_release(residency);
      // Only the admission pin was released; TearDown still owns the object.
      // NOLINTNEXTLINE(iree-refcount-lifecycle)
      EXPECT_EQ(loom_serve_residency_snapshot(residency).pin_count, 0u);
    }
    for (size_t i = 0; i < parameters_.count; ++i) {
      EXPECT_EQ(loom_serve_weights_activity(parameters_.plans[i]).activations,
                1u);
      EXPECT_EQ(loom_serve_weights_activity(parameters_.plans[i]).read_bytes,
                sizeof(output_));
    }
  }

 private:
  static void Complete(void* user_data, iree_status_t status,
                       loom_serve_stage_set_t* stages) {
    auto* self = static_cast<TextPlacementTest*>(user_data);
    std::lock_guard<std::mutex> lock(self->mutex_);
    self->status_ = status;
    self->compiled_ = stages;
    self->complete_ = true;
    self->condition_.notify_one();
  }

 protected:
  // Independently retained native compilation products in declaration order.
  loom_serve_stage_set_t* compiled_ = nullptr;

 private:
  // Owns every fixture allocation through native retirement.
  const iree_allocator_t allocator_ = iree_allocator_system();
  // Host shared by all physical devices and private compiler catalogs.
  inline static loom_serve_host_t* host_ = nullptr;
  // Application topology owner; all model resources retire before destruction.
  inline static loom_serve_device_group_t* group_ = nullptr;
  // Ordered deployment borrowed from the group.
  std::array<loom_serve_device_t*, 3> devices_ = {};
  // Explicit storage owners, independently selected from the device records.
  std::array<loom_serve_pool_t*, 3> pools_ = {};
  // Independent native lifetimes indexed by source execution ordinals.
  std::array<loom_serve_execution_t*, 3> executions_ = {};
  // Exact-family catalogs released before execution.
  std::array<loom_serve_jit_t*, 3> jits_ = {};
  // Separate report storage for independently synchronized catalogs.
  std::array<loom_serve_compilation_report_t, 3> reports_ = {};
  // Stable source directory through every asynchronous compilation.
  std::string directory_;
  // Source provider names borrowed by sources_.
  const std::array<iree_string_view_t, 1> providers_ = {
      IREE_SV("kernels.loom")};
  // Immutable local source namespace.
  loom_serve_sources_t sources_ = {};
  // Actual file input, retained through deferred native reads.
  iree::testing::TempFilePath checkpoint_{"loom_placement", ".safetensors"};
  // Source declaration ownership ends after recording and preparation.
  loom_serve_preparation_t* preparation_ = nullptr;
  // Shared indexed file catalog, released before activation.
  loom_serve_parameter_sources_t* indexed_ = nullptr;
  // Production text binder outputs, including partial failure ownership.
  std::array<loom_serve_text_stage_t, 3> stages_ = {};
  // Physical checkpoint domains independently owned by the three devices.
  loom_serve_text_parameters_t parameters_ = {};
  // Parameter protection in physical-owner order.
  std::array<loom_serve_residency_group_t*, 3> parameter_groups_ = {};
  // Complete model admission set, indexed by the same deployment.
  std::array<loom_serve_residency_t*, 3> residencies_ = {};
  // Separate live target bindings accessible by all declared peers.
  std::array<iree_hal_buffer_t*, 6> outputs_ = {};
  // Feedback bytes survive all accepted downloads and native retirement.
  std::array<int32_t, 8> output_ = {};
  // Reference type owner for declaration capture and the linked model VM.
  iree_vm_environment_t* environment_ = nullptr;
  // Borrowed HAL reference identities from environment_.
  iree_hal_module_types_t types_ = {};
  // Native runner retaining the complete execution and stage tables.
  iree_vm_module_t* module_ = nullptr;
  // Linked source control and serialized invocation.
  loom_serve_program_t* program_ = nullptr;
  // Bound source entry borrowing program_.
  iree_vm_function_t step_ = {};
  // Synchronizes final compilation publication with the outer test thread.
  std::mutex mutex_;
  // Test-thread notification; no compiler worker waits on a child.
  std::condition_variable condition_;
  // Final compilation or recording callback has relinquished its inputs.
  bool complete_ = false;
  // Owned final outcome, transferred by AwaitCompletion().
  iree_status_t status_ = iree_ok_status();
};

TEST_P(TextPlacementTest, SourcePlacementSurvivesCatalogsAndRetiresEveryOwner) {
  IREE_ASSERT_OK(Prepare(false));
  IREE_ASSERT_OK(Compile());
  ASSERT_NO_FATAL_FAILURE(Bind());
  ASSERT_NO_FATAL_FAILURE(Run());
}

TEST_P(TextPlacementTest, FailedOwnerCompilationJoinsOtherNativeProducts) {
  IREE_ASSERT_OK(Prepare(true));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_NOT_FOUND, Compile(true));
  ASSERT_NE(compiled_, nullptr);
  ASSERT_EQ(compiled_->count, 4u);
  EXPECT_NE(compiled_->values[0], nullptr);
  EXPECT_EQ(compiled_->values[1], nullptr);
  EXPECT_NE(compiled_->values[2], nullptr);
  EXPECT_NE(compiled_->values[3], nullptr);
}

TEST_P(TextPlacementTest, ReplacementRecordingPreservesEveryExecutionFamily) {
  IREE_ASSERT_OK(Prepare(false));
  IREE_ASSERT_OK(Compile());
  ASSERT_NO_FATAL_FAILURE(Bind(/*rebind=*/true));
  ASSERT_NO_FATAL_FAILURE(Run());
}

INSTANTIATE_TEST_SUITE_P(
    Placement, TextPlacementTest,
    ::testing::Combine(
        ::testing::Values(std::array<size_t, 3>{0, 1, 2},   // Fixed.
                          std::array<size_t, 3>{3, 4, 5},   // Elastic.
                          std::array<size_t, 3>{3, 1, 5},   // Mixed.
                          std::array<size_t, 3>{3, 3, 3}),  // Aliased.
        ::testing::Bool()));

}  // namespace
