// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/weights.h"

#include <array>
#include <filesystem>
#include <fstream>

#include "experimental/loom_serve/runtime/device.h"
#include "experimental/loom_serve/runtime/execution.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

IREE_FLAG(string, weight_sources, "", "Weight loader source catalog.");

namespace {

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
    IREE_ASSERT_OK(
        loom_serve_device_create(&options, &device_owner_, allocator_));
    device_ = loom_serve_device_handle(device_owner_);
    dispatch_ = loom_serve_device_dispatch_queue(device_owner_);
    transfer_ = loom_serve_device_transfer_queue(device_owner_);
    execution_ = loom_serve_device_execution(device_owner_);
    pool_ = loom_serve_device_memory_pool(device_owner_);
    directory_ =
        std::filesystem::path(FLAG_weight_sources).parent_path().string();
    IREE_ASSERT_OK(loom_serve_jit_create(
        device_, dispatch_, iree_make_cstring_view(directory_.c_str()), nullptr,
        allocator_, &jit_));
    WriteCheckpoint(path_.path(), 0);
    iree_hal_buffer_params_t params = {};
    params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    params.usage =
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device_), params, sizeof(output_),
        &output_buffer_));
  }

  void TearDown() override {
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
    for (auto* plan : plans_) {
      IREE_EXPECT_OK(loom_serve_weights_destroy(plan));
    }
    iree_hal_buffer_release(output_buffer_);
    loom_serve_jit_destroy(jit_);
    IREE_EXPECT_OK(loom_serve_device_destroy(device_owner_));
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

  iree_status_t LoadRoots(iree_host_size_t shared_count,
                          iree_host_size_t root_count,
                          const loom_serve_weight_root_t* roots,
                          const std::string& path) {
    const std::string policy = directory_ + "/policy.loom";
    auto** plan = &plans_[plan_count_++];
    iree_status_t status = loom_serve_weights_create(
        device_, transfer_, dispatch_, pool_, jit_,
        IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, shared_count, root_count, roots,
        iree_make_cstring_view(path.c_str()),
        iree_make_cstring_view(policy.c_str()), plan, allocator_);
    if (iree_status_is_ok(status)) {
      status = loom_serve_weights_activate(*plan);
    }
    return status;
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
  // Production runtime owner outliving every weight and accepted operation.
  loom_serve_device_t* device_owner_ = nullptr;
  // Borrowed live GPU used by compilation and execution.
  iree_hal_device_t* device_ = nullptr;
  // Exact dispatch queue borrowed from device.
  iree_hal_queue_t* dispatch_ = nullptr;
  // Exact transfer queue borrowed from device.
  iree_hal_queue_t* transfer_ = nullptr;
  // Timelines borrowed from device_owner_ and used by consuming commands.
  loom_serve_execution_t* execution_ = nullptr;
  // Explicitly elastic backing or NULL for the fixed comparison arm.
  loom_serve_memory_pool_t* pool_ = nullptr;
  // Physical slab size; the shared budget holds two unique roots.
  static constexpr uint64_t kSlabBytes = 2 * 1024 * 1024;
  // Owned checkpoint plans, including partially failed construction.
  std::array<loom_serve_weights_t*, 3> plans_ = {};
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
  // Readback backing retained until completion or terminal teardown.
  std::array<int32_t, 8> output_ = {};
};

TEST_P(WeightsTest, MultiplePreparersAndSharedRootsPrepareEachTensorOnce) {
  IREE_ASSERT_OK(Compile(0, "target"));
  IREE_ASSERT_OK(Compile(1, "target"));
  IREE_ASSERT_OK(Compile(2, "auxiliary"));
  IREE_ASSERT_OK(Load(2, 3));
  EXPECT_EQ(buffers_[0][0], buffers_[1][0]);
  Run(0, 70);
  Run(1, 70);
  Run(2, 93);
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
  IREE_ASSERT_OK(loom_serve_weights_activate(plans_[0]));
  IREE_ASSERT_OK(loom_serve_weights_activate(plans_[0]));
  EXPECT_EQ(buffers_[0][0], root);
  EXPECT_EQ(commands_[0], command);
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes,
            2 * kSlabBytes);
  Run(0, 70);
  Run(1, 70);
  Run(2, 93);
}

INSTANTIATE_TEST_SUITE_P(Backing, WeightsTest, ::testing::Values(false, true));

}  // namespace
