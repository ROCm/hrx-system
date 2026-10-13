// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/task/device.h"
#include "iree/hal/slab_pool.h"
#include "iree/module/hal/module.h"
#include "iree/module/hal/types.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/sync.h"
#include "loom/tooling/target/vm/hal_copy_bytecode.h"

namespace {

class HALSourceCopyTest : public ::testing::TestWithParam<const char*> {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(iree_allocator_system(), &environment_));
    const iree_vm_ref_type_table_t* table = nullptr;
    IREE_ASSERT_OK(iree_hal_module_register_types(environment_, &table));
    IREE_ASSERT_OK(iree_hal_module_types_resolve(table, &types_));
    IREE_ASSERT_OK(iree_hal_module_create(environment_, iree_allocator_system(),
                                          &hal_module_));
    const auto* file = loom_vm_hal_copy_bytecode_create();
    IREE_ASSERT_OK(iree_vm_bytecode_module_create(
        environment_, IREE_SV("copy"),
        {
            .contents = iree_make_const_byte_span(file[0].data, file[0].size),
            .deallocator = iree_allocator_null(),
        },
        iree_allocator_system(), &module_));
    iree_vm_module_t* libraries[] = {hal_module_};
    IREE_ASSERT_OK(iree_vm_program_create(
        {module_, iree_vm_module_span_from_array(libraries)},
        iree_allocator_system(), &program_));
    IREE_ASSERT_OK(iree_vm_invocation_initialize(
        iree_make_byte_span(storage_.data(), storage_.size()), &invocation_));
    IREE_ASSERT_OK(iree_vm_process_create(program_, invocation_,
                                          iree_vm_variant_span_empty(),
                                          iree_allocator_system(), &process_));

    iree_task_topology_t topology;
    iree_task_topology_initialize_from_group_count(1, &topology);
    iree_task_executor_options_t executor_options;
    iree_task_executor_options_initialize(&executor_options);
    iree_status_t status = iree_task_executor_create(
        executor_options, &topology, iree_allocator_system(), &executor_);
    iree_task_topology_deinitialize(&topology);
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(
        iree_hal_allocator_create_heap(iree_allocator_system(), &allocator_));
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        1, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &progress_));
    iree_hal_task_device_params_t params;
    iree_hal_task_device_params_initialize(&params);
    auto create_params = iree_hal_device_create_params_default();
    create_params.proactor_pool = progress_;
    iree_hal_device_t* device = nullptr;
    IREE_ASSERT_OK(iree_hal_task_device_create(
        IREE_SV("copy"), &params, 1, &executor_, 0, nullptr, allocator_,
        &create_params, iree_allocator_system(), &device));
    iree_async_frontier_tracker_t* tracker = nullptr;
    status = iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &tracker);
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_group_create_from_device(
          device, tracker, {this, CountGroupFree}, &group_);
    }
    iree_async_frontier_tracker_release(tracker);
    iree_hal_device_release(device);
    IREE_ASSERT_OK(status);
    group_address_ = group_;
    device = iree_hal_device_group_device_at(group_, 0);
    queue_ = iree_hal_device_queue(device, 0, 0);
    for (auto& semaphore : semaphores_) {
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
    }
    const iree_hal_pool_family_access_t family = {
        .family = iree_hal_queue_family(queue_),
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
    };
    const iree_hal_pool_scope_t scope = {
        .family_count = 1,
        .families = &family,
        .host =
            {
                .access = IREE_HAL_MEMORY_ACCESS_ALL,
                .modes = IREE_HAL_MAPPING_MODE_SCOPED,
            },
    };
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &options,
                                             iree_allocator_system(), &pool_));
    IREE_ASSERT_OK(iree_hal_vm_pool_wrap(group_, pool_, iree_allocator_system(),
                                         &pool_anchor_));
    auto request = iree_hal_make_pool_reservation_request(32, 0);
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_, request, iree_infinite_timeout(), &source_));
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_, request, iree_infinite_timeout(), &target_));
    for (size_t i = 0; i < input_.size(); ++i) {
      input_[i] = static_cast<uint8_t>(i * 7 + 3);
    }
    IREE_ASSERT_OK(
        iree_hal_buffer_map_write(source_, 0, input_.data(), input_.size()));
    std::array<uint8_t, 32> zero = {};
    IREE_ASSERT_OK(
        iree_hal_buffer_map_write(target_, 0, zero.data(), zero.size()));
  }

  void ReleaseNativeStorage() {
    iree_hal_buffer_release(source_);
    source_ = nullptr;
    iree_hal_buffer_release(target_);
    target_ = nullptr;
    iree_hal_pool_release(pool_);
    pool_ = nullptr;
    for (auto& semaphore : semaphores_) {
      iree_hal_semaphore_release(semaphore);
      semaphore = nullptr;
    }
    iree_vm_ref_reset(&pool_anchor_);
  }

  void PrepareArguments() {
    std::array<iree_vm_ref_t, 5> refs = {};
    IREE_ASSERT_OK(iree_hal_vm_queue_wrap(group_, queue_,
                                          iree_allocator_system(), &refs[0]));
    for (size_t i = 0; i < semaphores_.size(); ++i) {
      IREE_ASSERT_OK(iree_hal_vm_semaphore_wrap(
          group_, semaphores_[i], iree_allocator_system(), &refs[i + 1]));
    }
    arguments_ = {
        iree_vm_variant_from_ref_move(&refs[0]),
        iree_vm_variant_from_ref_move(&refs[1]),
        iree_vm_variant_from_ref_move(&refs[2]),
        iree_vm_variant_from_ref_move(&refs[3]),
        iree_vm_variant_from_ref_move(&refs[4]),
        iree_hal_buffer_variant_from_ptr_move(&types_, &source_),
        iree_hal_buffer_variant_from_ptr_retained(&types_, target_),
        iree_vm_variant_from_i64(3),
        iree_vm_variant_from_i64(7),
        iree_vm_variant_from_i64(13),
    };
    iree_hal_device_group_release(group_);
    group_ = nullptr;
  }

  iree_status_t Invoke() {
    iree_vm_function_t function;
    IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
        process_, IREE_SV("copy"), iree_make_cstring_view(GetParam()),
        &function));
    return iree_vm_invoke(invocation_, function,
                          {arguments_.data(), arguments_.size()},
                          {&result_, 1});
  }

  void TearDown() override {
    for (auto& argument : arguments_) {
      iree_vm_variant_reset(&argument);
    }
    ReleaseNativeStorage();
    iree_vm_variant_reset(&result_);
    iree_hal_device_group_release(group_);
    iree_async_proactor_pool_release(progress_);
    iree_hal_allocator_release(allocator_);
    iree_task_executor_release(executor_);
    iree_vm_process_release(process_);
    iree_vm_invocation_deinitialize(invocation_);
    iree_vm_program_release(program_);
    iree_vm_module_release(module_);
    iree_vm_module_release(hal_module_);
    iree_vm_environment_free(environment_);
  }

  static iree_status_t CountGroupFree(void* self,
                                      iree_allocator_command_t command,
                                      const void* parameters, void** pointer) {
    auto* test = static_cast<HALSourceCopyTest*>(self);
    if (command == IREE_ALLOCATOR_COMMAND_FREE &&
        *pointer == test->group_address_) {
      ++test->group_frees_;
    }
    auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  // Canonical VM construction registry, not retained by loaded modules.
  iree_vm_environment_t* environment_ = nullptr;
  // Resolved HAL identities used for the embedding call boundary.
  iree_hal_module_types_t types_ = {};
  // Optional native module, retained independently by the linked program.
  iree_vm_module_t* hal_module_ = nullptr;
  // Compiled source module borrowing build-produced immutable bytecode.
  iree_vm_module_t* module_ = nullptr;
  // Linked immutable executable and mutable process.
  iree_vm_program_t* program_ = nullptr;
  // Owned VM process; the HAL contributes no mutable storage.
  iree_vm_process_t* process_ = nullptr;
  // Fixed invocation storage, including linked native descriptor scratch.
  alignas(iree_max_align_t) std::array<uint8_t, 16384> storage_ = {};
  // Invocation initialized in storage_.
  iree_vm_invocation_t* invocation_ = nullptr;
  // Escaped owner retaining the device group until explicit completion.
  iree_vm_variant_t result_ = {};
  // Owners moved into invocation, or released on setup/argument failure.
  std::array<iree_vm_variant_t, 10> arguments_ = {};
  // Task worker owner retained independently by the device.
  iree_task_executor_t* executor_ = nullptr;
  // Native physical backing allocator.
  iree_hal_allocator_t* allocator_ = nullptr;
  // Progress service owner for the task device.
  iree_async_proactor_pool_t* progress_ = nullptr;
  // Embedding group owner released before invocation.
  iree_hal_device_group_t* group_ = nullptr;
  // Group allocation identity observed by its allocator.
  void* group_address_ = nullptr;
  // Number of final group destructions.
  int group_frees_ = 0;
  // Borrowed exact queue protected by the group and VM queue wrapper.
  iree_hal_queue_t* queue_ = nullptr;
  // Native owners for two waits and two completion timepoints.
  std::array<iree_hal_semaphore_t*, 4> semaphores_ = {};
  // Explicit embedding pool owner retained until submitted work is terminal.
  iree_hal_pool_t* pool_ = nullptr;
  // Explicit parent owner for the embedding's pool, including invocation
  // errors.
  iree_vm_ref_t pool_anchor_ = {};
  // Source owner moved into the VM invocation.
  iree_hal_buffer_t* source_ = nullptr;
  // Destination owner retained for exact output inspection.
  iree_hal_buffer_t* target_ = nullptr;
  // Independent byte oracle for offsets and copied length.
  std::array<uint8_t, 32> input_ = {};
};

TEST_P(HALSourceCopyTest, CapturesDescriptorsAndPreservesExplicitOwners) {
  PrepareArguments();
  IREE_ASSERT_OK(Invoke());
  // Submission returned while both waits are unsatisfied. The module cannot
  // retain pointers into call scratch or block on completion.
  uint64_t observed = UINT64_MAX;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphores_[2], &observed));
  EXPECT_EQ(observed, 0u);
  EXPECT_EQ(group_frees_, 0);
  // Reclaim the complete idle invocation before making the work runnable.
  // Native execution must use captured descriptors, never the old call scratch.
  iree_vm_invocation_deinitialize(invocation_);
  invocation_ = nullptr;
  storage_.fill(0xCD);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphores_[0], 1, nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphores_[1], 1, nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      semaphores_[2], 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  if (iree_string_view_equal(iree_make_cstring_view(GetParam()),
                             IREE_SV("explicit"))) {
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        semaphores_[3], 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }
  std::array<uint8_t, 32> actual;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(target_, 0, actual.data(), actual.size()));
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(actual[i], i >= 7 && i < 20 ? input_[i - 4] : 0) << i;
  }
  ReleaseNativeStorage();
  EXPECT_EQ(group_frees_, 0);
  iree_vm_variant_reset(&result_);
  EXPECT_EQ(group_frees_, 1);
}

TEST_P(HALSourceCopyTest, UnwindsPartiallyConvertedNullTimepoint) {
  PrepareArguments();
  iree_vm_variant_reset(&arguments_[2]);
  arguments_[2] = iree_vm_variant_null();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Invoke());
  uint64_t observed = UINT64_MAX;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphores_[2], &observed));
  EXPECT_EQ(observed, 0u);
  EXPECT_EQ(group_frees_, 0);
  EXPECT_TRUE(iree_vm_variant_is_null(result_));
  ReleaseNativeStorage();
  EXPECT_EQ(group_frees_, 1);
}

TEST_P(HALSourceCopyTest, PropagatesHALFailureWithoutRetirementState) {
  PrepareArguments();
  arguments_[9] = iree_vm_variant_from_i64(33);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Invoke());
  uint64_t observed = UINT64_MAX;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphores_[2], &observed));
  EXPECT_EQ(observed, 0u);
  EXPECT_EQ(group_frees_, 0);
  ReleaseNativeStorage();
  EXPECT_EQ(group_frees_, 1);
}

INSTANTIATE_TEST_SUITE_P(Barriers, HALSourceCopyTest,
                         ::testing::Values("defaults", "explicit"));

}  // namespace
