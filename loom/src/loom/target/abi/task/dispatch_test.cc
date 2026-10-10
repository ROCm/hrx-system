// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Architecture-independent consumer of the grid/fill_byte task library.
// Backend tests supply their artifact through --executable_file.

#include <array>
#include <fstream>
#include <iterator>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/init.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

IREE_FLAG(string, executable_file, "", "Loom-compiled native HAL library.");

namespace {

class HalDispatchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const iree_allocator_t allocator = iree_allocator_system();
    IREE_ASSERT_OK(iree_hal_driver_registry_allocate(allocator, &registry_));
    IREE_ASSERT_OK(iree_hal_register_all_available_drivers(registry_));
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        /*node_count=*/1, /*node_ids=*/nullptr,
        iree_async_proactor_pool_options_default(), allocator,
        &proactor_pool_));
    iree_hal_device_create_params_t params =
        iree_hal_device_create_params_default();
    params.proactor_pool = proactor_pool_;
    IREE_ASSERT_OK(iree_hal_create_device(registry_, IREE_SV("task"), &params,
                                          allocator, &device_));
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), allocator,
        &frontier_tracker_));
    IREE_ASSERT_OK(iree_hal_device_group_create_from_device(
        device_, frontier_tracker_, allocator, &device_group_));
    const iree_hal_device_queue_spec_t* queues =
        iree_hal_device_spec_queues(iree_hal_device_spec(device_));
    for (iree_host_size_t i = 0; i < queues->family_count; ++i) {
      if (queues->families[i].provisioned_queue_count &&
          iree_all_bits_set(queues->families[i].role_flags,
                            IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH |
                                IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER)) {
        queue_ = iree_hal_device_queue(
            device_, static_cast<iree_hal_queue_family_ordinal_t>(i), 0);
        break;
      }
    }
    ASSERT_NE(queue_, nullptr);
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore_));
  }

  void TearDown() override {
    iree_hal_buffer_release(buffer_);
    iree_hal_executable_release(executable_);
    iree_hal_semaphore_release(semaphore_);
    iree_hal_device_group_release(device_group_);
    iree_hal_device_release(device_);
    iree_async_frontier_tracker_release(frontier_tracker_);
    iree_async_proactor_pool_release(proactor_pool_);
    iree_hal_driver_registry_free(registry_);
  }

  void LoadImage() {
    // The compiler runs as a separate build action. This process owns only the
    // artifact, and releases even those input bytes before reflection/dispatch.
    std::ifstream input(FLAG_executable_file, std::ios::binary);
    ASSERT_TRUE(input.good()) << FLAG_executable_file;
    const std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                                  std::istreambuf_iterator<char>());
    ASSERT_FALSE(bytes.empty());
    iree_hal_executable_target_selection_t selection = {
        .family = IREE_SV("cpu"),
    };
    const auto target = iree_hal_device_spec_select_executable_target(
        iree_hal_device_spec(device_), &selection);
    ASSERT_EQ(target.outcome,
              IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
    iree_hal_executable_load_params_t params;
    iree_hal_executable_load_params_initialize(&params);
    params.executable_data =
        iree_make_const_byte_span(bytes.data(), bytes.size());
    IREE_ASSERT_OK(iree_hal_executable_load(
        iree_hal_queue_family(queue_), target.target, &params, &executable_));
  }

  iree_hal_semaphore_list_t Completion(uint64_t* value) {
    return {1, &semaphore_, value};
  }

  // Public driver registry owns the driver factories for the device lifetime.
  iree_hal_driver_registry_t* registry_ = nullptr;
  // Proactors used by the device and its semaphore domain.
  iree_async_proactor_pool_t* proactor_pool_ = nullptr;
  // Frontier state shared with the device group.
  iree_async_frontier_tracker_t* frontier_tracker_ = nullptr;
  // Public task device, retained until all executable resources are released.
  iree_hal_device_t* device_ = nullptr;
  // Installs the semaphore domain used by queue dependency edges.
  iree_hal_device_group_t* device_group_ = nullptr;
  // Borrowed transfer/dispatch queue from the device.
  iree_hal_queue_t* queue_ = nullptr;
  // Completion frontier for ordered uploads, dispatches, and downloads.
  iree_hal_semaphore_t* semaphore_ = nullptr;
  // Loaded native image, independent of compiler and input-byte lifetimes.
  iree_hal_executable_t* executable_ = nullptr;
  // Output allocation including prefix and suffix sentinel words.
  iree_hal_buffer_t* buffer_ = nullptr;
};

TEST_F(HalDispatchTest, GridCallsAndIndependentImageLifetime) {
  ASSERT_NO_FATAL_FAILURE(LoadImage());
  ASSERT_EQ(iree_hal_executable_function_count(executable_), 2u);
  iree_hal_executable_function_t function;
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable_, IREE_SV("grid"), &function));
  iree_hal_executable_function_info_t info;
  IREE_ASSERT_OK(
      iree_hal_executable_function_info(executable_, function, &info));
  EXPECT_EQ(info.binding_count, 1u);
  EXPECT_EQ(info.constant_byte_length, sizeof(uint32_t));
  ASSERT_EQ(info.parameter_count, 2u);
  EXPECT_THAT(info.workgroup_size, ::testing::ElementsAre(1, 1, 1));
  std::array<iree_hal_executable_function_parameter_t, 2> parameters;
  IREE_ASSERT_OK(iree_hal_executable_function_parameters(
      executable_, function, parameters.size(), parameters.data()));
  EXPECT_EQ(parameters[0].type,
            IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING);
  EXPECT_EQ(parameters[0].offset, 0u);
  EXPECT_EQ(parameters[1].type,
            IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT);
  EXPECT_EQ(parameters[1].offset, 0u);
  EXPECT_EQ(parameters[1].size, sizeof(uint32_t));

  constexpr size_t kCount = 7 * 3 * 2;
  constexpr size_t kPrefix = 3;
  std::array<uint32_t, kPrefix + kCount + 2> output = {};
  output.front() = 0x76543210;
  output.back() = 0xfedcba98;
  iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device_), buffer_params, sizeof(output),
      &buffer_));
  uint64_t uploaded = 1, first_done = 2, second_done = 3, downloaded = 4;
  IREE_ASSERT_OK(iree_hal_queue_upload(
      queue_, iree_hal_semaphore_list_empty(), Completion(&uploaded),
      output.data(), buffer_, 0, sizeof(output), /*barriers=*/NULL));
  const iree_hal_buffer_ref_t binding = iree_hal_make_buffer_ref(
      buffer_, kPrefix * sizeof(uint32_t), kCount * sizeof(uint32_t));
  const iree_hal_buffer_ref_list_t bindings = {1, &binding};
  const uint32_t first_bias = 11, second_bias = 29;
  iree_status_t status = iree_hal_queue_dispatch(
      queue_, Completion(&uploaded), Completion(&first_done), executable_,
      function, iree_hal_make_static_dispatch_config(7, 3, 2),
      iree_make_const_byte_span(&first_bias, sizeof(first_bias)), bindings,
      /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE);
  uint64_t completion = uploaded;
  if (iree_status_is_ok(status)) {
    completion = first_done;
    status = iree_hal_queue_dispatch(
        queue_, Completion(&first_done), Completion(&second_done), executable_,
        function, iree_hal_make_static_dispatch_config(3, 5, 2),
        iree_make_const_byte_span(&second_bias, sizeof(second_bias)), bindings,
        /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    completion = second_done;
    status = iree_hal_queue_download(
        queue_, Completion(&second_done), Completion(&downloaded), buffer_, 0,
        output.data(), sizeof(output), /*barriers=*/NULL);
  }
  if (iree_status_is_ok(status)) {
    completion = downloaded;
  }
  status = iree_status_join(
      status,
      iree_hal_semaphore_wait(semaphore_, completion, iree_infinite_timeout(),
                              IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(status);
  EXPECT_EQ(output.front(), 0x76543210u);
  EXPECT_EQ(output.back(), 0xfedcba98u);
  EXPECT_EQ(output[1], 0u);
  EXPECT_EQ(output[2], 0u);
  EXPECT_EQ(output[kPrefix + kCount], 0u);
  for (size_t i = 0; i < kCount; ++i) {
    const uint32_t expected =
        17 * i + first_bias + (i < 3 * 5 * 2 ? 17 * i + second_bias : 0);
    EXPECT_EQ(output[kPrefix + i], expected) << "workgroup " << i;
  }
}

TEST_F(HalDispatchTest, NarrowParameterAndZeroWork) {
  ASSERT_NO_FATAL_FAILURE(LoadImage());
  iree_hal_executable_function_t function;
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable_, IREE_SV("fill_byte"), &function));
  iree_hal_executable_function_info_t info;
  IREE_ASSERT_OK(
      iree_hal_executable_function_info(executable_, function, &info));
  EXPECT_EQ(info.constant_byte_length, 1u);

  std::array<uint8_t, 8> output;
  output.fill(0xc8);
  iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device_), buffer_params, output.size(),
      &buffer_));
  uint64_t uploaded = 1, filled = 2, skipped = 3, downloaded = 4;
  IREE_ASSERT_OK(iree_hal_queue_upload(
      queue_, iree_hal_semaphore_list_empty(), Completion(&uploaded),
      output.data(), buffer_, 0, output.size(), /*barriers=*/NULL));
  const iree_hal_buffer_ref_t binding = iree_hal_make_buffer_ref(buffer_, 1, 5);
  const iree_hal_buffer_ref_list_t bindings = {1, &binding};
  const uint8_t value = 0xef, unused_value = 0x13;
  iree_status_t status = iree_hal_queue_dispatch(
      queue_, Completion(&uploaded), Completion(&filled), executable_, function,
      iree_hal_make_static_dispatch_config(5, 1, 1),
      iree_make_const_byte_span(&value, sizeof(value)), bindings,
      /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE);
  uint64_t completion = uploaded;
  if (iree_status_is_ok(status)) {
    completion = filled;
    status = iree_hal_queue_dispatch(
        queue_, Completion(&filled), Completion(&skipped), executable_,
        function, iree_hal_make_static_dispatch_config(0, 1, 1),
        iree_make_const_byte_span(&unused_value, sizeof(unused_value)),
        bindings, /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    completion = skipped;
    status = iree_hal_queue_download(
        queue_, Completion(&skipped), Completion(&downloaded), buffer_, 0,
        output.data(), output.size(), /*barriers=*/NULL);
  }
  if (iree_status_is_ok(status)) {
    completion = downloaded;
  }
  status = iree_status_join(
      status,
      iree_hal_semaphore_wait(semaphore_, completion, iree_infinite_timeout(),
                              IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(status);
  EXPECT_THAT(output, ::testing::ElementsAre(0xc8, 0xef, 0xef, 0xef, 0xef, 0xef,
                                             0xc8, 0xc8));
}

}  // namespace
