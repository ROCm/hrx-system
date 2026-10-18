// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Architecture-independent consumer of a compiled task test library. Backend
// tests supply their artifact through --executable_file.

#include <array>
#include <cstring>
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
    const iree_hal_pool_family_access_t family_access = {
        .family = iree_hal_queue_family(queue_),
        .usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
    };
    const iree_hal_pool_scope_t pool_scope = {
        .family_count = 1,
        .families = &family_access,
    };
    iree_hal_slab_pool_options_t pool_options;
    iree_hal_slab_pool_options_initialize(&pool_options);
    pool_options.trace_name = IREE_SV("loom.task_abi");
    IREE_ASSERT_OK(iree_hal_slab_pool_create(device_group_, pool_scope,
                                             &pool_options, allocator, &pool_));
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore_));
  }

  void TearDown() override {
    iree_hal_buffer_release(buffer_);
    iree_hal_executable_release(executable_);
    iree_hal_semaphore_release(semaphore_);
    iree_hal_pool_release(pool_);
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
  // Explicit storage shared by dispatch and transfer on the selected family.
  iree_hal_pool_t* pool_ = nullptr;
  // Loaded native image, independent of compiler and input-byte lifetimes.
  iree_hal_executable_t* executable_ = nullptr;
  // Output allocation including prefix and suffix sentinel words.
  iree_hal_buffer_t* buffer_ = nullptr;
};

TEST_F(HalDispatchTest, GridCallsAndIndependentImageLifetime) {
  ASSERT_NO_FATAL_FAILURE(LoadImage());
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
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_,
      iree_hal_make_pool_reservation_request(sizeof(output), alignof(uint32_t)),
      iree_infinite_timeout(), &buffer_));
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
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(output.size(), 1),
      iree_infinite_timeout(), &buffer_));
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

TEST_F(HalDispatchTest, ConstantsCrossPhysicalCarriers) {
  ASSERT_NO_FATAL_FAILURE(LoadImage());
  iree_hal_executable_function_t function;
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable_, IREE_SV("constant_carriers"), &function));
  iree_hal_executable_function_info_t info;
  IREE_ASSERT_OK(
      iree_hal_executable_function_info(executable_, function, &info));
  EXPECT_EQ(info.binding_count, 1u);
  EXPECT_EQ(info.constant_byte_length, 72u);
  ASSERT_EQ(info.parameter_count, 6u);
  std::array<iree_hal_executable_function_parameter_t, 6> parameters;
  IREE_ASSERT_OK(iree_hal_executable_function_parameters(
      executable_, function, parameters.size(), parameters.data()));
  EXPECT_EQ(parameters[0].type,
            IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING);
  EXPECT_EQ(parameters[0].offset, 0u);
  const std::array<uint16_t, 5> constant_offsets = {0, 8, 16, 32, 64};
  const std::array<uint8_t, 5> constant_sizes = {4, 8, 16, 32, 8};
  for (size_t i = 0; i < constant_offsets.size(); ++i) {
    EXPECT_EQ(parameters[i + 1].type,
              IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT);
    EXPECT_EQ(parameters[i + 1].offset, constant_offsets[i]);
    EXPECT_EQ(parameters[i + 1].size, constant_sizes[i]);
  }

  const float single = 1.5f;
  const double double_value = -2.25;
  const std::array<double, 2> xmm = {3.5, -4.75};
  const std::array<float, 8> ymm = {1.25f, -2.5f,  3.75f, -4.0f,
                                    5.5f,  -6.25f, 7.0f,  -8.5f};
  const std::array<uint8_t, 8> predicate = {1, 0, 1, 0, 0, 1, 1, 0};
  std::array<uint8_t, 72> constants = {};
  const auto write_constant = [&constants](size_t offset, const auto& value) {
    std::memcpy(constants.data() + offset, &value, sizeof(value));
  };
  write_constant(0, single);
  write_constant(8, double_value);
  write_constant(16, xmm);
  write_constant(32, ymm);
  write_constant(64, predicate);

  std::array<uint8_t, 96> output = {};
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(output.size(), 8),
      iree_infinite_timeout(), &buffer_));
  uint64_t uploaded = 1, dispatched = 2, downloaded = 3;
  IREE_ASSERT_OK(iree_hal_queue_upload(
      queue_, iree_hal_semaphore_list_empty(), Completion(&uploaded),
      output.data(), buffer_, 0, output.size(), /*barriers=*/NULL));
  const iree_hal_buffer_ref_t binding =
      iree_hal_make_buffer_ref(buffer_, 0, output.size());
  const iree_hal_buffer_ref_list_t bindings = {1, &binding};
  iree_status_t status = iree_hal_queue_dispatch(
      queue_, Completion(&uploaded), Completion(&dispatched), executable_,
      function, iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_make_const_byte_span(constants.data(), constants.size()), bindings,
      /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE);
  uint64_t completion = uploaded;
  if (iree_status_is_ok(status)) {
    completion = dispatched;
    status = iree_hal_queue_download(
        queue_, Completion(&dispatched), Completion(&downloaded), buffer_, 0,
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

  float observed_single;
  double observed_double;
  std::array<double, 2> observed_xmm;
  std::array<float, 8> observed_ymm;
  std::array<float, 8> observed_selected;
  std::memcpy(&observed_single, output.data(), sizeof(observed_single));
  std::memcpy(&observed_double, output.data() + 8, sizeof(observed_double));
  std::memcpy(&observed_xmm, output.data() + 16, sizeof(observed_xmm));
  std::memcpy(&observed_ymm, output.data() + 32, sizeof(observed_ymm));
  std::memcpy(&observed_selected, output.data() + 64,
              sizeof(observed_selected));
  EXPECT_FLOAT_EQ(observed_single, single);
  EXPECT_DOUBLE_EQ(observed_double, double_value);
  EXPECT_THAT(observed_xmm, ::testing::ElementsAreArray(xmm));
  EXPECT_THAT(observed_ymm, ::testing::ElementsAreArray(ymm));
  for (size_t i = 0; i < observed_selected.size(); ++i) {
    EXPECT_FLOAT_EQ(observed_selected[i], predicate[i] ? ymm[i] : 0.0f);
  }
}

}  // namespace
