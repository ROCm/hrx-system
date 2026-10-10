// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/task/executable/elf/testdata/elementwise_mul_native.h"
#include "iree/hal/drivers/task/registration/driver_module.h"
#include "iree/hal/replay/execute.h"
#include "iree/hal/replay/file_reader.h"
#include "iree/hal/replay/recorder.h"
#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {

static iree_hal_device_group_t* CreateTaskDeviceGroup() {
  iree_async_proactor_pool_t* proactor_pool = nullptr;
  IREE_CHECK_OK(iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, iree_async_proactor_pool_options_default(),
      iree_allocator_system(), &proactor_pool));
  iree_hal_driver_registry_t* registry = nullptr;
  IREE_CHECK_OK(
      iree_hal_driver_registry_allocate(iree_allocator_system(), &registry));
  IREE_CHECK_OK(iree_hal_task_driver_module_register(registry));
  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;
  iree_hal_device_t* device = nullptr;
  IREE_CHECK_OK(iree_hal_create_device(registry, IREE_SV("task"),
                                       &create_params, iree_allocator_system(),
                                       &device));
  iree_hal_driver_registry_free(registry);
  iree_async_proactor_pool_release(proactor_pool);

  iree_async_frontier_tracker_t* frontier_tracker = nullptr;
  IREE_CHECK_OK(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &frontier_tracker));
  iree_hal_device_group_t* group = nullptr;
  IREE_CHECK_OK(iree_hal_device_group_create_from_device(
      device, frontier_tracker, iree_allocator_system(), &group));
  iree_async_frontier_tracker_release(frontier_tracker);
  iree_hal_device_release(device);
  return group;
}

class ReplayDispatchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    storage_.resize(1024 * 1024);
    iree_io_file_handle_t* file_handle = nullptr;
    IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(storage_.data(), storage_.size()),
        iree_io_file_handle_release_callback_null(), iree_allocator_system(),
        &file_handle));
    auto options = iree_hal_replay_recorder_options_default();
    options.external_file_validation =
        IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_VALIDATION_CONTENT_DIGEST;
    iree_status_t status = iree_hal_replay_recorder_create(
        file_handle, &options, iree_allocator_system(), &recorder_);
    iree_io_file_handle_release(file_handle);
    IREE_ASSERT_OK(status);
    source_group_ = CreateTaskDeviceGroup();
    IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
        recorder_, source_group_, iree_allocator_system(), &wrapped_group_));
    device_ = iree_hal_device_group_device_at(wrapped_group_, 0);
    family_ = iree_hal_device_queue_family(device_, 0);
    ASSERT_NE(nullptr, family_);
    const iree_hal_executable_target_selection_t selection = {
        .family = IREE_SV("cpu"),
    };
    auto result = iree_hal_device_spec_select_executable_target(
        iree_hal_device_spec(device_), &selection);
    ASSERT_EQ(IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED,
              result.outcome);
    target_ = result.target;
    ASSERT_EQ(1u, elementwise_mul_native_size());
    const auto* artifact = elementwise_mul_native_create();
    iree_hal_executable_load_params_initialize(&load_params_);
    load_params_.executable_data =
        iree_make_const_byte_span(artifact->data, artifact->size);
  }

  void ReleaseExecutionObjects() {
    iree_hal_command_buffer_release(command_buffer_);
    command_buffer_ = nullptr;
    iree_hal_executable_release(executable_);
    executable_ = nullptr;
    iree_hal_executable_release(base_executable_);
    base_executable_ = nullptr;
    for (auto*& buffer : buffers_) {
      iree_hal_buffer_release(buffer);
      buffer = nullptr;
    }
    iree_hal_file_release(file_);
    file_ = nullptr;
    iree_hal_semaphore_release(semaphore_);
    semaphore_ = nullptr;
    iree_hal_queue_release(queue_);
    queue_ = nullptr;
  }

  void TearDown() override {
    ReleaseExecutionObjects();
    iree_hal_device_group_release(replay_group_);
    iree_hal_device_group_release(wrapped_group_);
    iree_hal_device_group_release(source_group_);
    iree_hal_replay_recorder_release(recorder_);
  }

  void ExpectOutput(iree_string_view_t path) {
    const float expected[] = {100.0f, 400.0f, 900.0f, 1600.0f};
    iree_io_file_contents_t* contents = nullptr;
    IREE_ASSERT_OK(
        iree_io_file_contents_read(path, iree_allocator_system(), &contents));
    EXPECT_EQ(sizeof(expected), contents->const_buffer.data_length);
    if (contents->const_buffer.data_length == sizeof(expected)) {
      EXPECT_EQ(
          0, memcmp(expected, contents->const_buffer.data, sizeof(expected)));
    }
    iree_io_file_contents_free(contents);
  }

  // Capture storage remains live through replay execution.
  std::vector<uint8_t> storage_;
  // Recorder owning the capture stream.
  iree_hal_replay_recorder_t* recorder_ = nullptr;
  // Native devices used during capture.
  iree_hal_device_group_t* source_group_ = nullptr;
  // Recording wrappers owning the capture family identities.
  iree_hal_device_group_t* wrapped_group_ = nullptr;
  // Independent native devices used during replay.
  iree_hal_device_group_t* replay_group_ = nullptr;
  // Device borrowed from the recording group.
  iree_hal_device_t* device_ = nullptr;
  // Canonical family borrowed from the recording device.
  const iree_hal_queue_family_t* family_ = nullptr;
  // CPU target borrowed from the shared device spec.
  const iree_hal_executable_target_t* target_ = nullptr;
  // Native artifact backed by the embedded test library.
  iree_hal_executable_load_params_t load_params_ = {};
  // Dynamically acquired recording queue.
  iree_hal_queue_t* queue_ = nullptr;
  // Recording proxy for the native executable.
  iree_hal_executable_t* executable_ = nullptr;
  // Native executable used to check family incompatibility.
  iree_hal_executable_t* base_executable_ = nullptr;
  // Recorded dispatch command buffer.
  iree_hal_command_buffer_t* command_buffer_ = nullptr;
  // Input and output buffers retained until queue completion.
  iree_hal_buffer_t* buffers_[3] = {};
  // Imported file receiving the computed output.
  iree_hal_file_t* file_ = nullptr;
  // Explicit dispatch-to-write dependency and completion signal.
  iree_hal_semaphore_t* semaphore_ = nullptr;
};

TEST_F(ReplayDispatchTest, FamilyConstructorsReplayNativeOutput) {
#if IREE_FILE_IO_ENABLE
  iree_hal_device_t* base_device =
      iree_hal_device_group_device_at(source_group_, 0);
  const auto* base_family = iree_hal_device_queue_family(base_device, 0);
  EXPECT_EQ(device_, iree_hal_queue_family_device(family_));
  EXPECT_EQ(base_device, iree_hal_queue_family_device(base_family));
  EXPECT_NE(base_family, family_);
  EXPECT_EQ(iree_hal_queue_family_spec(base_family),
            iree_hal_queue_family_spec(family_));

  iree_hal_queue_params_t queue_params;
  iree_hal_queue_params_initialize(&queue_params);
  IREE_ASSERT_OK(iree_hal_queue_acquire(family_, &queue_params, &queue_));
  EXPECT_EQ(family_, iree_hal_queue_family(queue_));
  IREE_ASSERT_OK(
      iree_hal_executable_load(family_, target_, &load_params_, &executable_));
  EXPECT_EQ(family_, iree_hal_executable_queue_family(executable_));
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      family_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*binding_capacity=*/0,
      &command_buffer_));
  EXPECT_EQ(family_, iree_hal_command_buffer_queue_family(command_buffer_));

  const float initial_data[3][4] = {
      {1.0f, 2.0f, 3.0f, 4.0f},
      {100.0f, 200.0f, 300.0f, 400.0f},
      {0.0f, 0.0f, 0.0f, 0.0f},
  };
  const iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE |
               IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
               IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
  };
  iree_hal_buffer_ref_t binding_refs[3] = {};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(buffers_); ++i) {
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device_), buffer_params,
        sizeof(initial_data[i]), &buffers_[i]));
    IREE_ASSERT_OK(iree_hal_buffer_map_write(buffers_[i], 0, initial_data[i],
                                             sizeof(initial_data[i])));
    binding_refs[i] = {i, 0, buffers_[i], 0, sizeof(initial_data[i])};
  }
  iree_hal_executable_function_t function;
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable_, IREE_SV("elementwise_mul"), &function));
  auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  config.workgroup_size[0] = 1;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer_));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer_, executable_, function, config,
      iree_const_byte_span_empty(),
      {IREE_ARRAYSIZE(binding_refs), binding_refs},
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer_));

  iree::testing::TempFilePath output_path("iree_hal_replay_dispatch");
  const auto initial_output =
      iree_make_const_byte_span(initial_data[2], sizeof(initial_data[2]));
  IREE_ASSERT_OK(iree_io_file_contents_write(
      output_path.path_view(), initial_output, iree_allocator_system()));
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
          IREE_IO_FILE_MODE_RANDOM_ACCESS,
      output_path.path_view(), iree_allocator_system(), &file_handle));
  iree_status_t status = iree_hal_file_import(
      device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, file_handle,
      IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file_);
  iree_io_file_handle_release(file_handle);
  IREE_ASSERT_OK(status);
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore_));
  uint64_t dispatch_value = 1;
  uint64_t write_value = 2;
  const iree_hal_semaphore_list_t dispatch_complete = {1, &semaphore_,
                                                       &dispatch_value};
  const iree_hal_semaphore_list_t write_complete = {1, &semaphore_,
                                                    &write_value};
  IREE_ASSERT_OK(iree_hal_queue_execute(queue_, iree_hal_semaphore_list_empty(),
                                        dispatch_complete, command_buffer_,
                                        iree_hal_buffer_binding_table_empty(),
                                        IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
  // The capture records the output buffer reference, not the computed bytes.
  IREE_ASSERT_OK(iree_hal_queue_write(
      queue_, dispatch_complete, write_complete, buffers_[2], 0, file_, 0,
      sizeof(initial_data[2]), /*barriers=*/NULL, IREE_HAL_WRITE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      write_complete, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  ASSERT_NO_FATAL_FAILURE(ExpectOutput(output_path.path_view()));

  ReleaseExecutionObjects();
  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder_));
  iree_hal_device_group_release(wrapped_group_);
  wrapped_group_ = nullptr;
  iree_hal_device_group_release(source_group_);
  source_group_ = nullptr;

  iree_hal_replay_file_header_t header;
  iree_host_size_t offset = 0;
  IREE_ASSERT_OK(iree_hal_replay_file_parse_header(
      iree_make_const_byte_span(storage_.data(), storage_.size()), &header,
      &offset));
  ASSERT_LE(header.file_length, storage_.size());
  replay_group_ = CreateTaskDeviceGroup();
  const auto options = iree_hal_replay_execute_options_default();
  // Replay byte spans and the executable metadata within them can have any
  // alignment, regardless of the native artifact or target string lengths.
  constexpr size_t kAlignment =
      alignof(iree_hal_replay_executable_function_metadata_t);
  std::vector<uint8_t> replay_storage(header.file_length + kAlignment - 1);
  for (size_t offset = 0; offset < kAlignment; ++offset) {
    SCOPED_TRACE(offset);
    memcpy(replay_storage.data() + offset, storage_.data(), header.file_length);
    IREE_ASSERT_OK(iree_io_file_contents_write(
        output_path.path_view(), initial_output, iree_allocator_system()));
    IREE_ASSERT_OK(iree_hal_replay_execute_file(
        iree_make_const_byte_span(replay_storage.data() + offset,
                                  header.file_length),
        replay_group_, &options, iree_allocator_system()));
    ASSERT_NO_FATAL_FAILURE(ExpectOutput(output_path.path_view()));
  }
#else
  GTEST_SKIP() << "Native dispatch replay output requires file IO.";
#endif  // IREE_FILE_IO_ENABLE
}

TEST_F(ReplayDispatchTest, LoadFailurePreservesOutput) {
  IREE_ASSERT_OK(
      iree_hal_executable_load(family_, target_, &load_params_, &executable_));
  auto* output = executable_;
  const auto copied_target = *target_;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_executable_load(family_, &copied_target,
                                                 &load_params_, &output));
  EXPECT_EQ(executable_, output);

  auto invalid_params = load_params_;
  invalid_params.executable_data = iree_const_byte_span_empty();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_executable_load(family_, target_, &invalid_params, &output));
  EXPECT_EQ(executable_, output);
  invalid_params = load_params_;
  invalid_params.executable_data.data = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_executable_load(family_, target_, &invalid_params, &output));
  EXPECT_EQ(executable_, output);
  invalid_params = load_params_;
  invalid_params.constant_count = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_executable_load(family_, target_, &invalid_params, &output));
  EXPECT_EQ(executable_, output);

  // A truncated ELF header passes generic validation and fails in the loader.
  invalid_params = load_params_;
  invalid_params.executable_data.data_length = 4;
  IREE_EXPECT_NOT_OK(
      iree_hal_executable_load(family_, target_, &invalid_params, &output));
  EXPECT_EQ(executable_, output);
  iree_hal_executable_function_t function;
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable_, IREE_SV("elementwise_mul"), &function));
}

TEST_F(ReplayDispatchTest, SharedSpecDoesNotMakeFamiliesCompatible) {
  iree_hal_device_t* base_device =
      iree_hal_device_group_device_at(source_group_, 0);
  const auto* base_family = iree_hal_device_queue_family(base_device, 0);
  ASSERT_EQ(iree_hal_queue_family_spec(base_family),
            iree_hal_queue_family_spec(family_));
  IREE_ASSERT_OK(iree_hal_executable_load(base_family, target_, &load_params_,
                                          &base_executable_));
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      family_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*binding_capacity=*/0,
      &command_buffer_));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer_));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_command_buffer_dispatch(
                            command_buffer_, base_executable_,
                            iree_hal_executable_function_from_index(0),
                            iree_hal_make_static_dispatch_config(1, 1, 1),
                            iree_const_byte_span_empty(), {0, nullptr},
                            IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer_));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_queue_execute(iree_hal_device_queue(base_device, 0, 0),
                             iree_hal_semaphore_list_empty(),
                             iree_hal_semaphore_list_empty(), command_buffer_,
                             iree_hal_buffer_binding_table_empty(),
                             IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
}

}  // namespace
