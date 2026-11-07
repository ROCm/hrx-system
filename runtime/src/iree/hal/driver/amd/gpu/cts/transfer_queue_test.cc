// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/driver/amd/gpu/cts/test_device.h"
#include "iree/hal/driver/amd/gpu/cts/testdata.h"
#include "iree/hal/driver/amd/gpu/image/hip_images.h"
#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/testing/temp_file.h"

namespace {
using iree::hal::amd::gpu::cts::GpuNativeTest;
using iree::hal::cts::Ref;

class GpuTransferQueueTest : public GpuNativeTest,
                             public ::testing::WithParamInterface<bool> {
 protected:
  amdf_queue_command_type_t command_type() const override {
    return AMDF_QUEUE_COMMAND_TYPE_UNKNOWN;
  }

  void SetUp() override {
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    Create();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    const auto* specs =
        iree_hal_device_spec_queues(iree_hal_device_spec(device_));
    for (uint32_t i = 0; i < specs->family_count; ++i) {
      const auto roles = specs->families[i].role_flags;
      if ((roles & IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER) &&
          !(roles & IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH)) {
        transfer_family_ = iree_hal_device_queue_family(device_, i);
        break;
      }
    }
    if (!transfer_family_) {
      GTEST_SKIP() << "Native SDMA family is unavailable";
    }
    iree_hal_queue_params_t params;
    iree_hal_queue_params_initialize(&params);
    IREE_ASSERT_OK(
        iree_hal_queue_acquire(transfer_family_, &params, transfer_.out()));
    const std::array<iree_hal_pool_family_access_t, 2> access = {{
        {
            .family = iree_hal_queue_family(queues_[0]),
            .usage =
                IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
            .interfaces = UINT64_C(1)
                          << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
        },
        {
            .family = transfer_family_,
            .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
            .interfaces = UINT64_C(1)
                          << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
        },
    }};
    const iree_hal_pool_scope_t scope = {access.size(), access.data(), {}};
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    IREE_ASSERT_OK(iree_hal_slab_pool_create(
        group_, scope, &options, iree_allocator_system(), pool_.out()));
  }

  void TearDown() override {
    pool_.reset();
    transfer_.reset();
    if (path_.Exists()) {
      EXPECT_TRUE(path_.Remove());
    }
    GpuNativeTest::TearDown();
  }

  // Dynamic transfer owner, acquired only after public capability discovery.
  Ref<iree_hal_queue_t> transfer_;
  // Borrowed exact family identity for recording and resource qualification.
  const iree_hal_queue_family_t* transfer_family_ = nullptr;
  // Payload backing qualified for both native engines before allocation.
  Ref<iree_hal_pool_t> pool_;
  // Real asynchronous file backing; removed after accepted work retires.
  iree::testing::TempFilePath path_{"amd_transfer_queue"};
};

TEST_P(GpuTransferQueueTest,
       FileRecordingComputeAndWriteShareOneDependencyChain) {
  constexpr size_t kLength = 4 * 1024 * 1024;
  constexpr size_t kFileOffset = 4096;
  constexpr size_t kBufferOffset = 128;
  std::vector<uint32_t> contents((2 * kLength + 2 * kFileOffset) / 4);
  for (size_t i = 0; i < contents.size(); ++i) {
    contents[i] = 0x173921u + i * 17;
  }
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path_.path_view(),
      iree_make_const_byte_span(contents.data(), contents.size() * 4),
      iree_allocator_system()));
  iree_io_file_handle_t* handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
          IREE_IO_FILE_MODE_ASYNC | (GetParam() ? IREE_IO_FILE_MODE_DIRECT : 0),
      path_.path_view(), iree_allocator_system(), &handle));
  Ref<iree_hal_file_t> file;
  iree::Status import_status = iree_hal_file_import(
      device_, nullptr, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, handle, 0,
      file.out());
  iree_io_file_handle_release(handle);
  if (GetParam() && import_status.code() == iree::StatusCode::kUnavailable) {
    GTEST_SKIP() << iree::hal::cts::FormatSkipStatus(import_status);
  }
  IREE_ASSERT_OK(import_status);
  std::array<Ref<iree_hal_buffer_t>, 2> buffers;
  for (auto& buffer : buffers) {
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_,
        iree_hal_make_pool_reservation_request(kLength + kBufferOffset, 64),
        iree_infinite_timeout(), buffer.out()));
  }
  Ref<iree_hal_command_buffer_t> copy;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      transfer_family_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, 2, copy.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(copy));
  const iree_hal_barrier_t acquire = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
  };
  const iree_hal_barrier_t release = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(copy, &acquire));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      copy, iree_hal_make_indirect_buffer_ref(0, 0, kLength),
      iree_hal_make_indirect_buffer_ref(1, 0, kLength), 0));
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(copy, &release));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(copy));

  const auto* target =
      &iree_hal_device_spec_executables(iree_hal_device_spec(device_))
           ->targets[0];
  iree_hal_executable_load_params_t load;
  iree_hal_executable_load_params_initialize(&load);
  IREE_ASSERT_OK(iree::hal::amd::gpu::cts::LookupCodeObject(
      iree_hal_amd_gpu_hip_images_create(), iree_hal_amd_gpu_hip_images_size(),
      target->target_key, &load.executable_data));
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(iree_hal_executable_load(iree_hal_queue_family(queues_[0]),
                                          target, &load, executable.out()));
  iree_hal_executable_function_t function;
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable, IREE_SV("image_accumulate"), &function));
  Ref<iree_hal_command_buffer_t> compute;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      iree_hal_queue_family(queues_[0]), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, 1, compute.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(compute));
  auto compute_acquire = acquire;
  compute_acquire.target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH;
  auto compute_release = release;
  compute_release.source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH;
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(compute, &compute_acquire));
  auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  const uint32_t addend = 31;
  const auto reference = iree_hal_make_indirect_buffer_ref(0, 0, 4);
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      compute, executable, function, config,
      iree_make_const_byte_span(&addend, sizeof(addend)), {1, &reference}, 0));
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(compute, &compute_release));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(compute));

  Ref<iree_hal_semaphore_t> gate, done;
  CreateSemaphore(gate.out());
  CreateSemaphore(done.out());
  auto* gate_handle = gate.get();
  auto* done_handle = done.get();
  uint64_t one = 1, read = 1, copied = 2, computed = 3, written = 4;
  IREE_ASSERT_OK(iree_hal_queue_read(
      transfer_, {1, &gate_handle, &one}, {1, &done_handle, &read}, file,
      kFileOffset, buffers[0], kBufferOffset, kLength, nullptr, 0));
  // Growth happens with both kinds of resident recordings already registered
  // and with accepted file work still blocked on an input dependency.
  Ref<iree_hal_queue_t> extra_transfer, extra_compute;
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  IREE_ASSERT_OK(
      iree_hal_queue_acquire(transfer_family_, &params, extra_transfer.out()));
  IREE_ASSERT_OK(AcquireQueue(extra_compute.out()));
  std::array<iree_hal_buffer_binding_t, 2> bindings = {
      {{buffers[0], kBufferOffset, kLength},
       {buffers[1], kBufferOffset, kLength}}};
  IREE_ASSERT_OK(iree_hal_queue_execute(
      extra_transfer, {1, &done_handle, &read}, {1, &done_handle, &copied},
      copy, {bindings.size(), bindings.data()}, 0));
  IREE_ASSERT_OK(iree_hal_queue_execute(
      extra_compute, {1, &done_handle, &copied}, {1, &done_handle, &computed},
      compute, {1, &bindings[1]}, 0));
  IREE_ASSERT_OK(iree_hal_queue_write(
      transfer_, {1, &done_handle, &computed}, {1, &done_handle, &written},
      buffers[1], kBufferOffset, file, kFileOffset + kLength, kLength, nullptr,
      0));
  bindings.fill({});
  copy.reset();
  compute.reset();
  executable.reset();
  file.reset();
  for (auto& buffer : buffers) {
    buffer.reset();
  }
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, one, nullptr));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(done, written, iree_infinite_timeout(), 0));

  // The external file is the final output, including preservation of guards.
  std::memcpy(contents.data() + (kFileOffset + kLength) / 4,
              contents.data() + kFileOffset / 4, kLength);
  contents[(kFileOffset + kLength) / 4] += addend;
  iree_io_file_contents_t* actual = nullptr;
  IREE_ASSERT_OK(iree_io_file_contents_read(path_.path_view(),
                                            iree_allocator_system(), &actual));
  EXPECT_EQ(actual->buffer.data_length, contents.size() * 4);
  if (actual->buffer.data_length == contents.size() * 4) {
    EXPECT_EQ(
        std::memcmp(actual->buffer.data, contents.data(), contents.size() * 4),
        0);
  }
  iree_io_file_contents_free(actual);
}

INSTANTIATE_TEST_SUITE_P(Files, GpuTransferQueueTest, ::testing::Bool(),
                         [](const auto& info) {
                           return info.param ? "Direct" : "Buffered";
                         });

}  // namespace
