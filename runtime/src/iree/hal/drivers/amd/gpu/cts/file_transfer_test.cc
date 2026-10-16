// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstring>
#include <vector>

#include "iree/async/semaphore.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/drivers/amd/gpu/image/hip_images.h"
#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {
using iree::hal::amd::gpu::cts::GpuNativeTest;
using iree::hal::cts::Ref;

class GpuNativeFileTest : public GpuNativeTest {
 protected:
  void SetUp() override {
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    const uint32_t gfx_major = context_->target_info.gfx_ip.major;
    RecordProperty("gfx_ip_major", gfx_major);
    RecordProperty("gfx_ip_minor", context_->target_info.gfx_ip.minor);
    RecordProperty("gfx_ip_stepping", context_->target_info.gfx_ip.stepping);
    Create();
    if (HasFatalFailure()) {
      return;
    }
    const iree_hal_pool_family_access_t family = {
        .family = iree_hal_queue_family(queues_[0]),
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE,
    };
    const iree_hal_pool_scope_t scope = {
        .family_count = 1,
        .families = &family,
        .host =
            {
                .access =
                    IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
                .modes = IREE_HAL_MAPPING_MODE_SCOPED,
            },
    };
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    IREE_ASSERT_OK(iree_hal_slab_pool_create(
        group_, scope, &options, iree_allocator_system(), pool_.out()));
    CreateSemaphore(timeline_.out());
    iree_hal_executable_load_params_t load;
    iree_hal_executable_load_params_initialize(&load);
    const auto* images = iree_hal_amd_gpu_hip_images_create();
    for (size_t i = 0; i < iree_hal_amd_gpu_hip_images_size(); ++i) {
      if (std::strstr(images[i].name,
                      gfx_major == 11 ? "gfx11_generic" : "gfx12_generic")) {
        load.executable_data =
            iree_make_const_byte_span(images[i].data, images[i].size);
      }
    }
    ASSERT_NE(load.executable_data.data, nullptr);
    const auto* targets =
        iree_hal_device_spec_executables(iree_hal_device_spec(device_));
    IREE_ASSERT_OK(iree_hal_executable_load(family.family, &targets->targets[0],
                                            &load, executable_.out()));
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        executable_, IREE_SV("image_accumulate"), &function_));
  }

  void TearDown() override {
    executable_.reset();
    timeline_.reset();
    pool_.reset();
    EXPECT_TRUE(path_.Remove());
    GpuNativeTest::TearDown();
  }

  void Open(const std::vector<uint32_t>& contents, iree_hal_file_t** out_file) {
    IREE_ASSERT_OK(iree_io_file_contents_write(
        path_.path_view(),
        iree_make_const_byte_span(contents.data(), contents.size() * 4),
        iree_allocator_system()));
    iree_io_file_handle_t* handle = nullptr;
    IREE_ASSERT_OK(iree_io_file_handle_open(
        IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
            IREE_IO_FILE_MODE_ASYNC | IREE_IO_FILE_MODE_SHARE_READ |
            IREE_IO_FILE_MODE_SHARE_WRITE,
        path_.path_view(), iree_allocator_system(), &handle));
    iree_status_t status = iree_hal_file_import(
        device_, nullptr, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, handle, 0,
        out_file);
    iree_io_file_handle_release(handle);
    IREE_ASSERT_OK(status);
  }

  void ExpectFile(const std::vector<uint32_t>& expected) {
    iree_io_file_contents_t* contents = nullptr;
    IREE_ASSERT_OK(iree_io_file_contents_read(
        path_.path_view(), iree_allocator_system(), &contents));
    EXPECT_EQ(contents->buffer.data_length, expected.size() * 4);
    if (contents->buffer.data_length == expected.size() * 4) {
      EXPECT_EQ(std::memcmp(contents->buffer.data, expected.data(),
                            expected.size() * 4),
                0);
    }
    iree_io_file_contents_free(contents);
  }

  void Accumulate(iree_hal_buffer_t* buffer, uint64_t before, uint64_t after,
                  uint32_t value) {
    auto* semaphore = timeline_.get();
    auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
    config.workgroup_size[0] = 32;
    config.workgroup_size[1] = config.workgroup_size[2] = 1;
    const auto binding = iree_hal_make_buffer_ref(buffer, kBufferOffset, 4);
    const iree_hal_queue_barriers_t barriers = {nullptr, &empty_};
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        queues_[0], {1, &semaphore, &before}, {1, &semaphore, &after},
        executable_, function_, config,
        iree_make_const_byte_span(&value, sizeof(value)), {1, &binding},
        &barriers, 0));
  }

  void Wait(uint64_t value) {
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(timeline_, value, iree_infinite_timeout(), 0));
  }

  uint64_t Publications() {
    iree_async_single_frontier_t frontier;
    const uint8_t count = iree_async_semaphore_query_frontier(
        reinterpret_cast<iree_async_semaphore_t*>(timeline_.get()),
        iree_async_single_frontier_as_frontier(&frontier), 1);
    EXPECT_EQ(count, 1);
    return count == 1 ? frontier.entries[0].epoch : 0;
  }

  // Logical range begins inside the allocation, independently of file offset.
  static constexpr size_t kBufferOffset = 128;
  // Several old staging chunks with a non-power-of-two final tail.
  static constexpr size_t kLength = 8 * 1024 * 1024 + 12;
  // Native file owns its path through independent output verification.
  iree::testing::TempFilePath path_{"iree_gpu_file_transfer"};
  // Explicit host/device backing contract, released after captured uses end.
  Ref<iree_hal_pool_t> pool_;
  // Public operation timeline, independent of private native handoff points.
  Ref<iree_hal_semaphore_t> timeline_;
  // Real shader that leaves dirty lines before file access.
  Ref<iree_hal_executable_t> executable_;
  // One-word read/modify/write entry point.
  iree_hal_executable_function_t function_ = {};
  // Explicitly omits optional caller cache work.
  const iree_hal_barrier_list_t empty_ = {};
  // Internal file handoffs must remain correct with both lists empty.
  const iree_hal_queue_barriers_t barriers_ = {&empty_, &empty_};
};

TEST_F(GpuNativeFileTest, BufferedReadWritePreservesInternalCacheHandoffs) {
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(kLength + 256, 64),
      iree_infinite_timeout(), buffer.out()));
  std::vector<uint32_t> expected(kLength / 4 + 32);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = uint32_t(i * 37 + 3);
  }
  Ref<iree_hal_file_t> file;
  ASSERT_NO_FATAL_FAILURE(Open(expected, file.out()));
  auto* semaphore = timeline_.get();
  uint64_t one = 1, two = 2, three = 3, four = 4, five = 5, six = 6;
  uint32_t sentinel = 0xCDCDCDCDu;
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], {}, {1, &semaphore, &one},
                                     buffer, 0, kLength + 256, &sentinel,
                                     sizeof(sentinel), nullptr, 0));
  ASSERT_NO_FATAL_FAILURE(Accumulate(buffer, one, two, 7));
  ASSERT_NO_FATAL_FAILURE(Wait(two));
  const uint64_t before_read = Publications();
  IREE_ASSERT_OK(iree_hal_queue_read(queues_[0], {1, &semaphore, &two},
                                     {1, &semaphore, &three}, file, 64, buffer,
                                     kBufferOffset, kLength, &barriers_, 0));
  ASSERT_NO_FATAL_FAILURE(Wait(three));
  EXPECT_EQ(Publications() - before_read, 2u);
  ASSERT_NO_FATAL_FAILURE(Accumulate(buffer, three, four, 19));
  ASSERT_NO_FATAL_FAILURE(Wait(four));
  const uint64_t before_write = Publications();
  expected[16] += 19;
  IREE_ASSERT_OK(iree_hal_queue_write(
      queues_[0], {1, &semaphore, &four}, {1, &semaphore, &five}, buffer,
      kBufferOffset, file, 64, kLength, &barriers_, 0));
  ASSERT_NO_FATAL_FAILURE(Wait(five));
  EXPECT_EQ(Publications() - before_write, 2u);
  ASSERT_NO_FATAL_FAILURE(ExpectFile(expected));

  // Check both allocation guards through the public GPU transfer path.
  std::vector<uint32_t> observed((kLength + 256) / 4);
  IREE_ASSERT_OK(iree_hal_queue_download(
      queues_[0], {1, &semaphore, &five}, {1, &semaphore, &six}, buffer, 0,
      observed.data(), observed.size() * 4, nullptr));
  ASSERT_NO_FATAL_FAILURE(Wait(six));
  for (size_t i = 0; i < observed.size(); ++i) {
    EXPECT_EQ(observed[i],
              i >= kBufferOffset / 4 && i < (kBufferOffset + kLength) / 4
                  ? expected[i - kBufferOffset / 4 + 16]
                  : sentinel)
        << i;
  }
}

TEST_F(GpuNativeFileTest, PendingWriteRetiresCapturedBufferBeforeSignal) {
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(kLength + 256, 64),
      iree_infinite_timeout(), buffer.out()));
  std::vector<uint32_t> expected(kLength / 4, 0);
  Ref<iree_hal_file_t> file;
  ASSERT_NO_FATAL_FAILURE(Open(expected, file.out()));
  Ref<iree_hal_semaphore_t> gate;
  CreateSemaphore(gate.out());
  auto* gate_handle = gate.get();
  auto* semaphore = timeline_.get();
  uint64_t one = 1, two = 2, three = 3;
  // Submit the file consumer before the shader producer. File and buffer
  // references disappear from the caller before either operation can run.
  IREE_ASSERT_OK(iree_hal_queue_write(
      queues_[0], {1, &semaphore, &two}, {1, &semaphore, &three}, buffer,
      kBufferOffset, file, 0, kLength, &barriers_, 0));
  uint32_t value = 0x12345678u;
  IREE_ASSERT_OK(iree_hal_queue_fill(
      queues_[0], {1, &gate_handle, &one}, {1, &semaphore, &one}, buffer, 0,
      kLength + 256, &value, sizeof(value), nullptr, 0));
  ASSERT_NO_FATAL_FAILURE(Accumulate(buffer, one, two, 7));
  buffer.reset();
  file.reset();
  executable_.reset();
  uint64_t observed = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(timeline_, &observed));
  EXPECT_EQ(observed, 0u);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, one, nullptr));
  ASSERT_NO_FATAL_FAILURE(Wait(three));
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  pool_.reset();
  std::fill(expected.begin(), expected.end(), value);
  expected[0] += 7;
  ASSERT_NO_FATAL_FAILURE(ExpectFile(expected));
}

TEST_F(GpuNativeFileTest, TruncatedReadEndsCapturedUsesBeforeFailure) {
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(kLength, 64),
      iree_infinite_timeout(), buffer.out()));
  Ref<iree_hal_file_t> file;
  std::vector<uint32_t> contents(kLength / 4, 0xABCDu);
  ASSERT_NO_FATAL_FAILURE(Open(contents, file.out()));
  Ref<iree_hal_semaphore_t> gate;
  CreateSemaphore(gate.out());
  auto* gate_handle = gate.get();
  auto* semaphore = timeline_.get();
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_read(queues_[0], {1, &gate_handle, &one},
                                     {1, &semaphore, &one}, file, 0, buffer, 0,
                                     kLength, &barriers_, 0));
  buffer.reset();
  file.reset();
  // File length can change after public range validation but before the
  // dependency permits I/O. The accepted read must retire its partial prefix.
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path_.path_view(), iree_make_const_byte_span(contents.data(), 4096),
      iree_allocator_system()));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, one, nullptr));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_semaphore_wait(timeline_, one, iree_infinite_timeout(), 0));
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  pool_.reset();
}

}  // namespace
