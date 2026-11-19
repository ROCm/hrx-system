// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/driver/amd/gpu/cts/test_device.h"
#include "iree/hal/slab_pool.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu::cts {
namespace {

using iree::hal::cts::Ref;

class BlitTest
    : public GpuNativeTest,
      public ::testing::WithParamInterface<amdf_queue_command_type_t> {
 protected:
  amdf_queue_command_type_t command_type() const override {
    return GetParam() == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA
               ? AMDF_QUEUE_COMMAND_TYPE_UNKNOWN
               : GetParam();
  }

  void Create() {
    GpuNativeTest::Create();
    if (!HasFatalFailure() && !IsSkipped() &&
        GetParam() == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA) {
      AcquireTransferQueues();
    }
  }
};

TEST_P(BlitTest, ExactRangesAcrossVectorGroupAndGridBoundaries) {
  Create();
  if (HasFatalFailure() || IsSkipped()) {
    return;
  }
  const auto* family = iree_hal_queue_family(queues_[0]);
  const iree_hal_pool_family_access_t access = {
      .family = family,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
      .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
  };
  const iree_hal_pool_scope_t scope = {.family_count = 1, .families = &access};
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      group_, scope, &options, iree_allocator_system(), pool.out()));
  constexpr size_t kGridBytes = 256 * 64 * 16;
  constexpr size_t kCapacity = 3 * (kGridBytes + 64);
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(kCapacity, 16),
      iree_infinite_timeout(), buffer.out()));

  // Whole-range bindings exercise each protocol for every size, including empty
  // ranges and extents that require a second grid iteration. The same three
  // resident recordings serve every invocation without allocation churn.
  const std::array<uint8_t, 4> pattern = {0x17, 0x53, 0xB9, 0xED};
  const iree_hal_memory_barrier_t memory = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
  };
  const iree_hal_barrier_t dependency = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .memory_barrier_count = 1,
      .memory_barriers = &memory,
  };
  const iree_hal_barrier_t release = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  std::array<Ref<iree_hal_command_buffer_t>, 3> recordings;
  for (size_t i = 0; i < recordings.size(); ++i) {
    auto& recording = recordings[i];
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        family, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        IREE_HAL_COMMAND_CATEGORY_TRANSFER, 3, recording.out()));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
    const auto source =
        iree_hal_make_indirect_buffer_ref(0, 0, IREE_HAL_WHOLE_BUFFER);
    const auto target =
        iree_hal_make_indirect_buffer_ref(1, 0, IREE_HAL_WHOLE_BUFFER);
    const auto filled =
        iree_hal_make_indirect_buffer_ref(2, 0, IREE_HAL_WHOLE_BUFFER);
    IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer_linear(recording, source,
                                                              target, 0));
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &dependency));
    IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer_linear(
        recording, source, pattern.data(), size_t{1} << i, 0));
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &dependency));
    IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer_linear(recording, source,
                                                              filled, 0));
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &release));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
  }

  std::vector<uint8_t> initial(kCapacity);
  std::vector<uint8_t> expected(kCapacity);
  std::vector<uint8_t> actual(kCapacity);
  uint64_t submitted = 0;
  const auto join = [&submitted](iree_hal_semaphore_t* semaphore) {
    // Even an assertion after admission must join borrowed host storage before
    // releasing it, including an upload whose dependent execute was rejected.
    if (submitted) {
      IREE_EXPECT_OK(iree_hal_semaphore_wait(semaphore, submitted,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
    }
    iree_hal_semaphore_release(semaphore);
  };
  iree_hal_semaphore_t* semaphore = nullptr;
  CreateSemaphore(&semaphore);
  std::unique_ptr<iree_hal_semaphore_t, decltype(join)> timeline(semaphore,
                                                                 join);
  ASSERT_FALSE(HasFatalFailure());

  // Every partial-vector size and byte alignment occurs below and above a
  // vector, a workgroup, and the fixed grid's stride. Source and destination
  // offsets vary independently; fills retain their public pattern alignment.
  for (size_t boundary : {size_t{0}, size_t{16}, size_t{1008}, size_t{1024},
                          kGridBytes - 16, kGridBytes}) {
    for (size_t tail = 0; tail < 16; ++tail) {
      const size_t length = boundary + tail;
      const size_t stride = (length + 47) & ~size_t{15};
      const size_t byte_length = 3 * stride;
      for (size_t alignment = 0; alignment < 16; ++alignment) {
        const size_t source_offset = 16 + alignment;
        const size_t target_offset = stride + 16 + ((alignment * 7 + 3) & 15);
        const size_t filled_offset = 2 * stride + 16 + ((alignment * 5) & 15);
        const size_t pattern_index = ((length | source_offset) & 1)   ? 0
                                     : ((length | source_offset) & 2) ? 1
                                                                      : 2;
        const size_t pattern_length = size_t{1} << pattern_index;
        SCOPED_TRACE(::testing::Message()
                     << "length=" << length << " alignment=" << alignment
                     << " pattern_length=" << pattern_length);
        std::fill_n(initial.begin(), byte_length, 0xCC);
        for (size_t i = 0; i < length; ++i) {
          initial[source_offset + i] =
              static_cast<uint8_t>((i * 29) ^ (i >> 8) ^ alignment);
        }
        std::copy_n(initial.begin(), byte_length, expected.begin());
        for (size_t i = 0; i < length; ++i) {
          expected[target_offset + i] = initial[source_offset + i];
          expected[source_offset + i] = pattern[i % pattern_length];
          expected[filled_offset + i] = pattern[i % pattern_length];
        }
        std::fill_n(actual.begin(), byte_length, 0xDD);
        uint64_t next = submitted + 1;
        IREE_ASSERT_OK(iree_hal_queue_upload_linear(
            queues_[0], {}, {1, &semaphore, &next}, initial.data(), buffer, 0,
            byte_length, nullptr));
        submitted = next++;
        const iree_hal_buffer_binding_t bindings[] = {
            {buffer, source_offset, length},
            {buffer, target_offset, length},
            {buffer, filled_offset, length},
        };
        IREE_ASSERT_OK(iree_hal_queue_execute(
            queues_[0], {1, &semaphore, &submitted}, {1, &semaphore, &next},
            recordings[pattern_index], {IREE_ARRAYSIZE(bindings), bindings},
            IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
        submitted = next++;
        IREE_ASSERT_OK(iree_hal_queue_download_linear(
            queues_[1], {1, &semaphore, &submitted}, {1, &semaphore, &next},
            buffer, 0, actual.data(), byte_length, nullptr));
        submitted = next;
        IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, submitted,
                                               iree_infinite_timeout(),
                                               IREE_ASYNC_WAIT_FLAG_NONE));
        const auto mismatch = std::mismatch(
            expected.begin(), expected.begin() + byte_length, actual.begin());
        ASSERT_EQ(static_cast<size_t>(mismatch.first - expected.begin()),
                  byte_length);
      }
    }
  }
}

TEST_P(BlitTest, DirectAndRecordedRectanglesPreserveBytesAndBindings) {
  Create();
  if (HasFatalFailure() || IsSkipped()) {
    return;
  }
  const auto* family = iree_hal_queue_family(queues_[0]);
  const iree_hal_pool_family_access_t access = {
      .family = family,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
      .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
  };
  const iree_hal_pool_scope_t scope = {.family_count = 1, .families = &access};
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      group_, scope, &options, iree_allocator_system(), pool.out()));
  const iree_hal_memory_barrier_t memory = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_READ,
  };
  const iree_hal_barrier_t dependency = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .memory_barrier_count = 1,
      .memory_barriers = &memory,
  };
  const iree_hal_barrier_t release = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  constexpr size_t kRows = 128, kSlices = 2, kWindow = 80 * 1024;
  constexpr size_t kCapacity = 3 * kWindow + 32;
  std::array<Ref<iree_hal_buffer_t>, 2> buffers;
  for (auto& buffer : buffers) {
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool, iree_hal_make_pool_reservation_request(kCapacity, 16),
        iree_infinite_timeout(), buffer.out()));
  }
  std::vector<uint8_t> initial(kCapacity, 0xCC);
  std::vector<uint8_t> expected(kCapacity);
  std::vector<uint8_t> actual(kCapacity);
  uint64_t submitted = 0;
  const auto join = [&submitted](iree_hal_semaphore_t* semaphore) {
    if (submitted) {
      IREE_EXPECT_OK(iree_hal_semaphore_wait(semaphore, submitted,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
    }
    iree_hal_semaphore_release(semaphore);
  };
  iree_hal_semaphore_t* semaphore = nullptr;
  CreateSemaphore(&semaphore);
  std::unique_ptr<iree_hal_semaphore_t, decltype(join)> timeline(semaphore,
                                                                 join);
  ASSERT_FALSE(HasFatalFailure());

  for (size_t width : {size_t{37}, size_t{256}}) {
    SCOPED_TRACE(width);
    auto strides = [&](size_t padding) {
      const size_t pitch =
          width == 256 ? (width + padding + 3) & ~size_t{3} : width + padding;
      return iree_hal_transfer_strides_t{
          .row_stride = pitch,
          .slice_stride = pitch * kRows + 24,
      };
    };
    iree_hal_transfer_region_t update = {
        .row_length = width,
        .row_count = kRows,
        .slice_count = kSlices,
        .source_strides = strides(11),
        .target_strides = strides(3),
    };
    iree_hal_transfer_region_t copy = {
        .row_length = width,
        .row_count = kRows,
        .slice_count = kSlices,
        .source_strides = strides(3),
        .target_strides = strides(17),
    };
    const auto direct_copy = copy;
    iree_hal_fill_region_t fill = {
        .row_length = width,
        .row_count = kRows,
        .slice_count = kSlices,
        .target_strides = strides(20),
    };
    const std::array<uint8_t, 4> pattern = {0x17, 0x53, 0xB9, 0xED};
    const size_t pattern_length = width == 256 ? 4 : 1;
    std::vector<uint8_t> source(kWindow, 0xDD);
    expected = initial;
    for (size_t z = 0; z < kSlices; ++z) {
      for (size_t y = 0; y < kRows; ++y) {
        for (size_t x = 0; x < width; ++x) {
          const uint8_t value = static_cast<uint8_t>(x * 29 + y * 7 + z * 13);
          source[5 + z * update.source_strides.slice_stride +
                 y * update.source_strides.row_stride + x] = value;
          expected[7 + z * copy.source_strides.slice_stride +
                   y * copy.source_strides.row_stride + x] = value;
          expected[kWindow + 19 + z * copy.target_strides.slice_stride +
                   y * copy.target_strides.row_stride + x] = value;
          expected[2 * kWindow + 4 + z * fill.target_strides.slice_stride +
                   y * fill.target_strides.row_stride + x] =
              pattern[x % pattern_length];
        }
      }
    }
    Ref<iree_hal_command_buffer_t> recording;
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        family, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        IREE_HAL_COMMAND_CATEGORY_TRANSFER, 3, recording.out()));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
    IREE_ASSERT_OK(iree_hal_command_buffer_update_buffer(
        recording, source.data(), 5,
        iree_hal_make_indirect_buffer_ref(0, 0, kWindow), &update, 0));
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &dependency));
    IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
        recording, iree_hal_make_indirect_buffer_ref(0, 0, kWindow),
        iree_hal_make_indirect_buffer_ref(1, 0, kWindow), &copy, 0));
    IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
        recording, iree_hal_make_indirect_buffer_ref(2, 0, kWindow),
        pattern.data(), pattern_length, &fill, 0));
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &release));
    // Neither caller-owned bytes nor geometry remain live after recording.
    std::fill(source.begin(), source.end(), 0xEE);
    update = {};
    copy = {};
    fill = {};
    IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
    for (auto& buffer : buffers) {
      uint64_t next = submitted + 1;
      IREE_ASSERT_OK(iree_hal_queue_upload_linear(
          queues_[0], {}, {1, &semaphore, &next}, initial.data(), buffer, 0,
          kCapacity, nullptr));
      submitted = next++;
      const iree_hal_buffer_binding_t bindings[] = {
          {buffer, 7, kWindow},
          {buffer, kWindow + 19, kWindow},
          {buffer, 2 * kWindow + 4, kWindow},
      };
      IREE_ASSERT_OK(iree_hal_queue_execute(
          queues_[0], {1, &semaphore, &submitted}, {1, &semaphore, &next},
          recording, {IREE_ARRAYSIZE(bindings), bindings}, 0));
      submitted = next++;
      IREE_ASSERT_OK(iree_hal_queue_download_linear(
          queues_[1], {1, &semaphore, &submitted}, {1, &semaphore, &next},
          buffer, 0, actual.data(), kCapacity, nullptr));
      submitted = next;
      IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, submitted,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
      const auto mismatch =
          std::mismatch(expected.begin(), expected.end(), actual.begin());
      ASSERT_EQ(static_cast<size_t>(mismatch.first - expected.begin()),
                kCapacity);
      // Clear the copied window, then restore it with the direct queue API.
      // The host joins only the final download; both queue paths must preserve
      // the same bytes, padding, and cross-queue visibility.
      const uint8_t cleared = 0xCC;
      next = submitted + 1;
      IREE_ASSERT_OK(iree_hal_queue_fill_linear(
          queues_[0], {1, &semaphore, &submitted}, {1, &semaphore, &next},
          buffer, kWindow, kWindow, &cleared, sizeof(cleared), nullptr, 0));
      submitted = next++;
      IREE_ASSERT_OK(iree_hal_queue_copy(
          queues_[0], {1, &semaphore, &submitted}, {1, &semaphore, &next},
          buffer, 7, buffer, kWindow + 19, &direct_copy, nullptr, 0));
      submitted = next++;
      IREE_ASSERT_OK(iree_hal_queue_download_linear(
          queues_[1], {1, &semaphore, &submitted}, {1, &semaphore, &next},
          buffer, 0, actual.data(), kCapacity, nullptr));
      submitted = next;
      IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, submitted,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
      EXPECT_EQ(actual, expected);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Protocol, BlitTest,
    ::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                      AMDF_QUEUE_COMMAND_TYPE_GPU_AQL,
                      AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA),
    [](const ::testing::TestParamInfo<amdf_queue_command_type_t>& info) {
      return info.param == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA  ? "Sdma"
             : info.param == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL ? "Aql"
                                                             : "Pm4";
    });

}  // namespace
}  // namespace iree::hal::amd::gpu::cts
