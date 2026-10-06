// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <vector>

#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/amd/xdna/driver.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32_npu4.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class XdnaNativeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        iree_hal_amd_xdna_driver_create(iree_allocator_system(), &driver_));
    iree_host_size_t count = 0;
    iree_hal_device_info_t* infos = nullptr;
    IREE_ASSERT_OK(iree_hal_driver_query_available_devices(
        driver_, iree_allocator_system(), &count, &infos));
    iree_allocator_free(iree_allocator_system(), infos);
    if (!count) {
      GTEST_SKIP() << "No native XDNA endpoint";
    }
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        1, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &pool_));
    auto params = iree_hal_device_create_params_default();
    params.proactor_pool = pool_;
    params.event_sink = iree_hal_device_event_sink_stderr();
    IREE_ASSERT_OK(iree_hal_driver_create_default_device(
        driver_, &params, iree_allocator_system(), &device_));
    // The device's selected entry keeps the runner alive independently.
    iree_async_proactor_pool_release(pool_);
    pool_ = nullptr;
    queue_ = iree_hal_device_queue(device_, 0, 0);
    ASSERT_NE(queue_, nullptr);
  }

  void TearDown() override {
    for (auto* semaphore : semaphores_) {
      iree_hal_semaphore_release(semaphore);
    }
    for (auto* buffer : buffers_) {
      iree_hal_buffer_release(buffer);
    }
    iree_hal_executable_release(executable_);
    iree_hal_device_release(device_);
    iree_async_proactor_pool_release(pool_);
    iree_hal_driver_release(driver_);
  }

  void LoadMultiply() {
    const auto* targets =
        iree_hal_device_spec_executables(iree_hal_device_spec(device_));
    ASSERT_EQ(targets->target_count, 1u);
    const bool halo = iree_string_view_equal(
        targets->targets[0].target_key, IREE_SV("amd.xdna.strix_halo.17f0_11"));
    const auto* toc = halo ? iree_hal_amd_xdna_test_mul_i32_create()
                           : iree_hal_amd_xdna_test_mul_i32_npu4_create();
    std::vector<uint8_t> bytes(toc[0].data, toc[0].data + toc[0].size);
    iree_hal_executable_load_params_t params;
    iree_hal_executable_load_params_initialize(&params);
    params.executable_data =
        iree_make_const_byte_span(bytes.data(), bytes.size());
    IREE_ASSERT_OK(
        iree_hal_executable_load(iree_hal_device_queue_family(device_, 0),
                                 &targets->targets[0], &params, &executable_));
    std::fill(bytes.begin(), bytes.end(), 0xCC);
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        executable_, IREE_SV("mul_i32"), &function_));
  }

  void MakeBuffer(iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_params_t params = {};
    params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    params.usage =
        IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_MAPPING;
    iree_hal_buffer_t* allocation = nullptr;
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device_), params, 3 * kBytes, &allocation));
    buffers_.push_back(allocation);
    const uint8_t guard = 0xA5;
    IREE_ASSERT_OK(
        iree_hal_buffer_map_fill(allocation, 0, 3 * kBytes, &guard, 1));
    IREE_ASSERT_OK(iree_hal_buffer_subspan(
        allocation, kBytes, kBytes, iree_allocator_system(), out_buffer));
    buffers_.push_back(*out_buffer);
  }

  void MakeSemaphore(iree_hal_semaphore_t** out_semaphore) {
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, out_semaphore));
    semaphores_.push_back(*out_semaphore);
  }

  void CheckOutput(iree_hal_buffer_t* output,
                   const std::array<uint32_t, 16>& lhs,
                   const std::array<uint32_t, 16>& rhs) {
    std::array<uint32_t, 48> result;
    IREE_ASSERT_OK(
        iree_hal_buffer_map_read(iree_hal_buffer_allocated_buffer(output), 0,
                                 result.data(), sizeof(result)));
    for (size_t i = 0; i < 16; ++i) {
      EXPECT_EQ(result[i], 0xA5A5A5A5u) << i;
      EXPECT_EQ(result[16 + i], lhs[i] * rhs[i]) << i;
      EXPECT_EQ(result[32 + i], 0xA5A5A5A5u) << i;
    }
  }

  // Canonical compiler fixture binding extent.
  static constexpr size_t kBytes = 16 * sizeof(uint32_t);
  // Driver owning the native discovery instance.
  iree_hal_driver_t* driver_ = nullptr;
  // Construction-time aggregate pool, released after device creation.
  iree_async_proactor_pool_t* pool_ = nullptr;
  // Device dominating all native children.
  iree_hal_device_t* device_ = nullptr;
  // Borrowed provisioned queue.
  iree_hal_queue_t* queue_ = nullptr;
  // Loaded real compiler image.
  iree_hal_executable_t* executable_ = nullptr;
  // Reflected function token.
  iree_hal_executable_function_t function_ =
      iree_hal_executable_function_invalid();
  // Test-owned buffers, released before device teardown.
  std::vector<iree_hal_buffer_t*> buffers_;
  // Test-owned semaphores, released before device teardown.
  std::vector<iree_hal_semaphore_t*> semaphores_;
};

TEST_F(XdnaNativeTest, ConsumerBeforeProducerCapturesArguments) {
  ASSERT_NO_FATAL_FAILURE(LoadMultiply());
  iree_hal_buffer_t *lhs_buffer = nullptr, *rhs_buffer = nullptr,
                    *output = nullptr;
  ASSERT_NO_FATAL_FAILURE(MakeBuffer(&lhs_buffer));
  ASSERT_NO_FATAL_FAILURE(MakeBuffer(&rhs_buffer));
  ASSERT_NO_FATAL_FAILURE(MakeBuffer(&output));
  iree_hal_semaphore_t *ready = nullptr, *done = nullptr;
  ASSERT_NO_FATAL_FAILURE(MakeSemaphore(&ready));
  ASSERT_NO_FATAL_FAILURE(MakeSemaphore(&done));
  uint64_t value = 1;
  iree_hal_buffer_ref_t bindings[] = {
      iree_hal_make_buffer_ref(lhs_buffer, 0, kBytes),
      iree_hal_make_buffer_ref(rhs_buffer, 0, kBytes),
      iree_hal_make_buffer_ref(output, 0, kBytes),
  };
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      queue_, {1, &ready, &value}, {1, &done, &value}, executable_, function_,
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), {IREE_ARRAYSIZE(bindings), bindings},
      IREE_HAL_DISPATCH_FLAG_NONE));
  std::fill(std::begin(bindings), std::end(bindings), iree_hal_buffer_ref_t{});
  iree_hal_executable_release(executable_);
  executable_ = nullptr;

  std::array<uint32_t, 16> lhs, rhs;
  for (size_t i = 0; i < lhs.size(); ++i) {
    lhs[i] = static_cast<uint32_t>(i) - 7;
    rhs[i] = 0x12345678u + static_cast<uint32_t>(i);
  }
  const auto expected_lhs = lhs, expected_rhs = rhs;
  iree_hal_transfer_operation_t uploads[2] = {};
  uploads[0].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE;
  uploads[0].update.source_buffer = lhs.data();
  uploads[0].update.target_buffer = lhs_buffer;
  uploads[0].update.length = kBytes;
  uploads[1] = uploads[0];
  uploads[1].update.source_buffer = rhs.data();
  uploads[1].update.target_buffer = rhs_buffer;
  IREE_ASSERT_OK(
      iree_hal_queue_transfer(queue_, {}, {1, &ready, &value}, 2, uploads));
  lhs.fill(0);
  rhs.fill(0);
  IREE_ASSERT_OK(iree_hal_semaphore_wait(done, 1, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  CheckOutput(output, expected_lhs, expected_rhs);
}

TEST_F(XdnaNativeTest, RebindsExecutableAfterCheckedRetirement) {
  ASSERT_NO_FATAL_FAILURE(LoadMultiply());
  iree_hal_semaphore_t* done = nullptr;
  ASSERT_NO_FATAL_FAILURE(MakeSemaphore(&done));
  for (uint64_t iteration = 1; iteration <= 4; ++iteration) {
    iree_hal_buffer_t *lhs_buffer = nullptr, *rhs_buffer = nullptr,
                      *output = nullptr;
    ASSERT_NO_FATAL_FAILURE(MakeBuffer(&lhs_buffer));
    ASSERT_NO_FATAL_FAILURE(MakeBuffer(&rhs_buffer));
    ASSERT_NO_FATAL_FAILURE(MakeBuffer(&output));
    std::array<uint32_t, 16> lhs, rhs;
    for (size_t i = 0; i < lhs.size(); ++i) {
      lhs[i] = static_cast<uint32_t>(i + iteration * 17);
      rhs[i] = 0xFFFFFF00u + static_cast<uint32_t>(i + iteration);
    }
    IREE_ASSERT_OK(
        iree_hal_buffer_map_write(lhs_buffer, 0, lhs.data(), kBytes));
    IREE_ASSERT_OK(
        iree_hal_buffer_map_write(rhs_buffer, 0, rhs.data(), kBytes));
    const iree_hal_buffer_ref_t bindings[] = {
        iree_hal_make_buffer_ref(lhs_buffer, 0, kBytes),
        iree_hal_make_buffer_ref(rhs_buffer, 0, kBytes),
        iree_hal_make_buffer_ref(output, 0, kBytes),
    };
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        queue_, {}, {1, &done, &iteration}, executable_, function_,
        iree_hal_make_static_dispatch_config(1, 1, 1),
        iree_const_byte_span_empty(), {IREE_ARRAYSIZE(bindings), bindings},
        IREE_HAL_DISPATCH_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        done, iteration, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    CheckOutput(output, lhs, rhs);
  }
}

TEST_F(XdnaNativeTest, TransfersCaptureAndBorrowAtTheirDeclaredBoundary) {
  iree_hal_buffer_t *source = nullptr, *target = nullptr;
  ASSERT_NO_FATAL_FAILURE(MakeBuffer(&source));
  ASSERT_NO_FATAL_FAILURE(MakeBuffer(&target));
  iree_hal_semaphore_t *ready = nullptr, *done = nullptr;
  ASSERT_NO_FATAL_FAILURE(MakeSemaphore(&ready));
  ASSERT_NO_FATAL_FAILURE(MakeSemaphore(&done));
  uint64_t value = 1;
  uint32_t pattern = 0x12345678;
  std::array<uint32_t, 16> upload = {}, download = {};
  iree_hal_transfer_operation_t operations[4] = {};
  operations[0].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
  operations[0].upload.source = upload.data();
  operations[0].upload.target_buffer = source;
  operations[0].upload.length = kBytes;
  operations[1].type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY;
  operations[1].copy.source_buffer = source;
  operations[1].copy.target_buffer = target;
  operations[1].copy.length = kBytes;
  operations[2].type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
  operations[2].fill.target_buffer = target;
  operations[2].fill.length = sizeof(pattern);
  operations[2].fill.pattern = &pattern;
  operations[2].fill.pattern_length = sizeof(pattern);
  operations[3].type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
  operations[3].download.source_buffer = target;
  operations[3].download.target = download.data();
  operations[3].download.length = kBytes;
  IREE_ASSERT_OK(iree_hal_queue_transfer(queue_, {1, &ready, &value},
                                         {1, &done, &value}, 4, operations));
  pattern = 0;
  upload.fill(0xABCDEF01);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(ready, value, nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(done, value, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_EQ(download[0], 0x12345678u);
  for (size_t i = 1; i < download.size(); ++i) {
    EXPECT_EQ(download[i], 0xABCDEF01u);
  }
}

TEST_F(XdnaNativeTest, EmptyTransfersIgnoreUnusedPayloads) {
  iree_hal_semaphore_t* done = nullptr;
  ASSERT_NO_FATAL_FAILURE(MakeSemaphore(&done));
  uint64_t value = 1;
  const iree_hal_transfer_operation_type_t types[] = {
      IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
      IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
      IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
      IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
      IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
  };
  iree_hal_transfer_operation_t operations[5];
  for (size_t i = 0; i < 5; ++i) {
    memset(&operations[i], 0xA5, sizeof(operations[i]));
    operations[i].type = types[i];
  }
  operations[0].fill.length = 0;
  operations[1].update.length = 0;
  operations[2].copy.length = 0;
  operations[3].upload.length = 0;
  operations[4].download.length = 0;
  IREE_ASSERT_OK(
      iree_hal_queue_transfer(queue_, {}, {1, &done, &value}, 5, operations));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(done, value, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
}

TEST_F(XdnaNativeTest, FailedDependencyFailsOnlyItsConsumer) {
  iree_hal_semaphore_t *ready = nullptr, *failed = nullptr, *done = nullptr;
  ASSERT_NO_FATAL_FAILURE(MakeSemaphore(&ready));
  ASSERT_NO_FATAL_FAILURE(MakeSemaphore(&failed));
  ASSERT_NO_FATAL_FAILURE(MakeSemaphore(&done));
  uint64_t value = 1;
  IREE_ASSERT_OK(iree_hal_queue_barrier(queue_, {1, &ready, &value},
                                        {1, &failed, &value}, 0));
  iree_hal_semaphore_fail(ready, iree_status_from_code(IREE_STATUS_CANCELLED));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      iree_hal_semaphore_wait(failed, value, iree_infinite_timeout(),
                              IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_queue_barrier(queue_, {}, {1, &done, &value}, 0));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(done, value, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
}

TEST_F(XdnaNativeTest, AllocationPublishesItsActualMemoryProperties) {
  iree_hal_buffer_params_t params = {};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device_), params, kBytes, &buffer));
  buffers_.push_back(buffer);
  EXPECT_TRUE(iree_all_bits_set(
      iree_hal_buffer_memory_type(buffer),
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL));
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, kBytes, &mapping));
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  params.usage |= IREE_HAL_BUFFER_USAGE_SHARING_EXPORT;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(device_),
                                         params, kBytes, &buffer));
  EXPECT_EQ(buffer, nullptr);
}

}  // namespace
