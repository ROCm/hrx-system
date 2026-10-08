// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/amd/xdna/driver.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/add_i32.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/add_i32_npu4.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32_npu4.h"
#include "iree/hal/slab_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class XdnaNativeTest : public ::testing::Test {
 protected:
  enum class BinaryOperation {
    kAdd,
    kMultiply,
  };

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
    iree_hal_pool_release(memory_pool_);
    iree_hal_executable_release(alternate_executable_);
    iree_hal_executable_release(executable_);
    iree_hal_device_group_release(device_group_);
    iree_hal_device_release(device_);
    iree_async_proactor_pool_release(pool_);
    iree_hal_driver_release(driver_);
  }

  void LoadProgram(const iree_file_toc_t* halo_toc,
                   const iree_file_toc_t* npu4_toc,
                   iree_string_view_t function_name,
                   iree_hal_executable_t** out_executable,
                   iree_hal_executable_function_t* out_function) {
    const auto* targets =
        iree_hal_device_spec_executables(iree_hal_device_spec(device_));
    ASSERT_EQ(targets->target_count, 1u);
    const bool halo = iree_string_view_equal(
        targets->targets[0].target_key, IREE_SV("amd.xdna.strix_halo.17f0_11"));
    const auto* toc = halo ? halo_toc : npu4_toc;
    std::vector<uint8_t> bytes(toc[0].data, toc[0].data + toc[0].size);
    iree_hal_executable_load_params_t params;
    iree_hal_executable_load_params_initialize(&params);
    params.executable_data =
        iree_make_const_byte_span(bytes.data(), bytes.size());
    IREE_ASSERT_OK(iree_hal_executable_load(
        iree_hal_device_queue_family(device_, 0), &targets->targets[0], &params,
        out_executable));
    std::fill(bytes.begin(), bytes.end(), 0xCC);
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        *out_executable, function_name, out_function));
  }

  void LoadMultiply() {
    LoadProgram(iree_hal_amd_xdna_test_mul_i32_create(),
                iree_hal_amd_xdna_test_mul_i32_npu4_create(),
                IREE_SV("mul_i32"), &executable_, &function_);
  }

  void LoadAdd() {
    LoadProgram(iree_hal_amd_xdna_test_add_i32_create(),
                iree_hal_amd_xdna_test_add_i32_npu4_create(),
                IREE_SV("add_i32"), &alternate_executable_,
                &alternate_function_);
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

  void CreateMemoryPool(iree_hal_buffer_usage_t usage) {
    iree_async_frontier_tracker_t* tracker = nullptr;
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &tracker));
    iree_status_t status = iree_hal_device_group_create_from_device(
        device_, tracker, iree_allocator_system(), &device_group_);
    iree_async_frontier_tracker_release(tracker);
    IREE_ASSERT_OK(status);

    const iree_hal_pool_family_access_t family_access = {
        /*.family=*/iree_hal_queue_family(queue_),
        /*.usage=*/usage,
        /*.interfaces=*/UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA,
    };
    const iree_hal_pool_scope_t scope = {
        /*.family_count=*/1,
        /*.families=*/&family_access,
    };
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    IREE_ASSERT_OK(iree_hal_slab_pool_create(device_group_, scope, &options,
                                             iree_allocator_system(),
                                             &memory_pool_));
  }

  void CheckOutput(iree_hal_buffer_t* output,
                   const std::array<uint32_t, 16>& lhs,
                   const std::array<uint32_t, 16>& rhs,
                   BinaryOperation operation = BinaryOperation::kMultiply) {
    std::array<uint32_t, 48> result;
    iree_hal_buffer_mapping_t mapping;
    IREE_ASSERT_OK(iree_hal_buffer_map_range(
        iree_hal_buffer_allocated_buffer(output), IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_BUFFER_MAP_FLAG_NONE, 0,
        sizeof(result), &mapping));
    iree_status_t status =
        iree_hal_buffer_mapping_invalidate_range(&mapping, 0, sizeof(result));
    if (iree_status_is_ok(status)) {
      memcpy(result.data(), mapping.contents.data, sizeof(result));
    }
    status = iree_status_join(status, iree_hal_buffer_unmap_range(&mapping));
    IREE_ASSERT_OK(status);
    for (size_t i = 0; i < 16; ++i) {
      EXPECT_EQ(result[i], 0xA5A5A5A5u) << i;
      const uint32_t expected = operation == BinaryOperation::kAdd
                                    ? lhs[i] + rhs[i]
                                    : lhs[i] * rhs[i];
      EXPECT_EQ(result[16 + i], expected) << i;
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
  // Owned sealed topology for family-scoped memory construction.
  iree_hal_device_group_t* device_group_ = nullptr;
  // Owned native slab pool selected from the sealed topology.
  iree_hal_pool_t* memory_pool_ = nullptr;
  // Loaded real compiler image.
  iree_hal_executable_t* executable_ = nullptr;
  // Reflected function token.
  iree_hal_executable_function_t function_ =
      iree_hal_executable_function_invalid();
  // Independently loaded image used to prove executable switching.
  iree_hal_executable_t* alternate_executable_ = nullptr;
  // Reflected function token from |alternate_executable_|.
  iree_hal_executable_function_t alternate_function_ =
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

TEST_F(XdnaNativeTest, SlabPoolAllocationExecutesThroughPreparedBindings) {
  ASSERT_NO_FATAL_FAILURE(CreateMemoryPool(IREE_HAL_BUFFER_USAGE_TRANSFER |
                                           IREE_HAL_BUFFER_USAGE_STORAGE));
  ASSERT_NO_FATAL_FAILURE(LoadMultiply());
  iree_hal_semaphore_t* progress = nullptr;
  ASSERT_NO_FATAL_FAILURE(MakeSemaphore(&progress));

  std::array<iree_hal_pool_reservation_request_t, 3> requests = {};
  for (size_t i = 0; i < requests.size(); ++i) {
    requests[i].params.min_alignment = 64;
    requests[i].params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
                               IREE_HAL_BUFFER_USAGE_STORAGE_READ;
    requests[i].allocation_size = 3 * kBytes;
  }
  requests[2].params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE_WRITE;
  std::array<iree_hal_buffer_t*, 3> roots = {};
  uint64_t progress_value = 1;
  IREE_ASSERT_OK(iree_hal_queue_alloca(
      queue_, {}, {1, &progress, &progress_value}, memory_pool_,
      requests.size(), requests.data(), roots.data()));
  for (iree_hal_buffer_t* root : roots) {
    ASSERT_NE(root, nullptr);
    buffers_.push_back(root);
  }

  std::array<iree_hal_buffer_t*, 3> bindings = {};
  for (size_t i = 0; i < bindings.size(); ++i) {
    IREE_ASSERT_OK(iree_hal_buffer_subspan(
        roots[i], kBytes, kBytes, iree_allocator_system(), &bindings[i]));
    buffers_.push_back(bindings[i]);
  }

  const uint8_t guard = 0xA5;
  std::array<iree_hal_transfer_operation_t, 3> fills = {};
  for (size_t i = 0; i < fills.size(); ++i) {
    fills[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
    fills[i].fill.target_buffer = roots[i];
    fills[i].fill.length = requests[i].allocation_size;
    fills[i].fill.pattern = &guard;
    fills[i].fill.pattern_length = sizeof(guard);
  }
  uint64_t wait_value = progress_value;
  ++progress_value;
  IREE_ASSERT_OK(iree_hal_queue_transfer(queue_, {1, &progress, &wait_value},
                                         {1, &progress, &progress_value},
                                         fills.size(), fills.data()));

  std::array<uint32_t, 16> lhs, rhs;
  for (size_t i = 0; i < lhs.size(); ++i) {
    lhs[i] = static_cast<uint32_t>(i * 97 + 13);
    rhs[i] = static_cast<uint32_t>(0xFFFFFF00u + i * 11);
  }
  std::array<iree_hal_transfer_operation_t, 2> updates = {};
  for (size_t i = 0; i < updates.size(); ++i) {
    updates[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE;
    updates[i].update.source_buffer = i == 0 ? lhs.data() : rhs.data();
    updates[i].update.target_buffer = bindings[i];
    updates[i].update.length = kBytes;
  }
  wait_value = progress_value;
  ++progress_value;
  IREE_ASSERT_OK(iree_hal_queue_transfer(queue_, {1, &progress, &wait_value},
                                         {1, &progress, &progress_value},
                                         updates.size(), updates.data()));

  std::array<iree_hal_buffer_ref_t, 3> binding_refs;
  for (size_t i = 0; i < binding_refs.size(); ++i) {
    binding_refs[i] = iree_hal_make_buffer_ref(bindings[i], 0, kBytes);
  }
  wait_value = progress_value;
  ++progress_value;
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      queue_, {1, &progress, &wait_value}, {1, &progress, &progress_value},
      executable_, function_, iree_hal_make_static_dispatch_config(1, 1, 1), {},
      {binding_refs.size(), binding_refs.data()}, IREE_HAL_DISPATCH_FLAG_NONE));

  std::array<uint32_t, 48> output;
  iree_hal_transfer_operation_t download = {};
  download.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
  download.download.source_buffer = roots[2];
  download.download.target = output.data();
  download.download.length = sizeof(output);
  wait_value = progress_value;
  ++progress_value;
  IREE_ASSERT_OK(iree_hal_queue_transfer(queue_, {1, &progress, &wait_value},
                                         {1, &progress, &progress_value}, 1,
                                         &download));

  wait_value = progress_value;
  ++progress_value;
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queue_, {1, &progress, &wait_value},
                                         {1, &progress, &progress_value},
                                         roots.size(), roots.data()));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(progress, progress_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  for (size_t i = 0; i < lhs.size(); ++i) {
    EXPECT_EQ(output[i], 0xA5A5A5A5u) << i;
    EXPECT_EQ(output[lhs.size() + i], lhs[i] * rhs[i]) << i;
    EXPECT_EQ(output[2 * lhs.size() + i], 0xA5A5A5A5u) << i;
  }
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

TEST_F(XdnaNativeTest, ReusesIndependentBindingsAcrossPendingBatches) {
  ASSERT_NO_FATAL_FAILURE(LoadMultiply());
  struct Invocation {
    // Distinct logical input/output storage for each independent invocation.
    std::array<iree_hal_buffer_t*, 3> buffers = {};
    // Completion edge independent of the other invocation's timeline.
    iree_hal_semaphore_t* done = nullptr;
    // Expected left input, updated only after this invocation completes.
    std::array<uint32_t, 16> lhs;
    // Expected right input, updated only after this invocation completes.
    std::array<uint32_t, 16> rhs;
  };
  std::array<Invocation, 2> invocations;
  for (auto& invocation : invocations) {
    for (auto& buffer : invocation.buffers) {
      ASSERT_NO_FATAL_FAILURE(MakeBuffer(&buffer));
    }
    ASSERT_NO_FATAL_FAILURE(MakeSemaphore(&invocation.done));
  }
  for (uint64_t iteration = 1; iteration <= 4; ++iteration) {
    for (size_t i = 0; i < invocations.size(); ++i) {
      auto& invocation = invocations[i];
      for (size_t j = 0; j < invocation.lhs.size(); ++j) {
        invocation.lhs[j] = static_cast<uint32_t>(iteration * 97 + i * 13 + j);
        invocation.rhs[j] = static_cast<uint32_t>(iteration * 31 + i * 17 - j);
      }
      IREE_ASSERT_OK(iree_hal_buffer_map_write(invocation.buffers[0], 0,
                                               invocation.lhs.data(), kBytes));
      IREE_ASSERT_OK(iree_hal_buffer_map_write(invocation.buffers[1], 0,
                                               invocation.rhs.data(), kBytes));
      const iree_hal_buffer_ref_t bindings[] = {
          iree_hal_make_buffer_ref(invocation.buffers[0], 0, kBytes),
          iree_hal_make_buffer_ref(invocation.buffers[1], 0, kBytes),
          iree_hal_make_buffer_ref(invocation.buffers[2], 0, kBytes),
      };
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          queue_, {}, {1, &invocation.done, &iteration}, executable_, function_,
          iree_hal_make_static_dispatch_config(1, 1, 1), {},
          {IREE_ARRAYSIZE(bindings), bindings}, IREE_HAL_DISPATCH_FLAG_NONE));
    }
    for (const auto& invocation : invocations) {
      IREE_ASSERT_OK(iree_hal_semaphore_wait(invocation.done, iteration,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
      CheckOutput(invocation.buffers[2], invocation.lhs, invocation.rhs);
    }
  }
}

TEST_F(XdnaNativeTest, AlternatesExecutablesAcrossAcceptedChains) {
  ASSERT_NO_FATAL_FAILURE(LoadMultiply());
  ASSERT_NO_FATAL_FAILURE(LoadAdd());
  iree_hal_buffer_t *lhs_buffer = nullptr, *rhs_buffer = nullptr,
                    *output = nullptr;
  ASSERT_NO_FATAL_FAILURE(MakeBuffer(&lhs_buffer));
  ASSERT_NO_FATAL_FAILURE(MakeBuffer(&rhs_buffer));
  ASSERT_NO_FATAL_FAILURE(MakeBuffer(&output));
  std::array<uint32_t, 16> lhs, rhs;
  for (size_t i = 0; i < lhs.size(); ++i) {
    lhs[i] = static_cast<uint32_t>(i * 19 + 7);
    rhs[i] = static_cast<uint32_t>(i * 11 + 3);
  }
  IREE_ASSERT_OK(iree_hal_buffer_map_write(lhs_buffer, 0, lhs.data(), kBytes));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(rhs_buffer, 0, rhs.data(), kBytes));
  const iree_hal_buffer_ref_t bindings[] = {
      iree_hal_make_buffer_ref(lhs_buffer, 0, kBytes),
      iree_hal_make_buffer_ref(rhs_buffer, 0, kBytes),
      iree_hal_make_buffer_ref(output, 0, kBytes),
  };
  iree_hal_semaphore_t* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(MakeSemaphore(&completion));

  uint64_t completion_value = 0;
  uint64_t dispatch_ordinal = 0;
  for (uint64_t batch_length : {UINT64_C(16), UINT64_C(15)}) {
    BinaryOperation final_operation = BinaryOperation::kMultiply;
    for (uint64_t i = 0; i < batch_length; ++i, ++dispatch_ordinal) {
      uint64_t wait_value = completion_value;
      uint64_t signal_value = ++completion_value;
      const iree_hal_semaphore_list_t waits =
          wait_value ? iree_hal_semaphore_list_t{1, &completion, &wait_value}
                     : iree_hal_semaphore_list_t{};
      const iree_hal_semaphore_list_t signals = {1, &completion, &signal_value};
      final_operation = dispatch_ordinal % 2 == 0 ? BinaryOperation::kAdd
                                                  : BinaryOperation::kMultiply;
      iree_hal_executable_t* executable =
          final_operation == BinaryOperation::kAdd ? alternate_executable_
                                                   : executable_;
      const iree_hal_executable_function_t function =
          final_operation == BinaryOperation::kAdd ? alternate_function_
                                                   : function_;
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          queue_, waits, signals, executable, function,
          iree_hal_make_static_dispatch_config(1, 1, 1), {},
          {IREE_ARRAYSIZE(bindings), bindings}, IREE_HAL_DISPATCH_FLAG_NONE));
    }
    IREE_ASSERT_OK(iree_hal_semaphore_wait(completion, completion_value,
                                           iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE));
    CheckOutput(output, lhs, rhs, final_operation);
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
  EXPECT_EQ(iree_hal_buffer_allocation_placement(buffer).device, device_);
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
