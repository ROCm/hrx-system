// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "iree/hal/cts/util/registry.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amd/driver.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/add_i32.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/add_i32_npu4.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32_npu4.h"
#include "iree/hal/drivers/amd/xdna/testing/allocation_tracker.h"
#include "iree/hal/slab_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::cts {
namespace {

using amd::xdna::testing::HostAllocationCounters;
using amd::xdna::testing::SnapshotHostAllocations;
using amd::xdna::testing::TrackingAllocator;

// Counts only the HAL device allocator, independently of test/oracle storage.
HostAllocationCounters device_allocations;

iree_status_t CreateDevice(const iree_hal_device_create_params_t* params,
                           iree_hal_driver_t** out_driver,
                           iree_hal_device_t** out_device) {
  iree_hal_driver_t* driver = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_amd_driver_create(
      AMDF_NATIVE_LIFETIME_PROCESS, iree_allocator_system(), &driver));
  iree_host_size_t count = 0;
  iree_hal_device_info_t* infos = nullptr;
  iree_status_t status = iree_hal_driver_query_available_devices(
      driver, IREE_SV("xdna"), iree_allocator_system(), &count, &infos);
  iree_allocator_free(iree_allocator_system(), infos);
  if (iree_status_is_ok(status) && !count) {
    status =
        iree_make_status(IREE_STATUS_UNAVAILABLE, "no native XDNA endpoint");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_create_device_by_uri(
        driver, IREE_SV("xdna://0"), params,
        TrackingAllocator(&device_allocations), out_device);
  }
  if (iree_status_is_ok(status)) {
    *out_driver = driver;
  } else {
    iree_hal_driver_release(driver);
  }
  return status;
}

class NativeReplayStress : public CtsTestBase<> {
 protected:
  void Run(bool recorded) {
    auto* queue = QueueForCommandCategories(IREE_HAL_COMMAND_CATEGORY_DISPATCH);
    ASSERT_NE(queue, nullptr);
    const auto target = SelectExecutableTarget(IREE_SV("xdna"), {});
    ASSERT_EQ(target.outcome,
              IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
    const bool halo = iree_string_view_equal(
        target.target->target_key, IREE_SV("amd.xdna.strix_halo.17f0_11"));
    const iree_file_toc_t* images[] = {
        halo ? iree_hal_amd_xdna_test_mul_i32_create()
             : iree_hal_amd_xdna_test_mul_i32_npu4_create(),
        halo ? iree_hal_amd_xdna_test_add_i32_create()
             : iree_hal_amd_xdna_test_add_i32_npu4_create(),
    };
    const char* names[] = {"mul_i32", "add_i32"};
    std::array<Ref<iree_hal_executable_t>, 2> executables;
    std::array<iree_hal_executable_function_t, 2> functions;
    for (size_t i = 0; i < executables.size(); ++i) {
      IREE_ASSERT_OK(LoadExecutable(
          iree_hal_queue_family(queue), target.target,
          IREE_HAL_EXECUTABLE_LOAD_FLAG_NONE,
          iree_make_const_byte_span(images[i][0].data, images[i][0].size),
          executables[i].out()));
      IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
          executables[i], iree_make_cstring_view(names[i]), &functions[i]));
    }

    const iree_hal_pool_family_access_t family = {
        .family = iree_hal_queue_family(queue),
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE,
        .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA,
    };
    const iree_hal_pool_scope_t scope = {
        .family_count = 1,
        .families = &family,
        .host = {.access = IREE_HAL_MEMORY_ACCESS_ALL,
                 .modes = IREE_HAL_MAPPING_MODE_SCOPED},
    };
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    Ref<iree_hal_pool_t> pool;
    IREE_ASSERT_OK(iree_hal_slab_pool_create(
        device_group_, scope, &options, iree_allocator_system(), pool.out()));
    // The largest indirect binding table exceeds the small metadata arena.
    // Every invocation owns a separate result, including across queue windows.
    constexpr size_t kSlots = 65;
    constexpr size_t kPayloadWords = 16;
    constexpr size_t kPayloadBytes = kPayloadWords * sizeof(uint32_t);
    constexpr size_t kBufferWords = 5 * kPayloadWords;
    constexpr size_t kBufferBytes = kBufferWords * sizeof(uint32_t);
    constexpr size_t kDepths[] = {1, 5, 6, kSlots};
    std::array<Ref<iree_hal_buffer_t>, kSlots * 3> buffers;
    std::vector<std::array<uint32_t, kBufferWords>> expected(kSlots * 3);
    for (auto& buffer : buffers) {
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          pool, iree_hal_make_pool_reservation_request(kBufferBytes, 64),
          iree_infinite_timeout(), buffer.out()));
    }
    std::array<std::array<Ref<iree_hal_command_buffer_t>, 2>, 4> programs;
    if (recorded) {
      for (size_t shape = 0; shape < programs.size(); ++shape) {
        for (size_t switching = 0; switching < 2; ++switching) {
          auto& program = programs[shape][switching];
          IREE_ASSERT_OK(iree_hal_command_buffer_create(
              iree_hal_queue_family(queue),
              IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
              IREE_HAL_COMMAND_CATEGORY_DISPATCH, kDepths[shape] * 3,
              program.out()));
          IREE_ASSERT_OK(iree_hal_command_buffer_begin(program));
          for (size_t i = 0; i < kDepths[shape]; ++i) {
            const size_t operation = switching ? i % 2 : 0;
            const iree_hal_buffer_ref_t bindings[] = {
                iree_hal_make_indirect_buffer_ref(i * 3, 0, kPayloadBytes),
                iree_hal_make_indirect_buffer_ref(i * 3 + 1, 0, kPayloadBytes),
                iree_hal_make_indirect_buffer_ref(i * 3 + 2, 0, kPayloadBytes),
            };
            IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
                program, executables[operation], functions[operation],
                iree_hal_make_static_dispatch_config(1, 1, 1), {},
                {IREE_ARRAYSIZE(bindings), bindings},
                IREE_HAL_DISPATCH_FLAG_NONE));
          }
          IREE_ASSERT_OK(iree_hal_command_buffer_end(program));
        }
      }
    }
    Ref<iree_hal_semaphore_t> gate, completion;
    for (auto* semaphore : {&gate, &completion}) {
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_NONE, semaphore->out()));
    }
    auto* gate_pointer = gate.get();
    auto* completion_pointer = completion.get();
    uint64_t completed = 0;
    // Two passes warm every shape, executable pattern, and deferred path.
    constexpr uint64_t kWarmBatches = 32;
    constexpr uint64_t kStressBatches = 128;
    for (uint64_t batch = 0; batch < kWarmBatches + kStressBatches; ++batch) {
      const size_t shape = batch % IREE_ARRAYSIZE(kDepths);
      const size_t depth = kDepths[shape];
      const bool switching = (batch / 4) % 2;
      const bool deferred = (batch / 8) % 2;
      SCOPED_TRACE(batch);
      std::array<iree_hal_buffer_binding_t, kSlots * 3> table;
      std::array<size_t, kSlots * 3> owners;
      for (size_t i = 0; i < depth; ++i) {
        const size_t slot = (i + batch) % kSlots;
        const size_t operation = switching ? i % 2 : 0;
        std::array<size_t, 3> offsets;
        for (size_t role = 0; role < 3; ++role) {
          const size_t owner = slot * 3 + role;
          owners[i * 3 + role] = owner;
          offsets[role] = (1 + (batch + i + role) % 3) * kPayloadWords;
          auto& words = expected[owner];
          words.fill(0xA5A50000u + static_cast<uint32_t>(batch));
          if (role < 2) {
            for (size_t word = 0; word < kPayloadWords; ++word) {
              words[offsets[role] + word] = static_cast<uint32_t>(
                  (batch * kSlots + i + 1) * 997 + role * 61 + word * 17);
            }
          }
          IREE_ASSERT_OK(iree_hal_buffer_map_write(buffers[owner], 0,
                                                   words.data(), kBufferBytes));
          table[i * 3 + role] = {
              buffers[owner], offsets[role] * sizeof(uint32_t), kPayloadBytes};
        }
        for (size_t word = 0; word < kPayloadWords; ++word) {
          const uint32_t lhs = expected[slot * 3][offsets[0] + word];
          const uint32_t rhs = expected[slot * 3 + 1][offsets[1] + word];
          expected[slot * 3 + 2][offsets[2] + word] =
              operation ? lhs + rhs : lhs * rhs;
        }
      }
      const auto before = SnapshotHostAllocations(device_allocations);
      uint64_t gate_value = batch + 1;
      iree_hal_semaphore_list_t waits =
          deferred ? iree_hal_semaphore_list_t{1, &gate_pointer, &gate_value}
                   : iree_hal_semaphore_list_t{};
      if (recorded) {
        ++completed;
        IREE_ASSERT_OK(iree_hal_queue_execute(
            queue, waits, {1, &completion_pointer, &completed},
            programs[shape][switching], {depth * 3, table.data()}, 0));
      } else {
        for (size_t i = 0; i < depth; ++i) {
          const size_t operation = switching ? i % 2 : 0;
          uint64_t previous = completed++;
          if (i) {
            waits = {1, &completion_pointer, &previous};
          }
          std::array<iree_hal_buffer_ref_t, 3> bindings;
          for (size_t role = 0; role < 3; ++role) {
            const auto& binding = table[i * 3 + role];
            bindings[role] = iree_hal_make_buffer_ref(
                binding.buffer, binding.offset, binding.length);
          }
          IREE_ASSERT_OK(iree_hal_queue_dispatch(
              queue, waits, {1, &completion_pointer, &completed},
              executables[operation], functions[operation],
              iree_hal_make_static_dispatch_config(1, 1, 1), {},
              {bindings.size(), bindings.data()}, nullptr,
              IREE_HAL_DISPATCH_FLAG_NONE));
        }
      }
      // Submission owns captured bindings while an unresolved producer holds
      // it.
      table.fill({});
      if (deferred) {
        uint64_t observed = 0;
        IREE_ASSERT_OK(iree_hal_semaphore_query(completion, &observed));
        EXPECT_LT(observed, completed);
        IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, gate_value, nullptr));
      }
      IREE_ASSERT_OK(iree_hal_semaphore_wait(completion, completed,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
      const auto after = SnapshotHostAllocations(device_allocations);
      if (batch >= kWarmBatches) {
        EXPECT_EQ(after.allocations, before.allocations);
        EXPECT_EQ(after.reallocations, before.reallocations);
      }
      for (size_t i = 0; i < depth * 3; ++i) {
        SCOPED_TRACE(i);
        std::array<uint32_t, kBufferWords> actual;
        IREE_ASSERT_OK(iree_hal_buffer_map_read(buffers[owners[i]], 0,
                                                actual.data(), kBufferBytes));
        ASSERT_EQ(actual, expected[owners[i]]);
      }
    }
  }
};

TEST_P(NativeReplayStress, DirectBindingsSurviveRetirementAndReuse) {
  Run(false);
}
TEST_P(NativeReplayStress, RecordedBindingsSurviveRetirementAndReuse) {
  Run(true);
}

CTS_REGISTER_TEST_SUITE(NativeReplayStress);
static bool registered =
    (CtsRegistry::RegisterBackend({"xdna", {"xdna", CreateDevice}, {}}), true);

}  // namespace
}  // namespace iree::hal::cts
