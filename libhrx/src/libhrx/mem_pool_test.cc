// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <vector>

#include "hrx_runtime_cxx.h"
#include "iree/testing/gtest.h"

namespace {

testing::AssertionResult Check(hrx_status_t status) {
  auto result = hrx_status_is_ok(status) ? testing::AssertionSuccess()
                                         : testing::AssertionFailure();
  if (!hrx_status_is_ok(status)) {
    result << hrx::runtime::format_status(status);
  }
  hrx_status_ignore(status);
  return result;
}

class MemPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(Check(hrx_runtime_create(nullptr, hrx_host_allocator_system(),
                                         runtime_.for_output())));
    const hrx_device_create_params_t devices[] = {{.uri = "task://"},
                                                  {.uri = "task://"}};
    ASSERT_TRUE(Check(
        hrx_device_group_create(runtime_, 2, devices, group_.for_output())));
    for (size_t i = 0; i < 2; ++i) {
      devices_[i] = hrx_device_group_get_device(group_, i);
      queues_[i] = hrx_device_get_queue(devices_[i], 0, 0);
      ASSERT_NE(queues_[i], nullptr);
      ASSERT_TRUE(Check(hrx_semaphore_create(devices_[i], 0, &timelines_[i])));
    }
    ASSERT_TRUE(Check(hrx_semaphore_create(devices_[0], 0, &gate_)));
  }
  void TearDown() override {
    // Assertions can exit before the explicit gate opens. Retire accepted work
    // before releasing any source, finite range, or execution owner.
    uint64_t gate_value = 0;
    EXPECT_TRUE(Check(hrx_semaphore_query(gate_, &gate_value)));
    if (!gate_value) {
      EXPECT_TRUE(Check(hrx_semaphore_signal(gate_, 1)));
    }
    for (size_t i = 0; i < 2; ++i) {
      if (tails_[i]) {
        EXPECT_TRUE(
            Check(hrx_semaphore_wait(timelines_[i], tails_[i], UINT64_MAX)));
      }
    }
    for (auto buffer : buffers_) {
      hrx_buffer_release(buffer);
    }
    pools_.clear();
    hrx_semaphore_release(gate_);
    for (auto semaphore : timelines_) {
      hrx_semaphore_release(semaphore);
    }
  }
  hrx_status_t CreateSource(hrx_mem_pool_t* out_pool) {
    const hrx_pool_family_access_t families[] = {
        {
            .device = devices_[0],
            .family_ordinal = 0,
            .usage = HRX_BUFFER_USAGE_DEFAULT,
        },
        {
            .device = devices_[1],
            .family_ordinal = 0,
            .usage = HRX_BUFFER_USAGE_DEFAULT,
        },
    };
    const hrx_pool_scope_t scope = {
        .family_count = 2,
        .families = families,
        .host =
            {
                .access = HRX_MEMORY_ACCESS_ALL,
                .modes = HRX_MAPPING_MODE_SCOPED,
            },
    };
    return hrx_slab_pool_create(group_, scope, nullptr, out_pool);
  }
  hrx_status_t Allocate(size_t device, hrx_mem_pool_t pool,
                        const hrx_semaphore_list_t* waits, size_t count,
                        const hrx_pool_reservation_request_t* requests,
                        hrx_buffer_t* out_buffers) {
    uint64_t value = tails_[device] + 1;
    const hrx_semaphore_list_t signals = {&timelines_[device], &value, 1};
    hrx_status_t status = hrx_queue_alloca(queues_[device], waits, &signals,
                                           pool, count, requests, out_buffers);
    if (hrx_status_is_ok(status)) {
      tails_[device] = value;
    }
    return status;
  }
  hrx_status_t Free(size_t device, const hrx_semaphore_list_t* waits,
                    size_t count, const hrx_buffer_t* buffers) {
    uint64_t value = tails_[device] + 1;
    const hrx_semaphore_list_t signals = {&timelines_[device], &value, 1};
    hrx_status_t status =
        hrx_queue_dealloca(queues_[device], waits, &signals, count, buffers);
    if (hrx_status_is_ok(status)) {
      tails_[device] = value;
    }
    return status;
  }
  // Construction owners; retained independently by pools, buffers, and
  // timelines.
  hrx::runtime::runtime_ptr runtime_;
  // Immutable two-task-device access domain.
  hrx::runtime::device_group_ptr group_;
  // Borrowed devices in the sealed group.
  std::array<hrx_device_t, 2> devices_ = {};
  // Borrowed exact queues.
  std::array<hrx_queue_t, 2> queues_ = {};
  // Owned completion timelines.
  std::array<hrx_semaphore_t, 2> timelines_ = {};
  // Last successfully accepted signal for failure-safe cleanup.
  std::array<uint64_t, 2> tails_ = {};
  // Host gate proving allocation submission never waits for prerequisites.
  hrx_semaphore_t gate_ = nullptr;
  // Buffers released only after accepted work retires.
  std::vector<hrx_buffer_t> buffers_;
  // Independent allocation strategies retained through buffer cleanup.
  std::vector<hrx::runtime::mem_pool_ptr> pools_;
};

TEST_F(MemPoolTest, DeferredBatchesCarryDataAcrossDevices) {
  hrx::runtime::mem_pool_ptr source;
  ASSERT_TRUE(Check(CreateSource(source.for_output())));
  const hrx_tlsf_pool_options_t options = {.slab_size = 65536};
  pools_.resize(1);
  ASSERT_TRUE(
      Check(hrx_tlsf_pool_create(source, &options, pools_[0].for_output())));
  constexpr size_t count =
      9;  // Exercises overflow translation as well as batching.
  std::array<hrx_pool_reservation_request_t, count> requests;
  requests.fill({256, 16});
  buffers_.resize(count);
  uint64_t opened = 1;
  const hrx_semaphore_list_t wait_gate = {&gate_, &opened, 1};
  ASSERT_TRUE(Check(Allocate(0, pools_[0], &wait_gate, count, requests.data(),
                             buffers_.data())));
  uint64_t observed = UINT64_MAX;
  ASSERT_TRUE(Check(hrx_semaphore_query(timelines_[0], &observed)));
  EXPECT_EQ(observed, 0u);
  source.reset();
  group_.reset();
  runtime_.reset();
  // Both the allocation and its consumer are captured before the host gate.
  for (size_t i = 0; i < count; ++i) {
    const uint32_t pattern = 0x10203040u + static_cast<uint32_t>(i);
    const uint64_t prior = tails_[1];
    uint64_t next = prior + 1;
    hrx_semaphore_t waits[] = {timelines_[0], timelines_[1]};
    uint64_t values[] = {tails_[0], prior};
    const hrx_semaphore_list_t dependencies = {waits, values, 2};
    const hrx_semaphore_list_t signals = {&timelines_[1], &next, 1};
    ASSERT_TRUE(
        Check(hrx_queue_fill(queues_[1], &dependencies, &signals, buffers_[i],
                             0, 256, &pattern, sizeof(pattern), nullptr)));
    tails_[1] = next;
  }
  ASSERT_TRUE(Check(hrx_semaphore_signal(gate_, 1)));
  ASSERT_TRUE(Check(hrx_semaphore_wait(timelines_[1], tails_[1], UINT64_MAX)));
  for (size_t i = 0; i < count; ++i) {
    void* data = nullptr;
    ASSERT_TRUE(
        Check(hrx_buffer_map(buffers_[i], HRX_MAP_READ, 0, 256, &data)));
    for (size_t j = 0; j < 64; ++j) {
      EXPECT_EQ(static_cast<uint32_t*>(data)[j], 0x10203040u + i);
    }
    ASSERT_TRUE(Check(hrx_buffer_unmap(buffers_[i])));
  }
  const hrx_semaphore_list_t used = {&timelines_[1], &tails_[1], 1};
  ASSERT_TRUE(Check(Free(0, &used, count, buffers_.data())));
}

TEST_F(MemPoolTest, SiblingsReuseBackingAndTrimIndependently) {
  hrx::runtime::mem_pool_ptr source;
  ASSERT_TRUE(Check(CreateSource(source.for_output())));
  const hrx_tlsf_pool_options_t options = {.slab_size = 65536};
  hrx_slab_cache_options_t cache_options = {.max_count = 1};
  ASSERT_TRUE(Check(hrx_tlsf_pool_query_backing_request(source, &options,
                                                        &cache_options.slab)));
  pools_.resize(3);
  ASSERT_TRUE(Check(
      hrx_slab_cache_create(source, &cache_options, pools_[0].for_output())));
  ASSERT_TRUE(
      Check(hrx_tlsf_pool_create(pools_[0], &options, pools_[1].for_output())));
  ASSERT_TRUE(
      Check(hrx_tlsf_pool_create(pools_[0], &options, pools_[2].for_output())));
  buffers_.resize(2);
  const hrx_pool_reservation_request_t request = {256, 16};
  ASSERT_TRUE(
      Check(Allocate(0, pools_[1], nullptr, 1, &request, &buffers_[0])));
  ASSERT_TRUE(Check(hrx_semaphore_wait(timelines_[0], tails_[0], UINT64_MAX)));
  hrx_memory_scope_t scope;
  ASSERT_TRUE(Check(hrx_device_query_queue_family_memory_scope(
      devices_[0], 0, HRX_MEMORY_SITE_PROGRAM, &scope)));
  uint64_t first_address = 0;
  ASSERT_TRUE(Check(hrx_buffer_query_address(
      buffers_[0], scope, HRX_BUFFER_INTERFACE_HOST, &first_address)));
  uint64_t opened = 1;
  const hrx_semaphore_list_t gate = {&gate_, &opened, 1};
  ASSERT_TRUE(Check(Free(0, &gate, 1, &buffers_[0])));
  uint64_t freed = tails_[0];
  hrx_semaphore_t first_timeline = timelines_[0];
  const hrx_semaphore_list_t released = {&first_timeline, &freed, 1};
  ASSERT_TRUE(
      Check(Allocate(1, pools_[2], &released, 1, &request, &buffers_[1])));
  ASSERT_TRUE(Check(hrx_semaphore_signal(gate_, 1)));
  ASSERT_TRUE(Check(hrx_semaphore_wait(timelines_[1], tails_[1], UINT64_MAX)));
  uint64_t second_address = 0;
  ASSERT_TRUE(Check(hrx_buffer_query_address(
      buffers_[1], scope, HRX_BUFFER_INTERFACE_HOST, &second_address)));
  EXPECT_EQ(second_address, first_address);
  hrx_pool_stats_t stats;
  hrx_mem_pool_query_stats(source, &stats);
  EXPECT_EQ(stats.reserve_count, 1u);
  EXPECT_EQ(stats.reservation_count, 1u);
  hrx_mem_pool_trim(pools_[1], 0, HRX_POOL_TRIM_FLAG_ALL);
  hrx_mem_pool_trim(pools_[0], 0, HRX_POOL_TRIM_FLAG_ALL);
  hrx_mem_pool_query_stats(source, &stats);
  EXPECT_EQ(stats.reservation_count, 1u);
  void* data = nullptr;
  ASSERT_TRUE(Check(hrx_buffer_map(buffers_[1], HRX_MAP_WRITE, 0, 256, &data)));
  static_cast<uint32_t*>(data)[0] = 0xAABBCCDDu;
  ASSERT_TRUE(Check(hrx_buffer_unmap(buffers_[1])));
  const hrx_semaphore_list_t allocated = {&timelines_[1], &tails_[1], 1};
  ASSERT_TRUE(Check(Free(0, &allocated, 1, &buffers_[1])));
  ASSERT_TRUE(Check(hrx_semaphore_wait(timelines_[0], tails_[0], UINT64_MAX)));
  hrx_mem_pool_trim(pools_[0], 0, HRX_POOL_TRIM_FLAG_ALL);
  hrx_mem_pool_query_stats(source, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  hrx_mem_pool_query_stats(pools_[0], &stats);
  EXPECT_EQ(stats.bytes_committed, 0u);
}

TEST_F(MemPoolTest, FiniteRangesKeepOwnersThroughEverySuballocatorPair) {
  for (unsigned kinds = 0; kinds < 4; ++kinds) {
    SCOPED_TRACE(kinds);
    hrx::runtime::mem_pool_ptr source;
    ASSERT_TRUE(Check(CreateSource(source.for_output())));
    hrx::runtime::buffer_ptr backing;
    const hrx_pool_reservation_request_t backing_request = {65536, 4096};
    const hrx_semaphore_list_t prior = {&timelines_[0], &tails_[0], 1};
    ASSERT_TRUE(Check(Allocate(0, source, &prior, 1, &backing_request,
                               backing.for_output())));
    ASSERT_TRUE(
        Check(hrx_semaphore_wait(timelines_[0], tails_[0], UINT64_MAX)));
    hrx::runtime::mem_pool_ptr finite, child;
    const hrx_tlsf_pool_options_t finite_tlsf = {};
    const hrx_fixed_block_pool_options_t finite_blocks = {.block_size = 8192};
    if (kinds & 1) {
      ASSERT_TRUE(Check(hrx_fixed_block_pool_create_from_buffer(
          backing, 0, UINT64_MAX, &finite_blocks, finite.for_output())));
    } else {
      ASSERT_TRUE(Check(hrx_tlsf_pool_create_from_buffer(
          backing, 0, UINT64_MAX, &finite_tlsf, finite.for_output())));
    }
    const hrx_tlsf_pool_options_t tlsf = {.slab_size = 4096};
    const hrx_fixed_block_pool_options_t blocks = {
        .block_size = 256,
        .blocks_per_slab = 16,
    };
    if (kinds & 2) {
      hrx_pool_reservation_request_t geometry;
      ASSERT_TRUE(Check(hrx_fixed_block_pool_query_backing_request(
          finite, &blocks, &geometry)));
      EXPECT_GT(geometry.allocation_size, 0u);
      ASSERT_TRUE(Check(
          hrx_fixed_block_pool_create(finite, &blocks, child.for_output())));
    } else {
      ASSERT_TRUE(
          Check(hrx_tlsf_pool_create(finite, &tlsf, child.for_output())));
    }
    backing.reset();
    finite.reset();
    const size_t index = buffers_.size();
    buffers_.push_back(nullptr);
    const hrx_pool_reservation_request_t request = {256, 16};
    ASSERT_TRUE(
        Check(Allocate(0, child, &prior, 1, &request, &buffers_[index])));
    child.reset();
    ASSERT_TRUE(
        Check(hrx_semaphore_wait(timelines_[0], tails_[0], UINT64_MAX)));
    void* data = nullptr;
    ASSERT_TRUE(
        Check(hrx_buffer_map(buffers_[index], HRX_MAP_WRITE, 0, 256, &data)));
    static_cast<uint32_t*>(data)[63] = kinds;
    ASSERT_TRUE(Check(hrx_buffer_unmap(buffers_[index])));
    ASSERT_TRUE(Check(Free(0, &prior, 1, &buffers_[index])));
    ASSERT_TRUE(
        Check(hrx_semaphore_wait(timelines_[0], tails_[0], UINT64_MAX)));
    hrx_buffer_release(buffers_[index]);
    buffers_[index] = nullptr;
    hrx_pool_stats_t stats;
    hrx_mem_pool_query_stats(source, &stats);
    EXPECT_EQ(stats.reservation_count, 0u);
  }
}

TEST_F(MemPoolTest, RejectedBatchPublishesNoBuffersOrSignals) {
  hrx::runtime::mem_pool_ptr source;
  ASSERT_TRUE(Check(CreateSource(source.for_output())));
  hrx_pool_reservation_request_t requests[] = {{256, 16}, {256, 3}};
  hrx_buffer_t buffers[2] = {};
  uint64_t next = 1;
  const hrx_semaphore_list_t signals = {&timelines_[0], &next, 1};
  hrx_status_t status = hrx_queue_alloca(queues_[0], nullptr, &signals, source,
                                         2, requests, buffers);
  EXPECT_EQ(hrx_status_code(status), HRX_STATUS_INVALID_ARGUMENT);
  hrx_status_ignore(status);
  EXPECT_EQ(buffers[0], nullptr);
  EXPECT_EQ(buffers[1], nullptr);
  uint64_t observed = UINT64_MAX;
  ASSERT_TRUE(Check(hrx_semaphore_query(timelines_[0], &observed)));
  EXPECT_EQ(observed, 0u);
  hrx_pool_stats_t stats;
  hrx_mem_pool_query_stats(source, &stats);
  EXPECT_EQ(stats.reserve_count, 0u);
  // HRX preparation failure also unwinds all wrappers before native acceptance.
  requests[1] = {0, 0};
  status = hrx_queue_alloca(queues_[0], nullptr, &signals, source, 2, requests,
                            buffers);
  EXPECT_EQ(hrx_status_code(status), HRX_STATUS_OUT_OF_RANGE);
  hrx_status_ignore(status);
  EXPECT_EQ(buffers[0], nullptr);
  EXPECT_EQ(buffers[1], nullptr);
}

TEST_F(MemPoolTest, PlacementAndAtomicContractsAreQueryableBeforePayload) {
  uint32_t device_node = UINT32_MAX;
  for (size_t i = 0; i < hrx_device_group_topology_node_count(group_); ++i) {
    hrx_topology_node_t node;
    ASSERT_TRUE(Check(hrx_device_group_query_topology_node(group_, i, &node)));
    EXPECT_EQ(node.ordinal, i);
    if (node.kind == HRX_TOPOLOGY_NODE_KIND_LOGICAL_DEVICE &&
        node.device_ordinal == 0) {
      device_node = node.ordinal;
    }
  }
  ASSERT_NE(device_node, UINT32_MAX);
  const hrx_pool_family_access_t family = {
      .device = devices_[0],
      .family_ordinal = 0,
      .usage = HRX_BUFFER_USAGE_DEFAULT,
      .atomics =
          {
              .program =
                  {
                      .system_scope_32 = HRX_ATOMIC_OPERATION_FLAG_STORE,
                  },
          },
  };
  const hrx_pool_scope_t scope = {
      .family_count = 1,
      .families = &family,
      .host =
          {
              .access = HRX_MEMORY_ACCESS_ALL,
              .modes = HRX_MAPPING_MODE_SCOPED,
              .properties = HRX_HOST_MEMORY_PROPERTY_READ_CACHED |
                            HRX_HOST_MEMORY_PROPERTY_WRITE_BACK,
          },
  };
  const hrx_slab_pool_options_t options = {
      .placement = {.mode = HRX_POOL_PLACEMENT_REQUIRED, .node = device_node},
  };
  hrx::runtime::mem_pool_ptr source;
  // Task's ordinary malloc source cannot promise a physical allocation domain.
  // A required node fails instead of silently weakening placement.
  hrx_status_t rejected =
      hrx_slab_pool_create(group_, scope, &options, source.for_output());
  EXPECT_EQ(hrx_status_code(rejected), HRX_STATUS_UNAVAILABLE);
  hrx_status_ignore(rejected);
  EXPECT_FALSE(source);
  ASSERT_TRUE(
      Check(hrx_slab_pool_create(group_, scope, nullptr, source.for_output())));
  EXPECT_EQ(hrx_mem_pool_get_device_group(source), group_.get());
  hrx_pool_capabilities_t capabilities;
  hrx_mem_pool_query_capabilities(source, &capabilities);
  EXPECT_EQ(capabilities.placement.mode, HRX_POOL_PLACEMENT_AUTOMATIC);
  EXPECT_EQ(hrx_mem_pool_query_host_access(source).access,
            HRX_MEMORY_ACCESS_ALL);
  hrx_memory_scope_t program;
  ASSERT_TRUE(Check(hrx_device_query_queue_family_memory_scope(
      devices_[0], 0, HRX_MEMORY_SITE_PROGRAM, &program)));
  EXPECT_NE(
      hrx_mem_pool_query_atomic_operations(source, program).system_scope_32 &
          HRX_ATOMIC_OPERATION_FLAG_STORE,
      0u);
  EXPECT_EQ(hrx_mem_pool_query_zero_compute_atomic_operations(source, program)
                .system_scope_32,
            0u);
  hrx_pool_stats_t stats;
  hrx_mem_pool_query_stats(source, &stats);
  EXPECT_EQ(stats.bytes_committed, 0u);

  hrx_memory_scope_t excluded_program;
  ASSERT_TRUE(Check(hrx_device_query_queue_family_memory_scope(
      devices_[1], 0, HRX_MEMORY_SITE_PROGRAM, &excluded_program)));
  EXPECT_EQ(hrx_mem_pool_query_atomic_operations(source, excluded_program)
                .system_scope_32,
            0u);
  buffers_.resize(1);
  const hrx_pool_reservation_request_t request = {256, 16};
  uint64_t opened = 1;
  const hrx_semaphore_list_t wait_gate = {&gate_, &opened, 1};
  ASSERT_TRUE(
      Check(Allocate(0, source, &wait_gate, 1, &request, buffers_.data())));
  // The accepted handle already carries the immutable access contract. Queries
  // require neither physical backing readiness nor a host completion wait.
  EXPECT_NE(
      hrx_buffer_query_atomic_operations(buffers_[0], program).system_scope_32 &
          HRX_ATOMIC_OPERATION_FLAG_STORE,
      0u);
  EXPECT_EQ(hrx_buffer_query_atomic_operations(buffers_[0], excluded_program)
                .system_scope_32,
            0u);
  EXPECT_EQ(
      hrx_buffer_query_zero_compute_atomic_operations(buffers_[0], program)
          .system_scope_32,
      0u);
  const hrx_semaphore_list_t allocated = {&timelines_[0], &tails_[0], 1};
  ASSERT_TRUE(Check(Free(0, &allocated, 1, buffers_.data())));
}

}  // namespace
