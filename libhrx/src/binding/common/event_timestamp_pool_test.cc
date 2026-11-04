// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/event_timestamp_pool.h"

#include <atomic>
#include <cstdint>
#include <set>
#include <thread>
#include <vector>

#include "common/hrx_bridge.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/memory/heap_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "libhrx/src/libhrx/device.h"

namespace {

// Identity of a slot as the device sees it: the eight bytes it names.
struct SlotAddress {
  // Allocation containing the slot's bytes.
  iree_hal_buffer_t* buffer;
  // Byte offset within the allocation.
  iree_device_size_t offset;

  bool operator<(const SlotAddress& other) const {
    if (buffer != other.buffer) {
      return buffer < other.buffer;
    }
    return offset < other.offset;
  }
  bool operator==(const SlotAddress& other) const {
    return buffer == other.buffer && offset == other.offset;
  }
};

SlotAddress AddressOf(const iree_hal_streaming_event_timestamp_slot_t* slot) {
  return SlotAddress{iree_hal_streaming_event_timestamp_slot_buffer(slot),
                     iree_hal_streaming_event_timestamp_slot_offset(slot)};
}

// Forwards to the system allocator and counts the allocations it serves.
struct CountingAllocator {
  // Number of allocation requests served by this fixture.
  std::atomic<int64_t> allocation_count{0};

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* counter = static_cast<CountingAllocator*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE) {
      counter->allocation_count.fetch_add(1, std::memory_order_relaxed);
    }
    const iree_allocator_t system = iree_allocator_system();
    return system.ctl(system.self, command, params, inout_ptr);
  }

  iree_allocator_t AsAllocator() {
    return iree_allocator_t{this, &CountingAllocator::Control};
  }
};

// Pool backed by a host heap. One counting allocator backs both payload and
// metadata so the count exposes whether the timestamp pool grows lazily.
class EventTimestampPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const iree_hal_pool_host_access_t host_access = {
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
        .modes =
            IREE_HAL_MAPPING_MODE_SCOPED | IREE_HAL_MAPPING_MODE_PERSISTENT,
        .properties = (IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED |
                       IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK),
    };
    IREE_ASSERT_OK(
        iree_hal_heap_pool_create(host_access, counter_.AsAllocator(),
                                  counter_.AsAllocator(), &memory_pool_));
    iree_hal_streaming_event_timestamp_pool_initialize(
        memory_pool_, counter_.AsAllocator(), &pool_);
    // Source construction is not timestamp slab growth.
    counter_.allocation_count.store(0, std::memory_order_relaxed);
  }

  void TearDown() override {
    iree_hal_streaming_event_timestamp_pool_deinitialize(&pool_);
    iree_hal_pool_release(memory_pool_);
    memory_pool_ = nullptr;
  }

  iree_hal_streaming_event_timestamp_slot_t* Acquire() {
    iree_hal_streaming_event_timestamp_slot_t* slot = nullptr;
    IREE_EXPECT_OK(
        iree_hal_streaming_event_timestamp_pool_acquire(&pool_, &slot));
    EXPECT_NE(nullptr, slot);
    return slot;
  }

  std::vector<iree_hal_streaming_event_timestamp_slot_t*> AcquireMany(
      size_t count) {
    std::vector<iree_hal_streaming_event_timestamp_slot_t*> slots;
    slots.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      slots.push_back(Acquire());
    }
    return slots;
  }

  std::set<SlotAddress> AddressesOf(
      const std::vector<iree_hal_streaming_event_timestamp_slot_t*>& slots) {
    std::set<SlotAddress> addresses;
    for (auto* slot : slots) {
      addresses.insert(AddressOf(slot));
    }
    return addresses;
  }

  int64_t AllocationCount() const {
    return counter_.allocation_count.load(std::memory_order_relaxed);
  }

  CountingAllocator counter_;
  iree_hal_pool_t* memory_pool_ = nullptr;
  iree_hal_streaming_event_timestamp_pool_t pool_ = {};
};

// Enough slots to span several slabs whatever the pool's slab size is.
constexpr size_t kSlotsSpanningSeveralSlabs = 1500;

TEST_F(EventTimestampPoolTest, UnusedPoolAllocatesNothing) {
  EXPECT_EQ(0, AllocationCount());
  auto* slot = Acquire();
  EXPECT_GT(AllocationCount(), 0)
      << "the counting allocator saw nothing, so the zero above proves nothing";
  iree_hal_streaming_event_timestamp_slot_release(slot, nullptr, 0);
}

TEST_F(EventTimestampPoolTest, SlotsAreEightByteAlignedAndDistinct) {
  auto slots = AcquireMany(kSlotsSpanningSeveralSlabs);
  auto addresses = AddressesOf(slots);
  EXPECT_EQ(slots.size(), addresses.size())
      << "the pool handed the same eight bytes to two live holders";
  for (auto* slot : slots) {
    const SlotAddress address = AddressOf(slot);
    ASSERT_NE(nullptr, address.buffer);
    EXPECT_EQ(0u, address.offset % sizeof(uint64_t))
        << "a device timestamp target that is not eight-byte aligned is "
           "rejected by the driver";
    EXPECT_LE(address.offset + sizeof(uint64_t),
              iree_hal_buffer_allocation_size(address.buffer));
  }
  for (auto* slot : slots) {
    iree_hal_streaming_event_timestamp_slot_release(slot, nullptr, 0);
  }
}

TEST_F(EventTimestampPoolTest, SlotsDoNotAlias) {
  auto slots = AcquireMany(kSlotsSpanningSeveralSlabs);
  for (size_t i = 0; i < slots.size(); ++i) {
    const uint64_t tick = 0x0123456789abcdefull + i;
    IREE_ASSERT_OK(iree_hal_buffer_map_write(
        iree_hal_streaming_event_timestamp_slot_buffer(slots[i]),
        iree_hal_streaming_event_timestamp_slot_offset(slots[i]), &tick,
        sizeof(tick)));
  }
  for (size_t i = 0; i < slots.size(); ++i) {
    uint64_t tick = 0;
    IREE_ASSERT_OK(iree_hal_buffer_map_read(
        iree_hal_streaming_event_timestamp_slot_buffer(slots[i]),
        iree_hal_streaming_event_timestamp_slot_offset(slots[i]), &tick,
        sizeof(tick)));
    EXPECT_EQ(0x0123456789abcdefull + i, tick) << "slot " << i;
  }
  for (auto* slot : slots) {
    iree_hal_streaming_event_timestamp_slot_release(slot, nullptr, 0);
  }
}

TEST_F(EventTimestampPoolTest, ReleasedSlotsAreRecycled) {
  auto first = AcquireMany(kSlotsSpanningSeveralSlabs);
  auto first_addresses = AddressesOf(first);
  for (auto* slot : first) {
    iree_hal_streaming_event_timestamp_slot_release(slot, nullptr, 0);
  }

  auto second = AcquireMany(kSlotsSpanningSeveralSlabs);
  EXPECT_EQ(first_addresses, AddressesOf(second))
      << "the pool grew instead of reusing slots that were handed back";
  for (auto* slot : second) {
    iree_hal_streaming_event_timestamp_slot_release(slot, nullptr, 0);
  }
}

//===----------------------------------------------------------------------===//
// Reference counting
//===----------------------------------------------------------------------===//

// A reader holds the slot it reads a tick out of, so a record releasing the
// same slot concurrently cannot recycle it underneath that read.
TEST_F(EventTimestampPoolTest, SlotsHeldByASecondReferenceAreNotRecycled) {
  auto* held = Acquire();
  const SlotAddress address = AddressOf(held);
  iree_hal_streaming_event_timestamp_slot_retain(held);

  // No write is outstanding, so the reference count is the only thing keeping
  // this slot out of the free list.
  iree_hal_streaming_event_timestamp_slot_release(held, nullptr, 0);
  auto* other = Acquire();
  EXPECT_FALSE(address == AddressOf(other))
      << "a slot a second reference still holds was handed to a new holder";

  iree_hal_streaming_event_timestamp_slot_release(held, nullptr, 0);
  auto* recycled = Acquire();
  EXPECT_TRUE(address == AddressOf(recycled))
      << "the last reference did not return the slot to its pool";

  iree_hal_streaming_event_timestamp_slot_release(other, nullptr, 0);
  iree_hal_streaming_event_timestamp_slot_release(recycled, nullptr, 0);
}

//===----------------------------------------------------------------------===//
// Outstanding device writes
//===----------------------------------------------------------------------===//

// The CPU device creates the same public HAL timeline semaphore that a stream
// uses to retire its submitted timestamp writes. The pool only depends on that
// HAL contract; which driver implements the semaphore is deliberately hidden.
class EventTimestampPoolPendingTest : public EventTimestampPoolTest {
 protected:
  static void SetUpTestSuite() {
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_initialize(/*flags=*/0)));
    hrx_device_t hrx_device = nullptr;
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_device_get(/*index=*/0, &hrx_device)));
    device_ = hrx_device_hal(hrx_device);
  }
  static void TearDownTestSuite() {
    device_ = nullptr;
    IREE_EXPECT_OK(HRX_CALL(hrx_cpu_shutdown()));
  }

  void SetUp() override {
    EventTimestampPoolTest::SetUp();
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, /*initial_value=*/0ull,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore_));
  }
  void TearDown() override {
    iree_hal_semaphore_release(semaphore_);
    semaphore_ = nullptr;
    EventTimestampPoolTest::TearDown();
  }

  inline static iree_hal_device_t* device_ = nullptr;
  iree_hal_semaphore_t* semaphore_ = nullptr;
};

TEST_F(EventTimestampPoolPendingTest, SlotsWithWritesInFlightAreHeldBack) {
  auto in_flight = AcquireMany(kSlotsSpanningSeveralSlabs);
  auto in_flight_addresses = AddressesOf(in_flight);
  for (auto* slot : in_flight) {
    iree_hal_streaming_event_timestamp_slot_release(slot, semaphore_,
                                                    /*retire_value=*/1);
  }

  auto fresh = AcquireMany(kSlotsSpanningSeveralSlabs);
  for (auto* slot : fresh) {
    EXPECT_EQ(0u, in_flight_addresses.count(AddressOf(slot)))
        << "a slot with an outstanding device write was handed to a new holder";
  }
  for (auto* slot : fresh) {
    iree_hal_streaming_event_timestamp_slot_release(slot, nullptr, 0);
  }
}

// Growth is the observable: a pool that recovered nothing would ask the device
// for another slab every round.
TEST_F(EventTimestampPoolPendingTest, RetiredWritesReturnTheirSlots) {
  constexpr size_t kRoundCount = 5;
  int64_t allocations_after_first_round = 0;
  std::set<SlotAddress> all_addresses;
  for (size_t round = 0; round < kRoundCount; ++round) {
    auto slots = AcquireMany(kSlotsSpanningSeveralSlabs);
    for (auto* slot : slots) {
      all_addresses.insert(AddressOf(slot));
    }
    // Released while the write is still outstanding, so every slot lands on the
    // pending list, then retired so the next round can recover it.
    for (auto* slot : slots) {
      iree_hal_streaming_event_timestamp_slot_release(
          slot, semaphore_, /*retire_value=*/round + 1);
    }
    IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore_, round + 1, nullptr));
    if (round == 0) {
      allocations_after_first_round = AllocationCount();
    }
  }
  EXPECT_EQ(allocations_after_first_round, AllocationCount())
      << "the pool asked the device for more memory instead of recovering "
         "slots whose writes had retired";
  EXPECT_LT(all_addresses.size(), 2 * kSlotsSpanningSeveralSlabs)
      << "the pool handed out " << all_addresses.size()
      << " distinct slots over " << kRoundCount << " rounds of "
      << kSlotsSpanningSeveralSlabs
      << ", so slots held back for an outstanding write are not recovered";
}

TEST_F(EventTimestampPoolPendingTest, RetiredWritesRecycleImmediately) {
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore_, 4, nullptr));
  auto* slot = Acquire();
  const SlotAddress address = AddressOf(slot);
  iree_hal_streaming_event_timestamp_slot_release(slot, semaphore_,
                                                  /*retire_value=*/4);
  auto* recycled = Acquire();
  EXPECT_TRUE(address == AddressOf(recycled));
  iree_hal_streaming_event_timestamp_slot_release(recycled, nullptr, 0);
}

// Teardown frees the slabs only once every slot is back, because a slot record
// is storage inside a slab. This pins the returning side of that rule: the
// slots land on both the free and the pending list, so a teardown counting only
// one of them would read this pool as still holding slots out and keep slabs it
// should free. The other side is not reachable from a test here, because a pool
// torn down with a slot out trips the assertion and takes the process with it.
TEST_F(EventTimestampPoolPendingTest,
       DeinitializeReleasesTheSlabsOnceEverySlotIsBack) {
  iree_hal_streaming_event_timestamp_pool_t pool = {};
  iree_hal_streaming_event_timestamp_pool_initialize(
      memory_pool_, counter_.AsAllocator(), &pool);

  std::vector<iree_hal_streaming_event_timestamp_slot_t*> slots;
  slots.reserve(kSlotsSpanningSeveralSlabs);
  for (size_t i = 0; i < kSlotsSpanningSeveralSlabs; ++i) {
    iree_hal_streaming_event_timestamp_slot_t* slot = nullptr;
    IREE_ASSERT_OK(
        iree_hal_streaming_event_timestamp_pool_acquire(&pool, &slot));
    slots.push_back(slot);
  }
  ASSERT_NE(nullptr, pool.slabs) << "the pool served slots out of no slab";
  // The semaphore never reaches 1, so every odd slot is released with its write
  // still outstanding and stays on the pending list until teardown.
  for (size_t i = 0; i < slots.size(); ++i) {
    iree_hal_streaming_event_timestamp_slot_release(
        slots[i], i % 2 == 0 ? nullptr : semaphore_, /*retire_value=*/1);
  }

  iree_hal_streaming_event_timestamp_pool_deinitialize(&pool);
  EXPECT_EQ(nullptr, pool.slabs)
      << "the pool kept its slabs, which it does only for a slot no holder "
         "returned";
}

TEST_F(EventTimestampPoolPendingTest, FailedSemaphoresDoNotRecycleSlots) {
  auto in_flight = AcquireMany(kSlotsSpanningSeveralSlabs);
  const auto addresses = AddressesOf(in_flight);
  // Exercise failure both before release and after a slot is already pending.
  const size_t split = in_flight.size() / 2;
  for (size_t i = 0; i < split; ++i) {
    iree_hal_streaming_event_timestamp_slot_release(in_flight[i], semaphore_,
                                                    1);
  }
  iree_hal_semaphore_fail(
      semaphore_, iree_make_status(IREE_STATUS_ABORTED, "test failure"));
  for (size_t i = split; i < in_flight.size(); ++i) {
    iree_hal_streaming_event_timestamp_slot_release(in_flight[i], semaphore_,
                                                    1);
  }

  // Exhaust free slots so acquisition also drains the pending list. Neither
  // release order permits reusing bytes that an accepted write may still touch.
  auto fresh = AcquireMany(kSlotsSpanningSeveralSlabs);
  for (auto* slot : fresh) {
    EXPECT_EQ(0u, addresses.count(AddressOf(slot)));
    iree_hal_streaming_event_timestamp_slot_release(slot, nullptr, 0);
  }
}

TEST_F(EventTimestampPoolTest, ConcurrentHoldersNeverShareASlot) {
  constexpr size_t kThreadCount = 8;
  constexpr size_t kSlotsPerThread = 200;
  std::vector<std::vector<iree_hal_streaming_event_timestamp_slot_t*>> held(
      kThreadCount);
  std::atomic<size_t> ready{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (size_t t = 0; t < kThreadCount; ++t) {
    threads.emplace_back([&, t]() {
      // Churn first so acquires and releases overlap across threads, then hold
      // everything at once so the final state can be checked for overlap.
      for (size_t i = 0; i < kSlotsPerThread; ++i) {
        iree_hal_streaming_event_timestamp_slot_t* slot = nullptr;
        IREE_EXPECT_OK(
            iree_hal_streaming_event_timestamp_pool_acquire(&pool_, &slot));
        iree_hal_streaming_event_timestamp_slot_release(slot, nullptr, 0);
      }
      ready.fetch_add(1, std::memory_order_acq_rel);
      while (ready.load(std::memory_order_acquire) < kThreadCount) {
      }
      for (size_t i = 0; i < kSlotsPerThread; ++i) {
        iree_hal_streaming_event_timestamp_slot_t* slot = nullptr;
        IREE_EXPECT_OK(
            iree_hal_streaming_event_timestamp_pool_acquire(&pool_, &slot));
        held[t].push_back(slot);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  std::set<SlotAddress> addresses;
  size_t total = 0;
  for (const auto& per_thread : held) {
    for (auto* slot : per_thread) {
      ++total;
      EXPECT_TRUE(addresses.insert(AddressOf(slot)).second)
          << "two threads hold the same slot";
      iree_hal_streaming_event_timestamp_slot_release(slot, nullptr, 0);
    }
  }
  EXPECT_EQ(kThreadCount * kSlotsPerThread, total);
}

// Timestamp storage inherits its mapping contract from the selected source;
// allocation requests carry only slab geometry.
TEST_F(EventTimestampPoolTest, SlabsInheritConfiguredHostAccess) {
  const iree_hal_pool_host_access_t host_access =
      iree_hal_pool_query_host_access(memory_pool_);
  EXPECT_EQ(IREE_HAL_MEMORY_ACCESS_ALL, host_access.access);
  EXPECT_TRUE(iree_all_bits_set(
      host_access.modes,
      IREE_HAL_MAPPING_MODE_SCOPED | IREE_HAL_MAPPING_MODE_PERSISTENT));
  EXPECT_EQ((IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED |
             IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK),
            host_access.properties);

  auto* slot = Acquire();
  EXPECT_NE(nullptr, iree_hal_streaming_event_timestamp_slot_buffer(slot));
  iree_hal_streaming_event_timestamp_slot_release(slot, nullptr, 0);
}

}  // namespace
