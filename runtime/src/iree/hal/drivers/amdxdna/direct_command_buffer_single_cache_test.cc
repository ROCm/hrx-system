// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/direct_command_buffer_single_cache.h"

#include <cstdint>

#include "iree/testing/gtest.h"

namespace {

iree_allocator_t TestAllocator() { return iree_allocator_system(); }

iree_hal_amdxdna_native_queue_t* FakeQueue(uintptr_t value) {
  return reinterpret_cast<iree_hal_amdxdna_native_queue_t*>(value);
}

iree_hal_amdxdna_native_buffer_t* FakeBuffer(uintptr_t value) {
  return reinterpret_cast<iree_hal_amdxdna_native_buffer_t*>(value);
}

iree_hal_amdxdna_native_command_t* FakeCommand(uintptr_t value) {
  return reinterpret_cast<iree_hal_amdxdna_native_command_t*>(value);
}

void FreeSignature(iree_hal_amdxdna_device_single_command_cache_t* cache,
                   iree_hal_amdxdna_single_command_cache_entry_t* entry) {
  iree_allocator_free(cache->host_allocator, entry->ctrl_words);
  iree_allocator_free(cache->host_allocator, entry->binding_buffers);
  iree_allocator_free(cache->host_allocator, entry->binding_device_addrs);
  iree_allocator_free(cache->host_allocator, entry->binding_offsets);
  iree_allocator_free(cache->host_allocator, entry->binding_lengths);
  iree_allocator_free(cache->host_allocator, entry->owned_src_asm_inst.data);
  iree_allocator_free(cache->host_allocator, entry->owned_src_patches.data);
  *entry = {};
}

TEST(SingleCommandCacheTest, ExactHitReturnsPreparedEntry) {
  iree_hal_amdxdna_device_single_command_cache_t cache = {};
  cache.host_allocator = TestAllocator();
  const uint32_t ctrl_words[] = {1, 2, 3};
  iree_hal_amdxdna_native_buffer_t* binding_buffers[] = {FakeBuffer(0x100)};
  const uint64_t binding_device_addrs[] = {0x80000000};
  const iree_device_size_t binding_offsets[] = {16};
  const iree_device_size_t binding_lengths[] = {64};

  auto* stored = iree_hal_amdxdna_store_single_command_cache_entry(
      &cache, FakeQueue(0x200), /*cu_index=*/7, ctrl_words,
      IREE_ARRAYSIZE(ctrl_words), binding_buffers, binding_device_addrs,
      binding_offsets, binding_lengths, IREE_ARRAYSIZE(binding_buffers),
      FakeBuffer(0x300), FakeCommand(0x400));
  ASSERT_NE(stored, nullptr);

  iree_hal_amdxdna_single_command_cache_entry_t* found = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_find_single_command_cache_entry(
      &cache, FakeQueue(0x200), /*cu_index=*/7, ctrl_words,
      IREE_ARRAYSIZE(ctrl_words), binding_buffers, binding_device_addrs,
      binding_offsets, binding_lengths, IREE_ARRAYSIZE(binding_buffers),
      &found));
  EXPECT_EQ(found, stored);
  EXPECT_EQ(cache.use_clock, 2u);

  FreeSignature(&cache, stored);
}

TEST(SingleCommandCacheTest, DifferentQueueMissesWithoutUpdatingNativeCommand) {
  iree_hal_amdxdna_device_single_command_cache_t cache = {};
  cache.host_allocator = TestAllocator();
  const uint32_t ctrl_words[] = {4, 5};
  iree_hal_amdxdna_native_buffer_t* binding_buffers[] = {FakeBuffer(0x110)};
  const uint64_t binding_device_addrs[] = {0x81000000};
  const iree_device_size_t binding_offsets[] = {32};
  const iree_device_size_t binding_lengths[] = {128};

  auto* stored = iree_hal_amdxdna_store_single_command_cache_entry(
      &cache, FakeQueue(0x210), /*cu_index=*/3, ctrl_words,
      IREE_ARRAYSIZE(ctrl_words), binding_buffers, binding_device_addrs,
      binding_offsets, binding_lengths, IREE_ARRAYSIZE(binding_buffers),
      FakeBuffer(0x310), FakeCommand(0x410));
  ASSERT_NE(stored, nullptr);

  iree_hal_amdxdna_single_command_cache_entry_t* found = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_find_single_command_cache_entry(
      &cache, FakeQueue(0x220), /*cu_index=*/3, ctrl_words,
      IREE_ARRAYSIZE(ctrl_words), binding_buffers, binding_device_addrs,
      binding_offsets, binding_lengths, IREE_ARRAYSIZE(binding_buffers),
      &found));
  EXPECT_EQ(found, nullptr);

  FreeSignature(&cache, stored);
}

TEST(SingleCommandCacheTest, DescriptorTemplateHitIgnoresDynamicBindings) {
  iree_hal_amdxdna_device_single_command_cache_t cache = {};
  cache.host_allocator = TestAllocator();
  iree_slim_mutex_initialize(&cache.mutex);
  uint32_t ctrl_words[] = {9, 10, 11};
  iree_hal_amdxdna_native_buffer_t* binding_buffers[] = {FakeBuffer(0x140)};
  const uint64_t binding_device_addrs[] = {0x84000000};
  const iree_device_size_t binding_offsets[] = {80};
  const iree_device_size_t binding_lengths[] = {1024};
  iree_hal_amdxdna_u32_list_t asm_inst = {ctrl_words,
                                          IREE_ARRAYSIZE(ctrl_words)};
  uint32_t patch_words[] = {0, 0, 0};
  iree_hal_amdxdna_u32_list_t patches = {patch_words,
                                         IREE_ARRAYSIZE(patch_words)};

  auto* stored = iree_hal_amdxdna_store_single_command_cache_entry(
      &cache, FakeQueue(0x250), /*cu_index=*/8, ctrl_words,
      IREE_ARRAYSIZE(ctrl_words), binding_buffers, binding_device_addrs,
      binding_offsets, binding_lengths, IREE_ARRAYSIZE(binding_buffers),
      FakeBuffer(0x350), FakeCommand(0x450));
  ASSERT_NE(stored, nullptr);
  iree_hal_amdxdna_single_command_cache_entry_set_descriptor_template(
      &cache, stored, &asm_inst, &patches, /*constant_count=*/16,
      /*use_native_partial_elf=*/false, /*ctrl_code_mapped_ptr=*/nullptr);

  iree_hal_amdxdna_single_command_cache_entry_t* found = nullptr;
  IREE_CHECK_OK(
      iree_hal_amdxdna_find_single_command_cache_descriptor_template_entry(
          &cache, FakeQueue(0x250), /*cu_index=*/8, &asm_inst, &patches,
          /*constant_count=*/16, /*use_native_partial_elf=*/false,
          IREE_ARRAYSIZE(binding_buffers), &found));
  EXPECT_EQ(found, stored);

  iree_hal_amdxdna_single_command_cache_entry_acquire_in_flight(stored);
  found = nullptr;
  IREE_CHECK_OK(
      iree_hal_amdxdna_find_single_command_cache_descriptor_template_entry(
          &cache, FakeQueue(0x250), /*cu_index=*/8, &asm_inst, &patches,
          /*constant_count=*/16, /*use_native_partial_elf=*/false,
          IREE_ARRAYSIZE(binding_buffers), &found));
  EXPECT_EQ(found, nullptr);
  iree_hal_amdxdna_single_command_cache_entry_release_in_flight(&cache, stored);

  FreeSignature(&cache, stored);
  iree_slim_mutex_deinitialize(&cache.mutex);
}

TEST(SingleCommandCacheTest, DescriptorTemplateMissesDifferentBindingCount) {
  iree_hal_amdxdna_device_single_command_cache_t cache = {};
  cache.host_allocator = TestAllocator();
  iree_slim_mutex_initialize(&cache.mutex);
  uint32_t ctrl_words[] = {9, 10, 11};
  iree_hal_amdxdna_native_buffer_t* binding_buffers[] = {FakeBuffer(0x141)};
  const uint64_t binding_device_addrs[] = {0x84100000};
  const iree_device_size_t binding_offsets[] = {80};
  const iree_device_size_t binding_lengths[] = {1024};
  iree_hal_amdxdna_u32_list_t asm_inst = {ctrl_words,
                                          IREE_ARRAYSIZE(ctrl_words)};
  uint32_t patch_words[] = {0, 0, 0};
  iree_hal_amdxdna_u32_list_t patches = {patch_words,
                                         IREE_ARRAYSIZE(patch_words)};

  auto* stored = iree_hal_amdxdna_store_single_command_cache_entry(
      &cache, FakeQueue(0x251), /*cu_index=*/8, ctrl_words,
      IREE_ARRAYSIZE(ctrl_words), binding_buffers, binding_device_addrs,
      binding_offsets, binding_lengths, IREE_ARRAYSIZE(binding_buffers),
      FakeBuffer(0x351), FakeCommand(0x451));
  ASSERT_NE(stored, nullptr);
  iree_hal_amdxdna_single_command_cache_entry_set_descriptor_template(
      &cache, stored, &asm_inst, &patches, /*constant_count=*/16,
      /*use_native_partial_elf=*/false, /*ctrl_code_mapped_ptr=*/nullptr);

  iree_hal_amdxdna_single_command_cache_entry_t* found = nullptr;
  IREE_CHECK_OK(
      iree_hal_amdxdna_find_single_command_cache_descriptor_template_entry(
          &cache, FakeQueue(0x251), /*cu_index=*/8, &asm_inst, &patches,
          /*constant_count=*/16, /*use_native_partial_elf=*/false,
          IREE_ARRAYSIZE(binding_buffers) + 1, &found));
  EXPECT_EQ(found, nullptr);

  FreeSignature(&cache, stored);
  iree_slim_mutex_deinitialize(&cache.mutex);
}

TEST(SingleCommandCacheTest, DescriptorTemplateMissesDifferentTemplate) {
  iree_hal_amdxdna_device_single_command_cache_t cache = {};
  cache.host_allocator = TestAllocator();
  uint32_t ctrl_words[] = {12, 13};
  iree_hal_amdxdna_native_buffer_t* binding_buffers[] = {FakeBuffer(0x150)};
  const uint64_t binding_device_addrs[] = {0x85000000};
  const iree_device_size_t binding_offsets[] = {96};
  const iree_device_size_t binding_lengths[] = {2048};
  iree_hal_amdxdna_u32_list_t asm_inst = {ctrl_words,
                                          IREE_ARRAYSIZE(ctrl_words)};
  // Same shape/count as the stored template but different content: must miss.
  uint32_t other_ctrl_words[] = {12, 99};
  iree_hal_amdxdna_u32_list_t other_asm_inst = {
      other_ctrl_words, IREE_ARRAYSIZE(other_ctrl_words)};
  uint32_t patch_words[] = {0, 0, 0};
  iree_hal_amdxdna_u32_list_t patches = {patch_words,
                                         IREE_ARRAYSIZE(patch_words)};

  auto* stored = iree_hal_amdxdna_store_single_command_cache_entry(
      &cache, FakeQueue(0x260), /*cu_index=*/9, ctrl_words,
      IREE_ARRAYSIZE(ctrl_words), binding_buffers, binding_device_addrs,
      binding_offsets, binding_lengths, IREE_ARRAYSIZE(binding_buffers),
      FakeBuffer(0x360), FakeCommand(0x460));
  ASSERT_NE(stored, nullptr);
  iree_hal_amdxdna_single_command_cache_entry_set_descriptor_template(
      &cache, stored, &asm_inst, &patches, /*constant_count=*/0,
      /*use_native_partial_elf=*/false, /*ctrl_code_mapped_ptr=*/nullptr);

  iree_hal_amdxdna_single_command_cache_entry_t* found = nullptr;
  IREE_CHECK_OK(
      iree_hal_amdxdna_find_single_command_cache_descriptor_template_entry(
          &cache, FakeQueue(0x260), /*cu_index=*/9, &other_asm_inst, &patches,
          /*constant_count=*/0, /*use_native_partial_elf=*/false,
          IREE_ARRAYSIZE(binding_buffers), &found));
  EXPECT_EQ(found, nullptr);

  FreeSignature(&cache, stored);
}

// Regression test for the ABA collision: a cached descriptor template must be
// matched by content, never by the executable-owned pointer identity. A freed
// template's storage can be reallocated at the *same* address for a different
// same-shaped kernel (different content) -> that must MISS. Conversely an
// identical template presented through a *different* pointer must still HIT.
TEST(SingleCommandCacheTest, DescriptorTemplateMatchesByContentNotPointer) {
  iree_hal_amdxdna_device_single_command_cache_t cache = {};
  cache.host_allocator = TestAllocator();
  iree_slim_mutex_initialize(&cache.mutex);
  iree_hal_amdxdna_native_buffer_t* binding_buffers[] = {FakeBuffer(0x160)};
  const uint64_t binding_device_addrs[] = {0x86000000};
  const iree_device_size_t binding_offsets[] = {0};
  const iree_device_size_t binding_lengths[] = {4096};

  // Original template (e.g. the "PRED" gemm) recorded into the cache.
  uint32_t pred_words[] = {100, 101, 102, 103};
  iree_hal_amdxdna_u32_list_t pred_asm = {pred_words,
                                          IREE_ARRAYSIZE(pred_words)};
  uint32_t patch_words[] = {0, 0};
  iree_hal_amdxdna_u32_list_t patches = {patch_words,
                                         IREE_ARRAYSIZE(patch_words)};

  auto* stored = iree_hal_amdxdna_store_single_command_cache_entry(
      &cache, FakeQueue(0x270), /*cu_index=*/5, pred_words,
      IREE_ARRAYSIZE(pred_words), binding_buffers, binding_device_addrs,
      binding_offsets, binding_lengths, IREE_ARRAYSIZE(binding_buffers),
      FakeBuffer(0x370), FakeCommand(0x470));
  ASSERT_NE(stored, nullptr);
  iree_hal_amdxdna_single_command_cache_entry_set_descriptor_template(
      &cache, stored, &pred_asm, &patches, /*constant_count=*/0,
      /*use_native_partial_elf=*/false, /*ctrl_code_mapped_ptr=*/nullptr);

  // The cache must clone the template, so overwriting the caller's original
  // storage (simulating the executable being freed) must not affect matching.
  for (uint32_t& w : pred_words) w = 0xDEADBEEF;

  // ABA: a *different* kernel ("FAIL") whose same-shaped template happens to be
  // reallocated at an address indistinguishable from pointer identity. Emulate
  // the worst case by reusing the very same list struct/backing array the cache
  // recorded from, but with different content. This MUST miss.
  uint32_t fail_words[] = {200, 201, 202, 203};
  iree_hal_amdxdna_u32_list_t fail_asm = {fail_words,
                                          IREE_ARRAYSIZE(fail_words)};
  iree_hal_amdxdna_single_command_cache_entry_t* found = stored;
  IREE_CHECK_OK(
      iree_hal_amdxdna_find_single_command_cache_descriptor_template_entry(
          &cache, FakeQueue(0x270), /*cu_index=*/5, &fail_asm, &patches,
          /*constant_count=*/0, /*use_native_partial_elf=*/false,
          IREE_ARRAYSIZE(binding_buffers), &found));
  EXPECT_EQ(found, nullptr) << "different content must not reuse cached command";

  // An identical template presented through a brand-new pointer must still hit.
  uint32_t pred_words_copy[] = {100, 101, 102, 103};
  iree_hal_amdxdna_u32_list_t pred_asm_copy = {pred_words_copy,
                                               IREE_ARRAYSIZE(pred_words_copy)};
  found = nullptr;
  IREE_CHECK_OK(
      iree_hal_amdxdna_find_single_command_cache_descriptor_template_entry(
          &cache, FakeQueue(0x270), /*cu_index=*/5, &pred_asm_copy, &patches,
          /*constant_count=*/0, /*use_native_partial_elf=*/false,
          IREE_ARRAYSIZE(binding_buffers), &found));
  EXPECT_EQ(found, stored) << "identical content must reuse cached command";

  FreeSignature(&cache, stored);
  iree_slim_mutex_deinitialize(&cache.mutex);
}

TEST(SingleCommandCacheTest, InFlightEntryIsNotMatchedUntilReleased) {
  iree_hal_amdxdna_device_single_command_cache_t cache = {};
  cache.host_allocator = TestAllocator();
  iree_slim_mutex_initialize(&cache.mutex);
  const uint32_t ctrl_words[] = {6, 7};
  iree_hal_amdxdna_native_buffer_t* binding_buffers[] = {FakeBuffer(0x120)};
  const uint64_t binding_device_addrs[] = {0x82000000};
  const iree_device_size_t binding_offsets[] = {48};
  const iree_device_size_t binding_lengths[] = {256};

  auto* stored = iree_hal_amdxdna_store_single_command_cache_entry(
      &cache, FakeQueue(0x230), /*cu_index=*/4, ctrl_words,
      IREE_ARRAYSIZE(ctrl_words), binding_buffers, binding_device_addrs,
      binding_offsets, binding_lengths, IREE_ARRAYSIZE(binding_buffers),
      FakeBuffer(0x330), FakeCommand(0x430));
  ASSERT_NE(stored, nullptr);

  iree_hal_amdxdna_single_command_cache_entry_acquire_in_flight(stored);
  iree_hal_amdxdna_single_command_cache_entry_t* found = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_find_single_command_cache_entry(
      &cache, FakeQueue(0x230), /*cu_index=*/4, ctrl_words,
      IREE_ARRAYSIZE(ctrl_words), binding_buffers, binding_device_addrs,
      binding_offsets, binding_lengths, IREE_ARRAYSIZE(binding_buffers),
      &found));
  EXPECT_EQ(found, nullptr);

  iree_hal_amdxdna_single_command_cache_entry_release_in_flight(&cache, stored);
  IREE_CHECK_OK(iree_hal_amdxdna_find_single_command_cache_entry(
      &cache, FakeQueue(0x230), /*cu_index=*/4, ctrl_words,
      IREE_ARRAYSIZE(ctrl_words), binding_buffers, binding_device_addrs,
      binding_offsets, binding_lengths, IREE_ARRAYSIZE(binding_buffers),
      &found));
  EXPECT_EQ(found, stored);

  FreeSignature(&cache, stored);
  iree_slim_mutex_deinitialize(&cache.mutex);
}

TEST(SingleCommandCacheTest, InvalidateQueueDropsIdleAndDefersInFlight) {
  iree_hal_amdxdna_device_single_command_cache_t cache = {};
  cache.host_allocator = TestAllocator();
  iree_slim_mutex_initialize(&cache.mutex);
  const uint32_t ctrl_words[] = {20, 21};
  iree_hal_amdxdna_native_buffer_t* binding_buffers[] = {FakeBuffer(0x170)};
  const uint64_t binding_device_addrs[] = {0x87000000};
  const iree_device_size_t binding_offsets[] = {112};
  const iree_device_size_t binding_lengths[] = {4096};
  iree_hal_amdxdna_native_queue_t* target_queue = FakeQueue(0x280);
  iree_hal_amdxdna_native_queue_t* other_queue = FakeQueue(0x281);

  auto* idle = iree_hal_amdxdna_store_single_command_cache_entry(
      &cache, target_queue, /*cu_index=*/10, ctrl_words,
      IREE_ARRAYSIZE(ctrl_words), binding_buffers, binding_device_addrs,
      binding_offsets, binding_lengths, IREE_ARRAYSIZE(binding_buffers),
      /*ctrl_code_buffer=*/nullptr, /*command=*/nullptr);
  ASSERT_NE(idle, nullptr);
  auto* in_flight = iree_hal_amdxdna_store_single_command_cache_entry(
      &cache, target_queue, /*cu_index=*/11, ctrl_words,
      IREE_ARRAYSIZE(ctrl_words), binding_buffers, binding_device_addrs,
      binding_offsets, binding_lengths, IREE_ARRAYSIZE(binding_buffers),
      /*ctrl_code_buffer=*/nullptr, /*command=*/nullptr);
  ASSERT_NE(in_flight, nullptr);
  iree_hal_amdxdna_single_command_cache_entry_acquire_in_flight(in_flight);
  auto* other = iree_hal_amdxdna_store_single_command_cache_entry(
      &cache, other_queue, /*cu_index=*/12, ctrl_words,
      IREE_ARRAYSIZE(ctrl_words), binding_buffers, binding_device_addrs,
      binding_offsets, binding_lengths, IREE_ARRAYSIZE(binding_buffers),
      /*ctrl_code_buffer=*/nullptr, FakeCommand(0x480));
  ASSERT_NE(other, nullptr);

  iree_hal_amdxdna_single_command_cache_invalidate_queue(&cache, target_queue);

  EXPECT_EQ(idle->queue, nullptr);
  EXPECT_EQ(in_flight->queue, target_queue);
  EXPECT_TRUE(in_flight->invalidated);
  EXPECT_EQ(other->queue, other_queue);

  iree_hal_amdxdna_single_command_cache_entry_t* found = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_find_single_command_cache_entry(
      &cache, target_queue, /*cu_index=*/10, ctrl_words,
      IREE_ARRAYSIZE(ctrl_words), binding_buffers, binding_device_addrs,
      binding_offsets, binding_lengths, IREE_ARRAYSIZE(binding_buffers),
      &found));
  EXPECT_EQ(found, nullptr);

  IREE_CHECK_OK(iree_hal_amdxdna_find_single_command_cache_entry(
      &cache, other_queue, /*cu_index=*/12, ctrl_words,
      IREE_ARRAYSIZE(ctrl_words), binding_buffers, binding_device_addrs,
      binding_offsets, binding_lengths, IREE_ARRAYSIZE(binding_buffers),
      &found));
  EXPECT_EQ(found, other);

  iree_hal_amdxdna_single_command_cache_entry_release_in_flight(&cache,
                                                                in_flight);
  EXPECT_EQ(in_flight->queue, nullptr);
  EXPECT_FALSE(in_flight->invalidated);

  FreeSignature(&cache, other);
  iree_slim_mutex_deinitialize(&cache.mutex);
}

TEST(SingleCommandCacheTest, StoreReturnsNullWhenAllEntriesAreInFlight) {
  iree_hal_amdxdna_device_single_command_cache_t cache = {};
  cache.host_allocator = TestAllocator();
  const uint32_t ctrl_words[] = {8};
  iree_hal_amdxdna_native_buffer_t* binding_buffers[] = {FakeBuffer(0x130)};
  const uint64_t binding_device_addrs[] = {0x83000000};
  const iree_device_size_t binding_offsets[] = {64};
  const iree_device_size_t binding_lengths[] = {512};

  for (iree_host_size_t i = 0; i < kAmdxdnaSingleCommandCacheCapacity; ++i) {
    auto* stored = iree_hal_amdxdna_store_single_command_cache_entry(
        &cache, FakeQueue(0x240), /*cu_index=*/static_cast<uint32_t>(i),
        ctrl_words, IREE_ARRAYSIZE(ctrl_words), binding_buffers,
        binding_device_addrs, binding_offsets, binding_lengths,
        IREE_ARRAYSIZE(binding_buffers), FakeBuffer(0x340 + i),
        FakeCommand(0x440 + i));
    ASSERT_NE(stored, nullptr);
    iree_hal_amdxdna_single_command_cache_entry_acquire_in_flight(stored);
  }

  EXPECT_EQ(
      iree_hal_amdxdna_store_single_command_cache_entry(
          &cache, FakeQueue(0x240), /*cu_index=*/99, ctrl_words,
          IREE_ARRAYSIZE(ctrl_words), binding_buffers, binding_device_addrs,
          binding_offsets, binding_lengths, IREE_ARRAYSIZE(binding_buffers),
          FakeBuffer(0x399), FakeCommand(0x499)),
      nullptr);

  for (iree_host_size_t i = 0; i < cache.entry_count; ++i) {
    cache.entries[i].in_flight_count = 0;
    FreeSignature(&cache, &cache.entries[i]);
  }
}

}  // namespace
