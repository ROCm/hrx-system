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

}  // namespace
