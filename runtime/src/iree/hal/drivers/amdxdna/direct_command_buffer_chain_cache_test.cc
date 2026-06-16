// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/direct_command_buffer_chain_cache.h"

#include <cstdint>
#include <cstring>
#include <limits>

#include "iree/testing/gtest.h"

namespace {

iree_allocator_t TestAllocator() { return iree_allocator_system(); }

iree_hal_amdxdna_native_buffer_t* FakeBuffer(uintptr_t value) {
  return reinterpret_cast<iree_hal_amdxdna_native_buffer_t*>(value);
}

iree_hal_amdxdna_native_queue_t* FakeQueue(uintptr_t value) {
  return reinterpret_cast<iree_hal_amdxdna_native_queue_t*>(value);
}

const iree_hal_amdxdna_u32_list_t* FakeU32List(uintptr_t value) {
  return reinterpret_cast<const iree_hal_amdxdna_u32_list_t*>(value);
}

iree_hal_amdxdna_chain_cmd_t MakeCmd(iree_hal_amdxdna_native_buffer_t* buffer,
                                     uint64_t device_addr,
                                     size_t repeat_count = 1) {
  iree_hal_amdxdna_chain_cmd_t cmd;
  iree_hal_amdxdna_chain_cmd_initialize(&cmd);
  const uint32_t ctrl_words[] = {0xA, 0xB, 0xC};
  iree_hal_amdxdna_native_buffer_t* binding_buffers[] = {buffer};
  const uint64_t binding_device_addrs[] = {device_addr};
  const iree_device_size_t binding_offsets[] = {4};
  const iree_device_size_t binding_lengths[] = {128};
  IREE_CHECK_OK(iree_hal_amdxdna_chain_cmd_set_signature(
      TestAllocator(), &cmd, ctrl_words, IREE_ARRAYSIZE(ctrl_words),
      binding_buffers, binding_device_addrs, binding_offsets, binding_lengths,
      IREE_ARRAYSIZE(binding_device_addrs)));
  const uint8_t constants[] = {1, 2, 3, 4};
  IREE_CHECK_OK(iree_allocator_malloc_array(
      TestAllocator(), IREE_ARRAYSIZE(constants), sizeof(*cmd.src_constants),
      reinterpret_cast<void**>(&cmd.src_constants)));
  memcpy(cmd.src_constants, constants, sizeof(constants));
  cmd.src_constant_count = IREE_ARRAYSIZE(constants);
  cmd.src_asm_inst = FakeU32List(0x1000);
  cmd.src_patches = FakeU32List(0x2000);
  cmd.src_cu_idx.index = 7;
  cmd.src_use_native_partial_elf = true;
  cmd.repeat_count = repeat_count;
  return cmd;
}

iree_hal_amdxdna_chain_group_t MakeEmptyGroup() {
  iree_hal_amdxdna_chain_group_t group;
  iree_hal_amdxdna_chain_group_initialize(&group);
  group.queue = FakeQueue(0x3000);
  group.native_partial_elf = true;
  return group;
}

void AppendCmd(iree_hal_amdxdna_chain_group_t* group,
               iree_hal_amdxdna_chain_cmd_t* cmd) {
  IREE_CHECK_OK(iree_hal_amdxdna_chain_group_append_cmd_move(TestAllocator(),
                                                             group, cmd));
}

iree_hal_amdxdna_chain_group_t MakeGroup1(iree_hal_amdxdna_chain_cmd_t* cmd) {
  iree_hal_amdxdna_chain_group_t group = MakeEmptyGroup();
  AppendCmd(&group, cmd);
  return group;
}

iree_hal_amdxdna_chain_command_cache_entry_t MakeCacheEntry(
    iree_hal_amdxdna_chain_group_t* group, uint32_t max_slots = 24) {
  iree_hal_amdxdna_chain_command_cache_entry_t entry = {};
  iree_hal_amdxdna_chain_group_move(&entry.group, group);
  entry.max_slots = max_slots;
  // Matching requires at least one prepared parent chain; the tests only need
  // a non-empty marker and never destroy this fake command.
  entry.chain_count = 1;
  return entry;
}

TEST(ChainCommandCacheTest, DeviceMatchRejectsSameAddressDifferentBuffer) {
  auto cached_cmd = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto fresh_cmd = MakeCmd(FakeBuffer(0x20), /*device_addr=*/0x80000000);
  auto cached_group = MakeGroup1(&cached_cmd);
  auto fresh_group = MakeGroup1(&fresh_cmd);
  auto entry = MakeCacheEntry(&cached_group);

  EXPECT_FALSE(iree_hal_amdxdna_chain_command_cache_device_matches(
      &entry, &fresh_group, /*max_slots=*/24));
}

TEST(ChainCommandCacheTest, DeviceMatchAcceptsSameAddressAndBuffer) {
  auto cached_cmd = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto fresh_cmd = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto cached_group = MakeGroup1(&cached_cmd);
  auto fresh_group = MakeGroup1(&fresh_cmd);
  auto entry = MakeCacheEntry(&cached_group);

  EXPECT_TRUE(iree_hal_amdxdna_chain_command_cache_device_matches(
      &entry, &fresh_group, /*max_slots=*/24));
}

TEST(ChainCommandCacheTest, ShapeMatchAllowsDifferentBufferForRebind) {
  auto cached_cmd = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto fresh_cmd = MakeCmd(FakeBuffer(0x20), /*device_addr=*/0x80000000);
  auto cached_group = MakeGroup1(&cached_cmd);
  auto fresh_group = MakeGroup1(&fresh_cmd);
  auto entry = MakeCacheEntry(&cached_group);

  EXPECT_TRUE(iree_hal_amdxdna_chain_command_cache_shape_matches(
      &entry, &fresh_group, /*max_slots=*/24));
}

TEST(ChainCommandCacheTest, SignatureRewritePreservesSelfAliasedBindings) {
  auto cmd = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x11000);
  const uint32_t ctrl_words[] = {0xD, 0xE, 0xF};

  IREE_CHECK_OK(iree_hal_amdxdna_chain_cmd_set_signature(
      TestAllocator(), &cmd, ctrl_words, IREE_ARRAYSIZE(ctrl_words),
      cmd.binding_buffers, cmd.binding_device_addrs, cmd.binding_offsets,
      cmd.binding_lengths, cmd.binding_count));

  ASSERT_EQ(cmd.binding_count, 1u);
  EXPECT_EQ(cmd.binding_buffers[0], FakeBuffer(0x10));
  EXPECT_EQ(cmd.binding_device_addrs[0], 0x11000u);
  EXPECT_EQ(cmd.binding_offsets[0], 4u);
  EXPECT_EQ(cmd.binding_lengths[0], 128u);
  EXPECT_EQ(cmd.ctrl_word_count, IREE_ARRAYSIZE(ctrl_words));
  EXPECT_EQ(0, memcmp(cmd.ctrl_words, ctrl_words, sizeof(ctrl_words)));
  iree_hal_amdxdna_chain_cmd_deinitialize(TestAllocator(), &cmd);
}

TEST(ChainCommandCacheTest, DescriptorMatchExpandsRepeatCounts) {
  auto compact_cmd = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000,
                             /*repeat_count=*/3);
  auto cached_group = MakeGroup1(&compact_cmd);
  auto entry = MakeCacheEntry(&cached_group);
  auto group = MakeEmptyGroup();
  auto cmd0 = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto cmd1 = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto cmd2 = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  AppendCmd(&group, &cmd0);
  AppendCmd(&group, &cmd1);
  AppendCmd(&group, &cmd2);

  EXPECT_TRUE(iree_hal_amdxdna_chain_command_cache_descriptor_matches(
      &entry, &group, /*max_slots=*/24));
}

TEST(ChainCommandCacheTest, DescriptorMatchRejectsChangedBindings) {
  auto compact_cmd = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000,
                             /*repeat_count=*/3);
  auto cached_group = MakeGroup1(&compact_cmd);
  auto entry = MakeCacheEntry(&cached_group);
  auto group = MakeEmptyGroup();
  auto cmd0 = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto cmd1 = MakeCmd(FakeBuffer(0x20), /*device_addr=*/0x80000000);
  auto cmd2 = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  AppendCmd(&group, &cmd0);
  AppendCmd(&group, &cmd1);
  AppendCmd(&group, &cmd2);

  EXPECT_FALSE(iree_hal_amdxdna_chain_command_cache_descriptor_matches(
      &entry, &group, /*max_slots=*/24));
}

TEST(ChainCommandCacheTest, LogicalCommandCountSaturatesOnOverflow) {
  auto first = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000,
                       std::numeric_limits<size_t>::max());
  auto second = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000, 2);
  auto group = MakeEmptyGroup();
  AppendCmd(&group, &first);
  AppendCmd(&group, &second);

  EXPECT_EQ(iree_hal_amdxdna_chain_group_logical_command_count(&group),
            std::numeric_limits<size_t>::max());
}

}  // namespace
