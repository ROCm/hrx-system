// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/direct_command_buffer_chain_cache.h"

#include <cstdint>
#include <cstring>

#include "iree/testing/gtest.h"

namespace {

iree_allocator_t TestAllocator() { return iree_allocator_system(); }

iree_hal_amdxdna_native_buffer_t* FakeBuffer(uintptr_t value) {
  return reinterpret_cast<iree_hal_amdxdna_native_buffer_t*>(value);
}

iree_hal_amdxdna_native_queue_t* FakeQueue(uintptr_t value) {
  return reinterpret_cast<iree_hal_amdxdna_native_queue_t*>(value);
}

iree_hal_buffer_t* FakeHalBuffer(uintptr_t value) {
  return reinterpret_cast<iree_hal_buffer_t*>(value);
}

uint32_t kControlCodeData[] = {0x100, 0x101, 0x102, 0x103};
iree_hal_amdxdna_u32_list_t kControlCode = {
    kControlCodeData,
    IREE_ARRAYSIZE(kControlCodeData),
};
uint32_t kPatchTableData[] = {0, 0, 0};
iree_hal_amdxdna_u32_list_t kPatchTable = {
    kPatchTableData,
    IREE_ARRAYSIZE(kPatchTableData),
};

iree_hal_amdxdna_chain_cmd_t MakeCmd(iree_hal_amdxdna_native_buffer_t* buffer,
                                     uint64_t device_addr) {
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
  cmd.src_asm_inst = &kControlCode;
  cmd.src_patches = &kPatchTable;
  cmd.src_cu_idx.index = 7;
  cmd.src_use_native_partial_elf = true;
  return cmd;
}

void SetCmdConstants(iree_hal_amdxdna_chain_cmd_t* cmd,
                     const uint8_t* constants,
                     iree_host_size_t constant_count) {
  iree_allocator_free(TestAllocator(), cmd->src_constants);
  cmd->src_constants = nullptr;
  cmd->src_constant_count = 0;
  if (constant_count == 0) return;
  IREE_CHECK_OK(iree_allocator_malloc_array(
      TestAllocator(), constant_count, sizeof(*cmd->src_constants),
      reinterpret_cast<void**>(&cmd->src_constants)));
  memcpy(cmd->src_constants, constants, constant_count);
  cmd->src_constant_count = constant_count;
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

void AppendResourceOnlyCmd(iree_hal_amdxdna_chain_group_t* group,
                           iree_host_size_t ctrl_word_count) {
  iree_hal_amdxdna_chain_cmd_t cmd;
  iree_hal_amdxdna_chain_cmd_initialize(&cmd);
  cmd.ctrl_word_count = ctrl_word_count;
  AppendCmd(group, &cmd);
}

iree_hal_amdxdna_chain_group_t MakeResourceGroup(
    iree_host_size_t cmd_count, iree_host_size_t ctrl_word_count = 1) {
  iree_hal_amdxdna_chain_group_t group = MakeEmptyGroup();
  for (iree_host_size_t i = 0; i < cmd_count; ++i) {
    AppendResourceOnlyCmd(&group, ctrl_word_count);
  }
  return group;
}

void SetResourceEntry(iree_hal_amdxdna_chain_command_cache_entry_t* entry,
                      iree_host_size_t cmd_count, uint64_t last_use) {
  iree_hal_amdxdna_chain_group_t group = MakeResourceGroup(cmd_count);
  iree_hal_amdxdna_chain_group_move(&entry->group, &group);
  entry->last_use = last_use;
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

TEST(ChainCommandCacheTest, BindingRefsMatchExactRanges) {
  auto cached_group = MakeEmptyGroup();
  auto fresh_group = MakeEmptyGroup();
  IREE_CHECK_OK(iree_hal_amdxdna_chain_group_append_binding_ref_unique(
      TestAllocator(), &cached_group,
      iree_hal_make_buffer_ref(FakeHalBuffer(0x100), 64, 128)));
  IREE_CHECK_OK(iree_hal_amdxdna_chain_group_append_binding_ref_unique(
      TestAllocator(), &fresh_group,
      iree_hal_make_buffer_ref(FakeHalBuffer(0x100), 64, 128)));

  EXPECT_TRUE(iree_hal_amdxdna_chain_group_binding_refs_match(&cached_group,
                                                              &fresh_group));

  fresh_group.binding_refs[0].offset = 96;
  EXPECT_FALSE(iree_hal_amdxdna_chain_group_binding_refs_match(&cached_group,
                                                               &fresh_group));

  iree_hal_amdxdna_chain_group_deinitialize(TestAllocator(), &fresh_group);
  iree_hal_amdxdna_chain_group_deinitialize(TestAllocator(), &cached_group);
}

TEST(ChainCommandCacheTest, SetBindingRefsCopiesFreshRefs) {
  auto cached_group = MakeEmptyGroup();
  auto fresh_group = MakeEmptyGroup();
  IREE_CHECK_OK(iree_hal_amdxdna_chain_group_append_binding_ref_unique(
      TestAllocator(), &cached_group,
      iree_hal_make_buffer_ref(FakeHalBuffer(0x100), 64, 128)));
  IREE_CHECK_OK(iree_hal_amdxdna_chain_group_append_binding_ref_unique(
      TestAllocator(), &fresh_group,
      iree_hal_make_buffer_ref(FakeHalBuffer(0x200), 32, 256)));

  IREE_CHECK_OK(iree_hal_amdxdna_chain_group_set_binding_refs(
      TestAllocator(), &cached_group, &fresh_group));

  EXPECT_TRUE(iree_hal_amdxdna_chain_group_binding_refs_match(&cached_group,
                                                              &fresh_group));
  fresh_group.binding_refs[0].length = 512;
  EXPECT_FALSE(iree_hal_amdxdna_chain_group_binding_refs_match(&cached_group,
                                                               &fresh_group));

  iree_hal_amdxdna_chain_group_deinitialize(TestAllocator(), &fresh_group);
  iree_hal_amdxdna_chain_group_deinitialize(TestAllocator(), &cached_group);
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

TEST(ChainCommandCacheTest, DescriptorMatchAcceptsEqualMultiCommandGroup) {
  auto cached_cmd0 = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto cached_cmd1 = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto cached_cmd2 = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto cached_group = MakeEmptyGroup();
  AppendCmd(&cached_group, &cached_cmd0);
  AppendCmd(&cached_group, &cached_cmd1);
  AppendCmd(&cached_group, &cached_cmd2);
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
  auto cached_cmd0 = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto cached_cmd1 = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto cached_cmd2 = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto cached_group = MakeEmptyGroup();
  AppendCmd(&cached_group, &cached_cmd0);
  AppendCmd(&cached_group, &cached_cmd1);
  AppendCmd(&cached_group, &cached_cmd2);
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

TEST(ChainCommandCacheTest,
     DescriptorTemplateMatchAllowsSameShapeChangedConstants) {
  auto cached_cmd = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto fresh_cmd = MakeCmd(FakeBuffer(0x20), /*device_addr=*/0x90000000);
  const uint8_t fresh_constants[] = {5, 6, 7, 8};
  SetCmdConstants(&fresh_cmd, fresh_constants, IREE_ARRAYSIZE(fresh_constants));
  auto cached_group = MakeGroup1(&cached_cmd);
  auto fresh_group = MakeGroup1(&fresh_cmd);
  auto entry = MakeCacheEntry(&cached_group);

  EXPECT_FALSE(iree_hal_amdxdna_chain_command_cache_descriptor_matches(
      &entry, &fresh_group, /*max_slots=*/24));
  EXPECT_TRUE(iree_hal_amdxdna_chain_command_cache_descriptor_template_matches(
      &entry, &fresh_group, /*max_slots=*/24));
}

TEST(ChainCommandCacheTest,
     DescriptorTemplateMatchAcceptsDynamicStateForSameExecutableRun) {
  uint32_t cached_control_data[] = {0xA, 0x1000, 0x0, 0xD};
  uint32_t fresh_control_data[] = {0xB, 0x2000, 0x1, 0xE};
  iree_hal_amdxdna_u32_list_t cached_control = {
      cached_control_data,
      IREE_ARRAYSIZE(cached_control_data),
  };
  iree_hal_amdxdna_u32_list_t fresh_control = {
      fresh_control_data,
      IREE_ARRAYSIZE(fresh_control_data),
  };
  auto cached_cmd = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto fresh_cmd = MakeCmd(FakeBuffer(0x20), /*device_addr=*/0x90000000);
  cached_cmd.src_executable_identity = 7;
  fresh_cmd.src_executable_identity = 7;
  cached_cmd.src_entry_point = 2;
  fresh_cmd.src_entry_point = 2;
  cached_cmd.src_run_ordinal = 3;
  fresh_cmd.src_run_ordinal = 3;
  cached_cmd.src_asm_inst = &cached_control;
  fresh_cmd.src_asm_inst = &fresh_control;
  const uint8_t fresh_constants[] = {5, 6, 7, 8};
  SetCmdConstants(&fresh_cmd, fresh_constants, IREE_ARRAYSIZE(fresh_constants));
  auto cached_group = MakeGroup1(&cached_cmd);
  auto fresh_group = MakeGroup1(&fresh_cmd);
  auto entry = MakeCacheEntry(&cached_group);

  EXPECT_FALSE(iree_hal_amdxdna_chain_command_cache_descriptor_matches(
      &entry, &fresh_group, /*max_slots=*/24));
  EXPECT_TRUE(iree_hal_amdxdna_chain_command_cache_descriptor_template_matches(
      &entry, &fresh_group, /*max_slots=*/24));
}

TEST(ChainCommandCacheTest,
     PartialElfExactDescriptorRejectsChangedTxnForSameExecutableRun) {
  uint32_t cached_control_data[] = {0xA, 0x1000, 0x0, 0xD};
  uint32_t fresh_control_data[] = {0xA, 0x2000, 0x0, 0xD};
  iree_hal_amdxdna_u32_list_t cached_control = {
      cached_control_data,
      IREE_ARRAYSIZE(cached_control_data),
  };
  iree_hal_amdxdna_u32_list_t fresh_control = {
      fresh_control_data,
      IREE_ARRAYSIZE(fresh_control_data),
  };
  auto cached_cmd = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto fresh_cmd = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  cached_cmd.src_executable_identity = 7;
  fresh_cmd.src_executable_identity = 7;
  cached_cmd.src_entry_point = 2;
  fresh_cmd.src_entry_point = 2;
  cached_cmd.src_run_ordinal = 3;
  fresh_cmd.src_run_ordinal = 3;
  cached_cmd.src_asm_inst = &cached_control;
  fresh_cmd.src_asm_inst = &fresh_control;
  auto cached_group = MakeGroup1(&cached_cmd);
  auto fresh_group = MakeGroup1(&fresh_cmd);
  auto entry = MakeCacheEntry(&cached_group);

  EXPECT_FALSE(iree_hal_amdxdna_chain_command_cache_descriptor_matches(
      &entry, &fresh_group, /*max_slots=*/24));
  EXPECT_TRUE(iree_hal_amdxdna_chain_command_cache_descriptor_template_matches(
      &entry, &fresh_group, /*max_slots=*/24));
}

TEST(ChainCommandCacheTest, DeferredDescriptorReplacementOwnsFreshTxn) {
  uint32_t cached_control_data[] = {0xA, 0x1000, 0x0, 0xD};
  uint32_t fresh_control_data[] = {0xA, 0x2000, 0x0, 0xD};
  iree_hal_amdxdna_u32_list_t cached_control = {
      cached_control_data,
      IREE_ARRAYSIZE(cached_control_data),
  };
  iree_hal_amdxdna_u32_list_t fresh_control = {
      fresh_control_data,
      IREE_ARRAYSIZE(fresh_control_data),
  };
  auto cached_cmd = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto fresh_cmd = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  cached_cmd.src_asm_inst = &cached_control;
  fresh_cmd.src_asm_inst = &fresh_control;

  IREE_CHECK_OK(iree_hal_amdxdna_chain_cmd_make_deferred_lists_owned(
      TestAllocator(), &cached_cmd));
  IREE_CHECK_OK(iree_hal_amdxdna_chain_cmd_replace_deferred_lists_owned(
      TestAllocator(), &cached_cmd, &fresh_cmd));
  fresh_control_data[1] = 0x3000;

  ASSERT_EQ(cached_cmd.src_asm_inst, &cached_cmd.owned_src_asm_inst);
  ASSERT_EQ(cached_cmd.src_asm_inst->count, fresh_control.count);
  EXPECT_EQ(cached_cmd.src_asm_inst->data[1], 0x2000u);
  iree_hal_amdxdna_chain_cmd_deinitialize(TestAllocator(), &cached_cmd);
  iree_hal_amdxdna_chain_cmd_deinitialize(TestAllocator(), &fresh_cmd);
}

TEST(ChainCommandCacheTest,
     DescriptorMatchesRejectDifferentExecutableOrRunWithSameShape) {
  auto cached_cmd = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  cached_cmd.src_executable_identity = 7;
  cached_cmd.src_entry_point = 2;
  cached_cmd.src_run_ordinal = 3;
  auto cached_group = MakeGroup1(&cached_cmd);
  auto entry = MakeCacheEntry(&cached_group);

  auto different_executable_cmd =
      MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  different_executable_cmd.src_executable_identity = 8;
  different_executable_cmd.src_entry_point = 2;
  different_executable_cmd.src_run_ordinal = 3;
  auto different_executable_group = MakeGroup1(&different_executable_cmd);
  EXPECT_FALSE(iree_hal_amdxdna_chain_command_cache_descriptor_matches(
      &entry, &different_executable_group, /*max_slots=*/24));
  EXPECT_FALSE(iree_hal_amdxdna_chain_command_cache_descriptor_template_matches(
      &entry, &different_executable_group, /*max_slots=*/24));

  auto different_run_cmd =
      MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  different_run_cmd.src_executable_identity = 7;
  different_run_cmd.src_entry_point = 2;
  different_run_cmd.src_run_ordinal = 4;
  auto different_run_group = MakeGroup1(&different_run_cmd);
  EXPECT_FALSE(iree_hal_amdxdna_chain_command_cache_descriptor_matches(
      &entry, &different_run_group, /*max_slots=*/24));
  EXPECT_FALSE(iree_hal_amdxdna_chain_command_cache_descriptor_template_matches(
      &entry, &different_run_group, /*max_slots=*/24));
}

// A shape-match rewrite can realize control code from a program the cached
// command's descriptor was not cloned from, which retires that descriptor. Both
// descriptor paths reuse the cached descriptor as the source of truth for the
// control code (one resubmits the entry unchanged, the other re-realizes from
// the retained template), so neither may match a retired descriptor. The
// content-keyed paths compare the realized words and stay usable.
TEST(ChainCommandCacheTest, RetiredDescriptorOnlyMatchesByContent) {
  auto cached_cmd = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto fresh_cmd = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  cached_cmd.src_executable_identity = 7;
  fresh_cmd.src_executable_identity = 7;
  cached_cmd.src_entry_point = 2;
  fresh_cmd.src_entry_point = 2;
  cached_cmd.src_run_ordinal = 3;
  fresh_cmd.src_run_ordinal = 3;
  cached_cmd.descriptor_describes_contents = false;
  auto cached_group = MakeGroup1(&cached_cmd);
  auto fresh_group = MakeGroup1(&fresh_cmd);
  auto entry = MakeCacheEntry(&cached_group);

  EXPECT_FALSE(iree_hal_amdxdna_chain_command_cache_descriptor_matches(
      &entry, &fresh_group, /*max_slots=*/24));
  EXPECT_FALSE(iree_hal_amdxdna_chain_command_cache_descriptor_template_matches(
      &entry, &fresh_group, /*max_slots=*/24));
  EXPECT_TRUE(iree_hal_amdxdna_chain_command_cache_device_matches(
      &entry, &fresh_group, /*max_slots=*/24));
  EXPECT_TRUE(iree_hal_amdxdna_chain_command_cache_shape_matches(
      &entry, &fresh_group, /*max_slots=*/24));
}

TEST(ChainCommandCacheTest,
     DescriptorTemplateMatchRejectsChangedConstantShape) {
  auto cached_cmd = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto fresh_cmd = MakeCmd(FakeBuffer(0x20), /*device_addr=*/0x90000000);
  const uint8_t fresh_constants[] = {5, 6, 7, 8, 9};
  SetCmdConstants(&fresh_cmd, fresh_constants, IREE_ARRAYSIZE(fresh_constants));
  auto cached_group = MakeGroup1(&cached_cmd);
  auto fresh_group = MakeGroup1(&fresh_cmd);
  auto entry = MakeCacheEntry(&cached_group);

  EXPECT_FALSE(iree_hal_amdxdna_chain_command_cache_descriptor_template_matches(
      &entry, &fresh_group, /*max_slots=*/24));
}

TEST(ChainCommandCacheTest,
     DescriptorTemplateMatchAllowsChangedReconfigurationBuffers) {
  auto cached_cmd = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto fresh_cmd = MakeCmd(FakeBuffer(0x20), /*device_addr=*/0x90000000);
  auto cached_group = MakeGroup1(&cached_cmd);
  auto fresh_group = MakeGroup1(&fresh_cmd);
  IREE_CHECK_OK(iree_hal_amdxdna_chain_group_append_reconf_buffer(
      TestAllocator(), &cached_group, FakeBuffer(0x1000)));
  IREE_CHECK_OK(iree_hal_amdxdna_chain_group_append_reconf_buffer(
      TestAllocator(), &fresh_group, FakeBuffer(0x2000)));
  auto entry = MakeCacheEntry(&cached_group);

  EXPECT_FALSE(iree_hal_amdxdna_chain_command_cache_descriptor_matches(
      &entry, &fresh_group, /*max_slots=*/24));
  EXPECT_TRUE(iree_hal_amdxdna_chain_command_cache_descriptor_template_matches(
      &entry, &fresh_group, /*max_slots=*/24));
}

TEST(ChainCommandCacheTest,
     DescriptorTemplateMatchRejectsChangedReconfigurationCount) {
  auto cached_cmd = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto fresh_cmd = MakeCmd(FakeBuffer(0x20), /*device_addr=*/0x90000000);
  auto cached_group = MakeGroup1(&cached_cmd);
  auto fresh_group = MakeGroup1(&fresh_cmd);
  IREE_CHECK_OK(iree_hal_amdxdna_chain_group_append_reconf_buffer(
      TestAllocator(), &cached_group, FakeBuffer(0x1000)));
  IREE_CHECK_OK(iree_hal_amdxdna_chain_group_append_reconf_buffer(
      TestAllocator(), &fresh_group, FakeBuffer(0x2000)));
  IREE_CHECK_OK(iree_hal_amdxdna_chain_group_append_reconf_buffer(
      TestAllocator(), &fresh_group, FakeBuffer(0x3000)));
  auto entry = MakeCacheEntry(&cached_group);

  EXPECT_FALSE(iree_hal_amdxdna_chain_command_cache_descriptor_template_matches(
      &entry, &fresh_group, /*max_slots=*/24));
}

TEST(ChainCommandCacheTest,
     DescriptorTemplateMatchAllowsPatchTableDynamicControlWords) {
  uint32_t cached_control_data[] = {0xA, 0x1000, 0x0, 0xD};
  uint32_t fresh_control_data[] = {0xA, 0x2000, 0x1, 0xD};
  iree_hal_amdxdna_u32_list_t cached_control = {
      cached_control_data,
      IREE_ARRAYSIZE(cached_control_data),
  };
  iree_hal_amdxdna_u32_list_t fresh_control = {
      fresh_control_data,
      IREE_ARRAYSIZE(fresh_control_data),
  };
  auto cached_cmd = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto fresh_cmd = MakeCmd(FakeBuffer(0x20), /*device_addr=*/0x90000000);
  cached_cmd.src_asm_inst = &cached_control;
  fresh_cmd.src_asm_inst = &fresh_control;
  auto cached_group = MakeGroup1(&cached_cmd);
  auto fresh_group = MakeGroup1(&fresh_cmd);
  auto entry = MakeCacheEntry(&cached_group);

  EXPECT_FALSE(iree_hal_amdxdna_chain_command_cache_descriptor_matches(
      &entry, &fresh_group, /*max_slots=*/24));
  EXPECT_TRUE(iree_hal_amdxdna_chain_command_cache_descriptor_template_matches(
      &entry, &fresh_group, /*max_slots=*/24));
}

TEST(ChainCommandCacheTest,
     DescriptorTemplateMatchAllowsPatchTableDynamicAddends) {
  uint32_t cached_patch_data[] = {0, 0, 0};
  uint32_t fresh_patch_data[] = {0, 0, 64};
  iree_hal_amdxdna_u32_list_t cached_patches = {
      cached_patch_data,
      IREE_ARRAYSIZE(cached_patch_data),
  };
  iree_hal_amdxdna_u32_list_t fresh_patches = {
      fresh_patch_data,
      IREE_ARRAYSIZE(fresh_patch_data),
  };
  auto cached_cmd = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto fresh_cmd = MakeCmd(FakeBuffer(0x20), /*device_addr=*/0x90000000);
  cached_cmd.src_patches = &cached_patches;
  fresh_cmd.src_patches = &fresh_patches;
  auto cached_group = MakeGroup1(&cached_cmd);
  auto fresh_group = MakeGroup1(&fresh_cmd);
  auto entry = MakeCacheEntry(&cached_group);

  EXPECT_FALSE(iree_hal_amdxdna_chain_command_cache_descriptor_matches(
      &entry, &fresh_group, /*max_slots=*/24));
  EXPECT_TRUE(iree_hal_amdxdna_chain_command_cache_descriptor_template_matches(
      &entry, &fresh_group, /*max_slots=*/24));
}

TEST(ChainCommandCacheTest,
     DescriptorTemplateMatchAllowsPatchTableChangesWithSameNativeShape) {
  uint32_t cached_patch_data[] = {0, 0, 0};
  uint32_t offset_changed_patch_data[] = {4, 0, 0};
  uint32_t arg_changed_patch_data[] = {0, 1, 0};
  iree_hal_amdxdna_u32_list_t cached_patches = {
      cached_patch_data,
      IREE_ARRAYSIZE(cached_patch_data),
  };
  iree_hal_amdxdna_u32_list_t offset_changed_patches = {
      offset_changed_patch_data,
      IREE_ARRAYSIZE(offset_changed_patch_data),
  };
  iree_hal_amdxdna_u32_list_t arg_changed_patches = {
      arg_changed_patch_data,
      IREE_ARRAYSIZE(arg_changed_patch_data),
  };
  auto cached_cmd = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  cached_cmd.src_patches = &cached_patches;
  auto cached_group = MakeGroup1(&cached_cmd);
  auto entry = MakeCacheEntry(&cached_group);

  auto offset_changed_cmd =
      MakeCmd(FakeBuffer(0x20), /*device_addr=*/0x90000000);
  offset_changed_cmd.src_patches = &offset_changed_patches;
  auto offset_changed_group = MakeGroup1(&offset_changed_cmd);
  EXPECT_TRUE(iree_hal_amdxdna_chain_command_cache_descriptor_template_matches(
      &entry, &offset_changed_group, /*max_slots=*/24));

  auto arg_changed_cmd = MakeCmd(FakeBuffer(0x30), /*device_addr=*/0xA0000000);
  arg_changed_cmd.src_patches = &arg_changed_patches;
  auto arg_changed_group = MakeGroup1(&arg_changed_cmd);
  EXPECT_TRUE(iree_hal_amdxdna_chain_command_cache_descriptor_template_matches(
      &entry, &arg_changed_group, /*max_slots=*/24));
}

TEST(ChainCommandCacheTest, InFlightEntryIsNotMatchedUntilReleased) {
  auto cached_cmd = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto fresh_cmd = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto cached_group = MakeGroup1(&cached_cmd);
  auto fresh_group = MakeGroup1(&fresh_cmd);
  auto entry = MakeCacheEntry(&cached_group);
  iree_hal_amdxdna_device_chain_command_cache_t cache = {};
  cache.host_allocator = TestAllocator();
  iree_slim_mutex_initialize(&cache.mutex);

  iree_hal_amdxdna_chain_command_cache_entry_acquire_in_flight(&entry);
  EXPECT_FALSE(iree_hal_amdxdna_chain_command_cache_descriptor_matches(
      &entry, &fresh_group, /*max_slots=*/24));
  EXPECT_FALSE(iree_hal_amdxdna_chain_command_cache_descriptor_template_matches(
      &entry, &fresh_group, /*max_slots=*/24));

  iree_hal_amdxdna_chain_command_cache_entry_release_in_flight(&cache, &entry);
  EXPECT_TRUE(iree_hal_amdxdna_chain_command_cache_descriptor_matches(
      &entry, &fresh_group, /*max_slots=*/24));
  EXPECT_TRUE(iree_hal_amdxdna_chain_command_cache_descriptor_template_matches(
      &entry, &fresh_group, /*max_slots=*/24));

  iree_slim_mutex_deinitialize(&cache.mutex);
}

TEST(ChainCommandCacheTest, InvalidateQueueDropsIdleAndDefersInFlight) {
  iree_hal_amdxdna_device_chain_command_cache_t cache = {};
  cache.host_allocator = TestAllocator();
  cache.entry_count = 3;
  iree_slim_mutex_initialize(&cache.mutex);
  iree_hal_amdxdna_native_queue_t* target_queue = FakeQueue(0x3100);
  iree_hal_amdxdna_native_queue_t* other_queue = FakeQueue(0x3101);
  for (iree_host_size_t i = 0; i < cache.entry_count; ++i) {
    iree_hal_amdxdna_chain_group_initialize(&cache.entries[i].group);
    cache.entries[i].last_use = i + 1;
  }
  cache.entries[0].group.queue = target_queue;
  cache.entries[1].group.queue = target_queue;
  cache.entries[1].in_flight_count = 1;
  cache.entries[2].group.queue = other_queue;

  iree_hal_amdxdna_chain_command_cache_invalidate_queue(&cache, target_queue);

  EXPECT_EQ(cache.entries[0].group.queue, nullptr);
  EXPECT_EQ(cache.entries[1].group.queue, target_queue);
  EXPECT_TRUE(cache.entries[1].invalidated);
  EXPECT_EQ(cache.entries[2].group.queue, other_queue);

  iree_hal_amdxdna_chain_command_cache_entry_release_in_flight(
      &cache, &cache.entries[1]);
  EXPECT_EQ(cache.entries[1].group.queue, nullptr);
  EXPECT_FALSE(cache.entries[1].invalidated);

  for (iree_host_size_t i = 0; i < cache.entry_count; ++i) {
    iree_hal_amdxdna_chain_group_deinitialize(TestAllocator(),
                                              &cache.entries[i].group);
  }
  iree_slim_mutex_deinitialize(&cache.mutex);
}

TEST(ChainCommandCacheTest, AllocateEntryReturnsNullWhenAllEntriesAreInFlight) {
  iree_hal_amdxdna_device_chain_command_cache_t cache = {};
  cache.host_allocator = TestAllocator();
  cache.entry_count = kAmdxdnaChainCommandCacheCapacity;
  for (iree_host_size_t i = 0; i < cache.entry_count; ++i) {
    iree_hal_amdxdna_chain_group_initialize(&cache.entries[i].group);
    cache.entries[i].last_use = i + 1;
    cache.entries[i].in_flight_count = 1;
  }
  auto request_group = MakeResourceGroup(1);

  EXPECT_EQ(iree_hal_amdxdna_chain_command_cache_allocate_entry(
                &cache, &request_group, /*max_slots=*/24),
            nullptr);

  iree_hal_amdxdna_chain_group_deinitialize(TestAllocator(), &request_group);
}

TEST(ChainCommandCacheTest, AllocateEntryRejectsOverBudgetRequest) {
  iree_hal_amdxdna_device_chain_command_cache_t cache = {};
  cache.host_allocator = TestAllocator();
  auto request_group =
      MakeResourceGroup(kAmdxdnaChainCommandCacheDefaultMaxChildCommands + 1);

  EXPECT_EQ(iree_hal_amdxdna_chain_command_cache_allocate_entry(
                &cache, &request_group, /*max_slots=*/24),
            nullptr);

  iree_hal_amdxdna_chain_group_deinitialize(TestAllocator(), &request_group);
}

TEST(ChainCommandCacheTest, AllocateEntryEvictsLruToFitResourceBudget) {
  iree_hal_amdxdna_device_chain_command_cache_t cache = {};
  cache.host_allocator = TestAllocator();
  cache.entry_count = 2;
  iree_hal_amdxdna_chain_group_initialize(&cache.entries[0].group);
  iree_hal_amdxdna_chain_group_initialize(&cache.entries[1].group);
  SetResourceEntry(&cache.entries[0],
                   kAmdxdnaChainCommandCacheDefaultMaxChildCommands - 1,
                   /*last_use=*/1);
  SetResourceEntry(&cache.entries[1], 1, /*last_use=*/2);
  auto request_group = MakeResourceGroup(1);

  iree_hal_amdxdna_chain_command_cache_entry_t* entry =
      iree_hal_amdxdna_chain_command_cache_allocate_entry(
          &cache, &request_group, /*max_slots=*/24);

  EXPECT_EQ(entry, &cache.entries[0]);
  EXPECT_EQ(cache.entries[1].group.cmd_count, 1u);

  iree_hal_amdxdna_chain_group_deinitialize(TestAllocator(), &request_group);
  for (iree_host_size_t i = 0; i < cache.entry_count; ++i) {
    iree_hal_amdxdna_chain_group_deinitialize(TestAllocator(),
                                              &cache.entries[i].group);
  }
}

TEST(ChainCommandCacheTest, AllocateEntryUsesConfiguredChildBudget) {
  iree_hal_amdxdna_device_chain_command_cache_t cache = {};
  cache.host_allocator = TestAllocator();
  cache.max_child_commands =
      kAmdxdnaChainCommandCacheDefaultMaxChildCommands + 128;
  auto request_group = MakeResourceGroup(cache.max_child_commands);

  EXPECT_NE(iree_hal_amdxdna_chain_command_cache_allocate_entry(
                &cache, &request_group, /*max_slots=*/24),
            nullptr);

  iree_hal_amdxdna_chain_group_deinitialize(TestAllocator(), &request_group);
  for (iree_host_size_t i = 0; i < cache.entry_count; ++i) {
    iree_hal_amdxdna_chain_group_deinitialize(TestAllocator(),
                                              &cache.entries[i].group);
  }
}

}  // namespace
