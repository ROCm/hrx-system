// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/direct_command_buffer_planning.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

constexpr uint64_t kDdrAieAddrOffset = 0x80000000ULL;
constexpr uint32_t kWrite32ConstantSentinel = 0xA1EC0000u;

// --- iree_hal_amdxdna_txn_op_size --------------------------------------------

// Builds a byte buffer of `total` bytes with opcode `op` at offset 0 and the
// 32-bit value `size_field` written at `size_off`.
std::vector<uint8_t> MakeOp(uint8_t op, size_t total, size_t size_off,
                            uint32_t size_field) {
  std::vector<uint8_t> b(total, 0);
  b[0] = op;
  if (size_off + sizeof(uint32_t) <= total) {
    std::memcpy(b.data() + size_off, &size_field, sizeof(size_field));
  }
  return b;
}

TEST(TxnOpSizeTest, Write32ReadsSizeAtOffset20) {
  auto b = MakeOp(/*op=*/0, /*total=*/24, /*size_off=*/20, /*size=*/24);
  EXPECT_EQ(iree_hal_amdxdna_txn_op_size(b.data(), b.size(), 0), 24u);
}

TEST(TxnOpSizeTest, BlockWriteReadsSizeAtOffset12) {
  auto b = MakeOp(/*op=*/1, /*total=*/16, /*size_off=*/12, /*size=*/16);
  EXPECT_EQ(iree_hal_amdxdna_txn_op_size(b.data(), b.size(), 0), 16u);
}

TEST(TxnOpSizeTest, Ops3And4ReadSizeAtOffset24) {
  auto b3 = MakeOp(/*op=*/3, /*total=*/28, /*size_off=*/24, /*size=*/28);
  EXPECT_EQ(iree_hal_amdxdna_txn_op_size(b3.data(), b3.size(), 0), 28u);
  auto b4 = MakeOp(/*op=*/4, /*total=*/40, /*size_off=*/24, /*size=*/40);
  EXPECT_EQ(iree_hal_amdxdna_txn_op_size(b4.data(), b4.size(), 0), 40u);
}

TEST(TxnOpSizeTest, CustomOpReadsSizeAtOffset4) {
  auto b = MakeOp(/*op=*/200, /*total=*/8, /*size_off=*/4, /*size=*/8);
  EXPECT_EQ(iree_hal_amdxdna_txn_op_size(b.data(), b.size(), 0), 8u);
}

TEST(TxnOpSizeTest, UnknownSmallOpIsFourBytes) {
  std::vector<uint8_t> b(8, 0);
  b[0] = 2;  // not 0/1/3/4 and < 128.
  EXPECT_EQ(iree_hal_amdxdna_txn_op_size(b.data(), b.size(), 0), 4u);
}

TEST(TxnOpSizeTest, TruncatedOrOutOfBoundsReturnsZero) {
  auto b = MakeOp(/*op=*/0, /*total=*/10, /*size_off=*/0, /*size=*/0);
  EXPECT_EQ(iree_hal_amdxdna_txn_op_size(b.data(), b.size(), 0),
            0u);  // WRITE32
  EXPECT_EQ(iree_hal_amdxdna_txn_op_size(b.data(), b.size(), b.size()), 0u);
}

// --- iree_hal_amdxdna_patch_write32_constants --------------------------------

// Builds a one-op TXN with a single WRITE32 carrying `value` at the patch slot.
std::vector<uint32_t> MakeWrite32Txn(uint32_t value, size_t total_bytes = 40) {
  std::vector<uint32_t> txn(total_bytes / sizeof(uint32_t), 0);
  txn[2] = 1;  // num_ops
  uint8_t* b = reinterpret_cast<uint8_t*>(txn.data());
  b[16] = 0;                                        // WRITE32 opcode at p=16
  std::memcpy(b + 16 + 16, &value, sizeof(value));  // patch slot at p+16
  uint32_t op_size = 24;                            // WRITE32 op size
  std::memcpy(b + 16 + 20, &op_size, sizeof(op_size));  // size at p+20
  return txn;
}

TEST(PatchWrite32ConstantsTest, ReplacesSentinelWithConstant) {
  auto txn = MakeWrite32Txn(kWrite32ConstantSentinel | 1u);  // constant index 1
  std::vector<uint32_t> constants = {0xDEAD0000u, 0xCAFEBABEu};
  IREE_ASSERT_OK(iree_hal_amdxdna_patch_write32_constants(
      txn.data(), txn.size(),
      iree_make_const_byte_span(constants.data(),
                                constants.size() * sizeof(uint32_t))));
  uint32_t patched = 0;
  std::memcpy(&patched, reinterpret_cast<uint8_t*>(txn.data()) + 32,
              sizeof(patched));
  EXPECT_EQ(patched, 0xCAFEBABEu);
}

TEST(PatchWrite32ConstantsTest, LeavesNonSentinelUntouched) {
  auto txn = MakeWrite32Txn(0x12345678u);  // not a sentinel
  std::vector<uint32_t> constants = {0xCAFEBABEu};
  IREE_ASSERT_OK(iree_hal_amdxdna_patch_write32_constants(
      txn.data(), txn.size(),
      iree_make_const_byte_span(constants.data(),
                                constants.size() * sizeof(uint32_t))));
  uint32_t value = 0;
  std::memcpy(&value, reinterpret_cast<uint8_t*>(txn.data()) + 32,
              sizeof(value));
  EXPECT_EQ(value, 0x12345678u);
}

TEST(PatchWrite32ConstantsTest, RejectsOutOfRangeConstantIndex) {
  auto txn = MakeWrite32Txn(kWrite32ConstantSentinel | 7u);  // index 7
  std::vector<uint32_t> constants = {0xCAFEBABEu};           // only index 0
  iree_status_t status = iree_hal_amdxdna_patch_write32_constants(
      txn.data(), txn.size(),
      iree_make_const_byte_span(constants.data(),
                                constants.size() * sizeof(uint32_t)));
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

TEST(PatchWrite32ConstantsTest, RejectsMalformedTruncatedOp) {
  // num_ops=1 but only 20 bytes total: the WRITE32 op needs 24 -> size 0.
  std::vector<uint32_t> txn(5, 0);
  txn[2] = 1;
  reinterpret_cast<uint8_t*>(txn.data())[16] = 0;  // WRITE32 at p=16
  iree_status_t status = iree_hal_amdxdna_patch_write32_constants(
      txn.data(), txn.size(), iree_make_const_byte_span(nullptr, 0));
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

TEST(PatchWrite32ConstantsTest, ShortTxnIsNoOp) {
  std::vector<uint32_t> txn = {0, 0, 0};  // < 4 words
  IREE_ASSERT_OK(iree_hal_amdxdna_patch_write32_constants(
      txn.data(), txn.size(), iree_make_const_byte_span(nullptr, 0)));
}

TEST(PatchWrite32ConstantsTest, PrecomputedListMatchesScanner) {
  auto scanned_txn = MakeWrite32Txn(kWrite32ConstantSentinel | 1u);
  auto listed_txn = scanned_txn;
  std::vector<uint32_t> constants = {0xDEAD0000u, 0xCAFEBABEu};
  iree_hal_amdxdna_write32_constant_patch_list_t patch_list;
  IREE_ASSERT_OK(iree_hal_amdxdna_build_write32_constant_patch_list(
      iree_allocator_system(), listed_txn.data(), listed_txn.size(),
      &patch_list));
  EXPECT_EQ(patch_list.count, 1u);
  EXPECT_EQ(patch_list.data[0].byte_offset, 32u);
  EXPECT_EQ(patch_list.data[0].constant_index, 1u);

  IREE_ASSERT_OK(iree_hal_amdxdna_patch_write32_constants(
      scanned_txn.data(), scanned_txn.size(),
      iree_make_const_byte_span(constants.data(),
                                constants.size() * sizeof(uint32_t))));
  IREE_ASSERT_OK(iree_hal_amdxdna_patch_write32_constants_with_list(
      listed_txn.data(), listed_txn.size(), &patch_list,
      iree_make_const_byte_span(constants.data(),
                                constants.size() * sizeof(uint32_t))));
  EXPECT_EQ(listed_txn, scanned_txn);
  iree_hal_amdxdna_write32_constant_patch_list_deinitialize(
      iree_allocator_system(), &patch_list);
}

TEST(PatchWrite32ConstantsTest, PrecomputedListRejectsOutOfRangeConstantIndex) {
  auto txn = MakeWrite32Txn(kWrite32ConstantSentinel | 7u);
  std::vector<uint32_t> constants = {0xCAFEBABEu};
  iree_hal_amdxdna_write32_constant_patch_list_t patch_list;
  IREE_ASSERT_OK(iree_hal_amdxdna_build_write32_constant_patch_list(
      iree_allocator_system(), txn.data(), txn.size(), &patch_list));
  iree_status_t status = iree_hal_amdxdna_patch_write32_constants_with_list(
      txn.data(), txn.size(), &patch_list,
      iree_make_const_byte_span(constants.data(),
                                constants.size() * sizeof(uint32_t)));
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
  iree_hal_amdxdna_write32_constant_patch_list_deinitialize(
      iree_allocator_system(), &patch_list);
}

TEST(PatchWrite32ConstantsTest, PrecomputedListRejectsMalformedTruncatedOp) {
  std::vector<uint32_t> txn(5, 0);
  txn[2] = 1;
  reinterpret_cast<uint8_t*>(txn.data())[16] = 0;  // WRITE32 at p=16
  iree_hal_amdxdna_write32_constant_patch_list_t patch_list;
  iree_status_t status = iree_hal_amdxdna_build_write32_constant_patch_list(
      iree_allocator_system(), txn.data(), txn.size(), &patch_list);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

// --- iree_hal_amdxdna_build_host_patch_table --------------------------------

std::vector<uint32_t> MakeHostPatchTxn(uint32_t register_address,
                                       uint32_t argument_index,
                                       uint32_t argument_addend) {
  constexpr uint32_t kBlockWriteSize = 32;
  constexpr uint32_t kDdrPatchSize = 44;
  std::vector<uint32_t> txn((16 + kBlockWriteSize + kDdrPatchSize) / 4, 0);
  txn[2] = 2;
  uint8_t* bytes = reinterpret_cast<uint8_t*>(txn.data());
  size_t offset = 16;
  bytes[offset] = 1;
  std::memcpy(bytes + offset + 8, &register_address, sizeof(uint32_t));
  std::memcpy(bytes + offset + 12, &kBlockWriteSize, sizeof(uint32_t));
  offset += kBlockWriteSize;
  bytes[offset] = 0x81;
  std::memcpy(bytes + offset + 4, &kDdrPatchSize, sizeof(uint32_t));
  std::memcpy(bytes + offset + 24, &register_address, sizeof(uint32_t));
  std::memcpy(bytes + offset + 32, &argument_index, sizeof(uint32_t));
  std::memcpy(bytes + offset + 40, &argument_addend, sizeof(uint32_t));
  return txn;
}

TEST(BuildHostPatchTableTest, DerivesSemanticPatchTriple) {
  auto txn = MakeHostPatchTxn(/*register_address=*/0x1D000u,
                              /*argument_index=*/2u,
                              /*argument_addend=*/0x40u);
  iree_hal_amdxdna_host_patch_table_t table;
  IREE_ASSERT_OK(iree_hal_amdxdna_build_host_patch_table(
      iree_allocator_system(),
      iree_make_const_byte_span(txn.data(), txn.size() * sizeof(uint32_t)),
      &table));
  ASSERT_EQ(table.count, 3u);
  EXPECT_EQ(table.data[0], 32u);
  EXPECT_EQ(table.data[1], 2u);
  EXPECT_EQ(table.data[2], 0x40u);
  iree_hal_amdxdna_host_patch_table_deinitialize(iree_allocator_system(),
                                                  &table);
}

TEST(BuildHostPatchTableTest, RejectsPatchWithoutMatchingBlockWrite) {
  auto txn = MakeHostPatchTxn(/*register_address=*/0x1D000u,
                              /*argument_index=*/0u,
                              /*argument_addend=*/0u);
  uint32_t different_register = 0x201D000u;
  std::memcpy(reinterpret_cast<uint8_t*>(txn.data()) + 16 + 32 + 24,
              &different_register, sizeof(uint32_t));
  iree_hal_amdxdna_host_patch_table_t table;
  iree_status_t status = iree_hal_amdxdna_build_host_patch_table(
      iree_allocator_system(),
      iree_make_const_byte_span(txn.data(), txn.size() * sizeof(uint32_t)),
      &table);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
  iree_hal_amdxdna_host_patch_table_deinitialize(iree_allocator_system(),
                                                  &table);
}

TEST(BuildHostPatchTableTest, RejectsImpossibleOperationCount) {
  std::vector<uint32_t> txn(4, 0);
  txn[2] = 1;
  iree_hal_amdxdna_host_patch_table_t table;
  iree_status_t status = iree_hal_amdxdna_build_host_patch_table(
      iree_allocator_system(),
      iree_make_const_byte_span(txn.data(), txn.size() * sizeof(uint32_t)),
      &table);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

TEST(BuildHostPatchTableTest, RejectsNullOutputTable) {
  auto txn = MakeHostPatchTxn(/*register_address=*/0x1D000u,
                              /*argument_index=*/0u,
                              /*argument_addend=*/0u);
  iree_status_t status = iree_hal_amdxdna_build_host_patch_table(
      iree_allocator_system(),
      iree_make_const_byte_span(txn.data(), txn.size() * sizeof(uint32_t)),
      nullptr);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

TEST(BuildHostPatchTableTest, RejectsTrailingTransactionData) {
  auto txn = MakeHostPatchTxn(/*register_address=*/0x1D000u,
                              /*argument_index=*/0u,
                              /*argument_addend=*/0u);
  txn.push_back(0);
  iree_hal_amdxdna_host_patch_table_t table;
  iree_status_t status = iree_hal_amdxdna_build_host_patch_table(
      iree_allocator_system(),
      iree_make_const_byte_span(txn.data(), txn.size() * sizeof(uint32_t)),
      &table);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
  iree_hal_amdxdna_host_patch_table_deinitialize(iree_allocator_system(),
                                                  &table);
}

// --- iree_hal_amdxdna_apply_patch_table --------------------------------------

TEST(ApplyPatchTableTest, WritesShimDmaAddressIntoDescriptor) {
  std::vector<uint32_t> ctrl(8,
                             0);  // bd at byte 0: bd[1]=ctrl[1], bd[2]=ctrl[2]
  std::vector<uint32_t> patches = {/*offset=*/0u, /*arg_idx=*/0u,
                                   /*arg_plus=*/0x10u};
  uint64_t args[] = {0x1000u};
  EXPECT_TRUE(iree_hal_amdxdna_apply_patch_table(
      ctrl.data(), ctrl.size(), patches.data(), patches.size(), args, 1));
  // base = 0 + args[0] + arg_plus + AIE_DDR_offset.
  const uint64_t base = 0x1000u + 0x10u + kDdrAieAddrOffset;
  EXPECT_EQ(ctrl[1], static_cast<uint32_t>(base & 0xFFFFFFFC));
  EXPECT_EQ(ctrl[2], static_cast<uint32_t>(base >> 32));
}

TEST(ApplyPatchTableTest, SplitsHighAddressBitsIntoBd2) {
  std::vector<uint32_t> ctrl(8, 0);
  std::vector<uint32_t> patches = {0u, 0u, 0u};
  uint64_t args[] = {0x100000000ull};  // 4 GiB -> exercises the high 16 bits
  EXPECT_TRUE(iree_hal_amdxdna_apply_patch_table(
      ctrl.data(), ctrl.size(), patches.data(), patches.size(), args, 1));
  const uint64_t base = 0x100000000ull + kDdrAieAddrOffset;  // 0x1_8000_0000
  EXPECT_EQ(ctrl[1], static_cast<uint32_t>(base & 0xFFFFFFFC));
  EXPECT_EQ(ctrl[2], static_cast<uint32_t>(base >> 32));  // == 1
}

TEST(ApplyPatchTableTest, RejectsNonTripleTable) {
  std::vector<uint32_t> ctrl(8, 0);
  std::vector<uint32_t> patches = {0u, 0u};  // not a multiple of 3
  uint64_t args[] = {0u};
  EXPECT_FALSE(iree_hal_amdxdna_apply_patch_table(
      ctrl.data(), ctrl.size(), patches.data(), patches.size(), args, 1));
}

TEST(ApplyPatchTableTest, RejectsArgIndexOutOfRange) {
  std::vector<uint32_t> ctrl(8, 0);
  std::vector<uint32_t> patches = {0u, 5u, 0u};  // arg_idx 5 >= arg_count
  uint64_t args[] = {0u};
  EXPECT_FALSE(iree_hal_amdxdna_apply_patch_table(
      ctrl.data(), ctrl.size(), patches.data(), patches.size(), args, 1));
}

TEST(ApplyPatchTableTest, RejectsOutOfBoundsOrMisalignedOffset) {
  std::vector<uint32_t> ctrl(2, 0);  // total 8 bytes
  uint64_t args[] = {0u};
  std::vector<uint32_t> oob = {0u, 0u, 0u};  // offset+12 = 12 > 8
  EXPECT_FALSE(iree_hal_amdxdna_apply_patch_table(
      ctrl.data(), ctrl.size(), oob.data(), oob.size(), args, 1));
  std::vector<uint32_t> big(8, 0);
  std::vector<uint32_t> misaligned = {2u, 0u, 0u};  // offset & 0x3 != 0
  EXPECT_FALSE(iree_hal_amdxdna_apply_patch_table(
      big.data(), big.size(), misaligned.data(), misaligned.size(), args, 1));
}

TEST(ApplyPatchTableTest, RejectsUnalignedDescriptorAddress) {
  std::vector<uint32_t> ctrl(8, 0);
  // arg_plus 0x13 makes the computed base non-4-aligned; fail closed instead of
  // silently rounding the descriptor onto a neighboring address.
  std::vector<uint32_t> patches = {0u, 0u, 0x13u};
  uint64_t args[] = {0x1000u};
  EXPECT_FALSE(iree_hal_amdxdna_apply_patch_table(
      ctrl.data(), ctrl.size(), patches.data(), patches.size(), args, 1));
}

TEST(ApplyPatchTableTest, RejectsOverflowingDescriptorAddress) {
  std::vector<uint32_t> ctrl(8, 0);
  std::vector<uint32_t> patches = {0u, 0u, 0u};
  uint64_t args[] = {UINT64_MAX};
  EXPECT_FALSE(iree_hal_amdxdna_apply_patch_table(
      ctrl.data(), ctrl.size(), patches.data(), patches.size(), args, 1));
}

TEST(ApplyPatchTableTest, AddsAieApertureOffsetForAllArgIndices) {
  // The HRX host patch table carries raw buffer-relative byte offsets. The
  // runtime patcher owns conversion to AIE-visible addresses, so every patched
  // BD gets the DDR aperture offset regardless of arg index.
  std::vector<uint32_t> ctrl(8, 0);  // bd A at byte 0, bd B at byte 16.
  std::vector<uint32_t> patches = {/*offset=*/0u, /*arg_idx=*/4u,
                                   /*arg_plus=*/0u,
                                   /*offset=*/16u, /*arg_idx=*/5u,
                                   /*arg_plus=*/0u};
  uint64_t args[] = {0u, 0u, 0u, 0u, 0x1000u, 0x2000u};
  EXPECT_TRUE(iree_hal_amdxdna_apply_patch_table(
      ctrl.data(), ctrl.size(), patches.data(), patches.size(), args, 6));
  const uint64_t base4 = 0x1000u + 0u + kDdrAieAddrOffset;
  EXPECT_EQ(ctrl[1], static_cast<uint32_t>(base4 & 0xFFFFFFFC));
  EXPECT_EQ(ctrl[2], static_cast<uint32_t>(base4 >> 32));
  const uint64_t base5 = 0x2000u + kDdrAieAddrOffset;
  EXPECT_EQ(ctrl[5], static_cast<uint32_t>(base5 & 0xFFFFFFFC));
  EXPECT_EQ(ctrl[6], static_cast<uint32_t>(base5 >> 32));
}

TEST(ApplyPatchTableTest, DoesNotDoubleCountBakedSubBufferOffset) {
  // Regression for the multi-worker (e.g. transform_parallel) shim-DMA bug: for
  // a BD addressing a sub-range of a host buffer, the producer may bake the
  // intra-buffer byte offset into the BD address word while also emitting the
  // same offset as the DDR_PATCH addend (arg_plus). The patcher must apply the
  // offset once (buffer_base + arg_plus), overwriting the baked value.
  std::vector<uint32_t> ctrl(8, 0);
  ctrl[1] = 0x200u;  // baked intra-buffer offset (128 int32 elements).
  std::vector<uint32_t> patches = {/*offset=*/0u, /*arg_idx=*/0u,
                                   /*arg_plus=*/0x200u};
  uint64_t args[] = {0x1000u};
  EXPECT_TRUE(iree_hal_amdxdna_apply_patch_table(
      ctrl.data(), ctrl.size(), patches.data(), patches.size(), args, 1));
  const uint64_t base = 0x1000u + 0x200u + kDdrAieAddrOffset;  // one offset.
  EXPECT_EQ(ctrl[1], static_cast<uint32_t>(base & 0xFFFFFFFC));
  EXPECT_EQ(ctrl[2], static_cast<uint32_t>(base >> 32));
}

// --- iree_hal_amdxdna_patch_dynamic_fields_from_template ---------------------

TEST(PatchDynamicFieldsFromTemplateTest, RewritesConstantsAndPatchTable) {
  std::vector<uint32_t> templ = MakeWrite32Txn(kWrite32ConstantSentinel | 1u);
  templ.resize(12, 0);
  // Put a BD at byte offset 40. The high 16 bits of bd[2] are descriptor
  // metadata and must be preserved while the low 48 address bits are rewritten.
  templ[11] = 0xABCD0000u;
  std::vector<uint32_t> ctrl = templ;
  std::vector<uint32_t> patches = {/*offset=*/36u, /*arg_idx=*/0u,
                                   /*arg_plus=*/0x20u};
  std::vector<uint32_t> constants = {0xAAAA0000u, 0x12345678u};
  uint64_t args[] = {0x2000u};

  IREE_ASSERT_OK(iree_hal_amdxdna_patch_dynamic_fields_from_template(
      ctrl.data(), templ.data(), ctrl.size(), /*constant_patches=*/nullptr,
      iree_make_const_byte_span(constants.data(),
                                constants.size() * sizeof(uint32_t)),
      patches.data(), patches.size(), args, 1));

  uint32_t patched_constant = 0;
  std::memcpy(&patched_constant, reinterpret_cast<uint8_t*>(ctrl.data()) + 32,
              sizeof(patched_constant));
  EXPECT_EQ(patched_constant, 0x12345678u);
  const uint64_t base = 0x2000u + 0x20u + kDdrAieAddrOffset;
  EXPECT_EQ(ctrl[10], static_cast<uint32_t>(base & 0xFFFFFFFC));
  EXPECT_EQ(ctrl[11], 0xABCD0000u | static_cast<uint32_t>(base >> 32));
}

TEST(PatchDynamicFieldsFromTemplateTest,
     RewritesConstantsWithPrecomputedListAndPatchTable) {
  std::vector<uint32_t> templ = MakeWrite32Txn(kWrite32ConstantSentinel | 1u);
  templ.resize(12, 0);
  templ[11] = 0xABCD0000u;
  std::vector<uint32_t> ctrl = templ;
  std::vector<uint32_t> patches = {/*offset=*/36u, /*arg_idx=*/0u,
                                   /*arg_plus=*/0x20u};
  std::vector<uint32_t> constants = {0xAAAA0000u, 0x12345678u};
  uint64_t args[] = {0x2000u};

  iree_hal_amdxdna_write32_constant_patch_list_t patch_list;
  IREE_ASSERT_OK(iree_hal_amdxdna_build_write32_constant_patch_list(
      iree_allocator_system(), templ.data(), templ.size(), &patch_list));
  IREE_ASSERT_OK(iree_hal_amdxdna_patch_dynamic_fields_from_template(
      ctrl.data(), templ.data(), ctrl.size(), &patch_list,
      iree_make_const_byte_span(constants.data(),
                                constants.size() * sizeof(uint32_t)),
      patches.data(), patches.size(), args, 1));

  uint32_t patched_constant = 0;
  std::memcpy(&patched_constant, reinterpret_cast<uint8_t*>(ctrl.data()) + 32,
              sizeof(patched_constant));
  EXPECT_EQ(patched_constant, 0x12345678u);
  const uint64_t base = 0x2000u + 0x20u + kDdrAieAddrOffset;
  EXPECT_EQ(ctrl[10], static_cast<uint32_t>(base & 0xFFFFFFFC));
  EXPECT_EQ(ctrl[11], 0xABCD0000u | static_cast<uint32_t>(base >> 32));
  iree_hal_amdxdna_write32_constant_patch_list_deinitialize(
      iree_allocator_system(), &patch_list);
}

TEST(PatchDynamicFieldsFromTemplateTest, RepeatedRewriteOverwritesBakedAddress) {
  std::vector<uint32_t> templ(8, 0);
  // The template's BD address word carries a stale/baked offset (0x40). The
  // patcher must OVERWRITE it with buffer_base + arg_plus, not accumulate onto
  // it: firmware/XRT ignore the baked value, and the MLIR-AIE compiler already
  // supplies the intra-buffer offset via arg_plus. This also guards idempotency
  // across repeated rewrites of the same cached destination.
  templ[1] = 0x40u;
  std::vector<uint32_t> ctrl = templ;
  std::vector<uint32_t> patches = {0u, 0u, 0u};
  uint64_t first_args[] = {0x1000u};
  uint64_t second_args[] = {0x2000u};

  IREE_ASSERT_OK(iree_hal_amdxdna_patch_dynamic_fields_from_template(
      ctrl.data(), templ.data(), ctrl.size(), /*constant_patches=*/nullptr,
      iree_make_const_byte_span(nullptr, 0), patches.data(), patches.size(),
      first_args, 1));
  IREE_ASSERT_OK(iree_hal_amdxdna_patch_dynamic_fields_from_template(
      ctrl.data(), templ.data(), ctrl.size(), /*constant_patches=*/nullptr,
      iree_make_const_byte_span(nullptr, 0), patches.data(), patches.size(),
      second_args, 1));

  // Baked 0x40 is discarded, not added.
  const uint64_t expected = 0x2000u + kDdrAieAddrOffset;
  EXPECT_EQ(ctrl[1], static_cast<uint32_t>(expected & 0xFFFFFFFC));
  EXPECT_EQ(ctrl[2], static_cast<uint32_t>(expected >> 32));
}

TEST(PatchDynamicFieldsFromTemplateTest,
     AddsAieApertureOffsetForAllArgIndices) {
  // Same raw host-patch-table ABI as the apply_patch_table test, via the
  // template path: arg_plus is always buffer-relative and the patcher always
  // adds the AIE DDR aperture offset.
  std::vector<uint32_t> templ(8, 0);
  std::vector<uint32_t> ctrl = templ;
  std::vector<uint32_t> patches = {/*offset=*/0u, /*arg_idx=*/4u,
                                   /*arg_plus=*/0u,
                                   /*offset=*/16u, /*arg_idx=*/5u,
                                   /*arg_plus=*/0u};
  uint64_t args[] = {0u, 0u, 0u, 0u, 0x1000u, 0x2000u};

  IREE_ASSERT_OK(iree_hal_amdxdna_patch_dynamic_fields_from_template(
      ctrl.data(), templ.data(), ctrl.size(), /*constant_patches=*/nullptr,
      iree_make_const_byte_span(nullptr, 0), patches.data(), patches.size(),
      args, 6));

  const uint64_t base4 = 0x1000u + kDdrAieAddrOffset;
  EXPECT_EQ(ctrl[1], static_cast<uint32_t>(base4 & 0xFFFFFFFC));
  EXPECT_EQ(ctrl[2], static_cast<uint32_t>(base4 >> 32));
  const uint64_t base5 = 0x2000u + kDdrAieAddrOffset;
  EXPECT_EQ(ctrl[5], static_cast<uint32_t>(base5 & 0xFFFFFFFC));
  EXPECT_EQ(ctrl[6], static_cast<uint32_t>(base5 >> 32));
}

TEST(PatchDynamicFieldsFromTemplateTest, DoesNotDoubleCountBakedSubBufferOffset) {
  // Template-path counterpart of the apply_patch_table regression: the baked
  // sub-buffer offset in the template BD address must be overwritten, not added
  // to the DDR_PATCH addend that carries the same offset.
  std::vector<uint32_t> templ(8, 0);
  templ[1] = 0x200u;         // baked intra-buffer offset.
  templ[2] = 0xABCD0000u;    // bd[2] high 16 bits: BD control state, preserved.
  std::vector<uint32_t> ctrl = templ;
  std::vector<uint32_t> patches = {/*offset=*/0u, /*arg_idx=*/0u,
                                   /*arg_plus=*/0x200u};
  uint64_t args[] = {0x1000u};

  IREE_ASSERT_OK(iree_hal_amdxdna_patch_dynamic_fields_from_template(
      ctrl.data(), templ.data(), ctrl.size(), /*constant_patches=*/nullptr,
      iree_make_const_byte_span(nullptr, 0), patches.data(), patches.size(),
      args, 1));

  const uint64_t base = 0x1000u + 0x200u + kDdrAieAddrOffset;  // one offset.
  EXPECT_EQ(ctrl[1], static_cast<uint32_t>(base & 0xFFFFFFFC));
  EXPECT_EQ(ctrl[2], 0xABCD0000u | static_cast<uint32_t>(base >> 32));
}

TEST(PatchDynamicFieldsFromTemplateTest, RejectsMalformedPatchTable) {
  std::vector<uint32_t> templ(8, 0);
  std::vector<uint32_t> ctrl = templ;
  std::vector<uint32_t> patches = {0u, 0u};
  uint64_t args[] = {0u};

  iree_status_t status = iree_hal_amdxdna_patch_dynamic_fields_from_template(
      ctrl.data(), templ.data(), ctrl.size(), /*constant_patches=*/nullptr,
      iree_make_const_byte_span(nullptr, 0), patches.data(), patches.size(),
      args, 1);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

TEST(PatchDynamicFieldsFromTemplateTest, RejectsUnalignedDescriptorAddress) {
  std::vector<uint32_t> templ(8, 0);
  std::vector<uint32_t> ctrl = templ;
  std::vector<uint32_t> patches = {0u, 0u, 0x13u};
  uint64_t args[] = {0x1000u};

  iree_status_t status = iree_hal_amdxdna_patch_dynamic_fields_from_template(
      ctrl.data(), templ.data(), ctrl.size(), /*constant_patches=*/nullptr,
      iree_make_const_byte_span(nullptr, 0), patches.data(), patches.size(),
      args, 1);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

}  // namespace
