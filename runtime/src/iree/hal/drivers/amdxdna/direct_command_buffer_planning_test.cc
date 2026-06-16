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

TEST(ApplyPatchTableTest, DropsLowTwoBitsOfDescriptorAddress) {
  std::vector<uint32_t> ctrl(8, 0);
  // arg_plus 0x13 makes the computed base non-4-aligned; bd[1] must mask the
  // low two bits to keep the descriptor address word-aligned.
  std::vector<uint32_t> patches = {0u, 0u, 0x13u};
  uint64_t args[] = {0x1000u};
  EXPECT_TRUE(iree_hal_amdxdna_apply_patch_table(
      ctrl.data(), ctrl.size(), patches.data(), patches.size(), args, 1));
  const uint64_t base = 0x1000u + 0x13u + kDdrAieAddrOffset;  // 0x80001013
  EXPECT_EQ(ctrl[1], static_cast<uint32_t>(base & 0xFFFFFFFC));
  EXPECT_EQ(ctrl[1] & 0x3u, 0u);
  EXPECT_EQ(ctrl[2], static_cast<uint32_t>(base >> 32));
}

}  // namespace
