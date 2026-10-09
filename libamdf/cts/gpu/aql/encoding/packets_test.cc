// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/aql/encoding/packets.h"

#include <cstddef>

#include "gtest/gtest.h"

namespace {

TEST(AqlEncodingTest, BarrierSignalAddressesAndSystemScopes) {
  const auto packet = aql::Barrier(
      aql::BarrierType::kAnd, aql::HeaderBarrier::kEnabled,
      UINT64_C(0x1234567887654300), {UINT64_C(0x2345678998765400)});
  // HSA System Architecture 1.2, tables 2-4 through 2-6 and 2-9:
  // AND=3, barrier bit 8, SYSTEM=2 at bits 9 and 11; dependency byte 8,
  // completion byte 56. These literal words do not use the encoder's enums.
  const aql::Packet expected = {0x1503, 0, 0x98765400, 0x23456789, 0, 0,
                                0,      0, 0,          0,          0, 0,
                                0,      0, 0x87654300, 0x12345678};
  EXPECT_EQ(packet, expected);
  // ROCm 8d57824901ff, amd_hsa_signal.h, amd_signal_t.
  EXPECT_EQ(offsetof(aql::Signal, kind), 0u);
  EXPECT_EQ(offsetof(aql::Signal, value), 8u);
  EXPECT_EQ(sizeof(aql::Signal), 64u);
  EXPECT_EQ(alignof(aql::Signal), 64u);
}

TEST(AqlEncodingTest, BarrierAndEncodesAllFiveDependencies) {
  const auto packet =
      aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled,
                   UINT64_C(0x1234567887654300),
                   {UINT64_C(0x2345678998765400), UINT64_C(0x3456789aa9876500),
                    UINT64_C(0x456789abba987600), UINT64_C(0x56789abccba98700),
                    UINT64_C(0x6789abcddcba9800)});
  // HSA System Architecture 1.2 table 2-9 places five 64-bit handles at
  // bytes 8, 16, 24, 32 and 40; bytes 48 through 55 remain reserved zero.
  const aql::Packet expected = {0x1403,     0,          0x98765400, 0x23456789,
                                0xa9876500, 0x3456789a, 0xba987600, 0x456789ab,
                                0xcba98700, 0x56789abc, 0xdcba9800, 0x6789abcd,
                                0,          0,          0x87654300, 0x12345678};
  EXPECT_EQ(packet, expected);
}

TEST(AqlEncodingTest, BarrierOrEncodesSparseDependenciesAndNullCompletion) {
  const auto packet = aql::Barrier(
      aql::BarrierType::kOr, aql::HeaderBarrier::kDisabled, 0,
      {0, UINT64_C(0x1234567887654300), 0, UINT64_C(0x2345678998765400), 0});
  // HSA System Architecture 1.2 tables 2-4 and 2-10 define OR=5 with the
  // same layout as AND. Null dependency handles do not satisfy OR.
  const aql::Packet expected = {
      0x1405,     0,          0, 0, 0x87654300, 0x12345678, 0, 0,
      0x98765400, 0x23456789, 0, 0, 0,          0,          0, 0};
  EXPECT_EQ(packet, expected);
}

TEST(AqlEncodingTest, BarrierAndAllowsNullDependenciesAndCompletion) {
  const auto packet =
      aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kEnabled, 0);
  // HSA System Architecture 1.2 sections 2.9.2 and 2.9.8 permit no completion
  // signal and treat every null AND dependency as satisfied.
  const aql::Packet expected = {0x1503, 0, 0, 0, 0, 0, 0, 0,
                                0,      0, 0, 0, 0, 0, 0, 0};
  EXPECT_EQ(packet, expected);
}

TEST(AqlEncodingTest, BarrierOrReadinessHasNoImplicitFence) {
  const auto packet =
      aql::Barrier(aql::BarrierType::kOr, aql::HeaderBarrier::kDisabled, 0,
                   {UINT64_C(0x2345678998765400), UINT64_C(0x3456789aa9876500),
                    UINT64_C(0x456789abba987600), UINT64_C(0x56789abccba98700),
                    UINT64_C(0x6789abcddcba9800)},
                   {aql::FenceScope::kNone, aql::FenceScope::kNone});
  // HSA System Architecture 1.2 tables 2-4 and 2-10: OR=5, independent
  // NONE scopes, five complete handles and a null completion handle.
  const aql::Packet expected = {0x0005,     0,          0x98765400, 0x23456789,
                                0xa9876500, 0x3456789a, 0xba987600, 0x456789ab,
                                0xcba98700, 0x56789abc, 0xdcba9800, 0x6789abcd,
                                0,          0,          0,          0};
  EXPECT_EQ(packet, expected);
}

TEST(AqlEncodingTest, DependencyBarrierHasNoAdditionalCacheScopes) {
  const auto packet =
      aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled, 0,
                   {UINT64_C(0x2345678998765400)},
                   {aql::FenceScope::kNone, aql::FenceScope::kNone});
  // HSA System Architecture 1.2 table 2-4 encodes NONE as zero. The AND
  // dependency still blocks subsequent packet launches until satisfied.
  const aql::Packet expected = {0x0003, 0, 0x98765400, 0x23456789, 0, 0, 0, 0,
                                0,      0, 0,          0,          0, 0, 0, 0};
  EXPECT_EQ(packet, expected);
}

TEST(AqlEncodingTest, Dispatch1DEncodesGeometryResourcesAndSystemScopes) {
  const auto packet = aql::Dispatch(
      aql::HeaderBarrier::kDisabled, {1, {64, 1, 1}, {1024, 1, 1}}, 68, 128,
      UINT64_C(0x1234567887654300), UINT64_C(0x2345678998765400),
      UINT64_C(0x3456789aa9876500),
      {aql::FenceScope::kSystem, aql::FenceScope::kSystem});
  // ROCm 8d57824901ff hsa.h, hsa_kernel_dispatch_packet_t: dimensions at
  // setup bit 0, workgroup XYZ at bytes 4/6/8, grid XYZ at 12/16/20,
  // private/group bytes at 24/28, descriptor/kernarg/signal at 32/40/56.
  const aql::Packet expected = {0x00011402, 0x00010040, 0x00000001, 0x00000400,
                                0x00000001, 0x00000001, 0x00000044, 0x00000080,
                                0x87654300, 0x12345678, 0x98765400, 0x23456789,
                                0x00000000, 0x00000000, 0xa9876500, 0x3456789a};
  EXPECT_EQ(packet, expected);
}

TEST(AqlEncodingTest, Dispatch2DEncodesBothAxesAndInactiveZ) {
  const auto packet =
      aql::Dispatch(aql::HeaderBarrier::kDisabled, {2, {16, 4, 1}, {48, 8, 1}},
                    0, 0, UINT64_C(0x1234567887654300),
                    UINT64_C(0x2345678998765400), UINT64_C(0x3456789aa9876500));
  // HSA System Architecture 1.2 table 2-7: dimensions=2 at setup bits 0-1,
  // positive XY extents and both inactive Z sizes one. These literal words
  // distinguish 2D from the output-equivalent 3D dispatch with Z sizes one.
  const aql::Packet expected = {0x00021402, 0x00040010, 0x00000001, 0x00000030,
                                0x00000008, 0x00000001, 0x00000000, 0x00000000,
                                0x87654300, 0x12345678, 0x98765400, 0x23456789,
                                0x00000000, 0x00000000, 0xa9876500, 0x3456789a};
  EXPECT_EQ(packet, expected);
}

TEST(AqlEncodingTest, Dispatch3DEncodesEveryAxisAndReservedZeros) {
  const auto packet =
      aql::Dispatch(aql::HeaderBarrier::kDisabled, {3, {8, 4, 2}, {24, 8, 4}},
                    0, 0, UINT64_C(0x1234567887654300),
                    UINT64_C(0x2345678998765400), UINT64_C(0x3456789aa9876500));
  // HSA System Architecture 1.2 table 2-7: dimensions=3 at setup bits 0-1,
  // u16 workgroup XYZ at bytes 4/6/8 and u32 grid XYZ at bytes 12/16/20.
  // Byte 10 and the reserved u64 at byte 48 remain zero.
  const aql::Packet expected = {0x00031402, 0x00040008, 0x00000002, 0x00000018,
                                0x00000008, 0x00000004, 0x00000000, 0x00000000,
                                0x87654300, 0x12345678, 0x98765400, 0x23456789,
                                0x00000000, 0x00000000, 0xa9876500, 0x3456789a};
  EXPECT_EQ(packet, expected);
}

TEST(AqlEncodingTest, DispatchEncodesEarlierPacketBarrierAndSystemScopes) {
  const auto packet =
      aql::Dispatch(aql::HeaderBarrier::kEnabled, {1, {64, 1, 1}, {1024, 1, 1}},
                    0, 0, UINT64_C(0x1234567887654300),
                    UINT64_C(0x2345678998765400), UINT64_C(0x3456789aa9876500));
  // HSA System Architecture 1.2 tables 2-4 and 2-7: barrier bit 8 orders
  // earlier completion before this dispatch's SYSTEM acquire and execution.
  const aql::Packet expected = {0x00011502, 0x00010040, 0x00000001, 0x00000400,
                                0x00000001, 0x00000001, 0x00000000, 0x00000000,
                                0x87654300, 0x12345678, 0x98765400, 0x23456789,
                                0x00000000, 0x00000000, 0xa9876500, 0x3456789a};
  EXPECT_EQ(packet, expected);
}

TEST(AqlEncodingTest, DispatchChainUsesAgentDependencyAndSystemCompletion) {
  const auto producer = aql::Dispatch(
      aql::HeaderBarrier::kEnabled, {1, {64, 1, 1}, {1024, 1, 1}}, 0, 0,
      UINT64_C(0x1234567887654300), UINT64_C(0x2345678998765400), 0,
      {aql::FenceScope::kSystem, aql::FenceScope::kAgent});
  const auto consumer = aql::Dispatch(
      aql::HeaderBarrier::kEnabled, {1, {64, 1, 1}, {1024, 1, 1}}, 0, 0,
      UINT64_C(0x1234567887654300), UINT64_C(0x2345678998765440), 0,
      {aql::FenceScope::kAgent, aql::FenceScope::kSystem});
  const auto terminal =
      aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kEnabled,
                   UINT64_C(0x3456789aa9876500), {},
                   {aql::FenceScope::kNone, aql::FenceScope::kSystem});
  // HSA System Architecture 1.2 tables 2-4 through 2-7 and 2-9 encode
  // NONE/AGENT/SYSTEM as 0/1/2 at acquire bits 9-10 and release bits 11-12.
  // Every header enables barrier bit 8. The dispatches use distinct kernarg
  // slots 64 bytes apart and null completion handles; only the empty AND
  // carries the completion signal. Literal words include every reserved zero.
  const aql::Packet expected_producer = {
      0x00010d02, 0x00010040, 0x00000001, 0x00000400, 0x00000001, 0x00000001,
      0x00000000, 0x00000000, 0x87654300, 0x12345678, 0x98765400, 0x23456789,
      0x00000000, 0x00000000, 0x00000000, 0x00000000};
  const aql::Packet expected_consumer = {
      0x00011302, 0x00010040, 0x00000001, 0x00000400, 0x00000001, 0x00000001,
      0x00000000, 0x00000000, 0x87654300, 0x12345678, 0x98765440, 0x23456789,
      0x00000000, 0x00000000, 0x00000000, 0x00000000};
  const aql::Packet expected_terminal = {
      0x00001103, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0xa9876500, 0x3456789a};
  EXPECT_EQ(producer, expected_producer);
  EXPECT_EQ(consumer, expected_consumer);
  EXPECT_EQ(terminal, expected_terminal);
}

TEST(AqlEncodingTest, BarrierValueEncodesMaskedEpochAndLessThanCondition) {
  const auto packet = aql::BarrierValueLessThan(
      UINT64_C(0x2345678998765400), INT64_C(0x0000000200000001), INT64_MAX);
  // ROCm 8d57824901ff hsa_ext_amd.h: AMD format 2, signal/value/mask at
  // bytes 8/16/24, LT=2 at byte 32. The header sets barrier bit 8 and NONE
  // scopes; all reserved words and the unused completion handle stay zero.
  const aql::Packet expected = {0x00020100, 0x00000000, 0x98765400, 0x23456789,
                                0x00000001, 0x00000002, 0xffffffff, 0x7fffffff,
                                0x00000002, 0x00000000, 0x00000000, 0x00000000,
                                0x00000000, 0x00000000, 0x00000000, 0x00000000};
  EXPECT_EQ(packet, expected);
}

TEST(AqlEncodingTest, Gfx9CodeCacheInvalidateUsesBounded256ByteRange) {
  amdf_gpu_endpoint_info_t endpoint = {.gfx_ip = {9, 4, 2}};
  const auto commands =
      aql::CodeCacheInvalidate(endpoint, UINT64_C(0x1234567887654300), 1408);
  // ROCm 8d57824901ff amd_gpu_pm4.h: ACQUIRE_MEM opcode 0x58, seven
  // dwords, coherent actions at bits 18/23/27/29, base/size in 256 bytes.
  // InvalidateCodeCaches supplies zero poll interval. Linux soc15d.h agrees
  // on the action bits and base fields; no gfx10+ GCR word is appended.
  const std::vector<uint32_t> expected = {0xc0055800, 0x28840000, 0x00000006,
                                          0x00000000, 0x78876543, 0x00123456,
                                          0x00000000};
  EXPECT_EQ(commands, expected);
}

TEST(AqlEncodingTest, RdnaCodePublicationSelectsGenerationSpecificGcr) {
  for (uint32_t target : {110501u, 120000u, 120500u}) {
    SCOPED_TRACE(target);
    amdf_gpu_endpoint_info_t endpoint = {
        .gfx_ip = {target / 10000, (target / 100) % 100, target % 100}};
    const auto commands =
        aql::CodeCacheInvalidate(endpoint, UINT64_C(0x1234567887654300), 1408);
    // RDNA's eight-dword ACQUIRE_MEM appends GCR instead of COHER_CNTL.
    // Reserved fields and the GFX125x system scope follow the PM4 profile.
    const uint32_t gcr = target == 110501   ? 0xc3a1u
                         : target == 120000 ? 0xc181u
                                            : 0x1c1e1u;
    const std::vector<uint32_t> expected = {0xc0065800, 0,          6, 0,
                                            0x78876543, 0x00123456, 0, gcr};
    EXPECT_EQ(commands, expected);
  }
}

TEST(AqlEncodingTest, CodeCachePublicationOwnsIndirectBufferCompletion) {
  const auto packet = aql::IndirectBuffer(
      aql::HeaderBarrier::kDisabled, UINT64_C(0x123498765400), 7,
      UINT64_C(0x3456789aa9876500),
      {aql::FenceScope::kNone, aql::FenceScope::kNone});
  // ROCm 8d57824901ff AqlQueue::ExecutePM4: type0/format1; four-dword
  // INDIRECT_BUFFER opcode0x3f, 48-bit base, size7 and valid bit23; remaining
  // count0xa; eight reserved dwords; native completion signal at byte56.
  const aql::Packet expected = {0x00010000, 0xc0023f00, 0x98765400, 0x00001234,
                                0x00800007, 0x0000000a, 0x00000000, 0x00000000,
                                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                                0x00000000, 0x00000000, 0xa9876500, 0x3456789a};
  EXPECT_EQ(packet, expected);
}

TEST(AqlEncodingTest, DataCarrierOrdersSystemTransferCompletion) {
  const auto packet = aql::IndirectBuffer(
      aql::HeaderBarrier::kEnabled, UINT64_C(0xabcd98765400), 14,
      UINT64_C(0x3456789aa9876500),
      {aql::FenceScope::kSystem, aql::FenceScope::kSystem});
  // ROCm 8d57824901ff ExecutePM4 and amd_gpu_pm4.h: a 14-DWORD IB
  // carries VALID at bit 23, with an unshifted 48-bit address. HSA header
  // barrier bit 8 and SYSTEM at bits 9/11 precede vendor format 1.
  const aql::Packet expected = {0x00011500, 0xc0023f00, 0x98765400, 0x0000abcd,
                                0x0080000e, 0x0000000a, 0x00000000, 0x00000000,
                                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                                0x00000000, 0x00000000, 0xa9876500, 0x3456789a};
  EXPECT_EQ(packet, expected);
}

TEST(AqlEncodingTest, SingleXccHasNoExecutorPrefix) {
  std::array<uint32_t, 3> words = {0x12345678, 0xabcdef01, 0x98765432};
  const auto expected = words;
  EXPECT_EQ(aql::SingleExecutorPrefix(words.data(), 1, 12), 0u);
  EXPECT_EQ(words, expected);
}

TEST(AqlEncodingTest, MultiXccSelectsOneExecutorWithBodyCount) {
  // ROCm 8d57824901ff amd_gpu_pm4.h and Linux 50d05c7c76c9 soc15d.h:
  // PRED_EXEC opcode 0x23, two DWORDs; 14-bit body count and virtual-XCC
  // mask 1 at bit 24. These counts cover WRITE32/64 followed by COPY32/64.
  const std::array<uint32_t, 3> expected32 = {0xc0002300, 0x0100000b,
                                              0x98765432};
  const std::array<uint32_t, 3> expected64 = {0xc0002300, 0x0100000c,
                                              0x98765432};
  for (uint32_t xcc_count : {2u, 8u}) {
    SCOPED_TRACE(xcc_count);
    std::array<uint32_t, 3> words = {0, 0, 0x98765432};
    EXPECT_EQ(aql::SingleExecutorPrefix(words.data(), xcc_count, 11), 2u);
    EXPECT_EQ(words, expected32);
    EXPECT_EQ(aql::SingleExecutorPrefix(words.data(), xcc_count, 12), 2u);
    EXPECT_EQ(words, expected64);
  }
}

}  // namespace
