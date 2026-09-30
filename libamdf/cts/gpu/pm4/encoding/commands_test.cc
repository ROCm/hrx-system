// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/pm4/encoding/commands.h"

#include <array>
#include <initializer_list>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/cts/gpu/pm4/encoding/memory_commands.h"

namespace {

// Literal wire expectations come from PAL/Mesa memory commands and aqlprofile
// clock capture. No production constants or bitfield definitions are shared
// with the encoder; docs/reference/amd/gpu/pm4/memory-commands.md and
// docs/reference/amd/gpu/aql/profiling.md describe the source contracts.
template <size_t N>
void ExpectWords(const uint32_t* words,
                 const std::array<uint32_t, N>& expected) {
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(words[i], expected[i]) << "word " << i;
  }
}

const Pm4CommandProfile& Profile(uint32_t major, uint32_t minor) {
  amdf_gpu_endpoint_info_t info = {};
  info.gfx_ip = {major, minor, 0};
  return *Pm4CommandProfile::Find(info);
}

TEST(Pm4EncodingTest, TargetProfilesEncodeCacheAndRegisterDifferences) {
  const struct {
    // Compiler target family selecting the native compute representation.
    std::array<uint32_t, 2> target;
    // SET_SH_REG-relative RSRC3 address from the register definition.
    uint32_t resource3_offset;
    // Full acquire control including instruction invalidation.
    uint32_t acquire;
    // Release event plus generation-specific GCR in its native word.
    uint32_t release;
    // Isolated GL2 writeback with the target's scope.
    uint32_t writeback;
  } cases[] = {
      {{11, 0}, 0x228, 0xc3a1, 0x0030e528, 0x8000},
      {{11, 5}, 0x228, 0xc3a1, 0x0030e528, 0x8000},
      {{11, 7}, 0x228, 0xc3a1, 0x0030e528, 0x8000},
      {{12, 0}, 0x228, 0xc181, 0x00304528, 0x8000},
      {{12, 5}, 0x223, 0x1c1e1, 0x01706528, 0x8020},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(::testing::Message()
                 << test.target[0] << '.' << test.target[1]);
    std::array<uint32_t, 69> words;
    words.fill(0x24681357);
    const Pm4ComputeProgram program = {
        UINT64_C(0x0000123456789000), 0xe0af0000, 0x84, 0x30, 512, {128, 1, 1},
    };
    Pm4CommandWriter commands(words.data() + 1,
                              Profile(test.target[0], test.target[1]));
    commands.BindCompute(program, UINT64_C(0x00003456789abc00));
    commands.SystemBarrier();
    commands.ReleaseSystem32(UINT64_C(0x0000456789abcd00), 17);
    commands.WaitEndOfPipeAndWriteback(UINT64_C(0x0000456789abcd40), 19);
    ASSERT_EQ(commands.word_count(), 67u);
    // The binding preserves common register intervals and changes RSRC3 only.
    EXPECT_EQ(words[10], test.resource3_offset);
    EXPECT_EQ(words[11], 0x30u);
    EXPECT_EQ(words[8], 0x8084u);
    const std::array<uint32_t, 10> acquire = {
        0xc0004600, 0x407, 0xc0065800, 0,   UINT32_MAX,
        0xff,       0,     0,          0xa, test.acquire,
    };
    ExpectWords(words.data() + 27, acquire);
    const std::array<uint32_t, 8> release = {
        0xc0064900, test.release, 0x23010000, 0x89abcd00, 0x4567, 17, 0, 0,
    };
    ExpectWords(words.data() + 37, release);
    const std::array<uint32_t, 8> writeback = {
        0xc0065800, 0, UINT32_MAX, 0xff, 0, 0, 0xa, test.writeback,
    };
    ExpectWords(words.data() + 60, writeback);
    EXPECT_EQ(words.front(), 0x24681357u);
    EXPECT_EQ(words.back(), 0x24681357u);
  }
}

TEST(Pm4EncodingTest, OrderedDataAcquirePreservesTargetCachePolicy) {
  const struct {
    // Compiler target family selecting the native compute representation.
    std::array<uint32_t, 2> target;
    // Data acquire GCR with instruction invalidation disabled.
    uint32_t acquire;
  } cases[] = {
      {{11, 0}, 0xc3a0}, {{11, 5}, 0xc3a0},  {{11, 7}, 0xc3a0},
      {{12, 0}, 0xc180}, {{12, 5}, 0x1c1e0},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(::testing::Message()
                 << test.target[0] << '.' << test.target[1]);
    std::array<uint32_t, 10> words;
    words.fill(0x24681357);
    Pm4CommandWriter commands(words.data() + 1,
                              Profile(test.target[0], test.target[1]));
    commands.AcquireFromSystem();
    // One ACQUIRE_MEM, without CS_PARTIAL_FLUSH or an instruction invalidate.
    const std::array<uint32_t, 8> expected = {
        0xc0065800, 0, UINT32_MAX, 0xff, 0, 0, 0xa, test.acquire,
    };
    ASSERT_EQ(commands.word_count(), expected.size());
    ExpectWords(words.data() + 1, expected);
    EXPECT_EQ(words.front(), 0x24681357u);
    EXPECT_EQ(words.back(), 0x24681357u);
  }
}

TEST(Pm4EncodingTest, AtomicStoresPreserveWidthsAndClearUnusedFields) {
  std::array<uint32_t, 20> words;
  words.fill(0x24681357);
  Pm4CommandWriter commands(words.data() + 1, Profile(11, 0));
  commands.AtomicStore32(UINT64_C(0x000012345678903c), 0xfedcba98);
  commands.AtomicStore64(UINT64_C(0x00003456789abff0),
                         UINT64_C(0x13579bdf2468ace0));
  const std::array<uint32_t, 18> expected = {
      0xc0071e00, 0x07, 0x5678903c, 0x1234, 0xfedcba98, 0,          0, 0, 0,
      0xc0071e00, 0x27, 0x789abff0, 0x3456, 0x2468ace0, 0x13579bdf, 0, 0, 0,
  };
  ASSERT_EQ(commands.word_count(), expected.size());
  ExpectWords(words.data() + 1, expected);
  EXPECT_EQ(words.front(), 0x24681357u);
  EXPECT_EQ(words.back(), 0x24681357u);
}

TEST(Pm4EncodingTest, ComputeBindingPreservesNativeContextRegisters) {
  std::array<uint32_t, 27> words = {};
  words.back() = 0x24681357;
  const Pm4ComputeProgram program = {
      UINT64_C(0x0000123456789000), 0xe0af0000, 0x84, 0x20, 0, {64, 1, 1},
  };
  Pm4CommandWriter commands(words.data(), Profile(11, 0));
  commands.BindCompute(program, UINT64_C(0x00003456789abc00));
  // PAL's ordinary compute SET_SH_REG intervals contain only PGM_LO/HI,
  // RSRC1/2/3, RESOURCE_LIMITS, START/NUM_THREAD and two user-data words.
  // In particular, geometry ends at 0x2e09 before the profiling enables.
  const std::array<uint32_t, 26> expected = {
      0xc0027602, 0x20c,  0x34567890, 0x12,  0xc0027602, 0x212,
      0xe0af0000, 0x84,   0xc0017602, 0x228, 0x20,       0xc0017602,
      0x215,      0,      0xc0067602, 0x204, 0,          0,
      0,          64,     1,          1,     0xc0027602, 0x240,
      0x789abc00, 0x3456,
  };
  ASSERT_EQ(commands.word_count(), expected.size());
  ExpectWords(words.data(), expected);
  EXPECT_EQ(words.back(), 0x24681357u);
}

TEST(Pm4EncodingTest, ComputeBindingRealizesStaticLdsAndFourWavePolicy) {
  std::array<uint32_t, 28> words;
  words.fill(0x24681357);
  const Pm4ComputeProgram program = {
      UINT64_C(0x0000123456789000), 0xe0af0000, 0x84, 0x30, 512, {128, 1, 1},
  };
  Pm4CommandWriter commands(words.data() + 1, Profile(11, 0));
  commands.BindCompute(program, UINT64_C(0x00003456789abc00));
  // The compiler descriptor retains LDS_SIZE=0. PAL's HSA path derives one
  // 512-byte unit; the four-wave policy separately sets SIMD_DEST_CNTL bit 22.
  const std::array<uint32_t, 26> expected = {
      0xc0027602, 0x20c,      0x34567890, 0x12,  0xc0027602, 0x212,
      0xe0af0000, 0x8084,     0xc0017602, 0x228, 0x30,       0xc0017602,
      0x215,      0x00400000, 0xc0067602, 0x204, 0,          0,
      0,          128,        1,          1,     0xc0027602, 0x240,
      0x789abc00, 0x3456,
  };
  ASSERT_EQ(commands.word_count(), expected.size());
  ExpectWords(words.data() + 1, expected);
  EXPECT_EQ(words.front(), 0x24681357u);
  EXPECT_EQ(words.back(), 0x24681357u);
  EXPECT_EQ(program.resource2, 0x84u);
}

TEST(Pm4EncodingTest, ComputeBindingReplacesDynamicLdsRequirement) {
  std::array<uint32_t, 80> words;
  words.fill(0x24681357);
  Pm4ComputeProgram program = {
      UINT64_C(0x0000123456789000), 0xe0af0000, 0x84, 0x30, 512, {128, 1, 1},
  };
  Pm4CommandWriter commands(words.data() + 1, Profile(11, 0));
  for (uint32_t group_byte_length : {1024u, 2048u, 1024u}) {
    program.group_segment_byte_length = group_byte_length;
    commands.BindCompute(program, UINT64_C(0x00003456789abc00));
  }
  // Each complete binding replaces the launch requirement, including the
  // smaller final request. These words establish the emitted field values,
  // not the hardware's physical allocation or occupancy.
  const std::array<uint32_t, 78> expected = {
      0xc0027602, 0x20c,      0x34567890, 0x12,  0xc0027602, 0x212,
      0xe0af0000, 0x10084,    0xc0017602, 0x228, 0x30,       0xc0017602,
      0x215,      0x00400000, 0xc0067602, 0x204, 0,          0,
      0,          128,        1,          1,     0xc0027602, 0x240,
      0x789abc00, 0x3456,

      0xc0027602, 0x20c,      0x34567890, 0x12,  0xc0027602, 0x212,
      0xe0af0000, 0x20084,    0xc0017602, 0x228, 0x30,       0xc0017602,
      0x215,      0x00400000, 0xc0067602, 0x204, 0,          0,
      0,          128,        1,          1,     0xc0027602, 0x240,
      0x789abc00, 0x3456,

      0xc0027602, 0x20c,      0x34567890, 0x12,  0xc0027602, 0x212,
      0xe0af0000, 0x10084,    0xc0017602, 0x228, 0x30,       0xc0017602,
      0x215,      0x00400000, 0xc0067602, 0x204, 0,          0,
      0,          128,        1,          1,     0xc0027602, 0x240,
      0x789abc00, 0x3456,
  };
  ASSERT_EQ(commands.word_count(), expected.size());
  ExpectWords(words.data() + 1, expected);
  EXPECT_EQ(words.front(), 0x24681357u);
  EXPECT_EQ(words.back(), 0x24681357u);
  EXPECT_EQ(program.resource2, 0x84u);
}

TEST(Pm4EncodingTest, ComputeBindingSwitchesImmutableProgramsAndRestoresState) {
  std::array<uint32_t, 80> words;
  words.fill(0x24681357);
  const Pm4ComputeProgram transform = {
      UINT64_C(0x0000123456789000), 0xe0af0000, 0x84, 0x20, 0, {64, 1, 1},
  };
  const Pm4ComputeProgram lds = {
      UINT64_C(0x000023456789a000), 0xe0af0000, 0x84, 0x30, 512, {128, 1, 1},
  };
  Pm4CommandWriter commands(words.data() + 1, Profile(11, 0));
  commands.BindCompute(transform, UINT64_C(0x00003456789abc00));
  commands.BindCompute(lds, UINT64_C(0x00003456789abc40));
  commands.BindCompute(transform, UINT64_C(0x00003456789abc80));
  // PAL's changed-pipeline binding and per-dispatch arguments restore the
  // complete selected state. The final transform has zero LDS, its original
  // prefetch size and two-wave scheduling policy, and a fresh kernarg address.
  // These literal values do not measure physical resource reclamation.
  const std::array<uint32_t, 78> expected = {
      0xc0027602, 0x20c,      0x34567890, 0x12,  0xc0027602, 0x212,
      0xe0af0000, 0x84,       0xc0017602, 0x228, 0x20,       0xc0017602,
      0x215,      0,          0xc0067602, 0x204, 0,          0,
      0,          64,         1,          1,     0xc0027602, 0x240,
      0x789abc00, 0x3456,

      0xc0027602, 0x20c,      0x456789a0, 0x23,  0xc0027602, 0x212,
      0xe0af0000, 0x8084,     0xc0017602, 0x228, 0x30,       0xc0017602,
      0x215,      0x00400000, 0xc0067602, 0x204, 0,          0,
      0,          128,        1,          1,     0xc0027602, 0x240,
      0x789abc40, 0x3456,

      0xc0027602, 0x20c,      0x34567890, 0x12,  0xc0027602, 0x212,
      0xe0af0000, 0x84,       0xc0017602, 0x228, 0x20,       0xc0017602,
      0x215,      0,          0xc0067602, 0x204, 0,          0,
      0,          64,         1,          1,     0xc0027602, 0x240,
      0x789abc80, 0x3456,
  };
  ASSERT_EQ(commands.word_count(), expected.size());
  ExpectWords(words.data() + 1, expected);
  EXPECT_EQ(words.front(), 0x24681357u);
  EXPECT_EQ(words.back(), 0x24681357u);
}

TEST(Pm4EncodingTest, DirectWave32DispatchUsesCompleteThreadDimensions) {
  std::array<uint32_t, 6> words = {};
  words.back() = 0x24681357;
  Pm4CommandWriter commands(words.data(), Profile(11, 0));
  commands.DispatchWave32(1024, 1, 1);
  const std::array<uint32_t, 5> expected = {
      0xc0031502, 1024, 1, 1, 0x8025,
  };
  ASSERT_EQ(commands.word_count(), expected.size());
  ExpectWords(words.data(), expected);
  EXPECT_EQ(words.back(), 0x24681357u);
}

TEST(Pm4EncodingTest, IndirectWave32DispatchUsesAbsoluteByteAddressAndGroups) {
  std::array<uint32_t, 6> words;
  words.fill(0x24681357u);
  Pm4CommandWriter commands(words.data() + 1, Profile(11, 0));
  commands.DispatchIndirectWave32(UINT64_C(0x00000012abcd0100));
  const std::array<uint32_t, 4> expected = {
      0xc0021602,
      0xabcd0100,
      0x00000012,
      0x8005,
  };
  ASSERT_EQ(commands.word_count(), expected.size());
  ExpectWords(words.data() + 1, expected);
  EXPECT_EQ(words.front(), 0x24681357u);
  EXPECT_EQ(words.back(), 0x24681357u);
}

TEST(Pm4EncodingTest, IndirectBufferCallUsesMecAddressAndDwordCount) {
  std::array<uint32_t, 10> words;
  words.fill(0x24681357u);
  Pm4CommandWriter commands(words.data() + 1, Profile(11, 0));
  commands.CallIndirectBuffer(UINT64_C(0x0000123456789000), 40);
  // The upper count bit is a host encoding case, not a native launch size.
  commands.CallIndirectBuffer(UINT64_C(0x0000abcd87654324), 0x00081234);
  const std::array<uint32_t, 8> expected = {
      0xc0023f02, 0x56789000, 0x00001234, 0x00800028,
      0xc0023f02, 0x87654324, 0x0000abcd, 0x00881234,
  };
  ASSERT_EQ(commands.word_count(), expected.size());
  ExpectWords(words.data() + 1, expected);
  EXPECT_EQ(words.front(), 0x24681357u);
  EXPECT_EQ(words.back(), 0x24681357u);
}

TEST(Pm4EncodingTest, SystemReleaseUsesConfirmedEndOfPipeAndReleaseGcr) {
  std::array<uint32_t, 10> words;
  words.fill(0x24681357);
  Pm4CommandWriter commands(words.data() + 1, Profile(11, 0));
  commands.ReleaseSystem32(UINT64_C(0x1234567887654324), 0xfedcba98);
  // PAL's compute postamble uses BOTTOM_OF_PIPE_TS/index5 and release GCR
  // 0x30e. Immediate32/int_sel3/dst_sel1 requests confirmation without an
  // interrupt. The high payload and interrupt context remain zero.
  const std::array<uint32_t, 8> expected = {
      0xc0064900, 0x0030e528, 0x23010000, 0x87654324,
      0x12345678, 0xfedcba98, 0,          0,
  };
  ASSERT_EQ(commands.word_count(), expected.size());
  ExpectWords(words.data() + 1, expected);
  EXPECT_EQ(words.front(), 0x24681357u);
  EXPECT_EQ(words.back(), 0x24681357u);
}

TEST(Pm4EncodingTest, EndOfPipeClockUsesCachelessConfirmedWideWrite) {
  std::array<uint32_t, 10> words;
  words.fill(0x24681357);
  Pm4CommandWriter commands(words.data() + 1, Profile(11, 0));
  commands.ReleaseGpuClock64(UINT64_C(0x0000123487654328));
  const std::array<uint32_t, 8> expected = {
      0xc0064900, 0x00000528, 0x63010000, 0x87654328, 0x00001234, 0, 0, 0,
  };
  ASSERT_EQ(commands.word_count(), expected.size());
  ExpectWords(words.data() + 1, expected);
  EXPECT_EQ(words.front(), 0x24681357u);
  EXPECT_EQ(words.back(), 0x24681357u);
}

TEST(Pm4EncodingTest, EndOfPipeMarkerHasNoCacheActions) {
  std::array<uint32_t, 10> words;
  words.fill(0x24681357);
  Pm4CommandWriter commands(words.data() + 1, Profile(11, 0));
  commands.Release32(UINT64_C(0x0000123487654324), 0x89abcdef);
  const std::array<uint32_t, 8> expected = {
      0xc0064900, 0x00000528, 0x23010000, 0x87654324,
      0x00001234, 0x89abcdef, 0,          0,
  };
  ASSERT_EQ(commands.word_count(), expected.size());
  ExpectWords(words.data() + 1, expected);
  EXPECT_EQ(words.front(), 0x24681357u);
  EXPECT_EQ(words.back(), 0x24681357u);
}

TEST(Pm4EncodingTest, EndOfPipeWritebackJoinsPrivateFenceBeforeCacheWork) {
  std::array<uint32_t, 25> words;
  words.fill(0x24681357);
  Pm4CommandWriter commands(words.data() + 1, Profile(11, 0));
  commands.WaitEndOfPipeAndWriteback(UINT64_C(0x0000123487654324), 0x89abcdef);
  // PAL compute uses a cacheless release and an offloaded equality wait. The
  // standalone ACQUIRE has GL2_WB only and all reserved MEC size bits zero.
  const std::array<uint32_t, 23> expected = {
      0xc0064900, 0x00000528, 0x23010000, 0x87654324, 0x00001234, 0x89abcdef,
      0,          0,          0xc0053c00, 0x13,       0x87654324, 0x00001234,
      0x89abcdef, 0xffffffff, 0x8000000a, 0xc0065800, 0,          0xffffffff,
      0xff,       0,          0,          0x0a,       0x8000,
  };
  ASSERT_EQ(commands.word_count(), expected.size());
  ExpectWords(words.data() + 1, expected);
  EXPECT_EQ(words.front(), 0x24681357u);
  EXPECT_EQ(words.back(), 0x24681357u);
}

TEST(Pm4EncodingTest, ConfirmedCopiesPreserveAddressesAndSelectWidth) {
  std::array<uint32_t, 13> words = {};
  words.back() = 0x24681357;
  Pm4CommandWriter commands(words.data(), Profile(11, 0));
  commands.CopyData32(UINT64_C(0x0000123487654324),
                      UINT64_C(0x00005678fedcba94));
  commands.CopyData64(UINT64_C(0x0000123487654328),
                      UINT64_C(0x00005678fedcba98));
  const std::array<uint32_t, 12> expected = {
      0xc0044000, 0x00100202, 0x87654324, 0x00001234, 0xfedcba94, 0x00005678,
      0xc0044000, 0x00110202, 0x87654328, 0x00001234, 0xfedcba98, 0x00005678};
  ASSERT_EQ(commands.word_count(), expected.size());
  ExpectWords(words.data(), expected);
  EXPECT_EQ(words.back(), 0x24681357u);
}

TEST(Pm4EncodingTest, MecDmaCopyAndDrainKeepReservedControlsClear) {
  std::array<uint32_t, 16> words;
  words.fill(0x24681357);
  Pm4CommandWriter commands(words.data() + 1, Profile(11, 0));
  commands.DmaCopyL2(UINT64_C(0x1234567887654040), UINT64_C(0x2345678998765100),
                     1024);
  commands.WaitDma();
  // RADV's real compute copy/drain uses a direct byte count with RAW_WAIT,
  // enabled write confirmation and no PFP-layout CP_SYNC control.
  const std::array<uint32_t, 14> expected = {
      0xc0055000, 0x60300000, 0x87654040, 0x12345678, 0x98765100,
      0x23456789, 0x40000400, 0xc0055000, 0,          0,
      0,          0,          0,          0,
  };
  ASSERT_EQ(commands.word_count(), expected.size());
  ExpectWords(words.data() + 1, expected);
  EXPECT_EQ(words.front(), 0x24681357u);
  EXPECT_EQ(words.back(), 0x24681357u);
}

TEST(Pm4EncodingTest, SharedCopyDataPreservesFullPayloadAddresses) {
  std::array<uint32_t, 14> words;
  words.fill(0x24681357);
  const size_t first_count =
      pm4::CopyData(words.data() + 1, UINT64_C(0x1234567887654324),
                    UINT64_C(0x23456789fedcba94), pm4::CopyDataWidth::k32Bit);
  ASSERT_EQ(first_count, 6u);
  const size_t second_count = pm4::CopyData(
      words.data() + 1 + first_count, UINT64_C(0x3456789a76543218),
      UINT64_C(0x456789abedcba988), pm4::CopyDataWidth::k64Bit);
  ASSERT_EQ(second_count, 6u);
  // PAL's MEC COPY_DATA uses TC/L2 selectors 2, confirmation bit 20 and
  // width bit 16. Payload addresses retain full high DWORDs, independently
  // of any enclosing INDIRECT_BUFFER address limit.
  const std::array<uint32_t, 12> expected = {
      0xc0044000, 0x00100202, 0x87654324, 0x12345678, 0xfedcba94, 0x23456789,
      0xc0044000, 0x00110202, 0x76543218, 0x3456789a, 0xedcba988, 0x456789ab};
  ExpectWords(words.data() + 1, expected);
  EXPECT_EQ(words.front(), 0x24681357u);
  EXPECT_EQ(words.back(), 0x24681357u);
}

TEST(Pm4EncodingTest, SharedWriteDataCountsOneAndTwoPayloadWords) {
  std::array<uint32_t, 13> words;
  words.fill(0x24681357);
  const uint32_t first_value = 0xfedcba98;
  const size_t first_count = pm4::WriteData(
      words.data() + 1, UINT64_C(0x1234567887654324), &first_value, 1);
  ASSERT_EQ(first_count, 5u);
  const std::array<uint32_t, 2> values = {0x13579bdf, 0x2468ace0};
  const size_t second_count = pm4::WriteData(words.data() + 1 + first_count,
                                             UINT64_C(0x23456789fedcba94),
                                             values.data(), values.size());
  ASSERT_EQ(second_count, 6u);
  // WRITE_DATA's type-3 count includes its payload. Incrementing confirmed
  // TC/L2 writes require DWORD alignment even with two payload DWORDs.
  const std::array<uint32_t, 11> expected = {
      0xc0033700, 0x00100200, 0x87654324, 0x12345678, 0xfedcba98, 0xc0043700,
      0x00100200, 0xfedcba94, 0x23456789, 0x13579bdf, 0x2468ace0};
  ExpectWords(words.data() + 1, expected);
  EXPECT_EQ(words.front(), 0x24681357u);
  EXPECT_EQ(words.back(), 0x24681357u);
}

TEST(Pm4EncodingTest, IncrementingWriteIncludesEveryPayloadWordInCount) {
  std::array<uint32_t, 13> words = {};
  words.back() = 0x24681357;
  const std::array<uint32_t, 3> values = {0, UINT32_MAX, 0x13579bdf};
  Pm4CommandWriter commands(words.data(), Profile(11, 0));
  commands.WriteData32(UINT64_C(0x0000123487654324), 0x2468ace0);
  commands.WriteData(UINT64_C(0x00005678fedcba94), values.data(),
                     values.size());
  const std::array<uint32_t, 12> expected = {
      0xc0033700, 0x00100200, 0x87654324, 0x00001234, 0x2468ace0, 0xc0053700,
      0x00100200, 0xfedcba94, 0x00005678, 0,          UINT32_MAX, 0x13579bdf};
  ASSERT_EQ(commands.word_count(), expected.size());
  ExpectWords(words.data(), expected);
  EXPECT_EQ(words.back(), 0x24681357u);
}

TEST(Pm4EncodingTest, MaximumWritePayloadUsesAllFourteenCountBits) {
  // A 14-bit count holds total DWORDs minus two. Four fixed DWORDs leave
  // 16381 payload DWORDs. This is a wire-format boundary, not a native
  // workload.
  std::vector<uint32_t> values(16381, 0x13579bdf);
  std::vector<uint32_t> words(16386, 0x24681357);
  Pm4CommandWriter commands(words.data(), Profile(11, 0));
  commands.WriteData(UINT64_C(0x0000123487654324), values.data(),
                     values.size());
  ASSERT_EQ(commands.word_count(), 16385u);
  const std::array<uint32_t, 4> prefix = {0xffff3700, 0x00100200, 0x87654324,
                                          0x00001234};
  ExpectWords(words.data(), prefix);
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(words[4 + i], values[i]) << i;
  }
  EXPECT_EQ(words.back(), 0x24681357u);
}

TEST(Pm4EncodingTest, MemoryWaitUsesFullAddressAndDwordPacketCount) {
  std::array<uint32_t, 16> words = {};
  words.back() = 0x24681357;
  Pm4CommandWriter commands(words.data(), Profile(11, 0));
  commands.WaitMemory32(UINT64_C(0x1234567887654320), 17);
  // WAIT_REG_MEM's type-3 count excludes two DWORDs; memory/equal control
  // and the full address are separate fields in the MEC wire format.
  const std::array<uint32_t, 7> expected = {
      0xc0053c00, 0x13, 0x87654320, 0x12345678, 17, 0xffffffff, 4};
  ASSERT_EQ(commands.word_count(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(words[i], expected[i]) << i;
  }
  // The selected ordinary NOP form needs a header and payload, so padding
  // advances to the next eight-DWORD boundary. The special one-DWORD NOP
  // encoding is a separate form that this emitter does not use.
  commands.PadToEightWords();
  EXPECT_EQ(commands.word_count(), 16u);
  EXPECT_EQ(words[7], 0xc0071000u);
  EXPECT_EQ(words.back(), 0u);
}

TEST(Pm4EncodingTest, MaskedMemoryWaitsKeepOrdinaryMecExecution) {
  std::array<uint32_t, 22> words = {};
  words.back() = 0x24681357;
  Pm4CommandWriter commands(words.data(), Profile(11, 0));
  commands.WaitMemory32(UINT64_C(0x0000123487654324), 0x80000000,
                        Pm4MemoryComparison::kEqual, 0xff000000);
  commands.WaitMemory32(UINT64_C(0x00005678fedcba94), 0,
                        Pm4MemoryComparison::kNotEqual, 0x80000000);
  commands.WaitMemory32(UINT64_C(0x00009abcedcba984), 0x80000001,
                        Pm4MemoryComparison::kGreaterOrEqual);
  const std::array<uint32_t, 21> expected = {
      0xc0053c00, 0x13, 0x87654324, 0x00001234, 0x80000000, 0xff000000, 4,
      0xc0053c00, 0x14, 0xfedcba94, 0x00005678, 0,          0x80000000, 4,
      0xc0053c00, 0x15, 0xedcba984, 0x00009abc, 0x80000001, UINT32_MAX, 4};
  ASSERT_EQ(commands.word_count(), expected.size());
  ExpectWords(words.data(), expected);
  EXPECT_EQ(words.back(), 0x24681357u);
}

TEST(Pm4EncodingTest, LessThanWaitsPreserveBothOperandWidths) {
  std::array<uint32_t, 17> words = {};
  words.back() = 0x24681357;
  Pm4CommandWriter commands(words.data(), Profile(11, 0));
  commands.WaitMemory32(UINT64_C(0x0000123487654324), 0x40000000,
                        Pm4MemoryComparison::kLess);
  commands.WaitMemory64(
      UINT64_C(0x00005678fedcba98), UINT64_C(0x3fffffffffffffff),
      Pm4MemoryComparison::kLess, UINT64_C(0x7fffffffffffffff));
  // MEC function 1 is LT; function 2 would permit equality. Both forms use
  // ordinary memory waits with the ACE offload bit clear.
  const std::array<uint32_t, 16> expected = {
      0xc0053c00, 0x11,       0x87654324, 0x00001234, 0x40000000, 0xffffffff,
      4,          0xc0079300, 0x11,       0xfedcba98, 0x00005678, 0xffffffff,
      0x3fffffff, 0xffffffff, 0x7fffffff, 4};
  ASSERT_EQ(commands.word_count(), expected.size());
  ExpectWords(words.data(), expected);
  EXPECT_EQ(words.back(), 0x24681357u);
}

TEST(Pm4EncodingTest, WideWaitSeparatesReferenceAndMaskHalves) {
  std::array<uint32_t, 10> words = {};
  words.back() = 0x24681357;
  Pm4CommandWriter commands(words.data(), Profile(11, 0));
  commands.WaitMemory64(
      UINT64_C(0x0000123487654328), UINT64_C(0x8765000012340000),
      Pm4MemoryComparison::kGreaterOrEqual, UINT64_C(0xffff0000ffff0000));
  const std::array<uint32_t, 9> expected = {0xc0079300, 0x15,       0x87654328,
                                            0x00001234, 0x12340000, 0x87650000,
                                            0xffff0000, 0xffff0000, 4};
  ASSERT_EQ(commands.word_count(), expected.size());
  ExpectWords(words.data(), expected);
  EXPECT_EQ(words.back(), 0x24681357u);
}

TEST(Pm4EncodingTest, GpuClockCopyUsesConfirmedWideTimestampSource) {
  std::array<uint32_t, 7> words = {};
  words.back() = 0x24681357;
  Pm4CommandWriter commands(words.data(), Profile(11, 0));
  commands.CopyGpuClock64(UINT64_C(0x0000123487654328));
  const std::array<uint32_t, 6> expected = {0xc0044000, 0x00110509, 0,
                                            0,          0x87654328, 0x00001234};
  ASSERT_EQ(commands.word_count(), expected.size());
  ExpectWords(words.data(), expected);
  EXPECT_EQ(words.back(), 0x24681357u);
}

TEST(Pm4EncodingTest, ClockCopyUsesConfirmedStreamingPolicies) {
  std::array<uint32_t, 8> words;
  words.fill(0x24681357);
  const size_t word_count =
      pm4::CopyGpuClock64(words.data() + 1, UINT64_C(0x1234567887654328),
                          pm4::CopyDataPolicy::kStreaming);
  // aqlprofile's ClockRetrievePacket uses source 9, destination 5, STREAM
  // policy bits 13/25, 64-bit count 16 and confirmation 20. No source address
  // is present; the complete destination address is retained.
  const std::array<uint32_t, 6> expected = {0xc0044000, 0x02112509, 0,
                                            0,          0x87654328, 0x12345678};
  ASSERT_EQ(word_count, expected.size());
  ExpectWords(words.data() + 1, expected);
  EXPECT_EQ(words.front(), 0x24681357u);
  EXPECT_EQ(words.back(), 0x24681357u);
}

TEST(Pm4EncodingTest, SystemBarrierKeepsMecSizeAndGcrFieldsSeparate) {
  std::array<uint32_t, 11> words = {};
  words.back() = 0x24681357;
  Pm4CommandWriter commands(words.data(), Profile(11, 0));
  commands.SystemBarrier();
  // The MEC size-high field is eight bits. The remaining high bits stay
  // reserved even though some graphics-engine forms have a wider field.
  const std::array<uint32_t, 10> expected = {
      0xc0004600, 0x00000407, 0xc0065800, 0,          UINT32_MAX,
      0x000000ff, 0,          0,          0x0000000a, 0x0000c3a1};
  ASSERT_EQ(commands.word_count(), expected.size());
  ExpectWords(words.data(), expected);
  EXPECT_EQ(words.back(), 0x24681357u);
}

TEST(Pm4EncodingTest, PaddingUsesWholePacketsAtEveryResidue) {
  constexpr std::array<size_t, 8> kPaddingWords = {8, 7, 6, 5, 4, 3, 2, 9};
  constexpr std::array<uint32_t, 8> kPaddingHeaders = {
      0xc0061000, 0xc0051000, 0xc0041000, 0xc0031000,
      0xc0021000, 0xc0011000, 0xc0001000, 0xc0071000};
  const std::array<uint32_t, 8> values = {};
  for (size_t value_count = 1; value_count <= values.size(); ++value_count) {
    SCOPED_TRACE(value_count);
    std::array<uint32_t, 25> words;
    words.fill(0x24681357);
    Pm4CommandWriter commands(words.data(), Profile(11, 0));
    commands.WriteData(UINT64_C(0x0000123487654324), values.data(),
                       value_count);
    const size_t prefix_count = commands.word_count();
    const size_t residue = prefix_count % 8;
    commands.PadToEightWords();
    ASSERT_EQ(commands.word_count(), prefix_count + kPaddingWords[residue]);
    EXPECT_EQ(commands.word_count() % 8, 0u);
    EXPECT_EQ(words[prefix_count], kPaddingHeaders[residue]);
    for (size_t i = prefix_count + 1; i < commands.word_count(); ++i) {
      EXPECT_EQ(words[i], 0u) << i;
    }
    EXPECT_EQ(words[commands.word_count()], 0x24681357u);
  }
}

}  // namespace
