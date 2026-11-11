// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/descriptors/encoding.h"

#include <array>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/machine/machine.h"

namespace {

iree_status_t EncodeDescriptor(
    const loom_low_descriptor_set_t* descriptor_set, std::string_view key,
    const std::vector<std::string_view>& register_names,
    const std::vector<int64_t>& immediate_values,
    loom_aie2p_encoded_slot_t* out_slot) {
  const uint32_t descriptor_ordinal = loom_low_descriptor_set_lookup_descriptor(
      descriptor_set, iree_make_string_view(key.data(), key.size()));
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "AIE2P test descriptor was not found");
  }
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[descriptor_ordinal];
  iree_host_size_t encoded_operand_count = 0;
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        loom_low_descriptor_operand(descriptor_set, descriptor, i);
    encoded_operand_count += operand->encoding_field_id != 0;
  }
  if (register_names.size() != encoded_operand_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P test register count differs");
  }

  std::vector<loom_low_allocation_assignment_t> assignments(
      descriptor->operand_count);
  std::vector<const loom_low_allocation_assignment_t*> assignment_ptrs(
      descriptor->operand_count, nullptr);
  iree_host_size_t encoded_operand_index = 0;
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        loom_low_descriptor_operand(descriptor_set, descriptor, i);
    if (operand->encoding_field_id == 0) {
      continue;
    }
    const loom_aie2p_physical_register_id_t register_id =
        loom_aie2p_machine_find_physical_register(iree_make_string_view(
            register_names[encoded_operand_index].data(),
            register_names[encoded_operand_index].size()));
    ++encoded_operand_index;
    if (register_id == LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "AIE2P test register was not found");
    }
    assignments[i] = loom_low_allocation_assignment_t{
        .value_id = {},
        .descriptor_reg_class_id =
            descriptor_set->reg_class_alts[operand->reg_class_alt_start]
                .reg_class_id,
        .flags = {},
        .start_point = {},
        .end_point = {},
        .unit_count = 1,
        .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
        .location_base = register_id,
        .location_count = 1,
    };
    assignment_ptrs[i] = &assignments[i];
  }
  *out_slot = loom_aie2p_descriptor_encode(descriptor_set, descriptor_ordinal,
                                           assignment_ptrs.data(),
                                           immediate_values.data());
  return iree_ok_status();
}

iree_status_t AppendBundle(loom_aie2p_bundle_format_id_t format,
                           const std::vector<loom_aie2p_encoded_slot_t>& slots,
                           std::vector<uint8_t>* program) {
  loom_aie2p_encoding_packet_t packet;
  IREE_RETURN_IF_ERROR(loom_aie2p_encoding_pack_bundle(format, slots.data(),
                                                       slots.size(), &packet));
  program->insert(program->end(), packet.data,
                  packet.data + packet.data_length);
  return iree_ok_status();
}

iree_status_t EncodeSingleDescriptor(
    std::string_view descriptor_key,
    const std::vector<std::string_view>& register_names,
    const std::vector<int64_t>& immediate_values,
    std::string_view bundle_format, std::vector<uint8_t>* out_program) {
  loom_aie2p_encoded_slot_t slot;
  IREE_RETURN_IF_ERROR(EncodeDescriptor(loom_aie2p_core_descriptor_set(),
                                        descriptor_key, register_names,
                                        immediate_values, &slot));
  return AppendBundle(
      loom_aie2p_encoding_find_bundle_format(
          iree_make_string_view(bundle_format.data(), bundle_format.size())),
      {slot}, out_program);
}

iree_status_t BuildVectorAddLeaf(std::string_view vector_add_key,
                                 std::vector<uint8_t>* out_program) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_aie2p_core_descriptor_set();
  out_program->clear();

  loom_aie2p_encoded_slot_t load_a;
  IREE_RETURN_IF_ERROR(EncodeDescriptor(
      descriptor_set, "amd.xdna.aie2p.load.a.i8x64.indexed.immediate",
      {"x0", "p0"}, {0}, &load_a));
  loom_aie2p_encoded_slot_t load_b;
  IREE_RETURN_IF_ERROR(EncodeDescriptor(
      descriptor_set, "amd.xdna.aie2p.load.b.i8x64.indexed.immediate",
      {"x2", "p1"}, {0}, &load_b));
  loom_aie2p_encoded_slot_t vector_add;
  IREE_RETURN_IF_ERROR(EncodeDescriptor(descriptor_set, vector_add_key,
                                        {"x0", "x2", "x0"}, {}, &vector_add));
  loom_aie2p_encoded_slot_t store;
  IREE_RETURN_IF_ERROR(EncodeDescriptor(
      descriptor_set, "amd.xdna.aie2p.store.i8x64.indexed.immediate",
      {"x0", "p2"}, {0}, &store));
  loom_aie2p_encoded_slot_t nop;
  IREE_RETURN_IF_ERROR(
      EncodeDescriptor(descriptor_set, "amd.xdna.aie2p.nop", {}, {}, &nop));
  loom_aie2p_encoded_slot_t ret;
  IREE_RETURN_IF_ERROR(
      EncodeDescriptor(descriptor_set, "amd.xdna.aie2p.return", {}, {}, &ret));

  IREE_RETURN_IF_ERROR(AppendBundle(
      loom_aie2p_encoding_find_bundle_format(IREE_SV("I48_LDA_LDB")),
      {load_a, load_b}, out_program));
  const loom_aie2p_bundle_format_id_t nop_format =
      loom_aie2p_encoding_find_bundle_format(IREE_SV("I16_NOP"));
  for (int i = 0; i < 4; ++i) {
    IREE_RETURN_IF_ERROR(AppendBundle(nop_format, {nop}, out_program));
  }
  IREE_RETURN_IF_ERROR(
      AppendBundle(loom_aie2p_encoding_find_bundle_format(IREE_SV("I32_ALU")),
                   {ret}, out_program));
  IREE_RETURN_IF_ERROR(AppendBundle(nop_format, {nop}, out_program));
  IREE_RETURN_IF_ERROR(
      AppendBundle(loom_aie2p_encoding_find_bundle_format(IREE_SV("I32_MV")),
                   {vector_add}, out_program));
  IREE_RETURN_IF_ERROR(AppendBundle(nop_format, {nop}, out_program));
  IREE_RETURN_IF_ERROR(
      AppendBundle(loom_aie2p_encoding_find_bundle_format(IREE_SV("I32_ST")),
                   {store}, out_program));
  IREE_RETURN_IF_ERROR(AppendBundle(nop_format, {nop}, out_program));
  return iree_ok_status();
}

TEST(DescriptorEncodingTest, ConcreteConstantsSelectSignedElevenBitForms) {
  const uint32_t original_ordinals[] = {
      AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32,
      AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_STATIC_BYTE_OFFSET_I32,
      AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_STATIC_ADDRESS_INDEX_I20,
  };
  const uint32_t compact_ordinals[] = {
      AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_SHORT,
      AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_SHORT,
      AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_STATIC_ADDRESS_INDEX_I20_SHORT,
  };
  for (size_t i = 0; i < IREE_ARRAYSIZE(original_ordinals); ++i) {
    for (int64_t value : {-1024, -1, 0, 1, 1023}) {
      EXPECT_EQ(loom_aie2p_descriptor_select_constant_encoding(
                    original_ordinals[i], &value),
                compact_ordinals[i]);
    }
    for (int64_t value : {INT32_MIN, -1025, 1024, INT32_MAX}) {
      EXPECT_EQ(loom_aie2p_descriptor_select_constant_encoding(
                    original_ordinals[i], &value),
                original_ordinals[i]);
    }
  }
}

TEST(DescriptorEncodingTest, RelocationsAndOtherFormsKeepTheirEncoding) {
  const int64_t zero = 0;
  EXPECT_EQ(loom_aie2p_descriptor_select_constant_encoding(
                AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_LOCAL_ADDRESS_I32, &zero),
            AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_LOCAL_ADDRESS_I32);
  EXPECT_EQ(loom_aie2p_descriptor_select_constant_encoding(
                AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_MOVA, &zero),
            AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_MOVA);
  EXPECT_EQ(loom_aie2p_descriptor_select_constant_encoding(
                AIE2P_CORE_DESCRIPTOR_REF_MOVE_SCALAR, nullptr),
            AIE2P_CORE_DESCRIPTOR_REF_MOVE_SCALAR);
}

TEST(DescriptorEncodingTest, PhysicalAssignmentsReproduceVectorLeaves) {
  struct TestCase {
    std::string_view descriptor_key;
    std::array<uint8_t, 32> expected;
  };
  const TestCase test_cases[] = {
      {
          "amd.xdna.aie2p.add.i32x16",
          {0x3c, 0x68, 0x09, 0x72, 0x83, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
           0x00, 0x00, 0x00, 0x18, 0x00, 0x28, 0x10, 0x00, 0x00, 0x78, 0x2d,
           0x10, 0x18, 0x00, 0x00, 0x18, 0x13, 0x04, 0x0a, 0x00, 0x00},
      },
      {
          "amd.xdna.aie2p.add.i16x32",
          {0x3c, 0x68, 0x09, 0x72, 0x83, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
           0x00, 0x00, 0x00, 0x18, 0x00, 0x28, 0x10, 0x00, 0x00, 0x78, 0x1d,
           0x10, 0x18, 0x00, 0x00, 0x18, 0x13, 0x04, 0x0a, 0x00, 0x00},
      },
      {
          "amd.xdna.aie2p.add.i8x64",
          {0x3c, 0x68, 0x09, 0x72, 0x83, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
           0x00, 0x00, 0x00, 0x18, 0x00, 0x28, 0x10, 0x00, 0x00, 0x78, 0x0d,
           0x10, 0x18, 0x00, 0x00, 0x18, 0x13, 0x04, 0x0a, 0x00, 0x00},
      },
  };

  for (const TestCase& test_case : test_cases) {
    std::vector<uint8_t> program;
    IREE_ASSERT_OK(BuildVectorAddLeaf(test_case.descriptor_key, &program));
    EXPECT_EQ(program, std::vector<uint8_t>(test_case.expected.begin(),
                                            test_case.expected.end()));
  }
}

TEST(DescriptorEncodingTest, VolatileAliasesPreservePhysicalEncoding) {
  struct TestCase {
    std::string_view descriptor_key;
    std::vector<std::string_view> registers;
    std::vector<int64_t> immediates;
  };
  const TestCase test_cases[] = {
      {
          "amd.xdna.aie2p.load.scalar.i8.indexed.immediate",
          {"r3", "p0"},
          {4},
      },
      {
          "amd.xdna.aie2p.store.scalar.i8.indexed.immediate",
          {"r3", "p0"},
          {4},
      },
      {
          "amd.xdna.aie2p.load.scalar.i16.indexed.immediate",
          {"r3", "p0"},
          {4},
      },
      {
          "amd.xdna.aie2p.store.scalar.i16.indexed.immediate",
          {"r3", "p0"},
          {4},
      },
      {
          "amd.xdna.aie2p.load.scalar.i32.indexed.immediate",
          {"r3", "p0"},
          {4},
      },
      {
          "amd.xdna.aie2p.store.scalar.i32.indexed.immediate",
          {"r3", "p0"},
          {4},
      },
      {
          "amd.xdna.aie2p.load.a.i8x64.indexed.immediate",
          {"x0", "p0"},
          {64},
      },
      {
          "amd.xdna.aie2p.store.i8x64.indexed.immediate",
          {"x0", "p0"},
          {64},
      },
  };

  const loom_low_descriptor_set_t* descriptor_set =
      loom_aie2p_core_descriptor_set();
  for (const TestCase& test_case : test_cases) {
    loom_aie2p_encoded_slot_t ordinary_slot;
    IREE_ASSERT_OK(EncodeDescriptor(descriptor_set, test_case.descriptor_key,
                                    test_case.registers, test_case.immediates,
                                    &ordinary_slot));
    loom_aie2p_encoded_slot_t volatile_slot;
    IREE_ASSERT_OK(EncodeDescriptor(
        descriptor_set, std::string(test_case.descriptor_key) + ".volatile",
        test_case.registers, test_case.immediates, &volatile_slot));
    EXPECT_EQ(volatile_slot.slot, ordinary_slot.slot);
    EXPECT_EQ(volatile_slot.value, ordinary_slot.value);
  }
}

TEST(DescriptorEncodingTest, AccumulatorMovesMatchOracleInstructionEncodings) {
  struct TestCase {
    std::vector<std::string_view> registers;
    std::array<uint8_t, 4> expected;
  };
  const TestCase test_cases[] = {
      {{"bmll2", "bmhl0"}, {0xF8, 0x12, 0x02, 0x1A}},
      {{"bmhl0", "bmll2"}, {0xF8, 0x12, 0x88, 0x18}},
      {{"bmll2", "bmll0"}, {0xF8, 0x12, 0x00, 0x1A}},
  };

  for (const TestCase& test_case : test_cases) {
    std::vector<uint8_t> program;
    IREE_ASSERT_OK(EncodeSingleDescriptor("amd.xdna.aie2p.move.accumulator512",
                                          test_case.registers, {}, "I32_MV",
                                          &program));
    EXPECT_EQ(program, std::vector<uint8_t>(test_case.expected.begin(),
                                            test_case.expected.end()));
  }
}

TEST(DescriptorEncodingTest, DirectBranchesMatchOracleInstructionEncodings) {
  struct TestCase {
    std::string_view descriptor_key;
    std::vector<std::string_view> registers;
    int64_t target;
    std::array<uint8_t, 6> expected;
  };
  const TestCase test_cases[] = {
      {
          "amd.xdna.aie2p.branch.direct",
          {},
          0x12345,
          {0x84, 0x00, 0x80, 0xA2, 0x91, 0x00},  // j #0x12345
      },
      {
          "amd.xdna.aie2p.branch.nonzero",
          {"r3"},
          0x23456,
          {0x84, 0x01, 0x40, 0x2B, 0x1A, 0x19},  // jnz r3, #0x23456
      },
      {
          "amd.xdna.aie2p.branch.zero",
          {"r17"},
          0x34567,
          {0x84, 0x01, 0x80, 0xB3, 0xA2, 0x89},  // jz r17, #0x34567
      },
  };

  for (const TestCase& test_case : test_cases) {
    std::vector<uint8_t> program;
    IREE_ASSERT_OK(
        EncodeSingleDescriptor(test_case.descriptor_key, test_case.registers,
                               {test_case.target}, "I48_LNG", &program));
    EXPECT_EQ(program, std::vector<uint8_t>(test_case.expected.begin(),
                                            test_case.expected.end()));
  }
}

TEST(DescriptorEncodingTest, LocksMatchOracleInstructionEncodings) {
  struct TestCase {
    std::string_view descriptor_key;
    std::vector<std::string_view> registers;
    std::vector<int64_t> immediates;
    std::array<uint8_t, 4> expected;
  };
  const TestCase test_cases[] = {
      {"amd.xdna.aie2p.lock.acquire.register",
       {"r1", "r27"},
       {},
       {0x18, 0xB8, 0x53, 0x10}},
      {"amd.xdna.aie2p.lock.acquire.immediate",
       {"r26"},
       {4},
       {0x18, 0xA8, 0x83, 0x10}},
      {"amd.xdna.aie2p.lock.release.register",
       {"r2", "r3"},
       {},
       {0x18, 0x38, 0x90, 0x10}},
      {"amd.xdna.aie2p.lock.release.immediate",
       {"r26"},
       {63},
       {0x18, 0xA8, 0xE1, 0x17}},
      {"amd.xdna.aie2p.lock.acquire.conditional.register",
       {"r2", "r27"},
       {},
       {0x18, 0xB8, 0x97, 0x10}},
      {"amd.xdna.aie2p.lock.acquire.conditional.immediate",
       {"r2"},
       {0},
       {0x18, 0x28, 0x06, 0x10}},
      {"amd.xdna.aie2p.lock.release.conditional.register",
       {"r2", "r6"},
       {},
       {0x18, 0x68, 0x94, 0x10}},
      {"amd.xdna.aie2p.lock.release.conditional.immediate",
       {"r5"},
       {12},
       {0x18, 0x58, 0x84, 0x11}},
  };

  for (const TestCase& test_case : test_cases) {
    std::vector<uint8_t> program;
    IREE_ASSERT_OK(
        EncodeSingleDescriptor(test_case.descriptor_key, test_case.registers,
                               test_case.immediates, "I32_ALU", &program));
    EXPECT_EQ(program, std::vector<uint8_t>(test_case.expected.begin(),
                                            test_case.expected.end()));
  }
}

TEST(DescriptorEncodingTest, I16MultiplyMatchesOracleInstructionEncodings) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_aie2p_core_descriptor_set();
  std::vector<uint8_t> program;

  loom_aie2p_encoded_slot_t multiply_control;
  IREE_ASSERT_OK(EncodeDescriptor(descriptor_set,
                                  "amd.xdna.aie2p.constant.i32.mova", {"r0"},
                                  {0x35A}, &multiply_control));
  IREE_ASSERT_OK(
      AppendBundle(loom_aie2p_encoding_find_bundle_format(IREE_SV("I32_LDA")),
                   {multiply_control}, &program));

  loom_aie2p_encoded_slot_t narrow_shift;
  IREE_ASSERT_OK(EncodeDescriptor(descriptor_set,
                                  "amd.xdna.aie2p.constant.i32.shift", {"s0"},
                                  {0}, &narrow_shift));
  IREE_ASSERT_OK(
      AppendBundle(loom_aie2p_encoding_find_bundle_format(IREE_SV("I32_MV")),
                   {narrow_shift}, &program));

  loom_aie2p_encoded_slot_t multiply;
  IREE_ASSERT_OK(EncodeDescriptor(descriptor_set,
                                  "amd.xdna.aie2p.multiply.i16x32.configured",
                                  {"dm0", "x2", "x4", "r0"}, {}, &multiply));
  IREE_ASSERT_OK(
      AppendBundle(loom_aie2p_encoding_find_bundle_format(IREE_SV("I32_VEC")),
                   {multiply}, &program));

  for (const auto& [key, value] :
       std::array<std::pair<std::string_view, int64_t>, 3>{
           std::pair{"amd.xdna.aie2p.state.rounding.immediate", 0},
           std::pair{"amd.xdna.aie2p.state.srs-mode.immediate", 1},
           std::pair{"amd.xdna.aie2p.state.saturation.immediate", 0},
       }) {
    loom_aie2p_encoded_slot_t state_write;
    IREE_ASSERT_OK(
        EncodeDescriptor(descriptor_set, key, {}, {value}, &state_write));
    IREE_ASSERT_OK(
        AppendBundle(loom_aie2p_encoding_find_bundle_format(IREE_SV("I32_MV")),
                     {state_write}, &program));
  }

  loom_aie2p_encoded_slot_t narrow;
  IREE_ASSERT_OK(EncodeDescriptor(descriptor_set,
                                  "amd.xdna.aie2p.narrow.trunc.signed.i16x32",
                                  {"x0", "dm0", "s0"}, {}, &narrow));
  IREE_ASSERT_OK(
      AppendBundle(loom_aie2p_encoding_find_bundle_format(IREE_SV("I32_ST")),
                   {narrow}, &program));

  const std::array<uint8_t, 28> expected = {
      0x18, 0x00, 0x5A, 0x03,  // mova r0, #0x35a
      0xB8, 0x00, 0xE0, 0x18,  // mov s0, #0
      0x08, 0x81, 0xE4, 0x00,  // vmul dm0, x2, x4, r0
      0xB8, 0x00, 0xB0, 0x1F,  // mov crRnd, #0
      0xB8, 0x02, 0x70, 0x1C,  // mov crSRSMode, #1
      0xB8, 0x00, 0x70, 0x1A,  // mov crSat, #0
      0x18, 0x26, 0x64, 0x08,  // vsrs.4x x0, dm0, s0, srsSign1
  };
  EXPECT_EQ(program, std::vector<uint8_t>(expected.begin(), expected.end()));
}

TEST(DescriptorEncodingTest, NarrowExtendsMatchOracleInstructionEncodings) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_aie2p_core_descriptor_set();
  struct TestCase {
    std::string_view descriptor_key;
    std::array<uint8_t, 4> expected;
  };
  const TestCase test_cases[] = {
      {"amd.xdna.aie2p.extend.signed.i8", {0x18, 0x50, 0x40, 0x10}},
      {"amd.xdna.aie2p.extend.signed.i16", {0x18, 0x70, 0x40, 0x10}},
      {"amd.xdna.aie2p.extend.unsigned.i8", {0x18, 0x90, 0x40, 0x10}},
      {"amd.xdna.aie2p.extend.unsigned.i16", {0x18, 0xB0, 0x40, 0x10}},
  };

  for (const TestCase& test_case : test_cases) {
    std::vector<uint8_t> program;
    loom_aie2p_encoded_slot_t extend;
    IREE_ASSERT_OK(EncodeDescriptor(descriptor_set, test_case.descriptor_key,
                                    {"r0", "r1"}, {}, &extend));
    IREE_ASSERT_OK(
        AppendBundle(loom_aie2p_encoding_find_bundle_format(IREE_SV("I32_ALU")),
                     {extend}, &program));
    EXPECT_EQ(program, std::vector<uint8_t>(test_case.expected.begin(),
                                            test_case.expected.end()));
  }
}

TEST(DescriptorEncodingTest,
     IntegerPacketConversionsMatchOracleInstructionEncodings) {
  struct TestCase {
    std::string_view descriptor_key;
    std::vector<std::string_view> registers;
    std::vector<int64_t> immediates;
    std::string_view bundle_format;
    std::array<uint8_t, 4> expected;
  };
  const TestCase test_cases[] = {
      {"amd.xdna.aie2p.widen.2x.w-to-b.unsigned.configured",
       {"bmhl2", "wl3", "s0"},
       {},
       "I32_MV",
       {0xF8, 0x10, 0x87, 0x1A}},
      {"amd.xdna.aie2p.widen.2x.w-to-b.signed.configured",
       {"bmlh4", "wh7", "s2"},
       {},
       "I32_MV",
       {0xF8, 0xB0, 0x4E, 0x1C}},
      {"amd.xdna.aie2p.widen.2x.x-to-c.unsigned.configured",
       {"cmh4", "x4", "s1"},
       {},
       "I32_MV",
       {0xF8, 0x58, 0x88, 0x1C}},
      {"amd.xdna.aie2p.widen.2x.x-to-c.signed.configured",
       {"cmh0", "x9", "s2"},
       {},
       "I32_MV",
       {0xF8, 0xB8, 0x92, 0x18}},
      {"amd.xdna.aie2p.widen.4x.w-to-c.unsigned.configured",
       {"cmh4", "wl3", "s1"},
       {},
       "I32_MV",
       {0xF8, 0x48, 0x87, 0x1C}},
      {"amd.xdna.aie2p.widen.4x.w-to-c.signed.configured",
       {"cmh3", "wl0", "s3"},
       {},
       "I32_MV",
       {0xF8, 0xE8, 0x81, 0x1B}},
      {"amd.xdna.aie2p.widen.4x.x-to-d.unsigned.configured",
       {"dm2", "x6", "s2"},
       {},
       "I32_MV",
       {0xF8, 0x84, 0x0C, 0x1A}},
      {"amd.xdna.aie2p.widen.4x.x-to-d.signed.configured",
       {"dm2", "x2", "s3"},
       {},
       "I32_MV",
       {0xF8, 0xE4, 0x04, 0x1A}},
      {"amd.xdna.aie2p.pack.w.trunc.configured",
       {"wl9", "x11"},
       {},
       "I32_ST",
       {0x18, 0xD6, 0xDA, 0x0C}},
      {"amd.xdna.aie2p.pack.x.trunc.configured",
       {"x11", "y0"},
       {},
       "I32_ST",
       {0x18, 0x16, 0x84, 0x0D}},
      {"amd.xdna.aie2p.convert.floor.bf16x16.to.i32x16",
       {"x2", "wl4", "s1"},
       {},
       "I32_ST",
       {0x18, 0x36, 0x29, 0x09}},
      {"amd.xdna.aie2p.state.ups-mode.immediate",
       {},
       {1023},
       "I32_MV",
       {0xB8, 0xFE, 0x77, 0x1E}},
      {"amd.xdna.aie2p.state.pack-size.immediate",
       {},
       {-1024},
       "I32_MV",
       {0xB8, 0x00, 0xB8, 0x1B}},
  };

  for (const TestCase& test_case : test_cases) {
    SCOPED_TRACE(test_case.descriptor_key);
    std::vector<uint8_t> program;
    IREE_ASSERT_OK(EncodeSingleDescriptor(
        test_case.descriptor_key, test_case.registers, test_case.immediates,
        test_case.bundle_format, &program));
    EXPECT_EQ(program, std::vector<uint8_t>(test_case.expected.begin(),
                                            test_case.expected.end()));
  }
}

TEST(DescriptorEncodingTest,
     FusedVectorMemoryConversionsMatchOracleInstructionEncodings) {
  struct TestCase {
    std::string_view descriptor_key;
    std::vector<std::string_view> registers;
    std::vector<int64_t> immediates;
    std::string_view bundle_format;
    std::array<uint8_t, 4> expected;
  };
  const TestCase test_cases[] = {
      {"amd.xdna.aie2p.load.convert.bf16x16.to.f32x16.indexed.immediate",
       {"bmhl4", "p3"},
       {224},
       "I32_LDA",
       {0x98, 0x49, 0x76, 0x03}},
      {"amd.xdna.aie2p.load.convert.bf16x32.to.f32x32.indexed.immediate",
       {"cml2", "p2"},
       {-512},
       "I32_LDA",
       {0x98, 0x2B, 0x85, 0x02}},
      {"amd.xdna.aie2p.load.widen.2x.w-to-b.unsigned.configured.indexed."
       "immediate",
       {"bmhh3", "s1", "p1"},
       {32},
       "I32_LDA",
       {0x98, 0xE8, 0x15, 0x01}},
      {"amd.xdna.aie2p.load.widen.2x.x-to-c.unsigned.configured.indexed."
       "immediate",
       {"cmh3", "s1", "p7"},
       {-512},
       "I32_LDA",
       {0x18, 0xD3, 0x85, 0x07}},
      {"amd.xdna.aie2p.load.widen.4x.w-to-c.signed.configured.indexed."
       "immediate",
       {"cmh3", "s0", "p0"},
       {0},
       "I32_LDA",
       {0x18, 0xCD, 0x05, 0x00}},
      {"amd.xdna.aie2p.load.widen.4x.x-to-d.signed.configured.indexed."
       "immediate",
       {"dm2", "s3", "p6"},
       {-512},
       "I32_LDA",
       {0x18, 0x7F, 0x85, 0x06}},
      {"amd.xdna.aie2p.store.convert.f32x16.to.bf16x16.indexed.immediate",
       {"bmll0", "p0"},
       {-128},
       "I32_ST",
       {0x98, 0x12, 0xC4, 0x08}},
      {"amd.xdna.aie2p.store.convert.f32x32.to.bf16x32.indexed.immediate",
       {"cml0", "p0"},
       {-256},
       "I32_ST",
       {0x18, 0x23, 0xC4, 0x08}},
      {"amd.xdna.aie2p.store.pack.w.trunc.configured.indexed.immediate",
       {"x7", "p7"},
       {32},
       "I32_ST",
       {0x98, 0xC2, 0x15, 0x0F}},
      {"amd.xdna.aie2p.store.pack.x.trunc.configured.indexed.immediate",
       {"y5", "p7"},
       {448},
       "I32_ST",
       {0x98, 0x8E, 0x76, 0x0F}},
  };

  for (const TestCase& test_case : test_cases) {
    SCOPED_TRACE(test_case.descriptor_key);
    std::vector<uint8_t> program;
    IREE_ASSERT_OK(EncodeSingleDescriptor(
        test_case.descriptor_key, test_case.registers, test_case.immediates,
        test_case.bundle_format, &program));
    EXPECT_EQ(program, std::vector<uint8_t>(test_case.expected.begin(),
                                            test_case.expected.end()));
  }
}

TEST(DescriptorEncodingTest, IntegerMinMaxMatchOracleInstructionEncodings) {
  struct TestCase {
    std::string_view descriptor_key;
    std::array<uint8_t, 4> expected;
  };
  const TestCase test_cases[] = {
      {"amd.xdna.aie2p.min.signed.i8x64", {0x58, 0x4E, 0x12, 0x18}},
      {"amd.xdna.aie2p.min.unsigned.i8x64", {0x58, 0x4C, 0x12, 0x18}},
      {"amd.xdna.aie2p.max.signed.i8x64", {0x58, 0x52, 0x12, 0x18}},
      {"amd.xdna.aie2p.max.unsigned.i8x64", {0x58, 0x50, 0x12, 0x18}},
      {"amd.xdna.aie2p.min.signed.i16x32", {0x58, 0x2E, 0x12, 0x18}},
      {"amd.xdna.aie2p.min.unsigned.i16x32", {0x58, 0x2C, 0x12, 0x18}},
      {"amd.xdna.aie2p.max.signed.i16x32", {0x58, 0x32, 0x12, 0x18}},
      {"amd.xdna.aie2p.max.unsigned.i16x32", {0x58, 0x30, 0x12, 0x18}},
      {"amd.xdna.aie2p.min.signed.i32x16", {0x58, 0x0E, 0x12, 0x18}},
      {"amd.xdna.aie2p.min.unsigned.i32x16", {0x58, 0x0C, 0x12, 0x18}},
      {"amd.xdna.aie2p.max.signed.i32x16", {0x58, 0x12, 0x12, 0x18}},
      {"amd.xdna.aie2p.max.unsigned.i32x16", {0x58, 0x10, 0x12, 0x18}},
  };

  for (const TestCase& test_case : test_cases) {
    std::vector<uint8_t> program;
    IREE_ASSERT_OK(EncodeSingleDescriptor(
        test_case.descriptor_key, {"x0", "x2", "x4"}, {}, "I32_MV", &program));
    EXPECT_EQ(program, std::vector<uint8_t>(test_case.expected.begin(),
                                            test_case.expected.end()));
  }
}

TEST(DescriptorEncodingTest, VectorPredicatesMatchOracleInstructionEncodings) {
  struct TestCase {
    std::string_view descriptor_key;
    std::vector<std::string_view> registers;
    std::array<uint8_t, 4> expected;
  };
  const TestCase test_cases[] = {
      {"amd.xdna.aie2p.cmp.lt.signed.i8x64",
       {"l8", "x0", "x2"},
       {0x78, 0x22, 0x01, 0x1C}},
      {"amd.xdna.aie2p.cmp.ge.unsigned.i8x64",
       {"l8", "x0", "x2"},
       {0x78, 0x42, 0x01, 0x1C}},
      {"amd.xdna.aie2p.cmp.lt.signed.i16x32.el.low32",
       {"l8", "x0", "x2"},
       {0x78, 0x2A, 0x01, 0x18}},
      {"amd.xdna.aie2p.cmp.ge.unsigned.i16x32.el.low32",
       {"l8", "x0", "x2"},
       {0x78, 0x4A, 0x01, 0x18}},
      {"amd.xdna.aie2p.cmp.lt.signed.i32x16.el.low32",
       {"l8", "x0", "x2"},
       {0x78, 0x32, 0x01, 0x18}},
      {"amd.xdna.aie2p.cmp.ge.unsigned.i32x16.el.low32",
       {"l8", "x0", "x2"},
       {0x78, 0x52, 0x01, 0x18}},
      {"amd.xdna.aie2p.select.i8x64",
       {"x0", "x2", "x4", "l8"},
       {0x38, 0x44, 0x12, 0x18}},
      {"amd.xdna.aie2p.select.i16x32.mask64",
       {"x0", "x2", "x4", "l8"},
       {0x38, 0x02, 0x12, 0x18}},
      {"amd.xdna.aie2p.select.i32x16.mask64",
       {"x0", "x2", "x4", "l8"},
       {0x38, 0x00, 0x12, 0x18}},
  };

  for (const TestCase& test_case : test_cases) {
    std::vector<uint8_t> program;
    IREE_ASSERT_OK(EncodeSingleDescriptor(
        test_case.descriptor_key, test_case.registers, {}, "I32_MV", &program));
    EXPECT_EQ(program, std::vector<uint8_t>(test_case.expected.begin(),
                                            test_case.expected.end()));
  }
}

TEST(DescriptorEncodingTest, PredicatePartsMatchOracleInstructionEncodings) {
  struct TestCase {
    std::string_view descriptor_key;
    std::array<uint8_t, 4> expected;
  };
  const TestCase test_cases[] = {
      {"amd.xdna.aie2p.predicate.and.low32", {0x98, 0x24, 0x21, 0x15}},
      {"amd.xdna.aie2p.predicate.and.high32", {0x98, 0x34, 0x63, 0x15}},
      {"amd.xdna.aie2p.predicate.or.low32", {0x98, 0x25, 0x21, 0x15}},
      {"amd.xdna.aie2p.predicate.or.high32", {0x98, 0x35, 0x63, 0x15}},
      {"amd.xdna.aie2p.predicate.xor.low32", {0x98, 0x26, 0x21, 0x15}},
      {"amd.xdna.aie2p.predicate.xor.high32", {0x98, 0x36, 0x63, 0x15}},
  };

  for (const TestCase& test_case : test_cases) {
    std::vector<uint8_t> program;
    IREE_ASSERT_OK(EncodeSingleDescriptor(test_case.descriptor_key,
                                          {"l8", "l10", "l9"}, {}, "I32_ALU",
                                          &program));
    EXPECT_EQ(program, std::vector<uint8_t>(test_case.expected.begin(),
                                            test_case.expected.end()));
  }

  std::vector<uint8_t> complete_program;
  IREE_ASSERT_OK(
      EncodeSingleDescriptor("amd.xdna.aie2p.predicate.complete.zero.high32",
                             {"l8"}, {0}, "I32_LDA", &complete_program));
  EXPECT_EQ(complete_program, (std::vector<uint8_t>{0x18, 0x88, 0x00, 0x00}));
}

TEST(DescriptorEncodingTest, DirectMoveRoutesMatchEveryDescriptorPair) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_aie2p_core_descriptor_set();
  std::vector<uint32_t> move_descriptors;
  for (uint32_t ordinal = 0; ordinal < descriptor_set->descriptor_count;
       ++ordinal) {
    if (iree_any_bit_set(descriptor_set->descriptors[ordinal].flags,
                         LOOM_LOW_DESCRIPTOR_FLAG_ALLOCATION_MOVE)) {
      move_descriptors.push_back(ordinal);
    }
  }
  ASSERT_FALSE(move_descriptors.empty());
  // The reference interprets the actual descriptor register classes instead
  // of reproducing the generated mask representation. Include unsupported
  // pairs, overlapping classes, singleton selectors, and aggregate registers.
  for (uint32_t source = 0; source < descriptor_set->physical_register_count;
       ++source) {
    for (uint32_t destination = 0;
         destination < descriptor_set->physical_register_count; ++destination) {
      uint32_t expected = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
      for (uint32_t ordinal : move_descriptors) {
        const auto* descriptor = &descriptor_set->descriptors[ordinal];
        const auto* destination_operand =
            loom_low_descriptor_operand(descriptor_set, descriptor, 0);
        const auto* source_operand =
            loom_low_descriptor_operand(descriptor_set, descriptor, 1);
        const uint16_t destination_class =
            descriptor_set
                ->reg_class_alts[destination_operand->reg_class_alt_start]
                .reg_class_id;
        const uint16_t source_class =
            descriptor_set->reg_class_alts[source_operand->reg_class_alt_start]
                .reg_class_id;
        if (loom_low_descriptor_set_find_physical_register_candidate(
                descriptor_set, source_class, source, nullptr) &&
            loom_low_descriptor_set_find_physical_register_candidate(
                descriptor_set, destination_class, destination, nullptr)) {
          expected = ordinal;
          break;
        }
      }
      if (expected != LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
        const loom_aie2p_register_move_t move = {
            .source = (loom_aie2p_physical_register_id_t)source,
            .destination = (loom_aie2p_physical_register_id_t)destination,
        };
        const loom_aie2p_move_route_id_t route_id =
            loom_aie2p_descriptor_resolve_move_route(move.source,
                                                     move.destination);
        ASSERT_NE(route_id, LOOM_AIE2P_MOVE_ROUTE_ID_INVALID);
        ASSERT_EQ(loom_aie2p_descriptor_move_route_step_count(route_id), 1);
        const loom_aie2p_move_route_step_t step =
            loom_aie2p_descriptor_move_route_step(route_id, move, 0);
        const loom_aie2p_move_step_profile_t profile =
            loom_aie2p_descriptor_move_step_profile(step.profile_id);
        EXPECT_EQ(step.move.source, move.source);
        EXPECT_EQ(step.move.destination, move.destination);
        EXPECT_EQ(profile.descriptor_ordinal, expected);
        EXPECT_EQ(profile.resource_schedule_class_count, 0);
        EXPECT_EQ(profile.encoding, LOOM_AIE2P_MOVE_STEP_ENCODING_DESCRIPTOR);
      }
    }
  }
}

TEST(DescriptorEncodingTest, ScalarMoveRoutesCoverNativeOperandAdapters) {
  auto adapter_candidates = [](std::string_view name) {
    std::vector<loom_aie2p_physical_register_id_t> result;
    const auto adapter_id = loom_aie2p_machine_find_register_adapter(
        iree_make_string_view(name.data(), name.size()));
    EXPECT_NE(adapter_id, LOOM_AIE2P_REGISTER_ADAPTER_ID_INVALID);
    loom_aie2p_register_adapter_info_t adapter_info;
    if (!loom_aie2p_machine_query_register_adapter(adapter_id, &adapter_info)) {
      ADD_FAILURE() << "missing adapter " << name;
      return result;
    }
    loom_aie2p_register_class_info_t class_info;
    if (!loom_aie2p_machine_query_register_class(adapter_info.register_class_id,
                                                 &class_info)) {
      ADD_FAILURE() << "missing adapter class " << name;
      return result;
    }
    for (uint8_t i = 0; i < class_info.candidate_count; ++i) {
      result.push_back(loom_aie2p_machine_register_class_candidate(
          adapter_info.register_class_id, i));
    }
    return result;
  };
  const auto sources = adapter_candidates("OP_mMvSclSrc");
  const auto destinations = adapter_candidates("OP_mMvSclDst");
  ASSERT_FALSE(sources.empty());
  ASSERT_FALSE(destinations.empty());

  const loom_low_descriptor_set_t* descriptor_set =
      loom_aie2p_core_descriptor_set();
  const uint32_t scalar_descriptor = loom_low_descriptor_set_lookup_descriptor(
      descriptor_set, IREE_SV("amd.xdna.aie2p.move.scalar"));
  ASSERT_NE(scalar_descriptor, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  for (const auto source : sources) {
    for (const auto destination : destinations) {
      const loom_aie2p_register_move_t move = {
          .source = source,
          .destination = destination,
      };
      const loom_aie2p_move_route_id_t route_id =
          loom_aie2p_descriptor_resolve_move_route(source, destination);
      ASSERT_NE(route_id, LOOM_AIE2P_MOVE_ROUTE_ID_INVALID)
          << "source=" << source << " destination=" << destination;
      ASSERT_EQ(loom_aie2p_descriptor_move_route_step_count(route_id), 1);
      const loom_aie2p_move_route_step_t step =
          loom_aie2p_descriptor_move_route_step(route_id, move, 0);
      const loom_aie2p_move_step_profile_t profile =
          loom_aie2p_descriptor_move_step_profile(step.profile_id);
      EXPECT_EQ(step.move.source, source);
      EXPECT_EQ(step.move.destination, destination);
      EXPECT_EQ(profile.encoding, LOOM_AIE2P_MOVE_STEP_ENCODING_DESCRIPTOR);
      if (profile.resource_schedule_class_count == 0) {
        EXPECT_TRUE(iree_any_bit_set(
            descriptor_set->descriptors[profile.descriptor_ordinal].flags,
            LOOM_LOW_DESCRIPTOR_FLAG_ALLOCATION_MOVE));
      } else {
        EXPECT_EQ(profile.descriptor_ordinal, scalar_descriptor);
        EXPECT_GE(profile.resource_schedule_class_count, 1);
        EXPECT_LE(profile.resource_schedule_class_count,
                  LOOM_AIE2P_MOVE_STEP_MAX_RESOURCE_CLASS_COUNT);
        for (uint8_t i = 0; i < profile.resource_schedule_class_count; ++i) {
          EXPECT_LT(profile.resource_schedule_class_ids[i],
                    descriptor_set->schedule_class_count);
        }
      }
    }
  }
}

TEST(DescriptorEncodingTest, AggregateMoveRoutesPreserveNativeComponentOrder) {
  struct TestCase {
    std::string_view source;
    std::string_view destination;
    std::vector<std::pair<std::string_view, std::string_view>> steps;
    std::string_view event_descriptor;
    loom_aie2p_move_step_encoding_t encoding;
  };
  const std::vector<TestCase> test_cases = {
      {"l8",
       "l9",
       {{"r16", "r18"}, {"r17", "r19"}},
       "amd.xdna.aie2p.move.scalar",
       LOOM_AIE2P_MOVE_STEP_ENCODING_DESCRIPTOR},
      {"d0",
       "d1",
       {{"dc0", "dc1"}, {"dn0", "dn1"}, {"dj0", "dj1"}, {"m0", "m1"}},
       "amd.xdna.aie2p.move.scalar",
       LOOM_AIE2P_MOVE_STEP_ENCODING_DESCRIPTOR},
      {"d0_3d",
       "d1_3d",
       {{"dc0", "dc1"},
        {"dn0", "dn1"},
        {"dj0", "dj1"},
        {"m0", "m1"},
        {"dc4", "dc5"},
        {"dn4", "dn5"},
        {"dj4", "dj5"}},
       "amd.xdna.aie2p.move.scalar",
       LOOM_AIE2P_MOVE_STEP_ENCODING_DESCRIPTOR},
      {"dm0",
       "dm1",
       {{"cml0", "cml1"}, {"cmh0", "cmh1"}},
       "amd.xdna.aie2p.move.accumulator512",
       LOOM_AIE2P_MOVE_STEP_ENCODING_ACCUMULATOR_1024},
      {"lf0",
       "sf",
       {{"lfl0", "sfl"}, {"lfh0", "sfh"}},
       "amd.xdna.aie2p.move.accumulator512",
       LOOM_AIE2P_MOVE_STEP_ENCODING_FIFO_512},
  };
  const loom_low_descriptor_set_t* descriptor_set =
      loom_aie2p_core_descriptor_set();
  auto register_id = [](std::string_view name) {
    return loom_aie2p_machine_find_physical_register(
        iree_make_string_view(name.data(), name.size()));
  };
  for (const TestCase& test_case : test_cases) {
    const loom_aie2p_register_move_t move = {
        .source = register_id(test_case.source),
        .destination = register_id(test_case.destination),
    };
    ASSERT_NE(move.source, LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID);
    ASSERT_NE(move.destination, LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID);
    const loom_aie2p_move_route_id_t route_id =
        loom_aie2p_descriptor_resolve_move_route(move.source, move.destination);
    ASSERT_NE(route_id, LOOM_AIE2P_MOVE_ROUTE_ID_INVALID);
    ASSERT_EQ(loom_aie2p_descriptor_move_route_step_count(route_id),
              test_case.steps.size());
    const uint32_t expected_descriptor =
        loom_low_descriptor_set_lookup_descriptor(
            descriptor_set,
            iree_make_string_view(test_case.event_descriptor.data(),
                                  test_case.event_descriptor.size()));
    ASSERT_NE(expected_descriptor, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
    for (uint8_t i = 0; i < test_case.steps.size(); ++i) {
      const loom_aie2p_move_route_step_t step =
          loom_aie2p_descriptor_move_route_step(route_id, move, i);
      const loom_aie2p_move_step_profile_t profile =
          loom_aie2p_descriptor_move_step_profile(step.profile_id);
      EXPECT_EQ(step.move.source, register_id(test_case.steps[i].first));
      EXPECT_EQ(step.move.destination, register_id(test_case.steps[i].second));
      EXPECT_EQ(profile.descriptor_ordinal, expected_descriptor);
      EXPECT_EQ(profile.encoding, test_case.encoding);
      EXPECT_GE(profile.resource_schedule_class_count, 1);
      EXPECT_LE(profile.resource_schedule_class_count,
                LOOM_AIE2P_MOVE_STEP_MAX_RESOURCE_CLASS_COUNT);
      for (uint8_t resource = 0;
           resource < profile.resource_schedule_class_count; ++resource) {
        EXPECT_LT(profile.resource_schedule_class_ids[resource],
                  descriptor_set->schedule_class_count);
      }
    }
  }
}

TEST(DescriptorEncodingTest, AggregateMoveRoutesCoverAllocatedFamilies) {
  struct Family {
    std::array<uint16_t, 2> register_classes;
    uint8_t register_class_count;
    uint8_t step_count;
    loom_aie2p_move_step_encoding_t encoding;
  };
  const Family families[] = {
      {{{AIE2P_CORE_REG_CLASS_ID_AIE2P_ELPREDICATE,
         AIE2P_CORE_REG_CLASS_ID_AIE2P_ML8M}},
       2,
       2,
       LOOM_AIE2P_MOVE_STEP_ENCODING_DESCRIPTOR},
      {{{AIE2P_CORE_REG_CLASS_ID_AIE2P_ED, 0}},
       1,
       4,
       LOOM_AIE2P_MOVE_STEP_ENCODING_DESCRIPTOR},
      {{{AIE2P_CORE_REG_CLASS_ID_AIE2P_EDS, 0}},
       1,
       7,
       LOOM_AIE2P_MOVE_STEP_ENCODING_DESCRIPTOR},
      {{{AIE2P_CORE_REG_CLASS_ID_AIE2P_EDM, 0}},
       1,
       2,
       LOOM_AIE2P_MOVE_STEP_ENCODING_ACCUMULATOR_1024},
      {{{AIE2P_CORE_REG_CLASS_ID_AIE2P_ELDFIFOREG,
         AIE2P_CORE_REG_CLASS_ID_AIE2P_MSTFIFO}},
       2,
       2,
       LOOM_AIE2P_MOVE_STEP_ENCODING_FIFO_512},
  };
  const loom_low_descriptor_set_t* descriptor_set =
      loom_aie2p_core_descriptor_set();
  for (const Family& family : families) {
    for (uint8_t source_class_index = 0;
         source_class_index < family.register_class_count;
         ++source_class_index) {
      const loom_low_reg_class_t* source_class =
          &descriptor_set
               ->reg_classes[family.register_classes[source_class_index]];
      for (uint8_t destination_class_index = 0;
           destination_class_index < family.register_class_count;
           ++destination_class_index) {
        const loom_low_reg_class_t* destination_class =
            &descriptor_set->reg_classes
                 [family.register_classes[destination_class_index]];
        for (uint16_t source_ordinal = 0;
             source_ordinal < source_class->allocatable_count;
             ++source_ordinal) {
          const uint16_t source =
              descriptor_set->physical_register_candidate_ids
                  [source_class->physical_register_candidate_start +
                   source_ordinal];
          for (uint16_t destination_ordinal = 0;
               destination_ordinal < destination_class->allocatable_count;
               ++destination_ordinal) {
            const uint16_t destination =
                descriptor_set->physical_register_candidate_ids
                    [destination_class->physical_register_candidate_start +
                     destination_ordinal];
            const loom_aie2p_move_route_id_t route_id =
                loom_aie2p_descriptor_resolve_move_route(source, destination);
            ASSERT_NE(route_id, LOOM_AIE2P_MOVE_ROUTE_ID_INVALID);
            ASSERT_EQ(loom_aie2p_descriptor_move_route_step_count(route_id),
                      family.step_count);
            for (uint8_t step_index = 0; step_index < family.step_count;
                 ++step_index) {
              const loom_aie2p_move_route_step_t step =
                  loom_aie2p_descriptor_move_route_step(
                      route_id,
                      loom_aie2p_register_move_t{
                          .source = source,
                          .destination = destination,
                      },
                      step_index);
              EXPECT_EQ(loom_aie2p_descriptor_move_step_profile(step.profile_id)
                            .encoding,
                        family.encoding);
            }
          }
        }
      }
    }
  }
}

TEST(DescriptorEncodingTest, AggregateMoveStepsMatchNativeEncodings) {
  struct TestCase {
    std::string_view source;
    std::string_view destination;
    std::array<uint64_t, 2> expected_values;
  };
  const TestCase test_cases[] = {
      {"dm0", "dm1", {UINT64_C(0x00080457), UINT64_C(0x000c1457)}},
      {"lf0", "sf", {UINT64_C(0x001b2c97), UINT64_C(0x00330c97)}},
  };
  auto register_id = [](std::string_view name) {
    return loom_aie2p_machine_find_physical_register(
        iree_make_string_view(name.data(), name.size()));
  };
  for (const TestCase& test_case : test_cases) {
    const loom_aie2p_register_move_t move = {
        .source = register_id(test_case.source),
        .destination = register_id(test_case.destination),
    };
    ASSERT_NE(move.source, LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID);
    ASSERT_NE(move.destination, LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID);
    const loom_aie2p_move_route_id_t route_id =
        loom_aie2p_descriptor_resolve_move_route(move.source, move.destination);
    ASSERT_NE(route_id, LOOM_AIE2P_MOVE_ROUTE_ID_INVALID);
    ASSERT_EQ(loom_aie2p_descriptor_move_route_step_count(route_id), 2);
    for (uint8_t i = 0; i < 2; ++i) {
      const loom_aie2p_encoded_slot_t encoded =
          loom_aie2p_descriptor_encode_move_step(
              loom_aie2p_core_descriptor_set(),
              loom_aie2p_descriptor_move_route_step(route_id, move, i));
      EXPECT_EQ(encoded.slot, LOOM_AIE2P_SLOT_MV);
      EXPECT_EQ(encoded.value, test_case.expected_values[i]);
    }
  }
}

TEST(DescriptorEncodingTest, IncompatibleAggregateMoveRoutesAreAbsent) {
  auto register_id = [](std::string_view name) {
    return loom_aie2p_machine_find_physical_register(
        iree_make_string_view(name.data(), name.size()));
  };
  for (const auto& [source, destination] :
       std::array<std::pair<std::string_view, std::string_view>, 3>{
           {{"d0", "dm0"}, {"dm0", "lf0"}, {"l8", "d0"}}}) {
    EXPECT_EQ(loom_aie2p_descriptor_resolve_move_route(
                  register_id(source), register_id(destination)),
              LOOM_AIE2P_MOVE_ROUTE_ID_INVALID);
  }
}

TEST(DescriptorEncodingTest, ScalarPairMovesPreserveBothWords) {
  const uint32_t scalar_move = loom_low_descriptor_set_lookup_descriptor(
      loom_aie2p_core_descriptor_set(), IREE_SV("amd.xdna.aie2p.move.scalar"));
  ASSERT_NE(scalar_move, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  auto register_id = [](const std::string& name) {
    return loom_aie2p_machine_find_physical_register(
        iree_make_string_view(name.data(), name.size()));
  };
  for (unsigned source = 0; source < 16; ++source) {
    for (unsigned destination = 0; destination < 16; ++destination) {
      const loom_aie2p_register_move_t move = {
          .source = register_id("l" + std::to_string(source)),
          .destination = register_id("l" + std::to_string(destination)),
      };
      const loom_aie2p_move_route_id_t route_id =
          loom_aie2p_descriptor_resolve_move_route(move.source,
                                                   move.destination);
      ASSERT_NE(route_id, LOOM_AIE2P_MOVE_ROUTE_ID_INVALID);
      ASSERT_EQ(loom_aie2p_descriptor_move_route_step_count(route_id), 2);
      for (unsigned part = 0; part < 2; ++part) {
        const loom_aie2p_move_route_step_t step =
            loom_aie2p_descriptor_move_route_step(route_id, move, part);
        EXPECT_EQ(step.move.source,
                  register_id("r" + std::to_string(source * 2 + part)));
        EXPECT_EQ(step.move.destination,
                  register_id("r" + std::to_string(destination * 2 + part)));
        EXPECT_EQ(loom_aie2p_descriptor_move_step_profile(step.profile_id)
                      .descriptor_ordinal,
                  scalar_move);
      }
    }
  }
}

TEST(DescriptorEncodingTest, MachinePhysicalRegisterRowsRemainStablePrefix) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_aie2p_core_descriptor_set();
  const uint32_t machine_register_count =
      loom_aie2p_machine_physical_register_count();
  const std::array<std::string_view, 4> allocation_only_registers = {
      "predicate_pair0",
      "predicate_pair1",
      "predicate_pair2",
      "predicate_pair3",
  };
  ASSERT_EQ(descriptor_set->physical_register_count,
            machine_register_count + allocation_only_registers.size());
  for (uint32_t register_id = 0; register_id < machine_register_count;
       ++register_id) {
    const loom_low_physical_register_t* descriptor_register =
        loom_low_descriptor_set_physical_register_at(descriptor_set,
                                                     register_id);
    loom_aie2p_physical_register_info_t machine_register;
    ASSERT_TRUE(loom_aie2p_machine_query_physical_register(
        (loom_aie2p_physical_register_id_t)register_id, &machine_register));
    EXPECT_TRUE(iree_string_view_equal(
        loom_low_descriptor_set_string(descriptor_set,
                                       descriptor_register->name_string_ref),
        machine_register.name));

    uint16_t atomic_unit_count = 0;
    const uint16_t* atomic_units =
        loom_low_descriptor_set_physical_register_atomic_units(
            descriptor_set, register_id, &atomic_unit_count);
    ASSERT_EQ(atomic_unit_count, machine_register.atomic_unit_count);
    for (uint16_t i = 0; i < atomic_unit_count; ++i) {
      EXPECT_EQ(atomic_units[i],
                loom_aie2p_machine_physical_register_atomic_unit(
                    (loom_aie2p_physical_register_id_t)register_id, i));
    }
  }
  for (iree_host_size_t i = 0; i < allocation_only_registers.size(); ++i) {
    const loom_low_physical_register_t* descriptor_register =
        loom_low_descriptor_set_physical_register_at(
            descriptor_set, machine_register_count + i);
    const iree_string_view_t name = loom_low_descriptor_set_string(
        descriptor_set, descriptor_register->name_string_ref);
    EXPECT_EQ(std::string_view(name.data, name.size),
              allocation_only_registers[i]);
    EXPECT_EQ(loom_aie2p_machine_find_physical_register(name),
              LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID);
  }
}

}  // namespace
