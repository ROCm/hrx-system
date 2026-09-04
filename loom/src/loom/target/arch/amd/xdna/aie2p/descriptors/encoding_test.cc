// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/descriptors/encoding.h"

#include <array>
#include <string>
#include <string_view>
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
        &descriptor_set->operands[descriptor->operand_start + i];
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
        &descriptor_set->operands[descriptor->operand_start + i];
    if (operand->encoding_field_id == 0) continue;
    const loom_aie2p_physical_register_id_t register_id =
        loom_aie2p_machine_find_physical_register(iree_make_string_view(
            register_names[encoded_operand_index].data(),
            register_names[encoded_operand_index].size()));
    ++encoded_operand_index;
    if (register_id == LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "AIE2P test register was not found");
    }
    assignments[i] = (loom_low_allocation_assignment_t){
        .descriptor_reg_class_id =
            descriptor_set->reg_class_alts[operand->reg_class_alt_start]
                .reg_class_id,
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

TEST(DescriptorEncodingTest, PhysicalRegisterRowsAlignWithMachineTable) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_aie2p_core_descriptor_set();
  ASSERT_EQ(descriptor_set->physical_register_count,
            loom_aie2p_machine_physical_register_count());
  for (uint32_t register_id = 0;
       register_id < descriptor_set->physical_register_count; ++register_id) {
    const loom_low_physical_register_t* descriptor_register =
        loom_low_descriptor_set_physical_register_at(descriptor_set,
                                                     register_id);
    loom_aie2p_physical_register_info_t machine_register;
    ASSERT_TRUE(loom_aie2p_machine_query_physical_register(
        (loom_aie2p_physical_register_id_t)register_id, &machine_register));
    EXPECT_TRUE(iree_string_view_equal(
        loom_low_descriptor_set_string(descriptor_set,
                                       descriptor_register->name_string_offset),
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
}

}  // namespace
