// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/descriptors/encoding.h"

#include <array>
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
  if (register_names.size() != descriptor->operand_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P test register count differs");
  }

  std::vector<loom_low_allocation_assignment_t> assignments(
      descriptor->operand_count);
  std::vector<const loom_low_allocation_assignment_t*> assignment_ptrs(
      descriptor->operand_count);
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        &descriptor_set->operands[descriptor->operand_start + i];
    const loom_aie2p_physical_register_id_t register_id =
        loom_aie2p_machine_find_physical_register(iree_make_string_view(
            register_names[i].data(), register_names[i].size()));
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
