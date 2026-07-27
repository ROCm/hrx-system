// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/registers.h"

#include <string>

#include "iree/testing/gtest.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/records/target_records.h"

namespace {

const loom_low_reg_class_t* LookupRegisterClass(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t register_class_name) {
  uint16_t descriptor_register_class_id = LOOM_LOW_REG_CLASS_NONE;
  const loom_low_reg_class_t* reg_class = nullptr;
  EXPECT_TRUE(loom_low_descriptor_set_lookup_register_class(
      descriptor_set, register_class_name, &descriptor_register_class_id,
      &reg_class))
      << std::string(register_class_name.data, register_class_name.size);
  return reg_class;
}

const loom_low_descriptor_t* LookupDescriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t descriptor_key) {
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_lookup_descriptor(descriptor_set, descriptor_key);
  EXPECT_NE(descriptor_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE)
      << std::string(descriptor_key.data, descriptor_key.size);
  return loom_low_descriptor_set_descriptor_at(descriptor_set,
                                               descriptor_ordinal);
}

const loom_low_reg_class_t* OperandRegisterClass(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_operand_t& operand) {
  EXPECT_EQ(operand.reg_class_alt_count, 1);
  if (operand.reg_class_alt_count != 1) {
    return nullptr;
  }
  const loom_low_reg_class_alt_t& alt =
      descriptor_set->reg_class_alts[operand.reg_class_alt_start];
  EXPECT_NE(alt.reg_class_id, LOOM_LOW_REG_CLASS_NONE);
  if (alt.reg_class_id == LOOM_LOW_REG_CLASS_NONE) {
    return nullptr;
  }
  return &descriptor_set->reg_classes[alt.reg_class_id];
}

TEST(SpirvRegistersTest, LogicalIdsAndPointerValuesUseSeparateWidths) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_spirv_logical_core_descriptor_set();
  ASSERT_NE(descriptor_set, nullptr);

  const loom_low_reg_class_t* id_class =
      LookupRegisterClass(descriptor_set, IREE_SV("spirv.id"));
  ASSERT_NE(id_class, nullptr);
  EXPECT_EQ(id_class->alloc_unit_bits, 32);
  EXPECT_EQ(id_class->allocatable_count, 0);
  EXPECT_TRUE(
      iree_all_bits_set(id_class->flags, LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY));
  EXPECT_EQ(id_class->spill_slot_space, LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE);

  const loom_low_reg_class_t* offset64_class =
      LookupRegisterClass(descriptor_set, IREE_SV("spirv.offset64"));
  ASSERT_NE(offset64_class, nullptr);
  EXPECT_EQ(offset64_class->alloc_unit_bits, 64);
  EXPECT_EQ(offset64_class->allocatable_count, 0);
  EXPECT_TRUE(iree_all_bits_set(offset64_class->flags,
                                LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY));
  EXPECT_EQ(offset64_class->spill_slot_space,
            LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE);

  const loom_low_reg_class_t* function_ptr_class =
      LookupRegisterClass(descriptor_set, IREE_SV("spirv.ptr.function"));
  ASSERT_NE(function_ptr_class, nullptr);
  EXPECT_EQ(function_ptr_class->alloc_unit_bits, 32);
  EXPECT_TRUE(iree_all_bits_set(function_ptr_class->flags,
                                LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY));
  EXPECT_EQ(function_ptr_class->spill_slot_space,
            LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE);

  const loom_low_reg_class_t* storage_buffer_ptr_class =
      LookupRegisterClass(descriptor_set, IREE_SV("spirv.ptr.storage_buffer"));
  ASSERT_NE(storage_buffer_ptr_class, nullptr);
  EXPECT_EQ(storage_buffer_ptr_class->alloc_unit_bits, 64);
  EXPECT_TRUE(iree_all_bits_set(storage_buffer_ptr_class->flags,
                                LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY));
  EXPECT_EQ(storage_buffer_ptr_class->spill_slot_space,
            LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE);
}

TEST(SpirvRegistersTest, TypedWorkgroupPointersCarryScalarValueTypes) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_spirv_logical_core_descriptor_set();
  ASSERT_NE(descriptor_set, nullptr);

  const uint16_t workgroup_array_class_id =
      loom_spirv_ptr_workgroup_array_reg_class_id(LOOM_SPIRV_SCALAR_TYPE_S32);
  EXPECT_NE(workgroup_array_class_id, LOOM_LOW_REG_CLASS_NONE);
  const loom_low_reg_class_t* workgroup_array_class = LookupRegisterClass(
      descriptor_set, IREE_SV("spirv.ptr.workgroup.array.i32"));
  ASSERT_NE(workgroup_array_class, nullptr);
  EXPECT_EQ(workgroup_array_class->alloc_unit_bits, 32);
  EXPECT_TRUE(iree_all_bits_set(workgroup_array_class->flags,
                                LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY));
  EXPECT_TRUE(iree_all_bits_set(workgroup_array_class->flags,
                                LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE));

  loom_spirv_value_type_t value_type = {};
  ASSERT_TRUE(loom_spirv_value_type_from_reg_class_id(workgroup_array_class_id,
                                                      &value_type));
  EXPECT_EQ(value_type.value_class, LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY);
  EXPECT_EQ(value_type.scalar_type, LOOM_SPIRV_SCALAR_TYPE_S32);

  const uint16_t workgroup_scalar_class_id =
      loom_spirv_ptr_workgroup_reg_class_id(LOOM_SPIRV_SCALAR_TYPE_S32);
  EXPECT_NE(workgroup_scalar_class_id, LOOM_LOW_REG_CLASS_NONE);
  const loom_low_reg_class_t* workgroup_scalar_class =
      LookupRegisterClass(descriptor_set, IREE_SV("spirv.ptr.workgroup.i32"));
  ASSERT_NE(workgroup_scalar_class, nullptr);
  EXPECT_EQ(workgroup_scalar_class->alloc_unit_bits, 32);
  EXPECT_TRUE(iree_all_bits_set(workgroup_scalar_class->flags,
                                LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY));
  EXPECT_TRUE(iree_all_bits_set(workgroup_scalar_class->flags,
                                LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE));

  ASSERT_TRUE(loom_spirv_value_type_from_reg_class_id(workgroup_scalar_class_id,
                                                      &value_type));
  EXPECT_EQ(value_type.value_class, LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP);
  EXPECT_EQ(value_type.scalar_type, LOOM_SPIRV_SCALAR_TYPE_S32);
}

TEST(SpirvRegistersTest, VulkanTargetUsesBufferDeviceAddressWidths) {
  const loom_target_bundle_t& target = loom_spirv_low_target_bundle_vulkan1_3;
  ASSERT_NE(target.snapshot, nullptr);
  EXPECT_EQ(target.snapshot->default_pointer_bitwidth, 64u);
  EXPECT_EQ(target.snapshot->index_bitwidth, 32u);
  EXPECT_EQ(target.snapshot->offset_bitwidth, 64u);
  EXPECT_EQ(target.snapshot->memory_spaces.descriptor,
            LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER);
}

TEST(SpirvRegistersTest, StorageBufferEffectsStayDescriptorLocal) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_spirv_logical_core_descriptor_set();
  ASSERT_NE(descriptor_set, nullptr);

  const loom_low_descriptor_t* load_descriptor = LookupDescriptor(
      descriptor_set, IREE_SV("spirv.op_load.storage_buffer.i32"));
  ASSERT_NE(load_descriptor, nullptr);
  ASSERT_EQ(load_descriptor->effect_count, 1);
  const loom_low_effect_t& load_effect =
      descriptor_set->effects[load_descriptor->effect_start];
  EXPECT_EQ(load_effect.kind, LOOM_LOW_EFFECT_KIND_READ);
  EXPECT_EQ(load_effect.memory_space, LOOM_LOW_MEMORY_SPACE_GLOBAL);
  EXPECT_TRUE(
      iree_all_bits_set(load_effect.flags, LOOM_LOW_EFFECT_FLAG_DEPENDENCY));

  const loom_low_descriptor_t* store_descriptor = LookupDescriptor(
      descriptor_set, IREE_SV("spirv.op_store.storage_buffer.i32"));
  ASSERT_NE(store_descriptor, nullptr);
  ASSERT_EQ(store_descriptor->effect_count, 1);
  const loom_low_effect_t& store_effect =
      descriptor_set->effects[store_descriptor->effect_start];
  EXPECT_EQ(store_effect.kind, LOOM_LOW_EFFECT_KIND_WRITE);
  EXPECT_EQ(store_effect.memory_space, LOOM_LOW_MEMORY_SPACE_GLOBAL);
  EXPECT_TRUE(
      iree_all_bits_set(store_effect.flags, LOOM_LOW_EFFECT_FLAG_DEPENDENCY));
}

TEST(SpirvRegistersTest, StorageBufferAddressArithmeticUsesBdaRegisters) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_spirv_logical_core_descriptor_set();
  ASSERT_NE(descriptor_set, nullptr);

  const loom_low_descriptor_t* offset_add =
      LookupDescriptor(descriptor_set, IREE_SV("spirv.op_iadd.offset64"));
  ASSERT_NE(offset_add, nullptr);
  const loom_low_reg_class_t* offset_class =
      LookupRegisterClass(descriptor_set, IREE_SV("spirv.offset64"));
  ASSERT_NE(offset_class, nullptr);
  ASSERT_EQ(offset_add->operand_count, 3);
  const loom_low_operand_t* operands =
      &descriptor_set->operands[offset_add->operand_start];
  EXPECT_EQ(OperandRegisterClass(descriptor_set, operands[0]), offset_class);
  EXPECT_EQ(OperandRegisterClass(descriptor_set, operands[1]), offset_class);
  EXPECT_EQ(OperandRegisterClass(descriptor_set, operands[2]), offset_class);

  const loom_low_descriptor_t* ptr_access_chain = LookupDescriptor(
      descriptor_set,
      IREE_SV("spirv.op_ptr_access_chain.storage_buffer.i32.byte_offset"));
  ASSERT_NE(ptr_access_chain, nullptr);
  const loom_low_reg_class_t* storage_buffer_ptr_class =
      LookupRegisterClass(descriptor_set, IREE_SV("spirv.ptr.storage_buffer"));
  ASSERT_NE(storage_buffer_ptr_class, nullptr);
  ASSERT_EQ(ptr_access_chain->operand_count, 3);
  operands = &descriptor_set->operands[ptr_access_chain->operand_start];
  EXPECT_EQ(OperandRegisterClass(descriptor_set, operands[0]),
            storage_buffer_ptr_class);
  EXPECT_EQ(OperandRegisterClass(descriptor_set, operands[1]),
            storage_buffer_ptr_class);
  EXPECT_EQ(OperandRegisterClass(descriptor_set, operands[2]), offset_class);
  EXPECT_EQ(operands[1].role, LOOM_LOW_OPERAND_ROLE_RESOURCE);
  EXPECT_EQ(operands[2].role, LOOM_LOW_OPERAND_ROLE_OPERAND);
}

TEST(SpirvRegistersTest, WorkgroupAccessChainUsesTypedArrayBase) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_spirv_logical_core_descriptor_set();
  ASSERT_NE(descriptor_set, nullptr);

  const loom_low_descriptor_t* access_chain = LookupDescriptor(
      descriptor_set,
      IREE_SV("spirv.op_access_chain.workgroup.i32.element_index"));
  ASSERT_NE(access_chain, nullptr);
  const loom_low_reg_class_t* workgroup_array_class = LookupRegisterClass(
      descriptor_set, IREE_SV("spirv.ptr.workgroup.array.i32"));
  ASSERT_NE(workgroup_array_class, nullptr);
  const loom_low_reg_class_t* workgroup_scalar_class =
      LookupRegisterClass(descriptor_set, IREE_SV("spirv.ptr.workgroup.i32"));
  ASSERT_NE(workgroup_scalar_class, nullptr);
  const loom_low_reg_class_t* id_class =
      LookupRegisterClass(descriptor_set, IREE_SV("spirv.id"));
  ASSERT_NE(id_class, nullptr);

  ASSERT_EQ(access_chain->operand_count, 3);
  const loom_low_operand_t* operands =
      &descriptor_set->operands[access_chain->operand_start];
  EXPECT_EQ(OperandRegisterClass(descriptor_set, operands[0]),
            workgroup_scalar_class);
  EXPECT_EQ(OperandRegisterClass(descriptor_set, operands[1]),
            workgroup_array_class);
  EXPECT_EQ(OperandRegisterClass(descriptor_set, operands[2]), id_class);
  EXPECT_EQ(operands[1].role, LOOM_LOW_OPERAND_ROLE_RESOURCE);
  EXPECT_EQ(operands[2].role, LOOM_LOW_OPERAND_ROLE_OPERAND);
}

}  // namespace
