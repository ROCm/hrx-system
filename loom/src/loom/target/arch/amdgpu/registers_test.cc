// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/testing/gtest.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/planning/occupancy_tables.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"

namespace {

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

struct RegisterAltExpectation {
  // Descriptor-set-local register class accepted by the alternative.
  uint16_t reg_class_id;
  // Required alternative flags.
  loom_low_reg_class_alt_flags_t flags;
};

class AmdgpuRegistersTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_amdgpu_low_descriptor_registry_initialize(&low_registry_);
  }

  const loom_low_descriptor_set_t* LookupDescriptorSet(
      iree_string_view_t descriptor_set_key) const {
    return loom_low_descriptor_registry_lookup(&low_registry_.registry,
                                               descriptor_set_key);
  }

  loom_target_low_descriptor_registry_t low_registry_ = {};
};

void ExpectRegisterClass(const loom_low_descriptor_set_t* descriptor_set,
                         iree_string_view_t expected_name,
                         uint16_t expected_alloc_unit_bits,
                         uint16_t expected_allocatable_count,
                         loom_low_reg_class_flags_t expected_flags,
                         uint16_t* out_descriptor_reg_class_id = nullptr) {
  uint16_t descriptor_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  const loom_low_reg_class_t* reg_class = nullptr;
  ASSERT_TRUE(loom_low_descriptor_set_lookup_register_class(
      descriptor_set, expected_name, &descriptor_reg_class_id, &reg_class))
      << ToString(expected_name);
  ASSERT_NE(reg_class, nullptr) << ToString(expected_name);
  if (out_descriptor_reg_class_id != nullptr) {
    *out_descriptor_reg_class_id = descriptor_reg_class_id;
  }
  EXPECT_EQ(ToString(loom_low_descriptor_set_string(
                descriptor_set, reg_class->name_string_offset)),
            ToString(expected_name));
  EXPECT_EQ(reg_class->alloc_unit_bits, expected_alloc_unit_bits)
      << ToString(expected_name);
  EXPECT_EQ(reg_class->allocatable_count, expected_allocatable_count)
      << ToString(expected_name);
  EXPECT_TRUE(iree_all_bits_set(reg_class->flags, expected_flags))
      << ToString(expected_name);
}

void ExpectRegisterClassMissing(const loom_low_descriptor_set_t* descriptor_set,
                                iree_string_view_t register_class_name) {
  uint16_t descriptor_register_class_id = LOOM_LOW_REG_CLASS_NONE;
  const loom_low_reg_class_t* reg_class = nullptr;
  EXPECT_FALSE(loom_low_descriptor_set_lookup_register_class(
      descriptor_set, register_class_name, &descriptor_register_class_id,
      &reg_class))
      << ToString(register_class_name);
  EXPECT_EQ(descriptor_register_class_id, LOOM_LOW_REG_CLASS_NONE)
      << ToString(register_class_name);
  EXPECT_EQ(reg_class, nullptr) << ToString(register_class_name);
}

void ExpectOperandRegisterAlts(const loom_low_descriptor_set_t* descriptor_set,
                               uint32_t descriptor_ref, uint16_t operand_index,
                               loom_low_operand_role_t expected_role,
                               uint16_t expected_unit_count,
                               const RegisterAltExpectation* expected_alts,
                               iree_host_size_t expected_alt_count) {
  ASSERT_LT(descriptor_ref, descriptor_set->descriptor_count);
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[descriptor_ref];
  ASSERT_LT(operand_index, descriptor->operand_count);
  const loom_low_operand_t* operand =
      &descriptor_set->operands[descriptor->operand_start + operand_index];
  EXPECT_EQ(operand->role, expected_role);
  EXPECT_EQ(operand->unit_count, expected_unit_count);
  EXPECT_EQ(operand->address_map_kind, LOOM_LOW_OPERAND_ADDRESS_MAP_DIRECT);
  ASSERT_EQ(operand->reg_class_alt_count, expected_alt_count);
  for (iree_host_size_t i = 0; i < expected_alt_count; ++i) {
    const loom_low_reg_class_alt_t* alt =
        &descriptor_set
             ->reg_class_alts[operand->reg_class_alt_start + (uint16_t)i];
    EXPECT_EQ(alt->reg_class_id, expected_alts[i].reg_class_id)
        << "operand " << operand_index << " alt " << i;
    EXPECT_EQ(alt->flags, expected_alts[i].flags)
        << "operand " << operand_index << " alt " << i;
  }
}

void ExpectCdnaMfmaVgprAgprOperands(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t descriptor_ref,
    uint16_t vgpr_id, uint16_t agpr_id) {
  constexpr loom_low_reg_class_alt_flags_t kPreferred =
      LOOM_LOW_REG_CLASS_ALT_FLAG_PREFERRED;
  const RegisterAltExpectation vgpr_agpr[] = {
      {vgpr_id, kPreferred},
      {agpr_id, kPreferred},
  };
  const RegisterAltExpectation vgpr_agpr_or_immediate[] = {
      {vgpr_id, kPreferred},
      {agpr_id, kPreferred},
      {LOOM_LOW_REG_CLASS_NONE, LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE},
  };
  ExpectOperandRegisterAlts(descriptor_set, descriptor_ref, /*operand_index=*/0,
                            LOOM_LOW_OPERAND_ROLE_RESULT,
                            /*expected_unit_count=*/4, vgpr_agpr,
                            IREE_ARRAYSIZE(vgpr_agpr));
  ExpectOperandRegisterAlts(descriptor_set, descriptor_ref, /*operand_index=*/1,
                            LOOM_LOW_OPERAND_ROLE_OPERAND,
                            /*expected_unit_count=*/2, vgpr_agpr,
                            IREE_ARRAYSIZE(vgpr_agpr));
  ExpectOperandRegisterAlts(descriptor_set, descriptor_ref, /*operand_index=*/2,
                            LOOM_LOW_OPERAND_ROLE_OPERAND,
                            /*expected_unit_count=*/2, vgpr_agpr,
                            IREE_ARRAYSIZE(vgpr_agpr));
  ExpectOperandRegisterAlts(descriptor_set, descriptor_ref, /*operand_index=*/3,
                            LOOM_LOW_OPERAND_ROLE_OPERAND,
                            /*expected_unit_count=*/4, vgpr_agpr_or_immediate,
                            IREE_ARRAYSIZE(vgpr_agpr_or_immediate));
}

void ExpectOccupancyRegisterClass(const loom_amdgpu_occupancy_model_t* model,
                                  iree_host_size_t index,
                                  iree_string_view_t expected_name,
                                  uint32_t expected_pool_units,
                                  uint32_t expected_allocation_granularity) {
  ASSERT_LT(index, model->register_class_count) << ToString(expected_name);
  const loom_amdgpu_occupancy_register_class_model_t* reg_class =
      &model->register_classes[index];
  EXPECT_EQ(ToString(reg_class->register_class), ToString(expected_name));
  EXPECT_EQ(reg_class->pool_units, expected_pool_units)
      << ToString(expected_name);
  EXPECT_EQ(reg_class->allocation_granularity, expected_allocation_granularity)
      << ToString(expected_name);
}

void ExpectOccupancyPressureResource(
    const loom_amdgpu_occupancy_model_t* model, iree_host_size_t index,
    iree_string_view_t expected_name, uint32_t expected_pool_units,
    uint32_t expected_allocation_granularity, uint16_t expected_vgpr_index,
    uint32_t expected_vgpr_contribution_granularity,
    uint16_t expected_agpr_index,
    uint32_t expected_agpr_contribution_granularity) {
  ASSERT_LT(index, model->resource_count) << ToString(expected_name);
  const loom_amdgpu_occupancy_resource_model_t* resource =
      &model->resources[index];
  EXPECT_EQ(ToString(resource->resource), ToString(expected_name));
  EXPECT_EQ(resource->pool_units, expected_pool_units)
      << ToString(expected_name);
  EXPECT_EQ(resource->allocation_granularity, expected_allocation_granularity)
      << ToString(expected_name);
  ASSERT_EQ(resource->member_count, 2u) << ToString(expected_name);
  EXPECT_EQ(resource->members[0].register_class_index, expected_vgpr_index)
      << ToString(expected_name);
  EXPECT_EQ(resource->members[0].contribution_granularity,
            expected_vgpr_contribution_granularity)
      << ToString(expected_name);
  EXPECT_EQ(resource->members[1].register_class_index, expected_agpr_index)
      << ToString(expected_name);
  EXPECT_EQ(resource->members[1].contribution_granularity,
            expected_agpr_contribution_granularity)
      << ToString(expected_name);
}

TEST_F(AmdgpuRegistersTest,
       SelectedDescriptorSetsExposeAddressableRegisterNamespaces) {
  struct DescriptorSetCase {
    // Descriptor-set key under test when selected into this build.
    iree_string_view_t descriptor_set_key;
    // Compiler-visible SGPR allocation namespace size.
    uint16_t sgpr_allocatable_count;
    // Compiler-visible VGPR allocation namespace size.
    uint16_t vgpr_allocatable_count;
    // True when the descriptor set exposes accumulator VGPR storage.
    bool has_agpr;
    // True when the descriptor set exposes the MODE register namespace.
    bool has_mode;
  };
  const DescriptorSetCase cases[] = {
      {
          IREE_SV("amdgpu.cdna3.core"),
          102,
          256,
          true,
          false,
      },
      {
          IREE_SV("amdgpu.cdna4.core"),
          106,
          256,
          true,
          false,
      },
      {
          IREE_SV("amdgpu.rdna3.core"),
          106,
          256,
          false,
          false,
      },
      {
          IREE_SV("amdgpu.rdna4.core"),
          106,
          256,
          false,
          false,
      },
      {
          IREE_SV("amdgpu.rdna4.gfx125x.core"),
          106,
          1024,
          false,
          true,
      },
  };

  iree_host_size_t checked_count = 0;
  for (const DescriptorSetCase& c : cases) {
    const loom_low_descriptor_set_t* descriptor_set =
        LookupDescriptorSet(c.descriptor_set_key);
    if (descriptor_set == nullptr) {
      continue;
    }
    ++checked_count;
    ExpectRegisterClass(descriptor_set, IREE_SV("amdgpu.sgpr"), 32,
                        c.sgpr_allocatable_count,
                        LOOM_LOW_REG_CLASS_FLAG_PHYSICAL);
    ExpectRegisterClass(descriptor_set, IREE_SV("amdgpu.vgpr"), 32,
                        c.vgpr_allocatable_count,
                        LOOM_LOW_REG_CLASS_FLAG_PHYSICAL);
    ExpectRegisterClass(
        descriptor_set, IREE_SV("amdgpu.scc"), 1, 1,
        LOOM_LOW_REG_CLASS_FLAG_PHYSICAL | LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE);
    ExpectRegisterClass(
        descriptor_set, IREE_SV("amdgpu.exec"), 64, 1,
        LOOM_LOW_REG_CLASS_FLAG_PHYSICAL | LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE);

    if (!c.has_agpr) {
      ExpectRegisterClassMissing(descriptor_set, IREE_SV("amdgpu.agpr"));
    } else {
      ExpectRegisterClass(descriptor_set, IREE_SV("amdgpu.agpr"), 32, 256,
                          LOOM_LOW_REG_CLASS_FLAG_PHYSICAL);
    }

    if (!c.has_mode) {
      ExpectRegisterClassMissing(descriptor_set, IREE_SV("amdgpu.mode"));
    } else {
      ExpectRegisterClass(descriptor_set, IREE_SV("amdgpu.mode"), 32, 1,
                          LOOM_LOW_REG_CLASS_FLAG_PHYSICAL |
                              LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE);
    }
  }
  ASSERT_GT(checked_count, 0u);
}

TEST_F(AmdgpuRegistersTest,
       SelectedCdnaMfmaOperandsExposeAccumulatorRegisterNamespace) {
  struct MfmaCase {
    // Descriptor-set key under test when selected into this build.
    iree_string_view_t descriptor_set_key;
  };
  const MfmaCase cases[] = {
      {IREE_SV("amdgpu.cdna3.core")},
      {IREE_SV("amdgpu.cdna4.core")},
  };

  iree_host_size_t checked_count = 0;
  for (const MfmaCase& c : cases) {
    const loom_low_descriptor_set_t* descriptor_set =
        LookupDescriptorSet(c.descriptor_set_key);
    if (descriptor_set == nullptr) {
      continue;
    }
    ++checked_count;
    uint16_t vgpr_id = LOOM_LOW_REG_CLASS_NONE;
    uint16_t agpr_id = LOOM_LOW_REG_CLASS_NONE;
    ExpectRegisterClass(descriptor_set, IREE_SV("amdgpu.vgpr"), 32, 256,
                        LOOM_LOW_REG_CLASS_FLAG_PHYSICAL, &vgpr_id);
    ExpectRegisterClass(descriptor_set, IREE_SV("amdgpu.agpr"), 32, 256,
                        LOOM_LOW_REG_CLASS_FLAG_PHYSICAL, &agpr_id);
    const uint32_t descriptor_ref = loom_amdgpu_descriptor_ref_ordinal(
        descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MFMA_F32_16X16X16_F16);
    ASSERT_NE(descriptor_ref, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE)
        << ToString(c.descriptor_set_key);
    ExpectCdnaMfmaVgprAgprOperands(descriptor_set, descriptor_ref, vgpr_id,
                                   agpr_id);
  }
  if (checked_count == 0) {
    GTEST_SKIP() << "No CDNA descriptor set selected.";
  }
}

TEST_F(AmdgpuRegistersTest, OccupancyPoolsStaySeparateFromAddressability) {
  struct OccupancyCase {
    // Generated descriptor-set ordinal for the occupancy model.
    uint16_t descriptor_set_ordinal;
    // SGPR register-file pool available to resident waves.
    uint32_t sgpr_pool_units;
    // SGPR allocation granularity used by occupancy calculations.
    uint32_t sgpr_allocation_granularity;
    // VGPR register-file pool available to resident waves.
    uint32_t vgpr_pool_units;
    // VGPR allocation granularity used by occupancy calculations.
    uint32_t vgpr_allocation_granularity;
    // AGPR register-file pool available to resident waves, or zero when absent.
    uint32_t agpr_pool_units;
    // AGPR allocation granularity used by occupancy calculations.
    uint32_t agpr_allocation_granularity;
    // Combined VGPR+AGPR resource pool, or zero when absent.
    uint32_t combined_pool_units;
    // Combined VGPR+AGPR allocation granularity.
    uint32_t combined_allocation_granularity;
  };
  const OccupancyCase cases[] = {
      {
          LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_CDNA3,
          800,
          16,
          512,
          8,
          256,
          4,
          512,
          8,
      },
      {
          LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_CDNA4,
          800,
          16,
          1024,
          4,
          256,
          4,
          512,
          8,
      },
      {
          LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3,
          800,
          16,
          1024,
          4,
          0,
          0,
          0,
          0,
      },
      {
          LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4,
          800,
          16,
          1024,
          4,
          0,
          0,
          0,
          0,
      },
      {
          LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX125X,
          800,
          16,
          1024,
          4,
          0,
          0,
          0,
          0,
      },
  };

  for (const OccupancyCase& c : cases) {
    const loom_amdgpu_occupancy_model_t* model =
        loom_amdgpu_occupancy_model_for_descriptor_set_ordinal(
            c.descriptor_set_ordinal);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->descriptor_set_ordinal, c.descriptor_set_ordinal);
    ExpectOccupancyRegisterClass(model, 0, IREE_SV("amdgpu.sgpr"),
                                 c.sgpr_pool_units,
                                 c.sgpr_allocation_granularity);
    ExpectOccupancyRegisterClass(model, 1, IREE_SV("amdgpu.vgpr"),
                                 c.vgpr_pool_units,
                                 c.vgpr_allocation_granularity);
    if (c.agpr_pool_units == 0) {
      EXPECT_EQ(model->register_class_count, 2u);
      EXPECT_EQ(model->resource_count, 0u);
    } else {
      ASSERT_EQ(model->register_class_count, 3u);
      ExpectOccupancyRegisterClass(model, 2, IREE_SV("amdgpu.agpr"),
                                   c.agpr_pool_units,
                                   c.agpr_allocation_granularity);
      ASSERT_EQ(model->resource_count, 1u);
      ExpectOccupancyPressureResource(
          model, 0, IREE_SV("amdgpu.vgpr_agpr"), c.combined_pool_units,
          c.combined_allocation_granularity, /*expected_vgpr_index=*/1,
          /*expected_vgpr_contribution_granularity=*/4,
          /*expected_agpr_index=*/2,
          /*expected_agpr_contribution_granularity=*/1);
    }
  }
}

}  // namespace
