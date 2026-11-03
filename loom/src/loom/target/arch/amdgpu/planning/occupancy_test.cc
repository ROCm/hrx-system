// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/occupancy.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/planning/occupancy_model.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"

namespace loom {
namespace {

class AmdgpuOccupancyTargetResourcesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_amdgpu_occupancy_target_resources_t Build(
      iree_string_view_t processor_name, uint32_t wave_size,
      uint32_t scalar_register_count, uint32_t vector_register_count,
      uint32_t flat_workgroup_size, uint32_t local_memory_bytes,
      loom_target_residency_constraint_list_t* constraints = nullptr,
      uint32_t accumulator_register_count = 0) {
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_find_processor(processor_name);
    EXPECT_NE(processor, nullptr);
    const loom_amdgpu_occupancy_model_t* model =
        loom_amdgpu_occupancy_model_for_properties(&processor->properties,
                                                   wave_size);
    std::vector<uint32_t> direct_resource_units(
        model->descriptor_reg_class_count, 0);
    direct_resource_units[LOOM_AMDGPU_REG_CLASS_ID_SGPR] =
        scalar_register_count;
    direct_resource_units[LOOM_AMDGPU_REG_CLASS_ID_VGPR] =
        vector_register_count;
    for (iree_host_size_t i = 0; i < model->register_class_count; ++i) {
      const auto& resource = model->register_classes[i];
      if (iree_string_view_equal(resource.register_class,
                                 IREE_SV("amdgpu.agpr"))) {
        direct_resource_units[resource.descriptor_reg_class_id] =
            accumulator_register_count;
      }
    }
    loom_amdgpu_occupancy_target_resources_t resources = {};
    IREE_EXPECT_OK(loom_amdgpu_occupancy_build_target_resources(
        processor, wave_size, direct_resource_units.data(),
        direct_resource_units.size(), flat_workgroup_size, local_memory_bytes,
        &arena_, &resources, constraints));
    return resources;
  }

  loom_target_residency_view_t PlanningView(iree_string_view_t processor_name,
                                            uint32_t wave_size,
                                            uint32_t flat_workgroup_size,
                                            uint64_t local_memory_bytes) {
    loom_amdgpu_target_facts_t facts = {};
    facts.base.fact_type = &loom_amdgpu_target_fact_type;
    loom_amdgpu_target_identity_initialize(
        loom_amdgpu_target_info_find_target(processor_name), &facts.identity);
    facts.base.storage.snapshot.subgroup_size = wave_size;
    loom_target_fact_field_set_insert(&facts.base.explicit_fields,
                                      LOOM_TARGET_FACT_FIELD_SUBGROUP_SIZE);
    auto& export_plan = facts.base.storage.export_plan;
    export_plan.abi_kind = LOOM_TARGET_ABI_HAL_KERNEL;
    if (flat_workgroup_size != 0) {
      export_plan.hal_kernel.required_workgroup_size = {flat_workgroup_size, 1,
                                                        1};
    }
    loom_target_bundle_storage_rebind(&facts.base.storage);
    loom_amdgpu_target_facts_initialize(&facts);
    loom_low_resolved_target_t target = {.target_facts = &facts.base};
    loom_low_storage_layout_space_sizes_t storage_sizes = {
        .workgroup_bytes = local_memory_bytes,
    };
    return loom_amdgpu_occupancy_residency_view(&target, &storage_sizes);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
};

static const loom_target_residency_constraint_t* FindConstraint(
    const loom_target_residency_constraint_list_t& constraints,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < constraints.count; ++i) {
    if (iree_string_view_equal(constraints.rows[i].name, name)) {
      return &constraints.rows[i];
    }
  }
  return nullptr;
}

static const loom_amdgpu_occupancy_register_class_t* FindRegisterClass(
    const loom_amdgpu_occupancy_table_t& table, iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < table.register_class_count; ++i) {
    if (iree_string_view_equal(table.register_classes[i].register_class,
                               name)) {
      return &table.register_classes[i];
    }
  }
  return nullptr;
}

TEST_F(AmdgpuOccupancyTargetResourcesTest,
       PlanningCeilingMatchesFinalLaunchAccounting) {
  for (const auto* name :
       {"gfx1100", "gfx1151", "gfx1250", "gfx942", "gfx950"}) {
    SCOPED_TRACE(name);
    const auto processor_name = iree_make_cstring_view(name);
    const auto* processor =
        loom_amdgpu_target_info_find_processor(processor_name);
    ASSERT_NE(processor, nullptr);
    for (uint32_t wave_size : {32u, 64u}) {
      if (!loom_amdgpu_processor_properties_support_wavefront_size(
              &processor->properties, wave_size)) {
        continue;
      }
      SCOPED_TRACE(wave_size);
      for (uint32_t workgroup_size : {wave_size, 256u, 1024u}) {
        SCOPED_TRACE(workgroup_size);
        for (uint32_t local_memory_bytes : {0u, 1u, 65024u}) {
          SCOPED_TRACE(local_memory_bytes);
          const auto view = PlanningView(processor_name, wave_size,
                                         workgroup_size, local_memory_bytes);
          const auto final = Build(processor_name, wave_size, 0, 0,
                                   workgroup_size, local_memory_bytes);
          ASSERT_NE(view.model, nullptr);
          EXPECT_EQ(view.tier_limit, final.resident_waves_per_simd);
          EXPECT_LE(view.tier_limit, view.model->best_tier);
          EXPECT_EQ(view.required_tier, final.residency_summary.required_tier);
          EXPECT_TRUE(loom_target_residency_view_meets_required_tier(
              view, final.resident_waves_per_simd));
        }
      }
    }
  }
}

TEST_F(AmdgpuOccupancyTargetResourcesTest, UnknownLaunchDoesNotInventCeiling) {
  const auto view = PlanningView(IREE_SV("gfx1100"), 32, 0, 65024);
  ASSERT_NE(view.model, nullptr);
  EXPECT_EQ(view.tier_limit, view.model->best_tier);
  const auto final = Build(IREE_SV("gfx1100"), 32, 0, 0, 0, 65024);
  EXPECT_TRUE(iree_any_bit_set(
      final.residency_summary.flags,
      LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_UNKNOWN_WORKGROUP_SIZE));
}

TEST_F(AmdgpuOccupancyTargetResourcesTest, OversizedStorageHasZeroResidency) {
  const auto view = PlanningView(IREE_SV("gfx1100"), 32, 256, UINT64_MAX);
  ASSERT_NE(view.model, nullptr);
  EXPECT_EQ(view.tier_limit, 0u);
  const auto final = Build(IREE_SV("gfx1100"), 32, 0, 0, 256, UINT32_MAX);
  EXPECT_EQ(final.resident_waves_per_simd, 0u);
  const auto& summary = final.residency_summary;
  EXPECT_TRUE(loom_target_residency_summary_is_valid(&summary));
  EXPECT_EQ(summary.tier, 0u);
  EXPECT_EQ(summary.required_tier, 2u);
  EXPECT_TRUE(iree_any_bit_set(
      summary.flags,
      LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_REQUIRED_TIER_UNSATISFIED));
  EXPECT_TRUE(iree_any_bit_set(
      summary.flags, LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_NEXT_BETTER_TIER));
  EXPECT_GT(summary.next_better_tier, 0u);
  EXPECT_FALSE(iree_any_bit_set(
      summary.flags,
      LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_LIMITING_RESOURCE_NEXT_WORSE_TIER));
}

TEST_F(AmdgpuOccupancyTargetResourcesTest,
       RetainsJointRegisterAndLdsRequirements) {
  loom_target_residency_constraint_list_t constraints = {};
  const auto resources =
      Build(IREE_SV("gfx1151"), 64, 36, 88, 256, 15616, &constraints);
  ASSERT_EQ(constraints.count, 4u);
  EXPECT_EQ(resources.residency_summary.tier, 8u);
  EXPECT_EQ(resources.residency_summary.next_better_tier, 9u);
  EXPECT_EQ(resources.residency_summary.limiting_resource_count, 2u);
  const auto* vgpr = FindConstraint(constraints, IREE_SV("amdgpu.vgpr"));
  const auto* lds = FindConstraint(constraints, IREE_SV("amdgpu.lds"));
  ASSERT_NE(vgpr, nullptr);
  ASSERT_NE(lds, nullptr);
  for (const auto* constraint : {vgpr, lds}) {
    EXPECT_TRUE(iree_all_bits_set(
        constraint->flags,
        LOOM_TARGET_RESIDENCY_CONSTRAINT_FLAG_HAS_USAGE |
            LOOM_TARGET_RESIDENCY_CONSTRAINT_FLAG_HAS_TIER |
            LOOM_TARGET_RESIDENCY_CONSTRAINT_FLAG_HAS_LIMITING_RELATION |
            LOOM_TARGET_RESIDENCY_CONSTRAINT_FLAG_LIMITING |
            LOOM_TARGET_RESIDENCY_CONSTRAINT_FLAG_HAS_REDUCTION));
    EXPECT_EQ(constraint->tier, 8u);
  }
  EXPECT_EQ(vgpr->units, 88u);
  EXPECT_EQ(vgpr->rounded_units, 96u);
  EXPECT_EQ(vgpr->allocation_granularity, 12u);
  EXPECT_EQ(vgpr->pool_units, 768u);
  EXPECT_EQ(vgpr->reduction_units_to_next_better_tier, 4u);
  EXPECT_EQ(lds->units, 15616u);
  EXPECT_EQ(lds->rounded_units, 16384u);
  EXPECT_EQ(lds->allocation_granularity, 1024u);
  EXPECT_EQ(lds->pool_units, 131072u);
  EXPECT_EQ(lds->reduction_units_to_next_better_tier, 1280u);
  EXPECT_TRUE(iree_string_view_equal(lds->unit, IREE_SV("bytes")));
  EXPECT_TRUE(
      iree_string_view_equal(lds->allocation_scope, IREE_SV("workgroup")));
  const uint32_t next_vgpr =
      vgpr->units - vgpr->reduction_units_to_next_better_tier;
  const uint32_t next_lds =
      lds->units - lds->reduction_units_to_next_better_tier;
  EXPECT_EQ(Build(IREE_SV("gfx1151"), 64, 36, next_vgpr, 256, 15616)
                .resident_waves_per_simd,
            8u);
  EXPECT_EQ(Build(IREE_SV("gfx1151"), 64, 36, 88, 256, next_lds)
                .resident_waves_per_simd,
            8u);
  EXPECT_EQ(Build(IREE_SV("gfx1151"), 64, 36, next_vgpr, 256, next_lds)
                .resident_waves_per_simd,
            9u);
}

TEST_F(AmdgpuOccupancyTargetResourcesTest,
       ReportsPhysicalLocalMemoryGranularityAcrossFamilies) {
  struct TargetCase {
    // Exact processor whose final resource report is exercised.
    const char* processor;
    // Supported wave width selected for the processor.
    uint32_t wave_size;
    // Physical LDS allocation granularity in bytes per workgroup.
    uint32_t allocation_granularity;
    // Physical bytes reserved for an 11.5 KiB request.
    uint32_t rounded_units;
  };
  static constexpr TargetCase kCases[] = {
      {"gfx942", 64, 512, 11776},   {"gfx950", 64, 1280, 12800},
      {"gfx1100", 64, 1024, 12288}, {"gfx1151", 64, 1024, 12288},
      {"gfx1201", 64, 1024, 12288}, {"gfx1250", 32, 2048, 12288},
  };

  for (const TargetCase& test_case : kCases) {
    SCOPED_TRACE(test_case.processor);
    loom_target_residency_constraint_list_t constraints = {};
    Build(iree_make_cstring_view(test_case.processor), test_case.wave_size,
          /*scalar_register_count=*/4, /*vector_register_count=*/1,
          /*flat_workgroup_size=*/256, /*local_memory_bytes=*/11776,
          &constraints);
    const auto* lds = FindConstraint(constraints, IREE_SV("amdgpu.lds"));
    ASSERT_NE(lds, nullptr);
    EXPECT_EQ(lds->units, 11776u);
    EXPECT_EQ(lds->rounded_units, test_case.rounded_units);
    EXPECT_EQ(lds->allocation_granularity, test_case.allocation_granularity);
  }
}

TEST_F(AmdgpuOccupancyTargetResourcesTest,
       LimitsRdnaResidencyAtPhysicalLocalMemoryCliff) {
  for (const auto* processor : {"gfx1100", "gfx1151", "gfx1200", "gfx1201",
                                "gfx11-generic", "gfx12-generic"}) {
    SCOPED_TRACE(processor);
    const auto resources =
        Build(iree_make_cstring_view(processor), /*wave_size=*/64,
              /*scalar_register_count=*/4, /*vector_register_count=*/1,
              /*flat_workgroup_size=*/256, /*local_memory_bytes=*/11776);
    EXPECT_EQ(resources.resident_waves_per_simd, 10u);
  }
}

TEST_F(AmdgpuOccupancyTargetResourcesTest, DistinguishesNonlimitingRegisters) {
  loom_target_residency_constraint_list_t constraints = {};
  const auto resources =
      Build(IREE_SV("gfx1151"), 64, 36, 136, 128, 14848, &constraints);
  EXPECT_EQ(resources.residency_summary.next_better_tier, 5u);
  const auto* vgpr = FindConstraint(constraints, IREE_SV("amdgpu.vgpr"));
  const auto* sgpr = FindConstraint(constraints, IREE_SV("amdgpu.sgpr"));
  const auto* lds = FindConstraint(constraints, IREE_SV("amdgpu.lds"));
  ASSERT_NE(vgpr, nullptr);
  ASSERT_NE(sgpr, nullptr);
  ASSERT_NE(lds, nullptr);
  EXPECT_EQ(vgpr->tier, 5u);
  EXPECT_EQ(sgpr->kind,
            LOOM_TARGET_RESIDENCY_CONSTRAINT_UNCONSTRAINED_RESOURCE);
  EXPECT_EQ(sgpr->units, 36u);
  EXPECT_EQ(sgpr->rounded_units, 128u);
  EXPECT_EQ(sgpr->pool_units, 2048u);
  EXPECT_EQ(sgpr->allocation_granularity, 128u);
  EXPECT_EQ(sgpr->tier, 16u);
  for (const auto* constraint : {vgpr, sgpr}) {
    EXPECT_TRUE(iree_any_bit_set(
        constraint->flags,
        LOOM_TARGET_RESIDENCY_CONSTRAINT_FLAG_HAS_LIMITING_RELATION));
    EXPECT_FALSE(iree_any_bit_set(
        constraint->flags,
        LOOM_TARGET_RESIDENCY_CONSTRAINT_FLAG_LIMITING |
            LOOM_TARGET_RESIDENCY_CONSTRAINT_FLAG_HAS_REDUCTION));
  }
  EXPECT_EQ(lds->tier, 4u);
  EXPECT_EQ(lds->reduction_units_to_next_better_tier, 512u);
}

TEST_F(AmdgpuOccupancyTargetResourcesTest,
       ReportsPhysicalScalarRegisterBackingAcrossRdnaModels) {
  struct TargetCase {
    // Exact processor selecting one distinct RDNA occupancy model.
    const char* processor;
    // Supported wave width selecting one generated model row.
    uint32_t wave_size;
  };
  static constexpr TargetCase kCases[] = {
      {"gfx1100", 32}, {"gfx1100", 64}, {"gfx1151", 32}, {"gfx1151", 64},
      {"gfx1201", 32}, {"gfx1201", 64}, {"gfx1250", 32},
  };

  for (const TargetCase& test_case : kCases) {
    SCOPED_TRACE(test_case.processor);
    SCOPED_TRACE(test_case.wave_size);
    loom_target_residency_constraint_list_t constraints = {};
    Build(iree_make_cstring_view(test_case.processor), test_case.wave_size,
          /*scalar_register_count=*/1, /*vector_register_count=*/1,
          /*flat_workgroup_size=*/32, /*local_memory_bytes=*/0, &constraints);
    const auto* sgpr = FindConstraint(constraints, IREE_SV("amdgpu.sgpr"));
    ASSERT_NE(sgpr, nullptr);
    EXPECT_EQ(sgpr->kind,
              LOOM_TARGET_RESIDENCY_CONSTRAINT_UNCONSTRAINED_RESOURCE);
    EXPECT_EQ(sgpr->units, 1u);
    EXPECT_EQ(sgpr->rounded_units, 128u);
    EXPECT_EQ(sgpr->pool_units, 2048u);
    EXPECT_EQ(sgpr->allocation_granularity, 128u);
    EXPECT_EQ(sgpr->tier, 16u);
  }
}

TEST_F(AmdgpuOccupancyTargetResourcesTest,
       PreservesCdnaScalarAndAccumulatorRegisterContracts) {
  loom_target_residency_constraint_list_t constraints = {};
  Build(IREE_SV("gfx942"), 64, /*scalar_register_count=*/36,
        /*vector_register_count=*/1, /*flat_workgroup_size=*/64,
        /*local_memory_bytes=*/0, &constraints,
        /*accumulator_register_count=*/17);
  const auto* sgpr = FindConstraint(constraints, IREE_SV("amdgpu.sgpr"));
  const auto* agpr = FindConstraint(constraints, IREE_SV("amdgpu.agpr"));
  ASSERT_NE(sgpr, nullptr);
  ASSERT_NE(agpr, nullptr);

  EXPECT_EQ(sgpr->kind, LOOM_TARGET_RESIDENCY_CONSTRAINT_POOLED_RESOURCE);
  EXPECT_EQ(sgpr->units, 36u);
  EXPECT_EQ(sgpr->rounded_units, 48u);
  EXPECT_EQ(sgpr->pool_units, 800u);
  EXPECT_EQ(sgpr->allocation_granularity, 16u);

  EXPECT_EQ(agpr->kind,
            LOOM_TARGET_RESIDENCY_CONSTRAINT_UNCONSTRAINED_RESOURCE);
  EXPECT_EQ(agpr->units, 17u);
  EXPECT_EQ(agpr->rounded_units, 20u);
  EXPECT_EQ(agpr->pool_units, 256u);
  EXPECT_EQ(agpr->allocation_granularity, 4u);
}

TEST_F(AmdgpuOccupancyTargetResourcesTest,
       RetainsExactZeroForUnusedAccumulatorRegisters) {
  loom_target_residency_constraint_list_t constraints = {};
  const auto resources =
      Build(IREE_SV("gfx942"), 64, 36, 88, 64, 4096, &constraints);
  EXPECT_TRUE(
      loom_target_residency_summary_is_valid(&resources.residency_summary));
  EXPECT_FALSE(iree_any_bit_set(
      resources.residency_summary.flags,
      LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_INCOMPLETE_RESOURCE_COUNTS));
  for (const auto* name : {"amdgpu.agpr", "amdgpu.vgpr_agpr"}) {
    const auto* constraint =
        FindConstraint(constraints, iree_make_cstring_view(name));
    ASSERT_NE(constraint, nullptr);
    EXPECT_TRUE(iree_all_bits_set(
        constraint->flags,
        LOOM_TARGET_RESIDENCY_CONSTRAINT_FLAG_HAS_USAGE |
            LOOM_TARGET_RESIDENCY_CONSTRAINT_FLAG_HAS_TIER |
            LOOM_TARGET_RESIDENCY_CONSTRAINT_FLAG_HAS_LIMITING_RELATION));
  }
  const auto* accumulator = FindConstraint(constraints, IREE_SV("amdgpu.agpr"));
  ASSERT_NE(accumulator, nullptr);
  EXPECT_EQ(accumulator->units, 0u);
}

TEST_F(AmdgpuOccupancyTargetResourcesTest, CountsScratchSpillTraffic) {
  loom_amdgpu_target_facts_t facts = {};
  facts.base.fact_type = &loom_amdgpu_target_fact_type;
  loom_amdgpu_target_identity_initialize(
      loom_amdgpu_target_info_find_target(IREE_SV("gfx1100")), &facts.identity);
  facts.base.storage.snapshot.subgroup_size = 32;
  loom_target_fact_field_set_insert(&facts.base.explicit_fields,
                                    LOOM_TARGET_FACT_FIELD_SUBGROUP_SIZE);
  loom_target_bundle_storage_rebind(&facts.base.storage);
  loom_amdgpu_target_facts_initialize(&facts);

  const auto* model = loom_amdgpu_occupancy_model_for_properties(
      facts.properties.processor, /*wave_size=*/32);
  std::vector<uint32_t> physical_extents(model->descriptor_reg_class_count, 0);
  physical_extents[LOOM_AMDGPU_REG_CLASS_ID_VGPR] = 5;
  const loom_low_allocation_assignment_t assignment = {
      .descriptor_reg_class_id = LOOM_AMDGPU_REG_CLASS_ID_VGPR,
  };
  const loom_low_allocation_spill_plan_t spill_plan = {
      .assignment_index = 0,
      .slot_space = LOOM_LOW_SPILL_SLOT_SPACE_SCRATCH,
      .byte_size = 12,
      .store_count = 2,
      .reload_count = 3,
  };
  const loom_low_allocation_table_t allocation = {
      .target = {.target_facts = &facts.base},
      .assignments = &assignment,
      .assignment_count = 1,
      .physical_extents =
          {
              .ends_by_reg_class = physical_extents.data(),
              .count = physical_extents.size(),
          },
      .spill_plans = &spill_plan,
      .spill_plan_count = 1,
  };

  loom_amdgpu_occupancy_table_t table = {};
  IREE_ASSERT_OK(loom_amdgpu_occupancy_build(&allocation, /*options=*/nullptr,
                                             &arena_, &table));
  EXPECT_EQ(table.spill_count, 1u);
  EXPECT_EQ(table.scratch_spill_bytes, 12u);
  EXPECT_EQ(table.spill_store_count, 2u);
  EXPECT_EQ(table.spill_reload_count, 3u);
  const auto* vector = FindRegisterClass(table, IREE_SV("amdgpu.vgpr"));
  ASSERT_NE(vector, nullptr);
  EXPECT_EQ(vector->allocated_units, 5u);
  EXPECT_EQ(vector->spill_count, 1u);
  EXPECT_EQ(vector->spill_bytes, 12u);
  EXPECT_EQ(vector->spill_store_count, 2u);
  EXPECT_EQ(vector->spill_reload_count, 3u);
}

TEST_F(AmdgpuOccupancyTargetResourcesTest, AccountsForBothCdnaRegisterBanks) {
  for (const char* processor :
       {"gfx90a", "gfx942", "gfx950", "gfx9-4-generic"}) {
    SCOPED_TRACE(processor);
    for (uint32_t workgroup_size : {0u, 64u}) {
      for (uint32_t accumulators : {0u, 4u, 128u, 256u}) {
        SCOPED_TRACE(workgroup_size);
        SCOPED_TRACE(accumulators);
        loom_target_residency_constraint_list_t constraints = {};
        const auto resources =
            Build(iree_make_cstring_view(processor), 64, 36, 129,
                  workgroup_size, 0, &constraints, accumulators);
        const auto& summary = resources.residency_summary;
        EXPECT_FALSE(iree_any_bit_set(
            summary.flags,
            LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_INCOMPLETE_RESOURCE_COUNTS));
        EXPECT_EQ(loom_target_residency_summary_is_valid(&summary),
                  workgroup_size != 0);
        const auto* vector =
            FindConstraint(constraints, IREE_SV("amdgpu.vgpr"));
        const auto* accumulator =
            FindConstraint(constraints, IREE_SV("amdgpu.agpr"));
        const auto* combined =
            FindConstraint(constraints, IREE_SV("amdgpu.vgpr_agpr"));
        ASSERT_NE(vector, nullptr);
        ASSERT_NE(accumulator, nullptr);
        ASSERT_NE(combined, nullptr);
        EXPECT_EQ(vector->units, 129u);
        EXPECT_EQ(accumulator->units, accumulators);
        EXPECT_EQ(combined->units, 132u + accumulators);
        const uint32_t rounded_units = (132u + accumulators + 7u) & ~7u;
        EXPECT_EQ(combined->rounded_units, rounded_units);
        EXPECT_EQ(resources.resident_waves_per_simd, 512u / rounded_units);
        for (const auto* constraint : {vector, accumulator, combined}) {
          EXPECT_TRUE(iree_all_bits_set(
              constraint->flags,
              LOOM_TARGET_RESIDENCY_CONSTRAINT_FLAG_HAS_USAGE |
                  LOOM_TARGET_RESIDENCY_CONSTRAINT_FLAG_HAS_TIER));
        }
      }
    }
  }
}

TEST_F(AmdgpuOccupancyTargetResourcesTest, RetainsFixedLaunchCeiling) {
  loom_target_residency_constraint_list_t constraints = {};
  const auto resources =
      Build(IREE_SV("gfx1250"), 32, 36, 128, 64, 20480, &constraints);
  EXPECT_EQ(resources.residency_summary.tier, 8u);
  EXPECT_EQ(resources.residency_summary.limiting_resource_count, 3u);
  const auto* fixed =
      FindConstraint(constraints, IREE_SV("amdgpu.workgroup_slots"));
  ASSERT_NE(fixed, nullptr);
  EXPECT_EQ(fixed->kind, LOOM_TARGET_RESIDENCY_CONSTRAINT_FIXED_LIMIT);
  EXPECT_EQ(fixed->tier, 8u);
  EXPECT_TRUE(iree_any_bit_set(fixed->flags,
                               LOOM_TARGET_RESIDENCY_CONSTRAINT_FLAG_LIMITING));
  EXPECT_FALSE(iree_any_bit_set(
      fixed->flags, LOOM_TARGET_RESIDENCY_CONSTRAINT_FLAG_HAS_USAGE));
  for (iree_host_size_t i = 0; i < constraints.count; ++i) {
    EXPECT_FALSE(
        iree_any_bit_set(constraints.rows[i].flags,
                         LOOM_TARGET_RESIDENCY_CONSTRAINT_FLAG_HAS_REDUCTION));
  }
}

TEST_F(AmdgpuOccupancyTargetResourcesTest,
       ReportsSelectedGenericTargetTransition) {
  const loom_amdgpu_occupancy_target_resources_t resources =
      Build(IREE_SV("gfx11-generic"), /*wave_size=*/64,
            /*scalar_register_count=*/32, /*vector_register_count=*/160,
            /*flat_workgroup_size=*/128, /*local_memory_bytes=*/0);
  const loom_target_residency_summary_t& summary = resources.residency_summary;

  EXPECT_TRUE(loom_target_residency_summary_is_valid(&summary));
  EXPECT_TRUE(iree_all_bits_set(
      summary.flags,
      LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_NEXT_BETTER_TIER |
          LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_UNIQUE_LIMITING_RESOURCE |
          LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_LIMITING_RESOURCE_NEXT_WORSE_TIER));
  EXPECT_EQ(summary.best_tier, 16u);
  EXPECT_EQ(summary.tier, 3u);
  EXPECT_EQ(summary.next_better_tier, 4u);
  EXPECT_EQ(summary.limiting_resource_count, 1u);
  EXPECT_TRUE(iree_string_view_equal(summary.limiting_resource,
                                     IREE_SV("amdgpu.vgpr")));
  EXPECT_EQ(summary.limiting_resource_units, 160u);
  EXPECT_EQ(summary.limiting_resource_reduction_units_to_next_better_tier, 32u);
  EXPECT_GT(summary.limiting_resource_additional_units_to_next_worse_tier, 0u);
}

TEST_F(AmdgpuOccupancyTargetResourcesTest,
       DistinguishesConservativeGenericFromLargePoolExactProcessor) {
  const loom_target_residency_summary_t generic_summary =
      Build(IREE_SV("gfx11-generic"), /*wave_size=*/64,
            /*scalar_register_count=*/32, /*vector_register_count=*/160,
            /*flat_workgroup_size=*/128, /*local_memory_bytes=*/0)
          .residency_summary;
  const loom_target_residency_summary_t exact_summary =
      Build(IREE_SV("gfx1151"), /*wave_size=*/64,
            /*scalar_register_count=*/32, /*vector_register_count=*/160,
            /*flat_workgroup_size=*/128, /*local_memory_bytes=*/0)
          .residency_summary;

  EXPECT_TRUE(loom_target_residency_summary_is_valid(&generic_summary));
  EXPECT_TRUE(loom_target_residency_summary_is_valid(&exact_summary));
  EXPECT_EQ(generic_summary.tier, 3u);
  EXPECT_EQ(exact_summary.tier, 4u);
  EXPECT_EQ(generic_summary.next_better_tier, 4u);
  EXPECT_EQ(exact_summary.next_better_tier, 5u);
  EXPECT_GT(
      generic_summary.limiting_resource_reduction_units_to_next_better_tier,
      exact_summary.limiting_resource_reduction_units_to_next_better_tier);
  EXPECT_EQ(exact_summary.limiting_resource_reduction_units_to_next_better_tier,
            16u);
}

TEST_F(AmdgpuOccupancyTargetResourcesTest,
       ReportsTransitionWithExactUnusedAgprCount) {
  const loom_amdgpu_occupancy_target_resources_t resources =
      Build(IREE_SV("gfx942"), /*wave_size=*/64,
            /*scalar_register_count=*/32, /*vector_register_count=*/160,
            /*flat_workgroup_size=*/64, /*local_memory_bytes=*/0);

  EXPECT_TRUE(
      loom_target_residency_summary_is_valid(&resources.residency_summary));
  EXPECT_EQ(resources.residency_summary.tier, 3u);
  EXPECT_EQ(resources.residency_summary.required_tier, 1u);
}

TEST_F(AmdgpuOccupancyTargetResourcesTest,
       DistinguishesLaunchableLowOccupancyFromUnlaunchableWorkgroups) {
  struct TargetCase {
    const char* processor;
    uint32_t wave_size;
    uint32_t launchable_vgprs;
    uint32_t unlaunchable_vgprs;
  };
  static constexpr TargetCase kCases[] = {
      {"gfx1100", 32, 192, 205}, {"gfx1151", 64, 160, 193},
      {"gfx1250", 32, 128, 205}, {"gfx942", 64, 128, 169},
      {"gfx950", 64, 128, 169},
  };
  for (const auto& target : kCases) {
    SCOPED_TRACE(target.processor);
    const auto launchable =
        Build(iree_make_cstring_view(target.processor), target.wave_size,
              /*scalar_register_count=*/32, target.launchable_vgprs,
              /*flat_workgroup_size=*/1024, /*local_memory_bytes=*/0);
    EXPECT_TRUE(
        loom_target_residency_summary_is_valid(&launchable.residency_summary));
    EXPECT_EQ(launchable.waves_per_workgroup, 1024u / target.wave_size);
    EXPECT_GE(launchable.residency_summary.tier,
              launchable.residency_summary.required_tier);
    EXPECT_FALSE(iree_any_bit_set(
        launchable.residency_summary.flags,
        LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_REQUIRED_TIER_UNSATISFIED));

    const auto unlaunchable =
        Build(iree_make_cstring_view(target.processor), target.wave_size,
              /*scalar_register_count=*/32, target.unlaunchable_vgprs,
              /*flat_workgroup_size=*/1024, /*local_memory_bytes=*/0);
    EXPECT_TRUE(loom_target_residency_summary_is_valid(
        &unlaunchable.residency_summary));
    EXPECT_EQ(unlaunchable.waves_per_workgroup, 1024u / target.wave_size);
    EXPECT_LT(unlaunchable.residency_summary.tier,
              unlaunchable.residency_summary.required_tier);
    EXPECT_TRUE(iree_any_bit_set(
        unlaunchable.residency_summary.flags,
        LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_REQUIRED_TIER_UNSATISFIED));
  }
}

TEST_F(AmdgpuOccupancyTargetResourcesTest,
       OmitsTransitionWithoutExactLaunchInformation) {
  const loom_amdgpu_occupancy_target_resources_t resources =
      Build(IREE_SV("gfx11-generic"), /*wave_size=*/64,
            /*scalar_register_count=*/32, /*vector_register_count=*/160,
            /*flat_workgroup_size=*/0, /*local_memory_bytes=*/0);

  EXPECT_FALSE(
      loom_target_residency_summary_is_valid(&resources.residency_summary));
}

TEST_F(AmdgpuOccupancyTargetResourcesTest,
       ReportsTransitionLimitedByLocalMemory) {
  const loom_amdgpu_occupancy_target_resources_t resources =
      Build(IREE_SV("gfx11-generic"), /*wave_size=*/64,
            /*scalar_register_count=*/32, /*vector_register_count=*/81,
            /*flat_workgroup_size=*/512, /*local_memory_bytes=*/65536);

  const loom_target_residency_summary_t& summary = resources.residency_summary;
  EXPECT_TRUE(loom_target_residency_summary_is_valid(&summary));
  EXPECT_TRUE(iree_string_view_equal(resources.limiting_resource,
                                     IREE_SV("amdgpu.lds")));
  EXPECT_EQ(summary.tier, 4u);
  // LDS permits six waves after the reduction, but registers cap the result
  // at five. The requirement is still the same first LDS allocation cliff.
  EXPECT_EQ(summary.next_better_tier, 5u);
  EXPECT_EQ(summary.limiting_resource_count, 1u);
  EXPECT_EQ(summary.limiting_resource_reduction_units_to_next_better_tier,
            22528u);
}

TEST_F(AmdgpuOccupancyTargetResourcesTest, ReportsCapturedSwiGluLdsTransition) {
  const auto resources = Build(IREE_SV("gfx1151"), /*wave_size=*/64,
                               /*scalar_register_count=*/36,
                               /*vector_register_count=*/136,
                               /*flat_workgroup_size=*/128,
                               /*local_memory_bytes=*/14848);
  const auto& summary = resources.residency_summary;
  EXPECT_TRUE(loom_target_residency_summary_is_valid(&summary));
  EXPECT_EQ(summary.tier, 4u);
  EXPECT_EQ(summary.next_better_tier, 5u);
  EXPECT_EQ(summary.limiting_resource_count, 1u);
  EXPECT_TRUE(
      iree_string_view_equal(summary.limiting_resource, IREE_SV("amdgpu.lds")));
  EXPECT_EQ(summary.limiting_resource_units, 14848u);
  EXPECT_EQ(summary.limiting_resource_reduction_units_to_next_better_tier,
            512u);
  EXPECT_EQ(summary.limiting_resource_next_worse_cliff_units, 18433u);
  EXPECT_EQ(summary.limiting_resource_additional_units_to_next_worse_tier,
            3585u);
  EXPECT_EQ(summary.limiting_resource_next_worse_tier, 3u);

  EXPECT_EQ(Build(IREE_SV("gfx1151"), 64, 36, 136, 128, 14336)
                .resident_waves_per_simd,
            5u);
  EXPECT_EQ(Build(IREE_SV("gfx1151"), 64, 36, 136, 128, 14337)
                .resident_waves_per_simd,
            4u);
  EXPECT_EQ(Build(IREE_SV("gfx1151"), 64, 36, 136, 128, 18432)
                .resident_waves_per_simd,
            4u);
  EXPECT_EQ(Build(IREE_SV("gfx1151"), 64, 36, 136, 128, 18433)
                .resident_waves_per_simd,
            3u);
}

TEST_F(AmdgpuOccupancyTargetResourcesTest, RequiresBothTiedResourcesToImprove) {
  const auto resources = Build(IREE_SV("gfx1151"), /*wave_size=*/64,
                               /*scalar_register_count=*/36,
                               /*vector_register_count=*/160,
                               /*flat_workgroup_size=*/128,
                               /*local_memory_bytes=*/14848);
  const auto& summary = resources.residency_summary;
  EXPECT_TRUE(loom_target_residency_summary_is_valid(&summary));
  EXPECT_EQ(summary.tier, 4u);
  EXPECT_EQ(summary.next_better_tier, 5u);
  EXPECT_EQ(summary.limiting_resource_count, 2u);
  EXPECT_FALSE(iree_any_bit_set(
      summary.flags,
      LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_UNIQUE_LIMITING_RESOURCE));
  EXPECT_EQ(Build(IREE_SV("gfx1151"), 64, 36, 144, 128, 14848)
                .resident_waves_per_simd,
            4u);
  EXPECT_EQ(Build(IREE_SV("gfx1151"), 64, 36, 160, 128, 14336)
                .resident_waves_per_simd,
            4u);
  EXPECT_EQ(Build(IREE_SV("gfx1151"), 64, 36, 144, 128, 14336)
                .resident_waves_per_simd,
            5u);
}

TEST_F(AmdgpuOccupancyTargetResourcesTest, SkipsUnreachableLdsTiers) {
  const auto resources = Build(IREE_SV("gfx1100"), /*wave_size=*/32,
                               /*scalar_register_count=*/4,
                               /*vector_register_count=*/1,
                               /*flat_workgroup_size=*/32,
                               /*local_memory_bytes=*/2049);
  const auto& summary = resources.residency_summary;
  EXPECT_TRUE(loom_target_residency_summary_is_valid(&summary));
  EXPECT_EQ(summary.tier, 11u);
  EXPECT_EQ(summary.next_better_tier, 16u);
  EXPECT_EQ(summary.limiting_resource_reduction_units_to_next_better_tier, 1u);
  EXPECT_EQ(
      Build(IREE_SV("gfx1100"), 32, 4, 1, 32, 2048).resident_waves_per_simd,
      16u);
}

TEST_F(AmdgpuOccupancyTargetResourcesTest,
       ClipsRegisterTransitionToLdsCeiling) {
  const auto resources = Build(IREE_SV("gfx1151"), /*wave_size=*/64,
                               /*scalar_register_count=*/4,
                               /*vector_register_count=*/49,
                               /*flat_workgroup_size=*/128,
                               /*local_memory_bytes=*/5120);
  const auto& summary = resources.residency_summary;
  EXPECT_TRUE(loom_target_residency_summary_is_valid(&summary));
  EXPECT_EQ(summary.tier, 12u);
  EXPECT_EQ(summary.next_better_tier, 13u);
  EXPECT_EQ(summary.limiting_resource_count, 1u);
  EXPECT_TRUE(iree_string_view_equal(summary.limiting_resource,
                                     IREE_SV("amdgpu.vgpr")));
  EXPECT_EQ(summary.limiting_resource_reduction_units_to_next_better_tier, 1u);
  EXPECT_EQ(
      Build(IREE_SV("gfx1151"), 64, 4, 48, 128, 5120).resident_waves_per_simd,
      13u);
}

TEST_F(AmdgpuOccupancyTargetResourcesTest, RetainsFixedWorkgroupCeiling) {
  const auto resources = Build(IREE_SV("gfx1250"), /*wave_size=*/32,
                               /*scalar_register_count=*/4,
                               /*vector_register_count=*/1,
                               /*flat_workgroup_size=*/64,
                               /*local_memory_bytes=*/0);
  const auto& summary = resources.residency_summary;
  EXPECT_TRUE(loom_target_residency_summary_is_valid(&summary));
  EXPECT_EQ(summary.tier, 8u);
  EXPECT_EQ(summary.best_tier, 16u);
  EXPECT_EQ(summary.limiting_resource_count, 1u);
  EXPECT_FALSE(iree_any_bit_set(
      summary.flags, LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_NEXT_BETTER_TIER));
  // A fixed ceiling is not a reducible resource footprint.
  EXPECT_FALSE(iree_any_bit_set(
      summary.flags,
      LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_UNIQUE_LIMITING_RESOURCE));
  EXPECT_TRUE(iree_string_view_equal(resources.limiting_resource,
                                     IREE_SV("amdgpu.workgroup_slots")));

  const auto tied = Build(IREE_SV("gfx1250"), 32, 4, 128, 64, 20480);
  EXPECT_TRUE(loom_target_residency_summary_is_valid(&tied.residency_summary));
  EXPECT_EQ(tied.residency_summary.tier, 8u);
  EXPECT_EQ(tied.residency_summary.limiting_resource_count, 3u);
  EXPECT_FALSE(iree_any_bit_set(
      tied.residency_summary.flags,
      LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_NEXT_BETTER_TIER));
}

TEST_F(AmdgpuOccupancyTargetResourcesTest, BracketsLocalMemoryTransitions) {
  struct TargetCase {
    // Exact processor whose final-metadata API is exercised.
    const char* processor;
    // Selected wave width in lanes.
    uint32_t wave_size;
  };
  const TargetCase targets[] = {
      {"gfx1100", 32}, {"gfx1151", 64}, {"gfx1250", 32}, {"gfx1201", 64}};
  const uint32_t workgroup_sizes[] = {1, 32, 33, 64, 65, 128, 256, 512, 1024};
  const uint32_t local_sizes[] = {0,    1,    511,   512,   513,  2048,
                                  2049, 8704, 14848, 32769, 65536};
  for (const auto& target : targets) {
    SCOPED_TRACE(target.processor);
    SCOPED_TRACE(target.wave_size);
    for (uint32_t workgroup_size : workgroup_sizes) {
      SCOPED_TRACE(workgroup_size);
      for (uint32_t local_size : local_sizes) {
        SCOPED_TRACE(local_size);
        const auto resources =
            Build(iree_make_cstring_view(target.processor), target.wave_size,
                  /*scalar_register_count=*/4, /*vector_register_count=*/1,
                  workgroup_size, local_size);
        const auto& summary = resources.residency_summary;
        ASSERT_TRUE(loom_target_residency_summary_is_valid(&summary));
        EXPECT_EQ(summary.tier, resources.resident_waves_per_simd);
        if (!iree_any_bit_set(
                summary.flags,
                LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_NEXT_BETTER_TIER)) {
          continue;
        }
        ASSERT_TRUE(iree_string_view_equal(summary.limiting_resource,
                                           IREE_SV("amdgpu.lds")));
        ASSERT_GT(summary.limiting_resource_reduction_units_to_next_better_tier,
                  0u);
        const uint32_t boundary =
            local_size -
            summary.limiting_resource_reduction_units_to_next_better_tier;
        EXPECT_EQ(Build(iree_make_cstring_view(target.processor),
                        target.wave_size, 4, 1, workgroup_size, boundary)
                      .resident_waves_per_simd,
                  summary.next_better_tier);
        EXPECT_EQ(Build(iree_make_cstring_view(target.processor),
                        target.wave_size, 4, 1, workgroup_size, boundary + 1)
                      .resident_waves_per_simd,
                  summary.tier);
      }
    }
  }
}

TEST_F(AmdgpuOccupancyTargetResourcesTest,
       AppliesProcessorLaunchResourceLimits) {
  struct OccupancyCase {
    // Processor whose generated occupancy model is evaluated.
    const char* processor;
    // Wave width selected for the processor.
    uint32_t wave_size;
    // Fixed number of workitems in the workgroup.
    uint32_t flat_workgroup_size;
    // Fixed local-memory allocation in bytes.
    uint32_t local_memory_bytes;
    // Maximum resident waves admitted by the processor.
    uint32_t max_waves_per_simd;
    // Resident waves after applying launch resource limits.
    uint32_t resident_waves_per_simd;
    // Resident wave percentage after applying launch resource limits.
    uint32_t occupancy_percent;
    // Stable resource responsible for the resulting occupancy.
    const char* limiting_resource;
  };
  static constexpr OccupancyCase kCases[] = {
      {"gfx1100", 32, 64, 256, 16, 16, 100, "max_waves"},
      {"gfx1250", 32, 64, 256, 16, 8, 50, "amdgpu.workgroup_slots"},
      {"gfx1100", 32, 512, 32769, 16, 12, 75, "amdgpu.lds"},
      {"gfx942", 64, 512, 32769, 8, 2, 25, "amdgpu.lds"},
      {"gfx1250", 32, 512, 108545, 16, 8, 50, "amdgpu.lds"},
      {"gfx950", 64, 512, 108545, 8, 2, 25, "amdgpu.lds"},
  };

  for (const OccupancyCase& test_case : kCases) {
    SCOPED_TRACE(test_case.processor);
    const loom_amdgpu_occupancy_target_resources_t resources =
        Build(iree_make_cstring_view(test_case.processor), test_case.wave_size,
              /*scalar_register_count=*/4,
              /*vector_register_count=*/1, test_case.flat_workgroup_size,
              test_case.local_memory_bytes);

    EXPECT_TRUE(iree_string_view_equal(resources.scalar_register_class,
                                       IREE_SV("amdgpu.sgpr")));
    EXPECT_EQ(resources.scalar_register_count, 4u);
    EXPECT_TRUE(iree_string_view_equal(resources.vector_register_class,
                                       IREE_SV("amdgpu.vgpr")));
    EXPECT_EQ(resources.vector_register_count, 1u);
    EXPECT_EQ(resources.wave_size, test_case.wave_size);
    EXPECT_EQ(resources.max_waves_per_simd, test_case.max_waves_per_simd);
    EXPECT_EQ(resources.resident_waves_per_simd,
              test_case.resident_waves_per_simd);
    EXPECT_EQ(resources.occupancy_percent, test_case.occupancy_percent);
    EXPECT_TRUE(iree_string_view_equal(
        resources.limiting_resource,
        iree_make_cstring_view(test_case.limiting_resource)));
  }
}

}  // namespace
}  // namespace loom
