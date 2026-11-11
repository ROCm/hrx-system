// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/lower/context.h"
#include "loom/ir/context.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/lower/abi.h"
#include "loom/target/arch/amdgpu/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/util/fact_table.h"

namespace {

loom_value_facts_t VaryingFacts() {
  loom_value_facts_t facts = loom_value_facts_unknown();
  loom_value_facts_mark_lane_varying(&facts);
  return facts;
}

loom_type_t VectorType(loom_scalar_type_t element_type, uint32_t count) {
  return loom_type_shaped_1d(LOOM_TYPE_VECTOR, element_type,
                             loom_dim_pack_static(count), 0);
}

class AmdgpuSourceTypeMappingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = loom_test_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TEST,
                                                 vtables, (uint16_t)count));
    vtables = loom_index_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_INDEX,
                                                 vtables, (uint16_t)count));
    vtables = loom_kernel_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_KERNEL,
                                                 vtables, (uint16_t)count));
    vtables = loom_low_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_LOW,
                                                 vtables, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("mapping"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    loom_amdgpu_low_descriptor_registry_initialize(&registry_);
    mapping_.descriptor_set = loom_low_descriptor_registry_lookup(
        &registry_.registry, IREE_SV("amdgpu.rdna3.core"));
    ASSERT_NE(mapping_.descriptor_set, nullptr);
    options_.policy = loom_amdgpu_low_lower_policy();
    options_.fact_table = &facts_;
    mapping_.module = module_;
    mapping_.options = &options_;
    mapping_.policy = options_.policy;
    mapping_.result = &result_;
    mapping_.fact_table = &facts_;
    mapping_.lowering = &frame_;
    iree_arena_initialize(&pool_, &mapping_.analysis_arena);
    mapping_.function_arena = &mapping_.analysis_arena;
    IREE_ASSERT_OK(
        loom_value_fact_table_initialize(&facts_, &module_->arena, 0));
  }

  void TearDown() override {
    loom_local_value_domain_release(&frame_.value_domain);
    loom_low_lower_result_deinitialize(&result_);
    iree_arena_deinitialize(&mapping_.analysis_arena);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  // A function argument returned unchanged supplies a valid source owner for
  // both point queries and the fallback whole-function analysis.
  void CreateValue(loom_type_t type, loom_value_facts_t facts) {
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    loom_string_id_t name = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(
        loom_builder_intern_string(&builder, IREE_SV("identity"), &name));
    uint16_t symbol = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
    loom_op_t* function = nullptr;
    IREE_ASSERT_OK(loom_test_func_build(&builder, 0, 0, 0, {0, symbol}, &type,
                                        1, &type, 1, nullptr, 0, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &function));
    mapping_.source_function = loom_func_like_cast(module_, function);
    loom_block_t* entry =
        loom_region_entry_block(loom_func_like_body(mapping_.source_function));
    value_ = loom_block_arg_id(entry, 0);
    loom_builder_initialize(module_, &module_->arena, entry, &builder);
    IREE_ASSERT_OK(loom_test_yield_build(&builder, &value_, 1,
                                         LOOM_LOCATION_UNKNOWN, &user_));
    IREE_ASSERT_OK(loom_value_fact_table_compute(&facts_, module_,
                                                 mapping_.source_function));
    IREE_ASSERT_OK(loom_value_fact_table_define(&facts_, value_, facts));
  }

  void ExpectDirectMapping(uint16_t register_class, uint32_t unit_count,
                           bool needs_analysis) {
    loom_type_t low_type = loom_type_none();
    IREE_ASSERT_OK(loom_amdgpu_map_value(
        nullptr, &mapping_, user_, value_,
        loom_module_value_type(module_, value_), &low_type));
    if (unit_count == 0) {
      EXPECT_TRUE(loom_type_equal(low_type, loom_type_none()));
    } else {
      EXPECT_EQ(loom_low_register_type_class_id(low_type), register_class);
      EXPECT_EQ(loom_low_register_type_unit_count(low_type), unit_count);
    }
    ExpectAnalysis(needs_analysis);
  }

  void InvokeIdentity(uint16_t register_class, uint32_t unit_count) {
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    loom_string_id_t name;
    loom_string_id_t representation;
    loom_symbol_id_t symbol;
    IREE_ASSERT_OK(
        loom_builder_intern_string(&builder, IREE_SV("native"), &name));
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
    IREE_ASSERT_OK(loom_builder_intern_string(
        &builder, IREE_SV("amdgpu.rdna3.core"), &representation));
    const loom_type_t carrier = loom_low_register_type(
        mapping_.descriptor_set->stable_id, register_class, unit_count);
    loom_op_t* helper;
    IREE_ASSERT_OK(loom_low_func_def_build(
        &builder, 0, 0, 0, 0, 0, 0, 0, 0, 0, representation,
        loom_symbol_ref_null(), 0, {}, {}, LOOM_STRING_ID_INVALID, {},
        {0, symbol}, &carrier, 1, &carrier, 1, nullptr, 0, nullptr, 0,
        LOOM_LOCATION_UNKNOWN, &helper));
    loom_block_t* entry =
        loom_region_entry_block(loom_low_func_def_body(helper));
    loom_builder_set_block(&builder, entry);
    const loom_value_id_t argument = loom_block_arg_id(entry, 0);
    loom_op_t* return_op;
    IREE_ASSERT_OK(loom_low_return_build(&builder, &argument, 1,
                                         LOOM_LOCATION_UNKNOWN, &return_op));
    loom_builder_set_before(&builder, user_);
    const loom_type_t type = loom_module_value_type(module_, value_);
    loom_op_t* invoke;
    IREE_ASSERT_OK(loom_low_invoke_build(&builder, 0, 0, 0, 0, {0, symbol},
                                         &value_, 1, &type, 1, nullptr, 0,
                                         LOOM_LOCATION_UNKNOWN, &invoke));
    value_ = loom_low_invoke_results(invoke).values[0];
    IREE_ASSERT_OK(loom_op_set_operand(module_, user_, 0, value_));
    IREE_ASSERT_OK(loom_value_fact_table_compute(&facts_, module_,
                                                 mapping_.source_function));
  }

  void ExpectContractMapping(
      uint16_t register_class, uint32_t unit_count, bool needs_analysis,
      loom_target_contract_vector_lane_projection_t projection = {}) {
    loom_target_contract_query_environment_t environment = {
        .module = module_,
        .function = mapping_.source_function,
        .descriptor_set = mapping_.descriptor_set,
        .fact_table = &facts_,
        .vector_lane_projection = projection,
        .function_analysis =
            {
                .fn =
                    [](void* user_data, loom_local_value_domain_t** out_domain,
                       const loom_view_region_table_t** out_views) {
                      auto* mapping =
                          static_cast<loom_low_lower_context_t*>(user_data);
                      IREE_RETURN_IF_ERROR(loom_low_lower_context_view_regions(
                          mapping, out_views));
                      *out_domain = &mapping->lowering->value_domain;
                      return iree_ok_status();
                    },
                .user_data = &mapping_,
            },
        .arena = &mapping_.analysis_arena,
        .target_state_allocator =
            {
                .fn =
                    [](void* user_data, const void* key, iree_host_size_t size,
                       void** out_data) {
                      return loom_low_lower_get_or_allocate_target_state(
                          static_cast<loom_low_lower_context_t*>(user_data),
                          key, size, out_data);
                    },
                .user_data = &mapping_,
            },
    };
    loom_low_lower_rule_mapped_value_t mapped = {};
    IREE_ASSERT_OK(loom_amdgpu_map_contract_value(nullptr, &environment, user_,
                                                  value_, &mapped));
    EXPECT_EQ(mapped.is_register, unit_count != 0);
    EXPECT_EQ(mapped.descriptor_register_class_id, register_class);
    EXPECT_EQ(mapped.register_unit_count, unit_count);
    ExpectAnalysis(needs_analysis);
  }

  void ExpectAnalysis(bool acquired) {
    EXPECT_EQ(mapping_.function_analysis.phase,
              acquired ? LOOM_LOW_LOWER_FUNCTION_ANALYSIS_VIEW_REGIONS
                       : LOOM_LOW_LOWER_FUNCTION_ANALYSIS_EMPTY);
    EXPECT_EQ(loom_local_value_domain_is_acquired(&frame_.value_domain),
              acquired);
  }

  // Storage backing the module and query-owned analyses.
  iree_arena_block_pool_t pool_ = {};
  // Dialect metadata used by the identity function.
  loom_context_t context_ = {};
  // Source IR and independently retained value facts.
  loom_module_t* module_ = nullptr;
  // Actual AMDGPU register descriptors.
  loom_target_low_descriptor_registry_t registry_ = {};
  // Facts supplied by the mapping caller.
  loom_value_fact_table_t facts_ = {};
  // Target capabilities borrowed by the lowering options.
  loom_target_facts_t target_facts_ = {};
  // Active policy and fact-table binding.
  loom_low_lower_options_t options_ = {};
  // Function-local lowering and analysis owner.
  loom_low_lower_context_t mapping_ = {};
  // Domain and target query state retained by the owner.
  loom_low_lowering_frame_t frame_ = {};
  // Diagnostic storage for the lowering context.
  loom_low_lower_result_t result_ = {};
  // Source argument queried by both mapping APIs.
  loom_value_id_t value_ = LOOM_VALUE_ID_INVALID;
  // Return operation consuming the source value.
  loom_op_t* user_ = nullptr;
};

TEST_F(AmdgpuSourceTypeMappingTest, PredicateArgumentTransportAndBodyWidths) {
  // Every supported width includes both sides of each four-byte ABI packing
  // boundary. The body has one two-word lane mask per predicate element.
  loom_type_t types[32];
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(types); ++i) {
    types[i] = VectorType(LOOM_SCALAR_TYPE_I1, i + 1);
  }
  loom_builder_t builder;
  loom_builder_initialize(module_, &module_->arena, loom_module_block(module_),
                          &builder);
  loom_string_id_t name;
  loom_symbol_id_t symbol;
  IREE_ASSERT_OK(loom_module_intern_string(
      module_, IREE_SV("predicate_arguments"), &name));
  IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
  loom_op_t* kernel = nullptr;
  IREE_ASSERT_OK(loom_kernel_def_build(
      &builder, 0, 0, 0, loom_symbol_ref_null(), LOOM_STRING_ID_INVALID, 0,
      {0, symbol}, nullptr, 0, types, IREE_ARRAYSIZE(types), nullptr, 0,
      LOOM_LOCATION_UNKNOWN, &kernel));
  loom_builder_enter_region(&builder, kernel, loom_kernel_def_config(kernel));
  loom_op_t* unit = nullptr;
  IREE_ASSERT_OK(loom_index_constant_build(
      &builder, loom_attr_i64(1), loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      LOOM_LOCATION_UNKNOWN, &unit));
  const auto one = loom_index_constant_result(unit);
  loom_op_t* config = nullptr;
  IREE_ASSERT_OK(loom_kernel_launch_config_build(
      &builder, 0, one, one, one, one, one, one, LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, LOOM_LOCATION_UNKNOWN,
      &config));
  loom_builder_enter_region(&builder, kernel, loom_kernel_def_body(kernel));
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(
      loom_kernel_return_build(&builder, LOOM_LOCATION_UNKNOWN, &return_op));
  mapping_.source_function = loom_func_like_cast(module_, kernel);
  target_facts_.storage.export_plan.abi_kind = LOOM_TARGET_ABI_HAL_KERNEL;
  loom_target_bundle_storage_rebind(&target_facts_.storage);
  options_.target_facts = &target_facts_;
  const auto* entry = loom_region_entry_block(loom_kernel_def_body(kernel));
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(types); ++i) {
    SCOPED_TRACE(i + 1);
    loom_low_lower_abi_argument_t argument = {};
    IREE_ASSERT_OK(loom_amdgpu_map_argument(
        nullptr, &mapping_, kernel, i, loom_block_arg_id(entry, i), &argument));
    EXPECT_EQ(argument.kind, LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT);
    EXPECT_EQ(loom_low_register_type_class_id(argument.abi_type),
              LOOM_AMDGPU_REG_CLASS_ID_SGPR);
    EXPECT_EQ(loom_low_register_type_unit_count(argument.abi_type),
              (i + 4) / 4);
    EXPECT_EQ(loom_low_register_type_class_id(argument.body_type),
              LOOM_AMDGPU_REG_CLASS_ID_SGPR);
    EXPECT_EQ(loom_low_register_type_unit_count(argument.body_type),
              2 * (i + 1));
  }
}

struct MappingCase {
  // Source representation being mapped.
  loom_type_t type;
  // Required physical register class.
  uint16_t register_class;
  // Required number of 32-bit register units.
  uint32_t unit_count;
};

class AmdgpuInvokeMappingTest
    : public AmdgpuSourceTypeMappingTest,
      public ::testing::WithParamInterface<MappingCase> {};

TEST_P(AmdgpuInvokeMappingTest, DeclaredCarrierControlsBothMappingPaths) {
  const auto& test = GetParam();
  CreateValue(test.type, loom_value_facts_unknown());
  InvokeIdentity(test.register_class, test.unit_count);
  ExpectDirectMapping(test.register_class, test.unit_count, true);
  ExpectContractMapping(test.register_class, test.unit_count, true);
}

INSTANTIATE_TEST_SUITE_P(
    NativeResults, AmdgpuInvokeMappingTest,
    ::testing::Values(MappingCase{loom_type_scalar(LOOM_SCALAR_TYPE_I32),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1},
                      MappingCase{loom_type_scalar(LOOM_SCALAR_TYPE_I64),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2},
                      MappingCase{loom_type_scalar(LOOM_SCALAR_TYPE_F16),
                                  LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1},
                      MappingCase{loom_type_scalar(LOOM_SCALAR_TYPE_F32),
                                  LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1},
                      MappingCase{loom_type_scalar(LOOM_SCALAR_TYPE_F64),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2},
                      MappingCase{loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1},
                      MappingCase{loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2},
                      MappingCase{loom_type_scalar(LOOM_SCALAR_TYPE_I1),
                                  LOOM_AMDGPU_REG_CLASS_ID_SCC, 1},
                      MappingCase{loom_type_scalar(LOOM_SCALAR_TYPE_I1),
                                  LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1},
                      MappingCase{loom_type_scalar(LOOM_SCALAR_TYPE_I1),
                                  LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2},
                      MappingCase{VectorType(LOOM_SCALAR_TYPE_I32, 4),
                                  LOOM_AMDGPU_REG_CLASS_ID_SGPR, 4},
                      MappingCase{VectorType(LOOM_SCALAR_TYPE_I64, 2),
                                  LOOM_AMDGPU_REG_CLASS_ID_SGPR, 4},
                      MappingCase{VectorType(LOOM_SCALAR_TYPE_F32, 4),
                                  LOOM_AMDGPU_REG_CLASS_ID_SGPR, 4},
                      MappingCase{VectorType(LOOM_SCALAR_TYPE_F64, 2),
                                  LOOM_AMDGPU_REG_CLASS_ID_SGPR, 4}));

class AmdgpuVaryingTypeMappingTest
    : public AmdgpuSourceTypeMappingTest,
      public ::testing::WithParamInterface<MappingCase> {};

TEST_P(AmdgpuVaryingTypeMappingTest, DirectMappingNeedsNoFunctionAnalysis) {
  CreateValue(GetParam().type, VaryingFacts());
  ExpectDirectMapping(GetParam().register_class, GetParam().unit_count, false);
}

TEST_P(AmdgpuVaryingTypeMappingTest, ContractMappingNeedsNoFunctionAnalysis) {
  CreateValue(GetParam().type, VaryingFacts());
  ExpectContractMapping(GetParam().register_class, GetParam().unit_count,
                        false);
}

// Fixed storage, value-sensitive banks, address widths, packed vectors, wide
// vectors, and predicate masks all keep their own representation contracts.
INSTANTIATE_TEST_SUITE_P(
    StorageFamilies, AmdgpuVaryingTypeMappingTest,
    ::testing::Values(MappingCase{loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1},
                      MappingCase{loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2},
                      MappingCase{loom_type_scalar(LOOM_SCALAR_TYPE_I8),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1},
                      MappingCase{loom_type_scalar(LOOM_SCALAR_TYPE_I16),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1},
                      MappingCase{loom_type_scalar(LOOM_SCALAR_TYPE_I32),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1},
                      MappingCase{loom_type_scalar(LOOM_SCALAR_TYPE_I64),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2},
                      MappingCase{loom_type_scalar(LOOM_SCALAR_TYPE_F8E4M3),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1},
                      MappingCase{loom_type_scalar(LOOM_SCALAR_TYPE_F8E5M2),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1},
                      MappingCase{loom_type_scalar(LOOM_SCALAR_TYPE_F16),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1},
                      MappingCase{loom_type_scalar(LOOM_SCALAR_TYPE_BF16),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1},
                      MappingCase{loom_type_scalar(LOOM_SCALAR_TYPE_F32),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1},
                      MappingCase{loom_type_scalar(LOOM_SCALAR_TYPE_F64),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2},
                      MappingCase{VectorType(LOOM_SCALAR_TYPE_I1, 4),
                                  LOOM_AMDGPU_REG_CLASS_ID_SGPR, 8},
                      MappingCase{VectorType(LOOM_SCALAR_TYPE_I8, 8),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2},
                      MappingCase{VectorType(LOOM_SCALAR_TYPE_I16, 4),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2},
                      MappingCase{VectorType(LOOM_SCALAR_TYPE_I32, 4),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 4},
                      MappingCase{VectorType(LOOM_SCALAR_TYPE_I64, 4),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 8},
                      MappingCase{VectorType(LOOM_SCALAR_TYPE_F8E4M3, 8),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2},
                      MappingCase{VectorType(LOOM_SCALAR_TYPE_F8E5M2, 8),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2},
                      MappingCase{VectorType(LOOM_SCALAR_TYPE_F16, 4),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2},
                      MappingCase{VectorType(LOOM_SCALAR_TYPE_BF16, 4),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2},
                      MappingCase{VectorType(LOOM_SCALAR_TYPE_F32, 4),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 4},
                      MappingCase{VectorType(LOOM_SCALAR_TYPE_F64, 4),
                                  LOOM_AMDGPU_REG_CLASS_ID_VGPR, 8}));

TEST_F(AmdgpuSourceTypeMappingTest, UnknownDistributionStillRequiresAnalysis) {
  CreateValue(loom_type_scalar(LOOM_SCALAR_TYPE_I32),
              loom_value_facts_unknown());
  ExpectContractMapping(LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, true);
}

TEST_F(AmdgpuSourceTypeMappingTest, UniformDistributionStillRequiresAnalysis) {
  loom_value_facts_t facts = loom_value_facts_unknown();
  loom_value_facts_mark_subgroup_uniform(&facts);
  CreateValue(loom_type_scalar(LOOM_SCALAR_TYPE_I32), facts);
  ExpectDirectMapping(LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, true);
}

TEST_F(AmdgpuSourceTypeMappingTest,
       VaryingBooleanRetainsRepresentationAnalysis) {
  loom_value_facts_t facts = VaryingFacts();
  facts.range_lo = 0;
  facts.range_hi = 1;
  CreateValue(loom_type_scalar(LOOM_SCALAR_TYPE_I1), facts);
  ExpectContractMapping(LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, true);
}

TEST_F(AmdgpuSourceTypeMappingTest, LanePredicateRetainsMaskAnalysis) {
  loom_value_facts_t facts = VaryingFacts();
  facts.range_lo = 0;
  facts.range_hi = 1;
  loom_value_facts_mark_lane_predicate(&facts);
  CreateValue(loom_type_scalar(LOOM_SCALAR_TYPE_I1), facts);
  ExpectContractMapping(LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, true);
}

TEST_F(AmdgpuSourceTypeMappingTest, ProjectedWidthDoesNotReuseAuthoredWidth) {
  CreateValue(VectorType(LOOM_SCALAR_TYPE_F64, 8), VaryingFacts());
  ExpectContractMapping(LOOM_AMDGPU_REG_CLASS_ID_VGPR, 16, false);
  ExpectContractMapping(LOOM_AMDGPU_REG_CLASS_ID_VGPR, 4, false, {8, 2});
  ExpectContractMapping(LOOM_AMDGPU_REG_CLASS_ID_VGPR, 8, false, {8, 4});
}

TEST_F(AmdgpuSourceTypeMappingTest, NarrowOffsetRetainsRangeSelectedWidth) {
  loom_value_facts_t facts = VaryingFacts();
  facts.range_lo = 0;
  facts.range_hi = UINT32_MAX;
  CreateValue(loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), facts);
  ExpectDirectMapping(LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, false);
  ExpectContractMapping(LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, false);
}

TEST_F(AmdgpuSourceTypeMappingTest, WideIndexRetainsRangeSelectedWidth) {
  loom_value_facts_t facts = VaryingFacts();
  facts.range_lo = 0;
  facts.range_hi = int64_t{1} << 40;
  CreateValue(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), facts);
  ExpectDirectMapping(LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2, false);
  ExpectContractMapping(LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2, false);
}

struct BufferCase {
  // Memory space established by the reference fact producer.
  loom_value_fact_memory_space_t memory_space;
  // Whether the source preserves the unqualified entry-buffer ABI.
  bool entry_origin;
  // Two for raw pointers, one for private scratch offsets, and zero otherwise.
  uint32_t unit_count;
};

class AmdgpuBufferMappingTest
    : public AmdgpuSourceTypeMappingTest,
      public ::testing::WithParamInterface<BufferCase> {};

TEST_P(AmdgpuBufferMappingTest, VaryingBufferRetainsMemorySpaceContract) {
  CreateValue(loom_type_buffer(), VaryingFacts());
  loom_value_fact_buffer_reference_t reference = {
      .maximum_byte_extent = loom_value_facts_unknown(),
      .minimum_alignment = 1,
      .memory_space = GetParam().memory_space,
      .root_value_id = value_,
      .alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
  };
  if (GetParam().entry_origin) {
    reference.origin = {
        .function_symbol_id =
            loom_test_func_callee(mapping_.source_function.op).symbol_id,
        .region_index = 0,
        .kind = LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY,
        .entry_value_id = value_,
    };
  }
  loom_value_facts_t facts = {};
  IREE_ASSERT_OK(loom_value_facts_make_buffer_reference(&facts_.context,
                                                        reference, &facts));
  loom_value_facts_mark_lane_varying(&facts);
  IREE_ASSERT_OK(loom_value_fact_table_define(&facts_, value_, facts));
  const uint16_t register_class = GetParam().unit_count != 0
                                      ? LOOM_AMDGPU_REG_CLASS_ID_VGPR
                                      : LOOM_LOW_REG_CLASS_NONE;
  ExpectDirectMapping(register_class, GetParam().unit_count, false);
  ExpectContractMapping(register_class, GetParam().unit_count, false);
}

INSTANTIATE_TEST_SUITE_P(
    ReferenceSpaces, AmdgpuBufferMappingTest,
    ::testing::Values(
        BufferCase{LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN, false, 0},
        BufferCase{LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN, true, 2},
        BufferCase{LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL, false, 2},
        BufferCase{LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT, false, 2},
        BufferCase{LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR, false, 2},
        BufferCase{LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP, false, 0},
        BufferCase{LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE, false, 1},
        BufferCase{LOOM_VALUE_FACT_MEMORY_SPACE_HOST, false, 0},
        BufferCase{LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC, false, 0}));

}  // namespace
