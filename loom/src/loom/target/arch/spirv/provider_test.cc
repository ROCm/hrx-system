// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/provider.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/target/facts.h"
#include "loom/target/arch/spirv/bda_abi.h"
#include "loom/target/arch/spirv/error_catalog.h"
#include "loom/target/arch/spirv/features.h"
#include "loom/target/arch/spirv/profile.h"
#include "loom/target/arch/spirv/records/target_records.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;
using ::loom::testing::CapturedDiagnosticEmission;
using ::loom::testing::DiagnosticCapture;
using ::loom::testing::DiagnosticEmissionCapture;

class SpirvProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_target_environment_initialize(
        &loom_spirv_target_provider_set, &target_environment_));
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_target_environment_register_context(
        &target_environment_, &context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_symbol_fact_table_initialize(&requirement_fact_table_,
                                      &analysis_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
    loom_target_environment_deinitialize(&target_environment_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr Parse(iree_string_view_t source) {
    DiagnosticCapture capture;
    loom_text_parse_options_t parse_options = {
        .diagnostic_sink = capture.sink(),
        .max_errors = 20,
    };
    const loom_target_low_descriptor_registry_t registry =
        loom_target_environment_low_descriptor_registry(&target_environment_);
    loom_low_descriptor_text_asm_environment_initialize(
        &registry.registry, &parse_options.low_asm_environment);
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(source, IREE_SV("spirv_provider_test.loom"),
                                  &context_, &block_pool_, &parse_options,
                                  &module));
    for (const auto& diagnostic : capture.diagnostics) {
      ADD_FAILURE() << loom_error_def_summary(diagnostic.error) << " at line "
                    << diagnostic.origin_line << ", column "
                    << diagnostic.origin_column;
    }
    IREE_ASSERT(module != nullptr);
    return ModulePtr(module);
  }

  void VerifyLow(loom_module_t* module, DiagnosticEmissionCapture* capture,
                 loom_low_verify_result_t* out_result) {
    const loom_target_low_descriptor_registry_t registry =
        loom_target_environment_low_descriptor_registry(&target_environment_);
    const loom_low_verify_options_t options = {
        .descriptor_registry = &registry.registry,
        .function_versions = nullptr,
        .emitter = capture->emitter(),
        .provider_list = loom_target_environment_low_verify_provider_list(
            &target_environment_),
        .max_errors = 20,
    };
    loom_low_verify_scratch_t scratch =
        loom_low_verify_scratch_for_module(module);
    IREE_ASSERT_OK(
        loom_low_verify_module(module, &options, &scratch, out_result));
  }

  std::string BuildLogicalRawBdaKernel(uint32_t argument_count) {
    std::string source =
        "spirv.target<vulkan1_3> @target {abi = hal_kernel}\n"
        "low.kernel.def target<spirv.logical.core>(@target) "
        "abi_layout({signature = (";
    source.reserve(256 + argument_count * 48u);
    for (uint32_t i = 0; i < argument_count; ++i) {
      if (i != 0) {
        source.append(",\n");
      }
      source.append("i32");
    }
    source.append(") -> ()}) workgroup_size(1, 1, 1) @many_parameters(");
    for (uint32_t i = 0; i < argument_count; ++i) {
      if (i != 0) {
        source.append(",\n");
      }
      source.append("%arg");
      source.append(std::to_string(i));
      source.append(": reg<spirv.id : i32>");
    }
    source.append(") asm {\n  return\n}\n");
    return source;
  }

  loom_symbol_ref_t FindSymbolRef(const loom_module_t* module,
                                  iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    const uint16_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    return loom_symbol_ref_t{
        .module_id = 0,
        .symbol_id = symbol_id,
    };
  }

  const loom_target_symbol_facts_t* Target(loom_symbol_fact_table_t* fact_table,
                                           const loom_module_t* module,
                                           loom_symbol_ref_t target_ref) {
    const loom_symbol_facts_base_t* base_facts = nullptr;
    IREE_CHECK_OK(loom_symbol_fact_table_lookup_ref(fact_table, module,
                                                    target_ref, &base_facts));
    const loom_target_symbol_facts_t* target_facts =
        loom_target_symbol_facts_cast(base_facts);
    IREE_ASSERT(target_facts != nullptr);
    return target_facts;
  }

  const loom_target_symbol_facts_t* Requirement(const loom_module_t* module,
                                                iree_string_view_t name) {
    return Target(&requirement_fact_table_, module,
                  FindSymbolRef(module, name));
  }

  bool SatisfiesSpecialization(const loom_target_facts_t* effective,
                               const loom_target_symbol_facts_t* requirement) {
    return loom_target_facts_satisfy_specialization_requirement(
        effective, requirement->projection);
  }

  bool SatisfiesIdentity(const loom_target_facts_t* effective,
                         const loom_target_symbol_facts_t* requirement) {
    return loom_target_facts_satisfy_identity_requirement(
        effective, requirement->projection);
  }

  // Block pool shared by parsed modules and analysis storage.
  iree_arena_block_pool_t block_pool_;

  // Finalized context containing the production SPIR-V dialect package.
  loom_context_t context_;

  // Production SPIR-V target provider environment under test.
  loom_target_environment_t target_environment_;

  // Arena retaining indexed target facts for both test modules.
  iree_arena_allocator_t analysis_arena_;

  // Dense target fact index for authored requirements.
  loom_symbol_fact_table_t requirement_fact_table_;
};

TEST_F(SpirvProviderTest, VerifiesReflectedRawBdaParameterOffsetBoundary) {
  const uint32_t maximum_argument_count =
      LOOM_SPIRV_BDA_MAX_REFLECTED_CONSTANT_WORD_OFFSET + 1u;
  {
    const std::string source = BuildLogicalRawBdaKernel(maximum_argument_count);
    ModulePtr module =
        Parse(iree_make_string_view(source.data(), source.size()));
    DiagnosticEmissionCapture capture;
    loom_low_verify_result_t result = {};
    VerifyLow(module.get(), &capture, &result);
    EXPECT_EQ(result.error_count, 0u);
    EXPECT_TRUE(capture.emissions.empty());
  }

  const std::string source =
      BuildLogicalRawBdaKernel(maximum_argument_count + 1u);
  ModulePtr module = Parse(iree_make_string_view(source.data(), source.size()));
  DiagnosticEmissionCapture capture;
  loom_low_verify_result_t result = {};
  VerifyLow(module.get(), &capture, &result);
  EXPECT_EQ(result.error_count, 1u);
  ASSERT_EQ(capture.emissions.size(), 1u);
  const CapturedDiagnosticEmission& emission = capture.emissions[0];
  EXPECT_EQ(emission.error, LOOM_ERR_SPIRV_031);
  ASSERT_EQ(emission.string_params.size(), 2u);
  EXPECT_EQ(emission.string_params[0], "many_parameters");
  EXPECT_EQ(emission.string_params[1], "arg16376");
  ASSERT_EQ(emission.u32_params.size(), 2u);
  EXPECT_EQ(emission.u32_params[0], 65504u);
  EXPECT_EQ(emission.u32_params[1], 65536u);
}

TEST_F(SpirvProviderTest, RetainsLowCallsWithoutStorageOrResources) {
  EXPECT_EQ(loom_spirv_target_provider.target_fact_type,
            &loom_spirv_target_fact_type);
  ASSERT_NE(loom_spirv_target_provider.select_call_policy, nullptr);
  const loom_resolved_target_t resolved_target = {};
  EXPECT_EQ(loom_spirv_target_provider.select_call_policy(
                &resolved_target, nullptr, LOOM_CALL_LIKE_KIND_LOW_INTERNAL,
                loom_call_like_t{}, loom_func_like_t{}),
            LOOM_TARGET_CALL_POLICY_NO_RESOURCES |
                LOOM_TARGET_CALL_POLICY_NO_STORAGE);
}

TEST_F(SpirvProviderTest, AdmitsNumericSourceSignatureFamily) {
  const loom_resolved_target_t resolved_target = {};
  for (const char* element :
       {"i1", "i8", "i16", "i32", "i64", "index", "offset", "f16", "bf16",
        "f32", "f64", "f8E4M3", "f8E5M2"}) {
    for (unsigned lanes : {0, 1, 2, 3, 4, 5, 8, 17}) {
      const std::string type =
          lanes ? "vector<" + std::to_string(lanes) + "x" + element + ">"
                : std::string(element);
      SCOPED_TRACE(type);
      const std::string source =
          "func.def @swap(%left: " + type + ", %right: " + type + ") -> (" +
          type + ", " + type + ") { func.return %right, %left : " + type +
          ", " + type + " }";
      ModulePtr module =
          Parse(iree_make_string_view(source.data(), source.size()));
      const loom_symbol_ref_t symbol =
          FindSymbolRef(module.get(), IREE_SV("swap"));
      const loom_func_like_t function = loom_func_like_cast(
          module.get(), module->symbols.entries[symbol.symbol_id].defining_op);
      EXPECT_EQ(loom_spirv_target_provider.select_call_policy(
                    &resolved_target, module.get(),
                    LOOM_CALL_LIKE_KIND_SEMANTIC, loom_call_like_t{}, function),
                LOOM_TARGET_CALL_POLICY_DIRECT);
    }
  }
}

TEST_F(SpirvProviderTest, MaterializesAuthoredRefinements) {
  ModulePtr source =
      Parse(IREE_SV("spirv.target<vulkan1_3> @source {abi = hal_kernel, "
                    "max_workgroup_size_x = 128, subgroup_size = 32, "
                    "export_symbol = \"roundtrip_export\", "
                    "contract_set_key = \"roundtrip.contract\", "
                    "contract_feature_bits = 8}\n"));
  const loom_target_symbol_facts_t* source_symbol_facts =
      Requirement(source.get(), IREE_SV("source"));

  loom_module_t* raw_module = nullptr;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("materialized"),
                                      &block_pool_, nullptr,
                                      iree_allocator_system(), &raw_module));
  ModulePtr materialized(raw_module);
  loom_string_id_t symbol_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(
      materialized.get(), IREE_SV("sealed_target"), &symbol_name_id));
  loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(materialized.get(), symbol_name_id, &symbol_id));
  const loom_symbol_ref_t symbol = {
      .module_id = 0,
      .symbol_id = symbol_id,
  };
  loom_builder_t builder;
  loom_builder_initialize(materialized.get(), &materialized->arena,
                          loom_module_block(materialized.get()), &builder);
  const loom_resolved_target_t resolved_target = {
      .provider = &loom_spirv_target_provider,
      .facts = source_symbol_facts->projection,
  };
  ASSERT_NE(loom_spirv_target_provider.materialize_definition, nullptr);
  IREE_ASSERT_OK(loom_spirv_target_provider.materialize_definition(
      &builder, &resolved_target, symbol, LOOM_LOCATION_UNKNOWN));

  loom_symbol_fact_table_reset(&requirement_fact_table_);
  const loom_target_symbol_facts_t* materialized_symbol_facts =
      Target(&requirement_fact_table_, materialized.get(), symbol);
  EXPECT_TRUE(loom_target_facts_are_equivalent(
      source_symbol_facts->projection, materialized_symbol_facts->projection));
}

TEST_F(SpirvProviderTest, MaterializesStructuredProfileOverrides) {
  const auto& preset = loom_spirv_low_target_bundle_vulkan1_3;
  loom_target_bundle_storage_t storage = {
      .snapshot = *preset.snapshot,
      .export_plan = *preset.export_plan,
      .config = *preset.config,
      .bundle = preset,
  };
  loom_target_bundle_storage_rebind(&storage);
  storage.snapshot.subgroup_size = 32;
  storage.snapshot.max_workgroup_size.x = 128;
  storage.snapshot.max_workgroup_storage_bytes = 32768;
  storage.export_plan.export_symbol = IREE_SV("profile_export");
  loom_target_fact_field_set_t explicit_fields = 0;
  for (auto field : {LOOM_TARGET_FACT_FIELD_SUBGROUP_SIZE,
                     LOOM_TARGET_FACT_FIELD_MAX_WORKGROUP_SIZE_X,
                     LOOM_TARGET_FACT_FIELD_MAX_WORKGROUP_STORAGE_BYTES,
                     LOOM_TARGET_FACT_FIELD_EXPORT_SYMBOL,
                     LOOM_TARGET_FACT_FIELD_INDEX_BITWIDTH}) {
    loom_target_fact_field_set_insert(&explicit_fields, field);
  }
  loom_spirv_target_profile_t profile = {};
  loom_spirv_target_profile_initialize(&storage.bundle, explicit_fields, {},
                                       nullptr, &profile);
  loom_target_facts_t* profile_facts = nullptr;
  IREE_ASSERT_OK(loom_target_profile_project_facts(
      &profile.base, &analysis_arena_, &profile_facts));

  loom_module_t* raw_module = nullptr;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("materialized"),
                                      &block_pool_, nullptr,
                                      iree_allocator_system(), &raw_module));
  ModulePtr module(raw_module);
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module.get(), IREE_SV("device"), &name_id));
  loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module.get(), name_id, &symbol_id));
  const loom_symbol_ref_t symbol = {0, symbol_id};
  loom_builder_t builder;
  loom_builder_initialize(module.get(), &module->arena,
                          loom_module_block(module.get()), &builder);
  const loom_resolved_target_t resolved = {&loom_spirv_target_provider,
                                           profile_facts};
  IREE_ASSERT_OK(loom_spirv_target_provider.materialize_definition(
      &builder, &resolved, symbol, LOOM_LOCATION_UNKNOWN));

  const auto* materialized =
      Target(&requirement_fact_table_, module.get(), symbol)->projection;
  EXPECT_EQ(materialized->storage.snapshot.subgroup_size, 32u);
  EXPECT_EQ(materialized->storage.snapshot.max_workgroup_size.x, 128u);
  EXPECT_EQ(materialized->storage.snapshot.max_workgroup_storage_bytes, 32768u);
  EXPECT_TRUE(
      iree_string_view_equal(materialized->storage.export_plan.export_symbol,
                             storage.export_plan.export_symbol));
  // An explicitly supplied preset value must survive as a constraint too.
  EXPECT_TRUE(loom_target_facts_field_is_explicit(
      materialized, LOOM_TARGET_FACT_FIELD_INDEX_BITWIDTH));
  EXPECT_TRUE(loom_target_facts_are_equivalent(profile_facts, materialized));
}

TEST_F(SpirvProviderTest, ProjectedProfileSatisfiesStructuredRequirements) {
  ModulePtr requirements =
      Parse(IREE_SV("spirv.target<vulkan1_3> @baseline_a\n"
                    "spirv.target<vulkan1_3> @baseline_b\n"
                    "spirv.target<vulkan1_3> @fitting "
                    "{max_workgroup_size_x = 128, subgroup_size = 32}\n"
                    "spirv.target<vulkan1_3> @too_large "
                    "{max_workgroup_size_x = 512}\n"
                    "spirv.target<vulkan1_3> @subgroup64 {subgroup_size = 64}\n"
                    "spirv.target<vulkan1_3> @index64 {index_bitwidth = 64}\n"
                    "spirv.target<vulkan1_3> @float16 "
                    "{contract_feature_bits = 8}\n"
                    "spirv.target<vulkan1_3> @float64 "
                    "{contract_feature_bits = 16}\n"));
  loom_target_bundle_storage_t live_storage = {
      .snapshot = *loom_spirv_low_target_bundle_vulkan1_3.snapshot,
      .export_plan = *loom_spirv_low_target_bundle_vulkan1_3.export_plan,
      .config = *loom_spirv_low_target_bundle_vulkan1_3.config,
      .bundle = loom_spirv_low_target_bundle_vulkan1_3,
  };
  loom_target_bundle_storage_rebind(&live_storage);
  live_storage.bundle.name = IREE_SV("live-vulkan-device");
  live_storage.snapshot.max_workgroup_size = {
      .x = 256,
      .y = 128,
      .z = 64,
  };
  live_storage.snapshot.max_flat_workgroup_size = 256;
  live_storage.snapshot.subgroup_size = 32;
  live_storage.snapshot.max_workgroup_count = {
      .x = 65535,
      .y = 65535,
      .z = 65535,
  };
  live_storage.export_plan.abi_kind = LOOM_TARGET_ABI_HAL_KERNEL;
  live_storage.config.contract_feature_bits |= LOOM_SPIRV_FEATURE_FLOAT16;

  loom_target_fact_field_set_t explicit_fields = 0;
  for (auto field : {LOOM_TARGET_FACT_FIELD_MAX_WORKGROUP_SIZE_X,
                     LOOM_TARGET_FACT_FIELD_MAX_WORKGROUP_SIZE_Y,
                     LOOM_TARGET_FACT_FIELD_MAX_WORKGROUP_SIZE_Z,
                     LOOM_TARGET_FACT_FIELD_MAX_FLAT_WORKGROUP_SIZE,
                     LOOM_TARGET_FACT_FIELD_SUBGROUP_SIZE,
                     LOOM_TARGET_FACT_FIELD_MAX_WORKGROUP_COUNT_X,
                     LOOM_TARGET_FACT_FIELD_MAX_WORKGROUP_COUNT_Y,
                     LOOM_TARGET_FACT_FIELD_MAX_WORKGROUP_COUNT_Z,
                     LOOM_TARGET_FACT_FIELD_ABI,
                     LOOM_TARGET_FACT_FIELD_CONTRACT_FEATURE_BITS}) {
    loom_target_fact_field_set_insert(&explicit_fields, field);
  }
  loom_spirv_target_profile_t profile = {};
  loom_spirv_target_profile_initialize(
      &live_storage.bundle, explicit_fields, {},
      /*cooperative_properties=*/nullptr, &profile);
  loom_target_facts_t* effective = nullptr;
  IREE_ASSERT_OK(loom_target_profile_project_facts(
      &profile.base, &analysis_arena_, &effective));
  ASSERT_NE(effective, nullptr);
  EXPECT_EQ(effective->storage.export_plan.abi_kind,
            LOOM_TARGET_ABI_HAL_KERNEL);

  const loom_target_symbol_facts_t* baseline_a =
      Requirement(requirements.get(), IREE_SV("baseline_a"));
  const loom_target_symbol_facts_t* baseline_b =
      Requirement(requirements.get(), IREE_SV("baseline_b"));
  EXPECT_TRUE(SatisfiesIdentity(effective, baseline_a));
  EXPECT_TRUE(SatisfiesIdentity(effective, baseline_b));
  EXPECT_TRUE(SatisfiesSpecialization(effective, baseline_a));
  EXPECT_TRUE(SatisfiesSpecialization(effective, baseline_b));
  EXPECT_TRUE(SatisfiesSpecialization(
      effective, Requirement(requirements.get(), IREE_SV("fitting"))));
  EXPECT_FALSE(SatisfiesSpecialization(
      effective, Requirement(requirements.get(), IREE_SV("too_large"))));
  EXPECT_FALSE(SatisfiesSpecialization(
      effective, Requirement(requirements.get(), IREE_SV("subgroup64"))));
  EXPECT_FALSE(SatisfiesSpecialization(
      effective, Requirement(requirements.get(), IREE_SV("index64"))));
  EXPECT_TRUE(SatisfiesSpecialization(
      effective, Requirement(requirements.get(), IREE_SV("float16"))));
  EXPECT_FALSE(SatisfiesSpecialization(
      effective, Requirement(requirements.get(), IREE_SV("float64"))));

  EXPECT_EQ(baseline_a->projection->storage.export_plan.abi_kind,
            LOOM_TARGET_ABI_SHADER_ENTRY_POINT);
  EXPECT_FALSE(loom_target_facts_field_is_explicit(baseline_a->projection,
                                                   LOOM_TARGET_FACT_FIELD_ABI));
}

}  // namespace
}  // namespace loom
