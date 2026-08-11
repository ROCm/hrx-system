// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/provider.h"

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/pass/ops.h"
#include "loom/ops/target/facts.h"
#include "loom/pass/builder.h"
#include "loom/pass/registry.h"
#include "loom/pass/testing/registry_verify.h"
#include "loom/pass/tooling.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/profile.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/testing/module_ptr.h"
#include "loom/transforms/symbol/template_selection.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

struct PipelineBuildData {
  const loom_target_environment_t* environment;
};

static iree_status_t BuildMaterializationPipeline(loom_builder_t* builder,
                                                  void* user_data) {
  const PipelineBuildData* data =
      static_cast<const PipelineBuildData*>(user_data);
  return loom_target_environment_contribute_pipeline(
      data->environment, LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_MATERIALIZATION,
      loom_pass_environment_empty(), builder);
}

static iree_string_view_t ModuleString(loom_module_t* module,
                                       loom_string_id_t id) {
  return module->strings.entries[id];
}

static iree_string_view_t AttrStringValue(loom_module_t* module,
                                          loom_attribute_t attr) {
  return ModuleString(module, loom_attr_as_string_id(attr));
}

static const loom_named_attr_t* FindAttr(loom_module_t* module,
                                         loom_named_attr_slice_t attrs,
                                         iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    if (iree_string_view_equal(ModuleString(module, attrs.entries[i].name_id),
                               name)) {
      return &attrs.entries[i];
    }
  }
  return nullptr;
}

static loom_amdgpu_target_identity_t MakeTargetIdentity(
    const loom_amdgpu_target_info_t* target) {
  loom_amdgpu_target_identity_t identity = {};
  loom_amdgpu_target_identity_initialize(target, &identity);
  return identity;
}

static void ExpectTargetFactsEqual(const loom_amdgpu_target_facts_t& expected,
                                   const loom_amdgpu_target_facts_t& actual) {
  EXPECT_EQ(actual.base.fact_type, expected.base.fact_type);
  EXPECT_EQ(actual.base.selector, expected.base.selector);
  EXPECT_EQ(actual.base.explicit_fields, expected.base.explicit_fields);

  const loom_target_snapshot_t& expected_snapshot =
      expected.base.storage.snapshot;
  const loom_target_snapshot_t& actual_snapshot = actual.base.storage.snapshot;
  EXPECT_EQ(actual_snapshot.codegen_format, expected_snapshot.codegen_format);
  EXPECT_EQ(actual_snapshot.artifact_format, expected_snapshot.artifact_format);
  EXPECT_EQ(actual_snapshot.default_pointer_bitwidth,
            expected_snapshot.default_pointer_bitwidth);
  EXPECT_EQ(actual_snapshot.index_bitwidth, expected_snapshot.index_bitwidth);
  EXPECT_EQ(actual_snapshot.offset_bitwidth, expected_snapshot.offset_bitwidth);
  EXPECT_EQ(actual_snapshot.max_workgroup_size.x,
            expected_snapshot.max_workgroup_size.x);
  EXPECT_EQ(actual_snapshot.max_workgroup_size.y,
            expected_snapshot.max_workgroup_size.y);
  EXPECT_EQ(actual_snapshot.max_workgroup_size.z,
            expected_snapshot.max_workgroup_size.z);
  EXPECT_EQ(actual_snapshot.max_flat_workgroup_size,
            expected_snapshot.max_flat_workgroup_size);
  EXPECT_EQ(actual_snapshot.max_workgroup_storage_bytes,
            expected_snapshot.max_workgroup_storage_bytes);
  EXPECT_EQ(actual_snapshot.subgroup_size, expected_snapshot.subgroup_size);
  EXPECT_EQ(actual_snapshot.max_grid_size.x, expected_snapshot.max_grid_size.x);
  EXPECT_EQ(actual_snapshot.max_grid_size.y, expected_snapshot.max_grid_size.y);
  EXPECT_EQ(actual_snapshot.max_grid_size.z, expected_snapshot.max_grid_size.z);
  EXPECT_EQ(actual_snapshot.max_flat_grid_size,
            expected_snapshot.max_flat_grid_size);
  EXPECT_EQ(actual_snapshot.max_workgroup_count.x,
            expected_snapshot.max_workgroup_count.x);
  EXPECT_EQ(actual_snapshot.max_workgroup_count.y,
            expected_snapshot.max_workgroup_count.y);
  EXPECT_EQ(actual_snapshot.max_workgroup_count.z,
            expected_snapshot.max_workgroup_count.z);
  EXPECT_EQ(actual_snapshot.memory_spaces.generic,
            expected_snapshot.memory_spaces.generic);
  EXPECT_EQ(actual_snapshot.memory_spaces.global,
            expected_snapshot.memory_spaces.global);
  EXPECT_EQ(actual_snapshot.memory_spaces.workgroup,
            expected_snapshot.memory_spaces.workgroup);
  EXPECT_EQ(actual_snapshot.memory_spaces.constant,
            expected_snapshot.memory_spaces.constant);
  EXPECT_EQ(actual_snapshot.memory_spaces.private_memory,
            expected_snapshot.memory_spaces.private_memory);
  EXPECT_EQ(actual_snapshot.memory_spaces.host,
            expected_snapshot.memory_spaces.host);
  EXPECT_EQ(actual_snapshot.memory_spaces.descriptor,
            expected_snapshot.memory_spaces.descriptor);

  const loom_target_export_plan_t& expected_export =
      expected.base.storage.export_plan;
  const loom_target_export_plan_t& actual_export =
      actual.base.storage.export_plan;
  EXPECT_EQ(actual_export.abi_kind, expected_export.abi_kind);
  EXPECT_EQ(actual_export.linkage, expected_export.linkage);
  EXPECT_TRUE(iree_string_view_equal(actual_export.export_symbol,
                                     expected_export.export_symbol));
  EXPECT_EQ(actual_export.hal_kernel.required_workgroup_size.x,
            expected_export.hal_kernel.required_workgroup_size.x);
  EXPECT_EQ(actual_export.hal_kernel.required_workgroup_size.y,
            expected_export.hal_kernel.required_workgroup_size.y);
  EXPECT_EQ(actual_export.hal_kernel.required_workgroup_size.z,
            expected_export.hal_kernel.required_workgroup_size.z);
  EXPECT_EQ(actual_export.hal_kernel.flat_workgroup_size_min,
            expected_export.hal_kernel.flat_workgroup_size_min);
  EXPECT_EQ(actual_export.hal_kernel.flat_workgroup_size_max,
            expected_export.hal_kernel.flat_workgroup_size_max);

  EXPECT_TRUE(
      iree_string_view_equal(actual.base.storage.config.contract_set_key,
                             expected.base.storage.config.contract_set_key));
  EXPECT_EQ(actual.base.storage.config.contract_feature_bits,
            expected.base.storage.config.contract_feature_bits);
  EXPECT_TRUE(
      loom_amdgpu_target_identity_equal(&actual.identity, &expected.identity));
}

class AmdgpuProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_target_environment_initialize(
        &loom_amdgpu_target_provider_set, &target_environment_));
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_target_environment_register_context(
        &target_environment_, &context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_symbol_fact_table_initialize(&fact_table_, &analysis_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
    loom_target_environment_deinitialize(&target_environment_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t AllocateModule(iree_string_view_t name, ModulePtr* out_module) {
    *out_module = nullptr;
    loom_module_t* module = nullptr;
    IREE_RETURN_IF_ERROR(loom_module_allocate(&context_, name, &block_pool_,
                                              nullptr, iree_allocator_system(),
                                              &module));
    *out_module = ModulePtr(module);
    return iree_ok_status();
  }

  ModulePtr Parse(iree_string_view_t source) {
    loom_text_parse_options_t parse_options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/20,
    };
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(source, IREE_SV("amdgpu_provider_test.loom"),
                                  &context_, &block_pool_, &parse_options,
                                  &module));
    IREE_ASSERT(module != nullptr);
    return ModulePtr(module);
  }

  loom_symbol_ref_t FindSymbolRef(const loom_module_t* module,
                                  iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    const uint16_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    return (loom_symbol_ref_t){
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
    };
  }

  const loom_target_symbol_facts_t* Target(const loom_module_t* module,
                                           loom_symbol_ref_t target_ref) {
    const loom_symbol_facts_base_t* base_facts = nullptr;
    IREE_CHECK_OK(loom_symbol_fact_table_lookup_ref(&fact_table_, module,
                                                    target_ref, &base_facts));
    const loom_target_symbol_facts_t* target_facts =
        loom_target_symbol_facts_cast(base_facts);
    IREE_ASSERT(target_facts != nullptr);
    return target_facts;
  }

  bool SatisfiesSpecialization(const loom_target_facts_t* effective,
                               const loom_target_facts_t* requirement) {
    return loom_target_facts_satisfy_specialization_requirement(effective,
                                                                requirement);
  }

  bool SatisfiesIdentity(const loom_target_facts_t* effective,
                         const loom_target_facts_t* requirement) {
    return loom_target_facts_satisfy_identity_requirement(effective,
                                                          requirement);
  }

  bool SatisfiesSpecialization(const loom_target_symbol_facts_t* effective,
                               const loom_target_symbol_facts_t* requirement) {
    return SatisfiesSpecialization(effective->projection,
                                   requirement->projection);
  }

  bool SatisfiesIdentity(const loom_target_symbol_facts_t* effective,
                         const loom_target_symbol_facts_t* requirement) {
    return SatisfiesIdentity(effective->projection, requirement->projection);
  }

  const loom_op_t* TargetOpFromRef(const loom_module_t* module,
                                   loom_symbol_ref_t target_ref) {
    IREE_ASSERT(loom_symbol_ref_is_valid(target_ref));
    IREE_ASSERT(target_ref.module_id == 0);
    IREE_ASSERT(target_ref.symbol_id < module->symbols.count);
    const loom_symbol_t* symbol =
        &module->symbols.entries[target_ref.symbol_id];
    IREE_ASSERT(symbol->defining_op != nullptr);
    return symbol->defining_op;
  }

  loom_symbol_ref_t SelectedCallee(const loom_module_t* module,
                                   iree_string_view_t function_name) {
    const loom_symbol_ref_t function_ref = FindSymbolRef(module, function_name);
    loom_func_like_t function = loom_func_like_cast(
        module, module->symbols.entries[function_ref.symbol_id].defining_op);
    IREE_ASSERT(loom_func_like_isa(function));
    loom_block_t* block =
        loom_region_entry_block(loom_func_like_body(function));
    IREE_ASSERT(block != nullptr);
    for (const loom_op_t* op = block->first_op; op != nullptr;
         op = op->next_op) {
      if (loom_func_call_isa(op)) {
        return loom_func_call_callee(op);
      }
    }
    return loom_symbol_ref_null();
  }

  iree_status_t RunTemplateSelection(loom_module_t* module) {
    static const loom_pass_descriptor_t kPassDescriptor = {
        /*.key=*/IREE_SVL("select-templates"),
        /*.info=*/loom_template_selection_pass_info,
        /*.module_run=*/{loom_template_selection_run},
        /*.create=*/loom_template_selection_create,
    };
    static const loom_pass_registry_t kPassRegistry = {
        /*.descriptors=*/&kPassDescriptor,
        /*.descriptor_count=*/1,
    };
    loom_low_pass_environment_storage_t environment_storage = {};
    const loom_pass_environment_t environment =
        loom_low_pass_environment_storage_initialize(
            /*descriptor_registry=*/nullptr,
            /*lower_policy_registry=*/nullptr,
            /*legality_provider_list=*/nullptr,
            /*legalizer_provider_list=*/nullptr,
            /*math_policy_registry=*/nullptr,
            /*compile_report=*/nullptr, &target_environment_,
            /*function_versions=*/nullptr, &environment_storage);
    const loom_pass_tool_run_options_t options = {
        /*.registry=*/&kPassRegistry,
        /*.environment=*/environment,
        /*.function_versions=*/nullptr,
        /*.predicate_provider=*/{},
        /*.block_pool=*/&block_pool_,
    };
    loom_pass_run_result_t result = {};
    IREE_RETURN_IF_ERROR(loom_pass_tool_run_flat_pipeline(
        module, IREE_SV("select-templates"), &options, &result));
    if (result.error_count != 0) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "template selection emitted diagnostics");
    }
    return iree_ok_status();
  }

  // Block pool shared by parsed modules and analysis storage.
  iree_arena_block_pool_t block_pool_;

  // Finalized context containing the production AMDGPU dialect package.
  loom_context_t context_;

  // Production AMDGPU target provider environment under test.
  loom_target_environment_t target_environment_;

  // Arena retaining indexed target facts for each test.
  iree_arena_allocator_t analysis_arena_;

  // Dense target fact index attached to the test's module.
  loom_symbol_fact_table_t fact_table_;
};

TEST_F(AmdgpuProviderTest, ProvidesSortedPassRegistry) {
  const loom_pass_registry_t* registry =
      loom_target_environment_pass_registry(&target_environment_);
  IREE_ASSERT_OK(loom_pass_registry_verify(registry));

  const loom_pass_descriptor_t* descriptor = nullptr;
  IREE_ASSERT_OK(loom_pass_registry_lookup(
      registry, IREE_SV("amdgpu-materialize-hal-kernel-abi"), &descriptor));
  ASSERT_NE(descriptor, nullptr);
  ASSERT_NE(descriptor->info, nullptr);
  EXPECT_EQ(descriptor->info()->kind, LOOM_PASS_FUNCTION);
}

TEST_F(AmdgpuProviderTest, ContributesHalKernelAbiMaterialization) {
  ModulePtr module;
  IREE_ASSERT_OK(AllocateModule(IREE_SV("pipeline"), &module));

  PipelineBuildData build_data = {
      /*.environment=*/&target_environment_,
  };
  loom_op_t* pipeline_op = nullptr;
  IREE_ASSERT_OK(loom_pass_ir_build_pipeline(
      module.get(), IREE_SV("emit"), LOOM_PASS_ANCHOR_MODULE,
      BuildMaterializationPipeline, &build_data, &pipeline_op));

  loom_block_t* pipeline_body =
      loom_region_entry_block(loom_pass_pipeline_body(pipeline_op));
  ASSERT_NE(pipeline_body, nullptr);
  ASSERT_EQ(pipeline_body->op_count, 2u);

  loom_op_t* where_op = pipeline_body->first_op;
  ASSERT_TRUE(loom_pass_where_isa(where_op));
  EXPECT_TRUE(iree_string_view_equal(
      ModuleString(module.get(), loom_pass_where_predicate(where_op)),
      IREE_SV("target")));

  loom_named_attr_slice_t attrs = loom_pass_where_attrs(where_op);
  ASSERT_EQ(attrs.count, 3u);
  const loom_named_attr_t* family_attr =
      FindAttr(module.get(), attrs, IREE_SV("family"));
  ASSERT_NE(family_attr, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      AttrStringValue(module.get(), family_attr->value), IREE_SV("amdgpu")));
  const loom_named_attr_t* codegen_attr =
      FindAttr(module.get(), attrs, IREE_SV("codegen"));
  ASSERT_NE(codegen_attr, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(AttrStringValue(module.get(), codegen_attr->value),
                             IREE_SV("low_native")));
  const loom_named_attr_t* abi_attr =
      FindAttr(module.get(), attrs, IREE_SV("abi"));
  ASSERT_NE(abi_attr, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      AttrStringValue(module.get(), abi_attr->value), IREE_SV("hal_kernel")));

  loom_block_t* where_body =
      loom_region_entry_block(loom_pass_where_body(where_op));
  ASSERT_NE(where_body, nullptr);
  ASSERT_EQ(where_body->op_count, 2u);
  loom_op_t* run_op = where_body->first_op;
  ASSERT_TRUE(loom_pass_run_isa(run_op));
  EXPECT_TRUE(iree_string_view_equal(
      ModuleString(module.get(), loom_pass_run_key(run_op)),
      IREE_SV("amdgpu-materialize-hal-kernel-abi")));
  EXPECT_TRUE(loom_pass_yield_isa(where_body->last_op));
  EXPECT_TRUE(loom_pass_yield_isa(pipeline_body->last_op));
}

TEST_F(AmdgpuProviderTest, ProvidesLoweringPolicyForEveryDescriptorSet) {
  loom_low_lower_policy_registry_t registry = {0};
  IREE_ASSERT_OK(loom_target_environment_initialize_low_lower_policy_registry(
      &target_environment_, &registry));

  const iree_host_size_t descriptor_set_count =
      loom_amdgpu_target_info_descriptor_set_count();
  ASSERT_NE(descriptor_set_count, 0u);
  for (iree_host_size_t i = 0; i < descriptor_set_count; ++i) {
    const loom_amdgpu_descriptor_set_info_t* descriptor_set =
        loom_amdgpu_target_info_descriptor_set_at((uint16_t)i);
    ASSERT_NE(descriptor_set, nullptr);
    EXPECT_NE(
        loom_low_lower_policy_registry_lookup(&registry, descriptor_set->key),
        nullptr)
        << "descriptor set ordinal " << i;
  }
}

TEST_F(AmdgpuProviderTest, MaterializesEveryStructuredProfile) {
  const iree_host_size_t target_count = loom_amdgpu_target_info_target_count();
  ASSERT_NE(target_count, 0u);
  for (iree_host_size_t i = 0; i < target_count; ++i) {
    const loom_amdgpu_target_info_t* target =
        loom_amdgpu_target_info_target_at(i);
    ASSERT_NE(target, nullptr);
    loom_amdgpu_target_identity_t identity = {};
    loom_amdgpu_target_identity_initialize(target, &identity);
    loom_amdgpu_target_profile_t profile = {};
    IREE_ASSERT_OK(loom_amdgpu_target_profile_initialize(&identity, &profile));
    loom_target_facts_t* profile_facts = nullptr;
    IREE_ASSERT_OK(loom_target_profile_project_facts(
        &profile.base, &analysis_arena_, &profile_facts));

    ModulePtr module;
    IREE_ASSERT_OK(AllocateModule(target->name, &module));
    loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(module.get(), IREE_SV("target"),
                                             &target_name_id));
    loom_symbol_id_t target_symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module.get(), target_name_id,
                                          &target_symbol_id));
    const loom_symbol_ref_t target_ref = {
        /*.module_id=*/0,
        /*.symbol_id=*/target_symbol_id,
    };
    loom_builder_t builder;
    loom_builder_initialize(module.get(), &module->arena,
                            loom_module_block(module.get()), &builder);
    const loom_resolved_target_t resolved_target = {
        /*.provider=*/&loom_amdgpu_target_provider,
        /*.facts=*/profile_facts,
    };
    loom_op_t* target_op = nullptr;
    IREE_ASSERT_OK(loom_amdgpu_target_provider.materialize_definition(
        &builder, &resolved_target, target_ref, LOOM_LOCATION_UNKNOWN,
        &target_op));
    ASSERT_NE(target_op, nullptr);

    loom_symbol_fact_table_reset(&fact_table_);
    const loom_target_symbol_facts_t* materialized_symbol_facts =
        Target(module.get(), target_ref);
    const loom_amdgpu_target_facts_t* materialized_facts =
        loom_amdgpu_target_facts_cast(materialized_symbol_facts->projection);
    ASSERT_NE(materialized_facts, nullptr);
    const loom_amdgpu_target_facts_t* expected_facts =
        loom_amdgpu_target_facts_cast(profile_facts);
    ASSERT_NE(expected_facts, nullptr);
    ExpectTargetFactsEqual(*expected_facts, *materialized_facts);
  }
}

TEST_F(AmdgpuProviderTest, MaterializesAuthoredRefinements) {
  ModulePtr source =
      Parse(IREE_SV("amdgpu.target<gfx942> @source {subgroup_size = 64, "
                    "max_workgroup_storage_bytes = 32768, "
                    "export_symbol = \"roundtrip_export\", "
                    "contract_set_key = \"roundtrip.contract\", "
                    "contract_feature_bits = 5, sramecc = on, xnack = off}\n"));
  const loom_target_symbol_facts_t* source_symbol_facts =
      Target(source.get(), FindSymbolRef(source.get(), IREE_SV("source")));
  const loom_amdgpu_target_facts_t* source_facts =
      loom_amdgpu_target_facts_cast(source_symbol_facts->projection);
  ASSERT_NE(source_facts, nullptr);

  ModulePtr materialized;
  IREE_ASSERT_OK(AllocateModule(IREE_SV("materialized"), &materialized));
  loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(
      materialized.get(), IREE_SV("sealed_target"), &target_name_id));
  loom_symbol_id_t target_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(materialized.get(), target_name_id,
                                        &target_symbol_id));
  const loom_symbol_ref_t target_ref = {
      /*.module_id=*/0,
      /*.symbol_id=*/target_symbol_id,
  };
  loom_builder_t builder;
  loom_builder_initialize(materialized.get(), &materialized->arena,
                          loom_module_block(materialized.get()), &builder);
  const loom_resolved_target_t resolved_target = {
      /*.provider=*/&loom_amdgpu_target_provider,
      /*.facts=*/source_symbol_facts->projection,
  };
  loom_op_t* target_op = nullptr;
  IREE_ASSERT_OK(loom_amdgpu_target_provider.materialize_definition(
      &builder, &resolved_target, target_ref, LOOM_LOCATION_UNKNOWN,
      &target_op));
  ASSERT_NE(target_op, nullptr);

  loom_symbol_fact_table_reset(&fact_table_);
  const loom_target_symbol_facts_t* materialized_symbol_facts =
      Target(materialized.get(), target_ref);
  const loom_amdgpu_target_facts_t* materialized_facts =
      loom_amdgpu_target_facts_cast(materialized_symbol_facts->projection);
  ASSERT_NE(materialized_facts, nullptr);
  ExpectTargetFactsEqual(*source_facts, *materialized_facts);
}

TEST_F(AmdgpuProviderTest, SeparatesIdentityAndSpecializationRequirements) {
  ModulePtr module =
      Parse(IREE_SV("amdgpu.target<gfx1151> @gfx1151_a\n"
                    "amdgpu.target<gfx1151> @gfx1151_b\n"
                    "amdgpu.target<gfx1151> @gfx1151_explicit_wave32 "
                    "{subgroup_size = 32}\n"
                    "amdgpu.target<gfx1150> @gfx1150\n"
                    "amdgpu.target<gfx11-generic> @gfx11_generic\n"
                    "amdgpu.target<gfx11-generic> @gfx11_wave32 "
                    "{subgroup_size = 32}\n"
                    "amdgpu.target<gfx11-generic> @gfx11_wave64 "
                    "{subgroup_size = 64}\n"
                    "amdgpu.target<gfx11-generic> @gfx11_small_workgroup "
                    "{max_workgroup_size_x = 128}\n"
                    "amdgpu.target<gfx11-generic> @gfx11_feature "
                    "{contract_feature_bits = 1}\n"
                    "amdgpu.target<gfx11-generic> @gfx11_explicit_contract "
                    "{contract_set_key = \"amdgpu.gfx11.generic.core\"}\n"
                    "amdgpu.target<gfx1170> @gfx1170\n"));

  const loom_target_symbol_facts_t* gfx1151_a =
      Target(module.get(), FindSymbolRef(module.get(), IREE_SV("gfx1151_a")));
  const loom_target_symbol_facts_t* gfx1151_b =
      Target(module.get(), FindSymbolRef(module.get(), IREE_SV("gfx1151_b")));
  const loom_target_symbol_facts_t* gfx1151_explicit_wave32 =
      Target(module.get(),
             FindSymbolRef(module.get(), IREE_SV("gfx1151_explicit_wave32")));
  const loom_target_symbol_facts_t* gfx1150 =
      Target(module.get(), FindSymbolRef(module.get(), IREE_SV("gfx1150")));
  const loom_target_symbol_facts_t* gfx11_generic = Target(
      module.get(), FindSymbolRef(module.get(), IREE_SV("gfx11_generic")));
  const loom_target_symbol_facts_t* gfx11_wave32 = Target(
      module.get(), FindSymbolRef(module.get(), IREE_SV("gfx11_wave32")));
  const loom_target_symbol_facts_t* gfx11_wave64 = Target(
      module.get(), FindSymbolRef(module.get(), IREE_SV("gfx11_wave64")));
  const loom_target_symbol_facts_t* gfx11_small_workgroup =
      Target(module.get(),
             FindSymbolRef(module.get(), IREE_SV("gfx11_small_workgroup")));
  const loom_target_symbol_facts_t* gfx11_feature = Target(
      module.get(), FindSymbolRef(module.get(), IREE_SV("gfx11_feature")));
  const loom_target_symbol_facts_t* gfx11_explicit_contract =
      Target(module.get(),
             FindSymbolRef(module.get(), IREE_SV("gfx11_explicit_contract")));
  const loom_target_symbol_facts_t* gfx1170 =
      Target(module.get(), FindSymbolRef(module.get(), IREE_SV("gfx1170")));

  const loom_amdgpu_target_facts_t* gfx1151_a_facts =
      loom_amdgpu_target_facts_cast(gfx1151_a->projection);
  const loom_amdgpu_target_facts_t* gfx1151_explicit_wave32_facts =
      loom_amdgpu_target_facts_cast(gfx1151_explicit_wave32->projection);
  const loom_amdgpu_target_facts_t* gfx11_generic_facts =
      loom_amdgpu_target_facts_cast(gfx11_generic->projection);
  const loom_amdgpu_target_facts_t* gfx11_explicit_contract_facts =
      loom_amdgpu_target_facts_cast(gfx11_explicit_contract->projection);
  ASSERT_NE(gfx1151_a_facts, nullptr);
  ASSERT_NE(gfx1151_explicit_wave32_facts, nullptr);
  ASSERT_NE(gfx11_generic_facts, nullptr);
  ASSERT_NE(gfx11_explicit_contract_facts, nullptr);
  EXPECT_EQ(gfx1151_a_facts->base.storage.snapshot.subgroup_size,
            gfx1151_explicit_wave32_facts->base.storage.snapshot.subgroup_size);
  EXPECT_FALSE(gfx1151_a_facts->subgroup_size_explicit);
  EXPECT_TRUE(gfx1151_explicit_wave32_facts->subgroup_size_explicit);
  EXPECT_FALSE(gfx11_generic_facts->contract_set_key_explicit);
  EXPECT_TRUE(gfx11_explicit_contract_facts->contract_set_key_explicit);

  // Identity is only the generated code-object domain and target-ID
  // coordinates. Common projection explicitness cannot narrow it.
  EXPECT_TRUE(SatisfiesIdentity(gfx1151_a, gfx1151_a));
  EXPECT_TRUE(SatisfiesIdentity(gfx1151_a, gfx1151_b));
  EXPECT_TRUE(SatisfiesIdentity(gfx1151_a, gfx11_generic));
  EXPECT_TRUE(SatisfiesIdentity(gfx1151_a, gfx11_wave32));
  EXPECT_TRUE(SatisfiesIdentity(gfx1151_a, gfx11_wave64));
  EXPECT_TRUE(SatisfiesIdentity(gfx1151_explicit_wave32, gfx11_wave64));
  EXPECT_TRUE(SatisfiesIdentity(gfx1151_a, gfx11_small_workgroup));
  EXPECT_TRUE(SatisfiesIdentity(gfx1151_a, gfx11_feature));
  EXPECT_TRUE(SatisfiesIdentity(gfx1151_a, gfx11_explicit_contract));
  EXPECT_FALSE(SatisfiesIdentity(gfx11_generic, gfx1151_a));
  EXPECT_FALSE(SatisfiesIdentity(gfx1151_a, gfx1150));
  EXPECT_FALSE(SatisfiesIdentity(gfx1170, gfx11_generic));

  // Full specialization retains authored execution choices, limits, and
  // artifact contract requirements.
  EXPECT_TRUE(SatisfiesSpecialization(gfx1151_a, gfx1151_a));
  EXPECT_TRUE(SatisfiesSpecialization(gfx1151_a, gfx1151_b));
  EXPECT_TRUE(SatisfiesSpecialization(gfx1151_a, gfx11_generic));
  EXPECT_TRUE(SatisfiesSpecialization(gfx1151_a, gfx11_wave32));
  EXPECT_TRUE(SatisfiesSpecialization(gfx1151_a, gfx11_small_workgroup));
  EXPECT_FALSE(SatisfiesSpecialization(gfx11_generic, gfx1151_a));
  EXPECT_FALSE(SatisfiesSpecialization(gfx1151_a, gfx1150));
  EXPECT_FALSE(SatisfiesSpecialization(gfx1170, gfx11_generic));
  EXPECT_TRUE(SatisfiesSpecialization(gfx1151_a, gfx11_wave64));
  EXPECT_FALSE(SatisfiesSpecialization(gfx1151_explicit_wave32, gfx11_wave64));
  EXPECT_FALSE(SatisfiesSpecialization(gfx1151_a, gfx11_feature));
  EXPECT_FALSE(SatisfiesSpecialization(gfx1151_a, gfx11_explicit_contract));
}

TEST_F(AmdgpuProviderTest, PreservesTargetIdFeatureRequirements) {
  ModulePtr module =
      Parse(IREE_SV("amdgpu.target<gfx942> @any\n"
                    "amdgpu.target<gfx942> @xnack_on {xnack = on}\n"
                    "amdgpu.target<gfx942> @xnack_off {xnack = off}\n"));

  const loom_target_symbol_facts_t* any =
      Target(module.get(), FindSymbolRef(module.get(), IREE_SV("any")));
  const loom_target_symbol_facts_t* xnack_on =
      Target(module.get(), FindSymbolRef(module.get(), IREE_SV("xnack_on")));
  const loom_target_symbol_facts_t* xnack_off =
      Target(module.get(), FindSymbolRef(module.get(), IREE_SV("xnack_off")));

  EXPECT_TRUE(SatisfiesSpecialization(xnack_on, any));
  EXPECT_TRUE(SatisfiesSpecialization(xnack_on, xnack_on));
  EXPECT_FALSE(SatisfiesSpecialization(xnack_on, xnack_off));
  EXPECT_FALSE(SatisfiesSpecialization(any, xnack_on));
  EXPECT_FALSE(SatisfiesSpecialization(xnack_off, xnack_on));

  EXPECT_TRUE(SatisfiesIdentity(xnack_on, any));
  EXPECT_TRUE(SatisfiesIdentity(xnack_on, xnack_on));
  EXPECT_FALSE(SatisfiesIdentity(xnack_on, xnack_off));
  EXPECT_FALSE(SatisfiesIdentity(any, xnack_on));
  EXPECT_FALSE(SatisfiesIdentity(xnack_off, xnack_on));
}

TEST_F(AmdgpuProviderTest, PreservesTargetOverlayRequirements) {
  ModulePtr module =
      Parse(IREE_SV("amdgpu.target<gfx1250-a0> @gfx1250_a0\n"
                    "amdgpu.target<gfx1250> @gfx1250_b0\n"
                    "amdgpu.target<gfx12-5-generic> @gfx12_5_generic\n"));

  const loom_target_symbol_facts_t* gfx1250_a0 =
      Target(module.get(), FindSymbolRef(module.get(), IREE_SV("gfx1250_a0")));
  const loom_target_symbol_facts_t* gfx1250_b0 =
      Target(module.get(), FindSymbolRef(module.get(), IREE_SV("gfx1250_b0")));
  const loom_target_symbol_facts_t* gfx12_5_generic = Target(
      module.get(), FindSymbolRef(module.get(), IREE_SV("gfx12_5_generic")));

  EXPECT_TRUE(SatisfiesSpecialization(gfx1250_a0, gfx1250_a0));
  EXPECT_TRUE(SatisfiesSpecialization(gfx1250_b0, gfx1250_b0));
  EXPECT_FALSE(SatisfiesSpecialization(gfx1250_a0, gfx1250_b0));
  EXPECT_FALSE(SatisfiesSpecialization(gfx1250_b0, gfx1250_a0));
  EXPECT_TRUE(SatisfiesSpecialization(gfx1250_a0, gfx12_5_generic));
  EXPECT_TRUE(SatisfiesSpecialization(gfx1250_b0, gfx12_5_generic));
  EXPECT_FALSE(SatisfiesSpecialization(gfx12_5_generic, gfx1250_a0));
}

TEST_F(AmdgpuProviderTest, ExhaustsSupportedProfileRelations) {
  struct ProjectedProfile {
    // Generated canonical target represented by the profile.
    const loom_amdgpu_target_info_t* target_info;

    // Compiler-owned facts projected from the profile.
    const loom_target_facts_t* facts;
  };

  std::vector<ProjectedProfile> profiles;
  const iree_host_size_t target_count = loom_amdgpu_target_info_target_count();
  for (iree_host_size_t i = 0; i < target_count; ++i) {
    const loom_amdgpu_target_info_t* target_info =
        loom_amdgpu_target_info_target_at(i);
    ASSERT_NE(target_info, nullptr);
    loom_amdgpu_target_profile_t profile = {};
    const loom_amdgpu_target_identity_t identity =
        MakeTargetIdentity(target_info);
    IREE_ASSERT_OK(loom_amdgpu_target_profile_initialize(&identity, &profile));
    loom_target_facts_t* facts = nullptr;
    IREE_ASSERT_OK(loom_target_profile_project_facts(&profile.base,
                                                     &analysis_arena_, &facts));
    ASSERT_NE(facts, nullptr);
    profiles.push_back({
        /*.target_info=*/target_info,
        /*.facts=*/facts,
    });
  }

  ASSERT_FALSE(profiles.empty());
  for (const ProjectedProfile& effective : profiles) {
    for (const ProjectedProfile& requirement : profiles) {
      const bool expected =
          loom_amdgpu_target_satisfies_code_object_requirement(
              effective.target_info, requirement.target_info);
      EXPECT_EQ(SatisfiesSpecialization(effective.facts, requirement.facts),
                expected)
          << "effective "
          << std::string(effective.target_info->name.data,
                         effective.target_info->name.size)
          << ", requirement "
          << std::string(requirement.target_info->name.data,
                         requirement.target_info->name.size);
      EXPECT_EQ(SatisfiesIdentity(effective.facts, requirement.facts), expected)
          << "effective "
          << std::string(effective.target_info->name.data,
                         effective.target_info->name.size)
          << ", requirement "
          << std::string(requirement.target_info->name.data,
                         requirement.target_info->name.size);
    }
  }
}

TEST_F(AmdgpuProviderTest, SelectsRequirementsPerFunctionTarget) {
  ModulePtr module = Parse(
      IREE_SV("amdgpu.target<gfx1150> @gfx1150\n"
              "amdgpu.target<gfx1151> @gfx1151\n"
              "amdgpu.target<gfx11-generic> @gfx11_generic\n"
              "amdgpu.target<gfx1170> @gfx1170\n"
              "\n"
              "func.template<demo.pick> target(@gfx1150) priority(30) "
              "@gfx1150_provider(%value: i32) -> (i32) {\n"
              "  func.return %value : i32\n"
              "}\n"
              "\n"
              "func.template<demo.pick> target(@gfx11_generic) priority(20) "
              "@gfx11_provider(%value: i32) -> (i32) {\n"
              "  func.return %value : i32\n"
              "}\n"
              "\n"
              "func.template<demo.pick> priority(1) "
              "@fallback(%value: i32) -> (i32) {\n"
              "  func.return %value : i32\n"
              "}\n"
              "\n"
              "func.def public target(@gfx1151) "
              "@entry_gfx1151(%value: i32) -> (i32) {\n"
              "  %result = func.apply<demo.pick>(%value) : (i32) -> (i32)\n"
              "  func.return %result : i32\n"
              "}\n"
              "\n"
              "func.def public target(@gfx1170) "
              "@entry_gfx1170(%value: i32) -> (i32) {\n"
              "  %result = func.apply<demo.pick>(%value) : (i32) -> (i32)\n"
              "  func.return %result : i32\n"
              "}\n"));

  const loom_symbol_ref_t generic_target_ref =
      FindSymbolRef(module.get(), IREE_SV("gfx11_generic"));
  const loom_op_t* generic_target_op =
      TargetOpFromRef(module.get(), generic_target_ref);

  IREE_ASSERT_OK(RunTemplateSelection(module.get()));

  const loom_symbol_ref_t rewritten_generic_provider_ref =
      FindSymbolRef(module.get(), IREE_SV("gfx11_provider"));
  const loom_symbol_ref_t gfx1151_callee =
      SelectedCallee(module.get(), IREE_SV("entry_gfx1151"));
  EXPECT_EQ(gfx1151_callee.module_id, rewritten_generic_provider_ref.module_id);
  EXPECT_EQ(gfx1151_callee.symbol_id, rewritten_generic_provider_ref.symbol_id);
  const loom_symbol_ref_t fallback_ref =
      FindSymbolRef(module.get(), IREE_SV("fallback"));
  const loom_symbol_ref_t gfx1170_callee =
      SelectedCallee(module.get(), IREE_SV("entry_gfx1170"));
  EXPECT_EQ(gfx1170_callee.module_id, fallback_ref.module_id);
  EXPECT_EQ(gfx1170_callee.symbol_id, fallback_ref.symbol_id);

  const loom_symbol_ref_t rewritten_generic_target_ref =
      FindSymbolRef(module.get(), IREE_SV("gfx11_generic"));
  EXPECT_EQ(TargetOpFromRef(module.get(), rewritten_generic_target_ref),
            generic_target_op);
  loom_func_like_t generic_provider = loom_func_like_cast(
      module.get(),
      module->symbols.entries[rewritten_generic_provider_ref.symbol_id]
          .defining_op);
  ASSERT_TRUE(loom_func_like_isa(generic_provider));
  const loom_symbol_ref_t provider_target =
      loom_func_like_target(generic_provider);
  EXPECT_EQ(provider_target.module_id, rewritten_generic_target_ref.module_id);
  EXPECT_EQ(provider_target.symbol_id, rewritten_generic_target_ref.symbol_id);
}

}  // namespace
}  // namespace loom
