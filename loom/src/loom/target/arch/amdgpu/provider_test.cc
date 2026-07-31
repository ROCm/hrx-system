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
#include "loom/target/arch/amdgpu/ops/ops.h"
#include "loom/target/arch/amdgpu/ops/target.h"
#include "loom/target/arch/amdgpu/profile.h"
#include "loom/target/arch/amdgpu/records/target_records.h"
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

  bool Satisfies(const loom_target_symbol_facts_t* effective,
                 const loom_target_symbol_facts_t* requirement) {
    return loom_target_facts_satisfy_requirement(effective->projection,
                                                 requirement->projection);
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
            /*specialization_context=*/nullptr,
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

TEST_F(AmdgpuProviderTest, MaterializesEveryProfileTargetKind) {
  ModulePtr module;
  IREE_ASSERT_OK(AllocateModule(IREE_SV("materialize"), &module));

  for (uint32_t target_kind = 1; target_kind < LOOM_AMDGPU_TARGET_KIND_COUNT_;
       ++target_kind) {
    const loom_amdgpu_target_record_info_t* target_record =
        loom_amdgpu_target_record_info_for_kind(target_kind);
    ASSERT_NE(target_record, nullptr) << "target kind " << target_kind;
    EXPECT_EQ(target_record->target_kind, target_kind);

    const loom_amdgpu_target_info_t* target_info = nullptr;
    IREE_ASSERT_OK(loom_amdgpu_target_info_lookup_target(
        target_record->target_name, &target_info));
    ASSERT_NE(target_info, nullptr);
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_target_processor(target_info);
    ASSERT_NE(processor, nullptr);
    const loom_amdgpu_target_identity_t identity =
        MakeTargetIdentity(target_info);
    loom_amdgpu_target_profile_t target_profile = {};
    IREE_ASSERT_OK(
        loom_amdgpu_target_profile_initialize(&identity, &target_profile));
    loom_symbol_ref_t target_ref = loom_symbol_ref_null();
    IREE_ASSERT_OK(loom_target_environment_materialize_effective_target(
        &target_environment_, module.get(), &target_profile.base,
        /*authored_target_op=*/nullptr, &target_ref));
    const loom_op_t* target_op = TargetOpFromRef(module.get(), target_ref);
    ASSERT_TRUE(loom_amdgpu_target_isa(target_op));
    EXPECT_EQ(loom_amdgpu_target_kind(target_op), target_kind);
    EXPECT_EQ(loom_amdgpu_target_record_target(target_op), target_info);
    EXPECT_TRUE(iree_string_view_equal(
        loom_amdgpu_target_record_target_name(target_op), target_info->name));
    EXPECT_TRUE(iree_string_view_equal(
        loom_amdgpu_target_record_processor_name(target_op), processor->name));

    const loom_target_symbol_facts_t* target = Target(module.get(), target_ref);
    loom_amdgpu_target_identity_t record_identity = {};
    loom_amdgpu_target_record_resolve_identity(target_op, &record_identity);
    EXPECT_EQ(record_identity.target, target_profile.identity.target);
    EXPECT_EQ(record_identity.amdhsa_features.sramecc,
              target_profile.identity.amdhsa_features.sramecc);
    EXPECT_EQ(record_identity.amdhsa_features.xnack,
              target_profile.identity.amdhsa_features.xnack);

    loom_amdgpu_target_properties_t record_properties = {};
    loom_amdgpu_target_record_resolve_properties(
        target_op, &target->projection->storage.bundle, &record_properties);
    EXPECT_EQ(record_properties.target, target_profile.properties.target);
    EXPECT_EQ(record_properties.processor, target_profile.properties.processor);
    EXPECT_EQ(record_properties.common, &target->projection->storage.bundle);
    EXPECT_EQ(record_properties.instruction_constraints,
              target_profile.properties.instruction_constraints);
    EXPECT_TRUE(loom_target_snapshot_satisfies_requirement(
        record_properties.common->snapshot,
        target_profile.properties.common->snapshot));
    EXPECT_TRUE(loom_target_snapshot_satisfies_requirement(
        target_profile.properties.common->snapshot,
        record_properties.common->snapshot));

    loom_amdgpu_target_profile_t equal_profile = {};
    IREE_ASSERT_OK(
        loom_amdgpu_target_profile_initialize(&identity, &equal_profile));
    loom_symbol_ref_t reused_ref = loom_symbol_ref_null();
    IREE_ASSERT_OK(loom_target_environment_materialize_effective_target(
        &target_environment_, module.get(), &equal_profile.base,
        /*authored_target_op=*/nullptr, &reused_ref));
    EXPECT_EQ(reused_ref.module_id, target_ref.module_id);
    EXPECT_EQ(reused_ref.symbol_id, target_ref.symbol_id);
  }
}

TEST_F(AmdgpuProviderTest,
       MaterializesTargetsSharingAProcessorAsDistinctRecords) {
  ModulePtr module;
  IREE_ASSERT_OK(AllocateModule(IREE_SV("materialize_overlays"), &module));

  iree_host_size_t shared_processor_pair_count = 0;
  const iree_host_size_t target_count = loom_amdgpu_target_info_target_count();
  for (iree_host_size_t lhs_ordinal = 0; lhs_ordinal < target_count;
       ++lhs_ordinal) {
    const loom_amdgpu_target_info_t* lhs =
        loom_amdgpu_target_info_target_at(lhs_ordinal);
    ASSERT_NE(lhs, nullptr);
    for (iree_host_size_t rhs_ordinal = lhs_ordinal + 1;
         rhs_ordinal < target_count; ++rhs_ordinal) {
      const loom_amdgpu_target_info_t* rhs =
          loom_amdgpu_target_info_target_at(rhs_ordinal);
      ASSERT_NE(rhs, nullptr);
      if (lhs->processor_ordinal != rhs->processor_ordinal) {
        continue;
      }

      loom_symbol_ref_t refs[2] = {};
      const loom_amdgpu_target_info_t* targets[] = {lhs, rhs};
      for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(targets); ++i) {
        const loom_amdgpu_target_identity_t identity =
            MakeTargetIdentity(targets[i]);
        loom_amdgpu_target_profile_t profile = {};
        IREE_ASSERT_OK(
            loom_amdgpu_target_profile_initialize(&identity, &profile));
        IREE_ASSERT_OK(loom_target_environment_materialize_effective_target(
            &target_environment_, module.get(), &profile.base,
            /*authored_target_op=*/nullptr, &refs[i]));

        const loom_op_t* target_op = TargetOpFromRef(module.get(), refs[i]);
        EXPECT_EQ(loom_amdgpu_target_record_target(target_op), targets[i]);
        const loom_target_symbol_facts_t* target =
            Target(module.get(), refs[i]);
        loom_amdgpu_target_properties_t properties = {};
        loom_amdgpu_target_record_resolve_properties(
            target_op, &target->projection->storage.bundle, &properties);
        EXPECT_EQ(properties.target, targets[i]);
        EXPECT_EQ(properties.instruction_constraints,
                  targets[i]->instruction_constraints);
      }
      EXPECT_NE(refs[0].symbol_id, refs[1].symbol_id);
      ++shared_processor_pair_count;
    }
  }
  EXPECT_GT(shared_processor_pair_count, 0u);
}

TEST_F(AmdgpuProviderTest, ReusesAuthoredFeatureRecordAndSeparatesVariant) {
  ModulePtr module =
      Parse(IREE_SV("amdgpu.target<gfx942> @authored "
                    "{sramecc = on, xnack = off}\n"));

  const loom_amdgpu_target_info_t* target = nullptr;
  IREE_ASSERT_OK(
      loom_amdgpu_target_info_lookup_target(IREE_SV("gfx942"), &target));
  ASSERT_NE(target, nullptr);

  loom_amdgpu_target_identity_t authored_identity = MakeTargetIdentity(target);
  authored_identity.amdhsa_features.sramecc = LOOM_AMDGPU_TARGET_FEATURE_ON;
  authored_identity.amdhsa_features.xnack = LOOM_AMDGPU_TARGET_FEATURE_OFF;
  loom_amdgpu_target_profile_t authored_profile = {};
  IREE_ASSERT_OK(loom_amdgpu_target_profile_initialize(&authored_identity,
                                                       &authored_profile));
  loom_symbol_ref_t authored_ref = loom_symbol_ref_null();
  IREE_ASSERT_OK(loom_target_environment_materialize_effective_target(
      &target_environment_, module.get(), &authored_profile.base,
      /*authored_target_op=*/nullptr, &authored_ref));
  const loom_symbol_ref_t expected_authored_ref =
      FindSymbolRef(module.get(), IREE_SV("authored"));
  EXPECT_EQ(authored_ref.symbol_id, expected_authored_ref.symbol_id);

  loom_amdgpu_target_identity_t variant_identity = MakeTargetIdentity(target);
  variant_identity.amdhsa_features.sramecc = LOOM_AMDGPU_TARGET_FEATURE_OFF;
  variant_identity.amdhsa_features.xnack = LOOM_AMDGPU_TARGET_FEATURE_ON;
  loom_amdgpu_target_profile_t variant_profile = {};
  IREE_ASSERT_OK(loom_amdgpu_target_profile_initialize(&variant_identity,
                                                       &variant_profile));
  loom_symbol_ref_t variant_ref = loom_symbol_ref_null();
  IREE_ASSERT_OK(loom_target_environment_materialize_effective_target(
      &target_environment_, module.get(), &variant_profile.base,
      /*authored_target_op=*/nullptr, &variant_ref));
  EXPECT_NE(variant_ref.symbol_id, authored_ref.symbol_id);

  loom_amdgpu_target_identity_t record_identity = {};
  loom_amdgpu_target_record_resolve_identity(
      TargetOpFromRef(module.get(), variant_ref), &record_identity);
  EXPECT_EQ(record_identity.amdhsa_features.sramecc,
            LOOM_AMDGPU_TARGET_FEATURE_OFF);
  EXPECT_EQ(record_identity.amdhsa_features.xnack,
            LOOM_AMDGPU_TARGET_FEATURE_ON);

  loom_amdgpu_target_profile_t equal_variant_profile = {};
  IREE_ASSERT_OK(loom_amdgpu_target_profile_initialize(&variant_identity,
                                                       &equal_variant_profile));
  loom_symbol_ref_t equal_variant_ref = loom_symbol_ref_null();
  IREE_ASSERT_OK(loom_target_environment_materialize_effective_target(
      &target_environment_, module.get(), &equal_variant_profile.base,
      /*authored_target_op=*/nullptr, &equal_variant_ref));
  EXPECT_EQ(equal_variant_ref.symbol_id, variant_ref.symbol_id);
}

TEST_F(AmdgpuProviderTest, SatisfiesCanonicalProcessorRequirements) {
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
  EXPECT_FALSE(gfx1151_a_facts->subgroup_size_authored);
  EXPECT_TRUE(gfx1151_explicit_wave32_facts->subgroup_size_authored);
  EXPECT_FALSE(gfx11_generic_facts->contract_set_key_authored);
  EXPECT_TRUE(gfx11_explicit_contract_facts->contract_set_key_authored);

  EXPECT_TRUE(Satisfies(gfx1151_a, gfx1151_a));
  EXPECT_TRUE(Satisfies(gfx1151_a, gfx1151_b));
  EXPECT_TRUE(Satisfies(gfx1151_a, gfx11_generic));
  EXPECT_TRUE(Satisfies(gfx1151_a, gfx11_wave32));
  EXPECT_TRUE(Satisfies(gfx1151_a, gfx11_small_workgroup));
  EXPECT_FALSE(Satisfies(gfx11_generic, gfx1151_a));
  EXPECT_FALSE(Satisfies(gfx1151_a, gfx1150));
  EXPECT_FALSE(Satisfies(gfx1170, gfx11_generic));
  EXPECT_TRUE(Satisfies(gfx1151_a, gfx11_wave64));
  EXPECT_FALSE(Satisfies(gfx1151_explicit_wave32, gfx11_wave64));
  EXPECT_FALSE(Satisfies(gfx1151_a, gfx11_feature));
  EXPECT_FALSE(Satisfies(gfx1151_a, gfx11_explicit_contract));
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

  EXPECT_TRUE(Satisfies(xnack_on, any));
  EXPECT_TRUE(Satisfies(xnack_on, xnack_on));
  EXPECT_FALSE(Satisfies(xnack_on, xnack_off));
  EXPECT_FALSE(Satisfies(any, xnack_on));
  EXPECT_FALSE(Satisfies(xnack_off, xnack_on));
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

  EXPECT_TRUE(Satisfies(gfx1250_a0, gfx1250_a0));
  EXPECT_TRUE(Satisfies(gfx1250_b0, gfx1250_b0));
  EXPECT_FALSE(Satisfies(gfx1250_a0, gfx1250_b0));
  EXPECT_FALSE(Satisfies(gfx1250_b0, gfx1250_a0));
  EXPECT_TRUE(Satisfies(gfx1250_a0, gfx12_5_generic));
  EXPECT_TRUE(Satisfies(gfx1250_b0, gfx12_5_generic));
  EXPECT_FALSE(Satisfies(gfx12_5_generic, gfx1250_a0));
}

TEST_F(AmdgpuProviderTest, ExhaustsSupportedRecordSatisfactionRelation) {
  struct MaterializedRecord {
    // Generated canonical target row represented by the record.
    const loom_amdgpu_target_info_t* target_info;

    // Indexed materialized target record for the target.
    const loom_target_symbol_facts_t* target;
  };

  ModulePtr module = Parse(IREE_SV(""));
  std::vector<MaterializedRecord> records;
  const iree_host_size_t target_count = loom_amdgpu_target_info_target_count();
  for (iree_host_size_t i = 0; i < target_count; ++i) {
    const loom_amdgpu_target_info_t* target_info =
        loom_amdgpu_target_info_target_at(i);
    ASSERT_NE(target_info, nullptr);
    loom_amdgpu_target_profile_t profile = {};
    const loom_amdgpu_target_identity_t identity =
        MakeTargetIdentity(target_info);
    IREE_ASSERT_OK(loom_amdgpu_target_profile_initialize(&identity, &profile));
    loom_symbol_ref_t target_ref = loom_symbol_ref_null();
    IREE_ASSERT_OK(loom_target_environment_materialize_effective_target(
        &target_environment_, module.get(), &profile.base,
        /*authored_target_op=*/nullptr, &target_ref));
    const loom_target_symbol_facts_t* target = Target(module.get(), target_ref);
    ASSERT_TRUE(target != nullptr && target->projection != nullptr);
    records.push_back({
        /*.target_info=*/target_info,
        /*.target=*/target,
    });
  }

  ASSERT_FALSE(records.empty());
  for (const MaterializedRecord& effective : records) {
    for (const MaterializedRecord& requirement : records) {
      const bool expected =
          loom_amdgpu_target_satisfies_code_object_requirement(
              effective.target_info, requirement.target_info);
      EXPECT_EQ(Satisfies(effective.target, requirement.target), expected)
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
