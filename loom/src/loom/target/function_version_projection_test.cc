// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/function_version_projection.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/facts_builder.h"
#include "loom/target/provider.h"
#include "loom/target/specialization.h"
#include "loom/target/test/target_records.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

typedef struct TestTargetProfile {
  // Generic target profile base.
  loom_target_profile_t base;

  // Test target selector projected into facts.
  loom_test_target_kind_t kind;

  // Common target fields explicitly supplied by the profile.
  loom_target_fact_field_set_t explicit_fields;
} TestTargetProfile;

static iree_status_t ProjectTestProfileFacts(
    const loom_target_profile_t* base_profile, iree_arena_allocator_t* arena,
    loom_target_facts_t* out_facts) {
  (void)arena;
  const auto* profile =
      reinterpret_cast<const TestTargetProfile*>(base_profile);
  out_facts->selector = profile->kind;
  out_facts->explicit_fields = profile->explicit_fields;
  return iree_ok_status();
}

static const loom_target_profile_type_t kTestProfileType = {
    /*.name=*/IREE_SVL("function-version-projection-test"),
    /*.fact_type=*/&loom_test_target_fact_type,
    /*.project_facts=*/ProjectTestProfileFacts,
};

static iree_status_t MaterializeTestTargetDefinition(
    loom_builder_t* builder, const loom_resolved_target_t* resolved_target,
    loom_symbol_ref_t symbol, loom_location_id_t location) {
  const loom_target_facts_t* facts = resolved_target->facts;
  static_assert(LOOM_TARGET_FACT_FIELD_COUNT_ == 30,
                "test target flags reserve the first 30 bits for common "
                "target facts");
  static_assert(LOOM_TEST_TARGET_BUILD_FLAG_HAS_CODEGEN_FORMAT ==
                    (1u << LOOM_TARGET_FACT_FIELD_CODEGEN_FORMAT),
                "test target flags must follow target fact ordinals");
  static_assert(LOOM_TEST_TARGET_BUILD_FLAG_HAS_CONTRACT_FEATURE_BITS ==
                    (1u << LOOM_TARGET_FACT_FIELD_CONTRACT_FEATURE_BITS),
                "test target flags must follow target fact ordinals");
  const auto build_flags =
      static_cast<loom_test_target_build_flags_t>(facts->explicit_fields);

  loom_string_id_t export_symbol = LOOM_STRING_ID_INVALID;
  if (iree_any_bit_set(build_flags,
                       LOOM_TEST_TARGET_BUILD_FLAG_HAS_EXPORT_SYMBOL)) {
    IREE_RETURN_IF_ERROR(loom_builder_intern_string(
        builder, facts->storage.export_plan.export_symbol, &export_symbol));
  }
  loom_string_id_t contract_set_key = LOOM_STRING_ID_INVALID;
  if (iree_any_bit_set(build_flags,
                       LOOM_TEST_TARGET_BUILD_FLAG_HAS_CONTRACT_SET_KEY)) {
    IREE_RETURN_IF_ERROR(loom_builder_intern_string(
        builder, facts->storage.config.contract_set_key, &contract_set_key));
  }

  const loom_target_snapshot_t* snapshot = &facts->storage.snapshot;
  const loom_target_export_plan_t* export_plan = &facts->storage.export_plan;
  const loom_target_config_t* config = &facts->storage.config;
  loom_op_t* target_op = nullptr;
  return loom_test_target_build(
      builder, build_flags,
      static_cast<loom_test_target_kind_t>(facts->selector), symbol,
      snapshot->codegen_format, snapshot->artifact_format,
      snapshot->default_pointer_bitwidth, snapshot->index_bitwidth,
      snapshot->offset_bitwidth, snapshot->max_workgroup_size.x,
      snapshot->max_workgroup_size.y, snapshot->max_workgroup_size.z,
      snapshot->max_flat_workgroup_size, snapshot->max_workgroup_storage_bytes,
      snapshot->subgroup_size, snapshot->max_grid_size.x,
      snapshot->max_grid_size.y, snapshot->max_grid_size.z,
      snapshot->max_flat_grid_size, snapshot->max_workgroup_count.x,
      snapshot->max_workgroup_count.y, snapshot->max_workgroup_count.z,
      snapshot->memory_spaces.generic, snapshot->memory_spaces.global,
      snapshot->memory_spaces.workgroup, snapshot->memory_spaces.constant,
      snapshot->memory_spaces.private_memory, snapshot->memory_spaces.host,
      snapshot->memory_spaces.descriptor, export_plan->abi_kind, export_symbol,
      export_plan->linkage, contract_set_key, config->contract_feature_bits,
      location, &target_op);
}

static const loom_target_provider_t kTestProvider = {
    /*.profile_type=*/&kTestProfileType,
    /*.materialize_definition=*/MaterializeTestTargetDefinition,
};

static const loom_target_provider_t kMissingMaterializerProvider = {
    /*.profile_type=*/&kTestProfileType,
};

static const loom_target_provider_t* const kTestProviders[] = {
    &kTestProvider,
};

static const loom_target_provider_set_t kTestProviderSet =
    loom_target_provider_set_make(kTestProviders,
                                  IREE_ARRAYSIZE(kTestProviders));

static const loom_target_provider_t* const kMissingMaterializerProviders[] = {
    &kMissingMaterializerProvider,
};

static const loom_target_provider_set_t kMissingMaterializerProviderSet =
    loom_target_provider_set_make(
        kMissingMaterializerProviders,
        IREE_ARRAYSIZE(kMissingMaterializerProviders));

static TestTargetProfile MakeTestProfile(
    loom_test_target_kind_t kind,
    loom_target_fact_field_set_t explicit_fields = 0) {
  return TestTargetProfile{
      /*.base=*/
      {
          /*.type=*/&kTestProfileType,
          /*.target_bundle=*/
          loom_target_bundle_table_lookup(&loom_test_target_bundles, kind),
      },
      /*.kind=*/kind,
      /*.explicit_fields=*/explicit_fields,
  };
}

class TargetFunctionVersionProjectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(
        loom_target_environment_initialize(&kTestProviderSet, &environment_));
    iree_arena_initialize(&block_pool_, &version_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&version_arena_);
    loom_target_environment_deinitialize(&environment_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(loom_dialect_id_t dialect_id, DialectVtablesFn fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = fn(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, dialect_id, vtables, static_cast<uint16_t>(vtable_count)));
  }

  ModulePtr Parse(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_CHECK_OK(
        loom_text_parse(iree_make_cstring_view(source),
                        IREE_SV("function_version_projection_test.loom"),
                        &context_, &block_pool_, &options, &module));
    IREE_ASSERT(module != nullptr);
    return ModulePtr(module);
  }

  std::string Print(const loom_module_t* module) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    IREE_CHECK_OK(loom_text_print_module_to_builder(module, &builder,
                                                    LOOM_TEXT_PRINT_DEFAULT));
    std::string text(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
    iree_string_builder_deinitialize(&builder);
    return text;
  }

  loom_symbol_id_t FindSymbol(const loom_module_t* module,
                              iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    return symbol_id;
  }

  loom_func_like_t FindFunction(loom_module_t* module,
                                iree_string_view_t name) {
    const loom_symbol_id_t symbol_id = FindSymbol(module, name);
    loom_func_like_t function = loom_func_like_cast(
        module, module->symbols.entries[symbol_id].defining_op);
    IREE_ASSERT(loom_func_like_isa(function));
    return function;
  }

  iree_string_view_t FunctionTargetName(loom_module_t* module,
                                        iree_string_view_t function_name) {
    const loom_symbol_ref_t target_ref =
        loom_func_like_target(FindFunction(module, function_name));
    IREE_ASSERT(loom_symbol_ref_is_valid(target_ref));
    IREE_ASSERT_EQ(target_ref.module_id, 0);
    return module->strings
        .entries[module->symbols.entries[target_ref.symbol_id].name_id];
  }

  const loom_op_t* FunctionTarget(loom_module_t* module,
                                  iree_string_view_t function_name) {
    const loom_symbol_ref_t target_ref =
        loom_func_like_target(FindFunction(module, function_name));
    IREE_ASSERT(loom_symbol_ref_is_valid(target_ref));
    IREE_ASSERT_EQ(target_ref.module_id, 0);
    return module->symbols.entries[target_ref.symbol_id].defining_op;
  }

  iree_host_size_t CountTestTargets(const loom_module_t* module) {
    iree_host_size_t count = 0;
    for (loom_symbol_id_t symbol_id = 0; symbol_id < module->symbols.count;
         ++symbol_id) {
      if (loom_test_target_isa(
              module->symbols.entries[symbol_id].defining_op)) {
        ++count;
      }
    }
    return count;
  }

  loom_target_specialization_result_t Specialize(
      loom_module_t* module,
      const loom_target_specialization_request_t* requests,
      iree_host_size_t request_count) {
    return SpecializeWithEnvironment(&environment_, module, requests,
                                     request_count);
  }

  loom_target_specialization_result_t SpecializeWithEnvironment(
      const loom_target_environment_t* environment, loom_module_t* module,
      const loom_target_specialization_request_t* requests,
      iree_host_size_t request_count) {
    loom_target_specialization_result_t result = {};
    IREE_CHECK_OK(loom_target_specialize_functions(environment, module,
                                                   {
                                                       /*.values=*/requests,
                                                       /*.count=*/request_count,
                                                   },
                                                   /*.bindings=*/{},
                                                   /*.diagnostic_emitter=*/{},
                                                   &version_arena_, &result));
    IREE_ASSERT_EQ(result.error_count, 0u);
    return result;
  }

  ModulePtr Project(const loom_module_t* source_module,
                    const loom_function_version_list_t* function_versions) {
    loom_module_t* projected_module = nullptr;
    IREE_CHECK_OK(loom_target_function_versions_project_module(
        source_module, function_versions, &block_pool_, iree_allocator_system(),
        &projected_module));
    return ModulePtr(projected_module);
  }

  void InitializeFacts(loom_test_target_kind_t kind,
                       loom_target_fact_field_set_t explicit_fields,
                       loom_target_facts_t* out_facts) {
    loom_target_facts_builder_initialize(
        &loom_test_target_fact_type,
        loom_target_bundle_table_lookup(&loom_test_target_bundles, kind),
        out_facts);
    out_facts->selector = kind;
    out_facts->explicit_fields = explicit_fields;
  }

  loom_target_function_version_t MakeVersion(
      loom_module_t* module, iree_string_view_t function_name,
      const loom_target_provider_t* provider,
      const loom_target_facts_t* facts) {
    return loom_target_function_version_t{
        /*.base=*/
        {
            /*.type=*/&loom_target_function_version_type,
            /*.function=*/FindFunction(module, function_name),
        },
        /*.authored_target_name=*/{},
        /*.target_requirement_facts=*/nullptr,
        /*.resolved_target=*/
        {
            /*.provider=*/provider,
            /*.facts=*/facts,
        },
        /*.target_context_ordinal=*/0,
        /*.authored_target_is_exact=*/false,
        /*.function_target_facts=*/facts,
    };
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_environment_t environment_;
  iree_arena_allocator_t version_arena_;
};

TEST_F(TargetFunctionVersionProjectionTest,
       NoVersionsProducesAnExactIndependentClone) {
  ModulePtr source = Parse(R"(
test.target<low_core> @authored

func.def public target(@authored) @entry() {
  func.return
}
)");
  const std::string source_text = Print(source.get());
  loom_op_t* source_function = FindFunction(source.get(), IREE_SV("entry")).op;

  ModulePtr projected = Project(source.get(), /*function_versions=*/nullptr);

  ASSERT_NE(projected, nullptr);
  EXPECT_NE(projected.get(), source.get());
  EXPECT_NE(FindFunction(projected.get(), IREE_SV("entry")).op,
            source_function);
  EXPECT_EQ(Print(projected.get()), source_text);
  EXPECT_EQ(Print(source.get()), source_text);
}

TEST_F(TargetFunctionVersionProjectionTest,
       SharedTargetlessContextMaterializesOnce) {
  ModulePtr source = Parse(R"(
func.def public @left() {
  func.return
}

func.def public @right() {
  func.return
}
)");
  const TestTargetProfile profile =
      MakeTestProfile(LOOM_TEST_TARGET_KIND_LOW_CORE);
  const loom_target_specialization_request_t requests[] = {
      {
          /*.function_name=*/IREE_SV("left"),
          /*.target_profile=*/&profile.base,
      },
      {
          /*.function_name=*/IREE_SV("right"),
          /*.target_profile=*/&profile.base,
      },
  };
  const loom_target_specialization_result_t specialization =
      Specialize(source.get(), requests, IREE_ARRAYSIZE(requests));
  ASSERT_EQ(specialization.function_versions.list.count, 2u);
  const auto* left_version = loom_target_function_version_const_cast(
      specialization.function_versions.list.values[0]);
  const auto* right_version = loom_target_function_version_const_cast(
      specialization.function_versions.list.values[1]);
  ASSERT_NE(left_version, nullptr);
  ASSERT_NE(right_version, nullptr);
  EXPECT_EQ(left_version->target_context_ordinal,
            right_version->target_context_ordinal);
  const iree_host_size_t source_symbol_count = source->symbols.count;
  const std::string source_text = Print(source.get());

  ModulePtr projected =
      Project(source.get(), &specialization.function_versions.list);

  EXPECT_EQ(source->symbols.count, source_symbol_count);
  EXPECT_EQ(Print(source.get()), source_text);
  EXPECT_EQ(projected->symbols.count, source_symbol_count + 1);
  EXPECT_EQ(CountTestTargets(projected.get()), 1u);
  EXPECT_TRUE(iree_string_view_equal(
      FunctionTargetName(projected.get(), IREE_SV("left")),
      IREE_SV("__loom_target_context_0_0")));
  EXPECT_TRUE(iree_string_view_equal(
      FunctionTargetName(projected.get(), IREE_SV("right")),
      IREE_SV("__loom_target_context_0_0")));
}

TEST_F(TargetFunctionVersionProjectionTest,
       InexactAuthoredContextMaterializesResolvedDefinition) {
  ModulePtr source = Parse(R"(
test.target<low_core> @requirement

func.def public target(@requirement) @entry() {
  func.return
}
)");
  loom_target_fact_field_set_t explicit_fields = 0;
  loom_target_fact_field_set_insert(&explicit_fields,
                                    LOOM_TARGET_FACT_FIELD_INDEX_BITWIDTH);
  const TestTargetProfile profile =
      MakeTestProfile(LOOM_TEST_TARGET_KIND_LOW_CORE, explicit_fields);
  const loom_target_specialization_request_t request = {
      /*.function_name=*/IREE_SV("entry"),
      /*.target_profile=*/&profile.base,
  };
  const loom_target_specialization_result_t specialization =
      Specialize(source.get(), &request, 1);
  const auto* version = loom_target_function_version_const_cast(
      specialization.function_versions.list.values[0]);
  ASSERT_NE(version, nullptr);
  EXPECT_FALSE(version->authored_target_is_exact);
  const iree_host_size_t source_symbol_count = source->symbols.count;

  ModulePtr projected =
      Project(source.get(), &specialization.function_versions.list);

  const loom_op_t* projected_target =
      FunctionTarget(projected.get(), IREE_SV("entry"));
  EXPECT_EQ(projected->symbols.count, source_symbol_count + 1);
  EXPECT_EQ(CountTestTargets(projected.get()), 2u);
  EXPECT_TRUE(iree_string_view_equal(
      FunctionTargetName(projected.get(), IREE_SV("entry")),
      IREE_SV("__loom_target_context_0_0")));
  ASSERT_TRUE(loom_test_target_isa(projected_target));
  EXPECT_EQ(loom_test_target_kind(projected_target),
            LOOM_TEST_TARGET_KIND_LOW_CORE);
  const loom_attribute_t index_bitwidth = loom_op_const_attrs(
      projected_target)[loom_test_target_index_bitwidth_ATTR_INDEX];
  ASSERT_FALSE(loom_attr_is_absent(index_bitwidth));
  EXPECT_EQ(loom_attr_as_i64(index_bitwidth), 64);
}

TEST_F(TargetFunctionVersionProjectionTest,
       MaterializedDefinitionsFollowAppendedSymbolOrder) {
  ModulePtr source = Parse(R"(
func.def public @entry() {
  func.return
}
)");
  loom_target_facts_t facts = {};
  InitializeFacts(LOOM_TEST_TARGET_KIND_LOW_CORE, 0, &facts);
  loom_target_function_version_t version =
      MakeVersion(source.get(), IREE_SV("entry"), &kTestProvider, &facts);
  loom_function_version_t* version_values[] = {&version.base};
  const loom_function_version_list_t versions = {
      /*.values=*/version_values,
      /*.count=*/IREE_ARRAYSIZE(version_values),
  };

  ModulePtr projected = Project(source.get(), &versions);

  ASSERT_EQ(projected->symbols.count, 2u);
  const loom_symbol_t* target_symbol = &projected->symbols.entries[1];
  ASSERT_TRUE(loom_test_target_isa(target_symbol->defining_op));
  EXPECT_EQ(loom_module_block(projected.get())->last_op,
            target_symbol->defining_op);
  const std::string projected_text = Print(projected.get());
  ModulePtr reparsed = Parse(projected_text.c_str());
  EXPECT_EQ(Print(reparsed.get()), projected_text);
}

TEST_F(TargetFunctionVersionProjectionTest,
       ProjectsAnExactAuthoredDefinitionDirectly) {
  ModulePtr source = Parse(R"(
test.target<low_core> @exact

func.def public target(@exact) @entry() {
  func.return
}
)");
  const TestTargetProfile profile =
      MakeTestProfile(LOOM_TEST_TARGET_KIND_LOW_CORE);
  const loom_target_specialization_request_t request = {
      /*.function_name=*/IREE_SV("entry"),
      /*.target_profile=*/&profile.base,
  };
  const loom_target_specialization_result_t specialization =
      Specialize(source.get(), &request, 1);
  const iree_host_size_t source_symbol_count = source->symbols.count;

  ModulePtr projected =
      Project(source.get(), &specialization.function_versions.list);

  EXPECT_EQ(projected->symbols.count, source_symbol_count);
  EXPECT_EQ(CountTestTargets(projected.get()), 1u);
  EXPECT_TRUE(iree_string_view_equal(
      FunctionTargetName(projected.get(), IREE_SV("entry")), IREE_SV("exact")));
}

TEST_F(TargetFunctionVersionProjectionTest,
       SharedExactContextProjectsOneAuthoredDefinition) {
  ModulePtr source = Parse(R"(
test.target<low_core> @exact

func.def public target(@exact) @first() {
  func.return
}

func.def public target(@exact) @second() {
  func.return
}
)");
  const TestTargetProfile profile =
      MakeTestProfile(LOOM_TEST_TARGET_KIND_LOW_CORE);
  const loom_target_specialization_request_t requests[] = {
      {
          /*.function_name=*/IREE_SV("first"),
          /*.target_profile=*/&profile.base,
      },
      {
          /*.function_name=*/IREE_SV("second"),
          /*.target_profile=*/&profile.base,
      },
  };
  const loom_target_specialization_result_t specialization =
      Specialize(source.get(), requests, IREE_ARRAYSIZE(requests));
  const auto* first_version = loom_target_function_version_const_cast(
      specialization.function_versions.list.values[0]);
  const auto* second_version = loom_target_function_version_const_cast(
      specialization.function_versions.list.values[1]);
  ASSERT_NE(first_version, nullptr);
  ASSERT_NE(second_version, nullptr);
  EXPECT_TRUE(first_version->authored_target_is_exact);
  EXPECT_TRUE(second_version->authored_target_is_exact);
  EXPECT_EQ(first_version->target_context_ordinal,
            second_version->target_context_ordinal);

  ModulePtr projected =
      Project(source.get(), &specialization.function_versions.list);

  EXPECT_EQ(CountTestTargets(projected.get()), 1u);
  EXPECT_TRUE(iree_string_view_equal(
      FunctionTargetName(projected.get(), IREE_SV("first")), IREE_SV("exact")));
  EXPECT_TRUE(iree_string_view_equal(
      FunctionTargetName(projected.get(), IREE_SV("second")),
      IREE_SV("exact")));
}

TEST_F(TargetFunctionVersionProjectionTest,
       ExactContextWitnessBindsEarlierInheritedVersion) {
  ModulePtr source = Parse(R"(
test.target<low_core> @exact

func.def public @inherited() {
  func.return
}

func.def public target(@exact) @witness() {
  func.return
}
)");
  loom_target_facts_t facts = {};
  InitializeFacts(LOOM_TEST_TARGET_KIND_LOW_CORE, 0, &facts);
  loom_target_function_version_t inherited_version =
      MakeVersion(source.get(), IREE_SV("inherited"),
                  &kMissingMaterializerProvider, &facts);
  loom_target_function_version_t witness_version = MakeVersion(
      source.get(), IREE_SV("witness"), &kMissingMaterializerProvider, &facts);
  witness_version.authored_target_name = IREE_SV("exact");
  witness_version.target_requirement_facts = &facts;
  witness_version.authored_target_is_exact = true;
  loom_function_version_t* version_values[] = {
      &inherited_version.base,
      &witness_version.base,
  };
  const loom_function_version_list_t versions = {
      /*.values=*/version_values,
      /*.count=*/IREE_ARRAYSIZE(version_values),
  };

  ModulePtr projected = Project(source.get(), &versions);

  EXPECT_EQ(CountTestTargets(projected.get()), 1u);
  EXPECT_TRUE(iree_string_view_equal(
      FunctionTargetName(projected.get(), IREE_SV("inherited")),
      IREE_SV("exact")));
  EXPECT_TRUE(iree_string_view_equal(
      FunctionTargetName(projected.get(), IREE_SV("witness")),
      IREE_SV("exact")));
}

TEST_F(TargetFunctionVersionProjectionTest,
       EquivalentIndependentProfilesRemainDistinctContexts) {
  ModulePtr source = Parse(R"(
func.def public @left() {
  func.return
}

func.def public @right() {
  func.return
}
)");
  const TestTargetProfile left_profile =
      MakeTestProfile(LOOM_TEST_TARGET_KIND_LOW_CORE);
  const TestTargetProfile right_profile =
      MakeTestProfile(LOOM_TEST_TARGET_KIND_LOW_CORE);
  const loom_target_specialization_request_t requests[] = {
      {
          /*.function_name=*/IREE_SV("left"),
          /*.target_profile=*/&left_profile.base,
      },
      {
          /*.function_name=*/IREE_SV("right"),
          /*.target_profile=*/&right_profile.base,
      },
  };
  const loom_target_specialization_result_t specialization =
      Specialize(source.get(), requests, IREE_ARRAYSIZE(requests));
  const auto* left_version = loom_target_function_version_const_cast(
      specialization.function_versions.list.values[0]);
  const auto* right_version = loom_target_function_version_const_cast(
      specialization.function_versions.list.values[1]);
  ASSERT_NE(left_version, nullptr);
  ASSERT_NE(right_version, nullptr);
  EXPECT_NE(left_version->target_context_ordinal,
            right_version->target_context_ordinal);
  EXPECT_TRUE(
      loom_target_facts_are_equivalent(left_version->resolved_target.facts,
                                       right_version->resolved_target.facts));

  ModulePtr projected =
      Project(source.get(), &specialization.function_versions.list);

  EXPECT_EQ(CountTestTargets(projected.get()), 2u);
  EXPECT_FALSE(iree_string_view_equal(
      FunctionTargetName(projected.get(), IREE_SV("left")),
      FunctionTargetName(projected.get(), IREE_SV("right"))));
}

TEST_F(TargetFunctionVersionProjectionTest,
       HeterogeneousTargetsUseDeterministicSourceSymbolOrder) {
  ModulePtr source = Parse(R"(
func.def public @low_entry() {
  func.return
}

func.def public @low_entry_copy() {
  func.return
}

func.def public @quirky_entry() {
  func.return
}

func.def @__loom_target_context_0_17() {
  func.return
}
)");
  const TestTargetProfile low_profile =
      MakeTestProfile(LOOM_TEST_TARGET_KIND_LOW_CORE);
  const TestTargetProfile quirky_profile =
      MakeTestProfile(LOOM_TEST_TARGET_KIND_QUIRKY);
  const loom_target_specialization_request_t requests[] = {
      {
          /*.function_name=*/IREE_SV("quirky_entry"),
          /*.target_profile=*/&quirky_profile.base,
      },
      {
          /*.function_name=*/IREE_SV("low_entry_copy"),
          /*.target_profile=*/&low_profile.base,
      },
      {
          /*.function_name=*/IREE_SV("low_entry"),
          /*.target_profile=*/&low_profile.base,
      },
  };
  const loom_target_specialization_result_t specialization =
      Specialize(source.get(), requests, IREE_ARRAYSIZE(requests));

  ModulePtr first =
      Project(source.get(), &specialization.function_versions.list);
  ModulePtr second =
      Project(source.get(), &specialization.function_versions.list);

  EXPECT_EQ(Print(first.get()), Print(second.get()));
  EXPECT_EQ(CountTestTargets(first.get()), 2u);
  EXPECT_TRUE(iree_string_view_equal(
      FunctionTargetName(first.get(), IREE_SV("low_entry")),
      IREE_SV("__loom_target_context_1_0")));
  EXPECT_TRUE(iree_string_view_equal(
      FunctionTargetName(first.get(), IREE_SV("low_entry_copy")),
      IREE_SV("__loom_target_context_1_0")));
  EXPECT_TRUE(iree_string_view_equal(
      FunctionTargetName(first.get(), IREE_SV("quirky_entry")),
      IREE_SV("__loom_target_context_1_1")));
  EXPECT_EQ(
      loom_test_target_kind(FunctionTarget(first.get(), IREE_SV("low_entry"))),
      LOOM_TEST_TARGET_KIND_LOW_CORE);
  EXPECT_EQ(loom_test_target_kind(
                FunctionTarget(first.get(), IREE_SV("quirky_entry"))),
            LOOM_TEST_TARGET_KIND_QUIRKY);
}

TEST_F(TargetFunctionVersionProjectionTest,
       ExactAuthoredContextDoesNotRequireMaterializer) {
  ModulePtr source = Parse(R"(
test.target<low_core> @exact

func.def public target(@exact) @entry() {
  func.return
}
)");
  loom_target_environment_t environment = {};
  IREE_ASSERT_OK(loom_target_environment_initialize(
      &kMissingMaterializerProviderSet, &environment));
  const TestTargetProfile profile =
      MakeTestProfile(LOOM_TEST_TARGET_KIND_LOW_CORE);
  const loom_target_specialization_request_t request = {
      /*.function_name=*/IREE_SV("entry"),
      /*.target_profile=*/&profile.base,
  };
  const loom_target_specialization_result_t specialization =
      SpecializeWithEnvironment(&environment, source.get(), &request, 1);
  loom_target_environment_deinitialize(&environment);
  const auto* version = loom_target_function_version_const_cast(
      specialization.function_versions.list.values[0]);
  ASSERT_NE(version, nullptr);
  EXPECT_TRUE(version->authored_target_is_exact);

  ModulePtr projected =
      Project(source.get(), &specialization.function_versions.list);

  EXPECT_EQ(CountTestTargets(projected.get()), 1u);
  EXPECT_TRUE(iree_string_view_equal(
      FunctionTargetName(projected.get(), IREE_SV("entry")), IREE_SV("exact")));
}

TEST_F(TargetFunctionVersionProjectionTest,
       TargetlessContextRequiresProviderMaterializer) {
  ModulePtr source = Parse(R"(
func.def public @entry() {
  func.return
}
)");
  loom_target_environment_t environment = {};
  IREE_ASSERT_OK(loom_target_environment_initialize(
      &kMissingMaterializerProviderSet, &environment));
  const TestTargetProfile profile =
      MakeTestProfile(LOOM_TEST_TARGET_KIND_LOW_CORE);
  const loom_target_specialization_request_t request = {
      /*.function_name=*/IREE_SV("entry"),
      /*.target_profile=*/&profile.base,
  };
  const loom_target_specialization_result_t specialization =
      SpecializeWithEnvironment(&environment, source.get(), &request, 1);
  loom_target_environment_deinitialize(&environment);
  loom_module_t* projected_module = nullptr;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_target_function_versions_project_module(
          source.get(), &specialization.function_versions.list, &block_pool_,
          iree_allocator_system(), &projected_module));
  EXPECT_EQ(projected_module, nullptr);
}

TEST_F(TargetFunctionVersionProjectionTest,
       RejectsAFunctionVersionFromAnotherModule) {
  ModulePtr source = Parse(R"(
func.def public @entry() {
  func.return
}
)");
  ModulePtr foreign = Parse(R"(
func.def public @entry() {
  func.return
}
)");
  loom_target_facts_t facts = {};
  InitializeFacts(LOOM_TEST_TARGET_KIND_LOW_CORE, 0, &facts);
  loom_target_function_version_t version =
      MakeVersion(foreign.get(), IREE_SV("entry"), &kTestProvider, &facts);
  loom_function_version_t* version_values[] = {&version.base};
  const loom_function_version_list_t versions = {
      /*.values=*/version_values,
      /*.count=*/IREE_ARRAYSIZE(version_values),
  };
  loom_module_t* projected_module = nullptr;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_target_function_versions_project_module(
                            source.get(), &versions, &block_pool_,
                            iree_allocator_system(), &projected_module));
  EXPECT_EQ(projected_module, nullptr);
}

TEST_F(TargetFunctionVersionProjectionTest,
       RejectsAnUnrepresentableFunctionVersion) {
  ModulePtr source = Parse(R"(
func.def public @entry() {
  func.return
}
)");
  const loom_function_version_type_t other_type = {
      /*.name=*/IREE_SVL("other"),
  };
  loom_function_version_t version = {
      /*.type=*/&other_type,
      /*.function=*/FindFunction(source.get(), IREE_SV("entry")),
  };
  loom_function_version_t* version_values[] = {&version};
  const loom_function_version_list_t versions = {
      /*.values=*/version_values,
      /*.count=*/IREE_ARRAYSIZE(version_values),
  };
  loom_module_t* projected_module = nullptr;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_target_function_versions_project_module(
                            source.get(), &versions, &block_pool_,
                            iree_allocator_system(), &projected_module));
  EXPECT_EQ(projected_module, nullptr);
}

}  // namespace
}  // namespace loom
